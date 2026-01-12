// Copyright 2025 Filippo Guarda
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "graph_generator_node/graph_generator_node.hpp"

#include <algorithm>
#include <queue>
#include <limits>
#include <unordered_map>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std::chrono_literals;

namespace graph_generator_node
{

GraphGeneratorNode::GraphGeneratorNode(const rclcpp::NodeOptions & options)
: Node("graph_generator_node", options)
{
  // Parameters tuned to match Python defaults where relevant
  input_topic_ = this->declare_parameter<std::string>("shared_costmap_topic", "/shared_obstacles");
  global_frame_ = this->declare_parameter<std::string>("global_frame", "map");
  obstacle_size_threshold_ = this->declare_parameter<int>("obstacle_size_threshold", 20);
  max_bfs_steps_ = this->declare_parameter<int>("max_bfs_steps", 100);
  find_entrances_ = this->declare_parameter<bool>("find_entrances", true);
  merge_threshold_pix_ = this->declare_parameter<double>("merge_threshold_pix", 0.0);

  costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
    input_topic_,
    rclcpp::QoS(1).reliable().transient_local(),
    std::bind(&GraphGeneratorNode::costmapCallback, this, std::placeholders::_1));
    
  filtered_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/skeleton_graph/filtered_map", rclcpp::QoS(1).reliable());

  skeleton_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/skeleton_graph/skeleton_map", rclcpp::QoS(1).reliable());

  graph_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/skeleton_graph/graph_markers", rclcpp::QoS(1).transient_local().reliable());

  RCLCPP_INFO(this->get_logger(),
              "GraphGeneratorNode initialized, subscribing to %s",
              input_topic_.c_str());
}

void GraphGeneratorNode::costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(map_mutex_);
  last_map_ = msg;

  if (msg->data.empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Received empty costmap");
    return;
  }

  cv::Mat binary = costmapToBinary(*msg);
  cv::Mat cleaned = removeSmallObstacles(binary);
  cv::Mat filtered = gridFastLikeCleanup(cleaned);

  nav_msgs::msg::OccupancyGrid filtered_grid = *msg;
  filtered_grid.data.assign(filtered_grid.data.size(), 0);
  for (int y = 0; y < filtered.rows; ++y) {
    const uint8_t * row = filtered.ptr<uint8_t>(y);
    for (int x = 0; x < filtered.cols; ++x) {
      int idx = y * filtered.cols + x;
      filtered_grid.data[idx] = (row[x] > 0) ? 100 : 0;
    }
  }
  filtered_pub_->publish(filtered_grid);

  cv::Mat skeleton = buildSkeleton(filtered);

  if (cv::countNonZero(skeleton) == 0) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "Empty skeleton generated - map may be too cluttered or have no free space corridors");
    // Still publish empty skeleton for debugging
  }


  nav_msgs::msg::OccupancyGrid skeleton_grid = *msg;
  skeleton_grid.data.assign(skeleton_grid.data.size(), 0);
  for (int y = 0; y < skeleton.rows; ++y) {
    const uint8_t * row = skeleton.ptr<uint8_t>(y);
    for (int x = 0; x < skeleton.cols; ++x) {
      int idx = y * skeleton.cols + x;
      skeleton_grid.data[idx] = (row[x] > 0) ? 100 : 0;
    }
  }
  skeleton_pub_->publish(skeleton_grid);

  cv::Mat dist_map = computeDistanceMap(skeleton);

  std::vector<GraphNode> nodes;
  std::vector<GraphEdge> edges;
  buildGraph(skeleton, dist_map, nodes, edges);

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                       "Graph: %zu nodes, %zu edges",
                       nodes.size(), edges.size());

  publishGraphMarkers(*msg, nodes, edges);
}

cv::Mat GraphGeneratorNode::costmapToBinary(const nav_msgs::msg::OccupancyGrid & grid)
{
  // Interpret as: 255 = occupied, 0 = free, unknown -> occupied
  cv::Mat binary(grid.info.height, grid.info.width, CV_8UC1);
  for (int y = 0; y < static_cast<int>(grid.info.height); ++y) {
    uint8_t * row = binary.ptr<uint8_t>(y);
    for (int x = 0; x < static_cast<int>(grid.info.width); ++x) {
      int idx = y * grid.info.width + x;
      int8_t v = grid.data[idx];
      if (v == -1) {
        row[x] = 255;
      } else if (v > 50) {
        row[x] = 255;
      } else {
        row[x] = 0;
      }
    }
  }
  return binary;
}

cv::Mat GraphGeneratorNode::removeSmallObstacles(const cv::Mat & map_data)
{
  // map_data: 0 free, 255 occupied
  cv::Mat obstacle_mask = (map_data == 255);
  cv::Mat labels, stats, centroids;
  int num_labels = cv::connectedComponentsWithStats(
      obstacle_mask, labels, stats, centroids, 8, CV_32S);

  cv::Mat result = map_data.clone();
  for (int label = 1; label < num_labels; ++label) {
    int area = stats.at<int>(label, cv::CC_STAT_AREA);
    if (area <= obstacle_size_threshold_) {
      result.setTo(0, labels == label);
    }
  }
  return result;
}

cv::Mat GraphGeneratorNode::gridFastLikeCleanup(const cv::Mat & cleaned_map)
{
  // Convert to 0/1: free=1, occupied=0
  cv::Mat free_mask = (cleaned_map == 0);

  // Simple morphological closing to fill thin gaps
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::Mat closed;
  cv::morphologyEx(free_mask, closed, cv::MORPH_CLOSE, kernel);

  // Dilate free space slightly
  cv::Mat dilated;
  cv::dilate(closed, dilated, kernel);

  // Rebuild map: free (0) where dilated is 1, occupied (255) elsewhere
  cv::Mat filtered(cleaned_map.size(), CV_8UC1, cv::Scalar(255));
  filtered.setTo(0, dilated > 0);
  return filtered;
}

cv::Mat GraphGeneratorNode::buildSkeleton(const cv::Mat & filtered_map)
{
  // Skeletonize free space
  cv::Mat bin = (filtered_map == 0);
  bin.convertTo(bin, CV_8UC1);

  cv::Mat skel(filtered_map.size(), CV_8UC1, cv::Scalar(0));
  cv::Mat temp;
  cv::Mat eroded;

  cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
  bool done = false;
  while (!done) {
    cv::erode(bin, eroded, element);
    cv::dilate(eroded, temp, element);
    cv::subtract(bin, temp, temp);
    cv::bitwise_or(skel, temp, skel);
    eroded.copyTo(bin);

    if (cv::countNonZero(bin) == 0) {
      done = true;
    }
  }

  return skel;
}

cv::Mat GraphGeneratorNode::computeDistanceMap(const cv::Mat & skeleton)
{
  // Distance from skeleton pixels into free-space region
  cv::Mat inv;
  cv::threshold(skeleton, inv, 0, 255, cv::THRESH_BINARY_INV);
  cv::Mat dist;
  cv::distanceTransform(inv, dist, cv::DIST_L2, 3);

  cv::Mat dist32;
  dist.convertTo(dist32, CV_32FC1);
  return dist32;
}

void GraphGeneratorNode::findSkeletonPoints(const cv::Mat & skeleton,
                                            cv::Mat & intersections_mask,
                                            cv::Mat & endpoints_mask)
{
  // Convert to 0/1
  cv::Mat skel_bin;
  cv::threshold(skeleton, skel_bin, 0, 1, cv::THRESH_BINARY);

  // Convolution with 3x3 connectivity kernel; center weight 10
  cv::Mat kernel = (cv::Mat_<int>(3,3) <<
                    1, 1, 1,
                    1,10, 1,
                    1, 1, 1);
  cv::Mat neighbor_count;
  cv::filter2D(skel_bin, neighbor_count, CV_32S, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);

  // Endpoint: neighbor_count == 11 and skeleton>0
  endpoints_mask = (neighbor_count == 11) & (skel_bin > 0);

  // Intersection detection via templates (T, Y, +, X etc.)
  intersections_mask = cv::Mat::zeros(skeleton.size(), CV_8U);

  auto hit_or_miss = [&](const cv::Mat & templ) {
    cv::Mat hit;
    cv::Mat fg = (templ == 1);
    cv::Mat bg = (templ == 0);
    cv::Mat sk_fg, sk_bg;
    cv::erode(skel_bin, sk_fg, fg);
    cv::erode(1 - skel_bin, sk_bg, bg);
    cv::bitwise_and(sk_fg, sk_bg, hit);
    intersections_mask |= hit;
  };

  // Define templates
  int T_data[9]   = {0,1,0, 1,1,1, 0,0,0};
  int Y_data[9]   = {1,0,1, 0,1,0, 0,1,0};
  int HX_data[9]  = {1,0,1, 0,1,0, 1,0,0};
  int TY_data[9]  = {0,0,1, 1,1,0, 0,1,0};
  int YT_data[9]  = {1,0,0, 0,1,1, 0,1,0};
  int YH_data[9]  = {1,0,1, 0,1,1, 0,1,0};
  int HY_data[9]  = {1,0,1, 1,1,0, 0,1,0};
  int plus_data[9]= {0,1,0, 1,1,1, 0,1,0};
  int X_data[9]   = {1,0,1, 0,1,0, 1,0,1};

  cv::Mat T(3,3,CV_8S,T_data);
  cv::Mat Y(3,3,CV_8S,Y_data);
  cv::Mat HX(3,3,CV_8S,HX_data);
  cv::Mat TY(3,3,CV_8S,TY_data);
  cv::Mat YT(3,3,CV_8S,YT_data);
  cv::Mat YH(3,3,CV_8S,YH_data);
  cv::Mat HY(3,3,CV_8S,HY_data);
  cv::Mat plus(3,3,CV_8S,plus_data);
  cv::Mat X(3,3,CV_8S,X_data);

  auto rotate90 = [](const cv::Mat & m) {
    cv::Mat r;
    cv::rotate(m, r, cv::ROTATE_90_CLOCKWISE);
    return r;
  };

  std::vector<cv::Mat> base = {T, Y, TY, YT, HX, YH, HY};
  for (const auto & t : base) {
    cv::Mat r0 = t.clone();
    cv::Mat r1 = rotate90(r0);
    cv::Mat r2 = rotate90(r1);
    cv::Mat r3 = rotate90(r2);
    hit_or_miss(r0);
    hit_or_miss(r1);
    hit_or_miss(r2);
    hit_or_miss(r3);
  }
  hit_or_miss(X);
  hit_or_miss(plus);
}

std::vector<cv::Point2i> GraphGeneratorNode::extractCoordsFromMask(const cv::Mat & mask)
{
  std::vector<cv::Point2i> coords;
  for (int y = 0; y < mask.rows; ++y) {
    const uint8_t * row = mask.ptr<uint8_t>(y);
    for (int x = 0; x < mask.cols; ++x) {
      if (row[x]) {
        coords.emplace_back(x, y);
      }
    }
  }
  return coords;
}

void GraphGeneratorNode::buildGraph(const cv::Mat & skeleton,
                                    const cv::Mat & dist_map,
                                    std::vector<GraphNode> & nodes,
                                    std::vector<GraphEdge> & edges)
{
  // 1) Find intersections and endpoints
  cv::Mat inter_mask, endpoint_mask;
  findSkeletonPoints(skeleton, inter_mask, endpoint_mask);

  std::vector<cv::Point2i> inter_coords = extractCoordsFromMask(inter_mask);
  std::vector<cv::Point2i> endpoint_coords = extractCoordsFromMask(endpoint_mask);

  // 2) Build initial nodes
  std::vector<GraphNode> initial_nodes;
  int nid = 0;
  for (const auto & p : inter_coords) {
    GraphNode n;
    n.id = nid++;
    n.pix = p;
    n.type = "intersection";
    initial_nodes.push_back(n);
  }
  for (const auto & p : endpoint_coords) {
    GraphNode n;
    n.id = nid++;
    n.pix = p;
    n.type = "endpoint";
    initial_nodes.push_back(n);
  }

  if (initial_nodes.empty()) {
    RCLCPP_WARN(this->get_logger(), "No initial skeleton nodes found (no intersections/endpoints)");
    nodes.clear();
    edges.clear();
    return;
  }

  // 3) Execute multi-source BFS
  executeBFS(skeleton, dist_map, find_entrances_, initial_nodes, nodes, edges);
}

void GraphGeneratorNode::executeBFS(const cv::Mat & skeleton,
                                    const cv::Mat & dist_map,
                                    bool find_entrances,
                                    const std::vector<GraphNode> & initial_nodes,
                                    std::vector<GraphNode> & out_nodes,
                                    std::vector<GraphEdge> & out_edges)
{
  const int H = skeleton.rows;
  const int W = skeleton.cols;

  cv::Mat visited_src(H, W, CV_32S, cv::Scalar(-1));
  cv::Mat visited_dist(H, W, CV_32S);
  visited_dist.setTo(std::numeric_limits<int>::max());

  std::unordered_map<long long, cv::Point2i> parent_map;
  std::unordered_map<int, cv::Point2i> id_to_pix;
  out_nodes.clear();
  out_edges.clear();

  out_nodes = initial_nodes;
  for (const auto & n : out_nodes) {
    id_to_pix[n.id] = n.pix;
  }

  int next_node_id = 0;
  for (const auto & n : out_nodes) {
    next_node_id = std::max(next_node_id, n.id + 1);
  }

  std::queue<std::tuple<int,int,int,int>> q;
  for (const auto & node : out_nodes) {
    int x = node.pix.x;
    int y = node.pix.y;
    if (x < 0 || x >= W || y < 0 || y >= H) continue;
    if (skeleton.at<uint8_t>(y,x) == 0) continue;

    int initial_budget = static_cast<int>(dist_map.at<float>(y,x));
    visited_src.at<int>(y,x) = node.id;
    visited_dist.at<int>(y,x) = 0;
    parent_map[packKey(node.id, x, y)] = cv::Point2i(-1,-1);
    q.emplace(x, y, node.id, initial_budget);
  }

  auto neighbors8 = [&](int x, int y) {
    static const int dx[8] = {1,1,0,-1,-1,-1,0,1};
    static const int dy[8] = {0,1,1,1,0,-1,-1,-1};
    std::vector<cv::Point2i> nbs;
    nbs.reserve(8);
    for (int k=0;k<8;++k) {
      int nx = x + dx[k];
      int ny = y + dy[k];
      if (nx>=0 && nx<W && ny>=0 && ny<H) {
        nbs.emplace_back(nx,ny);
      }
    }
    return nbs;
  };

  while (!q.empty()) {
    auto [x,y,src_id,remaining_budget] = q.front();
    q.pop();

    for (const auto & nb : neighbors8(x,y)) {
      int nx = nb.x;
      int ny = nb.y;
      if (skeleton.at<uint8_t>(ny,nx) == 0) {
        continue;
      }
      int other_src = visited_src.at<int>(ny,nx);
      if (other_src == -1) {
        // Unvisited
        int new_distance = visited_dist.at<int>(y,x) + 1;
        int new_budget = remaining_budget - 1;
        if (new_budget > 0) {
          visited_src.at<int>(ny,nx) = src_id;
          visited_dist.at<int>(ny,nx) = new_distance;
          parent_map[packKey(src_id,nx,ny)] = cv::Point2i(x,y);
          q.emplace(nx,ny,src_id,new_budget);
        } else if (new_budget == 0) {
          visited_src.at<int>(ny,nx) = src_id;
          visited_dist.at<int>(ny,nx) = new_distance;
          parent_map[packKey(src_id,nx,ny)] = cv::Point2i(x,y);
          if (find_entrances) {
            int entrance_id = createEntranceNode(cv::Point2i(nx,ny), src_id,
                                                 out_nodes, out_edges,
                                                 id_to_pix, parent_map);
            visited_src.at<int>(ny,nx) = entrance_id;
            parent_map[packKey(entrance_id,nx,ny)] = cv::Point2i(-1,-1);
            q.emplace(nx,ny,entrance_id,max_bfs_steps_);
          } else {
            q.emplace(nx,ny,src_id,max_bfs_steps_);
          }
        }
      } else if (other_src != src_id) {
        // Collision
        bool current_budget_exhausted = (remaining_budget <= 0);
        int local_narrowness = static_cast<int>(dist_map.at<float>(ny,nx));
        bool other_budget_exhausted = (visited_dist.at<int>(ny,nx) >= local_narrowness);
        if (current_budget_exhausted && other_budget_exhausted) {
          visited_src.at<int>(ny,nx) = src_id;
          visited_dist.at<int>(ny,nx) = visited_dist.at<int>(y,x) + 1;
          parent_map[packKey(src_id,nx,ny)] = cv::Point2i(x,y);
          int collision_id = createCollisionNode(cv::Point2i(nx,ny),
                                                 src_id, other_src,
                                                 out_nodes, out_edges,
                                                 id_to_pix, parent_map);
          visited_src.at<int>(ny,nx) = collision_id;
          parent_map[packKey(collision_id,nx,ny)] = cv::Point2i(-1,-1);
          visited_dist.at<int>(ny,nx) = 0;
          q.emplace(nx,ny,collision_id,max_bfs_steps_);
        } else {
          // Standard collision: create edge if not existing
          int u = std::min(src_id, other_src);
          int v = std::max(src_id, other_src);
          bool exists = false;
          for (const auto & e : out_edges) {
            if ((e.u == u && e.v == v) || (e.u == v && e.v == u)) {
              exists = true;
              break;
            }
          }
          if (!exists) {
            std::vector<cv::Point2i> path = reconstructPathCollision(
                cv::Point2i(x,y), cv::Point2i(nx,ny),
                src_id, other_src, parent_map);
            if (!path.empty()) {
              GraphEdge edge;
              edge.u = u;
              edge.v = v;
              edge.path_pixels = path;
              edge.weight = static_cast<double>(path.size());
              out_edges.push_back(edge);
            }
          }
        }
      }
    }
  }
}

int GraphGeneratorNode::createEntranceNode(
    const cv::Point2i & p,
    int parent_src_id,
    std::vector<GraphNode> & nodes,
    std::vector<GraphEdge> & edges,
    const std::unordered_map<int, cv::Point2i> & id_to_pix,
    const std::unordered_map<long long, cv::Point2i> & parent_map)
{
  int new_id = 0;
  for (const auto & n : nodes) {
    new_id = std::max(new_id, n.id + 1);
  }

  GraphNode n;
  n.id = new_id;
  n.pix = p;
  n.type = "entrance";
  nodes.push_back(n);

  std::vector<cv::Point2i> path = reconstructPath(p, parent_src_id, parent_map);
  if (!path.empty()) {
    GraphEdge e;
    e.u = parent_src_id;
    e.v = new_id;
    e.path_pixels = path;
    e.weight = static_cast<double>(path.size());
    edges.push_back(e);
  }
  return new_id;
}

int GraphGeneratorNode::createCollisionNode(
    const cv::Point2i & p,
    int src1,
    int src2,
    std::vector<GraphNode> & nodes,
    std::vector<GraphEdge> & edges,
    const std::unordered_map<int, cv::Point2i> & id_to_pix,
    const std::unordered_map<long long, cv::Point2i> & parent_map)
{
  int new_id = 0;
  for (const auto & n : nodes) {
    new_id = std::max(new_id, n.id + 1);
  }

  GraphNode n;
  n.id = new_id;
  n.pix = p;
  n.type = "collision";
  nodes.push_back(n);

  for (int src : {src1, src2}) {
    std::vector<cv::Point2i> path = reconstructPath(p, src, parent_map);
    if (!path.empty()) {
      GraphEdge e;
      e.u = src;
      e.v = new_id;
      e.path_pixels = path;
      e.weight = static_cast<double>(path.size());
      edges.push_back(e);
    }
  }
  return new_id;
}

std::vector<cv::Point2i> GraphGeneratorNode::reconstructPath(
    const cv::Point2i & p,
    int src_id,
    const std::unordered_map<long long, cv::Point2i> & parent_map) const
{
  std::vector<cv::Point2i> path;
  cv::Point2i cur = p;
  while (true) {
    path.push_back(cur);
    long long key = packKey(src_id, cur.x, cur.y);
    auto it = parent_map.find(key);
    if (it == parent_map.end()) {
      break;
    }
    cv::Point2i par = it->second;
    if (par.x < 0 || par.y < 0) {
      break;
    }
    cur = par;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<cv::Point2i> GraphGeneratorNode::reconstructPathCollision(
    const cv::Point2i & cur_xy,
    const cv::Point2i & neigh_xy,
    int src_current,
    int src_neighbor,
    const std::unordered_map<long long, cv::Point2i> & parent_map) const
{
  std::vector<cv::Point2i> path_current = reconstructPath(cur_xy, src_current, parent_map);
  std::vector<cv::Point2i> path_neighbor = reconstructPath(neigh_xy, src_neighbor, parent_map);
  std::reverse(path_neighbor.begin(), path_neighbor.end());
  path_current.insert(path_current.end(), path_neighbor.begin(), path_neighbor.end());
  return path_current;
}

geometry_msgs::msg::Point GraphGeneratorNode::gridToWorld(
    const nav_msgs::msg::OccupancyGrid & grid,
    int gx, int gy) const
{
  geometry_msgs::msg::Point p;
  double res = grid.info.resolution;
  p.x = grid.info.origin.position.x + (gx + 0.5) * res;
  p.y = grid.info.origin.position.y + (gy + 0.5) * res;
  p.z = 0.0;
  return p;
}

void GraphGeneratorNode::publishGraphMarkers(
    const nav_msgs::msg::OccupancyGrid & base_grid,
    const std::vector<GraphNode> & nodes,
    const std::vector<GraphEdge> & edges)
{
  visualization_msgs::msg::MarkerArray ma;
  rclcpp::Time stamp = this->get_clock()->now();

  // Nodes as SPHERE_LIST
  visualization_msgs::msg::Marker node_marker;
  node_marker.header.frame_id = base_grid.header.frame_id;
  node_marker.header.stamp = stamp;
  node_marker.ns = "skeleton_graph_nodes";
  node_marker.id = 0;
  node_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  node_marker.action = visualization_msgs::msg::Marker::ADD;
  node_marker.scale.x = base_grid.info.resolution * 1.5;
  node_marker.scale.y = base_grid.info.resolution * 1.5;
  node_marker.scale.z = base_grid.info.resolution * 1.5;
  node_marker.color.a = 1.0;
  node_marker.color.r = 1.0;
  node_marker.color.g = 0.0;
  node_marker.color.b = 0.0;

  for (const auto & n : nodes) {
    geometry_msgs::msg::Point p = gridToWorld(base_grid, n.pix.x, n.pix.y);
    node_marker.points.push_back(p);
  }
  ma.markers.push_back(node_marker);

  // Edges as LINE_LIST
  visualization_msgs::msg::Marker edge_marker;
  edge_marker.header.frame_id = base_grid.header.frame_id;
  edge_marker.header.stamp = stamp;
  edge_marker.ns = "skeleton_graph_edges";
  edge_marker.id = 1;
  edge_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  edge_marker.action = visualization_msgs::msg::Marker::ADD;
  edge_marker.scale.x = base_grid.info.resolution * 0.5;
  edge_marker.color.a = 1.0;
  edge_marker.color.r = 0.0;
  edge_marker.color.g = 0.0;
  edge_marker.color.b = 1.0;

  for (const auto & e : edges) {
    if (e.path_pixels.size() < 2) continue;
    for (size_t i=1; i<e.path_pixels.size(); ++i) {
      geometry_msgs::msg::Point p1 = gridToWorld(base_grid,
                                                 e.path_pixels[i-1].x,
                                                 e.path_pixels[i-1].y);
      geometry_msgs::msg::Point p2 = gridToWorld(base_grid,
                                                 e.path_pixels[i].x,
                                                 e.path_pixels[i].y);
      edge_marker.points.push_back(p1);
      edge_marker.points.push_back(p2);
    }
  }
  ma.markers.push_back(edge_marker);

  graph_marker_pub_->publish(ma);
}

}  // namespace graph_generator_node

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(graph_generator_node::GraphGeneratorNode)