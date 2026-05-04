#include "graph_generator_node/skeleton_graph_builder.hpp"
#include <queue>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <limits>
#include <functional>

namespace graph_generator_node {

// Initialize a static logger name for this builder utility
static const rclcpp::Logger LOGGER = rclcpp::get_logger("skeleton_graph_builder");

SkeletonGraphBuilder::SkeletonGraphBuilder(
    const cv::Mat& skeleton,
    const cv::Mat& distmap)
    : skeleton_(skeleton),
      distmap_(distmap),
      height_(skeleton.rows),
      width_(skeleton.cols) {
  graph_ = std::make_shared<NetworkX>();
  next_node_id_ = 0;
}

std::pair<std::shared_ptr<NetworkX>, std::unordered_map<int, std::pair<int, int>>>
SkeletonGraphBuilder::buildGraph(int max_steps, int hysteresis, bool find_entrances) {
  RCLCPP_DEBUG(LOGGER, "Starting buildGraph...");

  // Step 1: Find skeleton features
  cv::Mat intersections_mask, endpoints_mask;
  findSkeletonPoints(intersections_mask, endpoints_mask);

  std::vector<cv::Point2i> inter_coords = extractCoordsFromMask(intersections_mask);
  std::vector<cv::Point2i> endpoint_coords = extractCoordsFromMask(endpoints_mask);

  RCLCPP_INFO(LOGGER, "Found %zu intersections and %zu endpoints.", inter_coords.size(), endpoint_coords.size());

  if (inter_coords.empty() && endpoint_coords.empty()) {
    RCLCPP_WARN(LOGGER, "No skeleton points found. Graph will be empty.");
    return {graph_, {}};
  }

  // Step 2: Create position dictionaries
  std::unordered_map<int, std::pair<int, int>> inter_positions;
  for (size_t i = 0; i < inter_coords.size(); ++i) {
    inter_positions[i] = {inter_coords[i].x, inter_coords[i].y};
  }

  std::unordered_map<int, std::pair<int, int>> endpoint_positions;
  for (size_t i = 0; i < endpoint_coords.size(); ++i) {
    endpoint_positions[inter_positions.size() + i] = {endpoint_coords[i].x, endpoint_coords[i].y};
  }

  // Step 3: Build graph with multi-source BFS
  buildGraphMultiSourceBFS(
      inter_positions,
      endpoint_positions,
      find_entrances,
      max_steps,
      hysteresis);

  // Step 4: Collect node positions
  std::unordered_map<int, std::pair<int, int>> node_positions;
  for (const auto& [nid, node] : graph_->nodes()) {
    node_positions[nid] = node.position;
  }

  pruneShortBranches(hysteresis);

  RCLCPP_DEBUG(LOGGER, "Finished. Graph nodes: %zu, Edges: %zu", graph_->nodes().size(), graph_->edges().size());

  return {graph_, node_positions};
}

void SkeletonGraphBuilder::buildGraphMultiSourceBFS(
    const std::unordered_map<int, std::pair<int, int>>& intersection_positions,
    const std::unordered_map<int, std::pair<int, int>>& endpoint_positions,
    bool find_entrances,
    int max_steps,
    int hysteresis) {
  // Initialize tracking structures
  cv::Mat visited_src(height_, width_, CV_32S, cv::Scalar(-1));
  cv::Mat visited_dist(height_, width_, CV_32S);
  visited_dist.setTo(std::numeric_limits<int>::max());
  
  // Pre-allocate map memory
  std::unordered_map<uint64_t, std::pair<int, int>> parent_data;
  parent_data.reserve(width_ * height_); 

  auto make_key = [&](int src_id, int x, int y) -> uint64_t{
    return (static_cast<uint64_t>(src_id) << 32) | static_cast<uint64_t>(y * width_ + x);
  };

  std::queue<std::tuple<int, int, int, int, bool>> queue;

  // Clear graph and reset node ID counter
  graph_->clear();
  next_node_id_ = 0;

  // Add all initial nodes to graph
  for (const auto& [nid, pos] : intersection_positions) {
    graph_->addNode(nid, pos, "intersection");
  }
  for (const auto& [nid, pos] : endpoint_positions) {
    graph_->addNode(nid, pos, "endpoint");
  }

  // Collect all initial nodes
  std::vector<std::tuple<int, std::pair<int, int>, std::string>> all_nodes;
  for (const auto& [nid, pos] : intersection_positions) {
    all_nodes.push_back({nid, pos, "intersection"});
  }
  for (const auto& [nid, pos] : endpoint_positions) {
    all_nodes.push_back({nid, pos, "endpoint"});
  }

  // Update next_node_id to avoid collisions with new nodes
  int max_id = -1;
  if (!all_nodes.empty()) {
    for (const auto& item : all_nodes) max_id = std::max(max_id, std::get<0>(item));
    next_node_id_ = max_id + 1;
  }

  // Initialize queue with all skeleton starting points
  for (const auto& [nid, pos, node_type] : all_nodes) {
    int x = pos.first;
    int y = pos.second;

    if (x < 0 || x >= width_ || y < 0 || y >= height_) continue;
    if (skeleton_.at<uint8_t>(y, x) == 0) continue;

    int initial_budget = max_steps; // Use flat budget for all sources

    visited_src.at<int>(y, x) = nid;
    visited_dist.at<int>(y, x) = 0;
    parent_data[make_key(nid, x, y)] = {-1, -1};

    queue.push(std::make_tuple(x, y, nid, initial_budget, true));
  }

  RCLCPP_DEBUG(LOGGER, "BFS Queue initialized with %zu sources.", queue.size());

  // Execute BFS
  executeBFS(
      queue,
      visited_src,
      visited_dist,
      parent_data,
      find_entrances,
      max_steps,
      hysteresis);
}

void SkeletonGraphBuilder::executeBFS(
    std::queue<std::tuple<int, int, int, int, bool>>& queue,
    cv::Mat& visited_src,
    cv::Mat& visited_dist,
    std::unordered_map<uint64_t, std::pair<int, int>>&  parent_data,
    bool find_entrances,
    int max_steps,
    int hysteresis) {

  auto make_key = [&](int src_id, int x, int y) -> uint64_t{
    return (static_cast<uint64_t>(src_id) << 32) | static_cast<uint64_t>(y * width_ + x);
  };

  while (!queue.empty()) {
    auto [x, y, srcid, remaining_budget, parent_decreasing] = queue.front();
    queue.pop();

    // Distance of parent skeleton pixel from map borders
    float parent_distance = distmap_.at<float>(y, x);

    for (const auto& nb : neighbors8(x, y)) {
      int nx = nb.x;
      int ny = nb.y;

      // Do not iterate pixels outside of skeleton
      if (skeleton_.at<uint8_t>(ny, nx) == 0) continue;

      // Distance of child skeleton pixel from map borders
      float child_distance = distmap_.at<float>(ny, nx);
      int neighbor_src = visited_src.at<int>(ny, nx);

      // Do not iterate pixel with same parent
      if (neighbor_src == srcid) continue;

      // Local minima/maxima detection along distance field
      bool is_local_minima = parent_decreasing && (child_distance > parent_distance);
      bool is_local_maxima = !parent_decreasing && (child_distance < parent_distance);

      if (neighbor_src == -1) {
        // Unvisited pixel - claim it
        int new_distance = visited_dist.at<int>(y, x) + 1;
        int new_budget = remaining_budget - 1;
        visited_src.at<int>(ny, nx) = srcid;
        visited_dist.at<int>(ny, nx) = new_distance;
        parent_data[make_key(srcid, nx, ny)] = {x, y};

        // Hysteresis based on robot size
        if (new_budget > max_steps / 2) {
          // Continue with current source
          queue.push(std::make_tuple(nx, ny, srcid, new_budget, parent_decreasing));
          continue;
        }

        if (is_local_minima) {
          // Since we reached minima, now next node will see increasing direction
          parent_decreasing = false;

          if (find_entrances && child_distance < static_cast<float>(hysteresis)) {
            // Create entrance node if the area is not twice the size of the robot
            int entrance_id = createEntranceNode(cv::Point2i(nx, ny), srcid, parent_data);
            visited_src.at<int>(ny, nx) = entrance_id;
            visited_dist.at<int>(ny, nx) = 0;
            parent_data[make_key(entrance_id, nx, ny)] = {-1, -1};
            queue.push(std::make_tuple(nx, ny, entrance_id, max_steps, parent_decreasing));
          } else {
            // Reset budget but keep same source
            queue.push(std::make_tuple(nx, ny, srcid, max_steps, parent_decreasing));
          }
          continue;
        } else {
          if (is_local_maxima) parent_decreasing = true;
          // Continue with current source
          queue.push(std::make_tuple(nx, ny, srcid, new_budget, parent_decreasing));
          continue;
        }

      } else if (neighbor_src != srcid) {
        int u = std::min(srcid, neighbor_src);
        int v = std::max(srcid, neighbor_src);

        if (remaining_budget <= 0 && visited_dist.at<int>(y, x) > max_steps && child_distance <= static_cast<float>(max_steps)) {
          // Both exhausted - create collision node
          visited_src.at<int>(ny, nx) = srcid;
          visited_dist.at<int>(ny, nx) = visited_dist.at<int>(y, x) + 1;
          parent_data[make_key(srcid, nx, ny)] = {x, y};

          int collision_id = createCollisionNode(cv::Point2i(nx, ny), srcid, neighbor_src, parent_data);
          visited_src.at<int>(ny, nx) = collision_id;
          parent_data[make_key(collision_id, nx, ny)] = {-1, -1};

          visited_dist.at<int>(ny, nx) = 0;
          queue.push(std::make_tuple(nx, ny, collision_id, max_steps, parent_decreasing));

          // Update u,v to link to the new collision node instead of the neighbor
          u = std::min(srcid, collision_id);
          v = std::max(srcid, collision_id);
        }

        // This edge creation must run for BOTH collision node creation AND normal boundary meetings
        if (!graph_->hasEdge(u, v)) {
          std::vector<cv::Point2i> path = reconstructPathCollision(
              cv::Point2i(x, y), cv::Point2i(nx, ny), srcid, neighbor_src, parent_data);
          if (!path.empty()) {
            graph_->addEdge(u, v, path, static_cast<double>(path.size()));
          }
        }
      }
    }
  }
}

void SkeletonGraphBuilder::pruneShortBranches(int max_length) {
  while (true) {
    bool removed_any = false;
    std::vector<int> nodes_snapshot;
    for (const auto& [nid, node] : graph_->nodes()) {
      nodes_snapshot.push_back(nid);
    }

    for (int n : nodes_snapshot) {
      if (graph_->nodes().find(n) == graph_->nodes().end()) continue;
      std::vector<int> node_neighbors;
      try {
        node_neighbors = graph_->getNeighbors(n);
      } catch (...) {
        continue;
      }

      if (static_cast<int>(node_neighbors.size()) == 1) {
        int nb = node_neighbors[0];
        const auto* edge = graph_->getEdge(n, nb);
        if (!edge) edge = graph_->getEdge(nb, n);
        if (edge && static_cast<int>(edge->path_pixels.size()) < max_length) {
          graph_->removeNode(n);
          removed_any = true;
        }
      }
    }
    if (!removed_any) break;
  }
}

int SkeletonGraphBuilder::createEntranceNode(
    const cv::Point2i& p,
    int parent_src_id,
    std::unordered_map<uint64_t, std::pair<int, int>>& parent_data) {
  int entrance_id = next_node_id_++;
  graph_->addNode(entrance_id, {p.x, p.y}, "entrance");

  std::vector<cv::Point2i> path = reconstructPath(p, parent_src_id, parent_data);
  if (!path.empty()) {
    graph_->addEdge(parent_src_id, entrance_id, path, static_cast<double>(path.size()));
  }

  return entrance_id;
}

int SkeletonGraphBuilder::createCollisionNode(
    const cv::Point2i& p,
    int src1,
    int src2,
    std::unordered_map<uint64_t, std::pair<int, int>>&  parent_data) {
  int collision_id = next_node_id_++;
  graph_->addNode(collision_id, {p.x, p.y}, "collision");

  for (int src : {src1, src2}) {
    std::vector<cv::Point2i> path = reconstructPath(p, src, parent_data);
    if (!path.empty()) {
      graph_->addEdge(src, collision_id, path, static_cast<double>(path.size()));
    }
  }

  return collision_id;
}

std::vector<cv::Point2i> SkeletonGraphBuilder::reconstructPath(
    const cv::Point2i& p,
    int src_id,
    const std::unordered_map<uint64_t, std::pair<int, int>>& parent_data) {

  std::vector<cv::Point2i> path;
  cv::Point2i current = p;
  while (true) {
    path.push_back(current);
    uint64_t key = (static_cast<uint64_t>(src_id) << 32) |
      static_cast<uint64_t>(current.y * width_ + current.x);
    auto it = parent_data.find(key);

    if (it == parent_data.end() || (it->second.first == -1 && it-> second.second == -1)){
      break;
    }
    
    current = cv::Point2i(it-> second.first, it-> second.second);
  }

  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<cv::Point2i> SkeletonGraphBuilder::reconstructPathCollision(
    const cv::Point2i& cur_xy,
    const cv::Point2i& neigh_xy,
    int src_current,
    int src_neighbor,
    const std::unordered_map<uint64_t, std::pair<int, int>>& parent_data) {
  std::vector<cv::Point2i> path_current = reconstructPath(cur_xy, src_current, parent_data);
  std::vector<cv::Point2i> path_neighbor = reconstructPath(neigh_xy, src_neighbor, parent_data);

  std::reverse(path_neighbor.begin(), path_neighbor.end());
  path_current.insert(path_current.end(), path_neighbor.begin(), path_neighbor.end());

  return path_current;
}

std::unordered_map<int, std::pair<int, int>> SkeletonGraphBuilder::mergeCloseNodes(
    double distance_threshold,
    const std::vector<std::string>& node_types_to_merge) {

  std::unordered_map<int, std::pair<int, int>> result;
  if (distance_threshold <= 0.0) goto end_merge;

  {
    // 1. Determine node types (default to intersection and endpoint)
    std::vector<std::string> types = node_types_to_merge.empty() ? 
        std::vector<std::string>{"intersection", "endpoint"} : node_types_to_merge;

    std::vector<int> nodes;
    for (const auto& [nid, node] : graph_->nodes()) {
      if (std::find(types.begin(), types.end(), node.type) != types.end()) {
        nodes.push_back(nid);
      }
    }
    if (nodes.size() < 2) goto end_merge;

    // 2. Purely Spatial Union-Find
    std::unordered_map<int, int> parent;
    for (int id : nodes) parent[id] = id;
    std::function<int(int)> find_root = [&](int x) {
      return parent[x] == x ? x : (parent[x] = find_root(parent[x]));
    };

    for (size_t i = 0; i < nodes.size(); ++i) {
      for (size_t j = i + 1; j < nodes.size(); ++j) {
        const auto& n1 = graph_->nodes().at(nodes[i]);
        const auto& n2 = graph_->nodes().at(nodes[j]);

        // Enforce same type
        if (n1.type != n2.type)
          continue;

        auto p1 = n1.position;
        auto p2 = n2.position;
        if (std::hypot(p1.first - p2.first, p1.second - p2.second) <= distance_threshold) {
          parent[find_root(nodes[i])] = find_root(nodes[j]);
        }
      }
    }

    // 3. Process Clusters
    std::unordered_map<int, std::vector<int>> clusters;
    for (int nid : nodes) clusters[find_root(nid)].push_back(nid);

    for (const auto& [root, cluster] : clusters) {
      if (cluster.size() <= 1) continue;

      double bx = 0, by = 0;
      std::unordered_map<std::string, int> type_counts;
      std::unordered_set<int> external_neighbors;

      // Accumulate cluster data
      for (int nid : cluster) {
        const auto& n = graph_->nodes().at(nid);
        bx += n.position.first;
        by += n.position.second;
        type_counts[n.type]++;
        
        for (int nb : graph_->getNeighbors(nid)) {
          if (std::find(cluster.begin(), cluster.end(), nb) == cluster.end()) {
            external_neighbors.insert(nb);
          }
        }
      }

      std::string merged_type = graph_->nodes().at(cluster[0]).type;

      // Snap barycenter (using max_search_radius=5 matching Python)
      auto [snapx, snapy] = snapToSkeleton(std::round(bx / cluster.size()), std::round(by / cluster.size()), 5);

      int merged_id = next_node_id_++;
      graph_->addNode(merged_id, {snapx, snapy}, merged_type);

      // Reconnect external neighbors
      for (int nb : external_neighbors) {
        auto nb_pos = graph_->nodes().at(nb).position;
        
        // CRITICAL FIX: Attempt new BFS path first
        std::vector<std::pair<int, int>> new_path = findSkeletonPath({snapx, snapy}, nb_pos);

        if (!new_path.empty()) {
          graph_->addEdge(merged_id, nb, new_path, static_cast<double>(new_path.size()));
        } else {
          // Fallback to the shortest existing path from deleted nodes
          double best_dist = std::numeric_limits<double>::infinity();
          std::vector<std::pair<int, int>> best_path;
          
          for (int nid : cluster) {
            auto edge = graph_->getEdge(nid, nb);
            if (!edge) edge = graph_->getEdge(nb, nid); // Check bidirectional
            if (edge && edge->weight < best_dist) {
              best_dist = edge->weight;
              best_path = edge->path_pixels;
            }
          }
          if (!best_path.empty()) {
            graph_->addEdge(merged_id, nb, best_path, static_cast<double>(best_path.size()));
          }
        }
      }

      // Purge old nodes
      for (int nid : cluster) graph_->removeNode(nid);
    }
  }

end_merge:
  for (const auto& [nid, node] : graph_->nodes()) {
    result[nid] = node.position;
  }
  return result;
}


std::pair<int, int> SkeletonGraphBuilder::snapToSkeleton(int x, int y, int max_search_radius) {
  x = std::max(0, std::min(x, width_ - 1));
  y = std::max(0, std::min(y, height_ - 1));

  if (skeleton_.at<uint8_t>(y, x) > 0) return {x, y};

  for (int radius = 1; radius <= max_search_radius; ++radius) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        int nx = x + dx, ny = y + dy;
        if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_ && skeleton_.at<uint8_t>(ny, nx) > 0) {
          return {nx, ny};
        }
      }
    }
  }
  return {x, y};
}

std::vector<std::pair<int, int>> SkeletonGraphBuilder::findSkeletonPath(
    std::pair<int, int> start_pos, 
    std::pair<int, int> end_pos) {
    
    std::vector<std::pair<int, int>> path;
    int start_x = start_pos.first;
    int start_y = start_pos.second;
    int end_x = end_pos.first;
    int end_y = end_pos.second;

    // Validate boundaries and skeleton presence
    if (start_x < 0 || start_x >= width_ || start_y < 0 || start_y >= height_ || 
        skeleton_.at<uint8_t>(start_y, start_x) == 0) return path;
        
    if (end_x < 0 || end_x >= width_ || end_y < 0 || end_y >= height_ || 
        skeleton_.at<uint8_t>(end_y, end_x) == 0) return path;

    // Local tracking structures
    std::vector<bool> visited(width_ * height_, false);
    std::vector<std::pair<int, int>> parent(width_ * height_, {-1, -1});
    std::queue<std::pair<int, int>> q;

    q.push({start_x, start_y});
    visited[start_y * width_ + start_x] = true;
    bool found = false;

    // Standard BFS
    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop();

        if (x == end_x && y == end_y) {
            found = true;
            break;
        }

        for (const auto& nb : neighbors8(x, y)) {
            int nx = nb.x;
            int ny = nb.y;
            int idx = ny * width_ + nx;

            if (skeleton_.at<uint8_t>(ny, nx) > 0 && !visited[idx]) {
                visited[idx] = true;
                parent[idx] = {x, y};
                q.push({nx, ny});
            }
        }
    }

    if (!found) return path;

    // Reconstruct path backwards
    std::pair<int, int> current = {end_x, end_y};
    while (current.first != -1 && current.second != -1) {
        path.push_back(current);
        if (current.first == start_x && current.second == start_y) break;
        current = parent[current.second * width_ + current.first];
    }

    std::reverse(path.begin(), path.end());
    return path;
}

void SkeletonGraphBuilder::findSkeletonPoints(cv::Mat& intersections_mask, cv::Mat& endpoints_mask) {

    cv::Mat skel_bin;
    cv::threshold(skeleton_, skel_bin, 0, 1, cv::THRESH_BINARY);
    skel_bin.convertTo(skel_bin, CV_16S); // Convert to signed 16-bit to prevent overflow during filtering

    // Filter kernel: Center pixel has weight 10, neighbors have weight 1
    // An endpoint (center + 1 neighbor) will sum exactly to 11
    cv::Mat kernel = (cv::Mat_<int16_t>(3, 3) << 
        1,  1,  1, 
        1, 10,  1, 
        1,  1,  1);
        
    cv::Mat neighbor_count;
    cv::filter2D(skel_bin, neighbor_count, CV_16S, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    endpoints_mask = (neighbor_count == 11);
    
    // Ensure endpoints only exist where skeleton actually is
    endpoints_mask.setTo(0, skeleton_ == 0); 
    endpoints_mask.convertTo(endpoints_mask, CV_8U, 255);


    intersections_mask = cv::Mat::zeros(skeleton_.size(), CV_8U);

    // OpenCV MORPH_HITMISS requires input image to be strictly 0 and 255
    cv::Mat skel_255;
    cv::threshold(skeleton_, skel_255, 0, 255, cv::THRESH_BINARY);

    // Define 3-state kernels:
    //  1 = strictly foreground
    // -1 = strictly background
    //  0 = don't care (can be either)


    cv::Mat Y = (cv::Mat_<int8_t>(3, 3) << 
         1, -1,  1,
         1,  1,  1,
        -1,  1, -1);

    cv::Mat YT = (cv::Mat_<int8_t>(3, 3) << 
        -1, -1,  1,
         1,  1,  1,
        -1,  1, -1);

    cv::Mat TY = (cv::Mat_<int8_t>(3, 3) << 
         1, -1, -1,
         1,  1,  1,
        -1,  1, -1);

    cv::Mat T = (cv::Mat_<int8_t>(3, 3) << 
        -1, -1, -1,
         1,  1,  1,
        -1,  1, -1);

    cv::Mat y = (cv::Mat_<int8_t>(3, 3) << 
         1, -1,  1,
        -1,  1, -1,
         -1,  1,  -1);

    cv::Mat dY = (cv::Mat_<int8_t>(3, 3) << 
        -1, -1,  1,
         1,  1, -1,
        -1,  1, -1);

    cv::Mat dT = (cv::Mat_<int8_t>(3, 3) << 
         1, -1,  1,
        -1,  1, -1,
         1, -1, -1);

    cv::Mat YH = (cv::Mat_<int8_t>(3, 3) << 
         1, -1,  1,
        -1,  1,  1,
        -1,  1, -1);

    cv::Mat HY = (cv::Mat_<int8_t>(3, 3) << 
         1, -1,  1,
         1,  1, -1,
        -1,  1, -1);
        
    cv::Mat plus = (cv::Mat_<int8_t>(3, 3) << 
        -1,  1, -1,
         1,  1,  1,
        -1,  1, -1);

    cv::Mat X = (cv::Mat_<int8_t>(3, 3) << 
         1, -1,  1,
        -1,  1, -1,
         1, -1,  1);

    std::vector<cv::Mat> templates = {T, y, Y, TY, YT, dY, dT, YH, HY, plus};

    for (const auto& tmpl : templates) {
        cv::Mat current = tmpl.clone();

        for (int i = 0; i < 4; ++i) {
            cv::Mat hit;
            // Execute true 3-state Hit-or-Miss
            cv::morphologyEx(skel_255, hit, cv::MORPH_HITMISS, current);
            
            intersections_mask = intersections_mask | hit;

            // Rotate kernel 90 degrees clockwise for next iteration
            cv::rotate(current, current, cv::ROTATE_90_CLOCKWISE);
        }
    }

    // Apply the symmetrical X pattern (rotation invariant, only needs one pass)
    cv::Mat hit_x;
    cv::morphologyEx(skel_255, hit_x, cv::MORPH_HITMISS, X);
    intersections_mask = intersections_mask | hit_x;
}

std::vector<cv::Point2i> SkeletonGraphBuilder::extractCoordsFromMask(const cv::Mat& mask) {
  std::vector<cv::Point2i> coords;

  for (int y = 0; y < mask.rows; ++y) {
    const uint8_t* row = mask.ptr<uint8_t>(y);

    for (int x = 0; x < mask.cols; ++x) {
      if (row[x] != 0) coords.emplace_back(x, y);
    }
  }
  return coords;
}

std::vector<cv::Point2i> SkeletonGraphBuilder::neighbors8(int x, int y) {
  static const int dx8[] = {-1, -1, 0, 1, 1, 1, 0, -1};
  static const int dy8[] = {0, 1, 1, 1, 0, -1, -1, -1};

  std::vector<cv::Point2i> nbs;
  nbs.reserve(8);

  for (int i = 0; i < 8; ++i) {
    int nx = x + dx8[i];
    int ny = y + dy8[i];
    if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
      nbs.emplace_back(nx, ny);
    }
  }
  return nbs;
}

} // namespace graph_generator_node