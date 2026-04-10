#include "graph_generator_node/skeleton_graph_builder.hpp"
#include <opencv2/imgproc.hpp>
#include <queue>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <unordered_set>
#include <functional>
#include <rclcpp/rclcpp.hpp>

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
SkeletonGraphBuilder::buildGraph(int max_steps, bool find_entrances) {
  RCLCPP_DEBUG(LOGGER, "Starting buildGraph...");

  // Step 1: Find skeleton features
  cv::Mat intersections_mask, endpoints_mask;
  findSkeletonPoints(intersections_mask, endpoints_mask);

  std::vector<cv::Point2i> inter_coords = extractCoordsFromMask(intersections_mask);
  std::vector<cv::Point2i> endpoint_coords = extractCoordsFromMask(endpoints_mask);

  RCLCPP_DEBUG(LOGGER, "Found %zu intersections and %zu endpoints.", inter_coords.size(), endpoint_coords.size());

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
  buildGraphMultiSourceBFS(inter_positions, endpoint_positions, find_entrances, max_steps);

  // Step 4: Collect node positions
  std::unordered_map<int, std::pair<int, int>> node_positions;
  for (const auto& [nid, node] : graph_->nodes()) {
    node_positions[nid] = node.position;
  }

  RCLCPP_DEBUG(LOGGER, "Finished. Graph nodes: %zu, Edges: %zu", graph_->nodes().size(), graph_->edges().size());

  return {graph_, node_positions};
}

void SkeletonGraphBuilder::buildGraphMultiSourceBFS(
    const std::unordered_map<int, std::pair<int, int>>& intersection_positions,
    const std::unordered_map<int, std::pair<int, int>>& endpoint_positions,
    bool find_entrances,
    int max_steps) {
  // Initialize tracking structures
  cv::Mat visited_src(height_, width_, CV_32S, cv::Scalar(-1));
  cv::Mat visited_dist(height_, width_, CV_32S);
  visited_dist.setTo(std::numeric_limits<int>::max());

  // OPTIMIZATION: Use vector instead of unordered_map for O(1) parent lookup
  std::vector<std::pair<int, int>> parent_data(height_ * width_, {-1, -1});

  auto get_parent = [&](int src_id, int x, int y) -> std::pair<int, int>& {
    return parent_data[y * width_ + x];  // O(1) direct access
  };

  std::queue<std::tuple<int, int, int, int>> queue;

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

    float dist_val = distmap_.at<float>(y, x);
    int initial_budget = static_cast<int>(dist_val);

    visited_src.at<int>(y, x) = nid;
    visited_dist.at<int>(y, x) = 0;
    get_parent(nid, x, y) = {-1, -1};

    queue.push(std::make_tuple(x, y, nid, initial_budget));
  }

  RCLCPP_DEBUG(LOGGER, "BFS Queue initialized with %zu sources.", queue.size());

  // Execute BFS
  executeBFS(queue, visited_src, visited_dist, parent_data, find_entrances, max_steps);
}

void SkeletonGraphBuilder::executeBFS(
    std::queue<std::tuple<int, int, int, int>>& queue,
    cv::Mat& visited_src,
    cv::Mat& visited_dist,
    std::vector<std::pair<int, int>>& parent_data,
    bool find_entrances,
    int max_steps) {
  auto get_parent = [&](int src_id, int x, int y) -> std::pair<int, int>& {
    return parent_data[y * width_ + x];
  };

  while (!queue.empty()) {
    auto [x, y, srcid, remaining_budget] = queue.front();
    queue.pop();

    for (const auto& nb : neighbors8(x, y)) {
      int nx = nb.x;
      int ny = nb.y;

      if (skeleton_.at<uint8_t>(ny, nx) == 0) continue;

      int other_src = visited_src.at<int>(ny, nx);

      if (other_src == -1) {
        // Unvisited pixel
        int new_distance = visited_dist.at<int>(y, x) + 1;
        int new_budget = remaining_budget - 1;

        if (new_budget > 0) {
          // Budget remains, continue exploring
          visited_src.at<int>(ny, nx) = srcid;
          visited_dist.at<int>(ny, nx) = new_distance;
          get_parent(srcid, nx, ny) = {x, y};
          queue.push(std::make_tuple(nx, ny, srcid, new_budget));

        } else if (new_budget == 0) {
          // Budget exhausted exactly at this pixel
          visited_src.at<int>(ny, nx) = srcid;
          visited_dist.at<int>(ny, nx) = new_distance;
          get_parent(srcid, nx, ny) = {x, y};

          if (find_entrances) {
            int entrance_id = createEntranceNode(cv::Point2i(nx, ny), srcid, parent_data);
            visited_src.at<int>(ny, nx) = entrance_id;
            get_parent(entrance_id, nx, ny) = {-1, -1};
            queue.push(std::make_tuple(nx, ny, entrance_id, max_steps));

          } else {
            // Reset budget but keep same source
            int reset_budget = static_cast<int>(distmap_.at<float>(ny, nx));
            queue.push(std::make_tuple(nx, ny, srcid, reset_budget));
          }
        }

      } else if (other_src != srcid) {
        // Collision with another source
        bool current_budget_exhausted = (remaining_budget <= 0);
        int local_narrowness = static_cast<int>(distmap_.at<float>(ny, nx));
        bool other_budget_exhausted = (visited_dist.at<int>(ny, nx) >= local_narrowness);

        if (current_budget_exhausted && other_budget_exhausted) {
          // Both budgets exhausted - create collision node
          visited_src.at<int>(ny, nx) = srcid;
          visited_dist.at<int>(ny, nx) = visited_dist.at<int>(y, x) + 1;
          get_parent(srcid, nx, ny) = {x, y};

          int collision_id = createCollisionNode(cv::Point2i(nx, ny), srcid, other_src, parent_data);
          visited_src.at<int>(ny, nx) = collision_id;
          get_parent(collision_id, nx, ny) = {-1, -1};

          queue.push(std::make_tuple(nx, ny, collision_id, max_steps));

        } else {
          // Standard merge - create edge between sources if not exists
          int u = std::min(srcid, other_src);
          int v = std::max(srcid, other_src);

          if (!graph_->hasEdge(u, v)) {
            std::vector<cv::Point2i> path = reconstructPathCollision(
                cv::Point2i(x, y), cv::Point2i(nx, ny), srcid, other_src, parent_data);

            if (!path.empty()) {
              graph_->addEdge(u, v, path, static_cast<double>(path.size()));
            }
          }
        }
      }
    }
  }
}

int SkeletonGraphBuilder::createEntranceNode(
    const cv::Point2i& p,
    int parent_src_id,
    std::vector<std::pair<int, int>>& parent_data) {
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
    std::vector<std::pair<int, int>>& parent_data) {
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
    const std::vector<std::pair<int, int>>& parent_data) {
  std::vector<cv::Point2i> path;

  cv::Point2i current = p;
  while (true) {
    path.push_back(current);

    int idx = current.y * width_ + current.x;
    auto [px, py] = parent_data[idx];

    if (px == -1 && py == -1) break;

    current = {px, py};
  }

  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<cv::Point2i> SkeletonGraphBuilder::reconstructPathCollision(
    const cv::Point2i& cur_xy,
    const cv::Point2i& neigh_xy,
    int src_current,
    int src_neighbor,
    const std::vector<std::pair<int, int>>& parent_data) {
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

  // SAFETY CHECK 1: Validate threshold
  if (distance_threshold <= 0.0) {
    RCLCPP_DEBUG(LOGGER, "Merge threshold <= 0, skipping merge.");
    for (const auto& [nid, node] : graph_->nodes()) {
      result[nid] = node.position;
    }
    return result;
  }

  // Determine node types to merge
  std::vector<std::string> types_to_merge = node_types_to_merge;
  if (types_to_merge.empty()) {
    types_to_merge = {"intersection", "collision", "entrance"};
  }

  // OPTIMIZATION: Collect nodes AND precompute type flags for O(1) comparison
  std::vector<int> nodes_to_check;
  std::unordered_map<int, uint8_t> node_type_flags;  // Bitmask: intersection=1, entrance=2, collision=4

  nodes_to_check.reserve(graph_->nodes().size());

  for (const auto& [nid, node] : graph_->nodes()) {
    for (const auto& t : types_to_merge) {
      if (node.type == t) {
        nodes_to_check.push_back(nid);

        // Precompute type flag
        if (node.type == "intersection")
          node_type_flags[nid] = 1;
        else if (node.type == "entrance")
          node_type_flags[nid] = 2;
        else if (node.type == "collision")
          node_type_flags[nid] = 4;

        break;
      }
    }
  }

  // SAFETY CHECK 2: Minimum nodes
  if (nodes_to_check.size() < 2) {
    RCLCPP_DEBUG(LOGGER, "Less than 2 nodes to merge, skipping.");
    for (const auto& [nid, node] : graph_->nodes()) {
      result[nid] = node.position;
    }
    return result;
  }

  RCLCPP_DEBUG(LOGGER, "Merging with EUCLIDEAN + CONNECTIVITY threshold %f, checking %zu nodes.", 
               distance_threshold, nodes_to_check.size());

  // OPTIMIZATION: Build adjacency cache for O(1) connectivity checks
  std::unordered_set<long long> edge_cache;
  for (const auto& [nid, node] : graph_->nodes()) {
    for (int nb : graph_->getNeighbors(nid)) {
      int u = std::min(nid, nb);
      int v = std::max(nid, nb);
      edge_cache.insert((static_cast<long long>(u) << 32) | static_cast<uint32_t>(v));
    }
  }

  auto has_edge_cached = [&](int n1, int n2) -> bool {
    int u = std::min(n1, n2);
    int v = std::max(n1, n2);
    return edge_cache.count((static_cast<long long>(u) << 32) | static_cast<uint32_t>(v)) > 0;
  };

  // Build position array
  std::vector<std::pair<int, int>> positions;
  positions.reserve(nodes_to_check.size());
  for (int nid : nodes_to_check) {
    positions.push_back(graph_->nodes().at(nid).position);
  }

  // Union-find
  std::unordered_map<int, int> parent;
  parent.reserve(nodes_to_check.size());
  for (int id : nodes_to_check) parent[id] = id;

  std::function<int(int)> find_root = [&](int x) -> int {
    auto it = parent.find(x);
    if (it == parent.end()) return x;
    if (it->second != x) it->second = find_root(it->second);
    return it->second;
  };

  auto unite = [&](int a, int b) {
    int ra = find_root(a);
    int rb = find_root(b);
    if (ra != rb) parent[ra] = rb;
  };

  // OPTIMIZATION: Euclidean distance + connectivity check
  auto euclidean = [&](size_t i, size_t j) -> double {
    double dx = positions[i].first - positions[j].first;
    double dy = positions[i].second - positions[j].second;
    return std::sqrt(dx * dx + dy * dy);
  };

  auto types_compatible = [&](uint8_t t1, uint8_t t2) -> bool {
    // intersection (1) + intersection (1) = OK
    if ((t1 & 1) && (t2 & 1)) return true;
    // entrance (2) or collision (4) mix = OK
    if ((t1 & 6) && (t2 & 6)) return true;
    return false;
  };

  int num_connected_pairs = 0;
  int num_merged_pairs = 0;

  for (size_t i = 0; i < nodes_to_check.size(); ++i) {
    int n1 = nodes_to_check[i];

    for (size_t j = i + 1; j < nodes_to_check.size(); ++j) {
      int n2 = nodes_to_check[j];

      // CRITICAL: Check connectivity from cache (O(1))
      if (!has_edge_cached(n1, n2)) {
        continue;
      }

      num_connected_pairs++;

      double d = euclidean(i, j);
      if (d <= distance_threshold) {
        num_merged_pairs++;

        // Check type compatibility (O(1) with precomputed flags)
        uint8_t t1 = node_type_flags[n1];
        uint8_t t2 = node_type_flags[n2];

        if (types_compatible(t1, t2)) {
          unite(n1, n2);
        }
      }
    }
  }

  RCLCPP_DEBUG(LOGGER, "Found %d connected node pairs, %d within distance threshold.", 
               num_connected_pairs, num_merged_pairs);

  // Group nodes by cluster root
  std::unordered_map<int, std::vector<int>> clusters;
  clusters.reserve(nodes_to_check.size());

  for (int node_id : nodes_to_check) {
    int r = find_root(node_id);
    clusters[r].push_back(node_id);
  }

  RCLCPP_DEBUG(LOGGER, "Found %zu clusters.", clusters.size());

  std::unordered_set<int> nodes_to_remove;
  int num_merged = 0;

  for (const auto& [root, cluster] : clusters) {
    if (cluster.size() <= 1) continue;

    num_merged++;

    // Compute barycenter
    double bx = 0.0, by = 0.0;
    int cnt = 0;

    for (int node_id : cluster) {
      auto itn = graph_->nodes().find(node_id);
      if (itn == graph_->nodes().end()) continue;

      bx += itn->second.position.first;
      by += itn->second.position.second;
      cnt++;
    }

    if (cnt == 0) continue;

    bx /= cnt;
    by /= cnt;

    int gx = static_cast<int>(std::round(bx));
    int gy = static_cast<int>(std::round(by));

    auto [snapx, snapy] = snapToSkeleton(gx, gy, 10);

    // Validate snap
    if (snapx < 0 || snapx >= width_ || snapy < 0 || snapy >= height_) {
      RCLCPP_WARN(LOGGER, "Snap out of bounds (%d, %d), skipping cluster.", snapx, snapy);
      continue;
    }

    if (skeleton_.at<uint8_t>(snapy, snapx) == 0) {
      RCLCPP_WARN(LOGGER, "Snap not on skeleton at (%d, %d), skipping cluster.", snapx, snapy);
      continue;
    }

    // Create merged node
    int merged_id = next_node_id_++;
    std::string merged_type = "intersection";

    // Check if this cluster contains entrance/collision nodes
    bool is_entrance_group = false;
    for (int id : cluster) {
      std::string t = graph_->nodes().at(id).type;
      if (t == "entrance" || t == "collision") {
        is_entrance_group = true;
        break;
      }
    }

    if (is_entrance_group) {
      merged_type = "entrance";
    }

    graph_->addNode(merged_id, {snapx, snapy}, merged_type);

    // Find external neighbors
    std::unordered_set<int> external_neighbors;

    for (int node_id : cluster) {
      try {
        for (int nb : graph_->getNeighbors(node_id)) {
          if (std::find(cluster.begin(), cluster.end(), nb) == cluster.end()) {
            external_neighbors.insert(nb);
          }
        }
      } catch (...) {
      }
    }

    // Reconnect to external neighbors using best edge from cluster
    for (int nb : external_neighbors) {
      if (graph_->nodes().find(nb) == graph_->nodes().end()) continue;

      double best_w = std::numeric_limits<double>::infinity();
      std::vector<std::pair<int, int>> best_path;
      bool found = false;

      for (int node_id : cluster) {
        if (graph_->nodes().find(node_id) == graph_->nodes().end()) continue;

        const auto e_fwd = graph_->getEdge(node_id, nb);
        const auto e_bwd = graph_->getEdge(nb, node_id);

        if (e_fwd && e_fwd->weight < best_w) {
          best_w = e_fwd->weight;
          best_path = e_fwd->path_pixels;
          found = true;
        }

        if (e_bwd && e_bwd->weight < best_w) {
          best_w = e_bwd->weight;
          best_path = e_bwd->path_pixels;
          found = true;
        }
      }

      if (found && !best_path.empty()) {
        std::vector<std::pair<int, int>> new_path;
        new_path.emplace_back(snapx, snapy);

        auto nbpos = graph_->nodes().at(nb).position;
        new_path.emplace_back(nbpos.first, nbpos.second);

        double new_weight = static_cast<double>(new_path.size());
        graph_->addEdge(merged_id, nb, new_path, new_weight);
      }
    }

    // Mark cluster nodes for removal
    for (int node_id : cluster) {
      nodes_to_remove.insert(node_id);
    }
  }

  // Remove old nodes
  RCLCPP_DEBUG(LOGGER, "Removing %zu old nodes...", nodes_to_remove.size());

  for (int node_id : nodes_to_remove) {
    try {
      graph_->removeNode(node_id);
    } catch (...) {
    }
  }

  // Changed to DEBUG (graph building output)
  RCLCPP_DEBUG(LOGGER, "Merged %d clusters. Final: %zu nodes, %zu edges.", 
               num_merged, graph_->nodes().size(), graph_->edges().size());

  // Return updated node positions
  for (const auto& [nid, node] : graph_->nodes()) {
    result[nid] = node.position;
  }

  return result;
}

std::pair<int, int> SkeletonGraphBuilder::snapToSkeleton(int x, int y, int max_search_radius) {
  x = std::max(0, std::min(x, width_ - 1));
  y = std::max(0, std::min(y, height_ - 1));

  if (skeleton_.at<uint8_t>(y, x) > 0) {
    return {x, y};
  }

  // Spiral search outward
  for (int radius = 1; radius <= max_search_radius; ++radius) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        int ny = y + dy;
        int nx = x + dx;

        if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
          if (skeleton_.at<uint8_t>(ny, nx) > 0) {
            return {nx, ny};
          }
        }
      }
    }
  }

  // Fallback: return clamped original position
  return {x, y};
}

void SkeletonGraphBuilder::findSkeletonPoints(cv::Mat& intersections_mask, cv::Mat& endpoints_mask) {
  // Ensure strict binary (0 or 1)
  cv::Mat skel_bin;
  cv::threshold(skeleton_, skel_bin, 0, 1, cv::THRESH_BINARY);

  // Endpoints detection
  cv::Mat kernel = (cv::Mat_<int>(3, 3) << 1, 1, 1, 1, 10, 1, 1, 1, 1);
  cv::Mat neighbor_count;
  cv::filter2D(skel_bin, neighbor_count, CV_16S, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

  endpoints_mask = (neighbor_count == 11) & (skeleton_ != 0);

  // Intersection detection using Hit-or-Miss
  intersections_mask = cv::Mat::zeros(skeleton_.size(), CV_8U);

  uint8_t T_d[] = {0, 1, 0, 1, 1, 1, 0, 0, 0};
  cv::Mat T(3, 3, CV_8U, T_d);
  uint8_t Y_d[] = {1, 0, 1, 0, 1, 0, 0, 1, 0};
  cv::Mat Y(3, 3, CV_8U, Y_d);
  uint8_t HX_d[] = {1, 0, 1, 0, 1, 0, 1, 0, 0};
  cv::Mat HX(3, 3, CV_8U, HX_d);
  uint8_t TY_d[] = {0, 0, 1, 1, 1, 0, 0, 1, 0};
  cv::Mat TY(3, 3, CV_8U, TY_d);
  uint8_t YT_d[] = {1, 0, 0, 0, 1, 1, 0, 1, 0};
  cv::Mat YT(3, 3, CV_8U, YT_d);
  uint8_t YH_d[] = {1, 0, 1, 0, 1, 1, 0, 1, 0};
  cv::Mat YH(3, 3, CV_8U, YH_d);
  uint8_t HY_d[] = {1, 0, 1, 1, 1, 0, 0, 1, 0};
  cv::Mat HY(3, 3, CV_8U, HY_d);
  uint8_t P_d[] = {0, 1, 0, 1, 1, 1, 0, 1, 0};
  cv::Mat plus(3, 3, CV_8U, P_d);
  uint8_t X_d[] = {1, 0, 1, 0, 1, 0, 1, 0, 1};
  cv::Mat X(3, 3, CV_8U, X_d);

  std::vector<cv::Mat> templates = {T, Y, TY, YT, HX, YH, HY, plus};

  for (const auto& tmpl : templates) {
    cv::Mat current = tmpl.clone();

    for (int i = 0; i < 4; ++i) {
      cv::Mat hit, miss;

      cv::morphologyEx(skeleton_, hit, cv::MORPH_ERODE, current);

      cv::Mat ones = cv::Mat::ones(3, 3, CV_8U);
      cv::morphologyEx(255 - skeleton_, miss, cv::MORPH_ERODE, ones - current);

      intersections_mask = intersections_mask | (hit & miss);

      cv::Mat rotated;
      cv::rotate(current, rotated, cv::ROTATE_90_CLOCKWISE);
      current = rotated;
    }
  }

  // Also apply X pattern
  {
    cv::Mat hit, miss;
    cv::Mat ones = cv::Mat::ones(3, 3, CV_8U);

    cv::morphologyEx(skeleton_, hit, cv::MORPH_ERODE, X);
    cv::morphologyEx(255 - skeleton_, miss, cv::MORPH_ERODE, ones - X);

    intersections_mask = intersections_mask | (hit & miss);
  }
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

  for (int k = 0; k < 8; ++k) {
    int nx = x + dx8[k];
    int ny = y + dy8[k];

    if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
      nbs.emplace_back(nx, ny);
    }
  }

  return nbs;
}

long long SkeletonGraphBuilder::packKey(int src_id, int x, int y) {
  return (static_cast<long long>(src_id) << 32) | ((static_cast<long long>(y) & 0xFFFF) << 16) |
         (static_cast<long long>(x) & 0xFFFF);
}

}  // namespace graph_generator_node