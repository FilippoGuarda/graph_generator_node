// Copyright 2025 Filippo Guarda
// Licensed under the Apache License, Version 2.0

#include "graph_generator_node/skeleton_graph_builder.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace graph_generator_node {

SkeletonGraphBuilder::SkeletonGraphBuilder(const cv::Mat& skeleton,
                                           const cv::Mat& dist_map)
  : skeleton_(skeleton.clone()), dist_map_(dist_map.clone()),
    height_(skeleton.rows), width_(skeleton.cols), next_node_id_(0) {
  // Ensure skeleton is binary (0 or 1)
  cv::threshold(skeleton_, skeleton_, 0, 1, cv::THRESH_BINARY);
}

// ============================================================================
// PUBLIC API
// ============================================================================

std::pair<std::vector<GraphNode>, std::vector<GraphEdge>>
SkeletonGraphBuilder::buildGraph(int max_bfs_steps, bool find_entrances,
                                  float merge_threshold) {
  nodes_.clear();
  edges_.clear();
  next_node_id_ = 0;

  // Step 1: Find initial skeleton points (intersections and endpoints)
  cv::Mat inter_mask, endpoint_mask;
  findSkeletonPoints(inter_mask, endpoint_mask);

  auto inter_coords = extractCoordsFromMask(inter_mask);
  auto endpoint_coords = extractCoordsFromMask(endpoint_mask);

  // Step 2: Create initial nodes
  std::vector<GraphNode> initial_nodes;

  for (const auto& p : inter_coords) {
    GraphNode n;
    n.id = next_node_id_++;
    n.pix = cv::Point2i(p.x, p.y);
    n.type = "intersection";
    initial_nodes.push_back(n);
  }

  for (const auto& p : endpoint_coords) {
    GraphNode n;
    n.id = next_node_id_++;
    n.pix = cv::Point2i(p.x, p.y);
    n.type = "endpoint";
    initial_nodes.push_back(n);
  }

  if (initial_nodes.empty()) {
    std::cerr << "WARNING: No initial skeleton points found\n";
    return {nodes_, edges_};
  }

  // Step 3: Execute BFS to build graph
  executeBFS(initial_nodes, find_entrances, max_bfs_steps);

  return {nodes_, edges_};
}

// ============================================================================
// SKELETON ANALYSIS
// ============================================================================

void SkeletonGraphBuilder::findSkeletonPoints(cv::Mat& intersections_mask,
                                               cv::Mat& endpoints_mask) {
  // Binarize skeleton
  cv::Mat skel_bin;
  cv::threshold(skeleton_, skel_bin, 0, 1, cv::THRESH_BINARY);

  // Count neighbors: kernel weights center 10x, neighbors 1x each
  cv::Mat kernel =
    (cv::Mat_<int>(3, 3) << 1, 1, 1, 1, 10, 1, 1, 1, 1);
  cv::Mat neighbor_count;
  cv::filter2D(skel_bin, neighbor_count, CV_16S, kernel, cv::Point(-1, -1), 0,
               cv::BORDER_CONSTANT);

  // Endpoint: exactly 1 neighbor (center weighted 10 + 1 neighbor = 11)
  endpoints_mask = cv::Mat::zeros(skeleton_.size(), CV_8U);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      if (neighbor_count.at<short>(y, x) == 11 && skel_bin.at<uchar>(y, x) > 0) {
        endpoints_mask.at<uchar>(y, x) = 255;
      }
    }
  }

  // Detect junctions via template matching
  detectAllJunctions(intersections_mask);
}

void SkeletonGraphBuilder::detectAllJunctions(cv::Mat& junctions_mask) {
  junctions_mask = cv::Mat::zeros(skeleton_.size(), CV_8U);

  // Binarize skeleton
  cv::Mat skel_bin;
  cv::threshold(skeleton_, skel_bin, 0, 1, cv::THRESH_BINARY);

  // Helper lambda for hit-or-miss: properly handling binary patterns
  // Pattern value: 0 = must be background, 1 = must be foreground, -1 = don't care
  auto applyHitOrMiss = [&](const cv::Mat& pattern) {
    for (int y = 1; y < height_ - 1; ++y) {
      for (int x = 1; x < width_ - 1; ++x) {
        if (skel_bin.at<uchar>(y, x) == 0) continue; // Center must be skeleton

        bool match = true;
        for (int py = -1; py <= 1; ++py) {
          for (int px = -1; px <= 1; ++px) {
            int pat_val = pattern.at<char>(py + 1, px + 1);
            if (pat_val == -1) continue; // Don't care

            int skel_val = skel_bin.at<uchar>(y + py, x + px);
            if (pat_val == 1 && skel_val == 0) {
              match = false;
              break;
            }
            if (pat_val == 0 && skel_val != 0) {
              match = false;
              break;
            }
          }
          if (!match) break;
        }

        if (match) {
          junctions_mask.at<uchar>(y, x) = 255;
        }
      }
    }
  };

  // Define template patterns (using -1 for "don't care")
  std::vector<cv::Mat> base_templates;

  // T-junction
  base_templates.push_back((cv::Mat_<char>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 0, 0));
  // Y-junction
  base_templates.push_back((cv::Mat_<char>(3, 3) << 1, 0, 1, 0, 1, 0, 0, 1, 0));
  // Asymmetric patterns (HX, TY, YT, YH, HY)
  base_templates.push_back(
    (cv::Mat_<char>(3, 3) << 1, 0, 1, 0, 1, 0, 1, 0, 0));
  base_templates.push_back(
    (cv::Mat_<char>(3, 3) << 0, 0, 1, 1, 1, 0, 0, 1, 0));
  base_templates.push_back(
    (cv::Mat_<char>(3, 3) << 1, 0, 0, 0, 1, 1, 0, 1, 0));
  base_templates.push_back(
    (cv::Mat_<char>(3, 3) << 1, 0, 1, 0, 1, 1, 0, 1, 0));
  base_templates.push_back(
    (cv::Mat_<char>(3, 3) << 1, 0, 1, 1, 1, 0, 0, 1, 0));

  // Apply all rotations
  for (const auto& templ : base_templates) {
    cv::Mat current = templ.clone();

    for (int rotation = 0; rotation < 4; ++rotation) {
      applyHitOrMiss(current);

      if (rotation < 3) {
        // Rotate 90 degrees clockwise
        cv::Mat rotated;
        cv::rotate(current, rotated, cv::ROTATE_90_CLOCKWISE);
        // For a 3x3 pattern, cv::rotate handles it fine
        current = rotated;
      }
    }
  }

  // Plus and X patterns (4-way and diagonal)
  cv::Mat plus = (cv::Mat_<char>(3, 3) << 0, 1, 0, 1, 1, 1, 0, 1, 0);
  cv::Mat x_pat = (cv::Mat_<char>(3, 3) << 1, 0, 1, 0, 1, 0, 1, 0, 1);

  applyHitOrMiss(plus);
  applyHitOrMiss(x_pat);
}

std::vector<cv::Point2i> SkeletonGraphBuilder::extractCoordsFromMask(
  const cv::Mat& mask) {
  std::vector<cv::Point2i> coords;
  for (int y = 0; y < mask.rows; ++y) {
    for (int x = 0; x < mask.cols; ++x) {
      if (mask.at<uchar>(y, x) > 0) {
        coords.push_back(cv::Point2i(x, y));
      }
    }
  }
  return coords;
}

// ============================================================================
// BFS EXECUTION (LEVEL-SYNCHRONIZED, BUDGET-BASED)
// ============================================================================

void SkeletonGraphBuilder::executeBFS(const std::vector<GraphNode>& initial_nodes,
                                       bool find_entrances, int max_bfs_steps) {
  // Initialize tracking arrays
  visited_src_ = cv::Mat::ones(height_, width_, CV_32S) * -1;
  visited_dist_ = cv::Mat(height_, width_, CV_32S, cv::Scalar(INT_MAX));
  parent_.clear();

  // Add initial nodes to graph and queue
  std::queue<BFSState> q;
  nodes_ = initial_nodes;

  for (const auto& node : initial_nodes) {
    int x = node.pix.x;
    int y = node.pix.y;

    // CRITICAL: Don't filter valid initial nodes!
    // Accept all initial nodes from skeleton point detection
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
      if (skeleton_.at<uchar>(y, x) > 0) {
        // Initialize with local distance as budget
        int initial_budget = static_cast<int>(dist_map_.at<float>(y, x));

        visited_src_.at<int>(y, x) = node.id;
        visited_dist_.at<int>(y, x) = 0;
        parent_[packKey(node.id, x, y)] = cv::Point2i(-1, -1);

        // Create BFS state with distance tracking for level-sync
        q.push(BFSState(x, y, node.id, initial_budget, 0));
      }
    }
  }

  // Execute level-synchronized BFS
  while (!q.empty()) {
    BFSState state = q.front();
    q.pop();

    int x = state.x;
    int y = state.y;
    int src_id = state.src_id;
    int remaining_budget = state.remaining_budget;

    // Expand to all 8-neighbors
    auto neighbors = neighbors8(x, y);
    for (const auto& neighbor : neighbors) {
      int nx = neighbor.x;
      int ny = neighbor.y;

      // Skip if not on skeleton
      if (skeleton_.at<uchar>(ny, nx) == 0) continue;

      int other_src = visited_src_.at<int>(ny, nx);

      if (other_src == -1) {
        // Unvisited pixel
        if (!processUnvisitedPixel(x, y, nx, ny, src_id, remaining_budget,
                                    find_entrances, max_bfs_steps, q)) {
          continue;
        }
      } else if (other_src != src_id) {
        // Collision with different source
        processCollision(x, y, nx, ny, src_id, remaining_budget, other_src,
                         max_bfs_steps, q);
      }
    }
  }
}

bool SkeletonGraphBuilder::processUnvisitedPixel(int x, int y, int nx, int ny,
                                                  int src_id, int remaining_budget,
                                                  bool find_entrances,
                                                  int max_bfs_steps,
                                                  std::queue<BFSState>& queue) {
  int new_distance = visited_dist_.at<int>(y, x) + 1;
  int new_budget = remaining_budget - 1;

  if (new_budget > 0) {
    // Continue with current source
    visited_src_.at<int>(ny, nx) = src_id;
    visited_dist_.at<int>(ny, nx) = new_distance;
    parent_[packKey(src_id, nx, ny)] = cv::Point2i(x, y);
    queue.push(BFSState(nx, ny, src_id, new_budget, new_distance));

  } else if (new_budget == 0) {
    // Budget EXACTLY exhausted - entrance condition
    visited_src_.at<int>(ny, nx) = src_id;
    visited_dist_.at<int>(ny, nx) = new_distance;
    parent_[packKey(src_id, nx, ny)] = cv::Point2i(x, y);

    if (find_entrances) {
      // Create entrance node at this boundary
      int entrance_id = createEntranceNode(cv::Point2i(nx, ny), src_id);

      visited_src_.at<int>(ny, nx) = entrance_id;
      parent_[packKey(entrance_id, nx, ny)] = cv::Point2i(-1, -1);

      // Entrance propagates with fresh budget
      int entrance_budget = static_cast<int>(dist_map_.at<float>(ny, nx));
      queue.push(BFSState(nx, ny, entrance_id, entrance_budget, 0));
    } else {
      // No entrances: reset budget and continue with same source
      int reset_budget = static_cast<int>(dist_map_.at<float>(ny, nx));
      queue.push(BFSState(nx, ny, src_id, reset_budget, new_distance));
    }

  } else {
    // new_budget < 0 (shouldn't happen in normal flow, but handle gracefully)
    visited_src_.at<int>(ny, nx) = src_id;
    visited_dist_.at<int>(ny, nx) = new_distance;
    parent_[packKey(src_id, nx, ny)] = cv::Point2i(x, y);
  }

  return true;
}

void SkeletonGraphBuilder::processCollision(int x, int y, int nx, int ny,
                                             int src_id, int remaining_budget,
                                             int other_src, int max_bfs_steps,
                                             std::queue<BFSState>& queue) {
  // Check budget exhaustion for both sources
  bool current_budget_exhausted = (remaining_budget <= 0);
  int local_narrowness = static_cast<int>(dist_map_.at<float>(ny, nx));
  int other_distance = visited_dist_.at<int>(ny, nx);

  // other_budget_exhausted: has other source traveled at least as far as local narrowness?
  bool other_budget_exhausted = (other_distance >= local_narrowness);

  if (current_budget_exhausted && other_budget_exhausted) {
    // Both exhausted - create collision node
    visited_src_.at<int>(ny, nx) = src_id;
    visited_dist_.at<int>(ny, nx) = visited_dist_.at<int>(y, x) + 1;
    parent_[packKey(src_id, nx, ny)] = cv::Point2i(x, y);

    int collision_id = createCollisionNode(cv::Point2i(nx, ny), src_id, other_src);

    visited_src_.at<int>(ny, nx) = collision_id;
    parent_[packKey(collision_id, nx, ny)] = cv::Point2i(-1, -1);
    visited_dist_.at<int>(ny, nx) = 0;

    // Collision propagates with fresh budget
    queue.push(BFSState(nx, ny, collision_id, max_bfs_steps, 0));

  } else {
    // Standard collision: create edge between sources
    int u = std::min(src_id, other_src);
    int v = std::max(src_id, other_src);

    // Check if edge already exists
    bool edge_exists = false;
    for (const auto& e : edges_) {
      if ((e.u == u && e.v == v) || (e.u == v && e.v == u)) {
        edge_exists = true;
        break;
      }
    }

    if (!edge_exists) {
      auto path = reconstructPathCollision(cv::Point2i(x, y), cv::Point2i(nx, ny),
                                            src_id, other_src);
      if (!path.empty()) {
        GraphEdge edge;
        edge.u = u;
        edge.v = v;
        edge.path_pixels = path;
        edge.weight = static_cast<float>(path.size());
        edges_.push_back(edge);
      }
    }
  }
}

// ============================================================================
// NODE CREATION
// ============================================================================

int SkeletonGraphBuilder::createEntranceNode(const cv::Point2i& p,
                                              int parent_src_id) {
  int entrance_id = next_node_id_++;

  GraphNode n;
  n.id = entrance_id;
  n.pix = p;
  n.type = "entrance";
  nodes_.push_back(n);

  // Connect to parent source
  auto path = reconstructPath(p, parent_src_id);
  if (!path.empty()) {
    GraphEdge e;
    e.u = parent_src_id;
    e.v = entrance_id;
    e.path_pixels = path;
    e.weight = static_cast<float>(path.size());
    edges_.push_back(e);
  }

  return entrance_id;
}

int SkeletonGraphBuilder::createCollisionNode(const cv::Point2i& p, int src1,
                                               int src2) {
  int collision_id = next_node_id_++;

  GraphNode n;
  n.id = collision_id;
  n.pix = p;
  n.type = "collision";
  nodes_.push_back(n);

  // Connect to both sources
  for (int src : {src1, src2}) {
    if (std::find_if(edges_.begin(), edges_.end(), [src, collision_id](const GraphEdge& e) {
          return (e.u == src && e.v == collision_id) ||
                 (e.u == collision_id && e.v == src);
        }) == edges_.end()) {
      auto path = reconstructPath(p, src);
      if (!path.empty()) {
        GraphEdge e;
        e.u = src;
        e.v = collision_id;
        e.path_pixels = path;
        e.weight = static_cast<float>(path.size());
        edges_.push_back(e);
      }
    }
  }

  return collision_id;
}

// ============================================================================
// PATH RECONSTRUCTION
// ============================================================================

std::vector<cv::Point2i> SkeletonGraphBuilder::reconstructPath(const cv::Point2i& p,
                                                                int src_id) {
  std::vector<cv::Point2i> path;
  cv::Point2i current = p;

  while (true) {
    path.push_back(current);

    int64_t key = packKey(src_id, current.x, current.y);
    auto it = parent_.find(key);

    if (it == parent_.end()) break;

    cv::Point2i parent_pix = it->second;
    if (parent_pix.x < 0 || parent_pix.y < 0) break;

    current = parent_pix;
  }

  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<cv::Point2i> SkeletonGraphBuilder::reconstructPathCollision(
  const cv::Point2i& curr_xy, const cv::Point2i& neigh_xy, int src_current,
  int src_neighbor) {
  auto path_current = reconstructPath(curr_xy, src_current);
  auto path_neighbor = reconstructPath(neigh_xy, src_neighbor);

  std::reverse(path_neighbor.begin(), path_neighbor.end());
  path_current.insert(path_current.end(), path_neighbor.begin(),
                      path_neighbor.end());

  return path_current;
}

// ============================================================================
// UTILITY
// ============================================================================

std::vector<cv::Point2i> SkeletonGraphBuilder::neighbors8(int x, int y) {
  std::vector<cv::Point2i> neighbors;
  static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  static const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

  for (int i = 0; i < 8; ++i) {
    int nx = x + dx[i];
    int ny = y + dy[i];
    if (nx >= 0 && nx < width_ && ny >= 0 && ny < height_) {
      neighbors.push_back(cv::Point2i(nx, ny));
    }
  }
  return neighbors;
}

} // namespace graph_generator_node
