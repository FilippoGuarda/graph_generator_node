// Copyright 2025 Filippo Guarda
// Licensed under the Apache License, Version 2.0

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <queue>
#include <cstdint>

namespace graph_generator_node {

/**
 * @struct GraphNode
 * Represents a node in the topological graph
 */
struct GraphNode {
  int id;
  cv::Point2i pix;  // Grid coordinates (x, y)
  std::string type; // "intersection", "endpoint", "entrance", "collision"
};

/**
 * @struct GraphEdge
 * Represents an edge in the topological graph
 */
struct GraphEdge {
  int u;
  int v;
  std::vector<cv::Point2i> path_pixels;
  float weight;
};

/**
 * @struct BFSState
 * Encapsulates BFS queue state
 */
struct BFSState {
  int x;
  int y;
  int src_id;      // Source node ID
  int remaining_budget;
  int distance;    // Distance from source (for level-sync)

  BFSState(int x_, int y_, int src_id_, int budget_, int dist_ = 0)
    : x(x_), y(y_), src_id(src_id_), remaining_budget(budget_), distance(dist_) {}
};

/**
 * @class SkeletonGraphBuilder
 * Builds topological graph from binary skeleton using budget-based BFS
 */
class SkeletonGraphBuilder {
public:
  /**
   * Constructor
   * @param skeleton Binary skeleton image (0=background, >0=skeleton)
   * @param dist_map Distance transform of the free space
   */
  SkeletonGraphBuilder(const cv::Mat& skeleton, const cv::Mat& dist_map);

  /**
   * Build complete topological graph
   * @param max_bfs_steps Maximum propagation steps for entrances
   * @param find_entrances Whether to detect entrance nodes
   * @param merge_threshold Distance threshold for node clustering (optional)
   * @return Pair of (nodes, edges)
   */
  std::pair<std::vector<GraphNode>, std::vector<GraphEdge>>
  buildGraph(int max_bfs_steps = 100,
             bool find_entrances = true,
             float merge_threshold = 0.0f);

private:
  // Input data
  cv::Mat skeleton_;
  cv::Mat dist_map_;
  int height_;
  int width_;

  // BFS tracking
  cv::Mat visited_src_;      // Source ID for each pixel (-1 = unvisited)
  cv::Mat visited_dist_;     // Distance from source for each pixel
  std::unordered_map<int64_t, cv::Point2i> parent_; // (src_id, x*W+y) -> parent

  // Graph data
  std::vector<GraphNode> nodes_;
  std::vector<GraphEdge> edges_;
  int next_node_id_;

  // ===== SKELETON ANALYSIS =====

  /**
   * Find intersection and endpoint pixels in skeleton
   * @param intersections_mask Output: binary mask of intersections
   * @param endpoints_mask Output: binary mask of endpoints
   */
  void findSkeletonPoints(cv::Mat& intersections_mask, cv::Mat& endpoints_mask);

  /**
   * Detect junctions using hit-or-miss templates
   * Correctly implements template matching with proper rotation and binary logic
   */
  void detectAllJunctions(cv::Mat& junctions_mask);

  /**
   * Extract (x, y) coordinates from binary mask
   */
  std::vector<cv::Point2i> extractCoordsFromMask(const cv::Mat& mask);

  // ===== BFS EXECUTION =====

  /**
   * Execute level-synchronized multi-source BFS
   * @param initial_nodes Starting intersection/endpoint nodes
   * @param find_entrances Whether to create entrance nodes
   * @param max_bfs_steps Max steps for entrance propagation
   */
  void executeBFS(const std::vector<GraphNode>& initial_nodes,
                  bool find_entrances,
                  int max_bfs_steps);

  /**
   * Process unvisited pixel during BFS
   * @return True if pixel was claimed, False otherwise
   */
  bool processUnvisitedPixel(int x, int y, int nx, int ny,
                              int src_id, int remaining_budget,
                              bool find_entrances, int max_bfs_steps,
                              std::queue<BFSState>& queue);

  /**
   * Process collision between two source wavefronts
   */
  void processCollision(int x, int y, int nx, int ny,
                        int src_id, int remaining_budget,
                        int other_src, int max_bfs_steps,
                        std::queue<BFSState>& queue);

  // ===== NODE CREATION =====

  /**
   * Create entrance node when budget exhausted
   * @param p Pixel position for entrance
   * @param parent_src_id Source that exhausted its budget
   * @return New node ID
   */
  int createEntranceNode(const cv::Point2i& p, int parent_src_id);

  /**
   * Create collision node when two sources meet with exhausted budgets
   */
  int createCollisionNode(const cv::Point2i& p, int src1, int src2);

  // ===== PATH RECONSTRUCTION =====

  /**
   * Reconstruct path from pixel back to source
   */
  std::vector<cv::Point2i> reconstructPath(const cv::Point2i& p, int src_id);

  /**
   * Reconstruct path when two sources collide
   */
  std::vector<cv::Point2i> reconstructPathCollision(
    const cv::Point2i& curr_xy,
    const cv::Point2i& neigh_xy,
    int src_current,
    int src_neighbor);

  // ===== UTILITY =====

  /**
   * Get 8-connected neighbors of (x, y)
   */
  std::vector<cv::Point2i> neighbors8(int x, int y);

  /**
   * Pack (src_id, x, y) into single key for map
   */
  int64_t packKey(int src_id, int x, int y) const {
    return ((int64_t)src_id << 32) | ((int64_t)x << 16) | (int64_t)y;
  }
};

} // namespace graph_generator_node
