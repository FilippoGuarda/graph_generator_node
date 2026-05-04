#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>
#include <rclcpp/rclcpp.hpp> 
#include "network_x.hpp"

namespace graph_generator_node {

/**
 * @class SkeletonGraphBuilder
 * @brief Builds a topological graph from skeleton images using multi-source BFS
 * 
 * Features:
 * - Multi-source BFS with budget-based propagation
 * - Automatic entrance node creation at budget boundaries
 * - Collision detection and node creation for intersecting paths
 * - Adaptive merging of close nodes with connectivity checks
 */
class SkeletonGraphBuilder {
public:
  /**
   * @brief Constructor
   * @param skeleton Binary skeleton image (H×W, 0=background, >0=skeleton)
   * @param distmap Distance transform of skeleton for narrowness estimation
   */
    SkeletonGraphBuilder(const cv::Mat& skeleton, const cv::Mat& distmap);

  /**
   * @brief Build graph from skeleton with multi-source BFS
   * 
   * Performs the following steps:
   * 1. Find skeleton intersection and endpoint nodes
   * 2. Initialize multi-source BFS with budget allocation
   * 3. Propagate from each source with budget-based termination
   * 4. Create entrance nodes at budget boundaries if enabled
   * 5. Create collision nodes where paths from different sources meet
   * 
   * @param max_steps Maximum propagation budget per source (typically 2x robot diameter)
   * @param find_entrances Create entrance nodes at budget boundaries (default: true)
   * @return Pair of (graph pointer, node_positions_map)
   */
    std::pair<std::shared_ptr<NetworkX>, std::unordered_map<int, std::pair<int, int>>>
    buildGraph(int max_steps = 100, int hysteresis = 20, bool find_entrances = true);

  /**
   * @brief Merge nodes within distance threshold
   * 
   * Merges nodes that are:
   * 1. Directly connected in the graph (edge exists between them)
   * 2. Within Euclidean distance threshold
   * 3. Compatible types (intersections merge with intersections, etc.)
   * 
   * Uses union-find clustering and connectivity cache for O(1) edge lookups.
   * 
   * @param distance_threshold Euclidean distance for merging (in pixels)
   * @param node_types_to_merge Which node types to consider (default: all except endpoints)
   * @return Updated node positions map after merging
   */
    std::unordered_map<int, std::pair<int, int>>
    mergeCloseNodes(double distance_threshold, const std::vector<std::string>& node_types_to_merge = {});

private:
    cv::Mat skeleton_;
    cv::Mat distmap_;
    int height_;
    int width_;

    std::shared_ptr<NetworkX> graph_;
    int next_node_id_;

  // ===== Private Methods =====

  /**
   * @brief Execute multi-source BFS from all skeleton sources
   * 
   * Each source propagates with its budget. When budgets exhaust simultaneously,
   * collision nodes are created. Entrances mark budget-only boundaries.
   */
    void buildGraphMultiSourceBFS(
        const std::unordered_map<int, std::pair<int, int>>& intersection_positions,
        const std::unordered_map<int, std::pair<int, int>>& endpoint_positions,
        bool find_entrances,
        int max_steps,
        int hysteresis);

  /**
   * @brief Execute multi-source BFS from queue
   * 
   * @param queue BFS queue of (x, y, source_id, remaining_budget)
   * @param visited_src Map of (x, y) -> source_id at that pixel
   * @param visited_dist Map of (x, y) -> distance from source to that pixel
   * @param parent_data Vector-indexed parent tracking: parent_data[y*width + x] = (px, py)
   * @param find_entrances Create entrance nodes at budget boundaries
   * @param hysteresis Budget hysteresis (default: 20)
   * @param max_steps Global maximum budget (for collision node propagation)
   */
    void executeBFS(
        std::queue<std::tuple<int, int, int, int, bool>>& queue,
        cv::Mat& visited_src,
        cv::Mat& visited_dist,
        std::unordered_map<uint64_t, std::pair<int, int>>& parent_data,
        bool find_entrances,
        int max_steps,
        int hysteresis);

      /**
   * @brief Create entrance node at budget boundary
   * @param p Pixel coordinates where entrance is created
   * @param parent_src_id Source ID that created this entrance
   * @param parent_data Vector-indexed parent tracking
   * @return New entrance node ID
   */
  int createEntranceNode(const cv::Point2i& p, 
                         int parent_src_id,
                         std::unordered_map<uint64_t, std::pair<int, int>>& parent_data);

    /**
   * @brief Create collision node where two sources meet with exhausted budgets
   * @param p Collision point coordinates
   * @param src1 First source ID
   * @param src2 Second source ID
   * @param parent_data Vector-indexed parent tracking
   * @return New collision node ID
   */
  int createCollisionNode(const cv::Point2i& p, 
                          int src1, 
                          int src2,
                          std::unordered_map<uint64_t, std::pair<int, int>>& parent_data);

      /**
   * @brief Reconstruct path from pixel back to source by following parent pointers
   * @param p End point (x, y)
   * @param src_id Source ID (not used in this implementation, for API compatibility)
   * @param parent_data Vector-indexed parent tracking
   * @return Path as vector of (x, y) points from source to p
   */
  std::vector<cv::Point2i> reconstructPath(
      const cv::Point2i& p,
      int src_id,
      const std::unordered_map<uint64_t, std::pair<int, int>>& parent_data);

    /**
   * @brief Reconstruct path when two sources collide
   * Combines path from first source backward to collision point
   * with path from second source backward to collision point
   * @param cur_xy Current position in first source path
   * @param neigh_xy Position in second source path
   * @param src_current Current source ID
   * @param src_neighbor Neighbor source ID
   * @param parent_data Vector-indexed parent tracking
   * @return Combined collision path
   */
  std::vector<cv::Point2i> reconstructPathCollision(
      const cv::Point2i& cur_xy,
      const cv::Point2i& neigh_xy,
      int src_current,
      int src_neighbor,
      const std::unordered_map<uint64_t, std::pair<int, int>>& parent_data);

  /**
   * @brief Find skeleton intersection and endpoint points
   * Uses morphological hit-or-miss operations to detect junction types
   * @param[out] intersections_mask Binary mask of intersection points
   * @param[out] endpoints_mask Binary mask of endpoint points
   */
    void findSkeletonPoints(cv::Mat& intersections_mask, cv::Mat& endpoints_mask);

    /**
   * @brief Extract (x, y) coordinates from binary mask
   * @param mask Binary mask where non-zero pixels are selected
   * @return Vector of (x, y) coordinates for each selected pixel
   */
    std::vector<cv::Point2i> extractCoordsFromMask(const cv::Mat& mask);

  /**
   * @brief Get 8-connected neighbors of a pixel
   * @param x X coordinate
   * @param y Y coordinate
   * @return Vector of 8 neighboring (x, y) positions (with bounds checking)
   */
    std::vector<cv::Point2i> neighbors8(int x, int y);

  /**
   * @brief Snap a point to nearest skeleton pixel using spiral search
   * @param x X coordinate (will be clamped to bounds)
   * @param y Y coordinate (will be clamped to bounds)
   * @param max_search_radius Maximum search radius in pixels
   * @return (x, y) coordinates of snapped point on skeleton, or clamped original if not found
   */
    std::pair<int, int> snapToSkeleton(int x, int y, int max_search_radius = 10);

  /**
   * @brief Find skeleton path between two points using BFS
   * @param start_pos Starting point (x, y)
   * @param end_pos Ending point (x, y)
   * @return Vector of (x, y) points along the path
   */
    std::vector<std::pair<int, int>> findSkeletonPath(
    std::pair<int, int> start_pos, 
    std::pair<int, int> end_pos);
  /**
   * @brief Pack three integers into a single 64-bit key
   * Used for parent tracking in legacy code (now replaced by vector indexing)
   * @param src_id Source ID
   * @param x X coordinate
   * @param y Y coordinate
   * @return Packed 64-bit key
   */
    long long packKey(int src_id, int x, int y);

  /**
   * @brief Prune branches shorter than max_length from skeleton
   * @param max_length Maximum path length to keep
   * @return Skeleton with removed unconnected branches
   */
    void pruneShortBranches(int max_length);

};

} // namespace graph_generator_node
