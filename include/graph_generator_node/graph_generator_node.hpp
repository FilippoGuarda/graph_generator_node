#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace graph_generator_node
{

// Simple graph structures, mirroring networkx usage in Python
struct GraphNode
{
  int id;
  cv::Point2i pix;     // Pixel coordinates (x,y) on grid
  std::string type;    // "intersection", "endpoint", "entrance", "collision"
};

struct GraphEdge
{
  int u;
  int v;
  std::vector<cv::Point2i> path_pixels;  // Skeleton path pixels along edge
  double weight;                         // path length in pixels
};

class GraphGeneratorNode : public rclcpp::Node
{
public:
  explicit GraphGeneratorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~GraphGeneratorNode() override = default;

private:
  // ROS I/O
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr filtered_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr skeleton_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_marker_pub_;

  // Parameters
  std::string input_topic_;
  std::string global_frame_;
  int obstacle_size_threshold_;     // As in GridFast.obstacle_size_threshold
  int max_bfs_steps_;              // max_steps in build_graph
  bool find_entrances_;            // find_entrances flag
  double merge_threshold_pix_;     // merge_threshold in pixels

  // Last costmap info
  std::mutex map_mutex_;
  nav_msgs::msg::OccupancyGrid::SharedPtr last_map_;

  // Core callbacks
  void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

  // Pipeline steps (costmap → binary → cleaned → skeleton → graph)
  cv::Mat costmapToBinary(const nav_msgs::msg::OccupancyGrid & grid);
  cv::Mat removeSmallObstacles(const cv::Mat & map_data);
  cv::Mat gridFastLikeCleanup(const cv::Mat & cleaned_map);
  cv::Mat buildSkeleton(const cv::Mat & filtered_map);
  cv::Mat computeDistanceMap(const cv::Mat & skeleton);

  // Skeleton feature extraction (intersections, endpoints)
  void findSkeletonPoints(const cv::Mat & skeleton,
                          cv::Mat & intersections_mask,
                          cv::Mat & endpoints_mask);

  // Multi-source BFS graph construction
  void buildGraph(const cv::Mat & skeleton,
                  const cv::Mat & dist_map,
                  std::vector<GraphNode> & nodes,
                  std::vector<GraphEdge> & edges);

  // Helper functions for graph building
  std::vector<cv::Point2i> extractCoordsFromMask(const cv::Mat & mask);
  void executeBFS(const cv::Mat & skeleton,
                  const cv::Mat & dist_map,
                  bool find_entrances,
                  const std::vector<GraphNode> & initial_nodes,
                  std::vector<GraphNode> & out_nodes,
                  std::vector<GraphEdge> & out_edges);

  int createEntranceNode(const cv::Point2i & p,
                         int parent_src_id,
                         std::vector<GraphNode> & nodes,
                         std::vector<GraphEdge> & edges,
                         const std::unordered_map<int, cv::Point2i> & id_to_pix,
                         const std::unordered_map<long long, cv::Point2i> & parent_map);

  int createCollisionNode(const cv::Point2i & p,
                          int src1,
                          int src2,
                          std::vector<GraphNode> & nodes,
                          std::vector<GraphEdge> & edges,
                          const std::unordered_map<int, cv::Point2i> & id_to_pix,
                          const std::unordered_map<long long, cv::Point2i> & parent_map);

  std::vector<cv::Point2i> reconstructPath(
      const cv::Point2i & p,
      int src_id,
      const std::unordered_map<long long, cv::Point2i> & parent_map) const;

  std::vector<cv::Point2i> reconstructPathCollision(
      const cv::Point2i & cur_xy,
      const cv::Point2i & neigh_xy,
      int src_current,
      int src_neighbor,
      const std::unordered_map<long long, cv::Point2i> & parent_map) const;

  inline long long packKey(int src_id, int x, int y) const
  {
    // src_id in upper 32 bits, (x,y) packed in lower bits
    return (static_cast<long long>(src_id) << 32) |
           (static_cast<long long>(y) << 16) |
           static_cast<long long>(x);
  }

  // RViz marker publishing
  void publishGraphMarkers(const nav_msgs::msg::OccupancyGrid & base_grid,
                           const std::vector<GraphNode> & nodes,
                           const std::vector<GraphEdge> & edges);

  geometry_msgs::msg::Point gridToWorld(
      const nav_msgs::msg::OccupancyGrid & grid,
      int gx, int gy) const;
};

}  // namespace graph_generator_node
