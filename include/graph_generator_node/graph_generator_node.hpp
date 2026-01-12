#ifndef GRAPH_GENERATOR_NODE__GRAPH_GENERATOR_NODE_HPP_
#define GRAPH_GENERATOR_NODE__GRAPH_GENERATOR_NODE_HPP_

#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <opencv2/core.hpp>

namespace graph_generator_node
{

struct GraphNode
{
  int id;
  cv::Point2i pix;
  std::string type;  // "intersection", "endpoint", "entrance", "collision"
};

struct GraphEdge
{
  int u;
  int v;
  std::vector<cv::Point2i> path_pixels;
  double weight{};
};

class GraphGeneratorNode : public rclcpp::Node
{
public:
  explicit GraphGeneratorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

  cv::Mat costmapToBinary(const nav_msgs::msg::OccupancyGrid & grid);
  cv::Mat removeSmallObstacles(const cv::Mat & map_data);
  cv::Mat gridFastLikeCleanup(const cv::Mat & cleaned_map);
  cv::Mat buildSkeleton(const cv::Mat & filtered_map);
  cv::Mat computeDistanceMap(const cv::Mat & skeleton);

  void findSkeletonPoints(const cv::Mat & skeleton,
                          cv::Mat & intersections_mask,
                          cv::Mat & endpoints_mask);
  std::vector<cv::Point2i> extractCoordsFromMask(const cv::Mat & mask);

  void buildGraph(const cv::Mat & skeleton,
                  const cv::Mat & dist_map,
                  std::vector<GraphNode> & nodes,
                  std::vector<GraphEdge> & edges);

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

  geometry_msgs::msg::Point gridToWorld(const nav_msgs::msg::OccupancyGrid & grid,
                                        int gx, int gy) const;

  void publishGraphMarkers(const nav_msgs::msg::OccupancyGrid & base_grid,
                           const std::vector<GraphNode> & nodes,
                           const std::vector<GraphEdge> & edges);

  static inline long long packKey(int src_id, int x, int y)
  {
    return (static_cast<long long>(src_id) << 32) |
           (static_cast<unsigned int>(y) << 16) |
           static_cast<unsigned int>(x);
  }

  // Parameters
  std::string input_topic_;
  std::string global_frame_;
  int obstacle_size_threshold_{20};
  int max_bfs_steps_{100};
  bool find_entrances_{true};
  double merge_threshold_pix_{0.0};  // reserved

  // ROS interfaces
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr filtered_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr skeleton_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_marker_pub_;

  // State
  nav_msgs::msg::OccupancyGrid::SharedPtr last_map_;
  std::mutex map_mutex_;
};

}  // namespace graph_generator_node

#endif  // GRAPH_GENERATOR_NODE__GRAPH_GENERATOR_NODE_HPP_
