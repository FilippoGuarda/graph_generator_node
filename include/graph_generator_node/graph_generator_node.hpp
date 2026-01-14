#pragma once

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "opencv2/opencv.hpp"
#include "skeleton_graph_builder.hpp"

namespace graph_generator_node {

using OccupancyGrid = nav_msgs::msg::OccupancyGrid;
using MarkerArray = visualization_msgs::msg::MarkerArray;
using Marker = visualization_msgs::msg::Marker;

class GraphGeneratorNode : public rclcpp::Node {
 public:
  explicit GraphGeneratorNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  // Callbacks
  void costmapCallback(OccupancyGrid::SharedPtr msg);

  // ROS2 Publishers and Subscribers
  rclcpp::Subscription<OccupancyGrid>::SharedPtr costmap_sub_;
  rclcpp::Publisher<OccupancyGrid>::SharedPtr filtered_pub_;
  rclcpp::Publisher<OccupancyGrid>::SharedPtr skeleton_pub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr graph_marker_pub_;

  // Map data
  std::mutex map_mutex_;
  OccupancyGrid::SharedPtr last_map_;

  // Parameters
  std::string input_topic_;
  std::string global_frame_;
  int obstacle_size_threshold_;
  int max_bfs_steps_;
  bool find_entrances_;
  double merge_threshold_pix_;
  double inflation_radius_;
  int min_junction_pixels_;
  double entrance_threshold_;

  // Map preprocessing functions
  cv::Mat costmapToBinary(const OccupancyGrid& costmap);
  cv::Mat removeSmallObstacles(const cv::Mat& map_data);
  cv::Mat gridFastLikeCleanup(const cv::Mat& cleaned_map);
  void thinning(const cv::Mat& src, cv::Mat& dst);

  // Skeleton and distance map
  cv::Mat buildSkeleton(const cv::Mat& filtered_map);
  cv::Mat computeDistanceMap(const cv::Mat& skeleton);

  // Visualization
  void publishGraphMarkers(
    const OccupancyGrid& costmap,
    const std::vector<GraphNode>& nodes,
    const std::vector<GraphEdge>& edges);

  // Coordinate conversion
  geometry_msgs::msg::Point gridToWorld(
    const OccupancyGrid& costmap, int x, int y) const;
};

}  // namespace graph_generator_node
