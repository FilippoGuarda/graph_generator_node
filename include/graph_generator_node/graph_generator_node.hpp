#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <opencv2/core.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "skeleton_graph_builder.hpp"

namespace graph_generator_node {

class GraphGeneratorNode : public rclcpp::Node {
public:
    explicit GraphGeneratorNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    // Callback
    void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

    // Processing pipeline
    cv::Mat costmapToBinary(const nav_msgs::msg::OccupancyGrid& grid);
    cv::Mat removeSmallObstacles(const cv::Mat& map);
    cv::Mat gridFastLikeCleanup(const cv::Mat& cleaned_map);
    cv::Mat buildSkeleton(const cv::Mat& filtered_map);
    cv::Mat computeDistanceMap(const cv::Mat& obstacle_map);
    cv::Mat momentumFieldSkeleton(const cv::Mat& dist_transform, double threshold);
    cv::Mat removeUnconnectedBranches(const cv::Mat& skeleton);

    // Publishing
    void publishFilteredMap(const nav_msgs::msg::OccupancyGrid& base_grid, const cv::Mat& filtered);
    void publishSkeletonMap(const nav_msgs::msg::OccupancyGrid& base_grid, const cv::Mat& skeleton);
    void publishEmptySkeleton(const nav_msgs::msg::OccupancyGrid& base_grid, const cv::Mat& skeleton);
    void publishGraphMarkers(const nav_msgs::msg::OccupancyGrid& base_grid,
                             const std::shared_ptr<NetworkX>& graph);
    void publishGraphJson(const nav_msgs::msg::OccupancyGrid& base_grid,
                          const std::shared_ptr<NetworkX>& graph);
    void addStaggeredPoints(cv::Mat& map, const cv::Mat& dist_map, double robot_size);

    // Utilities
    geometry_msgs::msg::Point gridToWorld(
        const nav_msgs::msg::OccupancyGrid& grid, int gx, int gy);

    // ROS2 interface
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr filtered_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr skeleton_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_marker_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr json_pub_;

    // Parameters
    std::string input_topic_;
    std::string global_frame_;
    int obstacle_size_threshold_;
    int max_bfs_steps_;
    int hysteresis_;
    int skeleton_threshold_;
    bool find_entrances_;
    double robot_radius_;

    // State
    std::mutex map_mutex_;
    nav_msgs::msg::OccupancyGrid::SharedPtr last_map_;
};

} // namespace graph_generator_node