// Copyright 2025 Filippo Guarda
// Licensed under the Apache License, Version 2.0

#include "graph_generator_node/graph_generator_node.hpp"
#include "graph_generator_node/skeleton_graph_builder.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>

using namespace std::chrono_literals;

namespace graph_generator_node {

GraphGeneratorNode::GraphGeneratorNode(const rclcpp::NodeOptions& options)
  : Node("graph_generator_node", options) {
  // Declare parameters
  input_topic_ = this->declare_parameter("shared_costmap_topic", "/shared_obstacles");
  global_frame_ = this->declare_parameter("global_frame", "map");
  obstacle_size_threshold_ = this->declare_parameter("obstacle_size_threshold", 20);
  max_bfs_steps_ = this->declare_parameter("max_bfs_steps", 100);
  find_entrances_ = this->declare_parameter("find_entrances", true);
  merge_threshold_pix_ = this->declare_parameter("merge_threshold_pix", 0.0);

  // Create subscriptions and publishers
  costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
    input_topic_, rclcpp::QoS(1).reliable().transient_local(),
    std::bind(&GraphGeneratorNode::costmapCallback, this, std::placeholders::_1));

  filtered_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
    "/skeleton_graph/filtered_map", rclcpp::QoS(1).reliable());

  skeleton_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
    "/skeleton_graph/skeleton_map", rclcpp::QoS(1).reliable());

  graph_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/skeleton_graph/graph_markers", rclcpp::QoS(1).transient_local().reliable());

  RCLCPP_INFO(this->get_logger(), "GraphGeneratorNode initialized, subscribing to %s",
              input_topic_.c_str());
}

void GraphGeneratorNode::costmapCallback(
  const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  std::lock_guard lock(map_mutex_);
  last_map_ = msg;

  if (msg->data.empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Received empty costmap");
    return;
  }

  // Step 1: Convert costmap to binary
  cv::Mat binary = costmapToBinary(*msg);

  // Step 2: Remove small obstacles
  cv::Mat cleaned = removeSmallObstacles(binary);

  // Step 3: Apply grid_fast_like cleanup
  cv::Mat filtered = gridFastLikeCleanup(cleaned);

  // Publish filtered map
  nav_msgs::msg::OccupancyGrid filtered_grid = *msg;
  filtered_grid.data.assign(filtered_grid.data.size(), 0);
  for (int y = 0; y < filtered.rows; ++y) {
    const uint8_t* row = filtered.ptr(y);
    for (int x = 0; x < filtered.cols; ++x) {
      int idx = y * filtered.cols + x;
      filtered_grid.data[idx] = (row[x] > 0) ? 100 : 0;
    }
  }
  filtered_pub_->publish(filtered_grid);

  // Step 4: Build skeleton
  cv::Mat skeleton = buildSkeleton(filtered);

  if (cv::countNonZero(skeleton) == 0) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Empty skeleton generated - map too cluttered or no corridors");
    return;
  }

  // Publish skeleton
  nav_msgs::msg::OccupancyGrid skeleton_grid = *msg;
  skeleton_grid.data.assign(skeleton_grid.data.size(), 0);
  for (int y = 0; y < skeleton.rows; ++y) {
    const uint8_t* row = skeleton.ptr(y);
    for (int x = 0; x < skeleton.cols; ++x) {
      int idx = y * skeleton.cols + x;
      skeleton_grid.data[idx] = (row[x] > 0) ? 100 : 0;
    }
  }
  skeleton_pub_->publish(skeleton_grid);

  // Step 5: Compute distance map
  cv::Mat dist_map = computeDistanceMap(skeleton);

  // Step 6: Build graph using SkeletonGraphBuilder
  SkeletonGraphBuilder builder(skeleton, dist_map);
  auto [nodes, edges] = builder.buildGraph(
    max_bfs_steps_,
    find_entrances_,
    static_cast<float>(merge_threshold_pix_)
  );

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                       "Graph: %zu nodes, %zu edges", nodes.size(), edges.size());

  // Publish visualization
  publishGraphMarkers(*msg, nodes, edges);
}

// ============================================================================
// MAP PROCESSING
// ============================================================================

cv::Mat GraphGeneratorNode::costmapToBinary(
  const nav_msgs::msg::OccupancyGrid& grid) {
  cv::Mat binary(grid.info.height, grid.info.width, CV_8UC1);

  for (int y = 0; y < static_cast<int>(grid.info.height); ++y) {
    uint8_t* row = binary.ptr(y);
    for (int x = 0; x < static_cast<int>(grid.info.width); ++x) {
      int idx = y * grid.info.width + x;
      int8_t v = grid.data[idx];

      if (v == -1 || v > 50) {
        row[x] = 255; // Occupied
      } else {
        row[x] = 0; // Free
      }
    }
  }
  return binary;
}

cv::Mat GraphGeneratorNode::removeSmallObstacles(const cv::Mat& map_data) {
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

cv::Mat GraphGeneratorNode::gridFastLikeCleanup(const cv::Mat& cleaned_map) {
  // Convert to free-space mask
  cv::Mat free_mask = (cleaned_map == 0);

  // Morphological closing to fill gaps
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::Mat closed;
  cv::morphologyEx(free_mask, closed, cv::MORPH_CLOSE, kernel);

  // Dilate slightly
  cv::Mat dilated;
  cv::dilate(closed, dilated, kernel);

  // Rebuild map
  cv::Mat filtered(cleaned_map.size(), CV_8UC1, cv::Scalar(255));
  filtered.setTo(0, dilated > 0);

  return filtered;
}

// ============================================================================
// SKELETONIZATION
// ============================================================================

void GraphGeneratorNode::thinning(const cv::Mat& src, cv::Mat& dst) {
  dst = src.clone();
  dst /= 255; // Convert to 0/1

  cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
  cv::Mat diff;

  do {
    dst.copyTo(prev);

    for (int iter = 0; iter < 2; ++iter) {
      cv::Mat marker = cv::Mat::zeros(dst.size(), CV_8UC1);

      for (int r = 1; r < dst.rows - 1; ++r) {
        const uchar* prev_row = dst.ptr(r - 1);
        const uchar* curr_row = dst.ptr(r);
        const uchar* next_row = dst.ptr(r + 1);
        uchar* marker_row = marker.ptr(r);

        for (int c = 1; c < dst.cols - 1; ++c) {
          int p2 = prev_row[c], p3 = prev_row[c + 1],
              p4 = curr_row[c + 1];
          int p5 = next_row[c + 1], p6 = next_row[c],
              p7 = next_row[c - 1];
          int p8 = curr_row[c - 1], p9 = prev_row[c - 1];

          int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                  (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                  (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                  (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

          int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

          int m1 = (iter == 0) ? (p2 * p4 * p6) : (p2 * p4 * p8);
          int m2 = (iter == 0) ? (p4 * p6 * p8) : (p2 * p6 * p8);

          if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
            marker_row[c] = 1;
          }
        }
      }

      dst &= ~marker;
    }

    cv::absdiff(dst, prev, diff);
  } while (cv::countNonZero(diff) > 0);

  dst *= 255;
}

cv::Mat GraphGeneratorNode::buildSkeleton(const cv::Mat& filtered_map) {
  cv::Mat bin;
  cv::threshold(filtered_map, bin, 127, 255, cv::THRESH_BINARY_INV);

  cv::Mat skeleton;
  thinning(bin, skeleton);

  return skeleton;
}

// ============================================================================
// DISTANCE TRANSFORM
// ============================================================================

cv::Mat GraphGeneratorNode::computeDistanceMap(const cv::Mat& skeleton) {
  cv::Mat inv;
  cv::threshold(skeleton, inv, 0, 255, cv::THRESH_BINARY_INV);

  cv::Mat dist;
  cv::distanceTransform(inv, dist, cv::DIST_L2, 3);

  cv::Mat dist32;
  dist.convertTo(dist32, CV_32FC1);

  return dist32;
}

// ============================================================================
// VISUALIZATION
// ============================================================================

geometry_msgs::msg::Point GraphGeneratorNode::gridToWorld(
  const nav_msgs::msg::OccupancyGrid& grid, int gx, int gy) const {
  geometry_msgs::msg::Point p;
  double res = grid.info.resolution;

  p.x = grid.info.origin.position.x + (gx + 0.5) * res;
  p.y = grid.info.origin.position.y + (gy + 0.5) * res;
  p.z = 0.0;

  return p;
}

void GraphGeneratorNode::publishGraphMarkers(
  const nav_msgs::msg::OccupancyGrid& base_grid,
  const std::vector<GraphNode>& nodes,
  const std::vector<GraphEdge>& edges) {
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

  for (const auto& n : nodes) {
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

  for (const auto& e : edges) {
    if (e.path_pixels.size() < 2) continue;

    for (size_t i = 1; i < e.path_pixels.size(); ++i) {
      geometry_msgs::msg::Point p1 =
        gridToWorld(base_grid, e.path_pixels[i - 1].x, e.path_pixels[i - 1].y);
      geometry_msgs::msg::Point p2 =
        gridToWorld(base_grid, e.path_pixels[i].x, e.path_pixels[i].y);

      edge_marker.points.push_back(p1);
      edge_marker.points.push_back(p2);
    }
  }
  ma.markers.push_back(edge_marker);

  graph_marker_pub_->publish(ma);
}

} // namespace graph_generator_node

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(graph_generator_node::GraphGeneratorNode)
