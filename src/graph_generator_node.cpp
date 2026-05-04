#include "graph_generator_node/graph_generator_node.hpp"
#include "graph_generator_node/skeleton_graph_builder.hpp"
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {


}// namespace

namespace graph_generator_node {

GraphGeneratorNode::GraphGeneratorNode(const rclcpp::NodeOptions& options)
    : Node("graph_generator_node", options) {
    // Parameters
    input_topic_ = this->declare_parameter<std::string>(
        "shared_costmap_topic", "/global_costmap");
    global_frame_ = this->declare_parameter<std::string>(
        "global_frame", "map");
    obstacle_size_threshold_ = this->declare_parameter<int>(
        "obstacle_size_threshold", 2);
    skeleton_threshold_ = this->declare_parameter<int>(
        "skeleton_threshold", 10);
    hysteresis_ = this->declare_parameter<int>(
        "max_steps", 20);
    max_entrance_distance_ = this->declare_parameter<int>(
        "max_entrance_distance", 10);
    robot_radius_ = this->declare_parameter<double>(
        "robot_radius", 15.0);
    cluster_radius_ = this->declare_parameter<double>(
        "cluster_radius", 3.0);
    find_entrances_ = this->declare_parameter<bool>(
        "find_entrances", true);
    cluster_nodes_ = this->declare_parameter<bool>(
        "cluster_nodes", true);

    // Subscriptions
    costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        input_topic_,
        rclcpp::QoS(1).reliable().transient_local(),
        std::bind(&GraphGeneratorNode::costmapCallback, this, std::placeholders::_1));

    // Publications
    filtered_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
        "skeleton_graph/filtered_map", rclcpp::QoS(1).reliable().transient_local());
    skeleton_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
        "skeleton_graph/skeleton_map", rclcpp::QoS(1).reliable().transient_local());
    graph_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "skeleton_graph/graph_markers", rclcpp::QoS(1).transient_local().reliable());
    json_pub_ = this->create_publisher<std_msgs::msg::String>(
        "skeleton_graph_json", rclcpp::QoS(1).transient_local().reliable());

    RCLCPP_INFO(this->get_logger(),
        "GraphGeneratorNode initialized, subscribing to %s", input_topic_.c_str());
}

void GraphGeneratorNode::costmapCallback(
    const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    last_map_ = msg;

    if (msg->data.empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Received empty costmap");
        return;
    }

    // Step 1: Convert to binary
    cv::Mat binary = costmapToBinary(*msg);

    // Step 2: Clean obstacles
    cv::Mat cleaned = removeSmallObstacles(binary);

    // Step 3: Grid-FAST-like cleanup
    cv::Mat filtered = gridFastLikeCleanup(cleaned);
    // filtered semantics: 0 = free, 255 = obstacle

    // Step 4: Distance map on the original filtered map if you still need it elsewhere
    cv::Mat distmap = computeDistanceMap(filtered);

    // Step 5: Build an explicit free-space mask for staggered point placement
    // free_mask semantics: 255 = free, 0 = obstacle
    cv::Mat free_mask = (filtered == 0);
    // free_mask.convertTo(free_mask, CV_8U, 255);

    // Step 6: Distance map in free space, used to decide where staggered points are valid
    cv::Mat free_dist;
    cv::distanceTransform(free_mask, free_dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    // Step 7: Apply staggered points INSIDE FREE SPACE
    // addStaggeredPoints expects map=255 where valid space exists and writes 0 where points are carved
    addStaggeredPoints(free_mask, free_dist, robot_radius_);

    // Step 8: Recompute distance transform from the staggered free-space mask
    cv::Mat map_lane_dist;
    cv::distanceTransform(free_mask, map_lane_dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);

    // Define maximum inflation distance in pixels for the gradient fade.
    // Replace 20.0f with your required radius (e.g., inflation_radius_meters / map_resolution)
    const float max_dist_px = robot_radius_ ;

    // Clamp the distances to max_dist_px
    cv::Mat clamped_dist;
    cv::min(map_lane_dist, max_dist_px, clamped_dist);

    // Linearly scale from [0, max_dist_px] to cost [99, 0]
    cv::Mat scaled_dist;
    clamped_dist.convertTo(scaled_dist, CV_32F, -99.0 / max_dist_px, 99.0);

    // Convert to 8-bit signed integers to match nav_msgs::msg::OccupancyGrid specification
    cv::Mat dist_vis;
    scaled_dist.convertTo(dist_vis, CV_8S);

    // Override the exact obstacle cells to 100 (lethal obstacle)
    dist_vis.setTo(100, map_lane_dist == 0.0f);

    // Publish the map directly
    publishFilteredMap(*msg, dist_vis);

    // Step 9: Build skeleton from distance map
    cv::Mat skeleton = buildSkeleton(map_lane_dist);

    skeleton = removeUnconnectedBranches(skeleton);

    if (cv::countNonZero(skeleton) == 0) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Empty skeleton generated - map may be too cluttered");
        publishEmptySkeleton(*msg, skeleton);
        return;
    }

    // Step 10: Publish skeleton
    publishSkeletonMap(*msg, skeleton);

    // Continue with graph generation...
    SkeletonGraphBuilder builder(skeleton, distmap);
    auto [graph, node_positions] =
        builder.buildGraph(hysteresis_, max_entrance_distance_, find_entrances_);

    if (robot_radius_ > 0.0 and cluster_nodes_) {
        std::vector<std::string> types_to_merge = {"intersection", "enpoint", "collision"};
        node_positions = builder.mergeCloseNodes(cluster_radius_ , types_to_merge);
    }

    publishGraphMarkers(*msg, graph);
    publishGraphJson(*msg, graph);

    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Graph: %zu nodes, %zu edges",
        graph->nodes().size(), graph->edges().size());
}

void GraphGeneratorNode::publishFilteredMap(
    const nav_msgs::msg::OccupancyGrid& base_grid,
    const cv::Mat& filtered) {
    
    // Ensure the input matrix is continuous and signed 8-bit to match ROS specifications
    cv::Mat continuous_filtered;
    if (filtered.type() != CV_8S) {
        filtered.convertTo(continuous_filtered, CV_8S);
    } else {
        continuous_filtered = filtered;
    }
    
    if (!continuous_filtered.isContinuous()) {
        continuous_filtered = continuous_filtered.clone();
    }

    nav_msgs::msg::OccupancyGrid filtered_msg = base_grid;
    
    // Directly copy the gradient data into the message vector
    size_t data_size = continuous_filtered.total();
    filtered_msg.data.assign(
        continuous_filtered.ptr<int8_t>(),
        continuous_filtered.ptr<int8_t>() + data_size
    );

    filtered_pub_->publish(filtered_msg);
}

void GraphGeneratorNode::publishSkeletonMap(
    const nav_msgs::msg::OccupancyGrid& base_grid,
    const cv::Mat& skeleton) {
    nav_msgs::msg::OccupancyGrid skeleton_msg = base_grid;
    skeleton_msg.data.assign(skeleton_msg.data.size(), 0);

    for (int y = 0; y < skeleton.rows; ++y) {
        const uint8_t* row = skeleton.ptr<uint8_t>(y);
        for (int x = 0; x < skeleton.cols; ++x) {
            int idx = y * skeleton.cols + x;
            skeleton_msg.data[idx] = (row[x] != 0) ? 100 : 0;
        }
    }
    skeleton_pub_->publish(skeleton_msg);
}

void GraphGeneratorNode::publishEmptySkeleton(
    const nav_msgs::msg::OccupancyGrid& base_grid,
    const cv::Mat& skeleton) {
    nav_msgs::msg::OccupancyGrid skeleton_msg = base_grid;
    skeleton_msg.data.assign(skeleton_msg.data.size(), 0);
    skeleton_pub_->publish(skeleton_msg);
}

cv::Mat GraphGeneratorNode::costmapToBinary(
    const nav_msgs::msg::OccupancyGrid& grid) {
    cv::Mat binary(grid.info.height, grid.info.width, CV_8UC1);

    for (int y = 0; y < static_cast<int>(grid.info.height); ++y) {
        uint8_t* row = binary.ptr<uint8_t>(y);
        for (int x = 0; x < static_cast<int>(grid.info.width); ++x) {
            int idx = y * grid.info.width + x;
            int8_t v = grid.data[idx];
            if (v == -1 || v >= 70) {
                row[x] = 255;
            } else {
                row[x] = 0;
            }
        }
    }
    return binary;
}

cv::Mat GraphGeneratorNode::removeSmallObstacles(const cv::Mat& map) {
    // map: 0 free, 255 occupied
    cv::Mat obstacle_mask = map == 255;
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(
        obstacle_mask, labels, stats, centroids, 8, CV_32S);

    cv::Mat result = map.clone();
    for (int label = 1; label < num_labels; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < obstacle_size_threshold_) {
            result.setTo(0, labels == label);
        }
    }
    return result;
}

cv::Mat GraphGeneratorNode::gridFastLikeCleanup(const cv::Mat& cleaned_map) {
    // Convert to 01: 1 = free, 0 = occupied
    cv::Mat free_mask = cleaned_map == 0;

    // Morphological closing to fill thin gaps
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat closed;
    cv::morphologyEx(free_mask, closed, cv::MORPH_CLOSE, kernel);

    // Dilate free space slightly
    cv::Mat dilated;
    cv::dilate(closed, dilated, kernel);

    // Rebuild map: free = 0 where dilated is 1, occupied = 255 elsewhere
    cv::Mat filtered = cleaned_map.clone();
    filtered.setTo(0, dilated != 0);
    return filtered;
}

cv::Mat GraphGeneratorNode::buildSkeleton(const cv::Mat& filtered_map) {

    cv::Mat momentum_skeleton = momentumFieldSkeleton(filtered_map, skeleton_threshold_);

    cv::Mat skeleton;
    cv::ximgproc::thinning(momentum_skeleton, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);

    return skeleton;
}

// Optional: remove unconnected branches from skeleton
cv::Mat GraphGeneratorNode::removeUnconnectedBranches(const cv::Mat& skeleton) {
    cv::Mat skel_u8;
    if (skeleton.type() != CV_8U) {
        skeleton.convertTo(skel_u8, CV_8U);
    } else {
        skel_u8 = skeleton.clone();
    }

    cv::Mat binary = skel_u8 != 0;

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S);
    if (num_labels <= 1) {
        return skel_u8;
    }

    int largest_label = 1;
    int largest_area = stats.at<int>(1, cv::CC_STAT_AREA);
    for (int label = 2; label < num_labels; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area > largest_area) {
            largest_area = area;
            largest_label = label;
        }
    }

    cv::Mat cleaned = (labels == largest_label);
    cleaned.convertTo(cleaned, CV_8U, 255);
    return cleaned;
}

// Used to add points to obstacle map, creates an exagonal pattern that is stable trough time
void GraphGeneratorNode::addStaggeredPoints(cv::Mat& map, const cv::Mat& dist_map, double robot_size) {
    int h = map.rows;
    int w = map.cols;

    int scaled_robot_size = static_cast<int>(robot_size * 2.0 / std::sqrt(3.0));

    int spacing_x = scaled_robot_size;
    int spacing_y = static_cast<int>(scaled_robot_size * std::sqrt(3.0) / 2.0);

    int row = 0;
    int column = 0;
    for (int y = 0; y < h; y += spacing_y) {
        int offset = (row % 2 == 0) ? 0 : spacing_x / 2;
        for (int x = offset; x < w; x += spacing_x) {
            float d = dist_map.at<float>(y, x);
            if (d > static_cast<float>(scaled_robot_size)) {
                map.at<uint8_t>(y, x) = 0;
            }
            column += 1;
        }
        column = 0;
        row += 1;
    }
}

cv::Mat GraphGeneratorNode::momentumFieldSkeleton(const cv::Mat& dist_transform, double threshold = 2.0) {
    cv::Mat dist_f;
    dist_transform.convertTo(dist_f, CV_64F);


    cv::Mat grad_x, grad_y;
    cv::Sobel(dist_f, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(dist_f, grad_y, CV_64F, 0, 1, 3);

    grad_x = -grad_x;
    grad_y = -grad_y;

    cv::Mat grad_magnitude;
    cv::magnitude(grad_x, grad_y, grad_magnitude);

    cv::Mat velocity_x = grad_x / (grad_magnitude + 1e-8);
    cv::Mat velocity_y = grad_y / (grad_magnitude + 1e-8);

    cv::Mat momentum_x = velocity_x.mul(dist_f);
    cv::Mat momentum_y = velocity_y.mul(dist_f);


    cv::Mat dmx_dx, dmy_dy;
    cv::Sobel(momentum_x, dmx_dx, CV_64F, 1, 0, 3);
    cv::Sobel(momentum_y, dmy_dy, CV_64F, 0, 1, 3);

    cv::Mat divergence = dmx_dx + dmy_dy;

    cv::Mat skeleton = divergence > threshold;
    skeleton.convertTo(skeleton, CV_8U, 255);

    return skeleton;
}

cv::Mat GraphGeneratorNode::computeDistanceMap(const cv::Mat& obstacle_map) {
    cv::Mat inv_obstacles;
    cv::threshold(obstacle_map, inv_obstacles, 0, 255, cv::THRESH_BINARY_INV);

    cv::Mat dist;
    cv::distanceTransform(inv_obstacles, dist, cv::DIST_L2, cv::DIST_MASK_5);

    cv::Mat dist32;
    dist.convertTo(dist32, CV_32FC1);
    return dist32;
}

void GraphGeneratorNode::publishGraphMarkers(
    const nav_msgs::msg::OccupancyGrid& base_grid,
    const std::shared_ptr<NetworkX>& graph) {
    visualization_msgs::msg::MarkerArray ma;
    rclcpp::Time stamp = this->get_clock()->now();

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

    for (const auto& [nid, node] : graph->nodes()) {
        geometry_msgs::msg::Point p = gridToWorld(base_grid, node.position.first, node.position.second);
        node_marker.points.push_back(p);
    }
    ma.markers.push_back(node_marker);

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

    for (const auto& [key, edge] : graph->edges()) {
        if (edge.path_pixels.size() < 2) continue;
        for (size_t i = 1; i < edge.path_pixels.size(); ++i) {
            geometry_msgs::msg::Point p1 = gridToWorld(base_grid,
                edge.path_pixels[i - 1].first,
                edge.path_pixels[i - 1].second);
            geometry_msgs::msg::Point p2 = gridToWorld(base_grid,
                edge.path_pixels[i].first,
                edge.path_pixels[i].second);
            edge_marker.points.push_back(p1);
            edge_marker.points.push_back(p2);
        }
    }
    ma.markers.push_back(edge_marker);

    graph_marker_pub_->publish(ma);
}

void GraphGeneratorNode::publishGraphJson(const nav_msgs::msg::OccupancyGrid& base_grid, const std::shared_ptr<NetworkX>& graph) {
    std_msgs::msg::String msg;
    std::stringstream ss;
    ss << "{\"directed\": false, \"multigraph\": false, \"graph\": {}, \"nodes\": [";

    bool first_node = true;
    for (const auto& [nid, node] : graph->nodes()) {
        if (!first_node) ss << ",";
        geometry_msgs::msg::Point p = gridToWorld(base_grid, node.position.first, node.position.second);
        ss << "{\"id\": " << nid << ", \"pos\": [" << p.x << ", " << p.y << "]}";
        first_node = false;
    }

    ss << "], \"links\": [";
    bool first_edge = true;
    std::unordered_set<std::string> seen;

    for (const auto& [nid, node] : graph->nodes()) {
        try {
            for (int nb : graph->getNeighbors(nid)) {
                int u = std::min(nid, nb);
                int v = std::max(nid, nb);
                std::string key = std::to_string(u) + "_" + std::to_string(v);

                if (seen.count(key)) continue;
                seen.insert(key);

                geometry_msgs::msg::Point p1 = gridToWorld(base_grid, graph->nodes().at(u).position.first, graph->nodes().at(u).position.second);
                geometry_msgs::msg::Point p2 = gridToWorld(base_grid, graph->nodes().at(v).position.first, graph->nodes().at(v).position.second);
                double weight = std::hypot(p1.x - p2.x, p1.y - p2.y);

                if (!first_edge) ss << ",";
                ss << "{\"source\": " << u << ", \"target\": " << v << ", \"weight\": " << weight << "}";
                first_edge = false;
            }
        } catch (...) {}
    }

    ss << "]}";
    msg.data = ss.str();
    json_pub_->publish(msg);
}

geometry_msgs::msg::Point GraphGeneratorNode::gridToWorld(
    const nav_msgs::msg::OccupancyGrid& grid,
    int gx,
    int gy) {
    geometry_msgs::msg::Point p;
    double res = grid.info.resolution;
    p.x = grid.info.origin.position.x + (gx + 0.5) * res;
    p.y = grid.info.origin.position.y + (gy + 0.5) * res;
    p.z = 0.0;
    return p;
}

} // namespace graph_generator_node

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(graph_generator_node::GraphGeneratorNode)