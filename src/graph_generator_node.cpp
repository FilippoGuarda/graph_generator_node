#include "graph_generator_node/graph_generator_node.hpp"
#include "graph_generator_node/skeleton_graph_builder.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

namespace graph_generator_node {

GraphGeneratorNode::GraphGeneratorNode(const rclcpp::NodeOptions& options)
    : Node("graph_generator_node", options) {
    // Parameters
    input_topic_ = this->declare_parameter(
        "shared_costmap_topic", "/global_costmap");
    global_frame_ = this->declare_parameter(
        "global_frame", "map");
    obstacle_size_threshold_ = this->declare_parameter(
        "obstacle_size_threshold", 2);
    max_bfs_steps_ = this->declare_parameter(
        "max_bfs_steps", 100);
    find_entrances_ = this->declare_parameter(
        "find_entrances", true);
    merge_threshold_pix_ = this->declare_parameter(
        "merge_threshold_pix", 50.0);

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

    // Step 4: Publish filtered map
    publishFilteredMap(*msg, filtered);

    // Step 5: Build skeleton
    cv::Mat skeleton = buildSkeleton(filtered);

    if (cv::countNonZero(skeleton) == 0) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "Empty skeleton generated - map may be too cluttered");
        publishEmptySkeleton(*msg, skeleton);
        return;
    }

    // Step 6: Publish skeleton
    publishSkeletonMap(*msg, skeleton);

    // Step 7: Compute distance map (from obstacles, not skeleton)
    cv::Mat distmap = computeDistanceMap(filtered);

    // Step 8: Build graph using SkeletonGraphBuilder
    SkeletonGraphBuilder builder(skeleton, distmap);
    auto [graph, node_positions] = builder.buildGraph(
        max_bfs_steps_,
        find_entrances_);

    // Step 8b: Merge close nodes if threshold is set
    if (merge_threshold_pix_ > 0.0) {
      std::vector<std::string> types_to_merge = {"intersection", "entrance", "collision"};
      node_positions = builder.mergeCloseNodes(merge_threshold_pix_, types_to_merge);
    }

    // Step 9: Publish graph visualization and networkx as json string
    publishGraphMarkers(*msg, graph);
    publishGraphJson(*msg, graph);

    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Graph: %zu nodes, %zu edges",
        graph->nodes().size(), graph->edges().size());
}

void GraphGeneratorNode::publishFilteredMap(
    const nav_msgs::msg::OccupancyGrid& base_grid,
    const cv::Mat& filtered) {
    nav_msgs::msg::OccupancyGrid filtered_msg = base_grid;
    filtered_msg.data.assign(filtered_msg.data.size(), 0);

    for (int y = 0; y < filtered.rows; ++y) {
        const uint8_t* row = filtered.ptr<uint8_t>(y);
        for (int x = 0; x < filtered.cols; ++x) {
            int idx = y * filtered.cols + x;
            filtered_msg.data[idx] = (row[x] != 0) ? 100 : 0;
        }
    }
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
            // -1 (unknown) and 50+ (occupied) -> 255 (obstacle)
            if (v == -1 || v >= 70) {
                row[x] = 255;
            } else {
                row[x] = 0; // Free
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
    cv::Mat bin;
    cv::threshold(filtered_map, bin, 127, 255, cv::THRESH_BINARY_INV);
    cv::Mat skeleton;
    thinning(bin, skeleton);
    return skeleton;
}

void GraphGeneratorNode::thinning(const cv::Mat& src, cv::Mat& dst) {
    dst = src.clone();
    dst.convertTo(dst, CV_8U);
    dst = dst / 255; // Convert to 01: 1 = skeleton, 0 = background

    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8U);
    cv::Mat marker = cv::Mat::zeros(dst.size(), CV_8U);

    while (true) {
        dst.copyTo(prev);

        // Iteration 1 and 2 for skeleton thinning
        for (int iter = 0; iter < 2; ++iter) {
            marker.setTo(0);
            
            for (int r = 1; r < dst.rows - 1; ++r) {
                const uchar* prev_row = dst.ptr<uchar>(r - 1);
                const uchar* curr_row = dst.ptr<uchar>(r);
                const uchar* next_row = dst.ptr<uchar>(r + 1);
                uchar* marker_row = marker.ptr<uchar>(r);

                for (int c = 1; c < dst.cols - 1; ++c) {
                    // 3x3 neighborhood
                    int p2 = prev_row[c], p3 = prev_row[c + 1],
                        p4 = curr_row[c + 1], p5 = next_row[c + 1],
                        p6 = next_row[c], p7 = next_row[c - 1],
                        p8 = curr_row[c - 1], p9 = prev_row[c - 1];

                    int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                            (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                            (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                            (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                    int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    int m1 = (iter == 0) ? (p2 * p4 * p6) : (p2 * p4 * p8);
                    int m2 = (iter == 0) ? (p4 * p6 * p8) : (p2 * p6 * p8);

                    if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
                        marker_row[c] = 1;
                    }
                }
            }
            // Remove marked pixels
            for(int r=0; r<dst.rows; ++r) {
                uchar* d = dst.ptr<uchar>(r);
                const uchar* m = marker.ptr<uchar>(r);
                for(int c=0; c<dst.cols; ++c) {
                    if(m[c]) d[c] = 0;
                }
            }
        }

        cv::Mat diff;
        cv::absdiff(dst, prev, diff);
        if (cv::countNonZero(diff) == 0) {
            break;
        }
    }

    dst = dst * 255; // Convert back to 255 = skeleton
}

cv::Mat GraphGeneratorNode::computeDistanceMap(const cv::Mat& obstacle_map) {
    // obstacle_map: 255=Obstacle, 0=Free
    // We want distance TO the nearest obstacle
    // distanceTransform calculates distance to nearest ZERO pixel
    // So invert: Obstacles(255)->0, Free(0)->255
    
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

    for (const auto& [nid, node] : graph->nodes()) {
        geometry_msgs::msg::Point p = gridToWorld(base_grid, node.position.first, node.position.second);
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

    for (const auto& [key, edge] : graph->edges()) {
        if (edge.path_pixels.size() < 2) continue;
        for (size_t i = 1; i < edge.path_pixels.size(); ++i) {
            geometry_msgs::msg::Point p1 = gridToWorld(base_grid,
            edge.path_pixels[i - 1].first,    // ← .first not .x
            edge.path_pixels[i - 1].second);  // ← .second not .y
            geometry_msgs::msg::Point p2 = gridToWorld(base_grid,
            edge.path_pixels[i].first,        // ← .first not .x
            edge.path_pixels[i].second);      // ← .second not .y
            edge_marker.points.push_back(p1);   // ← edge_marker not edgemarker
            edge_marker.points.push_back(p2);   // ← edge_marker not edgemarker
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
