/**
 * Topological Graph Generation Node for Multi-Robot Costmap
 *
 * Reads global costmap from multi_robot_costmap_plugin and generates
 * a topological graph with skeleton-based nodes, intersection detection,
 * and entrance node identification using GRID-FAST and multi-source BFS.
 *
 * Author: Filippo Guarda
 * Year: 2025
 */

#include "rclcpp/rclcpp.hpp"
#include <nav2_msgs/msg/costmap.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <queue>
#include <deque>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <algorithm>

#include "topological_graph_node/skeleton_graph_builder.hpp"
#include "topological_graph_node/grid_fast_processor.hpp"
#include "topological_graph_node/graph_types.hpp"
#include "topological_graph_node/graph_serializer.hpp"

namespace topological_graph
{

    class TopologicalGraphGeneratorNode : public rclcpp::Node
    {
    public:
        explicit TopologicalGraphGeneratorNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
        ~TopologicalGraphGeneratorNode() = default;

    private:
        // ROS2 subscriptions and publishers
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr skeleton_pub_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr filtered_map_pub_;

        // Timer for periodic processing
        rclcpp::TimerBase::SharedPtr timer_;

        // Parameters
        struct Parameters
        {
            // GRID-FAST parameters
            int grid_fast_num_directions{4};
            int grid_fast_cfilter_size{1};
            int grid_fast_min_group_size{3};
            int grid_fast_obstacle_threshold{20};

            // Skeleton and graph parameters
            int max_bfs_steps{100};
            bool find_entrances{true};
            float merge_threshold{5.0f};
            std::vector<std::string> merge_node_types{"intersection", "endpoint"};

            // Morphological operations
            bool apply_dilation{true};
            bool apply_erosion{false};
            int kernel_size{3};

            // Output parameters
            bool output_to_file{true};
            std::string output_directory{"/tmp/topological_graph"};
            bool include_skeleton_paths{true};

            // Processing parameters
            float occupancy_threshold{0.65f};
            float free_threshold{0.196f};
            bool process_continuously{true};
            int processing_period_ms{5000};
        } params_;

        // Current costmap data
        std::shared_ptr<nav_msgs::msg::OccupancyGrid> current_costmap_;
        std::shared_ptr<Graph> current_graph_;

        // Processing pipeline components
        std::shared_ptr<GridFastProcessor> grid_fast_;
        std::shared_ptr<SkeletonGraphBuilder> skeleton_builder_;
        std::shared_ptr<GraphSerializer> serializer_;

        // Callbacks
        void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
        void processingTimerCallback();

        // Processing pipeline
        void processGraph();
        cv::Mat costmapToMat(const nav_msgs::msg::OccupancyGrid &costmap);
        nav_msgs::msg::OccupancyGrid matToCostmap(const cv::Mat &mat, const std::string &frame_id);

        // Parameter management
        void declareAndLoadParameters();
        void onParametersChanged(const std::vector<rcl_interfaces::msg::Parameter> &parameters);

        // Logging utilities
        void logGraphStats(const Graph &graph);
        void publishGraphVisualization(const cv::Mat &skeleton, const cv::Mat &filtered_map);
    };

    TopologicalGraphGeneratorNode::TopologicalGraphGeneratorNode(const rclcpp::NodeOptions &options)
        : rclcpp::Node("topological_graph_generator", options)
    {

        RCLCPP_INFO(this->get_logger(), "Initializing Topological Graph Generator Node");

        // Declare and load parameters
        declareAndLoadParameters();

        // Initialize processing components
        grid_fast_ = std::make_shared<GridFastProcessor>(
            params_.grid_fast_num_directions,
            params_.grid_fast_cfilter_size,
            params_.grid_fast_min_group_size,
            params_.grid_fast_obstacle_threshold);

        skeleton_builder_ = std::make_shared<SkeletonGraphBuilder>();
        serializer_ = std::make_shared<GraphSerializer>();

        // Create output directory if needed
        if (params_.output_to_file)
        {
            std::system(("mkdir -p " + params_.output_directory).c_str());
        }

        // Subscribe to costmap
        costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "global_costmap",
            rclcpp::SensorDataQoS(),
            std::bind(&TopologicalGraphGeneratorNode::costmapCallback, this, std::placeholders::_1));

        // Publishers for visualization and debugging
        skeleton_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("skeleton_map", 10);
        filtered_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("filtered_map", 10);

        // Timer for continuous processing
        if (params_.process_continuously)
        {
            timer_ = this->create_wall_timer(
                std::chrono::milliseconds(params_.processing_period_ms),
                std::bind(&TopologicalGraphGeneratorNode::processingTimerCallback, this));
        }

        // Register parameter change callback
        auto on_set_parameters_callback = [this](const std::vector<rcl_interfaces::msg::Parameter> &parameters)
        {
            this->onParametersChanged(parameters);
            return rcl_interfaces::msg::SetParametersResult{.successful = true};
        };
        this->add_on_set_parameters_callback(on_set_parameters_callback);

        RCLCPP_INFO(this->get_logger(), "Topological Graph Generator Node initialized successfully");
    }

    void TopologicalGraphGeneratorNode::declareAndLoadParameters()
    {
        // GRID-FAST parameters
        this->declare_parameter("grid_fast.num_directions", params_.grid_fast_num_directions);
        this->declare_parameter("grid_fast.cfilter_size", params_.grid_fast_cfilter_size);
        this->declare_parameter("grid_fast.min_group_size", params_.grid_fast_min_group_size);
        this->declare_parameter("grid_fast.obstacle_threshold", params_.grid_fast_obstacle_threshold);

        // Skeleton and graph parameters
        this->declare_parameter("skeleton.max_bfs_steps", params_.max_bfs_steps);
        this->declare_parameter("skeleton.find_entrances", params_.find_entrances);
        this->declare_parameter("skeleton.merge_threshold", params_.merge_threshold);
        this->declare_parameter("skeleton.merge_node_types", params_.merge_node_types);

        // Morphological parameters
        this->declare_parameter("morphology.apply_dilation", params_.apply_dilation);
        this->declare_parameter("morphology.apply_erosion", params_.apply_erosion);
        this->declare_parameter("morphology.kernel_size", params_.kernel_size);

        // Output parameters
        this->declare_parameter("output.to_file", params_.output_to_file);
        this->declare_parameter("output.directory", params_.output_directory);
        this->declare_parameter("output.include_skeleton_paths", params_.include_skeleton_paths);

        // Processing parameters
        this->declare_parameter("processing.occupancy_threshold", params_.occupancy_threshold);
        this->declare_parameter("processing.free_threshold", params_.free_threshold);
        this->declare_parameter("processing.continuous", params_.process_continuously);
        this->declare_parameter("processing.period_ms", params_.processing_period_ms);

        // Load parameters
        this->get_parameter("grid_fast.num_directions", params_.grid_fast_num_directions);
        this->get_parameter("grid_fast.cfilter_size", params_.grid_fast_cfilter_size);
        this->get_parameter("grid_fast.min_group_size", params_.grid_fast_min_group_size);
        this->get_parameter("grid_fast.obstacle_threshold", params_.grid_fast_obstacle_threshold);

        this->get_parameter("skeleton.max_bfs_steps", params_.max_bfs_steps);
        this->get_parameter("skeleton.find_entrances", params_.find_entrances);
        this->get_parameter("skeleton.merge_threshold", params_.merge_threshold);
        this->get_parameter("skeleton.merge_node_types", params_.merge_node_types);

        this->get_parameter("morphology.apply_dilation", params_.apply_dilation);
        this->get_parameter("morphology.apply_erosion", params_.apply_erosion);
        this->get_parameter("morphology.kernel_size", params_.kernel_size);

        this->get_parameter("output.to_file", params_.output_to_file);
        this->get_parameter("output.directory", params_.output_directory);
        this->get_parameter("output.include_skeleton_paths", params_.include_skeleton_paths);

        this->get_parameter("processing.occupancy_threshold", params_.occupancy_threshold);
        this->get_parameter("processing.free_threshold", params_.free_threshold);
        this->get_parameter("processing.continuous", params_.process_continuously);
        this->get_parameter("processing.period_ms", params_.processing_period_ms);

        RCLCPP_INFO(this->get_logger(), "Parameters loaded successfully");
    }

    void TopologicalGraphGeneratorNode::costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        current_costmap_ = msg;

        if (!params_.process_continuously)
        {
            // Process immediately if not using timer-based processing
            processGraph();
        }
    }

    void TopologicalGraphGeneratorNode::processingTimerCallback()
    {
        if (current_costmap_)
        {
            processGraph();
        }
    }

    void TopologicalGraphGeneratorNode::processGraph()
    {
        if (!current_costmap_)
        {
            RCLCPP_WARN(this->get_logger(), "No costmap available for processing");
            return;
        }

        auto start_time = this->now();
        RCLCPP_INFO(this->get_logger(), "Starting graph generation pipeline");

        try
        {
            // Step 1: Convert costmap to OpenCV Mat
            cv::Mat costmap_mat = costmapToMat(*current_costmap_);

            // Step 2: GRID-FAST processing for intersection detection
            RCLCPP_INFO(this->get_logger(), "Running GRID-FAST intersection detection");
            auto [filtered_map, openings] = grid_fast_->process(costmap_mat);

            // Step 3: Morphological operations
            if (params_.apply_dilation)
            {
                cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(params_.kernel_size, params_.kernel_size));
                cv::dilate(filtered_map, filtered_map, kernel);
                RCLCPP_INFO(this->get_logger(), "Applied morphological dilation");
            }

            if (params_.apply_erosion)
            {
                cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(params_.kernel_size, params_.kernel_size));
                cv::erode(filtered_map, filtered_map, kernel);
                RCLCPP_INFO(this->get_logger(), "Applied morphological erosion");
            }

            // Step 4: Skeletonization
            RCLCPP_INFO(this->get_logger(), "Computing skeleton");
            cv::Mat skeleton = skeleton_builder_->computeSkeleton(filtered_map);

            // Step 5: Distance transform
            cv::Mat dist_map;
            cv::distanceTransform(filtered_map, dist_map, cv::DIST_L2, cv::DIST_5);

            // Step 6: Build topological graph
            RCLCPP_INFO(this->get_logger(), "Building topological graph from skeleton");
            current_graph_ = std::make_shared<Graph>();
            skeleton_builder_->buildGraphFromSkeleton(
                skeleton,
                dist_map,
                filtered_map,
                params_.max_bfs_steps,
                params_.find_entrances,
                params_.merge_threshold,
                *current_graph_);

            // Step 7: Graph numbering and annotation
            RCLCPP_INFO(this->get_logger(), "Numbering graph nodes");
            skeleton_builder_->numberGraphNodes(*current_graph_);

            // Step 8: Output results
            if (params_.output_to_file)
            {
                RCLCPP_INFO(this->get_logger(), "Serializing graph to file");
                std::string timestamp = std::to_string(this->get_clock()->now().nanoseconds());

                serializer_->saveToJSON(
                    *current_graph_,
                    params_.output_directory + "/graph_nodes_" + timestamp + ".json");

                if (params_.include_skeleton_paths)
                {
                    serializer_->saveGraphWithPaths(
                        *current_graph_,
                        params_.output_directory + "/graph_with_paths_" + timestamp + ".json");
                }

                serializer_->saveToPKL(
                    *current_graph_,
                    params_.output_directory + "/weighted_graph_" + timestamp + ".pkl");
            }

            // Publish visualization
            publishGraphVisualization(skeleton, filtered_map);

            // Log statistics
            logGraphStats(*current_graph_);

            auto end_time = this->now();
            auto duration = (end_time - start_time).seconds();
            RCLCPP_INFO(this->get_logger(),
                        "Graph generation completed in %.2f seconds. "
                        "Generated %lu nodes and %lu edges",
                        duration, current_graph_->nodes.size(), current_graph_->edges.size());
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "Error during graph generation: %s", e.what());
        }
    }

    cv::Mat TopologicalGraphGeneratorNode::costmapToMat(const nav_msgs::msg::OccupancyGrid &costmap)
    {
        int width = costmap.info.width;
        int height = costmap.info.height;

        cv::Mat mat(height, width, CV_8UC1);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = y * width + x;
                int8_t value = costmap.data[idx];

                // Convert ROS occupancy grid to binary: -1=unknown, 0-25=free, 26-100=occupied
                if (value < 0)
                {
                    mat.at<uint8_t>(y, x) = 127; // Unknown
                }
                else if (value >= 50)
                {
                    mat.at<uint8_t>(y, x) = 255; // Occupied
                }
                else
                {
                    mat.at<uint8_t>(y, x) = 0; // Free
                }
            }
        }

        return mat;
    }

    nav_msgs::msg::OccupancyGrid TopologicalGraphGeneratorNode::matToCostmap(
        const cv::Mat &mat,
        const std::string &frame_id)
    {

        nav_msgs::msg::OccupancyGrid costmap;
        costmap.header.frame_id = frame_id;
        costmap.header.stamp = this->now();

        costmap.info.width = mat.cols;
        costmap.info.height = mat.rows;
        costmap.info.resolution = current_costmap_->info.resolution;
        costmap.info.origin = current_costmap_->info.origin;

        costmap.data.resize(mat.rows * mat.cols);

        for (int y = 0; y < mat.rows; ++y)
        {
            for (int x = 0; x < mat.cols; ++x)
            {
                uint8_t value = mat.at<uint8_t>(y, x);
                int idx = y * mat.cols + x;

                if (value == 255)
                {
                    costmap.data[idx] = 100; // Occupied
                }
                else if (value == 127)
                {
                    costmap.data[idx] = -1; // Unknown
                }
                else
                {
                    costmap.data[idx] = 0; // Free
                }
            }
        }

        return costmap;
    }

    void TopologicalGraphGeneratorNode::logGraphStats(const Graph &graph)
    {
        RCLCPP_INFO(this->get_logger(), "\n=== Graph Statistics ===");
        RCLCPP_INFO(this->get_logger(), "Total nodes: %lu", graph.nodes.size());
        RCLCPP_INFO(this->get_logger(), "Total edges: %lu", graph.edges.size());

        // Count node types
        int intersection_count = 0, endpoint_count = 0, entrance_count = 0, collision_count = 0;
        for (const auto &node : graph.nodes)
        {
            if (node.second.type == NodeType::INTERSECTION)
                intersection_count++;
            else if (node.second.type == NodeType::ENDPOINT)
                endpoint_count++;
            else if (node.second.type == NodeType::ENTRANCE)
                entrance_count++;
            else if (node.second.type == NodeType::COLLISION)
                collision_count++;
        }

        RCLCPP_INFO(this->get_logger(),
                    "Node types - Intersections: %d, Endpoints: %d, Entrances: %d, Collisions: %d",
                    intersection_count, endpoint_count, entrance_count, collision_count);

        // Compute degree statistics
        std::vector<int> degrees;
        for (const auto &node : graph.nodes)
        {
            int degree = 0;
            for (const auto &edge : graph.edges)
            {
                if (edge.first.first == node.first || edge.first.second == node.first)
                {
                    degree++;
                }
            }
            degrees.push_back(degree);
        }

        if (!degrees.empty())
        {
            float avg_degree = 0;
            for (int d : degrees)
                avg_degree += d;
            avg_degree /= degrees.size();
            RCLCPP_INFO(this->get_logger(), "Average node degree: %.2f", avg_degree);
        }
    }

    void TopologicalGraphGeneratorNode::publishGraphVisualization(
        const cv::Mat &skeleton,
        const cv::Mat &filtered_map)
    {

        if (skeleton_pub_->get_subscription_count() > 0)
        {
            skeleton_pub_->publish(matToCostmap(skeleton, current_costmap_->header.frame_id));
        }

        if (filtered_map_pub_->get_subscription_count() > 0)
        {
            filtered_map_pub_->publish(matToCostmap(filtered_map, current_costmap_->header.frame_id));
        }
    }

    void TopologicalGraphGeneratorNode::onParametersChanged(
        const std::vector<rcl_interfaces::msg::Parameter> &parameters)
    {

        for (const auto &param : parameters)
        {
            RCLCPP_INFO(this->get_logger(), "Parameter changed: %s", param.name.c_str());
        }
    }

} // namespace topological_graph

RCLCPP_COMPONENTS_REGISTER_NODE(topological_graph::TopologicalGraphGeneratorNode)
