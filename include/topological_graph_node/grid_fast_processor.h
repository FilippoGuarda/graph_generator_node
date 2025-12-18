/**
 * GRID-FAST: Grid-based intersection detection
 * Optimized C++ implementation of the GRID-FAST algorithm
 *
 * Reference: Fredriksson, S. et al. "GRID-FAST: A Grid-based Intersection
 * Detection for Fast Semantic Topometric Mapping"
 */

#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <map>

namespace topological_graph
{

    struct Opening
    {
        cv::Point start;
        cv::Point end;
        int label = 1;
    };

    class GridFastProcessor
    {
    public:
        /**
         * Initialize GRID-FAST processor
         *
         * Args:
         *   num_directions: Number of scan directions (typically 4 or 8)
         *   cfilter_size: Connected filter size for gap filling
         *   min_group_size: Minimum group size to consider as gap
         *   obstacle_threshold: Minimum area to keep as obstacle
         */
        GridFastProcessor(
            int num_directions = 4,
            int cfilter_size = 1,
            int min_group_size = 3,
            int obstacle_threshold = 20);

        ~GridFastProcessor() = default;

        /**
         * Process occupancy map to detect openings and intersections
         *
         * Args:
         *   image: Binary or grayscale occupancy map (255=occupied, 0=free, 127=unknown)
         *   is_pgm: If true, interpret as ROS2 PGM format with thresholds
         *
         * Returns:
         *   Pair of (filtered_map, detected_openings)
         */
        std::pair<cv::Mat, std::vector<Opening>> process(
            const cv::Mat &image,
            bool is_pgm = false,
            float occupied_threshold = 0.65f,
            float free_threshold = 0.196f);

        /**
         * Convert ROS2 PGM map to internal representation
         */
        static cv::Mat pgmToMap(
            const cv::Mat &pgm_image,
            float occupied_thresh = 0.65f,
            float free_thresh = 0.196f,
            bool negate = false);

        /**
         * Convert internal map representation back to PGM
         */
        static cv::Mat mapToPgm(
            const cv::Mat &map_data,
            bool negate = false);

    private:
        int num_directions_;
        int cfilter_size_;
        int min_group_size_;
        int obstacle_threshold_;

        // Transform maps for each rotation direction
        std::vector<cv::Mat> transform_maps_x_;
        std::vector<cv::Mat> transform_maps_y_;

        static constexpr int MAP_OCCUPIED = 100;
        static constexpr int MAP_UNOCCUPIED = 0;
        static constexpr int MAP_UNKNOWN = -1;

        /**
         * Generate rotation transform maps
         */
        void updateTransforms(const cv::Size &map_size);

        /**
         * Analyze gaps in rotated view
         */
        void analyzeGaps(const cv::Mat &map_data, cv::Mat &filtered_map);

        /**
         * Remove small obstacles
         */
        cv::Mat removeSmallObstacles(const cv::Mat &map_data);

        /**
         * Detect doorway openings
         */
        std::vector<Opening> detectOpenings(const cv::Mat &map_data);

        /**
         * Bresenham line drawing for opening detection
         */
        std::vector<cv::Point> bresenhamLine(const cv::Point &start, const cv::Point &end);
    };

} // namespace topological_graph
