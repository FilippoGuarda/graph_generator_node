/**
 * Skeleton-based graph builder using multi-source BFS
 * Implements topological graph generation from skeleton images
 */

#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <queue>
#include <deque>
#include <map>
#include <set>
#include <memory>
#include <cmath>

#include "graph_types.h"

namespace topological_graph
{

    class SkeletonGraphBuilder
    {
    public:
        SkeletonGraphBuilder();
        ~SkeletonGraphBuilder() = default;

        /**
         * Compute skeleton using morphological thinning
         */
        cv::Mat computeSkeleton(const cv::Mat &binary_image);

        /**
         * Build topological graph from skeleton using multi-source BFS
         *
         * Args:
         *   skeleton: Binary skeleton image
         *   dist_map: Distance transform of the image
         *   filtered_map: Filtered occupancy map
         *   max_steps: Maximum BFS propagation steps
         *   find_entrances: Whether to create entrance nodes
         *   merge_threshold: Distance threshold for merging nodes
         *   graph: Output graph structure
         */
        void buildGraphFromSkeleton(
            const cv::Mat &skeleton,
            const cv::Mat &dist_map,
            const cv::Mat &filtered_map,
            int max_steps,
            bool find_entrances,
            float merge_threshold,
            Graph &graph);

        /**
         * Number all graph nodes sequentially
         */
        void numberGraphNodes(Graph &graph);

    private:
        static constexpr int VISITED_SENTINEL = -1;

        struct BFSState
        {
            int x, y;
            int src_id;
            int remaining_budget;
        };

        struct SkeletonPoint
        {
            bool is_intersection = false;
            bool is_endpoint = false;
            int neighbor_count = 0;
        };

        /**
         * Find intersection and endpoint pixels in skeleton
         */
        std::vector<Point> findIntersections(const cv::Mat &skeleton);
        std::vector<Point> findEndpoints(const cv::Mat &skeleton);

        /**
         * Multi-source BFS for graph construction
         */
        void executeMultiSourceBFS(
            const cv::Mat &skeleton,
            const cv::Mat &dist_map,
            const std::vector<Point> &intersections,
            const std::vector<Point> &endpoints,
            int max_steps,
            bool find_entrances,
            Graph &graph);

        /**
         * Process BFS queue
         */
        void processBFSQueue(
            std::deque<BFSState> &queue,
            const cv::Mat &skeleton,
            const cv::Mat &dist_map,
            cv::Mat &visited_src,
            cv::Mat &visited_dist,
            bool find_entrances,
            int max_steps,
            Graph &graph,
            std::map<std::pair<int, Point>, Point> &parent_map);

        /**
         * Create entrance node when budget exhausted
         */
        int createEntranceNode(
            int x, int y, int parent_src,
            const cv::Mat &skeleton,
            const std::map<std::pair<int, Point>, Point> &parent_map,
            Graph &graph);

        /**
         * Create collision node at intersection of two sources
         */
        int createCollisionNode(
            int x, int y, int src1, int src2,
            const cv::Mat &skeleton,
            const std::map<std::pair<int, Point>, Point> &parent_map,
            Graph &graph);

        /**
         * Reconstruct path from current position back to source
         */
        std::vector<Point> reconstructPath(
            int x, int y, int src_id,
            const std::map<std::pair<int, Point>, Point> &parent_map);

        /**
         * Find skeleton path between two points
         */
        std::vector<Point> findSkeletonPath(
            const Point &start,
            const Point &end,
            const cv::Mat &skeleton);

        /**
         * Merge nodes closer than threshold
         */
        void mergeCloseNodes(Graph &graph, float threshold);

        /**
         * Check if pixel is valid skeleton pixel
         */
        bool isValidSkeletonPixel(int x, int y, const cv::Mat &skeleton) const;

        /**
         * Get 8-connected neighbors
         */
        std::vector<Point> getNeighbors8(int x, int y, int height, int width) const;

        /**
         * Detect junctions in skeleton (3+ connected branches)
         */
        cv::Mat detectJunctions(const cv::Mat &skeleton);

        /**
         * Count neighbors of a pixel in skeleton
         */
        int countSkeletonNeighbors(int x, int y, const cv::Mat &skeleton) const;
    };

} // namespace topological_graph
