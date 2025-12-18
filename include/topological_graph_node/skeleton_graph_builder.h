#ifndef TOPOLOGICAL_GRAPH_SKELETON_GRAPH_BUILDER_H
#define TOPOLOGICAL_GRAPH_SKELETON_GRAPH_BUILDER_H

#include <opencv2/core.hpp>
#include <vector>
#include <map>
#include <deque>
#include <set>
#include <string>
#include "topological_graph_node/graph_types.h"

namespace topological_graph {

/**
 * @brief Builds a graph representation from a skeleton image using multi-source BFS.
 * 
 * STRICTLY IMPLEMENTS logic from Filippo Guarda's 2025 Python implementation.
 */
class SkeletonGraphBuilder {
public:
    SkeletonGraphBuilder();

    /**
     * @brief Computes the skeleton of a binary map using Zhang-Suen thinning.
     */
    cv::Mat computeSkeleton(const cv::Mat& binary_image);

    /**
     * @brief Main pipeline to build the graph.
     */
    void buildGraphFromSkeleton(
        const cv::Mat& skeleton,
        const cv::Mat& dist_map,
        const cv::Mat& filtered_map,
        int max_steps,
        bool find_entrances,
        float merge_threshold,
        Graph& graph);

    /**
     * @brief Renumbers nodes sequentially from 0 to N-1.
     */
    void numberGraphNodes(Graph& graph);

private:
    // Internal state
    struct BFSState {
        int x;
        int y;
        int src_id;
        int remaining_budget;
    };

    // Internal tracking for BFS (reset per build)
    int height_, width_;
    int next_node_id_;

    // --- Core Algorithms ---

    void executeMultiSourceBFS(
        const cv::Mat& skeleton,
        const cv::Mat& dist_map,
        const std::vector<Point>& intersections,
        const std::vector<Point>& endpoints,
        int max_steps,
        bool find_entrances,
        Graph& graph);

    // --- Feature Detection ---

    std::vector<Point> findIntersections(const cv::Mat& skeleton);
    std::vector<Point> findEndpoints(const cv::Mat& skeleton);
    std::vector<Point> extractCoordsFromMask(const cv::Mat& mask);
    cv::Mat detectAllJunctions(const cv::Mat& skeleton);
    
    // --- Node Creation & Path Finding ---

    int createEntranceNode(
        int x, int y, int parent_src,
        const std::map<std::pair<int, Point>, Point>& parent_map,
        const cv::Mat& dist_map,
        Graph& graph);

    int createCollisionNode(
        int x, int y, int src1, int src2,
        const std::map<std::pair<int, Point>, Point>& parent_map,
        Graph& graph);

    std::vector<Point> reconstructPath(
        int x, int y, int src_id,
        const std::map<std::pair<int, Point>, Point>& parent_map);

    std::vector<Point> reconstructPathCollision(
        const Point& curr, const Point& neighbor,
        int src_curr, int src_neighbor,
        const std::map<std::pair<int, Point>, Point>& parent_map);

    std::vector<Point> findSkeletonPath(
        const Point& start,
        const Point& end,
        const cv::Mat& skeleton);

    // --- Clustering & Merging ---

    void mergeCloseNodes(
        Graph& graph, 
        float threshold, 
        const cv::Mat& skeleton);

    Point snapToSkeleton(int x, int y, const cv::Mat& skeleton, int max_radius = 5);
    
    // --- Helpers ---
    std::vector<Point> getNeighbors8(int x, int y) const;
    bool isValidSkeletonPixel(int x, int y, const cv::Mat& skeleton) const;
};

} // namespace topological_graph

#endif // TOPOLOGICAL_GRAPH_SKELETON_GRAPH_BUILDER_H
