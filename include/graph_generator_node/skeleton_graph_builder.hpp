#pragma once

#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include <queue>

namespace graph_generator_node {

// Simple graph class to mimic NetworkX functionality
class NetworkX {
public:
    struct Node {
        int id;
        std::pair<int, int> position; // (x, y)
        std::string type; // "intersection", "endpoint", "entrance", "collision"
    };

    struct Edge {
        int u, v;
        std::vector<cv::Point2i> path_pixels;
        double weight;
    };

    void addNode(int id, const std::pair<int, int>& pos, const std::string& node_type) {
        nodes_[id] = {id, pos, node_type};
    }

    void addEdge(int u, int v, const std::vector<cv::Point2i>& path, double weight) {
        edges_.push_back({u, v, path, weight});
    }

    bool hasEdge(int u, int v) const {
        for (const auto& edge : edges_) {
            if ((edge.u == u && edge.v == v) || (edge.u == v && edge.v == u)) {
                return true;
            }
        }
        return false;
    }

    void clear() {
        nodes_.clear();
        edges_.clear();
    }

    const std::unordered_map<int, Node>& nodes() const {
        return nodes_;
    }

    const std::vector<Edge>& edges() const {
        return edges_;
    }

    // New method: Remove node by ID
    void removeNode(int id) {
        nodes_.erase(id);
        // Also remove edges connected to this node
        auto it = edges_.begin();
        while (it != edges_.end()) {
            if (it->u == id || it->v == id) {
                it = edges_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // New method: Get neighbors of a node
    std::vector<int> getNeighbors(int node_id) const {
        std::vector<int> neighbors;
        for (const auto& edge : edges_) {
            if (edge.u == node_id) neighbors.push_back(edge.v);
            else if (edge.v == node_id) neighbors.push_back(edge.u);
        }
        return neighbors;
    }

    // New method: Get edge between two nodes
    const Edge* getEdge(int u, int v) const {
        for (const auto& edge : edges_) {
            if ((edge.u == u && edge.v == v) || (edge.u == v && edge.v == u)) {
                return &edge;
            }
        }
        return nullptr;
    }

private:
    std::unordered_map<int, Node> nodes_;
    std::vector<Edge> edges_;
};

// Skeleton graph builder - implements the corrected algorithm
class SkeletonGraphBuilder {
public:
    SkeletonGraphBuilder(const cv::Mat& skeleton, const cv::Mat& distmap);

    // Main entry point: builds graph from skeleton and distance map
    std::pair<std::shared_ptr<NetworkX>, std::unordered_map<int, std::pair<int, int>>>
    buildGraph(int max_steps = 100, bool find_entrances = true);

    // Merge close nodes based on distance threshold
    std::unordered_map<int, std::pair<int, int>> mergeCloseNodes(
        double distance_threshold,
        const std::vector<std::string>& node_types_to_merge = {});

private:
    // Algorithm steps
    void buildGraphMultiSourceBFS(
        const std::unordered_map<int, std::pair<int, int>>& intersection_positions,
        const std::unordered_map<int, std::pair<int, int>>& endpoint_positions,
        bool find_entrances,
        int max_steps);

    void executeBFS(
        std::queue<std::tuple<int, int, int, int>>& queue,
        cv::Mat& visited_src,
        cv::Mat& visited_dist,
        std::unordered_map<long long, cv::Point2i>& parent_map,
        bool find_entrances,
        int max_steps);

    int createEntranceNode(
        const cv::Point2i& p,
        int parent_src_id,
        std::unordered_map<long long, cv::Point2i>& parent_map);

    int createCollisionNode(
        const cv::Point2i& p,
        int src1,
        int src2,
        std::unordered_map<long long, cv::Point2i>& parent_map);

    std::vector<cv::Point2i> reconstructPath(
        const cv::Point2i& p,
        int src_id,
        const std::unordered_map<long long, cv::Point2i>& parent_map);

    std::vector<cv::Point2i> reconstructPathCollision(
        const cv::Point2i& cur_xy,
        const cv::Point2i& neigh_xy,
        int src_current,
        int src_neighbor,
        const std::unordered_map<long long, cv::Point2i>& parent_map);

    // Skeleton analysis
    void findSkeletonPoints(cv::Mat& intersections_mask, cv::Mat& endpoints_mask);

    std::vector<cv::Point2i> extractCoordsFromMask(const cv::Mat& mask);

    // Merge helper
    std::pair<int, int> snapToSkeleton(int x, int y, int max_search_radius = 5);

    // Utilities
    std::vector<cv::Point2i> neighbors8(int x, int y);
    long long packKey(int src_id, int x, int y);

    // Member variables
    cv::Mat skeleton_;
    cv::Mat distmap_;
    int height_;
    int width_;
    std::shared_ptr<NetworkX> graph_;
    int next_node_id_;
};

} // namespace graph_generator_node
