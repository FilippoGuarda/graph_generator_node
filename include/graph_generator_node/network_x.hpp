#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <stdexcept>
#include <algorithm>

namespace graph_generator_node {

/**
 * @brief Minimal NetworkX-like graph class for C++
 * Stores nodes and undirected edges with path and weight metadata
 */
class NetworkX {
public:
    struct Node {
        std::pair<int, int> position; // (x, y)
        std::string type; // "intersection", "entrance", "collision", "endpoint"
    };

    struct Edge {
        std::vector<std::pair<int, int>> path_pixels; // Path coordinates
        double weight; // Edge weight (typically path length)
    };

    NetworkX() = default;
    ~NetworkX() = default;

    // Node operations
    void addNode(int node_id, const std::pair<int, int>& position, const std::string& type) {
        nodes_[node_id] = {position, type};
    }

    void removeNode(int node_id) {
        nodes_.erase(node_id);
        // Also remove all edges connected to this node
        std::vector<long long> edges_to_remove;
        for (auto& [key, edge] : edges_) {
            if ((key >> 32) == node_id || static_cast<int>(key & 0xFFFFFFFFLL) == node_id) {
                edges_to_remove.push_back(key);
            }
        }
        for (long long key : edges_to_remove) {
            edges_.erase(key);
        }
    }

    bool hasNode(int node_id) const {
        return nodes_.find(node_id) != nodes_.end();
    }

    const std::unordered_map<int, Node>& nodes() const {
        return nodes_;
    }

    // Edge operations - Version 1: std::pair vector
    void addEdge(int from, int to, const std::vector<std::pair<int, int>>& path, double weight) {
        long long key = makeEdgeKey(from, to);
        edges_[key] = {path, weight};
    }

    // Version 2: OpenCV point vector wrapper
    template <typename T>
    void addEdge(int from, int to, const std::vector<T>& path, double weight) {
        long long key = makeEdgeKey(from, to);
        std::vector<std::pair<int, int>> converted_path;
        converted_path.reserve(path.size());
        for (const auto& pt : path) {
            // Assumes type T has .x and .y attributes (like cv::Point)
            converted_path.emplace_back(pt.x, pt.y);
        }
        edges_[key] = {converted_path, weight};
    }

    bool hasEdge(int from, int to) const {
        long long key = makeEdgeKey(from, to);
        return edges_.find(key) != edges_.end();
    }

    const Edge* getEdge(int from, int to) const {
        long long key = makeEdgeKey(from, to);
        auto it = edges_.find(key);
        if (it != edges_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    std::vector<int> getNeighbors(int node_id) const {
        std::vector<int> neighbors;
        for (const auto& [key, edge] : edges_) {
            int from = key >> 32;
            int to = static_cast<int>(key & 0xFFFFFFFFLL);

            if (from == node_id) {
                neighbors.push_back(to);
            } else if (to == node_id) {
                neighbors.push_back(from);
            }
        }
        return neighbors;
    }

    const std::unordered_map<long long, Edge>& edges() const {
        return edges_;
    }

    void clear() {
        nodes_.clear();
        edges_.clear();
    }

private:
    std::unordered_map<int, Node> nodes_;
    std::unordered_map<long long, Edge> edges_;

    /**
     * Create unique edge key from two node IDs
     * Ensures undirected edge representation (always smaller ID first)
     */
    static long long makeEdgeKey(int from, int to) {
        if (from > to) std::swap(from, to);
        return (static_cast<long long>(from) << 32) | static_cast<long long>(to);
    }
};

} // namespace graph_generator_node