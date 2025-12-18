/**
 * Graph data structures and types for topological representation
 */

#pragma once

#include <vector>
#include <map>
#include <set>
#include <array>

namespace topological_graph
{

    enum class NodeType
    {
        INTERSECTION = 0,
        ENDPOINT = 1,
        ENTRANCE = 2,
        COLLISION = 3
    };

    struct Point
    {
        int x = 0;
        int y = 0;

        Point() = default;
        Point(int px, int py) : x(px), y(py) {}

        bool operator==(const Point &other) const
        {
            return x == other.x && y == other.y;
        }

        double distance(const Point &other) const
        {
            int dx = x - other.x;
            int dy = y - other.y;
            return std::sqrt(dx * dx + dy * dy);
        }
    };

    struct GraphNode
    {
        int id = -1;
        Point position;
        NodeType type = NodeType::INTERSECTION;
        std::vector<int> neighbor_ids;
        float local_narrowness = 0.0f; // Distance transform value
        int degree = 0;

        GraphNode() = default;
        GraphNode(int node_id, const Point &pos, NodeType node_type)
            : id(node_id), position(pos), type(node_type) {}
    };

    struct GraphEdge
    {
        int from_node = -1;
        int to_node = -1;
        std::vector<Point> path; // Skeleton path between nodes
        float weight = 0.0f;     // Path length
        int skeleton_distance = 0;

        GraphEdge() = default;
        GraphEdge(int from, int to, const std::vector<Point> &p)
            : from_node(from), to_node(to), path(p)
        {
            skeleton_distance = p.size();
            weight = static_cast<float>(skeleton_distance);
        }

        // For use in maps/sets
        std::pair<int, int> getKey() const
        {
            return {std::min(from_node, to_node), std::max(from_node, to_node)};
        }
    };

    struct Graph
    {
        std::map<int, GraphNode> nodes;
        std::map<std::pair<int, int>, GraphEdge> edges;
        int next_node_id = 0;

        int addNode(const Point &position, NodeType type)
        {
            int id = next_node_id++;
            nodes[id] = GraphNode(id, position, type);
            return id;
        }

        void addEdge(int from, int to, const std::vector<Point> &path)
        {
            auto key = std::make_pair(std::min(from, to), std::max(from, to));
            edges[key] = GraphEdge(from, to, path);

            // Update neighbor lists
            if (nodes.count(from) && nodes.count(to))
            {
                if (std::find(nodes[from].neighbor_ids.begin(),
                              nodes[from].neighbor_ids.end(), to) == nodes[from].neighbor_ids.end())
                {
                    nodes[from].neighbor_ids.push_back(to);
                }
                if (std::find(nodes[to].neighbor_ids.begin(),
                              nodes[to].neighbor_ids.end(), from) == nodes[to].neighbor_ids.end())
                {
                    nodes[to].neighbor_ids.push_back(from);
                }
            }
        }

        bool hasEdge(int from, int to) const
        {
            auto key = std::make_pair(std::min(from, to), std::max(from, to));
            return edges.find(key) != edges.end();
        }

        size_t nodeCount() const { return nodes.size(); }
        size_t edgeCount() const { return edges.size(); }
    };

} // namespace topological_graph
