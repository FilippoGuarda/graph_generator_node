/**
 * Graph serialization to JSON and pickle formats implementation
 */

#include "topological_graph_node/graph_serializer.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>

namespace topological_graph {

GraphSerializer::GraphSerializer() {}

void GraphSerializer::saveToJSON(
    const Graph& graph,
    const std::string& filepath) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    file << "{\n";
    file << "  \"nodes\": [\n";
    
    // Serialize nodes
    bool first = true;
    for (const auto& [id, node] : graph.nodes) {
        if (!first) file << ",\n";
        file << nodeToJSON(node);
        first = false;
    }
    
    file << "\n  ],\n";
    file << "  \"edges\": [\n";
    
    // Serialize edges
    first = true;
    for (const auto& [key, edge] : graph.edges) {
        if (!first) file << ",\n";
        file << edgeToJSON(edge);
        first = false;
    }
    
    file << "\n  ]\n";
    file << "}\n";
    file.close();
}

void GraphSerializer::saveGraphWithPaths(
    const Graph& graph,
    const std::string& filepath) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    file << "{\n";
    file << "  \"nodes\": [\n";
    
    // Serialize nodes with all details
    bool first = true;
    for (const auto& [id, node] : graph.nodes) {
        if (!first) file << ",\n";
        
        file << "    {\n";
        file << "      \"id\": " << node.id << ",\n";
        file << "      \"position\": [" << node.position.x << ", " << node.position.y << "],\n";
        file << "      \"type\": \"" << nodeTypeToString(node.type) << "\",\n";
        file << "      \"degree\": " << node.degree << ",\n";
        file << "      \"neighbors\": [";
        
        bool first_neighbor = true;
        for (int neighbor_id : node.neighbor_ids) {
            if (!first_neighbor) file << ", ";
            file << neighbor_id;
            first_neighbor = false;
        }
        
        file << "],\n";
        file << "      \"local_narrowness\": " << node.local_narrowness << "\n";
        file << "    }";
        first = false;
    }
    
    file << "\n  ],\n";
    file << "  \"edges\": [\n";
    
    // Serialize edges with complete paths
    first = true;
    for (const auto& [key, edge] : graph.edges) {
        if (!first) file << ",\n";
        
        file << "    {\n";
        file << "      \"from\": " << edge.from_node << ",\n";
        file << "      \"to\": " << edge.to_node << ",\n";
        file << "      \"weight\": " << edge.weight << ",\n";
        file << "      \"skeleton_distance\": " << edge.skeleton_distance << ",\n";
        file << "      \"path\": [\n";
        
        bool first_point = true;
        for (const auto& point : edge.path) {
            if (!first_point) file << ",\n";
            file << "        [" << point.x << ", " << point.y << "]";
            first_point = false;
        }
        
        file << "\n      ]\n";
        file << "    }";
        first = false;
    }
    
    file << "\n  ]\n";
    file << "}\n";
    file.close();
}

void GraphSerializer::saveToPKL(
    const Graph& graph,
    const std::string& filepath) {
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // Simple binary format (not standard pickle)
    // Header: TGRA (Topological GRAph)
    file.write("TGRA", 4);
    
    // Write version
    int32_t version = 1;
    file.write(reinterpret_cast<char*>(&version), sizeof(version));
    
    // Write number of nodes
    int32_t num_nodes = static_cast<int32_t>(graph.nodes.size());
    file.write(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
    
    // Write nodes
    for (const auto& [id, node] : graph.nodes) {
        int32_t node_id = node.id;
        int32_t x = node.position.x;
        int32_t y = node.position.y;
        int32_t type = static_cast<int32_t>(node.type);
        int32_t degree = node.degree;
        float narrowness = node.local_narrowness;
        
        file.write(reinterpret_cast<char*>(&node_id), sizeof(node_id));
        file.write(reinterpret_cast<char*>(&x), sizeof(x));
        file.write(reinterpret_cast<char*>(&y), sizeof(y));
        file.write(reinterpret_cast<char*>(&type), sizeof(type));
        file.write(reinterpret_cast<char*>(&degree), sizeof(degree));
        file.write(reinterpret_cast<char*>(&narrowness), sizeof(narrowness));
    }
    
    // Write number of edges
    int32_t num_edges = static_cast<int32_t>(graph.edges.size());
    file.write(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));
    
    // Write edges
    for (const auto& [key, edge] : graph.edges) {
        int32_t from = edge.from_node;
        int32_t to = edge.to_node;
        float weight = edge.weight;
        int32_t distance = edge.skeleton_distance;
        
        file.write(reinterpret_cast<char*>(&from), sizeof(from));
        file.write(reinterpret_cast<char*>(&to), sizeof(to));
        file.write(reinterpret_cast<char*>(&weight), sizeof(weight));
        file.write(reinterpret_cast<char*>(&distance), sizeof(distance));
        
        // Write path
        int32_t path_length = static_cast<int32_t>(edge.path.size());
        file.write(reinterpret_cast<char*>(&path_length), sizeof(path_length));
        
        for (const auto& point : edge.path) {
            int32_t px = point.x;
            int32_t py = point.y;
            file.write(reinterpret_cast<char*>(&px), sizeof(px));
            file.write(reinterpret_cast<char*>(&py), sizeof(py));
        }
    }
    
    file.close();
}

void GraphSerializer::saveNodePositions(
    const Graph& graph,
    const std::string& filepath) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    file << "{\n";
    file << "  \"nodes\": [\n";
    
    bool first = true;
    for (const auto& [id, node] : graph.nodes) {
        if (!first) file << ",\n";
        
        file << "    {\n";
        file << "      \"id\": " << node.id << ",\n";
        file << "      \"x\": " << node.position.x << ",\n";
        file << "      \"y\": " << node.position.y << ",\n";
        file << "      \"type\": \"" << nodeTypeToString(node.type) << "\"\n";
        file << "    }";
        first = false;
    }
    
    file << "\n  ]\n";
    file << "}\n";
    file.close();
}

void GraphSerializer::saveTaskPoses(
    const Graph& graph,
    const std::string& filepath) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    file << "{\n";
    file << "  \"tasks\": [\n";
    
    bool first = true;
    int task_id = 0;
    
    for (const auto& [id, node] : graph.nodes) {
        if (node.type == NodeType::ENDPOINT || node.type == NodeType::ENTRANCE) {
            if (!first) file << ",\n";
            
            file << "    {\n";
            file << "      \"task_id\": " << task_id++ << ",\n";
            file << "      \"node_id\": " << node.id << ",\n";
            file << "      \"position\": [" << node.position.x << ", " << node.position.y << "],\n";
            file << "      \"type\": \"" << nodeTypeToString(node.type) << "\"\n";
            file << "    }";
            first = false;
        }
    }
    
    file << "\n  ]\n";
    file << "}\n";
    file.close();
}

void GraphSerializer::saveNodeMapping(
    const Graph& graph,
    const std::string& filepath) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    file << "{\n";
    file << "  \"mappings\": [\n";
    
    bool first = true;
    for (const auto& [id, node] : graph.nodes) {
        if (!first) file << ",\n";
        
        file << "    {\n";
        file << "      \"original_id\": " << id << ",\n";
        file << "      \"sequential_id\": " << node.id << "\n";
        file << "    }";
        first = false;
    }
    
    file << "\n  ]\n";
    file << "}\n";
    file.close();
}

Graph GraphSerializer::loadFromJSON(const std::string& filepath) {
    // Simple JSON loader (would use proper JSON library in production)
    Graph graph;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    // For now, just return empty graph
    // In production, use nlohmann::json or similar
    
    file.close();
    return graph;
}

std::string GraphSerializer::nodeToJSON(const GraphNode& node) const {
    std::ostringstream json;
    json << "    {\n";
    json << "      \"id\": " << node.id << ",\n";
    json << "      \"position\": [" << node.position.x << ", " << node.position.y << "],\n";
    json << "      \"type\": \"" << nodeTypeToString(node.type) << "\",\n";
    json << "      \"degree\": " << node.degree << ",\n";
    json << "      \"local_narrowness\": " << node.local_narrowness << "\n";
    json << "    }";
    return json.str();
}

std::string GraphSerializer::edgeToJSON(const GraphEdge& edge) const {
    std::ostringstream json;
    json << "    {\n";
    json << "      \"from\": " << edge.from_node << ",\n";
    json << "      \"to\": " << edge.to_node << ",\n";
    json << "      \"weight\": " << edge.weight << ",\n";
    json << "      \"skeleton_distance\": " << edge.skeleton_distance << "\n";
    json << "    }";
    return json.str();
}

std::string GraphSerializer::nodeTypeToString(NodeType type) const {
    switch (type) {
        case NodeType::INTERSECTION: return "intersection";
        case NodeType::ENDPOINT: return "endpoint";
        case NodeType::ENTRANCE: return "entrance";
        case NodeType::COLLISION: return "collision";
        default: return "unknown";
    }
}

NodeType GraphSerializer::stringToNodeType(const std::string& str) const {
    if (str == "intersection") return NodeType::INTERSECTION;
    if (str == "endpoint") return NodeType::ENDPOINT;
    if (str == "entrance") return NodeType::ENTRANCE;
    if (str == "collision") return NodeType::COLLISION;
    return NodeType::INTERSECTION;
}

} // namespace topological_graph
