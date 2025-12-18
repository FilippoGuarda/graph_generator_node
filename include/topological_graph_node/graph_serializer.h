/**
 * Graph serialization to multiple formats (JSON, Python pickle, etc.)
 */

#pragma once

#include <string>
#include <memory>
#include "graph_types.h"

namespace topological_graph
{

    class GraphSerializer
    {
    public:
        GraphSerializer();
        ~GraphSerializer() = default;

        /**
         * Save graph to JSON format compatible with SMRTA
         */
        void saveToJSON(
            const Graph &graph,
            const std::string &filepath);

        /**
         * Save graph with complete skeleton paths
         */
        void saveGraphWithPaths(
            const Graph &graph,
            const std::string &filepath);

        /**
         * Save graph to Python pickle format
         */
        void saveToPKL(
            const Graph &graph,
            const std::string &filepath);

        /**
         * Save node positions as JSON
         */
        void saveNodePositions(
            const Graph &graph,
            const std::string &filepath);

        /**
         * Save task poses (endpoint/entrance nodes)
         */
        void saveTaskPoses(
            const Graph &graph,
            const std::string &filepath);

        /**
         * Save node mapping (original IDs to sequential IDs)
         */
        void saveNodeMapping(
            const Graph &graph,
            const std::string &filepath);

        /**
         * Load graph from JSON
         */
        Graph loadFromJSON(const std::string &filepath);

    private:
        /**
         * Helper to generate JSON node object
         */
        std::string nodeToJSON(const GraphNode &node) const;

        /**
         * Helper to generate JSON edge object
         */
        std::string edgeToJSON(const GraphEdge &edge) const;

        /**
         * Convert NodeType to string
         */
        std::string nodeTypeToString(NodeType type) const;

        /**
         * Convert string to NodeType
         */
        NodeType stringToNodeType(const std::string &str) const;
    };

} // namespace topological_graph
