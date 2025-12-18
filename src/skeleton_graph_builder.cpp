/**
 * Skeleton-based graph builder using multi-source BFS implementation
 */

#include "topological_graph_node/skeleton_graph_builder.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <queue>
#include <deque>
#include <algorithm>
#include <cmath>

namespace topological_graph {

SkeletonGraphBuilder::SkeletonGraphBuilder() {}

cv::Mat SkeletonGraphBuilder::computeSkeleton(const cv::Mat& binary_image) {
    // Convert to binary if needed
    cv::Mat binary = binary_image.clone();
    if (binary.type() != CV_8UC1) {
        cv::cvtColor(binary, binary, cv::COLOR_BGR2GRAY);
        cv::threshold(binary, binary, 127, 255, cv::THRESH_BINARY);
    }
    
    // Normalize to 0-1 for skeletonize
    cv::Mat normalized;
    binary.convertTo(normalized, CV_8U);
    
    // Apply Zhang-Suen thinning
    cv::Mat skeleton = normalized.clone();
    
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    
    // Iterative thinning
    for (int iter = 0; iter < 20; ++iter) {
        cv::Mat eroded, temp;
        
        cv::erode(skeleton, eroded, element);
        cv::dilate(eroded, temp, element);
        cv::subtract(skeleton, temp, temp);
        
        cv::Mat sub;
        cv::subtract(skeleton, temp, skeleton);
        
        if (cv::countNonZero(temp) == 0) break;
    }
    
    // Convert back to binary (255 for skeleton pixels)
    cv::Mat result = cv::Mat::zeros(skeleton.size(), CV_8UC1);
    for (int y = 0; y < skeleton.rows; ++y) {
        for (int x = 0; x < skeleton.cols; ++x) {
            if (skeleton.at<uint8_t>(y, x) > 127) {
                result.at<uint8_t>(y, x) = 255;
            }
        }
    }
    
    return result;
}

void SkeletonGraphBuilder::buildGraphFromSkeleton(
    const cv::Mat& skeleton,
    const cv::Mat& dist_map,
    const cv::Mat& filtered_map,
    int max_steps,
    bool find_entrances,
    float merge_threshold,
    Graph& graph) {
    
    // Find skeleton features
    std::vector<Point> intersections = findIntersections(skeleton);
    std::vector<Point> endpoints = findEndpoints(skeleton);
    
    // Add nodes to graph
    int node_id = 0;
    std::map<int, int> intersection_node_map;
    std::map<int, int> endpoint_node_map;
    
    for (size_t i = 0; i < intersections.size(); ++i) {
        int nid = graph.addNode(intersections[i], NodeType::INTERSECTION);
        intersection_node_map[i] = nid;
    }
    
    for (size_t i = 0; i < endpoints.size(); ++i) {
        int nid = graph.addNode(endpoints[i], NodeType::ENDPOINT);
        endpoint_node_map[i] = nid;
    }
    
    // Execute multi-source BFS
    executeMultiSourceBFS(
        skeleton, dist_map, intersections, endpoints,
        max_steps, find_entrances, graph
    );
    
    // Merge close nodes if threshold specified
    if (merge_threshold > 0) {
        mergeCloseNodes(graph, merge_threshold);
    }
}

void SkeletonGraphBuilder::executeMultiSourceBFS(
    const cv::Mat& skeleton,
    const cv::Mat& dist_map,
    const std::vector<Point>& intersections,
    const std::vector<Point>& endpoints,
    int max_steps,
    bool find_entrances,
    Graph& graph) {
    
    int height = skeleton.rows;
    int width = skeleton.cols;
    
    // Visited source tracking
    cv::Mat visited_src = cv::Mat::full(skeleton.size(), -1, CV_32S);
    cv::Mat visited_dist = cv::Mat::full(skeleton.size(), INT_MAX, CV_32S);
    
    // Parent tracking for path reconstruction
    std::map<std::pair<int, Point>, Point> parent_map;
    
    std::deque<BFSState> queue;
    
    // Collect all initial nodes
    std::vector<std::pair<Point, int>> all_sources;
    for (size_t i = 0; i < intersections.size(); ++i) {
        all_sources.push_back({intersections[i], static_cast<int>(i)});
    }
    for (size_t i = 0; i < endpoints.size(); ++i) {
        all_sources.push_back({endpoints[i], static_cast<int>(intersections.size() + i)});
    }
    
    // Initialize queue with all sources
    for (const auto& [pos, src_id] : all_sources) {
        if (isValidSkeletonPixel(pos.x, pos.y, skeleton)) {
            int initial_budget = static_cast<int>(dist_map.at<float>(pos.y, pos.x));
            visited_src.at<int32_t>(pos.y, pos.x) = src_id;
            visited_dist.at<int32_t>(pos.y, pos.x) = 0;
            parent_map[{src_id, pos}] = {-1, -1};
            
            queue.push_back({pos.x, pos.y, src_id, initial_budget});
        }
    }
    
    // Level-synchronized BFS
    while (!queue.empty()) {
        auto state = queue.front();
        queue.pop_front();
        
        int x = state.x;
        int y = state.y;
        int src_id = state.src_id;
        int remaining_budget = state.remaining_budget;
        
        // Expand to 8-connected neighbors
        for (const auto& neighbor : getNeighbors8(x, y, height, width)) {
            int nx = neighbor.x;
            int ny = neighbor.y;
            
            if (!isValidSkeletonPixel(nx, ny, skeleton)) continue;
            
            int32_t visited_source = visited_src.at<int32_t>(ny, nx);
            
            if (visited_source == -1) {
                // Unvisited pixel - claim it
                int new_dist = visited_dist.at<int32_t>(y, x) + 1;
                visited_src.at<int32_t>(ny, nx) = src_id;
                visited_dist.at<int32_t>(ny, nx) = new_dist;
                parent_map[{src_id, {nx, ny}}] = {x, y};
                
                if (remaining_budget > 1) {
                    // Continue with current source
                    queue.push_back({nx, ny, src_id, remaining_budget - 1});
                } else if (remaining_budget == 1 && find_entrances) {
                    // Budget exhausted - create entrance node
                    int entrance_id = createEntranceNode(nx, ny, src_id, skeleton, parent_map, graph);
                    int entrance_budget = static_cast<int>(dist_map.at<float>(ny, nx));
                    queue.push_back({nx, ny, entrance_id, entrance_budget});
                } else if (remaining_budget <= 0) {
                    // Reset budget but keep same source
                    int reset_budget = static_cast<int>(dist_map.at<float>(ny, nx));
                    queue.push_back({nx, ny, src_id, reset_budget});
                }
            } else if (visited_source != src_id) {
                // Collision with different source
                int other_dist = visited_dist.at<int32_t>(ny, nx);
                int local_narrowness = static_cast<int>(dist_map.at<float>(ny, nx));
                
                bool current_exhausted = (remaining_budget <= 0);
                bool other_exhausted = (other_dist >= local_narrowness);
                
                if (current_exhausted && other_exhausted) {
                    // Both budgets exhausted - create collision node
                    int collision_id = createCollisionNode(nx, ny, src_id, visited_source, skeleton, parent_map, graph);
                    visited_src.at<int32_t>(ny, nx) = collision_id;
                    visited_dist.at<int32_t>(ny, nx) = 0;
                    parent_map[{collision_id, {nx, ny}}] = {-1, -1};
                    
                    queue.push_back({nx, ny, collision_id, max_steps});
                } else {
                    // Standard collision - create edge between sources
                    // Find the actual source node IDs
                    for (const auto& [pos, orig_id] : all_sources) {
                        if (orig_id == src_id && graph.nodes.count(pos.x * width + pos.y)) {
                            int node1 = pos.x * width + pos.y;  // This is a simplified mapping
                            // Create edge
                            std::vector<Point> path = findSkeletonPath({x, y}, {nx, ny}, skeleton);
                            if (!path.empty()) {
                                // Add edge to graph
                                for (auto& [nid, node] : graph.nodes) {
                                    if (node.position == {x, y}) {
                                        for (auto& [nid2, node2] : graph.nodes) {
                                            if (node2.position == {nx, ny}) {
                                                graph.addEdge(nid, nid2, path);
                                                break;
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector<Point> SkeletonGraphBuilder::findIntersections(const cv::Mat& skeleton) {
    std::vector<Point> intersections;
    
    for (int y = 1; y < skeleton.rows - 1; ++y) {
        for (int x = 1; x < skeleton.cols - 1; ++x) {
            if (skeleton.at<uint8_t>(y, x) == 0) continue;
            
            int neighbors = countSkeletonNeighbors(x, y, skeleton);
            if (neighbors >= 3) {
                intersections.push_back({x, y});
            }
        }
    }
    
    return intersections;
}

std::vector<Point> SkeletonGraphBuilder::findEndpoints(const cv::Mat& skeleton) {
    std::vector<Point> endpoints;
    
    for (int y = 1; y < skeleton.rows - 1; ++y) {
        for (int x = 1; x < skeleton.cols - 1; ++x) {
            if (skeleton.at<uint8_t>(y, x) == 0) continue;
            
            int neighbors = countSkeletonNeighbors(x, y, skeleton);
            if (neighbors == 1) {
                endpoints.push_back({x, y});
            }
        }
    }
    
    return endpoints;
}

int SkeletonGraphBuilder::countSkeletonNeighbors(int x, int y, const cv::Mat& skeleton) const {
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < skeleton.cols && ny >= 0 && ny < skeleton.rows) {
                if (skeleton.at<uint8_t>(ny, nx) > 127) count++;
            }
        }
    }
    return count;
}

bool SkeletonGraphBuilder::isValidSkeletonPixel(int x, int y, const cv::Mat& skeleton) const {
    return x >= 0 && x < skeleton.cols && y >= 0 && y < skeleton.rows &&
           skeleton.at<uint8_t>(y, x) > 127;
}

std::vector<Point> SkeletonGraphBuilder::getNeighbors8(int x, int y, int height, int width) const {
    std::vector<Point> result;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                result.push_back({nx, ny});
            }
        }
    }
    return result;
}

int SkeletonGraphBuilder::createEntranceNode(
    int x, int y, int parent_src,
    const cv::Mat& skeleton,
    const std::map<std::pair<int, Point>, Point>& parent_map,
    Graph& graph) {
    
    int entrance_id = graph.addNode({x, y}, NodeType::ENTRANCE);
    
    // Reconstruct path from parent source
    std::vector<Point> path = reconstructPath(x, y, parent_src, parent_map);
    
    // Find parent node and create edge
    for (auto& [nid, node] : graph.nodes) {
        if (node.type == NodeType::INTERSECTION || node.type == NodeType::ENDPOINT) {
            if (!path.empty()) {
                Point first_on_path = path[0];
                for (const auto& skeleton_pos : path) {
                    if (isValidSkeletonPixel(skeleton_pos.x, skeleton_pos.y, skeleton)) {
                        // Check if this connects to parent node
                        if (node.position == skeleton_pos || 
                            (std::abs(node.position.x - skeleton_pos.x) <= 1 && 
                             std::abs(node.position.y - skeleton_pos.y) <= 1)) {
                            graph.addEdge(nid, entrance_id, path);
                            return entrance_id;
                        }
                    }
                }
            }
        }
    }
    
    return entrance_id;
}

int SkeletonGraphBuilder::createCollisionNode(
    int x, int y, int src1, int src2,
    const cv::Mat& skeleton,
    const std::map<std::pair<int, Point>, Point>& parent_map,
    Graph& graph) {
    
    int collision_id = graph.addNode({x, y}, NodeType::COLLISION);
    
    // Reconstruct paths from both sources
    std::vector<Point> path1 = reconstructPath(x, y, src1, parent_map);
    std::vector<Point> path2 = reconstructPath(x, y, src2, parent_map);
    
    // Connect to both source nodes
    for (auto& [nid, node] : graph.nodes) {
        if ((node.type == NodeType::INTERSECTION || node.type == NodeType::ENDPOINT) &&
            node.position != Point{x, y}) {
            if (!path1.empty()) {
                graph.addEdge(nid, collision_id, path1);
            }
        }
    }
    
    return collision_id;
}

std::vector<Point> SkeletonGraphBuilder::reconstructPath(
    int x, int y, int src_id,
    const std::map<std::pair<int, Point>, Point>& parent_map) {
    
    std::vector<Point> path;
    Point current{x, y};
    
    while (current.x != -1 && current.y != -1) {
        path.push_back(current);
        
        auto it = parent_map.find({src_id, current});
        if (it == parent_map.end()) break;
        
        current = it->second;
    }
    
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<Point> SkeletonGraphBuilder::findSkeletonPath(
    const Point& start,
    const Point& end,
    const cv::Mat& skeleton) {
    
    if (!isValidSkeletonPixel(start.x, start.y, skeleton) ||
        !isValidSkeletonPixel(end.x, end.y, skeleton)) {
        return {};
    }
    
    std::vector<Point> path;
    cv::Mat visited = cv::Mat::zeros(skeleton.size(), CV_8UC1);
    std::queue<std::pair<Point, std::vector<Point>>> q;
    
    q.push({start, {start}});
    visited.at<uint8_t>(start.y, start.x) = 1;
    
    while (!q.empty()) {
        auto [current, current_path] = q.front();
        q.pop();
        
        if (current.x == end.x && current.y == end.y) {
            return current_path;
        }
        
        for (const auto& neighbor : getNeighbors8(current.x, current.y, skeleton.rows, skeleton.cols)) {
            if (isValidSkeletonPixel(neighbor.x, neighbor.y, skeleton) &&
                !visited.at<uint8_t>(neighbor.y, neighbor.x)) {
                visited.at<uint8_t>(neighbor.y, neighbor.x) = 1;
                auto new_path = current_path;
                new_path.push_back(neighbor);
                q.push({neighbor, new_path});
            }
        }
    }
    
    return {};
}

void SkeletonGraphBuilder::mergeCloseNodes(Graph& graph, float threshold) {
    // Find clusters of close nodes
    std::vector<int> clusters(graph.nodes.size(), -1);
    int cluster_id = 0;
    
    auto it1 = graph.nodes.begin();
    for (size_t i = 0; i < graph.nodes.size(); ++it1, ++i) {
        if (clusters[i] != -1) continue;
        
        clusters[i] = cluster_id;
        auto it2 = it1;
        ++it2;
        
        for (size_t j = i + 1; j < graph.nodes.size(); ++it2, ++j) {
            if (clusters[j] != -1) continue;
            
            float dist = std::sqrt(
                std::pow(it1->second.position.x - it2->second.position.x, 2) +
                std::pow(it1->second.position.y - it2->second.position.y, 2)
            );
            
            if (dist <= threshold) {
                clusters[j] = cluster_id;
            }
        }
        
        cluster_id++;
    }
    
    // Merge nodes in same cluster
    // Implementation simplified - would need proper node merging logic
}

void SkeletonGraphBuilder::numberGraphNodes(Graph& graph) {
    // Renumber nodes sequentially starting from 0
    std::map<int, int> id_mapping;
    int new_id = 0;
    
    for (auto& [old_id, node] : graph.nodes) {
        id_mapping[old_id] = new_id++;
        node.id = id_mapping[old_id];
    }
    
    // Update graph structure with new IDs
    Graph new_graph;
    for (const auto& [old_id, node] : graph.nodes) {
        new_graph.nodes[id_mapping[old_id]] = node;
        new_graph.nodes[id_mapping[old_id]].id = id_mapping[old_id];
    }
    
    // Remap edges
    for (const auto& [key, edge] : graph.edges) {
        auto new_key = std::make_pair(
            id_mapping[key.first],
            id_mapping[key.second]
        );
        new_graph.edges[new_key] = edge;
        new_graph.edges[new_key].from_node = id_mapping[edge.from_node];
        new_graph.edges[new_key].to_node = id_mapping[edge.to_node];
    }
    
    graph = new_graph;
    graph.next_node_id = new_id;
}

} // namespace topological_graph
