/**
 * GRID-FAST: Grid-based intersection detection implementation
 * Optimized C++ implementation of the GRID-FAST algorithm for map cleanup
 */

#include "topological_graph_node/grid_fast_processor.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace topological_graph {

GridFastProcessor::GridFastProcessor(
    int num_directions,
    int cfilter_size,
    int min_group_size,
    int obstacle_threshold)
    : num_directions_(num_directions),
      cfilter_size_(cfilter_size),
      min_group_size_(min_group_size),
      obstacle_threshold_(obstacle_threshold) {}

std::pair<cv::Mat, std::vector<Opening>> GridFastProcessor::process(
    const cv::Mat& image,
    bool is_pgm,
    float occupied_threshold,
    float free_threshold) {
    
    // Convert to internal map representation
    cv::Mat map_data;
    if (is_pgm) {
        map_data = pgmToMap(image, occupied_threshold, free_threshold, false);
    } else {
        // Legacy binary conversion
        map_data = cv::Mat::zeros(image.size(), CV_32S);
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                uint8_t val = image.at<uint8_t>(y, x);
                if (val >= 250) {
                    map_data.at<int32_t>(y, x) = MAP_UNOCCUPIED;
                } else if (val <= 5) {
                    map_data.at<int32_t>(y, x) = MAP_OCCUPIED;
                } else {
                    map_data.at<int32_t>(y, x) = MAP_UNKNOWN;
                }
            }
        }
    }
    
    // Remove small obstacles
    map_data = removeSmallObstacles(map_data);
    
    // Update transforms for this map size
    updateTransforms(map_data.size());
    
    // Analyze gaps with rotations
    analyzeGaps(map_data, map_data);
    
    // Apply dilation for connectivity
    cv::Mat filtered_map = cv::Mat::zeros(map_data.size(), CV_8UC1);
    for (int y = 0; y < map_data.rows; ++y) {
        for (int x = 0; x < map_data.cols; ++x) {
            if (map_data.at<int32_t>(y, x) == MAP_UNOCCUPIED) {
                filtered_map.at<uint8_t>(y, x) = 255;
            } else if (map_data.at<int32_t>(y, x) == MAP_OCCUPIED) {
                filtered_map.at<uint8_t>(y, x) = 0;
            } else {
                filtered_map.at<uint8_t>(y, x) = 127;
            }
        }
    }
    
    // Apply morphological dilation
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(filtered_map, filtered_map, kernel, 1);
    
    // Detect openings
    std::vector<Opening> openings = detectOpenings(map_data);
    
    return {filtered_map, openings};
}

cv::Mat GridFastProcessor::pgmToMap(
    const cv::Mat& pgm_image,
    float occupied_thresh,
    float free_thresh,
    bool negate) {
    
    cv::Mat map_data = cv::Mat::full(pgm_image.size(), MAP_UNKNOWN, CV_32S);
    
    for (int y = 0; y < pgm_image.rows; ++y) {
        for (int x = 0; x < pgm_image.cols; ++x) {
            uint8_t pixel = pgm_image.at<uint8_t>(y, x);
            float p;
            
            if (negate) {
                p = pixel / 255.0f;
            } else {
                p = (255.0f - pixel) / 255.0f;
            }
            
            if (p > occupied_thresh) {
                map_data.at<int32_t>(y, x) = MAP_OCCUPIED;
            } else if (p < free_thresh) {
                map_data.at<int32_t>(y, x) = MAP_UNOCCUPIED;
            } else {
                map_data.at<int32_t>(y, x) = MAP_UNKNOWN;
            }
        }
    }
    
    return map_data;
}

cv::Mat GridFastProcessor::mapToPgm(
    const cv::Mat& map_data,
    bool negate) {
    
    cv::Mat pgm = cv::Mat::zeros(map_data.size(), CV_8UC1);
    
    for (int y = 0; y < map_data.rows; ++y) {
        for (int x = 0; x < map_data.cols; ++x) {
            int32_t val = map_data.at<int32_t>(y, x);
            
            if (negate) {
                if (val == MAP_OCCUPIED) pgm.at<uint8_t>(y, x) = 255;
                else if (val == MAP_UNOCCUPIED) pgm.at<uint8_t>(y, x) = 0;
                else pgm.at<uint8_t>(y, x) = 127;
            } else {
                if (val == MAP_OCCUPIED) pgm.at<uint8_t>(y, x) = 0;
                else if (val == MAP_UNOCCUPIED) pgm.at<uint8_t>(y, x) = 254;
                else pgm.at<uint8_t>(y, x) = 254;
            }
        }
    }
    
    return pgm;
}

void GridFastProcessor::updateTransforms(const cv::Size& map_size) {
    transform_maps_x_.clear();
    transform_maps_y_.clear();
    
    int map_size_x = map_size.width;
    int map_size_y = map_size.height;
    
    for (int angle_idx = 0; angle_idx < num_directions_; ++angle_idx) {
        double rotation = M_PI * angle_idx / num_directions_;
        double cos_a = std::cos(rotation);
        double sin_a = std::sin(rotation);
        
        // Calculate rotated bounds
        std::vector<cv::Point2d> corners = {
            {0, 0}, {map_size_x - 1, 0},
            {0, map_size_y - 1}, {map_size_x - 1, map_size_y - 1}
        };
        
        double center_x = map_size_x / 2.0;
        double center_y = map_size_y / 2.0;
        
        double x_min = 1e9, y_min = 1e9, x_max = -1e9, y_max = -1e9;
        
        for (const auto& corner : corners) {
            double cx = corner.x - center_x;
            double cy = corner.y - center_y;
            double rx = cx * cos_a - cy * sin_a + center_x;
            double ry = cx * sin_a + cy * cos_a + center_y;
            x_min = std::min(x_min, rx);
            y_min = std::min(y_min, ry);
            x_max = std::max(x_max, rx);
            y_max = std::max(y_max, ry);
        }
        
        int x_size = static_cast<int>(x_max - x_min + 1);
        int y_size = static_cast<int>(y_max - y_min + 1);
        
        // Create transform maps
        cv::Mat transform_x(y_size, x_size, CV_32S);
        cv::Mat transform_y(y_size, x_size, CV_32S);
        
        double cos_inv = std::cos(-rotation);
        double sin_inv = std::sin(-rotation);
        
        for (int ty = 0; ty < y_size; ++ty) {
            for (int tx = 0; tx < x_size; ++tx) {
                double xx_centered = tx - x_size / 2.0;
                double yy_centered = ty - y_size / 2.0;
                
                double orig_x = xx_centered * cos_inv - yy_centered * sin_inv + center_x;
                double orig_y = xx_centered * sin_inv + yy_centered * cos_inv + center_y;
                
                int ox = static_cast<int>(std::round(orig_x));
                int oy = static_cast<int>(std::round(orig_y));
                
                transform_x.at<int32_t>(ty, tx) = ox;
                transform_y.at<int32_t>(ty, tx) = oy;
            }
        }
        
        transform_maps_x_.push_back(transform_x);
        transform_maps_y_.push_back(transform_y);
    }
}

void GridFastProcessor::analyzeGaps(const cv::Mat& map_data, cv::Mat& filtered_map) {
    // For each rotation direction
    for (int angle_idx = 0; angle_idx < num_directions_; ++angle_idx) {
        if (angle_idx >= transform_maps_x_.size()) break;
        
        const cv::Mat& trans_x = transform_maps_x_[angle_idx];
        const cv::Mat& trans_y = transform_maps_y_[angle_idx];
        
        // For each row in the rotated view
        for (int row = 0; row < trans_x.rows; ++row) {
            std::vector<std::pair<int, int>> gaps;  // (start, end)
            int group_start = -1;
            int group_size = 0;
            int cfilter = 0;
            
            for (int col = 0; col < trans_x.cols; ++col) {
                int ox = trans_x.at<int32_t>(row, col);
                int oy = trans_y.at<int32_t>(row, col);
                
                // Check bounds
                if (ox < 0 || ox >= map_data.cols || oy < 0 || oy >= map_data.rows) {
                    if (group_size > 0 && group_size < min_group_size_) {
                        gaps.push_back({group_start, col - 1});
                    }
                    group_size = 0;
                    continue;
                }
                
                int32_t map_value = map_data.at<int32_t>(oy, ox);
                
                if (map_value == MAP_UNOCCUPIED && col != trans_x.cols - 1) {
                    if (group_size == 0) group_start = col;
                    group_size++;
                    cfilter = 0;
                } else if (map_value == MAP_UNKNOWN && cfilter < cfilter_size_ && 
                           group_size > 0 && col != trans_x.cols - 1) {
                    cfilter++;
                } else if (group_size > 0) {
                    int end_col = col - 1 - cfilter;
                    if (end_col - group_start >= min_group_size_) {
                        gaps.push_back({group_start, end_col});
                    } else {
                        // Fill small gap
                        for (int x = group_start; x <= end_col; ++x) {
                            int fx = trans_x.at<int32_t>(row, x);
                            int fy = trans_y.at<int32_t>(row, x);
                            if (fx >= 0 && fx < filtered_map.cols && 
                                fy >= 0 && fy < filtered_map.rows) {
                                filtered_map.at<int32_t>(fy, fx) = MAP_OCCUPIED;
                            }
                        }
                    }
                    cfilter = 0;
                    group_size = 0;
                }
            }
        }
    }
}

cv::Mat GridFastProcessor::removeSmallObstacles(const cv::Mat& map_data) {
    cv::Mat obstacle_mask = cv::Mat::zeros(map_data.size(), CV_8UC1);
    
    // Create binary mask of occupied cells
    for (int y = 0; y < map_data.rows; ++y) {
        for (int x = 0; x < map_data.cols; ++x) {
            if (map_data.at<int32_t>(y, x) == MAP_OCCUPIED) {
                obstacle_mask.at<uint8_t>(y, x) = 255;
            }
        }
    }
    
    // Connected components
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(
        obstacle_mask, labels, stats, centroids, 8);
    
    cv::Mat result = map_data.clone();
    
    // Remove small obstacles
    for (int label = 1; label < num_labels; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area <= obstacle_threshold_) {
            // Fill this component with unoccupied
            for (int y = 0; y < labels.rows; ++y) {
                for (int x = 0; x < labels.cols; ++x) {
                    if (labels.at<int>(y, x) == label) {
                        result.at<int32_t>(y, x) = MAP_UNOCCUPIED;
                    }
                }
            }
        }
    }
    
    return result;
}

std::vector<Opening> GridFastProcessor::detectOpenings(const cv::Mat& map_data) {
    std::vector<Opening> openings;
    
    // Simple opening detection: find narrow passages between occupied regions
    for (int y = 1; y < map_data.rows - 1; ++y) {
        for (int x = 1; x < map_data.cols - 1; ++x) {
            if (map_data.at<int32_t>(y, x) != MAP_UNOCCUPIED) continue;
            
            // Check if this is a narrow passage
            int occupied_neighbors = 0;
            std::vector<cv::Point> occupied_positions;
            
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (map_data.at<int32_t>(y + dy, x + dx) == MAP_OCCUPIED) {
                        occupied_neighbors++;
                        occupied_positions.push_back({x + dx, y + dy});
                    }
                }
            }
            
            // If surrounded by obstacles on 2+ sides, it's a potential opening
            if (occupied_neighbors >= 2 && occupied_neighbors <= 4) {
                if (occupied_positions.size() >= 2) {
                    Opening opening;
                    opening.start = occupied_positions[0];
                    opening.end = occupied_positions[occupied_positions.size() - 1];
                    opening.label = 1;
                    openings.push_back(opening);
                }
            }
        }
    }
    
    // Remove duplicate openings
    std::sort(openings.begin(), openings.end(),
        [](const Opening& a, const Opening& b) {
            auto key_a = std::make_tuple(a.start.x, a.start.y, a.end.x, a.end.y);
            auto key_b = std::make_tuple(b.start.x, b.start.y, b.end.x, b.end.y);
            return key_a < key_b;
        });
    
    auto last = std::unique(openings.begin(), openings.end(),
        [](const Opening& a, const Opening& b) {
            return a.start.x == b.start.x && a.start.y == b.start.y &&
                   a.end.x == b.end.x && a.end.y == b.end.y;
        });
    
    openings.erase(last, openings.end());
    
    return openings;
}

} // namespace topological_graph
