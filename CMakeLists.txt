cmake_minimum_required(VERSION 3.10)
project(topological_graph_node)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -fsanitize=address")

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/grid_fast_processor.cpp
    src/skeleton_graph_builder.cpp
    src/graph_serializer.cpp
    src/topological_graph_node.cpp
)

# Header files
set(HEADERS
    include/topological_graph_node/grid_fast_processor.h
    include/topological_graph_node/skeleton_graph_builder.h
    include/topological_graph_node/graph_serializer.h
    include/topological_graph_node/graph_types.h
    include/topological_graph_node/topological_graph_node.h
)

# Create library
add_library(topological_graph_lib STATIC ${SOURCES})
target_link_libraries(topological_graph_lib ${OpenCV_LIBRARIES})
target_include_directories(topological_graph_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Main executable
add_executable(topological_graph_node src/topological_graph_node.cpp)
target_link_libraries(topological_graph_node topological_graph_lib ${OpenCV_LIBRARIES})


# Compiler flags
if(MSVC)
    target_compile_options(topological_graph_lib PRIVATE /W4)
else()
    target_compile_options(topological_graph_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install targets
install(TARGETS topological_graph_node DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include/topological_graph_node)
