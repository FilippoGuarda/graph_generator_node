cmake_minimum_required(VERSION 3.16)
project(graph_generator_node)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Shared library for SkeletonGraphBuilder (core algorithm, no ROS dependencies)
add_library(skeleton_graph_builder SHARED
  src/skeleton_graph_builder.cpp
)
target_include_directories(skeleton_graph_builder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(skeleton_graph_builder
  ${OpenCV_LIBRARIES}
)

# Shared library for the ROS2 component
add_library(graph_generator_node_lib SHARED
  src/graph_generator_node.cpp
)
target_include_directories(graph_generator_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(graph_generator_node_lib
  skeleton_graph_builder
)
ament_target_dependencies(graph_generator_node_lib
  rclcpp
  rclcpp_components
  nav_msgs
  visualization_msgs
  geometry_msgs
  OpenCV
)

# Register ROS2 component and create executable
rclcpp_components_register_node(graph_generator_node_lib
  PLUGIN "graph_generator_node::GraphGeneratorNode"
  EXECUTABLE graph_generator_node_main
)

# Installation
install(TARGETS
  skeleton_graph_builder
  graph_generator_node_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  graph_generator_node_main
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY
  include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(skeleton_graph_builder)

ament_package()
