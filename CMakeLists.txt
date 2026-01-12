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

# Shared library for the component (NO main() here)
add_library(graph_generator_node_lib SHARED
  src/graph_generator_node.cpp
)
target_include_directories(graph_generator_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(graph_generator_node_lib
  rclcpp
  rclcpp_components
  nav_msgs
  visualization_msgs
  geometry_msgs
  OpenCV
)

# Register component, auto-create executable graph_generator_node_main
rclcpp_components_register_node(graph_generator_node_lib
  PLUGIN "graph_generator_node::GraphGeneratorNode"
  EXECUTABLE graph_generator_node_main
)

# INSTALL shared library
install(TARGETS
  graph_generator_node_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# INSTALL component executable
install(TARGETS
  graph_generator_node_main
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch/config/headers
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

ament_package()
