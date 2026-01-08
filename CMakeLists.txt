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

# Library for graph generator node
add_library(graph_generator_node_lib src/graph_generator_node.cpp)
target_include_directories(graph_generator_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(graph_generator_node_lib
  rclcpp
  rclcpp_components
  nav_msgs
  visualization_msgs
  geometry_msgs
  OpenCV)

# Component registration with executable
rclcpp_components_register_node(graph_generator_node_lib
  PLUGIN "multi_robot_topology::GraphGeneratorNode"
  EXECUTABLE graph_generator_node)

# Installation of library
install(TARGETS
  graph_generator_node_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Installation of executable
install(TARGETS
  graph_generator_node
  DESTINATION lib/${PROJECT_NAME})

# Installation of launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch)

# Installation of config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config)

# Installation of headers
install(DIRECTORY
  include/
  DESTINATION include)

ament_export_include_directories(include)

ament_package()