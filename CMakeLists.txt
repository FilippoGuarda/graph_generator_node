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
find_package(nav2_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Source files
set(SOURCES
    src/grid_fast_processor.cpp
    src/skeleton_graph_builder.cpp
    src/graph_serializer.cpp
    src/topological_graph_node.cpp
)


add_library(topological_graph_lib STATIC ${SOURCES})

target_link_libraries(topological_graph_lib ${OpenCV_LIBRARIES})

target_include_directories(topological_graph_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(topological_graph_lib PUBLIC ${OpenCV_INCLUDE_DIRS})


ament_target_dependencies(topological_graph_lib
  rclcpp
  rclcpp_components
  nav2_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
)


add_executable(topological_graph_node src/topological_graph_node.cpp)

target_link_libraries(topological_graph_node topological_graph_lib ${OpenCV_LIBRARIES})

ament_target_dependencies(topological_graph_node
  rclcpp
  rclcpp_components
  nav_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
)

# Compiler flags
if(MSVC)
    target_compile_options(topological_graph_lib PRIVATE /W4)
else()
    target_compile_options(topological_graph_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install targets
install(TARGETS topological_graph_node topological_graph_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/ DESTINATION include)

# --- 3. Ament Package ---
ament_package()
