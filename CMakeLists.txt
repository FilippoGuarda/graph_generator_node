cmake_minimum_required(VERSION 3.10)
project(topological_graph)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -fsanitize=address")

# Dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

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
    include/topological_graph/grid_fast_processor.h
    include/topological_graph/skeleton_graph_builder.h
    include/topological_graph/graph_serializer.h
    include/topological_graph/graph_types.h
    include/topological_graph/topological_graph_node.h
)

# Create library
add_library(topological_graph_lib STATIC ${SOURCES})
target_link_libraries(topological_graph_lib ${OpenCV_LIBRARIES})
target_include_directories(topological_graph_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Main executable
add_executable(topological_graph_node src/topological_graph_node.cpp)
target_link_libraries(topological_graph_node topological_graph_lib ${OpenCV_LIBRARIES})

# Unit tests
enable_testing()

# Test 1: GridFastProcessor
add_executable(test_grid_fast test/test_grid_fast.cpp)
target_link_libraries(test_grid_fast topological_graph_lib ${OpenCV_LIBRARIES})
add_test(NAME GridFastProcessorTest COMMAND test_grid_fast)

# Test 2: SkeletonGraphBuilder
add_executable(test_skeleton test/test_skeleton.cpp)
target_link_libraries(test_skeleton topological_graph_lib ${OpenCV_LIBRARIES})
add_test(NAME SkeletonGraphBuilderTest COMMAND test_skeleton)

# Test 3: GraphSerializer
add_executable(test_serializer test/test_serializer.cpp)
target_link_libraries(test_serializer topological_graph_lib ${OpenCV_LIBRARIES})
add_test(NAME GraphSerializerTest COMMAND test_serializer)

# Integration test
add_executable(test_integration test/test_integration.cpp)
target_link_libraries(test_integration topological_graph_lib ${OpenCV_LIBRARIES})
add_test(NAME IntegrationTest COMMAND test_integration)

# Compiler flags
if(MSVC)
    target_compile_options(topological_graph_lib PRIVATE /W4)
else()
    target_compile_options(topological_graph_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install targets
install(TARGETS topological_graph_node DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include/topological_graph)
