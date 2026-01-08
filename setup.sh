#!/bin/bash

# Quick setup script for graph_generator_node package
# Run this script from your ROS2 workspace root

set -e

PACKAGE_NAME="graph_generator_node"
WS_ROOT=$(pwd)

echo "Setting up $PACKAGE_NAME package..."
echo "Workspace: $WS_ROOT"

# Create directory structure
mkdir -p "$WS_ROOT/src/$PACKAGE_NAME/include/$PACKAGE_NAME"
mkdir -p "$WS_ROOT/src/$PACKAGE_NAME/src"
mkdir -p "$WS_ROOT/src/$PACKAGE_NAME/launch"
mkdir -p "$WS_ROOT/src/$PACKAGE_NAME/config"

echo "✓ Directory structure created"

# Build
echo ""
echo "Building package (this may take 1-2 minutes)..."
cd "$WS_ROOT"
colcon build --packages-select "$PACKAGE_NAME" --cmake-args -DCMAKE_BUILD_TYPE=Release

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "Next steps:"
    echo "1. Source the workspace:"
    echo "   source $WS_ROOT/install/setup.bash"
    echo ""
    echo "2. Run the node:"
    echo "   ros2 run $PACKAGE_NAME graph_generator_node"
    echo ""
    echo "   Or with custom parameters:"
    echo "   ros2 run $PACKAGE_NAME graph_generator_node --ros-args -p max_bfs_steps:=150"
    echo ""
    echo "3. View in RViz:"
    echo "   ros2 run rviz2 rviz2"
    echo "   Add MarkerArray display for /skeleton_graph/graph_markers"
    echo ""
else
    echo "✗ Build failed. Check CMake output above."
    exit 1
fi
