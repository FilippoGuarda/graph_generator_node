#include <rclcpp/rclcpp.hpp>
#include "graph_generator_node/graph_generator_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<multi_robot_topology::GraphGeneratorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}