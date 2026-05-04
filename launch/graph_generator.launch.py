
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """
    Launch the skeleton graph generator node.
    """

    config_file = PathJoinSubstitution([
        FindPackageShare('graph_generator_node'),
        'config',
        'graph_generator_params.yaml'
    ])

    return LaunchDescription([

        Node(
            package='graph_generator_node',
            executable='graph_generator_node_main',
            name='graph_generator_node',
            output='screen',
            parameters=[config_file],
        )
    ])
