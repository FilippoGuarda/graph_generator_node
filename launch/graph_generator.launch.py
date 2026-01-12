from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Launch the skeleton graph generator node.
    
    Arguments:
      shared_costmap_topic: Input topic from GlobalCostmapFusion (default: /shared_obstacles)
      global_frame: TF frame ID (default: map)
      obstacle_size_threshold: Min obstacle size to keep in cells (default: 20)
      max_bfs_steps: Max BFS propagation steps (default: 100)
      find_entrances: Create entrance nodes at budget boundaries (default: true)
    """
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'shared_costmap_topic',
            default_value='/robot1/global_costmap/costmap',
            description='Input occupancy grid topic from GlobalCostmapFusion'
        ),
        DeclareLaunchArgument(
            'global_frame',
            default_value='map',
            description='Global reference frame ID'
        ),
        DeclareLaunchArgument(
            'obstacle_size_threshold',
            default_value='2',
            description='Minimum obstacle size (cells) to preserve'
        ),
        DeclareLaunchArgument(
            'max_bfs_steps',
            default_value='100',
            description='Maximum BFS propagation budget for graph building'
        ),
        DeclareLaunchArgument(
            'find_entrances',
            default_value='true',
            description='Enable entrance node creation at budget boundaries'
        ),
        
        Node(
            package='graph_generator_node',
            executable='graph_generator_node_main',
            name='graph_generator_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {
                    'shared_costmap_topic': LaunchConfiguration('shared_costmap_topic'),
                    'global_frame': LaunchConfiguration('global_frame'),
                    'obstacle_size_threshold': LaunchConfiguration('obstacle_size_threshold'),
                    'max_bfs_steps': LaunchConfiguration('max_bfs_steps'),
                    'find_entrances': LaunchConfiguration('find_entrances'),
                }
            ],
            remappings=[
                # Override input topic if needed
                ('/shared_obstacles', LaunchConfiguration('shared_costmap_topic')),
            ]
        ),
    ])
