from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    pkg_share = get_package_share_directory('move_ps4')
    config = os.path.join(pkg_share, 'config', 'params_joy.yaml')
    
    return LaunchDescription([
        
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',
                'deadzone': 0.05,
                'autorepeat_rate': 20.0
            }]
        ),
        
        Node(
            package='joy_teleop',
            executable='joy_teleop',
            name='joy_teleop',
            parameters=[config],
        ),

        Node(
            package='move_ps4',
            executable='move',
            name='move',
        ),

        Node(
            package='move_ps4',
            executable='m_to_m',
            name='m_to_m',
        ),
    ])
