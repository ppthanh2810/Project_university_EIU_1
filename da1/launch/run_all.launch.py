from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('da1')
    joy_config = os.path.join(pkg_share, 'config', 'config.yaml')

    rs_launch = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    return LaunchDescription([

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rs_launch),
            launch_arguments={
                'align_depth.enable': 'true'
            }.items()
        ),

        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',
                'deadzone': 0.08,
                'autorepeat_rate': 15.0
            }]
        ),

        Node(
            package='joy_teleop',
            executable='joy_teleop',
            name='joy_teleop',
            parameters=[joy_config],
        ),

        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='da1',
                    executable='detect',
                    name='yolo_person_detector_onnx',
                    parameters=[{
                        'image_topic': '/camera/camera/color/image_raw',
                        'model': '/home/yolov8n.onnx',
                        'conf': 0.45,
                        'iou': 0.45,
                        'publish_debug_image': True,
                        'show_window': True,
                        'window_name': 'YOLOv8 ONNX Person Debug',
                        'providers': ['CPUExecutionProvider'],
                        'intra_op_num_threads': 2,
                        'inter_op_num_threads': 1,
                        'graph_optimization': 'basic',
                        'sequential_execution': True,
                        'topk': 200,
                        'min_box_size': 4.0,
                    }],
                    output='screen',
                )
            ]
        ),

        TimerAction(
            period=3.5,
            actions=[
                Node(
                    package='da1',
                    executable='algorithm',
                    name='bbox_percent_node',
                    parameters=[{
                        'image_topic': '/camera/camera/color/image_raw',
                        'detections_topic': '/person_detections',
                        'percent_topic': '/PT/percent',
                        'no_person_percent': 0.0,
                        'queue_size': 5,
                        'slop': 0.15,
                    }],
                    # output='screen',
                )
            ]
        ),

        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='da1',
                    executable='m_to_m',
                    name='m_to_m',
                    # output='screen',
                ),

                Node(
                    package='da1',
                    executable='move',
                    name='move',
                    # output='screen',
                ),
            ]
        ),
    ])