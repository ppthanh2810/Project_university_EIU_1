import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ---------- Paths ----------
    pkg_share = get_package_share_directory('da1')
    joy_teleop_yaml = os.path.join(pkg_share, 'config', 'config.yaml')

    # ---------- Launch args ----------
    align_depth   = LaunchConfiguration('align_depth')
    image_topic   = LaunchConfiguration('image_topic')

    start_delay   = LaunchConfiguration('start_delay')     # delay trước detect
    algo_delay    = LaunchConfiguration('algo_delay')      # delay trước algorithm
    control_delay = LaunchConfiguration('control_delay')   # delay trước m_to_m + move

    # detect params
    model         = LaunchConfiguration('model')
    conf          = LaunchConfiguration('conf')
    device        = LaunchConfiguration('device')
    publish_debug = LaunchConfiguration('publish_debug_image')
    show_window   = LaunchConfiguration('show_window')

    # joy params
    joy_dev       = LaunchConfiguration('joy_dev')
    deadzone      = LaunchConfiguration('deadzone')
    autorepeat    = LaunchConfiguration('autorepeat_rate')

    # venv python
    venv_python   = LaunchConfiguration('venv_python')

    # ---------- 1) RealSense ----------
    realsense_rs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            # realsense v4.x
            'align_depth.enable': align_depth,
        }.items()
    )

    # ---------- 2) Detect (da1.detect) ----------
    detect_node = Node(
        executable=venv_python,
        arguments=['-m', 'da1.detect'],  # <-- đổi nếu module khác
        name='yolo_person_detector',
        output='screen',
        emulate_tty=True,
        additional_env={
            'PYTHONNOUSERSITE': '1',
            # nếu bật show_window thì cần DISPLAY
            'DISPLAY': os.environ.get('DISPLAY', ''),
            'XAUTHORITY': os.environ.get('XAUTHORITY', ''),
        },
        parameters=[{
            'image_topic': image_topic,
            'model': model,
            'conf': conf,
            'device': device,
            'publish_debug_image': publish_debug,
            'show_window': show_window,
        }],
        respawn=True,
        respawn_delay=2.0,
    )
    detect_delayed = TimerAction(period=start_delay, actions=[detect_node])

    # ---------- 3) Algorithm (da1.algorithm) ----------
    algo_node = Node(
        executable=venv_python,
        arguments=['-m', 'da1.algorithm'],  # <-- đổi nếu module khác
        name='bbox_percent_node',
        output='screen',
        emulate_tty=True,
        additional_env={'PYTHONNOUSERSITE': '1'},
        parameters=[{
            'image_topic': image_topic,
            'detections_topic': '/person_detections',
            'percent_topic': '/PT/percent',
            'no_person_percent': 0.0,
            'queue_size': 10,
            'slop': 0.10,
        }],
        respawn=True,
        respawn_delay=2.0,
    )
    algo_delayed_action = TimerAction(period=algo_delay, actions=[algo_node])

    # ---------- 4) Joy + teleop ----------
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{
            'dev': joy_dev,
            'deadzone': deadzone,
            'autorepeat_rate': autorepeat,
        }]
    )

    joy_teleop_node = Node(
        package='joy_teleop',
        executable='joy_teleop',   # đúng theo máy bạn (ros2 pkg executables joy_teleop)
        name='joy_teleop',
        parameters=[joy_teleop_yaml],
    )

    # ---------- 5) Control nodes (m_to_m + move) ----------
    m_to_m_node = Node(
        package='da1',
        executable='m_to_m',
        name='m_to_m',
        output='screen',
        # respawn=True, respawn_delay=2.0,  # nếu bạn muốn tự hồi sinh khi crash thì mở
    )

    move_node = Node(
        package='da1',
        executable='move',
        name='move',
        output='screen',
        # respawn=True, respawn_delay=2.0,
    )

    control_delayed_action = TimerAction(
        period=control_delay,
        actions=[m_to_m_node, move_node]
    )

    # ---------- LaunchDescription ----------
    return LaunchDescription([
        # --- Realsense ---
        DeclareLaunchArgument('align_depth', default_value='true',
                              description='Align depth to color (true/false)'),
        DeclareLaunchArgument('image_topic', default_value='/camera/camera/color/image_raw',
                              description='RealSense color image topic'),

        # --- Delays (để hệ chạy mượt, tránh timeout lúc khởi động) ---
        DeclareLaunchArgument('start_delay', default_value='1.0',
                              description='Seconds to wait before starting detect'),
        DeclareLaunchArgument('algo_delay', default_value='2.0',
                              description='Seconds to wait before starting algorithm'),
        DeclareLaunchArgument('control_delay', default_value='2.5',
                              description='Seconds to wait before starting m_to_m + move'),

        # --- Detect params ---
        DeclareLaunchArgument('model', default_value='yolov8n.pt'),
        DeclareLaunchArgument('conf', default_value='0.4'),
        DeclareLaunchArgument('device', default_value='cpu'),
        # Mượt nhất: tắt debug image + tắt window
        DeclareLaunchArgument('publish_debug_image', default_value='false'),
        DeclareLaunchArgument('show_window', default_value='false'),

        # --- Joy params ---
        DeclareLaunchArgument('joy_dev', default_value='/dev/input/js0'),
        DeclareLaunchArgument('deadzone', default_value='0.05'),
        DeclareLaunchArgument('autorepeat_rate', default_value='20.0'),

        # --- venv python ---
        DeclareLaunchArgument(
            'venv_python',
            default_value=os.path.expanduser('~/ros_ws/.venv_ros/bin/python3'),
            description='Python executable inside your runtime venv'
        ),

        # --- Start ---
        joy_node,
        joy_teleop_node,

        realsense_rs_launch,
        detect_delayed,
        algo_delayed_action,
        control_delayed_action,
    ])
