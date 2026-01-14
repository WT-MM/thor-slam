from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

import launch


def generate_launch_description():
    num_cameras = LaunchConfiguration('num_cameras')
    rectified_images = LaunchConfiguration('rectified_images')
    enable_imu_fusion = LaunchConfiguration('enable_imu_fusion')
    imu_frame = LaunchConfiguration('imu_frame')
    enable_slam_visualization = LaunchConfiguration('enable_slam_visualization')
    enable_landmarks_view = LaunchConfiguration('enable_landmarks_view')
    enable_observations_view = LaunchConfiguration('enable_observations_view')
    enable_localization_n_mapping = LaunchConfiguration('enable_localization_n_mapping')
    multicam_mode = LaunchConfiguration('multicam_mode')
    enable_debug_mode = LaunchConfiguration('enable_debug_mode')
    image_jitter_threshold_ms = LaunchConfiguration('image_jitter_threshold_ms')
    image_sync_threshold_ms = LaunchConfiguration('image_sync_threshold_ms')
    image_buffer_size = LaunchConfiguration('image_buffer_size')
    debug_imu_mode = LaunchConfiguration('debug_imu_mode')
    verbosity = LaunchConfiguration('verbosity')
    gyroscope_noise_density = LaunchConfiguration('gyroscope_noise_density')
    accelerometer_noise_density = LaunchConfiguration('accelerometer_noise_density')
    gyroscope_random_walk = LaunchConfiguration('gyroscope_random_walk')
    accelerometer_random_walk = LaunchConfiguration('accelerometer_random_walk')
    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'num_cameras': num_cameras,
            'rectified_images': rectified_images,
            'enable_imu_fusion': enable_imu_fusion,
            'imu_frame': imu_frame,
            'enable_slam_visualization': enable_slam_visualization,
            'enable_landmarks_view': enable_landmarks_view,
            'enable_observations_view': enable_observations_view,
            'enable_localization_n_mapping': enable_localization_n_mapping,
            'enable_debug_mode': enable_debug_mode,
            'image_jitter_threshold_ms': image_jitter_threshold_ms,
            'image_sync_threshold_ms': image_sync_threshold_ms,
            'image_buffer_size': image_buffer_size,
            'debug_imu_mode': debug_imu_mode,
            'verbosity': verbosity,
            'multicam_mode': multicam_mode,
            'gyroscope_noise_density': gyroscope_noise_density,
            'accelerometer_noise_density': accelerometer_noise_density,
            'gyroscope_random_walk': gyroscope_random_walk,
            'accelerometer_random_walk': accelerometer_random_walk,
        }],
    )

    container = ComposableNodeContainer(
        name='visual_slam_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
    )

    return launch.LaunchDescription([
        DeclareLaunchArgument('num_cameras', default_value='2'),
        DeclareLaunchArgument('rectified_images', default_value='True'),
        DeclareLaunchArgument('enable_imu_fusion', default_value='True'),
        DeclareLaunchArgument('imu_frame', default_value='imu_link'),
        DeclareLaunchArgument('enable_slam_visualization', default_value='True'),
        DeclareLaunchArgument('enable_landmarks_view', default_value='True'),
        DeclareLaunchArgument('enable_observations_view', default_value='True'),
        DeclareLaunchArgument('enable_localization_n_mapping', default_value='True'),
        DeclareLaunchArgument('enable_debug_mode', default_value='True'),
        DeclareLaunchArgument('image_jitter_threshold_ms', default_value='60.0'),
        DeclareLaunchArgument('image_sync_threshold_ms', default_value='100.0'),
        DeclareLaunchArgument('image_buffer_size', default_value='10'),
        DeclareLaunchArgument('debug_imu_mode', default_value='False'),
        DeclareLaunchArgument('verbosity', default_value='1'),
        DeclareLaunchArgument('multicam_mode', default_value='1'),
        DeclareLaunchArgument('gyroscope_noise_density', default_value='8.27e-5', description='Gyroscope noise density (rad/s/√Hz)'),
        DeclareLaunchArgument('accelerometer_noise_density', default_value='2.553e-3', description='Accelerometer noise density (m/s²/√Hz)'),
        DeclareLaunchArgument('gyroscope_random_walk', default_value='0.00000001', description='Gyroscope random walk (rad/s²/√Hz)'),
        DeclareLaunchArgument('accelerometer_random_walk', default_value='0.00010493', description='Accelerometer random walk (m/s³/√Hz)'),
        container,
    ])
"""
Datasheet values for BNO085:
        DeclareLaunchArgument('gyroscope_noise_density', default_value='2.44e-4', description='Gyroscope noise density (rad/s/√Hz)'),
        DeclareLaunchArgument('accelerometer_noise_density', default_value='1.47e-3', description='Accelerometer noise density (m/s²/√Hz)'),
"""

"""
Values from 2.5 hr rosbag:
fs_used: 200.0 Hz
gyroscope_noise_density:      8.272408664902705e-05
gyroscope_random_walk:        None
accelerometer_noise_density:  0.0025532489945649236
accelerometer_random_walk:    0.00010493289270471003
"""

"""
Defaults from isaac ros visual slam node:
gyroscope_noise_density: 0.000244
[component_container-1] [INFO] [1768343753.129192437] [visual_slam_node]: gyroscope_random_walk: 0.000019
[component_container-1] [INFO] [1768343753.129198270] [visual_slam_node]: accelerometer_noise_density: 0.001862
[component_container-1] [INFO] [1768343753.129204836] [visual_slam_node]: accelerometer_random_walk: 0.003000

"""
