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
    enable_debug_mode = LaunchConfiguration('enable_debug_mode')

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
        container
    ])
