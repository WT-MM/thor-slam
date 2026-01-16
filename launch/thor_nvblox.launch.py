"""Launch file for nvblox with RGB-D cameras.

This launch file starts nvblox with support for multiple RGB-D cameras.
It expects topics in the format:
  /camera_0/rgb/image_raw, /camera_0/depth/image_raw
  /camera_1/rgb/image_raw, /camera_1/depth/image_raw
  etc.

Note: This assumes nvblox_ros is installed. For installation instructions,
run 'make nvblox-install'.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for nvblox."""
    num_cameras_arg = DeclareLaunchArgument("num_cameras", default_value="1", description="Number of RGB-D cameras")
    map_frame_arg = DeclareLaunchArgument("map_frame", default_value="map", description="Map frame name")
    global_frame_arg = DeclareLaunchArgument("global_frame", default_value="odom", description="Global frame name")
    voxel_size_arg = DeclareLaunchArgument("voxel_size", default_value="0.05", description="Voxel size in meters")
    tsdf_integrator_max_integration_distance_m_arg = DeclareLaunchArgument(
        "tsdf_integrator_max_integration_distance_m",
        default_value="10.0",
        description="Maximum integration distance for TSDF in meters",
    )
    tsdf_integrator_truncation_distance_vox_arg = DeclareLaunchArgument(
        "tsdf_integrator_truncation_distance_vox",
        default_value="4.0",
        description="Truncation distance in voxels",
    )

    num_cameras = LaunchConfiguration("num_cameras")
    map_frame = LaunchConfiguration("map_frame")
    global_frame = LaunchConfiguration("global_frame")
    voxel_size = LaunchConfiguration("voxel_size")
    tsdf_integrator_max_integration_distance_m = LaunchConfiguration(
        "tsdf_integrator_max_integration_distance_m"
    )
    tsdf_integrator_truncation_distance_vox = LaunchConfiguration(
        "tsdf_integrator_truncation_distance_vox"
    )

    # Build topic remappings for nvblox
    # nvblox expects: camera_N/color/image, camera_N/depth/image, camera_N/color/camera_info, camera_N/depth/camera_info
    # We publish: /camera_N/rgb/image_raw, /camera_N/depth/image_raw, etc.
    # Note: nvblox uses "color" not "rgb" in topic names
    remappings = []
    # For now, support single camera (camera_0)
    # Can be extended to support multiple cameras
    remappings.append(("camera_0/color/image", "/camera_0/rgb/image_raw"))
    remappings.append(("camera_0/depth/image", "/camera_0/depth/image_raw"))
    remappings.append(("camera_0/color/camera_info", "/camera_0/rgb/camera_info"))
    remappings.append(("camera_0/depth/camera_info", "/camera_0/depth/camera_info"))

    # nvblox node as ComposableNode (Isaac ROS style)
    nvblox_node = ComposableNode(
        name="nvblox_node",
        package="nvblox_ros",
        plugin="nvblox::NvbloxNode",
        remappings=remappings,
        parameters=[
            {
                # Basic configuration
                "map_frame": map_frame,
                "global_frame": global_frame,
                "voxel_size": voxel_size,
                # TSDF integrator settings
                "tsdf_integrator_max_integration_distance_m": tsdf_integrator_max_integration_distance_m,
                "tsdf_integrator_truncation_distance_vox": tsdf_integrator_truncation_distance_vox,
                # Camera configuration
                "num_cameras": num_cameras,
                "use_lidar": False,
            }
        ],
    )

    # Container for composable nodes
    container = ComposableNodeContainer(
        name="nvblox_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[nvblox_node],
        output="screen",
    )

    return LaunchDescription(
        [
            num_cameras_arg,
            map_frame_arg,
            global_frame_arg,
            voxel_size_arg,
            tsdf_integrator_max_integration_distance_m_arg,
            tsdf_integrator_truncation_distance_vox_arg,
            container,
        ]
    )

