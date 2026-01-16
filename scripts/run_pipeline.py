"""Run SLAM with Isaac ROS Visual SLAM and publish RGB-D data for nvblox.

Publishes camera frames to:
  /visual_slam/image_0, /visual_slam/image_1, ... (for SLAM)
  /visual_slam/camera_info_0, /visual_slam/camera_info_1, ... (for SLAM)
  /camera_0/rgb/image_raw, /camera_0/depth/image_raw (for nvblox, if enable_rgbd=true)
  /camera_1/rgb/image_raw, /camera_1/depth/image_raw (for nvblox, if enable_rgbd=true)
  ...

Example:
    # Use default config
    python -m scripts.run_pipeline

    # Use custom config
    python -m scripts.run_pipeline --config /path/to/config.yaml

    # Multiple cameras can have enable_rgbd: true in config
"""

import argparse
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

import depthai as dai

from thor_slam.camera.drivers.luxonis import (
    LuxonisCameraConfig,
    LuxonisCameraSource,
    LuxonisResolution,
)
from thor_slam.camera.rig import CameraRig
from thor_slam.camera.types import CameraSensorType, Extrinsics, IMUExtrinsics
from thor_slam.camera.utils import load_rig_extrinsics_from_urdf
from thor_slam.slam import IsaacRosAdapter

# Shutdown flag
_shutdown = False

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "slam_config.yaml"

# Fixed camera mapping: IP address -> URDF link name
CAMERA_MAP = {
    "192.168.2.25": "link_Camera_1_centroid",  # Front low cam
    "192.168.2.21": "link_Camera_2_centroid",  # Right cam
    "192.168.2.23": "link_Camera_3_centroid",  # Up cam
    "192.168.2.22": "link_Camera_4_centroid",  # Left cam
}


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    ip: str
    stereo: bool
    resolution: tuple[int, int]  # (width, height)
    sensor_type: str  # "COLOR" or "MONO"
    output_resolution: tuple[int, int] | None = None  # Optional output resolution to scale to
    enable_rgbd: bool = False  # Enable RGB-D streams for this camera
    rgb_sensor_resolution: tuple[int, int] | None = None  # Optional RGB sensor resolution (CAM_A) - auto-selected if not specified
    rgb_output_resolution: tuple[int, int] | None = None  # Optional RGB output resolution for RGB-D (independent from stereo resolution)


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    cameras: list[CameraConfig]
    fps: int = 30
    display: bool = False
    urdf_path: str = ""
    imu_report_rate: int = 400
    queue_size: int = 8
    rig_queue_size: int = 30
    rgbd_camera_ip: str | None = None  # Deprecated: use nvblox_cameras instead
    nvblox_cameras: list[str] | None = None  # List of camera IPs to use for nvblox (overrides enable_rgbd)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary."""
        cameras = []
        for cam_data in data.get("cameras", []):
            resolution = cam_data.get("resolution", [1280, 800])
            output_resolution = cam_data.get("output_resolution")
            rgb_sensor_resolution = cam_data.get("rgb_sensor_resolution")
            rgb_output_resolution = cam_data.get("rgb_output_resolution")
            cameras.append(
                CameraConfig(
                    ip=cam_data["ip"],
                    stereo=cam_data.get("stereo", True),
                    resolution=(resolution[0], resolution[1]),
                    sensor_type=cam_data.get("sensor_type", "COLOR").upper(),
                    output_resolution=(
                        (output_resolution[0], output_resolution[1]) if output_resolution is not None else None
                    ),
                    enable_rgbd=cam_data.get("enable_rgbd", False),
                    rgb_sensor_resolution=(
                        (rgb_sensor_resolution[0], rgb_sensor_resolution[1]) if rgb_sensor_resolution is not None else None
                    ),
                    rgb_output_resolution=(
                        (rgb_output_resolution[0], rgb_output_resolution[1]) if rgb_output_resolution is not None else None
                    ),
                )
            )

        # Default URDF path
        urdf_path = data.get("urdf_path", "")
        if not urdf_path:
            default_urdf = Path(__file__).parent.parent / "examples" / "assets" / "brackets.urdf"
            if default_urdf.exists():
                urdf_path = str(default_urdf)

        # Get nvblox_cameras list (list of IPs)
        nvblox_cameras = data.get("nvblox_cameras")
        if nvblox_cameras is None:
            # Fall back to deprecated rgbd_camera_ip if provided
            rgbd_camera_ip = data.get("rgbd_camera_ip")
            if rgbd_camera_ip:
                nvblox_cameras = [rgbd_camera_ip]
            else:
                # If neither specified, use cameras with enable_rgbd=True
                nvblox_cameras = [cam["ip"] for cam in data.get("cameras", []) if cam.get("enable_rgbd", False)]

        return cls(
            cameras=cameras,
            fps=data.get("fps", 30),
            display=data.get("display", False),
            urdf_path=urdf_path,
            imu_report_rate=data.get("imu_report_rate", 400),
            queue_size=data.get("queue_size", 8),
            rig_queue_size=data.get("rig_queue_size", 30),
            rgbd_camera_ip=data.get("rgbd_camera_ip"),
            nvblox_cameras=nvblox_cameras if isinstance(nvblox_cameras, list) else None,
        )

    def calculate_num_cameras(self) -> int:
        """Calculate total number of camera streams (stereo=2, mono=1 per camera)."""
        return sum(2 if cam.stereo else 1 for cam in self.cameras)


class RGBDPublisher(Node):
    """ROS 2 node that publishes RGB-D data for nvblox."""

    def __init__(self, camera: LuxonisCameraSource, camera_index: int, frame_id: str | None = None):
        """Initialize the RGB-D publisher node.

        Args:
            camera: Luxonis camera source with RGB-D enabled.
            camera_index: Index of this camera (for topic namespacing).
            frame_id: Frame ID for the RGB-D messages (default: camera_{index}_optical_frame).
        """
        super().__init__(f"rgbd_publisher_{camera_index}")
        self._camera = camera
        self._camera_index = camera_index
        self._frame_id = frame_id or f"camera_{camera_index}_optical_frame"
        self._bridge = CvBridge()
        self._spin_thread: threading.Thread | None = None

        # Create namespaced publishers
        namespace = f"/camera_{camera_index}"
        self._rgb_pub = self.create_publisher(Image, f"{namespace}/rgb/image_raw", qos_profile_sensor_data)
        self._rgb_info_pub = self.create_publisher(CameraInfo, f"{namespace}/rgb/camera_info", qos_profile_sensor_data)
        self._depth_pub = self.create_publisher(Image, f"{namespace}/depth/image_raw", qos_profile_sensor_data)
        self._depth_info_pub = self.create_publisher(CameraInfo, f"{namespace}/depth/camera_info", qos_profile_sensor_data)

        # Get intrinsics
        self._rgb_intrinsics, self._depth_intrinsics = camera.get_rgbd_intrinsics()

        self.get_logger().info(f"RGB-D publisher {camera_index} initialized, frame_id: {self._frame_id}")
        self.get_logger().info(f"  Topics: {namespace}/rgb/image_raw, {namespace}/depth/image_raw")
        self.get_logger().info(f"  RGB: {self._rgb_intrinsics.width}x{self._rgb_intrinsics.height}")
        self.get_logger().info(f"  Depth: {self._depth_intrinsics.width}x{self._depth_intrinsics.height}")

        # Start spinning in a separate thread
        node = self
        self._spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
        self._spin_thread.start()

    def shutdown(self) -> None:
        """Shutdown the publisher node."""
        if self._spin_thread and self._spin_thread.is_alive():
            # Node will be destroyed when thread exits
            pass

    def publish_rgbd(self, rgb_frame, depth_frame) -> None:
        """Publish RGB and depth frames.

        Args:
            rgb_frame: RGB CameraFrame
            depth_frame: Depth CameraFrame
        """
        # Create timestamp
        stamp = Time()
        stamp.sec = int(rgb_frame.timestamp)
        stamp.nanosec = int((rgb_frame.timestamp - stamp.sec) * 1e9)

        # Publish RGB image
        rgb_img = rgb_frame.image
        if len(rgb_img.shape) == 3:
            # Convert BGR to RGB for ROS
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_msg = self._bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
        else:
            rgb_msg = self._bridge.cv2_to_imgmsg(rgb_img, encoding="mono8")

        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self._frame_id
        self._rgb_pub.publish(rgb_msg)

        # Publish RGB camera info
        rgb_info = self._create_camera_info(self._rgb_intrinsics, stamp)
        self._rgb_info_pub.publish(rgb_info)

        # Publish depth image (uint16 in millimeters)
        depth_img = depth_frame.image
        depth_msg = self._bridge.cv2_to_imgmsg(depth_img, encoding="16UC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self._frame_id
        self._depth_pub.publish(depth_msg)

        # Publish depth camera info
        depth_info = self._create_camera_info(self._depth_intrinsics, stamp)
        self._depth_info_pub.publish(depth_info)

    def _create_camera_info(self, intrinsics, stamp: Time) -> CameraInfo:
        """Create CameraInfo message from intrinsics."""
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._frame_id
        info.width = intrinsics.width
        info.height = intrinsics.height

        # Distortion coefficients
        d = intrinsics.coeffs.flatten().tolist()
        if len(d) >= 8:
            info.distortion_model = "rational_polynomial"
            info.d = d[:8]  # k1 k2 p1 p2 k3 k4 k5 k6
        elif len(d) == 5:
            info.distortion_model = "plumb_bob"
            info.d = d
        elif len(d) == 4:
            info.distortion_model = "equidistant"
            info.d = d
        else:
            info.distortion_model = "plumb_bob"
            info.d = (d + [0, 0, 0, 0, 0])[:5]

        # Intrinsic matrix K
        info.k = intrinsics.matrix.flatten().tolist()

        # Rectification matrix R (identity for unrectified)
        info.r = np.eye(3).flatten().tolist()

        # Projection matrix P (same as K for monocular)
        p = np.zeros((3, 4))
        p[:3, :3] = intrinsics.matrix
        info.p = p.flatten().tolist()

        return info


def signal_handler(sig: int, frame: object) -> None:
    """Handle shutdown signal."""
    global _shutdown
    _shutdown = True
    print("\nShutting down...")


def load_config(config_path: Path) -> PipelineConfig:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return PipelineConfig.from_dict(data)


def create_sources(config: PipelineConfig) -> tuple[dict[str, LuxonisCameraSource], str | None, dict[str, LuxonisCameraSource]]:
    """Create camera sources directly from configuration.

    Returns:
        Tuple of (sources dict, first_ip, rgbd_cameras dict mapping IP to source)
    """
    print("\nCreating camera sources...")

    sources = {}
    first_ip = config.cameras[0].ip if config.cameras else None
    rgbd_cameras: dict[str, LuxonisCameraSource] = {}

    # Determine which cameras should have RGB-D enabled for nvblox
    nvblox_camera_ips = set()
    if config.nvblox_cameras:
        nvblox_camera_ips = set(config.nvblox_cameras)
        print(f"  nvblox cameras (from config): {sorted(nvblox_camera_ips)}")
    else:
        # Fall back to cameras with enable_rgbd=True
        nvblox_camera_ips = {cam.ip for cam in config.cameras if cam.enable_rgbd}
        if nvblox_camera_ips:
            print(f"  nvblox cameras (from enable_rgbd): {sorted(nvblox_camera_ips)}")

    # Validate nvblox camera IPs exist in camera list
    all_camera_ips = {cam.ip for cam in config.cameras}
    invalid_nvblox_ips = nvblox_camera_ips - all_camera_ips
    if invalid_nvblox_ips:
        raise ValueError(
            f"nvblox_cameras contains IPs not in cameras list: {sorted(invalid_nvblox_ips)}. "
            f"Available cameras: {sorted(all_camera_ips)}"
        )

    for cam_config in config.cameras:
        # Map config fields to new resolution structure
        mono_sensor_resolution = LuxonisResolution.from_dimensions(
            cam_config.resolution[0], cam_config.resolution[1]
        )
        
        slam_output_resolution = None
        if cam_config.output_resolution is not None:
            slam_output_resolution = LuxonisResolution.from_dimensions(
                cam_config.output_resolution[0], cam_config.output_resolution[1]
            )

        # Enable RGB-D if this camera is in nvblox_cameras list and is stereo
        enable_rgbd = (cam_config.ip in nvblox_camera_ips) and cam_config.stereo

        # RGB-D resolutions (optional)
        rgb_sensor_resolution = None
        if cam_config.rgb_sensor_resolution is not None:
            rgb_sensor_resolution = LuxonisResolution.from_dimensions(
                cam_config.rgb_sensor_resolution[0], cam_config.rgb_sensor_resolution[1]
            )
        # Will auto-select if None
        
        rgb_output_resolution = None
        if cam_config.rgb_output_resolution is not None:
            rgb_output_resolution = LuxonisResolution.from_dimensions(
                cam_config.rgb_output_resolution[0], cam_config.rgb_output_resolution[1]
            )
        
        depth_input_resolution = None  # Defaults to mono_sensor_resolution
        depth_output_resolution = None  # Defaults to rgb_output_resolution when aligned

        luxonis_config = LuxonisCameraConfig(
            ip=cam_config.ip,
            stereo=cam_config.stereo,
            mono_sensor_resolution=mono_sensor_resolution,
            slam_output_resolution=slam_output_resolution,
            depth_input_resolution=depth_input_resolution,
            fps=config.fps,
            queue_size=config.queue_size,
            queue_blocking=False,
            camera_mode=cast(CameraSensorType, cam_config.sensor_type),
            read_imu=(cam_config.ip == first_ip),  # Enable IMU on first camera
            imu_report_rate=config.imu_report_rate,
            enable_rgbd=enable_rgbd,
            rgb_sensor_resolution=rgb_sensor_resolution,
            rgb_output_resolution=rgb_output_resolution,
            depth_output_resolution=depth_output_resolution,
            depth_preset=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
            depth_lr_check=True,
            depth_align_to_rgb=True,
        )

        source = LuxonisCameraSource(luxonis_config)
        sources[cam_config.ip] = source

        if enable_rgbd and source.has_rgbd_streams:
            rgbd_cameras[cam_config.ip] = source

        imu_status = " (IMU enabled)" if cam_config.ip == first_ip else ""
        stereo_status = "stereo" if cam_config.stereo else "mono"
        slam_status = " [SLAM]" if True else ""  # All cameras used for SLAM
        nvblox_status = " [nvblox]" if cam_config.ip in nvblox_camera_ips and enable_rgbd and source.has_rgbd_streams else ""
        output_info = (
            f" -> {cam_config.output_resolution[0]}x{cam_config.output_resolution[1]}"
            if cam_config.output_resolution
            else ""
        )
        print(
            (
                f"  ✓ {cam_config.ip}: "
                f"{stereo_status},"
                f"{cam_config.resolution[0]}x{cam_config.resolution[1]}"
                f"{output_info}, "
                f"{cam_config.sensor_type}"
                f"{imu_status}"
                f"{slam_status}"
                f"{nvblox_status}"
            )
        )

    return sources, first_ip, rgbd_cameras


def run(config: PipelineConfig) -> None:
    """Run SLAM + RGB-D pipeline.

    Args:
        config: Pipeline configuration.
    """
    num_cameras = config.calculate_num_cameras()

    # Create sources first to get rgbd_cameras for display
    sources, first_ip, rgbd_cameras = create_sources(config)

    print("\n" + "=" * 60)
    print("Thor SLAM Pipeline -> Isaac ROS Visual SLAM + nvblox RGB-D")
    print("=" * 60)
    print(f"Total camera streams for SLAM: {num_cameras}")
    print(f"Physical cameras: {len(config.cameras)}")
    print("\nCameras for SLAM (all cameras):")
    for i, cam in enumerate(config.cameras):
        stereo_str = "stereo" if cam.stereo else "mono"
        print(f"  Camera {i + 1}: {cam.ip} ({stereo_str}, {cam.resolution[0]}x{cam.resolution[1]}, {cam.sensor_type})")
    
    if rgbd_cameras:
        print(f"\nCameras for nvblox ({len(rgbd_cameras)} camera(s)):")
        for i, (camera_ip, _) in enumerate(sorted(rgbd_cameras.items())):
            cam = next(c for c in config.cameras if c.ip == camera_ip)
            print(f"  nvblox Camera {i}: {camera_ip} (RGB-D, {cam.resolution[0]}x{cam.resolution[1]})")
    else:
        print("\nNo cameras configured for nvblox (set nvblox_cameras in config)")
    
    print(f"\nFPS: {config.fps}")
    if config.urdf_path:
        print(f"URDF: {config.urdf_path}")
    print(f"{'=' * 60}\n")

    rig = None
    slam = None
    rgbd_publishers: dict[str, RGBDPublisher] = {}

    try:
        # Initialize ROS 2
        if not rclpy.ok():
            rclpy.init()

        assert first_ip is not None

        # Load rig extrinsics from URDF if provided
        rig_extrinsics = None
        if config.urdf_path:
            camera_ips = [cam.ip for cam in config.cameras]
            # Build camera map for requested IPs only
            cam_map = {ip: CAMERA_MAP[ip] for ip in camera_ips if ip in CAMERA_MAP}

            # Check if all requested IPs have mappings
            missing = [ip for ip in camera_ips if ip not in CAMERA_MAP]
            if missing:
                raise RuntimeError(
                    f"These IPs are not in the fixed camera mapping: {missing}\n"
                    f"Available IPs: {list(CAMERA_MAP.keys())}"
                )

            print("\nLoading rig extrinsics from URDF...")
            rig_extrinsics = load_rig_extrinsics_from_urdf(config.urdf_path, cam_map)

            # Sanity: ensure every IP produced an extrinsics
            missing_out = [ip for ip in camera_ips if ip not in rig_extrinsics]
            if missing_out:
                raise RuntimeError(f"URDF did not contain base_link -> {missing_out} link joints (check link names).")
            print(f"  ✓ Loaded extrinsics for {len(rig_extrinsics)} camera(s)")

        # Create rig
        print("\nStarting camera rig...")

        # Compute IMU extrinsics in world frame
        imu_extrinsics_world = None

        imu_source_obj = sources[first_ip]

        # Get sensor extrinsics relative to camera source (CAM_A reference frame)
        imu_to_source_extrinsics = imu_source_obj.get_sensor_extrinsics()

        # Apply coordinate transformation: OAK D Pro IMU uses DRB, camera uses RDF
        # DRB: X down, Y right, Z back
        # RDF: X right, Y down, Z forward
        # Transformation matrix: DRB -> RDF
        drb_to_rdf_matrix = np.array(
            [
                [0, 1, 0, 0],  # DRB X (down) -> RDF Y (down)
                [1, 0, 0, 0],  # DRB Y (right) -> RDF X (right)
                [0, 0, -1, 0],  # ULB Z (back) -> RDF Z (forward)
                [0, 0, 0, 1],
            ]
        )

        assert imu_to_source_extrinsics is not None

        imu_to_source_extrinsics_rdf = drb_to_rdf_matrix @ imu_to_source_extrinsics.to_4x4_matrix()

        if rig_extrinsics:
            world_to_source_matrix = rig_extrinsics.get(first_ip, Extrinsics.from_4x4_matrix(np.eye(4))).to_4x4_matrix()
            world_to_imu_matrix = world_to_source_matrix @ imu_to_source_extrinsics_rdf
            imu_extrinsics_world = Extrinsics.from_4x4_matrix(world_to_imu_matrix)
        else:
            imu_extrinsics_world = Extrinsics.from_4x4_matrix(imu_to_source_extrinsics_rdf)

        print(f"  ✓ Computed IMU extrinsics in world frame for {first_ip}")

        # Use first camera as IMU source (name is the IP address)
        imu_extrinsics_obj = None
        if imu_extrinsics_world and first_ip:
            imu_extrinsics_obj = IMUExtrinsics(source_name=first_ip, extrinsics=imu_extrinsics_world)

        rig = CameraRig(
            sources=list(sources.values()),
            queue_size=config.rig_queue_size,
            rig_extrinsics=rig_extrinsics,
            imu_source=first_ip,
            imu_extrinsics=imu_extrinsics_obj,
        )
        rig.start()
        print("  ✓ Rig started")
        if first_ip:
            print(f"  ✓ IMU source: {first_ip}")

        # Create SLAM adapter
        print("\nInitializing SLAM adapter...")
        slam = IsaacRosAdapter(num_cameras=num_cameras)
        slam.initialize(rig.calibration)
        print(f"  ✓ Publishing to /visual_slam/image_0..{num_cameras - 1}")
        print("  ✓ Subscribing to /visual_slam/tracking/odometry")

        # Create RGB-D publishers for all RGB-D cameras
        if rgbd_cameras:
            print(f"\nInitializing RGB-D publishers for nvblox ({len(rgbd_cameras)} camera(s))...")
            # Assign camera indices based on order in config
            rgbd_camera_ips = [cam.ip for cam in config.cameras if cam.ip in rgbd_cameras]
            for idx, camera_ip in enumerate(rgbd_camera_ips):
                camera = rgbd_cameras[camera_ip]
                rgbd_publisher = RGBDPublisher(camera, camera_index=idx)
                rgbd_publishers[camera_ip] = rgbd_publisher
            print(f"  ✓ Publishing RGB-D data for {len(rgbd_publishers)} camera(s)")

        # Main loop
        print(f"\n{'─' * 60}")
        print("Running. Press Ctrl+C to stop.")
        print(f"{'─' * 60}\n")

        last_print = time.time()
        frame_count = 0
        rgbd_frame_counts: dict[str, int] = {ip: 0 for ip in rgbd_cameras.keys()}
        start_time = time.time()
        actual_fps = 0.0
        rgbd_fps_dict: dict[str, float] = {ip: 0.0 for ip in rgbd_cameras.keys()}
        latest_pose = None  # Store latest pose from process_frames()

        while not _shutdown:
            # Process SLAM frames
            sync = rig.get_synchronized_frames()
            if sync is not None:
                latest_pose = slam.process_frames(sync)
                frame_count += 1

            # Process RGB-D frames for all RGB-D cameras (non-blocking)
            for camera_ip, camera in rgbd_cameras.items():
                if camera_ip in rgbd_publishers:
                    rgbd_frames = camera.try_get_latest_rgbd_frames()
                    if rgbd_frames is not None:
                        rgb_frame, depth_frame = rgbd_frames
                        rgbd_publishers[camera_ip].publish_rgbd(rgb_frame, depth_frame)
                        rgbd_frame_counts[camera_ip] += 1

            # Calculate actual FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                actual_fps = frame_count / elapsed
                for camera_ip in rgbd_cameras.keys():
                    rgbd_fps_dict[camera_ip] = rgbd_frame_counts[camera_ip] / elapsed

            # Display
            if config.display:
                if sync is not None:
                    for source_name, fs in sync.frame_sets.items():
                        for i, frame in enumerate(fs.frames):
                            img = frame.image
                            if len(img.shape) == 2:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                            # Resize for display
                            h, w = img.shape[:2]
                            if w > 640:
                                scale = 640 / w
                                img = cv2.resize(img, (640, int(h * scale)))

                            cv2.imshow(f"{source_name}_{i}", img)

                # Display RGB-D frames
                for idx, (camera_ip, camera) in enumerate(rgbd_cameras.items()):
                    rgbd_frames = camera.try_get_latest_rgbd_frames()
                    if rgbd_frames is not None:
                        rgb_frame, depth_frame = rgbd_frames
                        rgb_img = rgb_frame.image
                        depth_img = depth_frame.image

                        # Convert depth to colorized display
                        depth_colored = cv2.applyColorMap(
                            (np.clip(depth_img.astype(np.float32) / 10.0, 0, 255)).astype(np.uint8),
                            cv2.COLORMAP_JET,
                        )

                        if len(rgb_img.shape) == 2:
                            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2BGR)

                        # Resize for display
                        h, w = rgb_img.shape[:2]
                        if w > 640:
                            scale = 640 / w
                            rgb_img = cv2.resize(rgb_img, (640, int(h * scale)))
                            depth_colored = cv2.resize(depth_colored, (640, int(h * scale)))

                        cv2.imshow(f"RGB-D RGB {idx} ({camera_ip})", rgb_img)
                        cv2.imshow(f"RGB-D Depth {idx} ({camera_ip})", depth_colored)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Status
            now = time.time()
            if now - last_print >= 2.0:
                state = slam.get_tracking_state() if slam else None
                pos = "N/A"
                if latest_pose:
                    p = latest_pose.position
                    pos = f"[{p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}]"
                state_str = state.name if state else "N/A"
                rgbd_status = ""
                if rgbd_cameras:
                    fps_strs = [f"{ip}: {rgbd_fps_dict[ip]:4.1f}" for ip in rgbd_cameras.keys()]
                    rgbd_status = f" | RGB-D FPS: {', '.join(fps_strs)}"
                print(
                    f"Frames: {frame_count:5d} | FPS: {actual_fps:4.1f} | State: {state_str:12s} | Pos: {pos}{rgbd_status}"
                )
                last_print = now

            if sync is None:
                time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        if config.display:
            cv2.destroyAllWindows()
        for publisher in rgbd_publishers.values():
            publisher.shutdown()
        if slam:
            slam.shutdown()
        if rig:
            rig.stop()
        if rclpy.ok():
            rclpy.shutdown()
        print("Done.")


def main() -> None:
    """Entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run SLAM with Isaac ROS Visual SLAM and publish RGB-D for nvblox")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config YAML file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--rgbd-camera",
        type=str,
        default=None,
        help="[Deprecated] Use enable_rgbd: true in camera config instead. Multiple cameras can have RGB-D enabled.",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nExample config structure:")
        print(
            yaml.dump(
                {
                    "cameras": [
                        {
                            "ip": "192.168.2.21",
                            "stereo": True,
                            "resolution": [1280, 800],
                            "sensor_type": "MONO",
                            # All cameras are used for SLAM
                        },
                        {
                            "ip": "192.168.2.22",
                            "stereo": True,
                            "resolution": [1280, 800],
                            "sensor_type": "MONO",
                        },
                        {
                            "ip": "192.168.2.23",
                            "stereo": True,
                            "resolution": [1920, 1200],
                            "sensor_type": "COLOR",
                            # This camera will be used for nvblox (see nvblox_cameras below)
                        },
                    ],
                    "fps": 30,
                    "display": False,
                    "urdf_path": "",
                    "imu_report_rate": 400,
                    "nvblox_cameras": ["192.168.2.23"],  # Specify which cameras to use for nvblox
                    # Alternative: use enable_rgbd per camera instead
                    # "nvblox_cameras": null,  # If null, uses cameras with enable_rgbd: true
                },
                default_flow_style=False,
                sort_keys=False,
            )
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    if not config.cameras:
        print("Error: No cameras specified in config")
        sys.exit(1)

    # Warn about deprecated argument
    if args.rgbd_camera:
        print("Warning: --rgbd-camera is deprecated. Use enable_rgbd: true in camera config instead.")

    run(config)


if __name__ == "__main__":
    main()

