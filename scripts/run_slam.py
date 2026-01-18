"""Run SLAM with Isaac ROS Visual SLAM.

Publishes camera frames to:
  /visual_slam/image_0, /visual_slam/image_1, ...
  /visual_slam/camera_info_0, /visual_slam/camera_info_1, ...

Example:
    # Use default config
    python -m scripts.run_slam

    # Use custom config
    python -m scripts.run_slam --config /path/to/config.yaml
"""

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import yaml

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


@dataclass
class SlamConfig:
    """SLAM configuration."""

    cameras: list[CameraConfig]
    fps: int = 30
    display: bool = False
    urdf_path: str = ""
    imu_report_rate: int = 400
    queue_size: int = 8
    rig_queue_size: int = 30

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlamConfig":
        """Create config from dictionary."""
        cameras = []
        for cam_data in data.get("cameras", []):
            resolution = cam_data.get("resolution", [1280, 800])
            output_resolution = cam_data.get("output_resolution")
            cameras.append(
                CameraConfig(
                    ip=cam_data["ip"],
                    stereo=cam_data.get("stereo", True),
                    resolution=(resolution[0], resolution[1]),
                    sensor_type=cam_data.get("sensor_type", "COLOR").upper(),
                    output_resolution=(
                        (output_resolution[0], output_resolution[1]) if output_resolution is not None else None
                    ),
                )
            )

        # Default URDF path
        urdf_path = data.get("urdf_path", "")
        if not urdf_path:
            default_urdf = Path(__file__).parent.parent / "examples" / "assets" / "brackets.urdf"
            if default_urdf.exists():
                urdf_path = str(default_urdf)

        return cls(
            cameras=cameras,
            fps=data.get("fps", 30),
            display=data.get("display", False),
            urdf_path=urdf_path,
            imu_report_rate=data.get("imu_report_rate", 400),
            queue_size=data.get("queue_size", 8),
            rig_queue_size=data.get("rig_queue_size", 30),
        )

    def calculate_num_cameras(self) -> int:
        """Calculate total number of camera streams (stereo=2, mono=1 per camera)."""
        return sum(2 if cam.stereo else 1 for cam in self.cameras)


def signal_handler(sig: int, frame: object) -> None:
    """Handle shutdown signal."""
    global _shutdown
    _shutdown = True
    print("\nShutting down...")


def load_config(config_path: Path) -> SlamConfig:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return SlamConfig.from_dict(data)


def create_sources(config: SlamConfig) -> tuple[dict[str, LuxonisCameraSource], str | None]:
    """Create camera sources directly from configuration."""
    print("\nCreating camera sources...")

    sources = {}
    first_ip = config.cameras[0].ip if config.cameras else None

    for cam_config in config.cameras:
        # Create camera config directly from config values
        output_resolution = None
        if cam_config.output_resolution is not None:
            output_resolution = LuxonisResolution.from_dimensions(
                cam_config.output_resolution[0], cam_config.output_resolution[1]
            )

        luxonis_config = LuxonisCameraConfig(
            ip=cam_config.ip,
            stereo=cam_config.stereo,
            mono_sensor_resolution=LuxonisResolution.from_dimensions(
                cam_config.resolution[0], cam_config.resolution[1]
            ),
            fps=config.fps,
            queue_size=config.queue_size,
            queue_blocking=False,
            camera_mode=cast(CameraSensorType, cam_config.sensor_type),
            read_imu=(cam_config.ip == first_ip),  # Enable IMU on first camera
            imu_report_rate=config.imu_report_rate,
            output_resolution=output_resolution,
        )

        source = LuxonisCameraSource(luxonis_config)
        sources[cam_config.ip] = source

        imu_status = " (IMU enabled)" if cam_config.ip == first_ip else ""
        stereo_status = "stereo" if cam_config.stereo else "mono"
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
            )
        )

    return sources, first_ip


def run(config: SlamConfig) -> None:
    """Run SLAM pipeline.

    Args:
        config: SLAM configuration.
    """
    num_cameras = config.calculate_num_cameras()

    print("\n" + "=" * 60)
    print("Thor SLAM -> Isaac ROS Visual SLAM")
    print("=" * 60)
    print(f"Total camera streams: {num_cameras}")
    print(f"Physical cameras: {len(config.cameras)}")
    for i, cam in enumerate(config.cameras):
        stereo_str = "stereo" if cam.stereo else "mono"
        print(f"  Camera {i + 1}: {cam.ip} ({stereo_str}, {cam.resolution[0]}x{cam.resolution[1]}, {cam.sensor_type})")
    print(f"FPS: {config.fps}")
    if config.urdf_path:
        print(f"URDF: {config.urdf_path}")
    print(f"{'=' * 60}\n")

    rig = None
    slam = None

    try:
        # Create sources
        sources, first_ip = create_sources(config)

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

        # Main loop
        print(f"\n{'─' * 60}")
        print("Running. Press Ctrl+C to stop.")
        print(f"{'─' * 60}\n")

        last_print = time.time()
        frame_count = 0
        start_time = time.time()
        actual_fps = 0.0

        while not _shutdown:
            sync = rig.get_synchronized_frames()
            # sync = rig.get_latest_frames()
            if sync is None:
                time.sleep(0.001)
                continue

            pose = slam.process_frames(sync)
            frame_count += 1

            # Calculate actual FPS as average over elapsed time
            # This measures the rate of synchronized frame sets being processed
            elapsed = time.time() - start_time
            if elapsed > 0:
                actual_fps = frame_count / elapsed

            # Display
            if config.display:
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

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Status
            now = time.time()
            if now - last_print >= 2.0:
                state = slam.get_tracking_state()
                pos = "N/A"
                if pose:
                    p = pose.position
                    pos = f"[{p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f}]"
                print(f"Frames: {frame_count:5d} | FPS: {actual_fps:4.1f} | State: {state.name:12s} | Pos: {pos}")
                last_print = now

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        if config.display:
            cv2.destroyAllWindows()
        if slam:
            slam.shutdown()
        if rig:
            rig.stop()
        print("Done.")


def main() -> None:
    """Entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run SLAM with Isaac ROS Visual SLAM")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config YAML file (default: {DEFAULT_CONFIG_PATH})",
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
                        {"ip": "192.168.2.21", "stereo": True, "resolution": [1280, 800], "sensor_type": "COLOR"}
                    ],
                    "fps": 30,
                    "display": False,
                    "urdf_path": "",
                    "imu_report_rate": 400,
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

    run(config)


if __name__ == "__main__":
    main()
