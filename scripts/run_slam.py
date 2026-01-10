#!/usr/bin/env python3
"""Run SLAM with Isaac ROS Visual SLAM.

Publishes camera frames to:
  /visual_slam/image_0, /visual_slam/image_1, ...
  /visual_slam/camera_info_0, /visual_slam/camera_info_1, ...

Example:
    # Single stereo camera (2 cameras)
    python -m scripts.run_slam --camera-ips 192.168.2.21

    # Two stereo cameras (4 cameras)
    python -m scripts.run_slam --camera-ips 192.168.2.21,192.168.2.22 --num-cameras 4
"""

import argparse
import logging
import signal
import sys
import time

import cv2
import depthai as dai
import numpy as np

from thor_slam.camera.drivers.luxonis import (
    LuxonisCameraConfig,
    LuxonisCameraSource,
    LuxonisResolution,
)
from thor_slam.camera.rig import CameraRig
from thor_slam.camera.types import Extrinsics, IPv4
from thor_slam.camera.utils import (
    get_luxonis_camera_valid_modes,
    get_luxonis_camera_valid_resolutions,
    get_luxonis_device,
)
from thor_slam.slam import IsaacRosAdapter

# Shutdown flag
_shutdown = False


def signal_handler(sig: int, frame: object) -> None:
    """Handle shutdown signal."""
    global _shutdown
    _shutdown = True
    print("\nShutting down...")


def parse_ips(value: str) -> list[str]:
    """Parse comma-separated IPs."""
    return [ip.strip() for ip in value.split(",") if ip.strip()]


def detect_cameras(ips: list[str]) -> list[dict]:
    """Detect camera capabilities."""
    cameras = []
    for ip in ips:
        print(f"  Checking {ip}...")
        device = get_luxonis_device(IPv4(ip))
        if device is None:
            raise RuntimeError(f"Cannot connect to {ip}")

        # Get valid resolutions for stereo cameras
        resolutions = get_luxonis_camera_valid_resolutions(device, dai.CameraBoardSocket.CAM_B)
        modes = get_luxonis_camera_valid_modes(device, dai.CameraBoardSocket.CAM_B)
        has_mono = dai.CameraSensorType.MONO in modes
        device.close()

        cameras.append(
            {
                "ip": ip,
                "resolutions": resolutions,
                "mono": has_mono,
            }
        )
        print(f"    ✓ {len(resolutions)} resolutions, {'MONO' if has_mono else 'COLOR'}")

    return cameras


def create_sources(cameras: list[dict], fps: int) -> tuple[list[LuxonisCameraSource], str | None]:
    """Create camera sources with largest common resolution."""
    # Find common resolutions
    common = set(cameras[0]["resolutions"])
    for cam in cameras[1:]:
        common &= set(cam["resolutions"])

    if not common:
        raise RuntimeError("No common resolution found")

    # Use largest
    resolution = max(common, key=lambda r: r[0] * r[1])
    print(f"  Resolution: {resolution[0]}x{resolution[1]}")

    # Wait for devices to be released after detection
    time.sleep(1.0)

    sources = []
    first_ip = cameras[0]["ip"] if cameras else None
    for cam in cameras:
        config = LuxonisCameraConfig(
            ip=cam["ip"],
            stereo=True,
            resolution=LuxonisResolution.from_dimensions(resolution[0], resolution[1]),
            fps=fps,
            queue_size=8,
            queue_blocking=False,
            camera_mode="MONO" if cam["mono"] else "COLOR",
            read_imu=(cam["ip"] == first_ip),  # Enable IMU on first camera
            imu_report_rate=400,
        )
        source = LuxonisCameraSource(config)
        sources.append(source)
        imu_status = " (IMU enabled)" if cam["ip"] == first_ip else ""
        print(f"  ✓ {cam['ip']}{imu_status}")

    return sources, first_ip


def run(camera_ips: list[str], num_cameras: int, fps: int, display: bool) -> None:
    """Run SLAM pipeline."""
    print("\n" + "=" * 60)
    print("Thor SLAM -> Isaac ROS Visual SLAM")
    print("=" * 60)
    print(f"Cameras: {num_cameras}")
    print(f"IPs: {', '.join(camera_ips)}")
    print(f"FPS: {fps}")
    print(f"{'=' * 60}\n")

    rig = None
    slam = None

    try:
        # Detect cameras
        print("Detecting cameras...")
        cameras = detect_cameras(camera_ips)

        time.sleep(5.0)

        # Create sources
        print("\nCreating camera sources...")
        sources, first_ip = create_sources(cameras, fps)

        # Create rig
        print("\nStarting camera rig...")

        imu_extrinsics = Extrinsics.from_4x4_matrix(
            np.array(
                [
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 1],
                ]
            )
        )
        # Use first camera as IMU source (name is the IP address)
        rig = CameraRig(sources=sources, queue_size=30, imu_source=first_ip, imu_extrinsics=imu_extrinsics)
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
        last_timestamp = 0.0
        actual_fps = 0.0

        while not _shutdown:
            sync = rig.get_synchronized_frames()
            if sync is None:
                time.sleep(0.001)
                continue

            pose = slam.process_frames(sync)
            frame_count += 1

            # Calculate actual FPS from camera timestamps
            if last_timestamp > 0:
                delta = sync.timestamp - last_timestamp
                if delta > 0:
                    actual_fps = 1.0 / delta
            last_timestamp = sync.timestamp

            # Display
            if display:
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
        if display:
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
        "--camera-ips",
        type=str,
        required=True,
        help="Camera IP(s), comma-separated",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=2,
        help="Number of cameras (default: 2 for stereo)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frame rate (default: 10)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display",
    )

    args = parser.parse_args()
    ips = parse_ips(args.camera_ips)

    if not ips:
        print("Error: No camera IPs provided")
        sys.exit(1)

    run(ips, args.num_cameras, args.fps, not args.no_display)


if __name__ == "__main__":
    main()
