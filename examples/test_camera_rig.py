#!/usr/bin/env python3
"""Test script for CameraRig - tests multi-camera synchronization."""

import argparse
import asyncio
import sys
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable

import cv2
import depthai as dai
import numpy as np
from askin import KeyboardController

from thor_slam.camera.drivers.luxonis import LuxonisCameraConfig, LuxonisCameraSource, LuxonisResolution
from thor_slam.camera.rig import CameraRig
from thor_slam.camera.types import IPv4
from thor_slam.camera.utils import (
    get_luxonis_camera_valid_modes,
    get_luxonis_camera_valid_resolutions,
    get_luxonis_device,
)


def find_available_cameras() -> list[dai.DeviceInfo]:
    """Find all available cameras on the network."""
    devices = dai.Device.getAllAvailableDevices()
    return devices


def setup_quit_listener() -> tuple[threading.Event, Callable[[], None]]:
    """Set up askin to listen for 'q' key press."""
    quit_event = threading.Event()

    async def key_handler(key: str) -> None:
        if key == "q":
            quit_event.set()

    async def run_controller() -> None:
        controller = KeyboardController(key_handler=key_handler, timeout=0.001)
        await controller.start()
        try:
            while not quit_event.is_set():
                await asyncio.sleep(0.1)
        finally:
            try:
                await controller.stop()
            except Exception:
                pass

    def run_async() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_controller())
        except Exception:
            pass
        finally:
            try:
                if loop and not loop.is_closed():
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                if loop and not loop.is_closed():
                    loop.close()
            except Exception:
                pass

    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()

    def cleanup() -> None:
        quit_event.set()
        if thread:
            thread.join(timeout=1.0)

    return quit_event, cleanup


def calculate_fps(frame_times: list[float], window_size: int = 30) -> float:
    """Calculate FPS from recent frame times."""
    if len(frame_times) < 2:
        return 0.0
    recent_times = frame_times[-window_size:]
    if len(recent_times) < 2:
        return 0.0
    time_diff = recent_times[-1] - recent_times[0]
    if time_diff <= 0:
        return 0.0
    return (len(recent_times) - 1) / time_diff


def draw_info(
    img: np.ndarray,
    fps: float,
    timestamp: float,
    sync_delta: float | None = None,
    source_name: str = "",
) -> np.ndarray:
    """Draw info overlay on image."""
    img_copy = img.copy()
    y_offset = 30
    line_height = 40

    # FPS
    cv2.putText(
        img_copy,
        f"FPS: {fps:.1f}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Source name
    if source_name:
        cv2.putText(
            img_copy,
            f"Source: {source_name}",
            (10, y_offset + line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    # Timestamp
    cv2.putText(
        img_copy,
        f"TS: {timestamp:.3f}s",
        (10, y_offset + line_height * 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Sync delta (if provided)
    if sync_delta is not None:
        color = (0, 255, 0) if abs(sync_delta) < 0.01 else (0, 165, 255) if abs(sync_delta) < 0.05 else (0, 0, 255)
        cv2.putText(
            img_copy,
            f"Sync Δ: {sync_delta * 1000:.1f}ms",
            (10, y_offset + line_height * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return img_copy


def resize_for_display(img: np.ndarray, max_width: int) -> np.ndarray:
    """Resize image to max width while maintaining aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def test_camera_rig(
    ips: list[str],
    fps: int = 30,
    queue_size: int = 30,
) -> None:
    """Test CameraRig with multiple cameras."""
    print(f"\n{'=' * 80}")
    print("CameraRig Test")
    print(f"{'=' * 80}\n")
    print(f"Testing {len(ips)} camera(s)")
    print(f"FPS: {fps}, Queue size: {queue_size}")
    print("Note: Resolution will be auto-selected based on camera capabilities\n")

    rig: CameraRig | None = None

    try:
        # Check camera capabilities: mode (mono vs stereo) and resolutions
        print("Checking camera capabilities...")
        camera_capabilities: list[dict[str, any]] = []  # type: ignore[valid-type]

        for ip in ips:
            device = get_luxonis_device(IPv4(ip))
            if device is None:
                raise ValueError(f"Could not connect to camera {ip}")

            # Check stereo mode (CAM_B and CAM_C)
            stereo_resolutions_b = get_luxonis_camera_valid_resolutions(device, dai.CameraBoardSocket.CAM_B)
            stereo_resolutions_c = get_luxonis_camera_valid_resolutions(device, dai.CameraBoardSocket.CAM_C)
            stereo_available = len(stereo_resolutions_b) > 0 and len(stereo_resolutions_c) > 0

            if not stereo_available:
                raise ValueError(f"No valid stereo camera modes found for camera {ip}")

            # Check sensor types available for stereo cameras (CAM_B) to determine MONO vs COLOR
            valid_modes = get_luxonis_camera_valid_modes(device, dai.CameraBoardSocket.CAM_B)
            sensor_type_mono_available = dai.CameraSensorType.MONO in valid_modes

            # Use CAM_B resolutions as representative for stereo
            valid_resolutions = stereo_resolutions_b

            # Determine preferred sensor type: MONO if available, else COLOR
            preferred_sensor_type = "MONO" if sensor_type_mono_available else "COLOR"

            camera_capabilities.append(
                {
                    "ip": ip,
                    "resolutions": valid_resolutions,
                    "sensor_type": preferred_sensor_type,
                }
            )
            print(f"  {ip}: STEREO mode, {len(valid_resolutions)} resolution(s), sensor: {preferred_sensor_type}")
            device.close()

        # Find intersection of all valid resolutions and pick the SMALLEST
        all_resolutions = [cap["resolutions"] for cap in camera_capabilities]
        if len(all_resolutions) > 1:
            common_resolutions: set[tuple[int, int]] = set(all_resolutions[0])
            for res_list in all_resolutions[1:]:
                common_resolutions &= set(res_list)

            if not common_resolutions:
                print("⚠ Warning: No common resolutions found. Using per-camera resolutions.")
                # Use the smallest resolution from each camera individually
                selected_resolutions = [min(res_list, key=lambda r: r[0] * r[1]) for res_list in all_resolutions]
            else:
                # Use the SMALLEST common resolution
                selected_resolution = min(common_resolutions, key=lambda r: r[0] * r[1])
                selected_resolutions = [selected_resolution] * len(ips)
                print(f"✓ Selected common resolution: {selected_resolution[0]}x{selected_resolution[1]}")
        else:
            # Single camera, pick the smallest
            selected_resolution = min(all_resolutions[0], key=lambda r: r[0] * r[1])
            selected_resolutions = [selected_resolution]
            print(f"✓ Selected resolution: {selected_resolution[0]}x{selected_resolution[1]}")

        # Create camera sources
        sources: list[LuxonisCameraSource] = []
        for i, ip in enumerate(ips):
            print(f"\nInitializing camera {ip}...")
            selected_res = selected_resolutions[i]
            resolution = LuxonisResolution.from_dimensions(selected_res[0], selected_res[1])
            resolution = LuxonisResolution.from_name("800")
            # Get preferred sensor type (MONO if available, else COLOR)
            cap = camera_capabilities[i]
            sensor_type = cap["sensor_type"]

            config = LuxonisCameraConfig(
                ip=ip,
                stereo=True,
                resolution=resolution,
                fps=fps,
                queue_size=8,
                queue_blocking=False,
                camera_mode=sensor_type,
            )
            camera = LuxonisCameraSource(config)
            sources.append(camera)
            print(f"  ✓ Camera {ip} initialized: {selected_res[0]}x{selected_res[1]} @ STEREO ({sensor_type})")

        # Create rig
        print(f"\nCreating CameraRig with {len(sources)} source(s)...")
        rig = CameraRig(sources=sources, queue_size=queue_size)  # type: ignore[arg-type]
        print("✓ CameraRig created")

        # Get calibration
        print("\n--- Rig Calibration Data ---")
        calibration = rig.calibration
        for source_name, intrinsics_list in calibration.intrinsics.items():
            print(f"\n{source_name}:")
            for i, intrinsics in enumerate(intrinsics_list):
                camera_type = ["Left", "Right", "RGB"][i] if i < 3 else f"Camera {i}"
                print(f"  {camera_type} Camera:")
                print(f"    Resolution: {intrinsics.width}x{intrinsics.height}")
                print(f"    Focal: fx={intrinsics.matrix[0, 0]:.1f}, fy={intrinsics.matrix[1, 1]:.1f}")
                print(f"    Principal: cx={intrinsics.matrix[0, 2]:.1f}, cy={intrinsics.matrix[1, 2]:.1f}")

        print("\n--- End Calibration Data ---\n")

        # Start rig
        rig.start()
        print("✓ CameraRig started\n")

        quit_event, cleanup_quit = setup_quit_listener()
        print("Displaying synchronized frames. Press 'q' to quit...\n")

        # Statistics tracking
        frame_times: dict[str, list[float]] = defaultdict(list)
        sync_deltas: list[float] = []
        frame_count = 0
        last_print_time = time.time()

        while not quit_event.is_set():
            try:
                # Get synchronized frames
                sync_set = rig.get_synchronized_frames()

                if sync_set is None:
                    # Not all cameras have frames yet, wait a bit
                    time.sleep(0.01)
                    continue

                # Track statistics
                current_time = time.time()
                frame_count += 1
                sync_deltas.append(sync_set.max_time_delta)

                # Process each source's frames
                display_images: dict[str, np.ndarray] = {}

                for source_name, frame_set in sync_set.frame_sets.items():
                    for frame in frame_set.frames:
                        # Track FPS per camera
                        camera_key = frame.camera_name
                        frame_times[camera_key].append(current_time)
                        if len(frame_times[camera_key]) > 60:
                            frame_times[camera_key].pop(0)

                        # Calculate FPS
                        camera_fps = calculate_fps(frame_times[camera_key])

                        # Calculate sync delta for this frame
                        frame_sync_delta = frame.timestamp - sync_set.timestamp

                        if frame_count % 50 == 0:

                            print(f"Frame timestamp: {frame.timestamp}")
                            print(f"Sync set timestamp: {sync_set.timestamp}")
                            print(f"Frame sync delta: {frame_sync_delta}")

                        # Prepare image for display
                        if len(frame.image.shape) == 2:
                            img = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
                        else:
                            img = frame.image.copy()

                        # Resize for display
                        img = resize_for_display(img, 640)

                        # Draw info
                        img = draw_info(img, camera_fps, frame.timestamp, frame_sync_delta, source_name)

                        # Add to display dict
                        display_images[camera_key] = img

                # Display all images
                if display_images:
                    for window_name, img in display_images.items():
                        cv2.imshow(window_name, img)

                # Print statistics periodically
                # if current_time - last_print_time >= 1.0:
                #     queue_depths = rig.get_queue_depths()
                #     avg_sync_delta = sum(sync_deltas[-30:]) / len(sync_deltas[-30:]) if sync_deltas else 0.0
                #     max_sync_delta = max(sync_deltas[-30:]) if sync_deltas else 0.0

                #     print(
                #         f"Frames: {frame_count} | "
                #         f"Sync Δ: avg={avg_sync_delta * 1000:.1f}ms, max={max_sync_delta * 1000:.1f}ms | "
                #         f"Queues: {queue_depths}"
                #     )
                #     last_print_time = current_time

                # Process OpenCV events
                cv2.waitKey(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                break

        cleanup_quit()
        if rig:
            rig.stop()
        cv2.destroyAllWindows()

        # Print final statistics
        print("\n" + "=" * 80)
        print("Final Statistics")
        print("=" * 80)
        print(f"Total synchronized frames: {frame_count}")
        if sync_deltas:
            print(f"Average sync delta: {sum(sync_deltas) / len(sync_deltas) * 1000:.2f}ms")
            print(f"Max sync delta: {max(sync_deltas) * 1000:.2f}ms")
            print(f"Min sync delta: {min(sync_deltas) * 1000:.2f}ms")
        print(f"Final queue depths: {rig.get_queue_depths() if rig else {}}")
        print("\n✓ Test completed")

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        if rig:
            try:
                rig.stop()
            except Exception:
                pass
        cv2.destroyAllWindows()


def interactive_select_cameras() -> list[str] | None:
    """Interactively select cameras."""
    devices = find_available_cameras()

    if not devices:
        print("No cameras found!")
        return None

    print("\nAvailable cameras:")
    for i, d in enumerate(devices, 1):
        print(f"  {i}. {d.name}")

    try:
        choice = input("\nSelect cameras (comma-separated numbers or IPs, or 'all'): ").strip()
        if choice.lower() == "all":
            return [d.name for d in devices]

        selected = []
        for item in choice.split(","):
            item = item.strip()
            if item.isdigit():
                idx = int(item) - 1
                if 0 <= idx < len(devices):
                    selected.append(devices[idx].name)
            else:
                # Treat as IP
                selected.append(item)
        return selected if selected else None
    except (ValueError, KeyboardInterrupt):
        pass

    return None


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test CameraRig synchronization")
    parser.add_argument(
        "--ips",
        type=str,
        nargs="+",
        help="Camera IP addresses (interactive selection if not provided)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS (default: 10)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=15,
        help="Queue size for each camera (default: 15)",
    )

    args = parser.parse_args()

    # Get camera IPs
    if args.ips:
        camera_ips = args.ips
    else:
        camera_ips = interactive_select_cameras()
        if camera_ips is None:
            sys.exit(1)

    if len(camera_ips) < 1:
        print("Need at least one camera!")
        sys.exit(1)

    try:
        test_camera_rig(camera_ips, fps=args.fps, queue_size=args.queue_size)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
