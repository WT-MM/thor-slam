"""Test stream resolutions - test RGB-D and stereo streams at specific resolutions."""

import argparse
import asyncio
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable

import cv2
import depthai as dai
import numpy as np
from askin import KeyboardController

from thor_slam.camera.drivers.luxonis import (
    LuxonisCameraConfig,
    LuxonisCameraSource,
    LuxonisResolution,
    LuxonisRGBDCameraConfig,
)

# Global shutdown flag
_shutdown = False


def signal_handler(sig: int, frame: object) -> None:
    """Handle shutdown signal."""
    global _shutdown
    _shutdown = True
    print("\nShutting down...")


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


def colorize_depth(
    depth: np.ndarray, colormap: int = cv2.COLORMAP_JET, min_depth: float = 100.0, max_depth: float = 10000.0
) -> np.ndarray:
    """Colorize a depth map for visualization.

    Args:
        depth: Depth map in millimeters (uint16)
        colormap: OpenCV colormap constant
        min_depth: Minimum depth in mm for visualization
        max_depth: Maximum depth in mm for visualization

    Returns:
        Colorized image (BGR)
    """
    # Clip depth to valid range
    depth_clipped = np.clip(depth, min_depth, max_depth)

    # Normalize to 0-255
    if max_depth > min_depth:
        normalized = ((depth_clipped - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(depth, dtype=np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(normalized, colormap)

    # Set invalid regions (depth == 0) to black
    valid_mask = depth > 0
    colored[~valid_mask] = 0

    return colored


def draw_info(
    img: np.ndarray,
    fps: float,
    info: dict[str, str] | None = None,
    label: str | None = None,
    resolution: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw info overlay on image."""
    img_copy = img.copy()
    y_offset = 30
    line_height = 30

    # FPS
    cv2.putText(img_copy, f"FPS: {fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Label and resolution (for stereo mode)
    if label and resolution:
        y_offset += line_height
        cv2.putText(
            img_copy,
            f"{label}: {resolution[0]}x{resolution[1]}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # Additional info (for RGB-D mode)
    if info:
        y_offset += line_height
        for key, value in info.items():
            cv2.putText(
                img_copy, f"{key}: {value}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            y_offset += line_height

    return img_copy


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


def run_test_stream_resolutions(
    camera_ip: str,
    stereo_output_resolution: tuple[int, int] = (640, 400),
    rgbd_output_resolution: tuple[int, int] = (1280, 800),
    fps: int = 30,
) -> None:
    """Run test stream with specific resolutions for RGB-D and stereo.

    Args:
        camera_ip: IP address of the camera
        stereo_output_resolution: Output resolution for stereo left/right (width, height)
        rgbd_output_resolution: Output resolution for RGB and depth (width, height)
        fps: Frame rate
    """
    print(f"\n{'=' * 60}")
    print("Test Stream: RGB-D and Stereo with Specific Resolutions")
    print(f"{'=' * 60}")
    print(f"Camera IP: {camera_ip}")
    print(f"FPS: {fps}")
    print("\nResolutions:")
    print(f"  - RGB: {rgbd_output_resolution[0]}x{rgbd_output_resolution[1]}")
    print(f"  - Depth: {rgbd_output_resolution[0]}x{rgbd_output_resolution[1]}")
    print(f"  - Stereo Left/Right: {stereo_output_resolution[0]}x{stereo_output_resolution[1]}")
    print(f"{'=' * 60}\n")

    camera: LuxonisCameraSource | None = None

    try:
        # RGB-D configuration
        rgbd_camera_config = LuxonisRGBDCameraConfig(
            enable_rgbd=True,
            rgb_output_resolution=LuxonisResolution.from_dimensions(
                rgbd_output_resolution[0], rgbd_output_resolution[1]
            ),
            depth_output_resolution=LuxonisResolution.from_dimensions(
                rgbd_output_resolution[0], rgbd_output_resolution[1]
            ),
            rgbd_sync=True,
            depth_preset=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
            depth_lr_check=True,
            depth_align_to_rgb=True,
        )

        # For simplicity, use rgbd_output_resolution as sensor resolution
        mono_sensor_resolution = LuxonisResolution.from_dimensions(
            rgbd_output_resolution[0], rgbd_output_resolution[1]
        )

        camera_config = LuxonisCameraConfig(
            ip=camera_ip,
            stereo=True,  # Required for RGB-D
            fps=fps,
            queue_size=8,
            queue_blocking=False,
            camera_mode="MONO",
            mono_sensor_resolution=mono_sensor_resolution,
            output_resolution=LuxonisResolution.from_dimensions(
                stereo_output_resolution[0], stereo_output_resolution[1]
            ),
            rgbd_camera_config=rgbd_camera_config,
        )

        # Create and start camera
        camera = LuxonisCameraSource(camera_config)
        print(f"✓ Connected to {camera_ip}")

        # Log intrinsics
        if camera.has_rgbd_streams:
            rgb_intrinsics, depth_intrinsics = camera.get_rgbd_intrinsics()
            print(f"\nRGB intrinsics: {rgb_intrinsics.width}x{rgb_intrinsics.height}")
            print(f"Depth intrinsics: {depth_intrinsics.width}x{depth_intrinsics.height}")

        stereo_intrinsics = camera.get_intrinsics()
        if len(stereo_intrinsics) >= 2:
            print(f"Stereo left intrinsics: {stereo_intrinsics[0].width}x{stereo_intrinsics[0].height}")
            print(f"Stereo right intrinsics: {stereo_intrinsics[1].width}x{stereo_intrinsics[1].height}")

        camera.start()
        print("✓ Camera started\n")
        print("Displaying streams. Press Ctrl+C or 'q' to quit...\n")

        # FPS tracking
        frame_times_rgbd: list[float] = []
        frame_times_stereo: list[float] = []

        quit_event, cleanup_quit = setup_quit_listener()

        while not quit_event.is_set() and not _shutdown:
            try:
                # Get RGB-D frames
                rgb_frame, depth_frame = camera.get_latest_rgbd_frames()
                rgb_img = rgb_frame.image
                depth_img = depth_frame.image  # Depth in millimeters (uint16)

                # Get stereo frames
                stereo_frames = camera.get_latest_frames()
                if len(stereo_frames) < 2:
                    continue
                left_img = stereo_frames[0].image
                right_img = stereo_frames[1].image

                # Track FPS
                current_time = time.time()
                frame_times_rgbd.append(current_time)
                frame_times_stereo.append(current_time)

                # Keep only last 30 frames for FPS calculation
                if len(frame_times_rgbd) > 30:
                    frame_times_rgbd.pop(0)
                if len(frame_times_stereo) > 30:
                    frame_times_stereo.pop(0)

                # Calculate FPS
                fps_rgbd = calculate_fps(frame_times_rgbd)
                fps_stereo = calculate_fps(frame_times_stereo)

                # Colorize depth
                depth_colored = colorize_depth(depth_img, min_depth=100.0, max_depth=10000.0)

                # Convert stereo to BGR if grayscale
                if len(left_img.shape) == 2:
                    left_display = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
                else:
                    left_display = left_img.copy()

                if len(right_img.shape) == 2:
                    right_display = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
                else:
                    right_display = right_img.copy()

                # Draw info overlays
                rgb_display = draw_info(
                    rgb_img.copy(), fps_rgbd, None, "RGB", (rgb_img.shape[1], rgb_img.shape[0])
                )
                depth_display = draw_info(
                    depth_colored, fps_rgbd, None, "Depth", (depth_img.shape[1], depth_img.shape[0])
                )
                left_display = draw_info(
                    left_display, fps_stereo, None, "Stereo Left", (left_img.shape[1], left_img.shape[0])
                )
                right_display = draw_info(
                    right_display, fps_stereo, None, "Stereo Right", (right_img.shape[1], right_img.shape[0])
                )

                # Resize for display (scale down if too large)
                display_width = 640

                # Resize RGB-D if needed
                if rgb_display.shape[1] > display_width:
                    scale = display_width / rgb_display.shape[1]
                    new_height = int(rgb_display.shape[0] * scale)
                    rgb_display = cv2.resize(rgb_display, (display_width, new_height))
                    depth_display = cv2.resize(depth_display, (display_width, new_height))

                # Resize stereo if needed
                if left_display.shape[1] > display_width:
                    scale = display_width / left_display.shape[1]
                    new_height = int(left_display.shape[0] * scale)
                    left_display = cv2.resize(left_display, (display_width, new_height))
                    right_display = cv2.resize(right_display, (display_width, new_height))

                # Ensure all images have same height for grid
                target_height = min(
                    rgb_display.shape[0], depth_display.shape[0], left_display.shape[0], right_display.shape[0]
                )

                if rgb_display.shape[0] != target_height:
                    new_w = int(rgb_display.shape[1] * (target_height / rgb_display.shape[0]))
                    rgb_display = cv2.resize(rgb_display, (new_w, target_height), interpolation=cv2.INTER_AREA)
                if depth_display.shape[0] != target_height:
                    new_w = int(depth_display.shape[1] * (target_height / depth_display.shape[0]))
                    depth_display = cv2.resize(depth_display, (new_w, target_height), interpolation=cv2.INTER_AREA)
                if left_display.shape[0] != target_height:
                    new_w = int(left_display.shape[1] * (target_height / left_display.shape[0]))
                    left_display = cv2.resize(left_display, (new_w, target_height), interpolation=cv2.INTER_AREA)
                if right_display.shape[0] != target_height:
                    new_w = int(right_display.shape[1] * (target_height / right_display.shape[0]))
                    right_display = cv2.resize(right_display, (new_w, target_height), interpolation=cv2.INTER_AREA)

                # Create 2x2 grid: top row (stereo left, stereo right), bottom row (RGB, depth)
                top_row = np.hstack([left_display, right_display])
                bottom_row = np.hstack([rgb_display, depth_display])
                combined = np.vstack([top_row, bottom_row])

                # Display
                cv2.imshow(
                    f"Test Stream: Stereo ({stereo_output_resolution[0]}x{stereo_output_resolution[1]}) | "
                    f"RGB-D ({rgbd_output_resolution[0]}x{rgbd_output_resolution[1]})",
                    combined,
                )

                # Check for 'q' key or window close
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                break

        cleanup_quit()
        print("\n✓ Done")

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
    finally:
        if camera:
            camera.stop()
        cv2.destroyAllWindows()


def interactive_select_camera() -> str | None:
    """Interactively select a camera."""
    devices = find_available_cameras()

    if not devices:
        print("No cameras found!")
        return None

    print("\nAvailable cameras:")
    for i, d in enumerate(devices, 1):
        print(f"  {i}. {d.name}")

    if len(devices) == 1:
        print(f"\nUsing only available camera: {devices[0].name}")
        return devices[0].name

    try:
        choice = input("\nSelect camera (number or IP): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(devices):
                return devices[idx].name
        else:
            # Treat as IP
            return choice
    except (ValueError, KeyboardInterrupt):
        pass

    return None


def main() -> None:
    """Main entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Test stream resolutions - RGB-D and stereo with specific resolutions")
    parser.add_argument(
        "--ip",
        type=str,
        help="Camera IP address. Interactive selection if not provided.",
    )
    parser.add_argument(
        "--stereo-output",
        type=str,
        default="400",
        help="Stereo output resolution name (default: 400 = 640x400)",
    )
    parser.add_argument(
        "--rgbd-output",
        type=str,
        default="800",
        help="RGB-D output resolution name (default: 800 = 1280x800)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS (default: 30)",
    )

    args = parser.parse_args()

    try:
        if args.ip:
            camera_ip = args.ip
        else:
            camera_ip = interactive_select_camera()
            if camera_ip is None:
                sys.exit(1)

        # Parse resolution names to tuples
        stereo_res = LuxonisResolution.from_name(args.stereo_output)
        rgbd_res = LuxonisResolution.from_name(args.rgbd_output)

        run_test_stream_resolutions(
            camera_ip=camera_ip,
            stereo_output_resolution=stereo_res.as_tuple(),
            rgbd_output_resolution=rgbd_res.as_tuple(),
            fps=args.fps,
        )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()

