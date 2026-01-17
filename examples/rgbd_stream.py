"""Script to stream and visualize RGB-D data from Luxonis cameras.

Uses LuxonisCameraSource with enable_rgbd=True to capture RGB and depth frames
from the camera's built-in StereoDepth node.
"""

import argparse
import asyncio
import sys
import threading
import time
import traceback
from collections.abc import Callable

import cv2
import depthai as dai
import numpy as np
from askin import KeyboardController

from thor_slam.camera.drivers.luxonis import LuxonisCameraConfig, LuxonisCameraSource, LuxonisResolution


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


def colorize_depth(depth: np.ndarray, colormap: int, min_depth: float = 0.0, max_depth: float = 10000.0) -> np.ndarray:
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


def draw_info(img: np.ndarray, fps: float, info: dict[str, str] | None = None) -> np.ndarray:
    """Draw info overlay on image."""
    img_copy = img.copy()
    y_offset = 30
    line_height = 30

    # FPS
    cv2.putText(img_copy, f"FPS: {fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Additional info
    if info:
        y_offset += line_height
        for key, value in info.items():
            cv2.putText(
                img_copy, f"{key}: {value}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            y_offset += line_height

    return img_copy


def run_rgbd_stream(
    camera_ip: str,
    resolution: str = "1200",
    rgb_resolution: str | None = None,
    fps: int = 30,
    colormap: int = cv2.COLORMAP_JET,
    min_depth: float = 100.0,
    max_depth: float = 10000.0,
    display_width: int = 1280,
    show_stereo: bool = False,
) -> None:
    """Run RGB-D streaming and visualization."""
    print(f"\n{'=' * 60}")
    print("RGB-D Streaming")
    print(f"Camera: {camera_ip}")
    print(f"Stereo resolution: {resolution}, FPS: {fps}")
    if rgb_resolution:
        print(f"RGB resolution: {rgb_resolution}")
    if show_stereo:
        print("Stereo display: Enabled")
    print(f"{'=' * 60}\n")

    camera: LuxonisCameraSource | None = None

    try:
        # Create camera config with RGB-D enabled
        camera_config = LuxonisCameraConfig(
            ip=camera_ip,
            stereo=True,  # Required for RGB-D
            resolution=LuxonisResolution.from_name(resolution),
            fps=fps,
            queue_size=8,
            queue_blocking=False,
            enable_rgbd=True,  # Enable RGB-D streams
            rgb_resolution=LuxonisResolution.from_name(rgb_resolution) if rgb_resolution else None,
            rgbd_sync=True,  # Use synced RGB-D stream
            depth_preset=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
            depth_lr_check=True,
            depth_align_to_rgb=True,  # Align depth to RGB camera
        )

        # Create camera source
        camera = LuxonisCameraSource(camera_config)
        print(f"✓ Connected to {camera_ip}")

        # Log RGB-D intrinsics
        if camera.has_rgbd_streams:
            rgb_intrinsics, depth_intrinsics = camera.get_rgbd_intrinsics()
            print("\nRGB camera intrinsics:")
            print(f"  Resolution: {rgb_intrinsics.width}x{rgb_intrinsics.height}")
            print(f"  Focal length: fx={rgb_intrinsics.matrix[0, 0]:.1f}, fy={rgb_intrinsics.matrix[1, 1]:.1f}")
            print(f"  Principal point: cx={rgb_intrinsics.matrix[0, 2]:.1f}, cy={rgb_intrinsics.matrix[1, 2]:.1f}")

            print("\nDepth camera intrinsics:")
            print(f"  Resolution: {depth_intrinsics.width}x{depth_intrinsics.height}")
            print(f"  Focal length: fx={depth_intrinsics.matrix[0, 0]:.1f}, fy={depth_intrinsics.matrix[1, 1]:.1f}")
            print(f"  Principal point: cx={depth_intrinsics.matrix[0, 2]:.1f}, cy={depth_intrinsics.matrix[1, 2]:.1f}")

        # Start camera
        camera.start()
        print("✓ Camera started\n")

        quit_event, cleanup_quit = setup_quit_listener()
        display_msg = "RGB-D stream" if not show_stereo else "RGB-D stream with stereo"
        print(f"Displaying {display_msg}. Press 'q' to quit...\n")

        frame_times: list[float] = []

        while not quit_event.is_set():
            try:
                # Get RGB-D frames
                rgb_frame, depth_frame = camera.get_latest_rgbd_frames()

                rgb_img = rgb_frame.image
                depth_img = depth_frame.image  # Depth in millimeters (uint16)

                # Get stereo frames if requested
                left_img = None
                right_img = None
                if show_stereo:
                    stereo_frames = camera.get_latest_frames()
                    if len(stereo_frames) >= 2:
                        left_img = stereo_frames[0].image
                        right_img = stereo_frames[1].image

                # Track FPS
                current_time = time.time()
                frame_times.append(current_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)

                fps = 0.0
                if len(frame_times) >= 2:
                    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

                # Colorize depth
                depth_colored = colorize_depth(depth_img, colormap, min_depth, max_depth)

                # Get depth statistics for info overlay
                valid_depth = depth_img[depth_img > 0]
                depth_info = {}
                if valid_depth.size > 0:
                    depth_info["Depth"] = f"{valid_depth.mean():.0f}mm (min: {valid_depth.min():.0f}, max: {valid_depth.max():.0f})"

                # Draw info overlays
                rgb_display = draw_info(rgb_img.copy(), fps, {"RGB": f"{rgb_img.shape[1]}x{rgb_img.shape[0]}"})
                depth_display = draw_info(depth_colored, fps, depth_info)

                # Resize for display if needed
                if rgb_display.shape[1] > display_width:
                    scale = display_width / rgb_display.shape[1]
                    new_width = display_width
                    new_height = int(rgb_display.shape[0] * scale)
                    rgb_display = cv2.resize(rgb_display, (new_width, new_height))
                    depth_display = cv2.resize(depth_display, (new_width, new_height))

                if show_stereo and left_img is not None and right_img is not None:
                    # Convert stereo to BGR if grayscale
                    if len(left_img.shape) == 2:
                        left_display = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
                    else:
                        left_display = left_img.copy()
                    if len(right_img.shape) == 2:
                        right_display = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
                    else:
                        right_display = right_img.copy()

                    # Resize stereo frames to match RGB-D size
                    if left_display.shape[:2] != rgb_display.shape[:2]:
                        left_display = cv2.resize(left_display, (rgb_display.shape[1], rgb_display.shape[0]))
                        right_display = cv2.resize(right_display, (rgb_display.shape[1], rgb_display.shape[0]))

                    # Add info overlays to stereo frames
                    left_display = draw_info(left_display, fps, {"Left": f"{left_img.shape[1]}x{left_img.shape[0]}"})
                    right_display = draw_info(right_display, fps, {"Right": f"{right_img.shape[1]}x{right_img.shape[0]}"})

                    # Create 2x2 grid: top row (left, right), bottom row (RGB, depth)
                    top_row = np.hstack([left_display, right_display])
                    bottom_row = np.hstack([rgb_display, depth_display])
                    combined = np.vstack([top_row, bottom_row])

                    window_name = "RGB-D Stream (Left | Right / RGB | Depth)"
                else:
                    # Combine RGB and depth side by side
                    combined = np.hstack([rgb_display, depth_display])
                    window_name = "RGB-D Stream (RGB | Depth)"

                # Display
                cv2.imshow(window_name, combined)
                cv2.waitKey(1)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                break

        cleanup_quit()
        if camera:
            camera.stop()
        cv2.destroyAllWindows()
        print("\n✓ Done")

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        if camera:
            try:
                camera.stop()
            except Exception:
                pass
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
    parser = argparse.ArgumentParser(description="RGB-D streaming and visualization from Luxonis cameras")
    parser.add_argument(
        "--ip",
        type=str,
        help="Camera IP address (interactive selection if not provided)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1200",
        help="Stereo resolution for depth computation (default: 1200)",
    )
    parser.add_argument(
        "--rgb-resolution",
        type=str,
        default=None,
        help="RGB camera resolution (default: same as stereo resolution)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS (default: 30)",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        choices=["jet", "hot", "bone", "rainbow", "turbo", "viridis"],
        help="Colormap for depth visualization (default: jet)",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=100.0,
        help="Minimum depth in mm for visualization (default: 100)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10000.0,
        help="Maximum depth in mm for visualization (default: 10000)",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Maximum width for display window (default: 1280)",
    )
    parser.add_argument(
        "--show-stereo",
        action="store_true",
        help="Also display stereo left/right frames in a 2x2 grid",
    )

    args = parser.parse_args()

    # Get camera IP
    if args.ip:
        camera_ip = args.ip
    else:
        camera_ip = interactive_select_camera()
        if camera_ip is None:
            sys.exit(1)

    # Map colormap string to cv2 constant
    colormap_map = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "bone": cv2.COLORMAP_BONE,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }

    try:
        run_rgbd_stream(
            camera_ip=camera_ip,
            resolution=args.resolution,
            rgb_resolution=args.rgb_resolution,
            fps=args.fps,
            colormap=colormap_map.get(args.colormap, cv2.COLORMAP_JET),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            display_width=args.display_width,
            show_stereo=args.show_stereo,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

