"""Test script to stream depth and RGB at 1280x800 and stereo at 640x400.

This script demonstrates streaming:
- RGB and depth components at 1280x800 resolution
- Stereo left/right at 640x400 resolution
"""

import argparse
import signal
import sys
import time
import traceback

import cv2
import depthai as dai
import numpy as np

from thor_slam.camera.drivers.luxonis import LuxonisCameraConfig, LuxonisCameraSource, LuxonisResolution

# Global shutdown flag
_shutdown = False


def signal_handler(sig: int, frame: object) -> None:
    """Handle shutdown signal."""
    global _shutdown
    _shutdown = True
    print("\nShutting down...")


def colorize_depth(depth: np.ndarray, min_depth: float = 100.0, max_depth: float = 10000.0) -> np.ndarray:
    """Colorize a depth map for visualization.

    Args:
        depth: Depth map in millimeters (uint16)
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
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    # Set invalid regions (depth == 0) to black
    valid_mask = depth > 0
    colored[~valid_mask] = 0

    return colored


def draw_info(img: np.ndarray, fps: float, label: str, resolution: tuple[int, int]) -> np.ndarray:
    """Draw info overlay on image."""
    img_copy = img.copy()
    y_offset = 30
    line_height = 30

    # Label and resolution
    cv2.putText(
        img_copy,
        f"{label}: {resolution[0]}x{resolution[1]}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # FPS
    y_offset += line_height
    cv2.putText(img_copy, f"FPS: {fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return img_copy


def run_test_stream(camera_ip: str, fps: int = 30) -> None:
    """Run test stream with specified resolutions.

    Args:
        camera_ip: IP address of the camera
        fps: Frame rate
    """
    print(f"\n{'=' * 60}")
    print("Test Stream: Depth/RGB at 1280x800, Stereo at 640x400")
    print(f"{'=' * 60}")
    print(f"Camera IP: {camera_ip}")
    print(f"FPS: {fps}")
    print("\nResolutions:")
    print("  - RGB: 1280x800")
    print("  - Depth: 1280x800")
    print("  - Stereo Left/Right: 640x400")
    print(f"{'=' * 60}\n")

    camera: LuxonisCameraSource | None = None

    try:
        # Create camera config
        # Stereo output (for SLAM) at 640x400
        # RGB-D output at 1280x800
        camera_config = LuxonisCameraConfig(
            ip=camera_ip,
            stereo=True,  # Required for RGB-D
            fps=fps,
            queue_size=8,
            queue_blocking=False,
            camera_mode="MONO",
            # Mono sensor resolution (for depth computation)
            # Use a resolution that supports both outputs
            mono_sensor_resolution=LuxonisResolution.from_dimensions(1280, 800),
            # SLAM stereo output resolution (left/right cameras)
            slam_output_resolution=LuxonisResolution.from_dimensions(640, 400),
            # RGB-D configuration
            enable_rgbd=True,
            rgb_output_resolution=LuxonisResolution.from_dimensions(1280, 800),
            depth_output_resolution=LuxonisResolution.from_dimensions(1280, 800),
            rgbd_sync=True,
            depth_preset=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
            depth_lr_check=True,
            depth_align_to_rgb=True,
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

        while not _shutdown:
            try:
                # Get RGB-D frames
                rgb_frame, depth_frame = camera.get_latest_rgbd_frames()
                rgb_img = rgb_frame.image
                depth_img = depth_frame.image  # Depth in millimeters (uint16)

                # Get stereo frames
                stereo_frames = camera.get_latest_frames()
                if len(stereo_frames) >= 2:
                    left_img = stereo_frames[0].image
                    right_img = stereo_frames[1].image
                else:
                    continue

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
                fps_rgbd = 0.0
                if len(frame_times_rgbd) >= 2:
                    fps_rgbd = (len(frame_times_rgbd) - 1) / (frame_times_rgbd[-1] - frame_times_rgbd[0])

                fps_stereo = 0.0
                if len(frame_times_stereo) >= 2:
                    fps_stereo = (len(frame_times_stereo) - 1) / (frame_times_stereo[-1] - frame_times_stereo[0])

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
                rgb_display = draw_info(rgb_img.copy(), fps_rgbd, "RGB", (rgb_img.shape[1], rgb_img.shape[0]))
                depth_display = draw_info(depth_colored, fps_rgbd, "Depth", (depth_img.shape[1], depth_img.shape[0]))
                left_display = draw_info(
                    left_display, fps_stereo, "Stereo Left", (left_img.shape[1], left_img.shape[0])
                )
                right_display = draw_info(
                    right_display, fps_stereo, "Stereo Right", (right_img.shape[1], right_img.shape[0])
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

                # Create 2x2 grid: top row (stereo left, stereo right), bottom row (RGB, depth)
                top_row = np.hstack([left_display, right_display])
                bottom_row = np.hstack([rgb_display, depth_display])
                combined = np.vstack([top_row, bottom_row])

                # Display
                cv2.imshow("Test Stream: Stereo (640x400) | RGB-D (1280x800)", combined)

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

        print("\n✓ Done")

    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
    finally:
        if camera:
            camera.stop()
        cv2.destroyAllWindows()


def main() -> None:
    """Main entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Test script to stream depth/RGB at 1280x800 and stereo at 640x400")
    parser.add_argument(
        "--ip",
        type=str,
        required=True,
        help="Camera IP address",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate (default: 30)",
    )

    args = parser.parse_args()

    try:
        run_test_stream(camera_ip=args.ip, fps=args.fps)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()
        sys.exit(0)


if __name__ == "__main__":
    main()
