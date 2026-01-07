#!/usr/bin/env python3
"""Script to compute and display depth maps from stereo vision using Luxonis cameras.

Uses LuxonisCameraSource to capture stereo frames and OpenCV StereoSGBM for depth computation.
"""

import argparse
import asyncio
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import depthai as dai
import numpy as np
from askin import KeyboardController

from thor_slam.camera.drivers.luxonis import LuxonisCameraConfig, LuxonisCameraSource, LuxonisResolution


@dataclass
class DepthConfig:
    """Configuration for stereo depth computation."""

    ip: str
    resolution: str = "1200"
    fps: int = 30
    # Stereo matching parameters
    num_disparities: int = 96  # Must be divisible by 16
    block_size: int = 5
    # Depth range in mm (for visualization)
    min_depth: int = 100
    max_depth: int = 10000
    # Colormap for visualization
    colormap: int = cv2.COLORMAP_JET
    # Rectification
    use_rectification: bool = True
    # Display size (max width for display window)
    display_width: int = 1280  # Resize images to this width for display


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


def get_rectification_maps(camera: LuxonisCameraSource) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Get stereo rectification maps from camera calibration.

    Returns:
        Tuple of (left_map1, left_map2, right_map1, right_map2) or None if failed
    """
    try:
        intrinsics = camera.get_intrinsics()
        if len(intrinsics) != 2:
            return None

        left_intrinsics = intrinsics[0]
        right_intrinsics = intrinsics[1]

        # Get extrinsics (left to right transformation)
        extrinsics = camera.get_extrinsics()
        if len(extrinsics) != 2:
            return None

        # Get the relative transformation between cameras
        # We have left-to-center and right-to-center, need left-to-right
        # For simplicity, use the calibration directly from the device
        calib = camera._calib_data

        # Get stereo rectification
        left_socket = dai.CameraBoardSocket.CAM_B
        right_socket = dai.CameraBoardSocket.CAM_C

        # Get intrinsic matrices
        left_k = np.array(
            calib.getCameraIntrinsics(
                left_socket,
                left_intrinsics.width,
                left_intrinsics.height,
            )
        )
        right_k = np.array(
            calib.getCameraIntrinsics(
                right_socket,
                right_intrinsics.width,
                right_intrinsics.height,
            )
        )

        # Get distortion coefficients
        left_d = np.array(calib.getDistortionCoefficients(left_socket))  # noqa: N806
        right_d = np.array(calib.getDistortionCoefficients(right_socket))  # noqa: N806

        # Get extrinsics (rotation and translation from left to right)
        extrinsic_matrix = np.array(calib.getCameraExtrinsics(left_socket, right_socket))
        r = extrinsic_matrix[:3, :3]  # noqa: N806
        t = extrinsic_matrix[:3, 3]  # noqa: N806

        img_size = (left_intrinsics.width, left_intrinsics.height)

        # Compute rectification transforms
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(  # noqa: N806
            left_k, left_d, right_k, right_d, img_size, r, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        # Compute rectification maps
        left_map1, left_map2 = cv2.initUndistortRectifyMap(left_k, left_d, r1, p1, img_size, cv2.CV_32FC1)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(right_k, right_d, r2, p2, img_size, cv2.CV_32FC1)

        return left_map1, left_map2, right_map1, right_map2

    except Exception as e:
        print(f"Warning: Could not compute rectification maps: {e}")
        return None


def compute_disparity(
    left_img: np.ndarray, right_img: np.ndarray, config: DepthConfig, stereo_matcher: cv2.StereoSGBM | None = None
) -> tuple[np.ndarray, cv2.StereoSGBM]:
    """Compute disparity map using OpenCV StereoSGBM.

    Args:
        left_img: Left camera image (grayscale or color)
        right_img: Right camera image (grayscale or color)
        config: Depth configuration
        stereo_matcher: Optional pre-created matcher (for reuse)

    Returns:
        Tuple of (disparity_map, stereo_matcher)
    """
    # Convert to grayscale if needed
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img

    if len(right_img.shape) == 3:
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right_img

    # Create stereo matcher if not provided
    if stereo_matcher is None:
        stereo_matcher = cv2.StereoSGBM_create(  # type: ignore[attr-defined]
            minDisparity=0,
            numDisparities=config.num_disparities,
            blockSize=config.block_size,
            uniquenessRatio=5,
            speckleWindowSize=200,
            speckleRange=2,
            disp12MaxDiff=0,
            P1=8 * 3 * config.block_size**2,
            P2=32 * 3 * config.block_size**2,
        )

    # Compute disparity
    disparity = stereo_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

    return disparity, stereo_matcher


def colorize_disparity(disparity: np.ndarray, colormap: int) -> np.ndarray:
    """Colorize a disparity map for visualization.

    Args:
        disparity: Disparity map
        colormap: OpenCV colormap constant

    Returns:
        Colorized image (BGR)
    """
    # Normalize disparity
    valid_mask = disparity > 0
    if valid_mask.any():
        min_val = disparity[valid_mask].min()
        max_val = disparity[valid_mask].max()
        if max_val > min_val:
            normalized = ((disparity - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(disparity, dtype=np.uint8)
    else:
        normalized = np.zeros_like(disparity, dtype=np.uint8)

    # Apply colormap
    colored = cv2.applyColorMap(normalized, colormap)

    # Set invalid regions to black
    colored[~valid_mask] = 0

    return colored


def draw_info(img: np.ndarray, fps: float, info: str = "") -> np.ndarray:
    """Draw info overlay on image."""
    img_copy = img.copy()
    cv2.putText(img_copy, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if info:
        cv2.putText(img_copy, info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return img_copy


def run_stereo_depth(config: DepthConfig) -> None:
    """Run stereo depth mapping using LuxonisCameraSource."""
    print(f"\n{'=' * 60}")
    print("Stereo Depth Mapping")
    print(f"Camera: {config.ip}")
    print(f"Resolution: {config.resolution}, FPS: {config.fps}")
    print(f"Rectification: {'Enabled' if config.use_rectification else 'Disabled'}")
    print(f"{'=' * 60}\n")

    camera: LuxonisCameraSource | None = None

    try:
        # Create camera config
        camera_config = LuxonisCameraConfig(
            ip=config.ip,
            stereo=True,
            resolution=LuxonisResolution.from_name(config.resolution),
            fps=config.fps,
            queue_size=8,
            queue_blocking=False,
        )

        # Create camera source
        camera = LuxonisCameraSource(camera_config)
        print(f"✓ Connected to {config.ip}")

        # Get rectification maps if enabled
        rect_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        if config.use_rectification:
            rect_maps = get_rectification_maps(camera)
            if rect_maps:
                print("✓ Stereo rectification enabled")
            else:
                print("⚠ Could not compute rectification maps, using unrectified images")

        # Log calibration info
        intrinsics = camera.get_intrinsics()
        if len(intrinsics) >= 2:
            print("\nLeft camera intrinsics:")
            print(f"  Focal length: fx={intrinsics[0].matrix[0, 0]:.1f}, fy={intrinsics[0].matrix[1, 1]:.1f}")
            print(f"  Principal point: cx={intrinsics[0].matrix[0, 2]:.1f}, cy={intrinsics[0].matrix[1, 2]:.1f}")

        # Start camera
        camera.start()
        print("✓ Camera started\n")

        quit_event, cleanup_quit = setup_quit_listener()
        print("Displaying depth map. Press 'q' to quit...\n")

        frame_times: list[float] = []
        stereo_matcher: cv2.StereoSGBM | None = None

        while not quit_event.is_set():
            try:
                # Get stereo frames
                frames = camera.get_latest_frames()

                if len(frames) != 2:
                    continue

                left_frame = frames[0]
                right_frame = frames[1]

                left_img = left_frame.image
                right_img = right_frame.image

                # Apply rectification if available
                if rect_maps is not None:
                    left_map1, left_map2, right_map1, right_map2 = rect_maps
                    left_img = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
                    right_img = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

                # Track FPS
                current_time = time.time()
                frame_times.append(current_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)

                fps = 0.0
                if len(frame_times) >= 2:
                    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

                # Compute disparity using OpenCV
                disparity, stereo_matcher = compute_disparity(left_img, right_img, config, stereo_matcher)

                # Colorize disparity
                disparity_colored = colorize_disparity(disparity, config.colormap)
                info_str = "Rectified" if rect_maps else "Unrectified"
                disparity_colored = draw_info(disparity_colored, fps, info_str)

                if len(left_img.shape) == 2:
                    left_display = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
                else:
                    left_display = left_img

                if disparity_colored.shape[:2] != left_display.shape[:2]:
                    disparity_colored = cv2.resize(disparity_colored, (left_display.shape[1], left_display.shape[0]))

                combined = np.hstack([left_display, disparity_colored])

                # Display
                cv2.imshow("Left | Disparity", combined)
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
    parser = argparse.ArgumentParser(description="Stereo depth mapping from Luxonis cameras")
    parser.add_argument(
        "--ip",
        type=str,
        help="Camera IP address (interactive selection if not provided)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1200",
        help="Resolution (default: 1200)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS (default: 30)",
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=96,
        help="Number of disparities (must be divisible by 16, default: 96)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
        help="Block size for SGBM (default: 5)",
    )
    parser.add_argument(
        "--no-rectify",
        action="store_true",
        help="Disable stereo rectification",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        choices=["jet", "hot", "bone", "rainbow", "turbo"],
        help="Colormap for disparity visualization (default: jet)",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Maximum width for display window (default: 1280)",
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
    }

    config = DepthConfig(
        ip=camera_ip,
        resolution=args.resolution,
        fps=args.fps,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        use_rectification=not args.no_rectify,
        colormap=colormap_map.get(args.colormap, cv2.COLORMAP_JET),
        display_width=args.display_width,
    )

    try:
        run_stereo_depth(config)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
