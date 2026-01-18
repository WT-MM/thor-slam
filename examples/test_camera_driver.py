"""Test script for Luxonis camera driver - tests both stereo and mono modes."""

import argparse
import asyncio
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


def find_available_cameras() -> list[dai.DeviceInfo]:
    """Find all available cameras on the network."""
    devices = dai.Device.getAllAvailableDevices()
    return devices


def draw_fps(image: cv2.Mat | np.ndarray, fps: float, label: str = "") -> cv2.Mat | np.ndarray:
    """Draw FPS text on image."""
    img_copy = image.copy()
    text = f"{label}FPS: {fps:.1f}" if label else f"FPS: {fps:.1f}"
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img_copy


def draw_fps_and_timestamp(
    image: cv2.Mat | np.ndarray, fps: float, timestamp: float, label: str = ""
) -> cv2.Mat | np.ndarray:
    """Draw FPS and timestamp text on image."""
    img_copy = image.copy()
    fps_text = f"{label}FPS: {fps:.1f}" if label else f"FPS: {fps:.1f}"
    # Format timestamp as seconds with 3 decimal places
    ts_text = f"TS: {timestamp:.3f}s"
    cv2.putText(img_copy, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_copy, ts_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
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


def setup_quit_listener() -> tuple[threading.Event, Callable[[], None]]:
    """Set up askin to listen for 'q' key press.

    Returns:
        A tuple of (quit_event, cleanup_function).
        The cleanup function should be called before exiting to properly stop the controller.
    """
    quit_event = threading.Event()
    controller: KeyboardController | None = None
    loop_ref: list[asyncio.AbstractEventLoop | None] = [None]
    thread: threading.Thread | None = None

    async def key_handler(key: str) -> None:
        if key == "q":
            quit_event.set()

    async def run_controller() -> None:
        nonlocal controller
        controller = KeyboardController(key_handler=key_handler, timeout=0.001)
        await controller.start()
        # Keep the event loop running until quit
        try:
            while not quit_event.is_set():
                await asyncio.sleep(0.1)
        finally:
            if controller:
                try:
                    await controller.stop()
                except Exception:
                    pass

    def run_async() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_ref[0] = loop
        try:
            loop.run_until_complete(run_controller())
        except Exception:
            pass
        finally:
            # Cancel all pending tasks before closing
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
        """Clean up the keyboard controller and event loop."""
        quit_event.set()  # Signal to stop
        # Give the thread a moment to clean up naturally
        if thread:
            thread.join(timeout=1.0)

    return quit_event, cleanup


def test_stereo_camera(ip: str, resolution_name: str = "1200", fps: int = 30) -> None:
    """Test stereo camera (left and right mono cameras)."""
    print(f"\n{'=' * 60}")
    print(f"Testing STEREO camera at {ip}")
    print(f"{'=' * 60}\n")

    try:
        # Create config for stereo camera
        config = LuxonisCameraConfig(
            ip=ip,
            stereo=True,
            mono_sensor_resolution=LuxonisResolution.from_name(resolution_name),
            fps=fps,
            queue_size=8,
            queue_blocking=False,
        )

        # Create camera source
        camera = LuxonisCameraSource(config)
        print("✓ Camera initialized successfully")
        print(f"  Resolution: {config.resolution.width}x{config.resolution.height}")
        print(f"  Target FPS: {fps}")

        # Log intrinsics and extrinsics
        print("\n--- Camera Calibration Data ---")
        intrinsics_list = camera.get_intrinsics()
        extrinsics_list = camera.get_extrinsics()

        print("Left Camera Intrinsics:")
        print(f"  Resolution: {intrinsics_list[0].width}x{intrinsics_list[0].height}")
        print(f"  Matrix:\n{intrinsics_list[0].matrix}")
        print(f"  Distortion Coeffs: {intrinsics_list[0].coeffs}")

        print("\nRight Camera Intrinsics:")
        print(f"  Resolution: {intrinsics_list[1].width}x{intrinsics_list[1].height}")
        print(f"  Matrix:\n{intrinsics_list[1].matrix}")
        print(f"  Distortion Coeffs: {intrinsics_list[1].coeffs}")

        print("\nLeft-to-Center Extrinsics:")
        print(f"  Rotation:\n{extrinsics_list[0].rotation}")
        print(f"  Translation (cm): {extrinsics_list[0].translation}")

        print("\nRight-to-Center Extrinsics:")
        print(f"  Rotation:\n{extrinsics_list[1].rotation}")
        print(f"  Translation (cm): {extrinsics_list[1].translation}")

        print("--- End Calibration Data ---\n")

        # Start camera
        camera.start()
        print("✓ Camera started")

        print("\nDisplaying stereo frames. Press 'q' to quit...\n")

        # Set up quit listener
        quit_event, cleanup_quit = setup_quit_listener()

        # FPS tracking
        left_frame_times: list[float] = []
        right_frame_times: list[float] = []
        frame_count = 0

        while not quit_event.is_set():
            try:
                # Get frames (blocking)
                frames = camera.get_latest_frames()

                if len(frames) == 2:
                    left_frame = frames[0]
                    right_frame = frames[1]

                    # Update FPS tracking
                    current_time = time.time()
                    left_frame_times.append(current_time)
                    right_frame_times.append(current_time)

                    # Keep only recent times (last 60 frames)
                    if len(left_frame_times) > 60:
                        left_frame_times.pop(0)
                        right_frame_times.pop(0)

                    # Calculate FPS
                    left_fps = calculate_fps(left_frame_times)
                    right_fps = calculate_fps(right_frame_times)

                    # Draw FPS and timestamp on images
                    left_img = draw_fps_and_timestamp(left_frame.image, left_fps, left_frame.timestamp, "Left - ")
                    right_img = draw_fps_and_timestamp(right_frame.image, right_fps, right_frame.timestamp, "Right - ")

                    # Display frames
                    cv2.imshow(f"Left Camera - {ip}", left_img)
                    cv2.imshow(f"Right Camera - {ip}", right_img)

                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"Frames: {frame_count} | Left FPS: {left_fps:.1f} | Right FPS: {right_fps:.1f}")

                # Process OpenCV window events (non-blocking)
                cv2.waitKey(1)

                # Check for quit from command line
                if quit_event.is_set():
                    break

            except KeyboardInterrupt:
                break

        # Cleanup
        cleanup_quit()
        camera.stop()
        cv2.destroyAllWindows()
        print(f"\n✓ Test completed. Total frames: {frame_count}")

    except Exception as e:
        print(f"✗ Error testing stereo camera: {e}")

        traceback.print_exc()
        raise


def test_mono_camera(ip: str, resolution_name: str = "1200", fps: int = 30) -> None:
    """Test mono/RGB camera."""
    print(f"\n{'=' * 60}")
    print(f"Testing MONO/RGB camera at {ip}")
    print(f"{'=' * 60}\n")

    try:
        # Create config for RGB camera
        config = LuxonisCameraConfig(
            ip=ip,
            stereo=False,
            mono_sensor_resolution=LuxonisResolution.from_name(resolution_name),
            fps=fps,
            queue_size=8,
            queue_blocking=False,
        )

        # Create camera source
        camera = LuxonisCameraSource(config)
        print("✓ Camera initialized successfully")
        print(f"  Resolution: {config.resolution.width}x{config.resolution.height}")
        print(f"  Target FPS: {fps}")

        # Log intrinsics and extrinsics
        print("\n--- Camera Calibration Data ---")
        intrinsics_list = camera.get_intrinsics()
        extrinsics_list = camera.get_extrinsics()

        print("RGB Camera Intrinsics:")
        print(f"  Resolution: {intrinsics_list[0].width}x{intrinsics_list[0].height}")
        print(f"  Matrix:\n{intrinsics_list[0].matrix}")
        print(f"  Distortion Coeffs: {intrinsics_list[0].coeffs}")

        print("\nExtrinsics (Identity):")
        print(f"  Rotation:\n{extrinsics_list[0].rotation}")
        print(f"  Translation: {extrinsics_list[0].translation}")

        print("--- End Calibration Data ---\n")

        # Start camera
        camera.start()
        print("✓ Camera started")

        print("\nDisplaying RGB frames. Press 'q' to quit...\n")

        # Set up quit listener
        quit_event, cleanup_quit = setup_quit_listener()

        # FPS tracking
        frame_times: list[float] = []
        frame_count = 0

        while not quit_event.is_set():
            try:
                # Get frames (blocking)
                frames = camera.get_latest_frames()

                if len(frames) == 1:
                    rgb_frame = frames[0]

                    # Update FPS tracking
                    current_time = time.time()
                    frame_times.append(current_time)

                    # Keep only recent times (last 60 frames)
                    if len(frame_times) > 60:
                        frame_times.pop(0)

                    # Calculate FPS
                    current_fps = calculate_fps(frame_times)

                    # Draw FPS and timestamp on image
                    rgb_img = draw_fps_and_timestamp(rgb_frame.image, current_fps, rgb_frame.timestamp)

                    # Display frame
                    cv2.imshow(f"RGB Camera - {ip}", rgb_img)

                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"Frames: {frame_count} | FPS: {current_fps:.1f}")

                # Process OpenCV window events (non-blocking)
                cv2.waitKey(1)

                # Check for quit from command line
                if quit_event.is_set():
                    break

            except KeyboardInterrupt:
                break

        # Cleanup
        cleanup_quit()
        camera.stop()
        cv2.destroyAllWindows()
        print(f"\n✓ Test completed. Total frames: {frame_count}")

    except Exception as e:
        print(f"✗ Error testing mono camera: {e}")

        traceback.print_exc()
        raise


def _process_frames(
    ip: str,
    frames: list,
    mode: str,
    frame_times: dict[str, list[float]],
    display_images: dict[str, cv2.Mat | np.ndarray],
) -> None:
    """Helper to process raw frames into display images."""
    current_time = time.time()

    if mode == "stereo" and len(frames) == 2:
        left_frame = frames[0]
        right_frame = frames[1]

        # Track FPS
        left_key = f"{ip}_left"
        right_key = f"{ip}_right"
        frame_times[left_key].append(current_time)
        frame_times[right_key].append(current_time)

        # Keep only recent times (last 30 frames)
        if len(frame_times[left_key]) > 30:
            frame_times[left_key].pop(0)
        if len(frame_times[right_key]) > 30:
            frame_times[right_key].pop(0)

        # Calculate FPS
        left_fps = calculate_fps(frame_times[left_key])
        right_fps = calculate_fps(frame_times[right_key])

        # Draw FPS and timestamp, then add to display
        display_images[left_key] = draw_fps_and_timestamp(
            left_frame.image, left_fps, left_frame.timestamp, f"{ip} L - "
        )
        display_images[right_key] = draw_fps_and_timestamp(
            right_frame.image, right_fps, right_frame.timestamp, f"{ip} R - "
        )

    elif mode == "mono" and len(frames) == 1:
        rgb_frame = frames[0]

        # Track FPS
        frame_times[ip].append(current_time)
        if len(frame_times[ip]) > 30:
            frame_times[ip].pop(0)

        # Calculate FPS
        current_fps = calculate_fps(frame_times[ip])

        # Draw FPS and timestamp, then add to display
        display_images[ip] = draw_fps_and_timestamp(rgb_frame.image, current_fps, rgb_frame.timestamp, f"{ip} - ")


def test_multiple_cameras(ips: list[str], mode: str, resolution_name: str = "1200", fps: int = 30) -> None:
    """Test multiple cameras simultaneously using master-slave synchronization."""
    print(f"\n{'=' * 60}")
    print(f"Testing {len(ips)} camera(s) in {mode.upper()} mode")
    print(f"{'=' * 60}\n")

    cameras: list[LuxonisCameraSource] = []
    frame_times: dict[str, list[float]] = defaultdict(list)

    try:
        # Initialize all cameras
        for ip in ips:
            print(f"Initializing camera {ip}...")
            config = LuxonisCameraConfig(
                ip=ip,
                stereo=(mode == "stereo"),
                mono_sensor_resolution=LuxonisResolution.from_name(resolution_name),
                fps=fps,
                queue_size=8,
                queue_blocking=False,
            )
            camera = LuxonisCameraSource(config)

            # Log intrinsics and extrinsics for this camera
            print(f"\n--- Camera {ip} Calibration Data ---")
            intrinsics_list = camera.get_intrinsics()
            extrinsics_list = camera.get_extrinsics()

            if mode == "stereo" and len(intrinsics_list) == 2:
                print("Left Camera Intrinsics:")
                print(f"  Resolution: {intrinsics_list[0].width}x{intrinsics_list[0].height}")
                print(f"  Matrix:\n{intrinsics_list[0].matrix}")
                print(f"  Distortion Coeffs: {intrinsics_list[0].coeffs}")

                print("\nRight Camera Intrinsics:")
                print(f"  Resolution: {intrinsics_list[1].width}x{intrinsics_list[1].height}")
                print(f"  Matrix:\n{intrinsics_list[1].matrix}")
                print(f"  Distortion Coeffs: {intrinsics_list[1].coeffs}")

                print("\nLeft-to-Center Extrinsics:")
                print(f"  Rotation:\n{extrinsics_list[0].rotation}")
                print(f"  Translation (cm): {extrinsics_list[0].translation}")

                print("\nRight-to-Center Extrinsics:")
                print(f"  Rotation:\n{extrinsics_list[1].rotation}")
                print(f"  Translation (cm): {extrinsics_list[1].translation}")
            else:
                print("RGB Camera Intrinsics:")
                print(f"  Resolution: {intrinsics_list[0].width}x{intrinsics_list[0].height}")
                print(f"  Matrix:\n{intrinsics_list[0].matrix}")
                print(f"  Distortion Coeffs: {intrinsics_list[0].coeffs}")

                print("\nExtrinsics (Identity):")
                print(f"  Rotation:\n{extrinsics_list[0].rotation}")
                print(f"  Translation: {extrinsics_list[0].translation}")

            print(f"--- End Camera {ip} Calibration Data ---\n")

            camera.start()
            cameras.append(camera)
            print(f"✓ Camera {ip} started")

        print(f"\nDisplaying frames from {len(cameras)} camera(s). Press 'q' to quit...\n")

        # Set up quit listener
        quit_event, cleanup_quit = setup_quit_listener()

        # Identify the "Master" camera (the first one) to pace the loop
        master_cam = cameras[0]
        slave_cams = cameras[1:]

        frame_count = 0

        while not quit_event.is_set():
            try:
                display_images: dict[str, cv2.Mat | np.ndarray] = {}

                # 1. BLOCK on the Master Camera
                # This ensures the loop runs at exactly the camera's FPS (e.g., 30Hz)
                # We use get_latest_frames (Blocking) instead of try_get
                try:
                    master_frames = master_cam.get_latest_frames()
                except RuntimeError:
                    # Occurs if camera stops
                    break

                # Process Master Frames
                _process_frames(master_cam.name, master_frames, mode, frame_times, display_images)

                # 2. POLL the Slave Cameras
                # Since we waited for Master, Slaves should likely have data ready.
                for cam in slave_cams:
                    # We still use try_get here to prevent hanging if a slave dies
                    slave_frames = cam.try_get_latest_frames()

                    if slave_frames:
                        _process_frames(cam.name, slave_frames, mode, frame_times, display_images)
                    # If slave has no frame, we just skip it for this iteration
                    # rather than dropping the Master's frame.

                # 3. Display whatever we got
                if display_images:
                    for window_name, img in display_images.items():
                        cv2.imshow(window_name, img)

                    frame_count += 1
                    if frame_count % 30 == 0:
                        # Calculate average FPS across all active streams
                        fps_list = [calculate_fps(times) for times in frame_times.values() if len(times) > 1]
                        if fps_list:
                            avg_fps = sum(fps_list) / len(fps_list)
                            print(f"Frames: {frame_count} | Avg FPS: {avg_fps:.1f}")

                # Process OpenCV window events (non-blocking)
                cv2.waitKey(1)

                # Check for quit from command line
                if quit_event.is_set():
                    break

            except KeyboardInterrupt:
                break

        # Cleanup
        for camera in cameras:
            camera.stop()
        cv2.destroyAllWindows()
        print(f"\n✓ Test completed. Total frames: {frame_count}")

    except Exception as e:
        print(f"✗ Error testing multiple cameras: {e}")

        traceback.print_exc()
        # Cleanup on error
        try:
            cleanup_quit()
        except Exception:
            pass
        for camera in cameras:
            try:
                camera.stop()
            except Exception:
                pass
        raise


def interactive_test() -> None:
    """Interactive mode to select camera(s) and test mode."""
    print("Finding available cameras...")
    devices = find_available_cameras()

    if len(devices) == 0:
        print("No cameras found!")
        return

    print(f"\nFound {len(devices)} camera(s):")
    for i, device_info in enumerate(devices):
        print(f"  [{i + 1}] IP: {device_info.name}, MXID: {device_info.deviceId}, State: {device_info.state}")

    # Select camera(s)
    while True:
        try:
            choice = input(
                f"\nSelect camera(s) (1-{len(devices)}, comma-separated for multiple, 'all' for all, or 'q' to quit): "
            ).strip()
            if choice.lower() == "q":
                return

            if choice.lower() == "all":
                selected_devices = devices
                break

            # Parse comma-separated indices
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            if all(0 <= idx < len(devices) for idx in indices):
                selected_devices = [devices[idx] for idx in indices]
                break
            else:
                print(f"Invalid choice. Please enter numbers between 1-{len(devices)}")
        except ValueError:
            print("Invalid input. Please enter numbers, 'all', or 'q'")

    ips = [device.name for device in selected_devices]
    print(f"\nSelected camera(s): {', '.join(ips)}")

    # Select mode
    while True:
        mode = input("Test mode - [s]tereo or [m]ono? (default: stereo): ").strip().lower()
        if mode in {"", "s"}:
            mode_str = "stereo"
            break
        elif mode == "m":
            mode_str = "mono"
            break
        else:
            print("Invalid choice. Enter 's' for stereo or 'm' for mono")

    # Select resolution
    resolution = input("Resolution (default: 1200): ").strip() or "1200"

    # Select FPS
    while True:
        try:
            fps_input = input("FPS (default: 30): ").strip()
            fps = int(fps_input) if fps_input else 30
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Run test
    if len(ips) == 1:
        if mode_str == "stereo":
            test_stereo_camera(ips[0], resolution, fps)
        else:
            test_mono_camera(ips[0], resolution, fps)
    else:
        test_multiple_cameras(ips, mode_str, resolution, fps)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Luxonis camera driver with stereo and mono modes")
    parser.add_argument(
        "--ip",
        type=str,
        help="IP address(es) of camera(s) to test (comma-separated, or 'all' for all cameras). "
        "If not provided, interactive mode will be used.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["stereo", "mono"], default="stereo", help="Camera mode to test (default: stereo)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720",
        help="Resolution name (default: 720, works with OAK-D Pro W). Options: 720, 800, 400, 480, 1200, 4000x3000, 4224x3136",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    args = parser.parse_args()

    if args.ip:
        # Parse IP addresses
        if args.ip.lower() == "all":
            # Get all available cameras
            devices = find_available_cameras()
            if len(devices) == 0:
                print("No cameras found!")
                return
            ips = [device.name for device in devices]
            print(f"Found {len(ips)} camera(s): {', '.join(ips)}")
        else:
            # Parse comma-separated IPs
            ips = [ip.strip() for ip in args.ip.split(",")]

        if len(ips) == 1:
            # Single camera - use simpler test functions
            if args.mode == "stereo":
                test_stereo_camera(ips[0], args.resolution, args.fps)
            else:
                test_mono_camera(ips[0], args.resolution, args.fps)
        else:
            # Multiple cameras
            test_multiple_cameras(ips, args.mode, args.resolution, args.fps)
    else:
        # Interactive mode
        interactive_test()


if __name__ == "__main__":
    main()
