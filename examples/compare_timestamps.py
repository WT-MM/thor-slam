"""Script to run all cameras and compare timestamps to analyze synchronization."""

import argparse
import asyncio
import statistics
import sys
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable

import depthai as dai
from askin import KeyboardController

from thor_slam.camera.drivers.luxonis import LuxonisCameraConfig, LuxonisCameraSource, LuxonisResolution


def find_available_cameras() -> list[dai.DeviceInfo]:
    """Find all available cameras on the network."""
    devices = dai.Device.getAllAvailableDevices()
    return devices


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


def compare_timestamps(
    device_infos: list[dai.DeviceInfo],
    duration: float = 10.0,
    resolution_name: str = "1200",
    fps: int = 30,
    stereo: bool = True,
) -> None:
    """Run all cameras and compare timestamps.

    Args:
        device_infos: List of device info objects
        duration: How long to collect data (seconds)
        resolution_name: Resolution to use
        fps: FPS to use
        stereo: Whether to use stereo mode (default: True)
    """
    print(f"\n{'=' * 80}")
    print("Timestamp Comparison Test")
    print(f"{'=' * 80}\n")
    print(f"Testing {len(device_infos)} camera(s) for {duration} seconds")
    print(f"Mode: {'Stereo' if stereo else 'Mono'}, Resolution: {resolution_name}, FPS: {fps}\n")

    # Initialize all cameras
    camera_configs: dict[str, LuxonisCameraSource] = {}
    for device_info in device_infos:
        ip = device_info.name if hasattr(device_info, "name") else str(device_info)
        print(f"Initializing camera: {ip}...")

        try:
            config = LuxonisCameraConfig(
                ip=ip,
                stereo=stereo,
                resolution=LuxonisResolution.from_name(resolution_name),
                fps=fps,
                queue_size=8,
                queue_blocking=False,
            )
            camera = LuxonisCameraSource(config)
            camera.start()
            camera_configs[ip] = camera
            print(f"  ✓ Camera {ip} started")
        except Exception as e:
            print(f"  ✗ Failed to initialize {ip}: {e}")
            traceback.print_exc()
            continue

    if not camera_configs:
        print("No cameras could be initialized!")
        return

    print(f"\n✓ All {len(camera_configs)} camera(s) initialized\n")

    # Set up quit listener
    quit_event, cleanup_quit = setup_quit_listener()

    # Collect timestamp data
    timestamp_data: dict[str, list[float]] = defaultdict(list)
    sequence_data: dict[str, list[int]] = defaultdict(list)
    host_timestamps: dict[str, list[float]] = defaultdict(list)

    # Statistics
    frame_counts: dict[str, int] = defaultdict(int)
    start_time = time.time()
    last_print_time = start_time

    print("Collecting timestamp data... (Press 'q' to quit early)\n")

    # Use master-slave pattern: first camera paces the loop
    master_ip = list(camera_configs.keys())[0]
    master_cam = camera_configs[master_ip]
    slave_cams = {ip: cam for ip, cam in camera_configs.items() if ip != master_ip}

    try:
        while (time.time() - start_time) < duration and not quit_event.is_set():
            try:
                # Get master camera frames (blocking)
                master_frames = master_cam.get_latest_frames()
                host_time = time.time()

                # Process master frames
                for frame in master_frames:
                    camera_key = frame.camera_name
                    timestamp_data[camera_key].append(frame.timestamp)
                    sequence_data[camera_key].append(frame.sequence_num)
                    host_timestamps[camera_key].append(host_time)
                    frame_counts[camera_key] += 1

                # Get slave camera frames (non-blocking)
                for ip, cam in slave_cams.items():
                    slave_frames = cam.try_get_latest_frames()
                    if slave_frames:
                        host_time = time.time()
                        for frame in slave_frames:
                            camera_key = frame.camera_name
                            timestamp_data[camera_key].append(frame.timestamp)
                            sequence_data[camera_key].append(frame.sequence_num)
                            host_timestamps[camera_key].append(host_time)
                            frame_counts[camera_key] += 1

                # Print progress every second
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    elapsed = current_time - start_time
                    print(
                        f"  Elapsed: {elapsed:.1f}s / {duration:.1f}s | Frames collected: {sum(frame_counts.values())}",
                        end="\r",
                    )
                    last_print_time = current_time

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError during collection: {e}")
                traceback.print_exc()
                break

    finally:
        cleanup_quit()
        for camera in camera_configs.values():
            try:
                camera.stop()
            except Exception:
                pass

    print("\n\n" + "=" * 80)
    print("TIMESTAMP ANALYSIS")
    print("=" * 80 + "\n")

    if not timestamp_data:
        print("No timestamp data collected!")
        return

    # Print per-camera statistics
    print("Per-Camera Statistics:")
    print("-" * 80)
    for camera_name in sorted(timestamp_data.keys()):
        timestamps = timestamp_data[camera_name]
        sequences = sequence_data[camera_name]
        host_times = host_timestamps[camera_name]

        if len(timestamps) < 2:
            print(f"\n{camera_name}:")
            print(f"  Frames: {len(timestamps)} (insufficient data)")
            continue

        # Calculate frame intervals (device timestamps)
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        expected_interval = 1.0 / fps

        # Calculate host time intervals
        host_intervals = [host_times[i + 1] - host_times[i] for i in range(len(host_times) - 1)]

        # Calculate timestamp differences (device vs host)
        time_diffs = [host_times[i] - timestamps[i] for i in range(len(timestamps))]

        print(f"\n{camera_name}:")
        print(f"  Total frames: {len(timestamps)}")
        print(f"  Sequence range: {min(sequences)} - {max(sequences)}")
        print(f"  Timestamp range: {min(timestamps):.6f}s - {max(timestamps):.6f}s")
        print(f"  Duration: {max(timestamps) - min(timestamps):.6f}s")
        print("  Device frame interval:")
        print(f"    Mean: {statistics.mean(intervals):.6f}s (expected: {expected_interval:.6f}s)")
        print(f"    Std dev: {statistics.stdev(intervals) if len(intervals) > 1 else 0:.6f}s")
        print(f"    Min: {min(intervals):.6f}s, Max: {max(intervals):.6f}s")
        print("  Host frame interval:")
        print(f"    Mean: {statistics.mean(host_intervals):.6f}s")
        print(f"    Std dev: {statistics.stdev(host_intervals) if len(host_intervals) > 1 else 0:.6f}s")
        print("  Host-Device time difference:")
        print(f"    Mean: {statistics.mean(time_diffs):.6f}s")
        print(f"    Std dev: {statistics.stdev(time_diffs) if len(time_diffs) > 1 else 0:.6f}s")
        print(f"    Min: {min(time_diffs):.6f}s, Max: {max(time_diffs):.6f}s")

    # Cross-camera timestamp comparison
    print("\n" + "=" * 80)
    print("Cross-Camera Synchronization Analysis")
    print("=" * 80 + "\n")

    camera_names = sorted(timestamp_data.keys())

    if len(camera_names) < 2:
        print("Need at least 2 cameras for synchronization analysis")
        return

    # Find common time windows and calculate differences
    print("Timestamp differences between cameras:")
    print("-" * 80)

    for i, cam1 in enumerate(camera_names):
        for cam2 in camera_names[i + 1 :]:
            ts1 = timestamp_data[cam1]
            ts2 = timestamp_data[cam2]

            if not ts1 or not ts2:
                continue

            # Align by sequence numbers if possible, or by closest timestamps
            # For simplicity, compare timestamps directly
            min_len = min(len(ts1), len(ts2))
            if min_len < 2:
                continue

            # Compare first and last timestamps
            first_diff = ts1[0] - ts2[0]
            last_diff = ts1[-1] - ts2[-1]

            # Calculate average difference over overlapping period
            # Use the shorter sequence and compare element-wise
            diffs = [ts1[j] - ts2[j] for j in range(min_len)]
            mean_diff = statistics.mean(diffs)
            std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0

            print(f"\n{cam1} vs {cam2}:")
            print(f"  First frame diff: {first_diff:.6f}s")
            print(f"  Last frame diff: {last_diff:.6f}s")
            print(f"  Mean diff: {mean_diff:.6f}s")
            print(f"  Std dev: {std_diff:.6f}s")
            print(f"  Min diff: {min(diffs):.6f}s, Max diff: {max(diffs):.6f}s")

            # Calculate drift rate
            if len(ts1) > 1 and len(ts2) > 1:
                time_span = max(*ts1, *ts2) - min(*ts1, *ts2)
                if time_span > 0:
                    drift = (last_diff - first_diff) / time_span
                    print(f"  Drift rate: {drift * 1e6:.2f} μs/s ({drift * 1e3:.4f} ms/s)")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total frames collected: {sum(frame_counts.values())}")
    print(f"Collection duration: {time.time() - start_time:.2f}s")
    print(f"Average frame rate: {sum(frame_counts.values()) / (time.time() - start_time):.2f} fps")
    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare timestamps from all cameras")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration to collect data in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720",
        help="Resolution to use (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS to use (default: 30)",
    )
    parser.add_argument(
        "--ips",
        type=str,
        nargs="+",
        help="Specific camera IPs to test (default: all available)",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Use mono/RGB mode instead of stereo (default: stereo)",
    )

    args = parser.parse_args()

    try:
        # Find cameras
        if args.ips:
            devices = []
            all_devices = find_available_cameras()
            for ip in args.ips:
                matching = [d for d in all_devices if d.name == ip]
                if matching:
                    devices.extend(matching)
                else:
                    print(f"Warning: Camera {ip} not found")
            if not devices:
                print("No matching cameras found!")
                sys.exit(1)
        else:
            devices = find_available_cameras()
            if not devices:
                print("No cameras found!")
                sys.exit(1)

        print(f"Found {len(devices)} camera(s):")
        for device in devices:
            print(f"  - {device.name}")

        # Run comparison
        compare_timestamps(devices, args.duration, args.resolution, args.fps, stereo=not args.mono)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
