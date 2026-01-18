"""Test script for IMU readings from Luxonis camera(s).

Supports both single camera and synchronized multi-camera (CameraRig) modes.
"""

import argparse
import sys
import time
from collections import deque

import cv2  # type: ignore[import-untyped]
import depthai as dai
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from thor_slam.camera.drivers.luxonis import IMUData, LuxonisCameraConfig, LuxonisCameraSource, LuxonisResolution
from thor_slam.camera.rig import CameraRig


def find_available_cameras() -> list[dai.DeviceInfo]:
    """Find all available cameras on the network."""
    return dai.Device.getAllAvailableDevices()


def interactive_camera_selection_single() -> str | None:
    """Interactively select a single camera from available devices."""
    print("Finding available cameras...")
    devices = find_available_cameras()

    if len(devices) == 0:
        print("No cameras found on the network!")
        return None

    print(f"\nFound {len(devices)} camera(s):")
    for i, device_info in enumerate(devices):
        print(f"  [{i + 1}] IP: {device_info.name}, MXID: {device_info.deviceId}, State: {device_info.state}")

    # Select camera
    while True:
        try:
            choice = input(f"\nSelect camera (1-{len(devices)}, or 'q' to quit): ").strip()
            if choice.lower() == "q":
                return None

            index = int(choice) - 1
            if 0 <= index < len(devices):
                selected_device = devices[index]
                camera_ip = selected_device.name
                print(f"\nSelected camera: {camera_ip}")
                return camera_ip
            else:
                print(f"Invalid choice. Please enter a number between 1-{len(devices)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")


def interactive_camera_selection_multiple() -> list[str] | None:
    """Interactively select multiple cameras from available devices."""
    print("Finding available cameras...")
    devices = find_available_cameras()

    if len(devices) == 0:
        print("No cameras found on the network!")
        return None

    print(f"\nFound {len(devices)} camera(s):")
    for i, device_info in enumerate(devices):
        print(f"  [{i + 1}] IP: {device_info.name}, MXID: {device_info.deviceId}, State: {device_info.state}")

    # Select cameras
    while True:
        try:
            choice = input(
                f"\nSelect camera(s) (comma-separated numbers 1-{len(devices)}, e.g., '1,2,3' or 'q' to quit): "
            ).strip()
            if choice.lower() == "q":
                return None

            # Parse comma-separated indices
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            if all(0 <= idx < len(devices) for idx in indices):
                selected_ips = [devices[idx].name for idx in indices]
                print(f"\nSelected cameras: {', '.join(selected_ips)}")
                return selected_ips
            else:
                print(f"Invalid choice. Please enter numbers between 1-{len(devices)}")
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers (e.g., '1,2,3') or 'q'")


def format_imu_data(imu: IMUData) -> str:
    """Format IMU data for display."""
    accel = imu.accelerometer
    gyro = imu.gyroscope
    return (
        f"Accel: [{accel[0]:+7.3f}, {accel[1]:+7.3f}, {accel[2]:+7.3f}] m/s² | "
        f"Gyro: [{gyro[0]:+7.3f}, {gyro[1]:+7.3f}, {gyro[2]:+7.3f}] rad/s | "
        f"TS: {imu.timestamp:.6f} | Seq: {imu.sequence_num}"
    )


def calculate_imu_rate(timestamps: deque[float], window_size: int = 100) -> float:
    """Calculate IMU data rate from recent timestamps."""
    if len(timestamps) < 2:
        return 0.0
    recent = list(timestamps)[-window_size:]
    if len(recent) < 2:
        return 0.0
    time_diff = recent[-1] - recent[0]
    if time_diff <= 0:
        return 0.0
    return (len(recent) - 1) / time_diff


def calculate_stats(data_list: list[float]) -> dict[str, float]:
    """Calculate statistics for a list of values."""
    if not data_list:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    arr = np.array(data_list)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


class IMUPlotter:
    """Real-time IMU data plotter."""

    def __init__(self, max_points: int = 1000):
        """Initialize the plotter.

        Args:
            max_points: Maximum number of points to display in the plot.
        """
        self.max_points = max_points
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle("IMU Data - Real-time", fontsize=14, fontweight="bold")

        # Accelerometer plot
        self.ax_accel = self.axes[0]
        self.ax_accel.set_title("Accelerometer (m/s²)")
        self.ax_accel.set_ylabel("Acceleration (m/s²)")
        self.ax_accel.grid(True, alpha=0.3)

        # Gyroscope plot
        self.ax_gyro = self.axes[1]
        self.ax_gyro.set_title("Gyroscope (rad/s)")
        self.ax_gyro.set_xlabel("Time (seconds)")
        self.ax_gyro.set_ylabel("Angular Velocity (rad/s)")
        self.ax_gyro.grid(True, alpha=0.3)

        # Data storage
        self.times = deque(maxlen=max_points)
        self.accel_x = deque(maxlen=max_points)
        self.accel_y = deque(maxlen=max_points)
        self.accel_z = deque(maxlen=max_points)
        self.gyro_x = deque(maxlen=max_points)
        self.gyro_y = deque(maxlen=max_points)
        self.gyro_z = deque(maxlen=max_points)

        # Plot lines
        self.line_accel_x, = self.ax_accel.plot([], [], "r-", label="X", linewidth=1.5, alpha=0.8)
        self.line_accel_y, = self.ax_accel.plot([], [], "g-", label="Y", linewidth=1.5, alpha=0.8)
        self.line_accel_z, = self.ax_accel.plot([], [], "b-", label="Z", linewidth=1.5, alpha=0.8)
        self.line_gyro_x, = self.ax_gyro.plot([], [], "r-", label="X", linewidth=1.5, alpha=0.8)
        self.line_gyro_y, = self.ax_gyro.plot([], [], "g-", label="Y", linewidth=1.5, alpha=0.8)
        self.line_gyro_z, = self.ax_gyro.plot([], [], "b-", label="Z", linewidth=1.5, alpha=0.8)

        # Add legends
        self.ax_accel.legend(loc="upper right")
        self.ax_gyro.legend(loc="upper right")

        self.start_time = None

    def add_data(self, imu_data: IMUData, timestamp: float) -> None:
        """Add IMU data point to the plotter."""
        if self.start_time is None:
            self.start_time = timestamp

        relative_time = timestamp - self.start_time
        self.times.append(relative_time)
        self.accel_x.append(imu_data['accelerometer'][0])
        self.accel_y.append(imu_data['accelerometer'][1])
        self.accel_z.append(imu_data['accelerometer'][2])
        self.gyro_x.append(imu_data['gyroscope'][0])
        self.gyro_y.append(imu_data['gyroscope'][1])
        self.gyro_z.append(imu_data['gyroscope'][2])

    def update_plot(self) -> None:
        """Update the plot with latest data."""
        if len(self.times) == 0:
            return

        times_array = np.array(self.times)

        # Update accelerometer plot
        self.line_accel_x.set_data(times_array, np.array(self.accel_x))
        self.line_accel_y.set_data(times_array, np.array(self.accel_y))
        self.line_accel_z.set_data(times_array, np.array(self.accel_z))

        # Update gyroscope plot
        self.line_gyro_x.set_data(times_array, np.array(self.gyro_x))
        self.line_gyro_y.set_data(times_array, np.array(self.gyro_y))
        self.line_gyro_z.set_data(times_array, np.array(self.gyro_z))

        # Auto-scale axes
        if len(times_array) > 0:
            time_range = [max(0, times_array[-1] - 10), times_array[-1] + 1]

            # Accelerometer auto-scale
            all_accel = list(self.accel_x) + list(self.accel_y) + list(self.accel_z)
            if all_accel:
                accel_min, accel_max = min(all_accel), max(all_accel)
                accel_range = accel_max - accel_min
                margin = accel_range * 0.1 if accel_range > 0 else 1.0
                self.ax_accel.set_xlim(time_range)
                self.ax_accel.set_ylim(accel_min - margin, accel_max + margin)

            # Gyroscope auto-scale
            all_gyro = list(self.gyro_x) + list(self.gyro_y) + list(self.gyro_z)
            if all_gyro:
                gyro_min, gyro_max = min(all_gyro), max(all_gyro)
                gyro_range = gyro_max - gyro_min
                margin = gyro_range * 0.1 if gyro_range > 0 else 1.0
                self.ax_gyro.set_xlim(time_range)
                self.ax_gyro.set_ylim(gyro_min - margin, gyro_max + margin)

    def show(self) -> None:
        """Show the plot window."""
        plt.tight_layout()
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

    def close(self) -> None:
        """Close the plot window."""
        plt.close(self.fig)


def resize_for_display(img: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Resize image for display if it's too large."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def draw_text_on_image(img: np.ndarray, text: str, position: tuple[int, int] = (10, 30)) -> np.ndarray:
    """Draw text on image."""
    img_copy = img.copy()
    cv2.putText(img_copy, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img_copy


def run_single_camera(
    camera_ip: str,
    resolution: str,
    fps: int,
    imu_rate: int,
    imu_raw: bool,
    batch_threshold: int,
    max_batch: int,
    stats_window: int,
    plot: bool,
) -> None:
    """Run IMU test with a single camera."""
    # Create camera configuration
    try:
        config = LuxonisCameraConfig(
            ip=camera_ip,
            stereo=True,  # Assume stereo for testing
            mono_sensor_resolution=LuxonisResolution.from_name(resolution),
            fps=fps,
            read_imu=True,
            imu_report_rate=imu_rate,
            imu_batch_threshold=batch_threshold,
            imu_max_batch_reports=max_batch,
            imu_raw=imu_raw,
        )
    except ValueError as e:
        print(f"Error creating camera config: {e}")
        sys.exit(1)

    # Create and start camera source
    print("\nInitializing camera...")
    try:
        camera = LuxonisCameraSource(config)
        camera.start()
        print("Camera started successfully!")
    except Exception as e:
        print(f"Error starting camera: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if not camera.has_sensor_data:
        print("Warning: Camera does not have IMU sensor data enabled!")
        camera.stop()
        sys.exit(1)

    print("\nIMU Configuration:")
    print(f"  Report Rate: {imu_rate} Hz")
    print(f"  Batch Threshold: {batch_threshold}")
    print(f"  Max Batch Reports: {max_batch}")
    print(f"  Raw Data: {imu_raw}")

    # Initialize plotter if enabled
    plotter: IMUPlotter | None = None
    if plot:
        print("\nStarting real-time plot. Close the plot window to stop...\n")
        plotter = IMUPlotter(max_points=stats_window)
        plotter.show()
    else:
        print("\nPress Ctrl+C to stop...\n")

    # Statistics tracking
    timestamps = deque(maxlen=stats_window)
    accel_x = deque(maxlen=stats_window)
    accel_y = deque(maxlen=stats_window)
    accel_z = deque(maxlen=stats_window)
    gyro_x = deque(maxlen=stats_window)
    gyro_y = deque(maxlen=stats_window)
    gyro_z = deque(maxlen=stats_window)

    packet_count = 0
    total_imu_readings = 0
    start_time = time.time()
    last_stats_time = time.time()

    try:
        while True:
            # Check if plot window is still open
            if plot and plotter and not plt.get_fignums():
                break

            # Get IMU data (non-blocking)
            if plot:
                sensor_data, timestamp = camera.try_get_timestamped_sensor_data()
                if sensor_data is None:
                    plt.pause(0.001)
                    continue

                imu_data = sensor_data
                if imu_data is not None:
                    plotter.add_data(imu_data, timestamp)
                    plotter.update_plot()
                    plt.pause(0.01)
            else:
                sensor_data = camera.try_get_sensor_data()
                if sensor_data is None:
                    time.sleep(0.001)
                    continue

                imu_data = sensor_data.get("imu")
                imu_packets = sensor_data.get("imu_packets")

                if imu_data is not None:
                    # Process single IMU reading
                    timestamps.append(imu_data.timestamp)
                    accel_x.append(imu_data.accelerometer[0])
                    accel_y.append(imu_data.accelerometer[1])
                    accel_z.append(imu_data.accelerometer[2])
                    gyro_x.append(imu_data.gyroscope[0])
                    gyro_y.append(imu_data.gyroscope[1])
                    gyro_z.append(imu_data.gyroscope[2])
                    total_imu_readings += 1

                    # Display current reading
                    print(f"\r{format_imu_data(imu_data)}", end="", flush=True)

                if imu_packets is not None:
                    # Process batch of IMU packets
                    for packet in imu_packets:
                        timestamps.append(packet.timestamp)
                        accel_x.append(packet.accelerometer[0])
                        accel_y.append(packet.accelerometer[1])
                        accel_z.append(packet.accelerometer[2])
                        gyro_x.append(packet.gyroscope[0])
                        gyro_y.append(packet.gyroscope[1])
                        gyro_z.append(packet.gyroscope[2])
                        total_imu_readings += 1

                    packet_count += 1

            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= 2.0:  # Every 2 seconds
                last_stats_time = current_time
                elapsed = current_time - start_time
                rate = calculate_imu_rate(timestamps)

                if plot:
                    print(f"IMU Rate: {rate:.1f} Hz | Total Readings: {total_imu_readings} | "
                          f"Packets: {packet_count} | Elapsed: {elapsed:.1f}s")
                else:
                    print(f"\n\n--- Statistics (last {len(timestamps)} samples) ---")
                    print(f"IMU Rate: {rate:.1f} Hz")
                    print(f"Total Readings: {total_imu_readings}")
                    print(f"Total Packets: {packet_count}")
                    print(f"Elapsed Time: {elapsed:.1f}s")

                    if len(accel_x) > 0:
                        accel_x_stats = calculate_stats(list(accel_x))
                        accel_y_stats = calculate_stats(list(accel_y))
                        accel_z_stats = calculate_stats(list(accel_z))
                        gyro_x_stats = calculate_stats(list(gyro_x))
                        gyro_y_stats = calculate_stats(list(gyro_y))
                        gyro_z_stats = calculate_stats(list(gyro_z))

                        print("\nAccelerometer (m/s²):")
                        print(f"  X: min={accel_x_stats['min']:+.3f}, max={accel_x_stats['max']:+.3f}, "
                              f"mean={accel_x_stats['mean']:+.3f}, std={accel_x_stats['std']:.3f}")
                        print(f"  Y: min={accel_y_stats['min']:+.3f}, max={accel_y_stats['max']:+.3f}, "
                              f"mean={accel_y_stats['mean']:+.3f}, std={accel_y_stats['std']:.3f}")
                        print(f"  Z: min={accel_z_stats['min']:+.3f}, max={accel_z_stats['max']:+.3f}, "
                              f"mean={accel_z_stats['mean']:+.3f}, std={accel_z_stats['std']:.3f}")

                        print("\nGyroscope (rad/s):")
                        print(f"  X: min={gyro_x_stats['min']:+.3f}, max={gyro_x_stats['max']:+.3f}, "
                              f"mean={gyro_x_stats['mean']:+.3f}, std={gyro_x_stats['std']:.3f}")
                        print(f"  Y: min={gyro_y_stats['min']:+.3f}, max={gyro_y_stats['max']:+.3f}, "
                              f"mean={gyro_y_stats['mean']:+.3f}, std={gyro_y_stats['std']:.3f}")
                        print(f"  Z: min={gyro_z_stats['min']:+.3f}, max={gyro_z_stats['max']:+.3f}, "
                              f"mean={gyro_z_stats['mean']:+.3f}, std={gyro_z_stats['std']:.3f}")

                    print("\n" + "=" * 80)
                    print(f"\r{format_imu_data(imu_data) if imu_data else 'Waiting for IMU data...'}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        if plotter:
            plotter.close()
        camera.stop()
        print("Camera stopped.")


def run_synchronized(
    camera_ips: list[str],
    imu_source_ip: str | None,
    resolution: str,
    fps: int,
    imu_rate: int,
    duration: float,
    display: bool,
    plot: bool,
) -> None:
    """Run IMU test with synchronized multi-camera rig."""
    print(f"\n{'=' * 80}")
    print("Synchronized IMU Test")
    print(f"{'=' * 80}\n")

    # Create camera sources
    print("Creating camera sources...")
    sources = []
    imu_source_name = None

    for i, ip in enumerate(camera_ips):
        config = LuxonisCameraConfig(
            ip=ip,
            stereo=True,
            mono_sensor_resolution=LuxonisResolution.from_name(resolution),
            fps=fps,
            read_imu=(imu_source_ip is None and i == 0) or (imu_source_ip == ip),
            camera_mode="COLOR",
            imu_report_rate=imu_rate,
        )

        source = LuxonisCameraSource(config)
        sources.append(source)
        print(f"  ✓ Created source: {ip} (IMU: {'enabled' if source.has_sensor_data else 'disabled'})")

        # Determine IMU source name
        if imu_source_ip is None and i == 0 and source.has_sensor_data:
            imu_source_name = source.name
        elif imu_source_ip == ip and source.has_sensor_data:
            imu_source_name = source.name

    if imu_source_name is None:
        print("\nError: No camera with IMU enabled found!")
        print("Please ensure at least one camera has read_imu=True")
        return

    print(f"\nUsing '{imu_source_name}' as IMU source")

    # Create camera rig
    print("\nCreating CameraRig...")
    try:
        rig = CameraRig(
            sources=sources,
            queue_size=30,
            imu_source=imu_source_name,
        )
        print("  ✓ CameraRig created successfully")
    except Exception as e:
        print(f"  ✗ Error creating CameraRig: {e}")
        import traceback

        traceback.print_exc()
        return

    # Start rig
    print("\nStarting CameraRig...")
    try:
        rig.start()
        print("  ✓ CameraRig started")
    except Exception as e:
        print(f"  ✗ Error starting CameraRig: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\nStarting live streaming and IMU plotting...")
    print("Press 'q' in any window or Ctrl+C to stop\n")

    # Initialize plotter if enabled
    plotter: IMUPlotter | None = None
    if plot:
        plotter = IMUPlotter(max_points=1000)
        plotter.show()

    # Statistics
    frame_count = 0
    imu_count = 0
    imu_timestamp_deltas = []
    frame_timestamps = []
    imu_timestamps = []
    start_time = time.time()
    last_print_time = start_time

    try:
        while True:
            # Check if plot window is still open
            if plot and plotter and not plt.get_fignums():
                break

            sync_frames = rig.get_synchronized_frames()

            if sync_frames is None:
                if plot and plotter:
                    plt.pause(0.01)
                else:
                    time.sleep(0.01)
                continue

            frame_count += 1
            frame_timestamps.append(sync_frames.timestamp)

            # Display camera frames
            if display:
                for source_name, frame_set in sync_frames.frame_sets.items():
                    for i, frame in enumerate(frame_set.frames):
                        img = frame.image.copy()

                        # Convert grayscale to BGR for display
                        if len(img.shape) == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                        # Resize for display
                        img = resize_for_display(img, max_width=640)

                        # Add text overlay
                        camera_label = f"{source_name}_{i}"
                        timestamp_text = f"TS: {frame.timestamp:.3f}"
                        img = draw_text_on_image(img, camera_label, (10, 30))
                        img = draw_text_on_image(img, timestamp_text, (10, 60))

                        # Show frame
                        cv2.imshow(camera_label, img)

            # Check for IMU data
            if sync_frames.sensor_data is not None and sync_frames.sensor_timestamp is not None:
                imu_count += 1
                imu_timestamps.append(sync_frames.sensor_timestamp)

                # Calculate timestamp delta between frame and IMU
                timestamp_delta = abs(sync_frames.sensor_timestamp - sync_frames.timestamp)
                imu_timestamp_deltas.append(timestamp_delta)

                # Get IMU data (should be IMUData object directly)
                imu = sync_frames.sensor_data

                if imu and "accelerometer" in imu:
                    # Add to plotter
                    if plotter:
                        plotter.add_data(imu, sync_frames.sensor_timestamp)
                        plotter.update_plot()

                    # Print status periodically
                    current_time = time.time()
                    if current_time - last_print_time >= 2.0:
                        avg_delta = np.mean(imu_timestamp_deltas) if imu_timestamp_deltas else 0.0
                        max_delta = np.max(imu_timestamp_deltas) if imu_timestamp_deltas else 0.0
                        print(
                            f"Frames: {frame_count:5d} | IMU: {imu_count:5d} | "
                            f"Avg Δt: {avg_delta*1000:.2f}ms | Max Δt: {max_delta*1000:.2f}ms"
                        )
                        last_print_time = current_time

            # Update plots
            if plot and plotter:
                plt.pause(0.01)

            # Check for quit key
            if display:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Check duration
            if duration > 0 and (time.time() - start_time) >= duration:
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        # Cleanup display
        if display:
            cv2.destroyAllWindows()

        # Cleanup plotter
        if plotter:
            plotter.close()

        print("\n" + "=" * 80)
        print("Test Results")
        print("=" * 80)
        print(f"Total synchronized frames: {frame_count}")
        print(f"Frames with IMU data: {imu_count}")
        print(f"IMU coverage: {imu_count/frame_count*100:.1f}%" if frame_count > 0 else "N/A")

        if imu_timestamp_deltas:
            imu_timestamp_deltas = np.array(imu_timestamp_deltas)
            print("\nTimestamp Synchronization Statistics:")
            print(f"  Mean delta: {np.mean(imu_timestamp_deltas * 1000):.2f} ms")
            print(f"  Std dev:   {np.std(imu_timestamp_deltas)*1000:.2f} ms")
            print(f"  Min delta: {np.min(imu_timestamp_deltas)*1000:.2f} ms")
            print(f"  Max delta: {np.max(imu_timestamp_deltas)*1000:.2f} ms")
            print(f"  Median:    {np.median(imu_timestamp_deltas)*1000:.2f} ms")

        if frame_timestamps and imu_timestamps:
            elapsed = time.time() - start_time
            frame_rate = len(frame_timestamps) / elapsed if elapsed > 0 else 0
            imu_rate_actual = len(imu_timestamps) / elapsed if elapsed > 0 else 0
            print("\nData Rates:")
            print(f"  Frame rate: {frame_rate:.1f} Hz")
            print(f"  IMU rate:   {imu_rate_actual:.1f} Hz")

        print("\nStopping CameraRig...")
        rig.stop()
        print("  ✓ CameraRig stopped")
        print("\nTest completed!")


def main() -> None:
    """Main function to test IMU readings."""
    parser = argparse.ArgumentParser(description="Test IMU readings from Luxonis camera(s)")
    parser.add_argument(
        "--synchronized",
        action="store_true",
        help="Use synchronized multi-camera mode (CameraRig) instead of single camera",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="IP address of the camera (single camera mode). If not provided, will use interactive selection.",
    )
    parser.add_argument(
        "--ips",
        type=str,
        default=None,
        help="Comma-separated list of camera IP addresses (synchronized mode). If not provided, will use interactive selection.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720",
        help="Camera resolution (default: 720, works with OAK-D Pro W)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS (default: 30)",
    )
    parser.add_argument(
        "--imu-rate",
        type=int,
        default=400,
        help="IMU report rate in Hz (default: 400)",
    )
    parser.add_argument(
        "--imu-raw",
        action="store_true",
        help="Use raw IMU data instead of calibrated data",
    )
    parser.add_argument(
        "--batch-threshold",
        type=int,
        default=1,
        help="IMU batch report threshold (default: 1)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=10,
        help="Maximum IMU batch reports (default: 10)",
    )
    parser.add_argument(
        "--stats-window",
        type=int,
        default=1000,
        help="Number of samples to use for statistics (default: 1000)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting (use text output only)",
    )
    parser.add_argument(
        "--imu-source",
        type=str,
        default=None,
        help="IP address of the camera to use as IMU source (synchronized mode). If not provided, uses first camera.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Test duration in seconds (synchronized mode, default: 0.0 = run until stopped)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable camera frame display (synchronized mode)",
    )

    args = parser.parse_args()

    if args.synchronized:
        # Synchronized multi-camera mode
        if args.ips:
            camera_ips = [ip.strip() for ip in args.ips.split(",") if ip.strip()]
        else:
            camera_ips = interactive_camera_selection_multiple()
            if camera_ips is None or not camera_ips:
                print("No cameras selected. Exiting.")
                sys.exit(0)

        if not camera_ips:
            print("Error: No camera IPs provided!")
            sys.exit(1)

        run_synchronized(
            camera_ips=camera_ips,
            imu_source_ip=args.imu_source,
            resolution=args.resolution,
            fps=args.fps,
            imu_rate=args.imu_rate,
            duration=args.duration,
            display=not args.no_display,
            plot=not args.no_plot,
        )
    else:
        # Single camera mode
        if args.ip:
            camera_ip = args.ip
            print(f"Using camera at IP: {camera_ip}")
        else:
            camera_ip = interactive_camera_selection_single()
            if camera_ip is None:
                print("No camera selected. Exiting.")
                sys.exit(0)

        run_single_camera(
            camera_ip=camera_ip,
            resolution=args.resolution,
            fps=args.fps,
            imu_rate=args.imu_rate,
            imu_raw=args.imu_raw,
            batch_threshold=args.batch_threshold,
            max_batch=args.max_batch,
            stats_window=args.stats_window,
            plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
