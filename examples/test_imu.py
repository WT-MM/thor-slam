#!/usr/bin/env python3
"""Test script for IMU readings from a single Luxonis camera."""

import argparse
import sys
import time
from collections import deque

import depthai as dai
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from thor_slam.camera.drivers.luxonis import IMUData, LuxonisCameraConfig, LuxonisCameraSource, LuxonisResolution


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


def find_available_cameras() -> list[dai.DeviceInfo]:
    """Find all available cameras on the network."""
    return dai.Device.getAllAvailableDevices()


def interactive_camera_selection() -> str | None:
    """Interactively select a camera from available devices."""
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

    def add_data(self, imu_data: IMUData) -> None:
        """Add IMU data point to the plotter."""
        if self.start_time is None:
            self.start_time = imu_data.timestamp

        relative_time = imu_data.timestamp - self.start_time
        self.times.append(relative_time)
        self.accel_x.append(imu_data.accelerometer[0])
        self.accel_y.append(imu_data.accelerometer[1])
        self.accel_z.append(imu_data.accelerometer[2])
        self.gyro_x.append(imu_data.gyroscope[0])
        self.gyro_y.append(imu_data.gyroscope[1])
        self.gyro_z.append(imu_data.gyroscope[2])

    def update_plot(self, frame) -> tuple:
        """Update the plot with latest data."""
        if len(self.times) == 0:
            return (
                self.line_accel_x,
                self.line_accel_y,
                self.line_accel_z,
                self.line_gyro_x,
                self.line_gyro_y,
                self.line_gyro_z,
            )

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

        return (
            self.line_accel_x,
            self.line_accel_y,
            self.line_accel_z,
            self.line_gyro_x,
            self.line_gyro_y,
            self.line_gyro_z,
        )

    def show(self) -> None:
        """Show the plot window."""
        plt.tight_layout()
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

    def close(self) -> None:
        """Close the plot window."""
        plt.close(self.fig)


def run_with_plotting(
    camera_ip: str,
    resolution: str,
    fps: int,
    imu_rate: int,
    imu_raw: bool,
    batch_threshold: int,
    max_batch: int,
    stats_window: int,
) -> None:
    """Run IMU test with real-time plotting."""
    # Create camera configuration
    try:
        config = LuxonisCameraConfig(
            ip=camera_ip,
            stereo=True,  # Assume stereo for testing
            resolution=LuxonisResolution.from_name(resolution),
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
    print("\nStarting real-time plot. Close the plot window to stop...\n")

    # Initialize plotter
    plotter = IMUPlotter(max_points=stats_window)
    plotter.show()

    # Statistics tracking
    timestamps = deque(maxlen=stats_window)
    packet_count = 0
    total_imu_readings = 0
    start_time = time.time()
    last_stats_time = time.time()

    try:
        while plt.get_fignums():  # Continue while plot window is open
            # Get IMU data (non-blocking)
            sensor_data = camera.try_get_sensor_data()

            if sensor_data is None:
                plt.pause(0.001)  # Small pause to allow plot updates
                continue

            imu_data = sensor_data.get("imu")
            imu_packets = sensor_data.get("imu_packets")

            if imu_data is not None:
                # Add to plotter
                plotter.add_data(imu_data)

                # Update statistics
                timestamps.append(imu_data.timestamp)
                total_imu_readings += 1

            if imu_packets is not None:
                # Process batch of IMU packets
                for packet in imu_packets:
                    plotter.add_data(packet)
                    timestamps.append(packet.timestamp)
                    total_imu_readings += 1

                packet_count += 1

            # Update plot
            plotter.update_plot(None)
            plt.pause(0.01)  # Small pause to allow plot updates

            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= 2.0:  # Every 2 seconds
                last_stats_time = current_time
                elapsed = current_time - start_time
                rate = calculate_imu_rate(timestamps)

                print(f"IMU Rate: {rate:.1f} Hz | Total Readings: {total_imu_readings} | "
                      f"Packets: {packet_count} | Elapsed: {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        plotter.close()
        camera.stop()
        print("Camera stopped.")


def run_without_plotting(
    camera_ip: str,
    resolution: str,
    fps: int,
    imu_rate: int,
    imu_raw: bool,
    batch_threshold: int,
    max_batch: int,
    stats_window: int,
) -> None:
    """Run IMU test without plotting (text output only)."""
    # Create camera configuration
    try:
        config = LuxonisCameraConfig(
            ip=camera_ip,
            stereo=True,  # Assume stereo for testing
            resolution=LuxonisResolution.from_name(resolution),
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
            # Get IMU data (non-blocking)
            sensor_data = camera.try_get_sensor_data()

            if sensor_data is None:
                time.sleep(0.001)  # Small sleep to avoid busy waiting
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
        camera.stop()
        print("Camera stopped.")


def main() -> None:
    """Main function to test IMU readings."""
    parser = argparse.ArgumentParser(description="Test IMU readings from a Luxonis camera")
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="IP address of the camera (e.g., 192.168.2.21). If not provided, will use interactive selection.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="800",
        help="Camera resolution (default: 800)",
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

    args = parser.parse_args()

    # Select camera
    if args.ip:
        camera_ip = args.ip
        print(f"Using camera at IP: {camera_ip}")
    else:
        camera_ip = interactive_camera_selection()
        if camera_ip is None:
            print("No camera selected. Exiting.")
            sys.exit(0)

    # Run with or without plotting
    if args.no_plot:
        run_without_plotting(
            camera_ip=camera_ip,
            resolution=args.resolution,
            fps=args.fps,
            imu_rate=args.imu_rate,
            imu_raw=args.imu_raw,
            batch_threshold=args.batch_threshold,
            max_batch=args.max_batch,
            stats_window=args.stats_window,
        )
    else:
        run_with_plotting(
            camera_ip=camera_ip,
            resolution=args.resolution,
            fps=args.fps,
            imu_rate=args.imu_rate,
            imu_raw=args.imu_raw,
            batch_threshold=args.batch_threshold,
            max_batch=args.max_batch,
            stats_window=args.stats_window,
        )


if __name__ == "__main__":
    main()
