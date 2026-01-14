"""Publish IMU data from a Luxonis camera to /imu/data_raw at a fixed frequency."""

import argparse
import signal
import sys
import time
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from thor_slam.camera.drivers.luxonis import LuxonisCameraConfig, LuxonisCameraSource, LuxonisResolution


class IMUPublisher(Node):
    """ROS 2 node that publishes IMU data at a fixed frequency."""

    def __init__(self, camera: LuxonisCameraSource, publish_rate: float, frame_id: str = "imu_link"):
        """Initialize the IMU publisher node.

        Args:
            camera: Luxonis camera source with IMU enabled.
            publish_rate: Publishing frequency in Hz.
            frame_id: Frame ID for the IMU messages.
        """
        super().__init__("imu_publisher")
        self._camera = camera
        self._frame_id = frame_id
        self._latest_imu_data: Optional[dict] = None
        self._latest_timestamp: Optional[float] = None

        # Create publisher
        self._imu_pub = self.create_publisher(Imu, "/imu/data_raw", qos_profile_sensor_data)

        # Create timer for fixed-rate publishing
        timer_period = 1.0 / publish_rate
        self._timer = self.create_timer(timer_period, self._publish_imu_callback)

        self.get_logger().info(f"IMU publisher initialized: {publish_rate} Hz, frame_id: {frame_id}")

    def _publish_imu_callback(self) -> None:
        """Timer callback to publish IMU data at fixed rate."""
        # Try to get latest IMU data (non-blocking)
        result = self._camera.try_get_timestamped_sensor_data()
        
        # Handle return value (may be None or tuple due to bug in luxonis.py line 646)
        if result is None:
            # No data available, use latest if we have it
            pass
        elif isinstance(result, tuple):
            sensor_data, timestamp = result
            if sensor_data is not None and timestamp is not None:
                self._latest_imu_data = sensor_data
                self._latest_timestamp = timestamp
        else:
            # Unexpected return type (shouldn't happen, but handle gracefully)
            self.get_logger().warn(f"Unexpected return type from try_get_timestamped_sensor_data: {type(result)}")
            return

        # Publish latest available data
        if self._latest_imu_data is not None and self._latest_timestamp is not None:
            self._publish_imu(self._latest_imu_data, self._latest_timestamp)

    def _publish_imu(self, sensor_data: dict, timestamp: float) -> None:
        """Publish IMU data to ROS topic.

        Args:
            sensor_data: Dictionary containing IMU data with 'accelerometer' and 'gyroscope' keys.
            timestamp: Timestamp of the IMU reading in seconds.
        """
        # Create IMU message
        imu_msg = Imu()

        # Set header
        stamp = Time()
        stamp.sec = int(timestamp)
        stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = self._frame_id

        # Extract accelerometer and gyroscope data
        accel = sensor_data.get("accelerometer")
        gyro = sensor_data.get("gyroscope")

        if accel is not None and len(accel) >= 3:
            # Accelerometer data in m/s²
            imu_msg.linear_acceleration.x = float(accel[0])
            imu_msg.linear_acceleration.y = float(accel[1])
            imu_msg.linear_acceleration.z = float(accel[2])

        if gyro is not None and len(gyro) >= 3:
            # Gyroscope data in rad/s
            imu_msg.angular_velocity.x = float(gyro[0])
            imu_msg.angular_velocity.y = float(gyro[1])
            imu_msg.angular_velocity.z = float(gyro[2])

        # Set covariance matrices (unknown)
        # Using -1.0 to indicate unknown/not available
        imu_msg.linear_acceleration_covariance[0] = -1.0
        imu_msg.angular_velocity_covariance[0] = -1.0

        self._imu_pub.publish(imu_msg)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Publish IMU data from Luxonis camera to /imu/data_raw")
    parser.add_argument(
        "--ip",
        type=str,
        required=True,
        help="IP address of the camera (e.g., 192.168.2.21)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Publishing rate in Hz (default: 100.0)",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default="imu_link",
        help="Frame ID for IMU messages (default: imu_link)",
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

    args = parser.parse_args()

    # Create camera configuration
    try:
        config = LuxonisCameraConfig(
            ip=args.ip,
            stereo=False,  # Not needed for IMU only
            resolution=LuxonisResolution.from_name("800"),  # Default resolution
            fps=30,  # Not critical for IMU only
            read_imu=True,
            imu_report_rate=args.imu_rate,
            imu_raw=args.imu_raw,
        )
    except ValueError as e:
        print(f"Error creating camera config: {e}")
        sys.exit(1)

    # Create and start camera source
    print(f"\nInitializing camera at {args.ip}...")
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
        print("Error: Camera does not have IMU sensor data enabled!")
        camera.stop()
        sys.exit(1)

    # Initialize ROS 2
    rclpy.init()

    # Create publisher node
    node = IMUPublisher(camera, args.rate, args.frame_id)

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        node.destroy_node()
        rclpy.shutdown()
        camera.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"\nPublishing IMU data to /imu/data_raw at {args.rate} Hz")
    print("Press Ctrl+C to stop...\n")

    try:
        # Spin the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        node.destroy_node()
        rclpy.shutdown()
        camera.stop()
        print("Done.")


if __name__ == "__main__":
    main()

