"""Publish TF from odometry messages.

This script subscribes to an odometry topic and broadcasts the pose as a TF transform.
This is useful when Visual SLAM publishes Odometry but not TF directly.

Usage:
    python -m scripts.odom_to_tf --odom-topic /visual_slam/tracking/odometry --default-child-frame base_link
"""

import argparse
import sys
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdomToTF(Node):
    """Node that converts odometry messages to TF transforms."""

    def __init__(self, odom_topic: str, parent_frame: str, child_frame: str):
        """Initialize the odom-to-TF converter.

        Args:
            odom_topic: Topic name for odometry messages.
            parent_frame: Parent frame for the TF transform (e.g., 'odom').
            child_frame: Child frame for the TF transform (e.g., 'base_link').
        """
        super().__init__("odom_to_tf")
        self._parent_frame = parent_frame
        self._child_frame = child_frame
        self._tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(Odometry, odom_topic, self._odom_callback, 10)
        self.get_logger().info(
            f"Subscribing to {odom_topic} and broadcasting TF: {parent_frame} -> {child_frame}"
        )

    def _odom_callback(self, msg: Odometry) -> None:
        """Convert odometry message to TF transform."""
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self._parent_frame
        t.child_frame_id = self._child_frame

        # Copy pose from odometry
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation.x = msg.pose.pose.orientation.x
        t.transform.rotation.y = msg.pose.pose.orientation.y
        t.transform.rotation.z = msg.pose.pose.orientation.z
        t.transform.rotation.w = msg.pose.pose.orientation.w

        self._tf_broadcaster.sendTransform(t)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Convert odometry messages to TF transforms")
    parser.add_argument(
        "--odom-topic",
        type=str,
        default="/visual_slam/tracking/odometry",
        help="Odometry topic to subscribe to (default: /visual_slam/tracking/odometry)",
    )
    parser.add_argument(
        "--parent-frame",
        type=str,
        default="odom",
        help="Parent frame for TF transform (default: odom)",
    )
    parser.add_argument(
        "--default-child-frame",
        type=str,
        default="base_link",
        dest="child_frame",
        help="Child frame for TF transform (default: base_link)",
    )

    args = parser.parse_args()

    rclpy.init()
    node = OdomToTF(args.odom_topic, args.parent_frame, args.child_frame)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

