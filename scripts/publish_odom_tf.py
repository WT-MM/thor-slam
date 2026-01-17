"""Publish map->odom transform from visual SLAM odometry.

This script subscribes to /visual_slam/tracking/odometry and publishes
the transform from map to odom frame. This is needed for proper TF tree
connectivity in RViz and other ROS tools.
"""

import signal

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import TransformBroadcaster


class OdomTfPublisher(Node):
    """Publishes map->odom transform from odometry."""

    def __init__(self) -> None:
        """Initialize the odom TF publisher."""
        super().__init__("odom_tf_publisher")

        # Create transform broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to odometry
        self._odom_sub = self.create_subscription(Odometry, "/visual_slam/tracking/odometry", self._odom_callback, 10)

        self.get_logger().info("Odom TF publisher initialized")
        self.get_logger().info("  Subscribing to: /visual_slam/tracking/odometry")
        self.get_logger().info("  Publishing: map -> odom transform")

    def _odom_callback(self, msg: Odometry) -> None:
        """Handle odometry message and publish transform.

        The odometry from visual SLAM typically publishes the pose of base_link
        in the map frame. We need to publish map->odom transform.

        If odometry frame_id is "map", then:
        - The pose is base_link in map frame
        - We publish map->odom as the inverse of this pose (assuming odom starts at origin)

        If odometry frame_id is "odom", then:
        - The pose is base_link in odom frame
        - We publish map->odom as identity (or track separately)
        """
        # Extract pose from odometry
        pose = msg.pose.pose
        odom_frame = msg.header.frame_id

        # Create transform message
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        if odom_frame == "map":
            # Odometry is in map frame - pose is base_link in map
            # To get map->odom, we compute the inverse of base_link pose in map
            # (assuming odom frame starts at the same origin as base_link)

            # Get rotation matrix from quaternion
            q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            rot = Rotation.from_quat(q)
            rot_matrix = rot.as_matrix()

            # Position in map frame
            pos = [pose.position.x, pose.position.y, pose.position.z]

            # Transform: map->odom = inverse of base_link pose in map
            # If base_link in map is (R, t), then map->odom = (R^T, -R^T * t)
            pos_odom = -rot_matrix.T @ pos

            # Set translation
            t.transform.translation.x = float(pos_odom[0])
            t.transform.translation.y = float(pos_odom[1])
            t.transform.translation.z = float(pos_odom[2])

            # Set rotation (inverse quaternion)
            q_inv = rot.inv().as_quat()
            t.transform.rotation.x = float(q_inv[0])
            t.transform.rotation.y = float(q_inv[1])
            t.transform.rotation.z = float(q_inv[2])
            t.transform.rotation.w = float(q_inv[3])
        else:
            # Odometry is in odom frame - publish identity transform
            # (or could track separately if needed)
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

        # Publish transform
        self._tf_broadcaster.sendTransform(t)


# Global shutdown flag
_shutdown = False


def signal_handler(sig: int, frame: object) -> None:
    """Handle shutdown signal."""
    global _shutdown
    _shutdown = True
    print("\nShutting down...")


def main() -> None:
    """Main entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize ROS 2
    if not rclpy.ok():
        rclpy.init()

    # Create and run node
    node = OdomTfPublisher()

    print("Odom TF publisher running. Press Ctrl+C to stop.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
