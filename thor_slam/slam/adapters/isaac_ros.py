"""SLAM adapter for Isaac ROS Visual SLAM.

Publishes to: /visual_slam/image_0..N, /visual_slam/camera_info_0..N, /visual_slam/imu
Subscribes to: /visual_slam/tracking/odometry
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_ros import StaticTransformBroadcaster

from thor_slam.camera.rig import RigCalibration
from thor_slam.camera.types import Extrinsics, SynchronizedFrameSet
from thor_slam.slam.interface import CameraConfig, SlamConfig, SlamEngine, SlamMap, SlamPose, TrackingState

logger = logging.getLogger(__name__)

# Coordinate frame transformation matrices
# Luxonis uses RDF (Right-Down-Forward): +x right, +y down, +z forward
# Isaac ROS base_link uses FLU (Forward-Left-Up): +x forward, +y left, +z up
# Camera optical frames need to be RDF: +z forward, +y down, +x right

# OAK-D Pro IMU uses ULB (Up-Left-Back)
# OAK-D Long Range IMU uses RDF (Right-Down-Forward)

# Transformation from RDF to FLU
RDF_TO_FLU_MATRIX = np.array(
    [
        [0, 0, 1, 0],  # x_flu = z_rdf (forward)
        [-1, 0, 0, 0],  # y_flu = -x_rdf (left)
        [0, -1, 0, 0],  # z_flu = -y_rdf (up)
        [0, 0, 0, 1],
    ]
)


@dataclass
class IsaacRosConfig(SlamConfig):
    """Configuration for Isaac ROS Visual SLAM adapter."""

    queue_size: int = 10


class IsaacRosAdapter(SlamEngine):
    """Adapter that bridges to Isaac ROS Visual SLAM."""

    def __init__(self, num_cameras: int = 2, config: IsaacRosConfig | None = None) -> None:
        self._num_cameras = num_cameras
        self._config = config or IsaacRosConfig(num_cameras=num_cameras)
        self._tracking_state = TrackingState.NOT_INITIALIZED

        # Extracted at init from RigCalibration
        self._cameras: list[CameraConfig] = []
        self._calibration: RigCalibration | None = None

        # ROS
        self._node: Node | None = None
        self._bridge: CvBridge | None = None
        self._tf_broadcaster: StaticTransformBroadcaster | None = None
        self._spin_thread: threading.Thread | None = None
        self._image_pubs: list = []
        self._info_pubs: list = []
        self._imu_pub: Publisher | None = None

        # Pose
        self._latest_pose: SlamPose | None = None
        self._pose_lock = threading.Lock()
        self._frame_count = 0

    def initialize(self, calibration: RigCalibration, config: SlamConfig | None = None) -> None:
        """Initialize with calibration data."""
        self._calibration = calibration
        self._cameras = self._extract_cameras(calibration)

        if len(self._cameras) < self._num_cameras:
            logger.warning(
                "Calibration has %d cameras, expected %d",
                len(self._cameras),
                self._num_cameras,
            )

        # Init ROS
        if not rclpy.ok():
            rclpy.init()

        self._node = Node("thor_slam_bridge")
        self._bridge = CvBridge()
        self._tf_broadcaster = StaticTransformBroadcaster(self._node)

        # Create publishers
        for i in range(self._num_cameras):
            self._image_pubs.append(
                self._node.create_publisher(Image, f"/visual_slam/image_{i}", self._config.queue_size)
            )
            self._info_pubs.append(
                self._node.create_publisher(CameraInfo, f"/visual_slam/camera_info_{i}", self._config.queue_size)
            )

        # Create IMU publisher with sensor QoS profile
        self._imu_pub = self._node.create_publisher(Imu, "/visual_slam/imu", qos_profile_sensor_data)

        # Subscribe to odometry
        self._node.create_subscription(Odometry, "/visual_slam/tracking/odometry", self._odom_cb, 10)

        # Publish static TF
        self._publish_tf()

        # Publish IMU link transform if IMU extrinsics are available
        if self._calibration.imu_extrinsics is not None:
            self._publish_imu_tf()
        else:
            logger.warning("No IMU extrinsics in calibration, skipping IMU TF publication")

        # Start spin
        node = self._node
        self._spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
        self._spin_thread.start()

        self._tracking_state = TrackingState.INITIALIZING
        logger.info("Initialized with %d cameras", len(self._cameras))
        logger.info("IMU publisher created at /visual_slam/imu")

    def _extract_cameras(self, cal: RigCalibration) -> list[CameraConfig]:
        """Extract flat list of camera configs from calibration.

        Transforms extrinsics from RDF (Luxonis) to ROS/cuVSLAM coordinate frame.
        """
        cameras: list[CameraConfig] = []
        for source_name in sorted(cal.intrinsics.keys()):
            intrinsics_list = cal.intrinsics[source_name]
            extrinsics_list = cal.get_world_extrinsics(source_name) or cal.extrinsics.get(source_name, [])

            for cam_idx, intr in enumerate(intrinsics_list):
                if len(cameras) >= self._num_cameras:
                    break
                extr = (
                    extrinsics_list[cam_idx] if cam_idx < len(extrinsics_list) else Extrinsics(np.eye(3), np.zeros(3))
                )

                cameras.append(CameraConfig(intr, extr, source_name, cam_idx))

        return cameras

    def _publish_tf(self) -> None:
        """Publish static TF from base_link to each camera and optical frames."""
        if not self._tf_broadcaster or not self._node:
            return

        stamp = self._node.get_clock().now().to_msg()
        transforms = []

        for i, cam in enumerate(self._cameras):
            # base_link -> camera_{i}
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "base_link"
            t.child_frame_id = f"camera_{i}"

            t.transform.translation.x = float(cam.extrinsics.translation[0])
            t.transform.translation.y = float(cam.extrinsics.translation[1])
            t.transform.translation.z = float(cam.extrinsics.translation[2])

            q = Rotation.from_matrix(cam.extrinsics.rotation).as_quat()
            t.transform.rotation.x = float(q[0])
            t.transform.rotation.y = float(q[1])
            t.transform.rotation.z = float(q[2])
            t.transform.rotation.w = float(q[3])

            transforms.append(t)

            # camera_{i} -> camera_{i}_optical_frame
            # Optical frame uses RDF convention: +x right, +y down, +z forward
            # Camera frame (from Isaac ROS) uses FLU: +x forward, +y left, +z up
            # Transform: FLU -> RDF (inverse of RDF_TO_FLU_MATRIX)
            t_optical = TransformStamped()
            t_optical.header.stamp = stamp
            t_optical.header.frame_id = f"camera_{i}"
            t_optical.child_frame_id = f"camera_{i}_optical_frame"

            # Identity translation (optical frame is at same position as camera frame)
            t_optical.transform.translation.x = 0.0
            t_optical.transform.translation.y = 0.0
            t_optical.transform.translation.z = 0.0

            # FLU -> RDF transformation matrix (inverse of RDF_TO_FLU_MATRIX)
            # FLU: +x forward, +y left, +z up
            # RDF: +x right, +y down, +z forward
            # x_rdf = -y_flu, y_rdf = -z_flu, z_rdf = x_flu
            flu_to_rdf_matrix = np.array(
                [
                    [0, -1, 0, 0],  # x_rdf = -y_flu
                    [0, 0, -1, 0],  # y_rdf = -z_flu
                    [1, 0, 0, 0],  # z_rdf = x_flu
                    [0, 0, 0, 1],
                ]
            )
            q_optical = Rotation.from_matrix(flu_to_rdf_matrix[:3, :3]).as_quat()
            t_optical.transform.rotation.x = float(q_optical[0])
            t_optical.transform.rotation.y = float(q_optical[1])
            t_optical.transform.rotation.z = float(q_optical[2])
            t_optical.transform.rotation.w = float(q_optical[3])

            transforms.append(t_optical)

        self._tf_broadcaster.sendTransform(transforms)
        logger.info(
            "Published TF: base_link -> camera_0..%d, camera_0..%d -> camera_0..%d_optical_frame",
            len(self._cameras) - 1,
            len(self._cameras) - 1,
            len(self._cameras) - 1,
        )

    def _publish_imu_tf(self) -> None:
        """Publish static TF from base_link to imu_link."""
        if not self._tf_broadcaster or not self._node or not self._calibration:
            return

        if self._calibration.imu_extrinsics is None:
            logger.warning("No IMU extrinsics in calibration, skipping IMU TF publication")
            return

        # Get IMU extrinsics (already in base_link/world frame)
        imu_extrinsics = self._calibration.imu_extrinsics.extrinsics

        # Create transform message
        stamp = self._node.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "base_link"
        t.child_frame_id = "imu_link"

        # Set translation
        t.transform.translation.x = float(imu_extrinsics.translation[0])
        t.transform.translation.y = float(imu_extrinsics.translation[1])
        t.transform.translation.z = float(imu_extrinsics.translation[2])

        # Set rotation
        q = Rotation.from_matrix(imu_extrinsics.rotation).as_quat()
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        # Publish transform
        self._tf_broadcaster.sendTransform([t])
        logger.info("Published TF: base_link -> imu_link")

    def _publish_imu(self, sensor_data: dict, timestamp: Time) -> None:
        """Publish IMU data to ROS topic.

        Args:
            sensor_data: Dictionary containing 'accelerometer' and 'gyroscope' keys with numpy arrays.
            timestamp: Time message for the IMU message header.
        """
        if not self._node or not self._imu_pub:
            return

        # Create IMU message
        imu_msg = Imu()

        # Set header
        imu_msg.header.stamp = timestamp
        imu_msg.header.frame_id = "base_link"

        # Extract accelerometer and gyroscope data
        accel = sensor_data.get("accelerometer")
        gyro = sensor_data.get("gyroscope")

        if accel is not None and len(accel) >= 3:
            # Accelerometer data in m/s²
            # Note: IMU data from Luxonis is in RDF frame, but we need to transform it
            # to match the camera optical frame (RDF) or base_link (FLU) as needed
            # For now, we'll publish as-is and let the coordinate frame transformation
            # be handled by the frame_id convention
            imu_msg.linear_acceleration.x = float(accel[0])
            imu_msg.linear_acceleration.y = float(accel[1])
            imu_msg.linear_acceleration.z = float(accel[2])

        if gyro is not None and len(gyro) >= 3:
            # Gyroscope data in rad/s
            imu_msg.angular_velocity.x = float(gyro[0])
            imu_msg.angular_velocity.y = float(gyro[1])
            imu_msg.angular_velocity.z = float(gyro[2])

        # Set covariance matrices (unknown for now)
        # Diagonal covariance: [x, y, z, roll, pitch, yaw]
        # Using large values to indicate uncertainty
        imu_msg.linear_acceleration_covariance[0] = -1.0  # Unknown
        imu_msg.angular_velocity_covariance[0] = -1.0  # Unknown

        self._imu_pub.publish(imu_msg)

    def _odom_cb(self, msg: Odometry) -> None:
        """Handle odometry."""
        p = msg.pose.pose
        cov = np.array(msg.pose.covariance).reshape(6, 6)
        conf = max(0.0, min(1.0, 1.0 / (1.0 + np.trace(cov[:3, :3]))))

        with self._pose_lock:
            self._latest_pose = SlamPose(
                position=np.array([p.position.x, p.position.y, p.position.z]),
                rotation=np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]),
                timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                tracking_state=TrackingState.TRACKING,
                confidence=conf,
                covariance=cov,
            )
            if self._tracking_state == TrackingState.INITIALIZING:
                self._tracking_state = TrackingState.TRACKING
                logger.info("Tracking started")

    def process_frames(self, frame_set: SynchronizedFrameSet) -> SlamPose | None:
        """Publish frames, return latest pose."""
        if not self._node or not self._bridge:
            raise RuntimeError("Not initialized")

        self._frame_count += 1

        # Map frames to global camera indices
        published = 0
        for i, cam in enumerate(self._cameras):
            if cam.source_name not in frame_set.frame_sets:
                continue
            fs = frame_set.frame_sets[cam.source_name]
            if cam.cam_idx >= len(fs.frames):
                continue

            frame = fs.frames[cam.cam_idx]
            frame_id = f"camera_{i}"

            stamp = Time()

            stamp.sec = int(frame.timestamp / 1)
            stamp.nanosec = int((frame.timestamp / 1 - int(frame.timestamp / 1)) * 1e9)

            img = frame.image
            if len(img.shape) == 2:
                img_msg = self._bridge.cv2_to_imgmsg(img, encoding="mono8")
            else:
                # Convert BGR to RGB for Isaac ROS (expects rgb8, not bgr8)
                # NOTE: Have image encoding information somewhere  within the frameset.
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_msg = self._bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")

            img_msg.header.stamp = stamp
            img_msg.header.frame_id = frame_id
            self._image_pubs[i].publish(img_msg)

            # Publish camera info
            info = CameraInfo()
            info.header.stamp = stamp
            info.header.frame_id = frame_id
            info.width = cam.intrinsics.width
            info.height = cam.intrinsics.height
            d = cam.intrinsics.coeffs.flatten().tolist()

            if len(d) >= 8:
                info.distortion_model = "rational_polynomial"
                info.d = d[:8]  # k1 k2 p1 p2 k3 k4 k5 k6
            elif len(d) == 5:
                info.distortion_model = "plumb_bob"
                info.d = d
            elif len(d) == 4:
                info.distortion_model = "equidistant"
                info.d = d
            else:
                info.distortion_model = "plumb_bob"
                info.d = (d + [0, 0, 0, 0, 0])[:5]
            info.k = cam.intrinsics.matrix.flatten().tolist()
            info.r = np.eye(3).flatten().tolist()

            # Build projection matrix P
            # For right cameras in stereo pairs, P[0,3] = -fx * baseline
            p = np.zeros((3, 4))
            p[:3, :3] = cam.intrinsics.matrix

            # Right camera in stereo pair (cam_idx=1)
            if cam.cam_idx == 1 and i > 0 and self._cameras[i - 1].source_name == cam.source_name:
                left_cam = self._cameras[i - 1]

                # Compute baseline in left camera frame: T_lr = inv(T_l) * T_r
                rot_l = left_cam.extrinsics.rotation
                t_l, t_r = left_cam.extrinsics.translation, cam.extrinsics.translation

                # TODO: check if this is correct.
                t_lr = rot_l.T @ (t_r - t_l)  # Transform to left camera frame

                baseline = float(t_lr[0])  # x component in left camera frame
                fx = float(cam.intrinsics.matrix[0, 0])
                p[0, 3] = -fx * baseline  # ROS stereo convention: negative

                if self._frame_count == 1:
                    logger.info("Camera %d baseline: %.4f m, Tx: %.2f", i, baseline, p[0, 3])

            info.p = p.flatten().tolist()
            self._info_pubs[i].publish(info)

            published += 1

        # Publish IMU data if available
        if frame_set.sensor_data is not None and self._imu_pub is not None:
            # Use sensor timestamp if available, otherwise use frame timestamp
            imu_timestamp = frame_set.sensor_timestamp
            if imu_timestamp is not None:
                # Convert float timestamp to Time message
                time_msg = Time()
                time_msg.sec = int(imu_timestamp)
                time_msg.nanosec = int((imu_timestamp - time_msg.sec) * 1e9)
            else:
                # Fall back to frame timestamp
                time_msg = stamp
            self._publish_imu(frame_set.sensor_data, time_msg)

        with self._pose_lock:
            return self._latest_pose

    def get_tracking_state(self) -> TrackingState:
        return self._tracking_state

    def get_map(self) -> SlamMap:
        return SlamMap()

    def reset(self) -> None:
        with self._pose_lock:
            self._latest_pose = None
        self._tracking_state = TrackingState.INITIALIZING
        self._frame_count = 0

    def shutdown(self) -> None:
        if self._node:
            self._node.destroy_node()
            self._node = None
        if rclpy.ok():
            rclpy.shutdown()
        self._tracking_state = TrackingState.NOT_INITIALIZED

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def num_cameras(self) -> int:
        return self._num_cameras
