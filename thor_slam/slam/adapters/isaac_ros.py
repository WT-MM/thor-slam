"""SLAM adapter for Isaac ROS Visual SLAM.

Publishes to: /visual_slam/image_0..N, /visual_slam/camera_info_0..N
Subscribes to: /visual_slam/tracking/odometry
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import rclpy
from cv_bridge import CvBridge  # type: ignore[import-not-found]
from geometry_msgs.msg import TransformStamped  # type: ignore[import-untyped]
from nav_msgs.msg import Odometry  # type: ignore[import-untyped]
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image  # type: ignore[import-untyped]
from tf2_ros import StaticTransformBroadcaster  # type: ignore[import-untyped]

from thor_slam.camera.rig import RigCalibration
from thor_slam.camera.types import Extrinsics, SynchronizedFrameSet
from thor_slam.slam.interface import CameraConfig, SlamConfig, SlamEngine, SlamMap, SlamPose, TrackingState

logger = logging.getLogger(__name__)


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

        # ROS
        self._node: Node | None = None
        self._bridge: CvBridge | None = None
        self._tf_broadcaster: StaticTransformBroadcaster | None = None
        self._spin_thread: threading.Thread | None = None
        self._image_pubs: list = []
        self._info_pubs: list = []

        # Pose
        self._latest_pose: SlamPose | None = None
        self._pose_lock = threading.Lock()
        self._frame_count = 0

    def initialize(self, calibration: RigCalibration, config: SlamConfig | None = None) -> None:
        """Initialize with calibration data."""
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

        # Subscribe to odometry
        self._node.create_subscription(Odometry, "/visual_slam/tracking/odometry", self._odom_cb, 10)

        # Publish static TF
        self._publish_tf()

        # Start spin
        node = self._node
        self._spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
        self._spin_thread.start()

        self._tracking_state = TrackingState.INITIALIZING
        logger.info("Initialized with %d cameras", len(self._cameras))

    def _extract_cameras(self, cal: RigCalibration) -> list[CameraConfig]:
        """Extract flat list of camera configs from calibration."""
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
        """Publish static TF from base_link to each camera."""
        if not self._tf_broadcaster or not self._node:
            return

        stamp = self._node.get_clock().now().to_msg()
        transforms = []

        for i, cam in enumerate(self._cameras):
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

        self._tf_broadcaster.sendTransform(transforms)
        logger.info("Published TF: base_link -> camera_0..%d", len(transforms) - 1)

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

        # TODO: switch this back to frame_set timestamp. Check monotonicity
        stamp = self._node.get_clock().now().to_msg()

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

            img = frame.image
            if len(img.shape) == 2:
                img_msg = self._bridge.cv2_to_imgmsg(img, encoding="mono8")
            else:
                img_msg = self._bridge.cv2_to_imgmsg(img, encoding="bgr8")

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
