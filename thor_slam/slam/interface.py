"""Interface for SLAM engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from types import TracebackType
from typing import Self

import numpy as np
from scipy.spatial.transform import Rotation

from thor_slam.camera.rig import RigCalibration
from thor_slam.camera.types import SynchronizedFrameSet


class TrackingState(Enum):
    """Tracking state of the SLAM system."""

    NOT_INITIALIZED = auto()  # System not yet initialized
    INITIALIZING = auto()  # Gathering initial frames for initialization
    TRACKING = auto()  # Normal tracking operation
    LOST = auto()  # Tracking lost, attempting recovery
    RELOCALIZING = auto()  # Attempting to relocalize in known map


@dataclass
class SlamPose:
    """Estimated camera/robot pose from SLAM.

    Attributes:
        position: Translation vector [x, y, z] in world frame (meters).
        rotation: Quaternion [qx, qy, qz, qw] representing orientation.
        timestamp: Timestamp of the pose estimate (seconds).
        tracking_state: Current tracking state.
        confidence: Confidence score in [0, 1], where 1 is highest confidence.
        covariance: Optional 6x6 pose covariance matrix (translation + rotation).
    """

    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [qx, qy, qz, qw]
    timestamp: float
    tracking_state: TrackingState = TrackingState.TRACKING
    confidence: float = 1.0
    covariance: np.ndarray | None = None  # 6x6 covariance matrix

    def to_4x4_matrix(self) -> np.ndarray:
        """Convert pose to 4x4 homogeneous transformation matrix.

        Returns:
            4x4 transformation matrix (world_T_camera).
        """
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(self.rotation).as_matrix()
        matrix[:3, 3] = self.position
        return matrix

    @classmethod
    def from_4x4_matrix(
        cls,
        matrix: np.ndarray,
        timestamp: float,
        tracking_state: TrackingState = TrackingState.TRACKING,
        confidence: float = 1.0,
    ) -> Self:
        """Create SlamPose from 4x4 transformation matrix.

        Args:
            matrix: 4x4 homogeneous transformation matrix.
            timestamp: Timestamp of the pose.
            tracking_state: Current tracking state.
            confidence: Confidence score.
        """
        rotation = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        position = matrix[:3, 3]
        return cls(
            position=position,
            rotation=rotation,
            timestamp=timestamp,
            tracking_state=tracking_state,
            confidence=confidence,
        )

    @classmethod
    def identity(cls, timestamp: float = 0.0) -> Self:
        """Create an identity pose (no translation or rotation)."""
        return cls(
            position=np.zeros(3),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),  # Identity quaternion
            timestamp=timestamp,
        )


@dataclass
class MapPoint:
    """A 3D point in the SLAM map.

    Attributes:
        position: 3D position [x, y, z] in world frame.
        color: Optional RGB color [r, g, b] in [0, 255].
        normal: Optional surface normal vector.
        observations: Number of times this point has been observed.
    """

    position: np.ndarray  # [x, y, z]
    color: np.ndarray | None = None  # [r, g, b]
    normal: np.ndarray | None = None  # [nx, ny, nz]
    observations: int = 1


@dataclass
class SlamMap:
    """Sparse or dense map from SLAM.

    Attributes:
        points: List of map points.
        keyframe_poses: Poses of keyframes in the map.
        timestamp: Timestamp when map was retrieved.
    """

    points: list[MapPoint] = field(default_factory=list)
    keyframe_poses: list[SlamPose] = field(default_factory=list)
    timestamp: float = 0.0

    def to_point_cloud(self) -> np.ndarray:
        """Convert map points to Nx3 numpy array."""
        if not self.points:
            return np.empty((0, 3))
        return np.array([p.position for p in self.points])


@dataclass
class SlamConfig:
    """Configuration for SLAM engines.

    This base config contains common parameters. Engine-specific configs
    should extend this class.
    """

    # Number of cameras (1 for mono, 2 for stereo, >2 for multi-cam)
    num_cameras: int = 2

    # Whether input images are already rectified
    rectified_images: bool = True

    # Enable loop closure / place recognition
    enable_loop_closure: bool = True

    # Enable mapping (vs pure visual odometry)
    enable_mapping: bool = True

    # Maximum number of map points to maintain
    max_map_size: int = 100000

    # Frame rate hint (helps with motion prediction)
    expected_fps: float = 30.0


class SlamEngine(ABC):
    """Abstract base class for SLAM engines.

    Supports context manager protocol:
        with MySlamEngine(config) as slam:
            pose = slam.process_frames(frame_set)
    """

    @abstractmethod
    def initialize(self, calibration: RigCalibration, config: SlamConfig | None = None) -> None:
        """Initialize the SLAM system with camera calibration.

        This must be called before process_frames(). Some engines may start
        internal threads or allocate GPU resources here.

        Args:
            calibration: Camera rig calibration data (intrinsics + extrinsics).
            config: Optional engine-specific configuration.

        Raises:
            RuntimeError: If initialization fails.
        """

    @abstractmethod
    def process_frames(self, frame_set: SynchronizedFrameSet) -> SlamPose | None:
        """Process a synchronized set of camera frames.

        Args:
            frame_set: Synchronized frames from all cameras.

        Returns:
            Estimated pose, or None if tracking failed.
        """

    @abstractmethod
    def get_tracking_state(self) -> TrackingState:
        """Get the current tracking state."""

    @abstractmethod
    def get_map(self) -> SlamMap:
        """Get the current sparse map.

        Returns:
            Current map with 3D points and keyframe poses.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the SLAM system.

        Clears the map and resets tracking state.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the SLAM system.

        Releases resources, stops threads, etc.
        """

    def save_map(self, path: str) -> bool:
        """Save the current map to disk.

        Args:
            path: Path to save the map.

        Returns:
            True if successful, False otherwise.
        """
        raise NotImplementedError("This SLAM engine does not support map saving")

    def load_map(self, path: str) -> bool:
        """Load a map from disk.

        Args:
            path: Path to load the map from.

        Returns:
            True if successful, False otherwise.
        """
        raise NotImplementedError("This SLAM engine does not support map loading")

    def relocalize(self) -> bool:
        """Attempt to relocalize in a previously loaded map.

        Returns:
            True if relocalization succeeded, False otherwise.
        """
        raise NotImplementedError("This SLAM engine does not support relocalization")

    # Context manager support
    def __enter__(self) -> Self:
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, shutting down the engine."""
        self.shutdown()
