"""Types for the camera package."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np

CameraSensorType = Literal["COLOR", "MONO"]


class IPv4(str):
    """Represents an IPv4 address."""

    _ip: str

    def __init__(self, ip: str) -> None:
        if not re.match(r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$", ip):
            raise ValueError(f"Invalid IPv4 address: {ip}")
        self._ip = ip

    def __str__(self) -> str:
        return self.ip

    @property
    def ip(self) -> str:
        return self._ip


@dataclass
class Intrinsics:
    """Intrinsics of a camera."""

    width: int
    height: int
    matrix: np.ndarray  # 3x3
    coeffs: np.ndarray  # distortion coefficients


@dataclass
class Extrinsics:
    """Extrinsics of a camera."""

    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3x1 (units in meters)

    @classmethod
    def from_4x4_matrix(cls, matrix: np.ndarray | list[list[float]]) -> Self:
        """Create Extrinsics from a 4x4 homogeneous transformation matrix."""
        matrix = np.array(matrix)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got shape {matrix.shape}")

        rotation = matrix[:3, :3]  # Top-left 3x3
        translation = matrix[:3, 3]  # Top-right 3x1 column

        return cls(rotation=rotation, translation=translation)

    def to_4x4_matrix(self) -> np.ndarray:
        """Convert Extrinsics to a 4x4 homogeneous transformation matrix.

        Returns:
            4x4 transformation matrix
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix


@dataclass
class IMUExtrinsics:
    """Dataclass to hold the IMU extrinsics and source name."""

    source_name: str
    extrinsics: Extrinsics

    def to_4x4_matrix(self) -> np.ndarray:
        """Convert IMUExtrinsics to a 4x4 homogeneous transformation matrix."""
        return self.extrinsics.to_4x4_matrix()


@dataclass
class CameraFrame:
    """Standardized output format."""

    image: np.ndarray
    timestamp: float
    sequence_num: int
    camera_name: str


class SensorData(ABC):
    """Abstract base class for sensor data."""

    @abstractmethod
    def get_timestamp(self) -> float:
        """Get the timestamp of the sensor data."""
        pass

    @abstractmethod
    def get_sequence_num(self) -> int:
        """Get the sequence number of the sensor data."""
        pass

    @abstractmethod
    def get_data(self) -> dict:
        """Get the data of the sensor data."""
        pass


class IMUData(SensorData):
    """IMU sensor data."""

    accelerometer: np.ndarray
    gyroscope: np.ndarray
    timestamp: float
    sequence_num: int

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_sequence_num(self) -> int:
        return self.sequence_num

    def get_data(self) -> dict:
        return {"accelerometer": self.accelerometer, "gyroscope": self.gyroscope}


class CameraSource(ABC):
    """Abstract base class for camera sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the camera source."""
        pass

    @abstractmethod
    def start(self) -> None:
        """Start the camera source."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera source."""
        pass

    @abstractmethod
    def get_latest_frames(self) -> list[CameraFrame]:
        """Get the latest frame from the camera source (blocking)."""
        pass

    @abstractmethod
    def try_get_latest_frames(self) -> list[CameraFrame] | None:
        """Try to get the latest frames without blocking.

        Returns None if no frames are available.
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> list[Intrinsics]:
        """Get the intrinsics of the camera source."""
        pass

    @abstractmethod
    def get_extrinsics(self) -> list[Extrinsics]:
        """Get the extrinsics of the camera source."""
        pass

    @abstractmethod
    def get_sensor_extrinsics(self) -> Extrinsics | None:
        """Get the extrinsics of a non-camera sensor relative to the CameraSource's reference frame.

        This method is for non-camera sensors (e.g., IMU). For camera sensors, use get_extrinsics().

        Returns:
            Extrinsics transformation from the sensor to the CameraSource's reference frame,
            or None if not available.
        """
        pass

    @abstractmethod
    def get_timestamped_sensor_data(self) -> tuple[dict | None, float | None]:
        """Get the (optional) sensor data of the camera source. E.g. imu data."""
        pass

    def try_get_timestamped_sensor_data(self) -> tuple[dict | None, float | None]:
        """Try to get sensor data without blocking.

        Default implementation calls get_timestamped_sensor_data(). Override in subclasses
        for non-blocking behavior.

        Returns:
            Tuple of (sensor data if available, timestamp if available), or (None, None) if no data available.
        """
        if not self.has_sensor_data:
            return None, None
        try:
            return self.get_timestamped_sensor_data()
        except Exception:
            return None, None

    @property
    @abstractmethod
    def has_sensor_data(self) -> bool:
        """Check if the camera source has sensor data."""
        pass

@dataclass
class FrameSet:
    """A set of frames from a single camera source.

    For stereo cameras, contains [left_frame, right_frame].
    For mono cameras, contains [rgb_frame].

    Each frame has its own timestamp (accessible via frames[i].timestamp).
    The `timestamp` field is a reference timestamp (typically from the first frame).
    The `sensor_data` field is an optional dictionary of sensor data. E.g. imu data.
    """

    timestamp: float  # Reference timestamp
    frames: list[CameraFrame]
    source_name: str
    sensor_data: dict | None = None
    sensor_timestamp: float | None = None

    @classmethod
    def from_frames(cls, frames: list[CameraFrame], source_name: str) -> Self:
        """Create a FrameSet from a list of frames."""
        if not frames:
            raise ValueError("Cannot create FrameSet from empty frame list")
        # Use the timestamp of the first frame as reference
        return cls(timestamp=frames[0].timestamp, frames=frames, source_name=source_name)

    def get_timestamps(self) -> list[float]:
        """Get timestamps for all frames in this set."""
        return [frame.timestamp for frame in self.frames]

    def get_max_timestamp(self) -> float:
        """Get the maximum (newest) timestamp in this set."""
        return max(frame.timestamp for frame in self.frames)

    def get_min_timestamp(self) -> float:
        """Get the minimum (oldest) timestamp in this set."""
        return min(frame.timestamp for frame in self.frames)

    def get_timestamp_spread(self) -> float:
        """Get the time difference between oldest and newest frames."""
        timestamps = self.get_timestamps()
        return max(timestamps) - min(timestamps)


@dataclass
class SynchronizedFrameSet:
    """A set of synchronized frames from multiple camera sources.

    All frames in this set are synchronized to the same reference timestamp.
    Individual frame timestamps are preserved and accessible via the FrameSet objects.
    """

    timestamp: float  # The reference timestamp (from the slowest camera)
    frame_sets: dict[str, FrameSet]  # source_name -> FrameSet
    max_time_delta: float  # Maximum time difference between any frame and reference
    sensor_data: dict | None = None
    sensor_timestamp: float | None = None

    def get_all_frames(self) -> list[CameraFrame]:
        """Get all frames from all sources as a flat list."""
        all_frames = []
        for frame_set in self.frame_sets.values():
            all_frames.extend(frame_set.frames)
        return all_frames

    def get_frames_for_source(self, source_name: str) -> list[CameraFrame] | None:
        """Get frames for a specific source."""
        if source_name in self.frame_sets:
            return self.frame_sets[source_name].frames
        return None

    def get_all_timestamps(self) -> dict[str, list[float]]:
        """Get timestamps for all frames, organized by source.

        Returns:
            Dictionary mapping source_name -> list of timestamps for each frame.
        """
        return {name: fs.get_timestamps() for name, fs in self.frame_sets.items()}

    def get_timestamp_for_frame(self, source_name: str, frame_index: int) -> float | None:
        """Get the timestamp for a specific frame.

        Args:
            source_name: Name of the camera source.
            frame_index: Index of the frame (0=left/first, 1=right/second, etc.).

        Returns:
            The frame's timestamp, or None if not found.
        """
        if source_name not in self.frame_sets:
            return None
        frames = self.frame_sets[source_name].frames
        if frame_index < 0 or frame_index >= len(frames):
            return None
        return frames[frame_index].timestamp
