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
    translation: np.ndarray  # 3x1

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
class CameraFrame:
    """Standardized output format."""

    image: np.ndarray
    timestamp: float
    sequence_num: int
    camera_name: str


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
