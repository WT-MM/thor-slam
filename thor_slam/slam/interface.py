"""Interface for SLAM engines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SlamPose:
    """Standardized output format."""

    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [qx, qy, qz, qw]
    timestamp: float
    confidence: float


class SlamEngine(ABC):
    """Abstract base class for SLAM engines."""

    @abstractmethod
    def initialize(self, calibration_file: str) -> None:
        """Load calibration and setup the system."""
        pass

    @abstractmethod
    def process_frames(self, frames: dict[str, np.ndarray], timestamp: float) -> SlamPose:
        """Accepts a dictionary of images { 'cam_left': img, ... } and returns the calculated pose."""
        pass

    @abstractmethod
    def get_map_point_cloud(self) -> np.ndarray:
        """Return sparse or dense map points."""
        pass
