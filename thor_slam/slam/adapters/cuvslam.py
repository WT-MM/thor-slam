"""SLAM adapter for cuVSLAM."""

import numpy as np

from thor_slam.camera.rig import CameraRig
from thor_slam.slam.interface import SlamEngine, SlamPose


class CuVSLAMAdapter(SlamEngine):
    """SLAM adapter for cuVSLAM."""

    def __init__(self, camera_rig: CameraRig) -> None:
        """Initialize the SLAM engine."""
        self.camera_rig = camera_rig

    def initialize(self, calibration_file: str) -> None:
        """Initialize the SLAM engine."""
        raise NotImplementedError("Not implemented")

    def process_frames(self, frames: dict[str, np.ndarray], timestamp: float) -> SlamPose:
        """Process frames and return the pose."""
        raise NotImplementedError("Not implemented")
