"""Coordinating system for multiple cameras."""

import numpy as np

from thor_slam.camera.types import CameraSource


class CameraRig:
    """Coordinating system for multiple cameras."""

    cameras: list[CameraSource]

    def __init__(self, cameras: list[CameraSource]) -> None:
        """Initialize the camera rig."""
        self.cameras = cameras

    def get_synchronized_frames(self) -> dict[str, np.ndarray]:
        """Get synchronized frames from all cameras."""
        raise NotImplementedError("Not implemented")

    def close(self) -> None:
        """Close the cameras."""
        raise NotImplementedError("Not implemented")
