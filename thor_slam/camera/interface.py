"""Interface for Luxonis cameras."""

import depthai as dai

from thor_slam.camera.types import IPv4
from thor_slam.camera.utils import get_device


class CameraInterface:
    """Interface for Luxonis cameras."""

    ip: IPv4
    device: dai.Device | None
    pipeline: dai.Pipeline

    def __init__(self, ip: str) -> None:
        """Initialize the camera interface."""
        self.ip = IPv4(ip)
        self.device = get_device(self.ip)

        if self.device is None:
            raise ValueError(f"Device with IP address {self.ip} not available.")

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> dai.Pipeline:
        # pipeline = dai.Pipeline(self.device)

        # sockets = self.device.getConnectedCameras()

        raise NotImplementedError("Not implemented")
