"""Driver for Luxonis cameras."""

from dataclasses import dataclass
from typing import Self

import depthai as dai
import numpy as np

from thor_slam.camera.types import CameraFrame, CameraSensorType, CameraSource, Extrinsics, Intrinsics, IPv4
from thor_slam.camera.utils import (
    get_luxonis_camera_valid_modes,
    get_luxonis_camera_valid_resolutions,
    get_luxonis_device,
)

# Supported resolutions as (width, height) tuples
SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720": (1280, 720),
    "800": (1280, 800),
    "400": (640, 400),
    "480": (640, 480),
    "1200": (1920, 1200),
    "4000x3000": (4000, 3000),
    "4224x3136": (4224, 3136),
}

camera_sensor_type_to_dai: dict[CameraSensorType, dai.CameraSensorType] = {
    "COLOR": dai.CameraSensorType.COLOR,
    "MONO": dai.CameraSensorType.MONO,
}

dai_to_camera_sensor_type: dict[dai.CameraSensorType, CameraSensorType] = {
    dai.CameraSensorType.COLOR: "COLOR",
    dai.CameraSensorType.MONO: "MONO",
}


@dataclass
class LuxonisResolution:
    """Resolution for Luxonis cameras."""

    width: int
    height: int

    @classmethod
    def from_dimensions(cls, width: int, height: int) -> Self:
        """Create a LuxonisResolution from width and height."""
        return cls(width=width, height=height)

    @classmethod
    def from_name(cls, name: str) -> Self:
        """Create a LuxonisResolution from a simple name."""
        # Try exact match first
        if name in SUPPORTED_RESOLUTIONS:
            width, height = SUPPORTED_RESOLUTIONS[name]
            return cls(width=width, height=height)

        # Try case-insensitive match
        name_lower = name.lower()
        for key, dims in SUPPORTED_RESOLUTIONS.items():
            if key.lower() == name_lower:
                return cls(width=dims[0], height=dims[1])

        raise ValueError(f"Unknown resolution name: {name}. Supported names: {sorted(SUPPORTED_RESOLUTIONS.keys())}")

    def as_tuple(self) -> tuple[int, int]:
        """Return resolution as (width, height) tuple."""
        return (self.width, self.height)


@dataclass
class LuxonisCameraConfig:
    """Configuration for Luxonis cameras."""

    ip: str
    stereo: bool  # True if the camera is a stereo camera
    resolution: LuxonisResolution
    fps: int
    queue_size: int = 8  # Size of output queues
    queue_blocking: bool = False  # If True, blocks when queue is full
    camera_mode: CameraSensorType = "MONO"


class LuxonisCameraSource(CameraSource):
    """Driver for Luxonis cameras."""

    ip: IPv4
    device: dai.Device
    cfg: LuxonisCameraConfig
    _calib_data: dai.CalibrationHandler
    _pipeline: dai.Pipeline | None
    _output_queues: dict[str, dai.MessageQueue]
    _running: bool
    _valid_resolutions: list[tuple[int, int]]

    def __init__(self, cfg: LuxonisCameraConfig) -> None:
        """Initialize the camera source."""
        self.ip = IPv4(cfg.ip)
        device = get_luxonis_device(self.ip)
        if device is None:
            raise ValueError(f"Device with IP address {self.ip} not found")

        self.device = device
        self.cfg = cfg
        self._running = False
        self._output_queues = {}
        self._pipeline = None

        sockets_to_check = (
            [dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]
            if self.cfg.stereo
            else [dai.CameraBoardSocket.CAM_A]
        )

        valid_resolutions: list[tuple[int, int]] = []
        valid_modes: list[dai.CameraSensorType] = []

        resolution_valid = False
        mode_valid = False
        for socket in sockets_to_check:
            valid_resolutions = get_luxonis_camera_valid_resolutions(self.device, socket)
            valid_modes = get_luxonis_camera_valid_modes(self.device, socket)

            if self.cfg.resolution.as_tuple() in valid_resolutions:
                resolution_valid = True

            if camera_sensor_type_to_dai[self.cfg.camera_mode] in valid_modes:
                mode_valid = True

            if resolution_valid and mode_valid:
                break
        else:
            errors = []
            if not resolution_valid:
                supported_resolutions = [f"{width}x{height}" for width, height in valid_resolutions]
                errors.append(
                    ValueError(
                        f"Resolution {self.cfg.resolution.as_tuple()} not supported for device {self.ip}. "
                        f"Supported resolutions: {', '.join(supported_resolutions)}"
                        f"for socket {socket}"
                    )
                )
            if not mode_valid:
                errors.append(
                    ValueError(
                        f"Camera mode {self.cfg.camera_mode} not supported for device {self.ip}. "
                        f"Supported modes: {', '.join([dai_to_camera_sensor_type[mode] for mode in valid_modes])} "
                        f"for socket {socket}"
                    )
                )

            raise ExceptionGroup("Invalid camera configuration", errors) from errors[0]

        # Load calibration data
        self._calib_data = self.device.readCalibration()

        # Store camera mode (sensor type) for pipeline building
        self._camera_mode = camera_sensor_type_to_dai[self.cfg.camera_mode]

        # Initialize intrinsics and extrinsics
        self._intrinsics: list[Intrinsics] | None = None
        self._extrinsics: list[Extrinsics] | None = None

    def _build_and_start_pipeline(self) -> None:
        """Build and start the pipeline."""
        # Create pipeline with device
        self._pipeline = dai.Pipeline(self.device)

        resolution = self.cfg.resolution.as_tuple()
        fps = float(self.cfg.fps)

        if self.cfg.stereo:
            # Create stereo pair
            left_camera = self._pipeline.create(dai.node.Camera)
            right_camera = self._pipeline.create(dai.node.Camera)

            # Build cameras with socket, resolution, and fps
            left_camera.build(boardSocket=dai.CameraBoardSocket.CAM_B, sensorResolution=resolution, sensorFps=fps)
            right_camera.build(boardSocket=dai.CameraBoardSocket.CAM_C, sensorResolution=resolution, sensorFps=fps)

            left_camera.setSensorType(self._camera_mode)
            right_camera.setSensorType(self._camera_mode)

            # Request outputs and create queues
            left_output = left_camera.requestOutput(size=resolution, fps=fps)
            right_output = right_camera.requestOutput(size=resolution, fps=fps)

            self._output_queues["left"] = left_output.createOutputQueue(
                maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
            )
            self._output_queues["right"] = right_output.createOutputQueue(
                maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
            )
        else:
            # Create RGB camera
            rgb_camera = self._pipeline.create(dai.node.Camera)

            # Build camera with socket, resolution, and fps
            rgb_camera.build(boardSocket=dai.CameraBoardSocket.CAM_A, sensorResolution=resolution, sensorFps=fps)

            rgb_camera.setSensorType(self._camera_mode)

            # Request output and create queue
            rgb_output = rgb_camera.requestOutput(size=resolution, fps=fps)

            self._output_queues["rgb"] = rgb_output.createOutputQueue(
                maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
            )

        # Start the pipeline
        self._pipeline.start()

    def get_intrinsics(self) -> list[Intrinsics]:
        """Get the intrinsics of the camera source. If stereo, returns [left, right]."""
        if self._intrinsics is not None:
            return self._intrinsics

        intrinsics_list: list[Intrinsics] = []

        if self.cfg.stereo:
            # For stereo cameras, return both left and right intrinsics
            # Left camera (CAM_B)
            left_matrix = np.array(
                self._calib_data.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_B, self.cfg.resolution.width, self.cfg.resolution.height
                )
            )
            left_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
            intrinsics_list.append(
                Intrinsics(
                    width=self.cfg.resolution.width,
                    height=self.cfg.resolution.height,
                    matrix=left_matrix,
                    coeffs=left_coeffs,
                )
            )

            # Right camera (CAM_C)
            right_matrix = np.array(
                self._calib_data.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_C, self.cfg.resolution.width, self.cfg.resolution.height
                )
            )
            right_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
            intrinsics_list.append(
                Intrinsics(
                    width=self.cfg.resolution.width,
                    height=self.cfg.resolution.height,
                    matrix=right_matrix,
                    coeffs=right_coeffs,
                )
            )
        else:
            rgb_matrix = np.array(
                self._calib_data.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A, self.cfg.resolution.width, self.cfg.resolution.height
                )
            )
            rgb_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
            intrinsics_list.append(
                Intrinsics(
                    width=self.cfg.resolution.width,
                    height=self.cfg.resolution.height,
                    matrix=rgb_matrix,
                    coeffs=rgb_coeffs,
                )
            )

        self._intrinsics = intrinsics_list
        return self._intrinsics

    def get_extrinsics(self) -> list[Extrinsics]:
        """Get the extrinsics of the camera source. If stereo, returns [left, right]."""
        if self._extrinsics is not None:
            return self._extrinsics

        extrinsics_list: list[Extrinsics] = []

        if self.cfg.stereo:
            left_to_center_matrix = np.array(
                self._calib_data.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A)
            )
            extrinsics_list.append(Extrinsics.from_4x4_matrix(left_to_center_matrix))

            right_to_center_matrix = np.array(
                self._calib_data.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A)
            )
            extrinsics_list.append(Extrinsics.from_4x4_matrix(right_to_center_matrix))
        else:
            # For RGB cameras, return identity (no relative transformation)
            extrinsics_list.append(Extrinsics.from_4x4_matrix(np.eye(4)))

        self._extrinsics = extrinsics_list
        return self._extrinsics

    @property
    def name(self) -> str:
        """Get the name of the camera source."""
        return str(self.ip)

    def start(self) -> None:
        """Start the camera source and pipeline."""
        if self._running:
            return

        # Build and start the pipeline
        self._build_and_start_pipeline()
        self._running = True

    def stop(self) -> None:
        """Stop the camera source."""
        if not self._running:
            return

        # Stop the pipeline
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

        self._output_queues.clear()
        self._running = False

    def is_running(self) -> bool:
        """Check if the camera is running."""
        return self._running

    def get_latest_frames(self) -> list[CameraFrame]:
        """Get the latest frames from the camera source.

        For stereo cameras, returns [left_frame, right_frame].
        For RGB cameras, returns [rgb_frame].
        """
        if not self._running:
            raise RuntimeError("Camera source not started. Call start() first.")

        frames: list[CameraFrame] = []

        if self.cfg.stereo:
            # Get left frame
            left_data = self._output_queues["left"].get()
            left_frame = left_data.getCvFrame()  # type: ignore[attr-defined]
            left_timestamp = left_data.getTimestamp()  # type: ignore[attr-defined]
            left_seq = left_data.getSequenceNum()  # type: ignore[attr-defined]

            frames.append(
                CameraFrame(
                    image=left_frame,
                    timestamp=left_timestamp.total_seconds(),
                    sequence_num=left_seq,
                    camera_name=f"{self.name}_left",
                )
            )

            # Get right frame
            right_data = self._output_queues["right"].get()
            right_frame = right_data.getCvFrame()  # type: ignore[attr-defined]

            # note that getTimestamp() is fixed to the host machine's clock
            # so comparison to other cameras is meaningful if they are on the same machine.
            right_timestamp = right_data.getTimestamp()  # type: ignore[attr-defined]
            right_seq = right_data.getSequenceNum()  # type: ignore[attr-defined]

            frames.append(
                CameraFrame(
                    image=right_frame,
                    timestamp=right_timestamp.total_seconds(),
                    sequence_num=right_seq,
                    camera_name=f"{self.name}_right",
                )
            )
        else:
            # Get RGB frame
            rgb_data = self._output_queues["rgb"].get()
            rgb_frame = rgb_data.getCvFrame()  # type: ignore[attr-defined]
            rgb_timestamp = rgb_data.getTimestamp()  # type: ignore[attr-defined]
            rgb_seq = rgb_data.getSequenceNum()  # type: ignore[attr-defined]

            frames.append(
                CameraFrame(
                    image=rgb_frame,
                    timestamp=rgb_timestamp.total_seconds(),
                    sequence_num=rgb_seq,
                    camera_name=f"{self.name}_rgb",
                )
            )

        return frames

    def try_get_latest_frames(self) -> list[CameraFrame] | None:
        """Try to get the latest frames without blocking.

        Returns None if no frames are available.
        """
        if not self._running:
            return None

        frames: list[CameraFrame] = []

        if self.cfg.stereo:
            left_data = self._output_queues["left"].tryGet()
            right_data = self._output_queues["right"].tryGet()

            if left_data is None or right_data is None:
                return None

            frames.append(
                CameraFrame(
                    image=left_data.getCvFrame(),  # type: ignore[attr-defined]
                    timestamp=left_data.getTimestamp().total_seconds(),  # type: ignore[attr-defined]
                    sequence_num=left_data.getSequenceNum(),  # type: ignore[attr-defined]
                    camera_name=f"{self.name}_left",
                )
            )
            frames.append(
                CameraFrame(
                    image=right_data.getCvFrame(),  # type: ignore[attr-defined]
                    timestamp=right_data.getTimestamp().total_seconds(),  # type: ignore[attr-defined]
                    sequence_num=right_data.getSequenceNum(),  # type: ignore[attr-defined]
                    camera_name=f"{self.name}_right",
                )
            )
        else:
            rgb_data = self._output_queues["rgb"].tryGet()

            if rgb_data is None:
                return None

            frames.append(
                CameraFrame(
                    image=rgb_data.getCvFrame(),  # type: ignore[attr-defined]
                    timestamp=rgb_data.getTimestamp().total_seconds(),  # type: ignore[attr-defined]
                    sequence_num=rgb_data.getSequenceNum(),  # type: ignore[attr-defined]
                    camera_name=f"{self.name}_rgb",
                )
            )

        return frames
