"""Driver for Luxonis cameras."""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Self, TypedDict, cast

import depthai as dai
import numpy as np

from thor_slam.camera.types import CameraFrame, CameraSensorType, CameraSource, Extrinsics, Intrinsics, IPv4
from thor_slam.camera.utils import (
    get_luxonis_camera_valid_modes,
    get_luxonis_camera_valid_resolutions,
    get_luxonis_device,
)

logger = logging.getLogger(__name__)


class IMUData(TypedDict):
    """IMU sensor data from Luxonis camera.

    Keys:
        accelerometer: Accelerometer data [x, y, z] in m/s²
        gyroscope: Gyroscope data [x, y, z] in rad/s
        timestamp: Timestamp of the IMU reading in seconds
        sequence_num: Sequence number of the IMU packet
    """

    accelerometer: np.ndarray  # [x, y, z] in m/s²
    gyroscope: np.ndarray  # [x, y, z] in rad/s
    timestamp: float
    sequence_num: int


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
    resolution: LuxonisResolution
    fps: int
    stereo: bool = False  # True if the camera is a stereo camera
    queue_size: int = 8  # Size of output queues
    queue_blocking: bool = False  # If True, blocks when queue is full
    camera_mode: CameraSensorType = "MONO"
    read_imu: bool = False
    imu_report_rate: int = 400  # IMU report rate in Hz
    imu_batch_threshold: int = 1  # Number of IMU packets to batch before reporting
    imu_max_batch_reports: int = 10  # Maximum number of batched reports
    imu_raw: bool = False  # If True, returns raw IMU data
    output_resolution: LuxonisResolution | None = None  # If set, rescale output to this resolution

    # Optional RGB-D streams (requires stereo=True)
    enable_rgbd: bool = False
    rgb_resolution: LuxonisResolution | None = None  # Sensor resolution for RGB (CAM_A)
    rgb_output_resolution: LuxonisResolution | None = None  # Host output size for RGB
    depth_output_resolution: LuxonisResolution | None = None  # Host output size for depth
    rgbd_sync: bool = True  # Provide a synced RGB+depth stream via Sync node
    rgbd_sync_threshold_ms: int = 50
    rgbd_sync_attempts: int = 10

    # StereoDepth configuration (used for RGB-D depth)
    depth_preset: dai.node.StereoDepth.PresetMode = dai.node.StereoDepth.PresetMode.HIGH_DETAIL
    depth_lr_check: bool = True  # Required for depth alignment
    depth_subpixel: bool = False
    depth_extended_disparity: bool = False
    depth_align_to_rgb: bool = True  # If True, aligns depth to CAM_A


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
    _has_sensor_data: bool
    _output_resolution: LuxonisResolution | None

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
        self._has_sensor_data = cfg.read_imu

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

        if self.cfg.enable_rgbd:
            if not self.cfg.stereo:
                raise ValueError("RGB-D requires stereo=True (needs CAM_B/C for depth)")

            rgb_socket = dai.CameraBoardSocket.CAM_A
            rgb_resolution = (self.cfg.rgb_resolution or self.cfg.resolution).as_tuple()
            rgb_valid_resolutions = get_luxonis_camera_valid_resolutions(self.device, rgb_socket)
            rgb_valid_modes = get_luxonis_camera_valid_modes(self.device, rgb_socket)

            rgb_errors: list[Exception] = []
            if rgb_resolution not in rgb_valid_resolutions:
                supported = [f"{w}x{h}" for w, h in rgb_valid_resolutions]
                rgb_errors.append(
                    ValueError(
                        f"RGB resolution {rgb_resolution} not supported for device {self.ip}. "
                        f"Supported resolutions: {', '.join(supported)} for socket {rgb_socket}"
                    )
                )
            if dai.CameraSensorType.COLOR not in rgb_valid_modes:
                rgb_errors.append(
                    ValueError(
                        f"RGB stream not supported for device {self.ip} on socket {rgb_socket}. "
                        f"Supported modes: {', '.join([dai_to_camera_sensor_type[m] for m in rgb_valid_modes])}"
                    )
                )

            if rgb_errors:
                raise ExceptionGroup("Invalid RGB-D configuration", rgb_errors) from rgb_errors[0]

        # Load calibration data
        self._calib_data = self.device.readCalibration()

        # Store camera mode (sensor type) for pipeline building
        self._camera_mode = camera_sensor_type_to_dai[self.cfg.camera_mode]

        self._output_resolution = self.cfg.output_resolution

        if self._output_resolution is None:
            self._output_resolution = self.cfg.resolution

        # Initialize intrinsics and extrinsics
        self._intrinsics: list[Intrinsics] | None = None
        self._extrinsics: list[Extrinsics] | None = None

    def _build_and_start_pipeline(self) -> None:
        """Build and start the pipeline."""
        # Create pipeline with device
        self._pipeline = dai.Pipeline(self.device)

        mono_sensor_resolution = self.cfg.resolution.as_tuple()
        fps = float(self.cfg.fps)

        # Output sizes (may differ from sensor sizes)
        lr_output_size = (self.cfg.output_resolution or self.cfg.resolution).as_tuple()

        rgb_sensor_resolution = (self.cfg.rgb_resolution or self.cfg.resolution).as_tuple()
        rgb_output_size = (
            self.cfg.rgb_output_resolution
            or self.cfg.output_resolution
            or LuxonisResolution(width=rgb_sensor_resolution[0], height=rgb_sensor_resolution[1])
        ).as_tuple()
        depth_output_size = (
            self.cfg.depth_output_resolution
            or self.cfg.rgb_output_resolution
            or self.cfg.output_resolution
            or LuxonisResolution(width=rgb_output_size[0], height=rgb_output_size[1])
        ).as_tuple()

        resize_mode = dai.ImgResizeMode.STRETCH

        if self.cfg.stereo:
            # Left camera (CAM_B)
            left_cam = self._pipeline.create(dai.node.Camera)
            left_cam.setSensorType(self._camera_mode)
            left_cam.build(
                boardSocket=dai.CameraBoardSocket.CAM_B,
                sensorResolution=mono_sensor_resolution,
                sensorFps=fps,
            )

            if lr_output_size != mono_sensor_resolution:
                left_out = left_cam.requestOutput(size=lr_output_size, resize_mode=resize_mode, fps=fps)
            else:
                left_out = left_cam.requestOutput(size=lr_output_size, fps=fps)

            self._output_queues["left"] = left_out.createOutputQueue(
                maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
            )

            # Right camera (CAM_C)
            right_cam = self._pipeline.create(dai.node.Camera)
            right_cam.setSensorType(self._camera_mode)
            right_cam.build(
                boardSocket=dai.CameraBoardSocket.CAM_C,
                sensorResolution=mono_sensor_resolution,
                sensorFps=fps,
            )

            if lr_output_size != mono_sensor_resolution:
                right_out = right_cam.requestOutput(size=lr_output_size, resize_mode=resize_mode, fps=fps)
            else:
                right_out = right_cam.requestOutput(size=lr_output_size, fps=fps)

            self._output_queues["right"] = right_out.createOutputQueue(
                maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
            )

            if self.cfg.enable_rgbd:
                # RGB camera (CAM_A)
                rgb_cam = self._pipeline.create(dai.node.Camera)
                rgb_cam.setSensorType(dai.CameraSensorType.COLOR)
                rgb_cam.build(
                    boardSocket=dai.CameraBoardSocket.CAM_A,
                    sensorResolution=rgb_sensor_resolution,
                    sensorFps=fps,
                )

                if rgb_output_size != rgb_sensor_resolution:
                    rgb_out = rgb_cam.requestOutput(size=rgb_output_size, resize_mode=resize_mode, fps=fps)
                else:
                    rgb_out = rgb_cam.requestOutput(size=rgb_output_size, fps=fps)

                self._output_queues["rgb"] = rgb_out.createOutputQueue(
                    maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
                )

                # Stereo depth from full-resolution mono streams
                left_full = left_cam.requestFullResolutionOutput()
                right_full = right_cam.requestFullResolutionOutput()

                stereo = self._pipeline.create(dai.node.StereoDepth)
                stereo.setDefaultProfilePreset(self.cfg.depth_preset)
                # Camera node isn't MonoCamera/ColorCamera, so be explicit about the input resolution
                stereo.setInputResolution(self.cfg.resolution.width, self.cfg.resolution.height)

                left_full.link(stereo.left)
                right_full.link(stereo.right)

                stereo.setLeftRightCheck(self.cfg.depth_lr_check)
                stereo.setSubpixel(self.cfg.depth_subpixel)
                stereo.setExtendedDisparity(self.cfg.depth_extended_disparity)
                if self.cfg.depth_align_to_rgb:
                    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                    stereo.setOutputSize(depth_output_size[0], depth_output_size[1])

                self._output_queues["depth"] = stereo.depth.createOutputQueue(
                    maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
                )

                if self.cfg.rgbd_sync:
                    sync = self._pipeline.create(dai.node.Sync)
                    sync.setRunOnHost(True)
                    sync.setSyncThreshold(timedelta(milliseconds=self.cfg.rgbd_sync_threshold_ms))
                    sync.setSyncAttempts(self.cfg.rgbd_sync_attempts)

                    rgb_out.link(sync.inputs["rgb"])
                    stereo.depth.link(sync.inputs["depth"])

                    self._output_queues["rgbd"] = sync.out.createOutputQueue(
                        maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
                    )

        else:
            # Single camera (CAM_A)
            cam = self._pipeline.create(dai.node.Camera)
            cam.setSensorType(self._camera_mode)
            cam.build(
                boardSocket=dai.CameraBoardSocket.CAM_A,
                sensorResolution=mono_sensor_resolution,
                sensorFps=fps,
            )

            if lr_output_size != mono_sensor_resolution:
                out = cam.requestOutput(size=lr_output_size, resize_mode=resize_mode, fps=fps)
            else:
                out = cam.requestOutput(size=lr_output_size, fps=fps)

            self._output_queues["rgb"] = out.createOutputQueue(
                maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
            )

        # Create IMU node if requested
        if self._has_sensor_data:
            # Create IMU node
            imu_node = self._pipeline.create(dai.node.IMU)

            # Enable accelerometer and gyroscope sensors
            imu_node.enableIMUSensor(
                dai.IMUSensor.ACCELEROMETER_RAW if self.cfg.imu_raw else dai.IMUSensor.ACCELEROMETER,
                self.cfg.imu_report_rate,
            )
            imu_node.enableIMUSensor(
                dai.IMUSensor.GYROSCOPE_RAW if self.cfg.imu_raw else dai.IMUSensor.GYROSCOPE_CALIBRATED,
                self.cfg.imu_report_rate,
            )

            # Configure batch reporting
            imu_node.setBatchReportThreshold(self.cfg.imu_batch_threshold)
            imu_node.setMaxBatchReports(self.cfg.imu_max_batch_reports)

            imu_queue = imu_node.out.createOutputQueue(maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking)

            self._output_queues["imu"] = imu_queue

        # Start the pipeline
        self._pipeline.start()

    def get_intrinsics(self) -> list[Intrinsics]:
        """Get intrinsics. If output_resolution is set, scale intrinsics accordingly.

        If stereo, returns [left, right].
        """
        if self._intrinsics is not None:
            return self._intrinsics

        intrinsics_list: list[Intrinsics] = []

        if self.cfg.output_resolution is not None:
            output_width = self.cfg.output_resolution.width
            output_height = self.cfg.output_resolution.height
            scale_x = output_width / self.cfg.resolution.width
            scale_y = output_height / self.cfg.resolution.height
        else:
            output_width = self.cfg.resolution.width
            output_height = self.cfg.resolution.height
            scale_x = 1.0
            scale_y = 1.0

        if self.cfg.stereo:
            left_matrix = np.array(
                self._calib_data.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_B, self.cfg.resolution.width, self.cfg.resolution.height
                )
            )
            left_matrix_scaled = left_matrix.copy()
            left_matrix_scaled[0, 0] *= scale_x
            left_matrix_scaled[1, 1] *= scale_y
            left_matrix_scaled[0, 2] *= scale_x
            left_matrix_scaled[1, 2] *= scale_y

            left_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
            intrinsics_list.append(
                Intrinsics(width=output_width, height=output_height, matrix=left_matrix_scaled, coeffs=left_coeffs)
            )

            right_matrix = np.array(
                self._calib_data.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_C, self.cfg.resolution.width, self.cfg.resolution.height
                )
            )
            right_matrix_scaled = right_matrix.copy()
            right_matrix_scaled[0, 0] *= scale_x
            right_matrix_scaled[1, 1] *= scale_y
            right_matrix_scaled[0, 2] *= scale_x
            right_matrix_scaled[1, 2] *= scale_y

            right_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
            intrinsics_list.append(
                Intrinsics(width=output_width, height=output_height, matrix=right_matrix_scaled, coeffs=right_coeffs)
            )
        else:
            rgb_matrix = np.array(
                self._calib_data.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A, self.cfg.resolution.width, self.cfg.resolution.height
                )
            )
            rgb_matrix_scaled = rgb_matrix.copy()
            rgb_matrix_scaled[0, 0] *= scale_x
            rgb_matrix_scaled[1, 1] *= scale_y
            rgb_matrix_scaled[0, 2] *= scale_x
            rgb_matrix_scaled[1, 2] *= scale_y

            rgb_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
            intrinsics_list.append(
                Intrinsics(width=output_width, height=output_height, matrix=rgb_matrix_scaled, coeffs=rgb_coeffs)
            )

        self._intrinsics = intrinsics_list
        return self._intrinsics

    def get_extrinsics(self) -> list[Extrinsics]:
        """Get the extrinsics of the camera source. If stereo, returns [left, right].

        Note: DepthAI returns translation in centimeters, so we convert to meters.
        """
        if self._extrinsics is not None:
            return self._extrinsics

        extrinsics_list: list[Extrinsics] = []

        if self.cfg.stereo:
            # This returns where the left camera is relative to the center camera
            # (so if our coordinate system has x positive right, then left_to_center_matrix[0, 3] is negative)
            # i.e. the translation is in center camera coordinate frame.
            # Then our matrix of left_to_center can be multiplied by a point in the left camera coordinate
            # frame to get a point in the center camera coordinate frame.
            left_to_center_matrix = np.array(
                self._calib_data.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A)
            )
            # Convert translation from cm to meters
            left_to_center_matrix[:3, 3] /= 100.0
            extrinsics_list.append(Extrinsics.from_4x4_matrix(left_to_center_matrix))

            right_to_center_matrix = np.array(
                self._calib_data.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A)
            )
            # Convert translation from cm to meters
            right_to_center_matrix[:3, 3] /= 100.0
            extrinsics_list.append(Extrinsics.from_4x4_matrix(right_to_center_matrix))
        else:
            # For RGB cameras, return identity (no relative transformation)
            extrinsics_list.append(Extrinsics.from_4x4_matrix(np.eye(4)))

        self._extrinsics = extrinsics_list
        return self._extrinsics

    def get_sensor_extrinsics(self) -> Extrinsics | None:
        """Get the extrinsics of a non-camera sensor relative to the CameraSource's reference frame.

        This method is for non-camera sensors (e.g., IMU). For camera sensors, use get_extrinsics().
        The reference frame is CAM_A (center camera).
        """
        try:
            imu_extrinsics = self._calib_data.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A)
        except RuntimeError as e:
            logger.warning("Failed to get IMU extrinsics: %s. Returning identity transformation.", e)
            return Extrinsics.from_4x4_matrix(np.eye(4))

        extrinsics_matrix = np.array(imu_extrinsics)
        # Convert translation from cm to meters
        extrinsics_matrix[:3, 3] /= 100.0
        return Extrinsics.from_4x4_matrix(imu_extrinsics)

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

    @property
    def has_rgbd_streams(self) -> bool:
        """Check if RGB-D streams are available."""
        return self.cfg.stereo and self.cfg.enable_rgbd

    def get_latest_rgbd_frames(self) -> tuple[CameraFrame, CameraFrame]:
        """Get the latest RGB and depth frames (blocking).

        Returns:
            Tuple of (rgb_frame, depth_frame).
        """
        if not self._running:
            raise RuntimeError("Camera source not started. Call start() first.")

        if not self.has_rgbd_streams:
            raise RuntimeError("RGB-D streams not enabled. Set enable_rgbd=True and stereo=True.")

        if self.cfg.rgbd_sync and "rgbd" in self._output_queues:
            # Use synced RGB-D stream
            rgbd_group = self._output_queues["rgbd"].get()
            rgb_data = rgbd_group["rgb"]
            depth_data = rgbd_group["depth"]
        else:
            # Get RGB and depth separately
            rgb_data = self._output_queues["rgb"].get()
            depth_data = self._output_queues["depth"].get()

        rgb_frame = rgb_data.getCvFrame()  # type: ignore[attr-defined]
        rgb_timestamp = rgb_data.getTimestamp()  # type: ignore[attr-defined]
        rgb_seq = rgb_data.getSequenceNum()  # type: ignore[attr-defined]

        depth_frame = depth_data.getCvFrame()  # type: ignore[attr-defined]
        depth_timestamp = depth_data.getTimestamp()  # type: ignore[attr-defined]
        depth_seq = depth_data.getSequenceNum()  # type: ignore[attr-defined]

        return (
            CameraFrame(
                image=rgb_frame,
                timestamp=rgb_timestamp.total_seconds(),
                sequence_num=rgb_seq,
                camera_name=f"{self.name}_rgb",
            ),
            CameraFrame(
                image=depth_frame,
                timestamp=depth_timestamp.total_seconds(),
                sequence_num=depth_seq,
                camera_name=f"{self.name}_depth",
            ),
        )

    def try_get_latest_rgbd_frames(self) -> tuple[CameraFrame, CameraFrame] | None:
        """Try to get the latest RGB and depth frames without blocking.

        Returns:
            Tuple of (rgb_frame, depth_frame) if available, None otherwise.
        """
        if not self._running:
            return None

        if not self.has_rgbd_streams:
            return None

        if self.cfg.rgbd_sync and "rgbd" in self._output_queues:
            # Use synced RGB-D stream
            rgbd_group = self._output_queues["rgbd"].tryGet()
            if rgbd_group is None:
                return None
            rgb_data = rgbd_group["rgb"]
            depth_data = rgbd_group["depth"]
        else:
            # Get RGB and depth separately
            rgb_data = self._output_queues["rgb"].tryGet()
            depth_data = self._output_queues["depth"].tryGet()
            if rgb_data is None or depth_data is None:
                return None

        rgb_frame = rgb_data.getCvFrame()  # type: ignore[attr-defined]
        rgb_timestamp = rgb_data.getTimestamp()  # type: ignore[attr-defined]
        rgb_seq = rgb_data.getSequenceNum()  # type: ignore[attr-defined]

        depth_frame = depth_data.getCvFrame()  # type: ignore[attr-defined]
        depth_timestamp = depth_data.getTimestamp()  # type: ignore[attr-defined]
        depth_seq = depth_data.getSequenceNum()  # type: ignore[attr-defined]

        return (
            CameraFrame(
                image=rgb_frame,
                timestamp=rgb_timestamp.total_seconds(),
                sequence_num=rgb_seq,
                camera_name=f"{self.name}_rgb",
            ),
            CameraFrame(
                image=depth_frame,
                timestamp=depth_timestamp.total_seconds(),
                sequence_num=depth_seq,
                camera_name=f"{self.name}_depth",
            ),
        )

    def get_rgbd_intrinsics(self) -> tuple[Intrinsics, Intrinsics]:
        """Get RGB and depth camera intrinsics.

        Returns:
            Tuple of (rgb_intrinsics, depth_intrinsics).
        """
        if not self.has_rgbd_streams:
            raise RuntimeError("RGB-D streams not enabled. Set enable_rgbd=True and stereo=True.")

        # Get RGB resolution and output size
        rgb_sensor_resolution = self.cfg.rgb_resolution or self.cfg.resolution
        rgb_output_resolution = (
            self.cfg.rgb_output_resolution
            or self.cfg.output_resolution
            or LuxonisResolution(width=rgb_sensor_resolution.width, height=rgb_sensor_resolution.height)
        )

        # Get depth output size
        depth_output_resolution = (
            self.cfg.depth_output_resolution
            or self.cfg.rgb_output_resolution
            or self.cfg.output_resolution
            or LuxonisResolution(width=rgb_output_resolution.width, height=rgb_output_resolution.height)
        )

        # Calculate scaling factors
        rgb_scale_x = rgb_output_resolution.width / rgb_sensor_resolution.width
        rgb_scale_y = rgb_output_resolution.height / rgb_sensor_resolution.height

        depth_scale_x = depth_output_resolution.width / rgb_sensor_resolution.width
        depth_scale_y = depth_output_resolution.height / rgb_sensor_resolution.height

        # RGB intrinsics (CAM_A)
        rgb_matrix = np.array(
            self._calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, rgb_sensor_resolution.width, rgb_sensor_resolution.height
            )
        )
        rgb_matrix_scaled = rgb_matrix.copy()
        rgb_matrix_scaled[0, 0] *= rgb_scale_x
        rgb_matrix_scaled[1, 1] *= rgb_scale_y
        rgb_matrix_scaled[0, 2] *= rgb_scale_x
        rgb_matrix_scaled[1, 2] *= rgb_scale_y

        rgb_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))

        # Depth intrinsics - if aligned to RGB, use RGB intrinsics scaled to depth output; otherwise use left camera
        if self.cfg.depth_align_to_rgb:
            # When aligned, depth intrinsics match RGB but may be at different output resolution
            if depth_output_resolution.width != rgb_output_resolution.width or depth_output_resolution.height != rgb_output_resolution.height:
                # Scale RGB intrinsics to depth output resolution
                depth_scale_from_rgb_x = depth_output_resolution.width / rgb_output_resolution.width
                depth_scale_from_rgb_y = depth_output_resolution.height / rgb_output_resolution.height
                depth_matrix = rgb_matrix_scaled.copy()
                depth_matrix[0, 0] *= depth_scale_from_rgb_x
                depth_matrix[1, 1] *= depth_scale_from_rgb_y
                depth_matrix[0, 2] *= depth_scale_from_rgb_x
                depth_matrix[1, 2] *= depth_scale_from_rgb_y
            else:
                depth_matrix = rgb_matrix_scaled.copy()
            depth_coeffs = rgb_coeffs.copy()
        else:
            depth_matrix = np.array(
                self._calib_data.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_B, self.cfg.resolution.width, self.cfg.resolution.height
                )
            )
            depth_matrix_scaled = depth_matrix.copy()
            depth_matrix_scaled[0, 0] *= depth_scale_x
            depth_matrix_scaled[1, 1] *= depth_scale_y
            depth_matrix_scaled[0, 2] *= depth_scale_x
            depth_matrix_scaled[1, 2] *= depth_scale_y
            depth_matrix = depth_matrix_scaled
            depth_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))

        return (
            Intrinsics(
                width=rgb_output_resolution.width,
                height=rgb_output_resolution.height,
                matrix=rgb_matrix_scaled,
                coeffs=rgb_coeffs,
            ),
            Intrinsics(
                width=depth_output_resolution.width,
                height=depth_output_resolution.height,
                matrix=depth_matrix,
                coeffs=depth_coeffs,
            ),
        )

    def get_rgbd_extrinsics(self) -> tuple[Extrinsics, Extrinsics]:
        """Get RGB and depth camera extrinsics.

        Returns:
            Tuple of (rgb_extrinsics, depth_extrinsics). RGB is identity (CAM_A reference).
            Depth is relative to RGB (CAM_B to CAM_A transformation).

        Note: DepthAI returns translation in centimeters, so we convert to meters.
        """
        if not self.has_rgbd_streams:
            raise RuntimeError("RGB-D streams not enabled. Set enable_rgbd=True and stereo=True.")

        # RGB camera is at CAM_A (identity)
        rgb_extrinsics = Extrinsics.from_4x4_matrix(np.eye(4))

        # Depth camera extrinsics relative to RGB (from stereo left camera CAM_B to color CAM_A)
        depth_to_rgb_matrix = np.array(
            self._calib_data.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A)
        )
        # Convert translation from cm to meters
        depth_to_rgb_matrix[:3, 3] /= 100.0
        depth_extrinsics = Extrinsics.from_4x4_matrix(depth_to_rgb_matrix)

        return (rgb_extrinsics, depth_extrinsics)

    @property
    def has_sensor_data(self) -> bool:
        """Check if the camera source has sensor data."""
        return self._has_sensor_data

    def _process_imu_data(self, imu_data: dai.IMUData) -> list[IMUData]:
        """Process IMU data packets into IMUData objects.

        Args:
            imu_data: Raw IMU data from DepthAI queue.

        Returns:
            List of IMUData objects extracted from the packets.
        """
        imu_packets = []
        for packet in imu_data.packets:
            # Extract accelerometer data
            accel = packet.acceleroMeter
            accelerometer = np.array([accel.x, accel.y, accel.z])

            # Extract gyroscope data
            gyro = packet.gyroscope
            gyroscope = np.array([gyro.x, gyro.y, gyro.z])

            # Extract timestamp (use device timestamp for accuracy)
            timestamp = packet.acceleroMeter.getTimestamp().total_seconds()

            # Extract sequence number
            sequence_num = packet.acceleroMeter.getSequenceNum()

            imu_packets.append(
                IMUData(
                    accelerometer=accelerometer,
                    gyroscope=gyroscope,
                    timestamp=timestamp,
                    sequence_num=sequence_num,
                )
            )

        return imu_packets

    def get_timestamped_sensor_data(self) -> tuple[dict | None, float | None]:
        """Get the timestamped sensor data of the camera source (blocking).

        Returns:
            Tuple of (IMUData object if available, timestamp if available), or (None, None) if no data available.
        """
        if not self._has_sensor_data:
            return None, None

        if "imu" not in self._output_queues:
            return {}, None

        if not self._running:
            raise RuntimeError("Camera source not started. Call start() first.")

        try:
            # Get IMU data from queue (blocking)
            imu_data = self._output_queues["imu"].get()

            assert isinstance(imu_data, dai.IMUData)
            # Process IMU packets
            imu_packets = self._process_imu_data(imu_data)

            # Return the latest IMU data and all packets
            return (cast(dict, imu_packets[-1])) if imu_packets else None, (
                imu_packets[-1]["timestamp"] if imu_packets else None
            )

        except Exception as e:
            logger.warning("Failed to get IMU data: %s", e)
            return {}, None

    def try_get_timestamped_sensor_data(self) -> tuple[dict | None, float | None]:
        """Try to get timestamped sensor data without blocking.

        Returns:
            Tuple of (IMUData object if available, timestamp if available), or (None, None) if no data available.
        """
        if not self._has_sensor_data:
            return None, None

        if "imu" not in self._output_queues:
            return None, None

        if not self._running:
            return None, None

        try:
            # Try to get IMU data from queue (non-blocking)
            imu_data = self._output_queues["imu"].tryGet()

            if imu_data is None:
                return None

            assert isinstance(imu_data, dai.IMUData)

            # Process IMU packets
            imu_packets = self._process_imu_data(imu_data)
            # Return the latest IMU data and all packets
            return (cast(dict, imu_packets[-1])) if imu_packets else None, (
                imu_packets[-1]["timestamp"] if imu_packets else None
            )

        except Exception as e:
            logger.warning("Failed to get IMU data: %s", e)
            return None, None
