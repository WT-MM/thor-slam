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
class LuxonisRGBDCameraConfig:
    """Configuration for Luxonis RGB-D cameras."""

    # RGB-D streams (requires stereo=True)\
    depth_input_resolution: LuxonisResolution | None = None  # Optional: resize mono frames to this before StereoDepth

    enable_rgbd: bool = False
    rgb_sensor_resolution: LuxonisResolution | None = None  # Actual CAM_A sensor mode (auto-selected if None)
    rgb_output_resolution: LuxonisResolution | None = None  # Published RGB resolution for nvblox (auto-set if None)
    depth_output_resolution: LuxonisResolution | None = (
        None  # Published depth resolution (auto-set to match rgb_output if aligned)
    )

    rgbd_sync: bool = True  # Provide a synced RGB+depth stream via Sync node
    rgbd_sync_threshold_ms: int = 50
    rgbd_sync_attempts: int = 10

    # StereoDepth configuration (used for RGB-D depth)
    depth_preset: dai.node.StereoDepth.PresetMode = dai.node.StereoDepth.PresetMode.HIGH_DETAIL
    depth_lr_check: bool = True  # Required for depth alignment
    depth_subpixel: bool = False
    depth_extended_disparity: bool = False
    depth_align_to_rgb: bool = True  # If True, aligns depth to CAM_A (depth_output must match rgb_output)


@dataclass
class LuxonisCameraConfig:
    """Configuration for Luxonis cameras.

    Supports independent resolutions for SLAM stereo output and RGB-D nvblox output.
    """

    ip: str
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

    # Mono/stereo sensor and output resolutions
    mono_sensor_resolution: LuxonisResolution | None = None  # Actual CAM_B/C sensor mode for depth quality
    output_resolution: LuxonisResolution | None = None  # Published resolution for SLAM left/right

    rgbd_camera_config: LuxonisRGBDCameraConfig | None = None


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

        # Validate mono sensor resolution for stereo cameras
        if self.cfg.stereo:
            if self.cfg.mono_sensor_resolution is None:
                raise ValueError("mono_sensor_resolution must be set")
            mono_sensor_res = self.cfg.mono_sensor_resolution.as_tuple()
            sockets_to_check = [dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C]

            resolution_valid = False
            mode_valid = False
            for socket in sockets_to_check:
                valid_resolutions = get_luxonis_camera_valid_resolutions(self.device, socket)
                valid_modes = get_luxonis_camera_valid_modes(self.device, socket)

                if mono_sensor_res in valid_resolutions:
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
                            f"Mono sensor resolution {mono_sensor_res} not supported for device {self.ip}. "
                            f"Supported resolutions: {', '.join(supported_resolutions)} for socket {socket}"
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
        else:
            # Single camera mode - validate CAM_A
            if self.cfg.mono_sensor_resolution is None:
                raise ValueError("mono_sensor_resolution must be set")
            mono_sensor_res = self.cfg.mono_sensor_resolution.as_tuple()
            valid_resolutions = get_luxonis_camera_valid_resolutions(self.device, dai.CameraBoardSocket.CAM_A)
            valid_modes = get_luxonis_camera_valid_modes(self.device, dai.CameraBoardSocket.CAM_A)

            if mono_sensor_res not in valid_resolutions:
                supported = [f"{w}x{h}" for w, h in valid_resolutions]
                raise ValueError(
                    f"Mono sensor resolution {mono_sensor_res} not supported for device {self.ip}. "
                    f"Supported resolutions: {', '.join(supported)}"
                )

            if camera_sensor_type_to_dai[self.cfg.camera_mode] not in valid_modes:
                raise ValueError(
                    f"Camera mode {self.cfg.camera_mode} not supported for device {self.ip}. "
                    f"Supported modes: {', '.join([dai_to_camera_sensor_type[m] for m in valid_modes])}"
                )

        if self.cfg.rgbd_camera_config is not None and self.cfg.rgbd_camera_config.enable_rgbd:
            if not self.cfg.stereo:
                raise ValueError("RGB-D requires stereo=True (needs CAM_B/C for depth)")

            rgbd_cfg = self.cfg.rgbd_camera_config
            rgb_socket = dai.CameraBoardSocket.CAM_A
            rgb_valid_resolutions = get_luxonis_camera_valid_resolutions(self.device, rgb_socket)
            rgb_valid_modes = get_luxonis_camera_valid_modes(self.device, rgb_socket)

            rgb_errors: list[Exception] = []
            if dai.CameraSensorType.COLOR not in rgb_valid_modes:
                rgb_errors.append(
                    ValueError(
                        f"RGB stream not supported for device {self.ip} on socket {rgb_socket}. "
                        f"Supported modes: {', '.join([dai_to_camera_sensor_type[m] for m in rgb_valid_modes])}"
                    )
                )

            if rgb_errors:
                raise ExceptionGroup("Invalid RGB-D configuration", rgb_errors) from rgb_errors[0]

            # Determine RGB sensor resolution
            # If rgb_sensor_resolution is explicitly set, validate it; otherwise auto-select
            if rgbd_cfg.rgb_sensor_resolution is not None:
                rgb_sensor_res = rgbd_cfg.rgb_sensor_resolution.as_tuple()
                if rgb_sensor_res not in rgb_valid_resolutions:
                    supported = [f"{w}x{h}" for w, h in rgb_valid_resolutions]
                    raise ValueError(
                        f"RGB sensor resolution {rgb_sensor_res} not supported for device {self.ip}. "
                        f"Supported resolutions: {', '.join(supported)} for socket {rgb_socket}"
                    )
                # Explicitly set, no need for auto-selected resolution
                self._auto_rgb_sensor_resolution = None
            else:
                # Auto-select a valid color sensor resolution
                # If rgb_output_resolution is specified, prefer sensor resolutions that can support it
                rgb_output_res = rgbd_cfg.rgb_output_resolution.as_tuple() if rgbd_cfg.rgb_output_resolution else None
                if self.cfg.mono_sensor_resolution is None:
                    raise ValueError("mono_sensor_resolution must be set")
                mono_res = self.cfg.mono_sensor_resolution.as_tuple()

                best_res = None
                best_score = float("inf")

                for res in rgb_valid_resolutions:
                    score = 0.0

                    # Prefer resolutions that can support the desired output (sensor >= output)
                    if rgb_output_res is not None:
                        # Prefer sensor resolutions >= output resolution
                        if res[0] >= rgb_output_res[0] and res[1] >= rgb_output_res[1]:
                            # Good: sensor can support output, prefer smaller valid sensor
                            score = res[0] * res[1]  # Prefer smaller valid sensor
                        else:
                            # Bad: sensor too small for output, heavily penalize
                            score = 1000000 + (rgb_output_res[0] * rgb_output_res[1] - res[0] * res[1])
                    else:
                        # No output specified: prefer resolutions close to mono resolution
                        pixel_diff = abs(res[0] * res[1] - mono_res[0] * mono_res[1])
                        aspect_ratio_diff = abs((res[0] / res[1]) - (mono_res[0] / mono_res[1]))
                        score = pixel_diff + aspect_ratio_diff * 10000

                    if score < best_score:
                        best_score = score
                        best_res = res

                if best_res is None:
                    # Fallback: use smallest resolution
                    best_res = min(rgb_valid_resolutions, key=lambda r: r[0] * r[1])

                rgb_sensor_res = best_res
                # Store the auto-selected resolution for use in pipeline building
                self._auto_rgb_sensor_resolution = LuxonisResolution(width=rgb_sensor_res[0], height=rgb_sensor_res[1])
                logger.info(
                    "Auto-selected RGB sensor resolution for %s: %s (output: %s)",
                    self.ip,
                    rgb_sensor_res,
                    rgb_output_res if rgb_output_res else "not specified",
                )

            # Validate RGB output resolution is set
            if rgbd_cfg.rgb_output_resolution is None:
                # Default to RGB sensor resolution if not specified
                rgb_output_res_cfg = rgbd_cfg.rgb_sensor_resolution or self._auto_rgb_sensor_resolution
                if rgb_output_res_cfg is None:
                    raise ValueError("rgb_output_resolution must be set when enable_rgbd=True")
                if not isinstance(rgb_output_res_cfg, LuxonisResolution):
                    raise ValueError("rgb_output_resolution must be LuxonisResolution")
                rgbd_cfg.rgb_output_resolution = rgb_output_res_cfg

            # Validate depth output resolution
            if rgbd_cfg.depth_align_to_rgb:
                # When aligned, depth output must match RGB output
                if rgbd_cfg.depth_output_resolution is not None:
                    if rgbd_cfg.rgb_output_resolution is None:
                        raise ValueError("rgb_output_resolution must be set when depth_align_to_rgb=True")
                    if rgbd_cfg.depth_output_resolution.as_tuple() != rgbd_cfg.rgb_output_resolution.as_tuple():
                        raise ValueError(
                            "When depth_align_to_rgb=True, depth_output_resolution (%s) "
                            "must match rgb_output_resolution (%s)"
                            % (rgbd_cfg.depth_output_resolution, rgbd_cfg.rgb_output_resolution)
                        )
                else:
                    # Auto-set to match RGB output
                    rgbd_cfg.depth_output_resolution = rgbd_cfg.rgb_output_resolution

        # Set defaults for output resolutions
        if self.cfg.output_resolution is None:
            # Default to mono sensor resolution if not specified
            self.cfg.output_resolution = self.cfg.mono_sensor_resolution

        # Set depth input resolution default
        if self.cfg.rgbd_camera_config is not None and self.cfg.rgbd_camera_config.depth_input_resolution is None:
            # Default to mono sensor resolution (use full res for depth)
            self.cfg.rgbd_camera_config.depth_input_resolution = self.cfg.mono_sensor_resolution

        # Load calibration data
        self._calib_data = self.device.readCalibration()

        # Store camera mode (sensor type) for pipeline building
        self._camera_mode = camera_sensor_type_to_dai[self.cfg.camera_mode]

        # Initialize intrinsics and extrinsics
        self._intrinsics: list[Intrinsics] | None = None
        self._extrinsics: list[Extrinsics] | None = None

        # Initialize auto-selected RGB sensor resolution (set above if enable_rgbd, otherwise None)
        if not hasattr(self, "_auto_rgb_sensor_resolution"):
            self._auto_rgb_sensor_resolution = None

    def _build_and_start_pipeline(self) -> None:
        """Build and start the pipeline with independent resolutions for SLAM and RGB-D."""
        # Create pipeline with device
        self._pipeline = dai.Pipeline(self.device)

        # Get resolution tuples
        if self.cfg.mono_sensor_resolution is None:
            raise ValueError("mono_sensor_resolution must be set")
        if self.cfg.output_resolution is None:
            raise ValueError("output_resolution must be set")
        mono_sensor_res = self.cfg.mono_sensor_resolution.as_tuple()
        slam_output_res = self.cfg.output_resolution.as_tuple()
        fps = float(self.cfg.fps)

        # RGB-D resolutions (if enabled)
        rgb_sensor_res = None
        rgb_output_res = None
        depth_output_res = None
        depth_input_res = None

        if self.cfg.rgbd_camera_config is not None and self.cfg.rgbd_camera_config.enable_rgbd:
            rgbd_cfg = self.cfg.rgbd_camera_config
            if rgbd_cfg.depth_input_resolution is None:
                raise ValueError("depth_input_resolution must be set when enable_rgbd=True")
            depth_input_res = rgbd_cfg.depth_input_resolution.as_tuple()

            # Get RGB sensor resolution (explicit or auto-selected)
            if rgbd_cfg.rgb_sensor_resolution is not None:
                rgb_sensor_res = rgbd_cfg.rgb_sensor_resolution.as_tuple()
            elif hasattr(self, "_auto_rgb_sensor_resolution") and self._auto_rgb_sensor_resolution is not None:
                rgb_sensor_res = self._auto_rgb_sensor_resolution.as_tuple()
            else:
                raise RuntimeError("RGB sensor resolution not determined (should have been set in __init__)")

            if rgbd_cfg.rgb_output_resolution is None:
                raise ValueError("rgb_output_resolution must be set when enable_rgbd=True")
            if rgbd_cfg.depth_output_resolution is None:
                raise ValueError("depth_output_resolution must be set when enable_rgbd=True")
            rgb_output_res = rgbd_cfg.rgb_output_resolution.as_tuple()
            depth_output_res = rgbd_cfg.depth_output_resolution.as_tuple()

        resize_mode = dai.ImgResizeMode.LETTERBOX  # Use letterbox to preserve aspect ratio

        if self.cfg.stereo:
            # Left camera (CAM_B) - request TWO outputs: SLAM output and depth input
            left_cam = self._pipeline.create(dai.node.Camera)
            left_cam.setSensorType(self._camera_mode)
            left_cam.build(
                boardSocket=dai.CameraBoardSocket.CAM_B,
                sensorResolution=mono_sensor_res,
                sensorFps=fps,
            )

            # SLAM output (resized if needed)
            if slam_output_res != mono_sensor_res:
                left_slam_out = left_cam.requestOutput(size=slam_output_res, resizeMode=resize_mode, fps=fps)
            else:
                left_slam_out = left_cam.requestOutput(size=slam_output_res, fps=fps)

            self._output_queues["left"] = left_slam_out.createOutputQueue(
                maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
            )

            # Right camera (CAM_C) - request SLAM output
            right_cam = self._pipeline.create(dai.node.Camera)
            right_cam.setSensorType(self._camera_mode)
            right_cam.build(
                boardSocket=dai.CameraBoardSocket.CAM_C,
                sensorResolution=mono_sensor_res,
                sensorFps=fps,
            )

            # SLAM output (resized if needed)
            if slam_output_res != mono_sensor_res:
                right_slam_out = right_cam.requestOutput(size=slam_output_res, resizeMode=resize_mode, fps=fps)
            else:
                right_slam_out = right_cam.requestOutput(size=slam_output_res, fps=fps)

            self._output_queues["right"] = right_slam_out.createOutputQueue(
                maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
            )

            # Depth input streams - only request if RGB-D is enabled
            left_depth_in = None
            right_depth_in = None

            if self.cfg.rgbd_camera_config is not None and self.cfg.rgbd_camera_config.enable_rgbd:
                rgbd_cfg = self.cfg.rgbd_camera_config
                assert depth_input_res is not None
                # Request depth input streams from mono cameras (only if RGB-D enabled)
                if depth_input_res == mono_sensor_res:
                    left_depth_in = left_cam.requestFullResolutionOutput()
                else:
                    left_depth_in = left_cam.requestOutput(size=depth_input_res, resizeMode=resize_mode, fps=fps)

                if depth_input_res == mono_sensor_res:
                    right_depth_in = right_cam.requestFullResolutionOutput()
                else:
                    right_depth_in = right_cam.requestOutput(size=depth_input_res, resizeMode=resize_mode, fps=fps)

                # RGB camera (CAM_A)
                logger.info(
                    "RGB-D config for %s: rgb_sensor=%s, rgb_output=%s, depth_output=%s",
                    self.ip,
                    rgb_sensor_res,
                    rgb_output_res,
                    depth_output_res,
                )

                rgb_cam = self._pipeline.create(dai.node.Camera)
                rgb_cam.setSensorType(dai.CameraSensorType.COLOR)
                try:
                    # Build with explicit sensor resolution if set, otherwise let DepthAI auto-select
                    if rgb_sensor_res is not None:
                        rgb_cam.build(
                            boardSocket=dai.CameraBoardSocket.CAM_A,
                            sensorResolution=rgb_sensor_res,
                            sensorFps=fps,
                        )
                    else:
                        # Auto-select sensor mode
                        rgb_cam.build(
                            boardSocket=dai.CameraBoardSocket.CAM_A,
                            sensorFps=fps,
                        )
                        # Get the actual selected resolution for intrinsics
                        # Note: DepthAI v3 doesn't expose this directly, so we'll use rgb_output_res for intrinsics
                except Exception as e:
                    raise RuntimeError(f"Failed to build RGB camera (CAM_A) for {self.ip}: {e}") from e

                try:
                    if rgb_output_res != rgb_sensor_res:
                        # Request resized output
                        rgb_out = rgb_cam.requestOutput(
                            size=cast(tuple[int, int], rgb_output_res), resizeMode=resize_mode, fps=fps
                        )
                    else:
                        # Use full resolution output when output size matches sensor resolution
                        rgb_out = rgb_cam.requestFullResolutionOutput()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to request RGB output for {self.ip} with size {rgb_output_res} "
                        f"(sensor resolution: {rgb_sensor_res}): {e}"
                    ) from e

                self._output_queues["rgb"] = rgb_out.createOutputQueue(
                    maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
                )

                # StereoDepth: use depth input streams from mono cameras
                stereo = self._pipeline.create(dai.node.StereoDepth)
                stereo.setDefaultProfilePreset(rgbd_cfg.depth_preset)
                # Set input resolution to match the actual depth input stream size
                stereo.setInputResolution(depth_input_res[0], depth_input_res[1])

                left_depth_in.link(stereo.left)
                right_depth_in.link(stereo.right)

                stereo.setLeftRightCheck(rgbd_cfg.depth_lr_check)
                stereo.setSubpixel(rgbd_cfg.depth_subpixel)
                stereo.setExtendedDisparity(rgbd_cfg.depth_extended_disparity)

                if rgbd_cfg.depth_align_to_rgb:
                    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                    # ALWAYS set output size when aligned to avoid expensive upscaling
                    assert depth_output_res is not None
                    stereo.setOutputSize(depth_output_res[0], depth_output_res[1])
                elif depth_output_res is not None:
                    stereo.setOutputSize(depth_output_res[0], depth_output_res[1])

                self._output_queues["depth"] = stereo.depth.createOutputQueue(
                    maxSize=self.cfg.queue_size, blocking=self.cfg.queue_blocking
                )

                if rgbd_cfg.rgbd_sync:
                    sync = self._pipeline.create(dai.node.Sync)
                    sync.setRunOnHost(True)
                    sync.setSyncThreshold(timedelta(milliseconds=rgbd_cfg.rgbd_sync_threshold_ms))
                    sync.setSyncAttempts(rgbd_cfg.rgbd_sync_attempts)

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
                sensorResolution=mono_sensor_res,
                sensorFps=fps,
            )

            if slam_output_res != mono_sensor_res:
                out = cam.requestOutput(size=slam_output_res, resizeMode=resize_mode, fps=fps)
            else:
                out = cam.requestOutput(size=slam_output_res, fps=fps)

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
        """Get intrinsics for SLAM output (left/right at output_resolution).

        If stereo, returns [left, right] at output_resolution.
        """
        if self._intrinsics is not None:
            return self._intrinsics

        intrinsics_list: list[Intrinsics] = []

        # Use SLAM output resolution (what we publish for SLAM)
        if self.cfg.output_resolution is None:
            raise ValueError("output_resolution must be set")
        if self.cfg.mono_sensor_resolution is None:
            raise ValueError("mono_sensor_resolution must be set")
        slam_output = self.cfg.output_resolution
        mono_sensor = self.cfg.mono_sensor_resolution

        # Get intrinsics at sensor resolution, then scale to output
        if self.cfg.stereo:
            # Left camera (CAM_B)
            left_matrix: np.ndarray = np.array(
                self._calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, mono_sensor.width, mono_sensor.height)
            )
            # Scale to SLAM output resolution
            scale_x = slam_output.width / mono_sensor.width
            scale_y = slam_output.height / mono_sensor.height
            left_matrix_scaled = left_matrix.copy()
            left_matrix_scaled[0, 0] *= scale_x
            left_matrix_scaled[1, 1] *= scale_y
            left_matrix_scaled[0, 2] *= scale_x
            left_matrix_scaled[1, 2] *= scale_y

            left_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
            intrinsics_list.append(
                Intrinsics(
                    width=slam_output.width, height=slam_output.height, matrix=left_matrix_scaled, coeffs=left_coeffs
                )
            )

            # Right camera (CAM_C)
            right_matrix: np.ndarray = np.array(
                self._calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, mono_sensor.width, mono_sensor.height)
            )
            right_matrix_scaled = right_matrix.copy()
            right_matrix_scaled[0, 0] *= scale_x
            right_matrix_scaled[1, 1] *= scale_y
            right_matrix_scaled[0, 2] *= scale_x
            right_matrix_scaled[1, 2] *= scale_y

            right_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
            intrinsics_list.append(
                Intrinsics(
                    width=slam_output.width, height=slam_output.height, matrix=right_matrix_scaled, coeffs=right_coeffs
                )
            )
        else:
            # Single camera (CAM_A)
            rgb_matrix: np.ndarray = np.array(
                self._calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, mono_sensor.width, mono_sensor.height)
            )
            scale_x = slam_output.width / mono_sensor.width
            scale_y = slam_output.height / mono_sensor.height
            rgb_matrix_scaled = rgb_matrix.copy()
            rgb_matrix_scaled[0, 0] *= scale_x
            rgb_matrix_scaled[1, 1] *= scale_y
            rgb_matrix_scaled[0, 2] *= scale_x
            rgb_matrix_scaled[1, 2] *= scale_y

            rgb_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
            intrinsics_list.append(
                Intrinsics(
                    width=slam_output.width, height=slam_output.height, matrix=rgb_matrix_scaled, coeffs=rgb_coeffs
                )
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
        return Extrinsics.from_4x4_matrix(extrinsics_matrix)

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
        return self.cfg.stereo and self.cfg.rgbd_camera_config is not None and self.cfg.rgbd_camera_config.enable_rgbd

    def get_latest_rgbd_frames(self) -> tuple[CameraFrame, CameraFrame]:
        """Get the latest RGB and depth frames (blocking).

        Returns:
            Tuple of (rgb_frame, depth_frame).
        """
        if not self._running:
            raise RuntimeError("Camera source not started. Call start() first.")

        if not self.has_rgbd_streams:
            raise RuntimeError("RGB-D streams not enabled. Set enable_rgbd=True and stereo=True.")

        rgbd_cfg = self.cfg.rgbd_camera_config
        assert rgbd_cfg is not None
        if rgbd_cfg.rgbd_sync and "rgbd" in self._output_queues:
            # Use synced RGB-D stream
            rgbd_group = self._output_queues["rgbd"].get()
            rgb_data = rgbd_group["rgb"]  # type: ignore[index]
            depth_data = rgbd_group["depth"]  # type: ignore[index]
        else:
            # Get RGB and depth separately
            rgb_data = self._output_queues["rgb"].get()
            depth_data = self._output_queues["depth"].get()

        rgb_frame = rgb_data.getCvFrame()
        rgb_timestamp = rgb_data.getTimestamp()
        rgb_seq = rgb_data.getSequenceNum()

        depth_frame = depth_data.getCvFrame()
        depth_timestamp = depth_data.getTimestamp()
        depth_seq = depth_data.getSequenceNum()

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

        rgbd_cfg = self.cfg.rgbd_camera_config
        assert rgbd_cfg is not None
        if rgbd_cfg.rgbd_sync and "rgbd" in self._output_queues:
            # Use synced RGB-D stream
            rgbd_group = self._output_queues["rgbd"].tryGet()
            if rgbd_group is None:
                return None
            rgb_data = rgbd_group["rgb"]  # type: ignore[index]
            depth_data = rgbd_group["depth"]  # type: ignore[index]
        else:
            # Get RGB and depth separately
            rgb_data = self._output_queues["rgb"].tryGet()
            depth_data = self._output_queues["depth"].tryGet()
            if rgb_data is None or depth_data is None:
                return None

        rgb_frame = rgb_data.getCvFrame()
        rgb_timestamp = rgb_data.getTimestamp()
        rgb_seq = rgb_data.getSequenceNum()

        depth_frame = depth_data.getCvFrame()
        depth_timestamp = depth_data.getTimestamp()
        depth_seq = depth_data.getSequenceNum()

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
        """Get RGB and depth camera intrinsics at their published output resolutions.

        Returns:
            Tuple of (rgb_intrinsics, depth_intrinsics).
        """
        if not self.has_rgbd_streams:
            raise RuntimeError("RGB-D streams not enabled. Set enable_rgbd=True and stereo=True.")

        rgbd_cfg = self.cfg.rgbd_camera_config
        assert rgbd_cfg is not None

        # Get RGB sensor and output resolutions
        if rgbd_cfg.rgb_sensor_resolution is not None:
            rgb_sensor_res = rgbd_cfg.rgb_sensor_resolution
        elif hasattr(self, "_auto_rgb_sensor_resolution") and self._auto_rgb_sensor_resolution is not None:
            rgb_sensor_res = self._auto_rgb_sensor_resolution
        else:
            raise RuntimeError("RGB sensor resolution not determined")

        if rgbd_cfg.rgb_output_resolution is None:
            raise ValueError("rgb_output_resolution must be set when enable_rgbd=True")
        if rgbd_cfg.depth_output_resolution is None:
            raise ValueError("depth_output_resolution must be set when enable_rgbd=True")
        rgb_output_res = rgbd_cfg.rgb_output_resolution
        depth_output_res = rgbd_cfg.depth_output_resolution

        # RGB intrinsics (CAM_A) - get at sensor resolution, scale to output
        rgb_matrix = np.array(
            self._calib_data.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A, rgb_sensor_res.width, rgb_sensor_res.height
            )
        )
        rgb_scale_x = rgb_output_res.width / rgb_sensor_res.width
        rgb_scale_y = rgb_output_res.height / rgb_sensor_res.height
        rgb_matrix_scaled = rgb_matrix.copy()
        rgb_matrix_scaled[0, 0] *= rgb_scale_x
        rgb_matrix_scaled[1, 1] *= rgb_scale_y
        rgb_matrix_scaled[0, 2] *= rgb_scale_x
        rgb_matrix_scaled[1, 2] *= rgb_scale_y

        rgb_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))

        # Depth intrinsics
        if rgbd_cfg.depth_align_to_rgb:
            # When aligned, depth intrinsics match RGB (same K and D) at depth output resolution
            # Since depth_output must equal rgb_output when aligned, they should be the same
            if depth_output_res.width != rgb_output_res.width or depth_output_res.height != rgb_output_res.height:
                # Scale RGB intrinsics to depth output resolution (shouldn't happen if validation passed)
                depth_scale_from_rgb_x = depth_output_res.width / rgb_output_res.width
                depth_scale_from_rgb_y = depth_output_res.height / rgb_output_res.height
                depth_matrix_final = rgb_matrix_scaled.copy()
                depth_matrix_final[0, 0] *= depth_scale_from_rgb_x
                depth_matrix_final[1, 1] *= depth_scale_from_rgb_y
                depth_matrix_final[0, 2] *= depth_scale_from_rgb_x
                depth_matrix_final[1, 2] *= depth_scale_from_rgb_y
            else:
                depth_matrix_final = rgb_matrix_scaled.copy()
            depth_coeffs = rgb_coeffs.copy()
        else:
            # Not aligned: use left camera (CAM_B) intrinsics at depth output resolution
            if self.cfg.mono_sensor_resolution is None:
                raise ValueError("mono_sensor_resolution must be set")
            mono_sensor_res = self.cfg.mono_sensor_resolution
            depth_matrix: np.ndarray = np.array(
                self._calib_data.getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_B, mono_sensor_res.width, mono_sensor_res.height
                )
            )
            depth_scale_x = depth_output_res.width / mono_sensor_res.width
            depth_scale_y = depth_output_res.height / mono_sensor_res.height
            depth_matrix_scaled = depth_matrix.copy()
            depth_matrix_scaled[0, 0] *= depth_scale_x
            depth_matrix_scaled[1, 1] *= depth_scale_y
            depth_matrix_scaled[0, 2] *= depth_scale_x
            depth_matrix_scaled[1, 2] *= depth_scale_y
            depth_matrix_final = depth_matrix_scaled
            depth_coeffs = np.array(self._calib_data.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))

        return (
            Intrinsics(
                width=rgb_output_res.width,
                height=rgb_output_res.height,
                matrix=rgb_matrix_scaled,
                coeffs=rgb_coeffs,
            ),
            Intrinsics(
                width=depth_output_res.width,
                height=depth_output_res.height,
                matrix=depth_matrix_final,
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
                return None, None

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
