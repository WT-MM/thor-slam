"""Coordinating system for multiple cameras with frame synchronization."""

import logging
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from types import TracebackType
from typing import Self

import numpy as np

from thor_slam.camera.types import CameraSource, Extrinsics, FrameSet, Intrinsics, SynchronizedFrameSet

logger = logging.getLogger(__name__)


@dataclass
class RigCalibration:
    """Calibration data for the entire camera rig.

    Attributes:
        intrinsics: Per-source camera intrinsics (source_name -> [intrinsics per camera]).
        extrinsics: Per-source camera extrinsics relative to that source's reference frame
                    (source_name -> [extrinsics per camera]).
        rig_extrinsics: Position/orientation of each source relative to the rig coordinate frame
                        (source_name -> Extrinsics). Used to transform camera extrinsics to rig space.
    """

    intrinsics: dict[str, list[Intrinsics]]  # source_name -> [intrinsics per camera]
    extrinsics: dict[str, list[Extrinsics]]  # source_name -> [extrinsics per camera]
    rig_extrinsics: dict[str, Extrinsics] = field(default_factory=dict)  # source_name -> rig pose

    def get_world_extrinsics(self, source_name: str) -> list[Extrinsics] | None:
        """Get extrinsics transformed to rig/world coordinate frame.

        Combines camera extrinsics with rig extrinsics:
            world_T_camera = rig_T_source @ source_T_camera

        The resulting matrix is a transformation from a point in the camera coordinate frame
        to a point in the world coordinate frame.

        Args:
            source_name: Name of the camera source.

        Returns:
            List of extrinsics in rig coordinate frame, or None if source not found.
        """
        if source_name not in self.extrinsics:
            return None

        camera_extrinsics = self.extrinsics[source_name]

        # If no rig extrinsics defined for this source, return camera extrinsics as-is
        if source_name not in self.rig_extrinsics:
            logger.warning("No rig extrinsics defined for source %s, returning camera extrinsics as-is", source_name)
            return camera_extrinsics

        rig_ext = self.rig_extrinsics[source_name]
        rig_matrix = rig_ext.to_4x4_matrix()

        # Transform each camera's extrinsics to world frame
        world_extrinsics = []
        for cam_ext in camera_extrinsics:
            cam_matrix = cam_ext.to_4x4_matrix()
            world_matrix = rig_matrix @ cam_matrix
            world_extrinsics.append(Extrinsics.from_4x4_matrix(world_matrix))

        return world_extrinsics


class CameraRig:
    """Coordinating system for multiple cameras with frame synchronization.

    The rig keeps a queue of recent frames from each camera source and provides
    synchronized frame sets. Synchronization is done by finding the camera that
    is furthest behind (slowest) and matching frames from other cameras to that
    timestamp.
    """

    sources: dict[str, CameraSource]
    queue_size: int
    _frame_queues: dict[str, deque[FrameSet]]
    _lock: Lock
    _running: bool
    _calibration: RigCalibration
    _imu_source: str | None
    _imu_queue: deque[tuple[float, dict]]

    def __init__(
        self,
        sources: list[CameraSource],
        queue_size: int = 30,
        rig_extrinsics: dict[str, Extrinsics] | None = None,
        imu_source: str | None = None,
    ) -> None:
        """Initialize the camera rig.

        Args:
            sources: List of camera sources to synchronize.
            queue_size: Maximum number of frame sets to keep in each queue.
            rig_extrinsics: Optional dict mapping source names to their pose in the rig frame.
                            If not provided, identity transforms are used for all sources.
            imu_source: Optional name of the camera source to use as the primary IMU.
                        If specified, IMU data will be pulled from this source.
        """
        self.sources = {src.name: src for src in sources}
        self.queue_size = queue_size
        self._frame_queues = {name: deque(maxlen=queue_size) for name in self.sources}
        self._lock = Lock()
        self._running = False
        self._imu_source = imu_source
        self._imu_queue: deque[tuple[float, dict]] = deque(maxlen=queue_size)

        # Validate IMU source if specified
        if self._imu_source is not None:
            if self._imu_source not in self.sources:
                raise ValueError(
                    f"IMU source '{self._imu_source}' not found in sources. "
                    f"Available sources: {list(self.sources.keys())}"
                )
            imu_source_obj = self.sources[self._imu_source]
            if not imu_source_obj.has_sensor_data:
                raise ValueError(
                    f"IMU source '{self._imu_source}' does not have sensor data enabled. "
                    "Set read_imu=True when creating the camera source."
                )
            logger.info("Using '%s' as IMU source", self._imu_source)

        # Build rig extrinsics (identity if not provided)
        if not rig_extrinsics:
            logger.warning("No rig extrinsics provided, using identity transformation for all sources")
            rig_extrinsics = {name: Extrinsics.from_4x4_matrix(np.eye(4)) for name in self.sources}

        # Build calibration
        self._calibration = self._build_calibration(rig_extrinsics)

    """
    Supports context manager protocol for automatic cleanup:
        with CameraRig(sources) as rig:
            rig.get_synchronized_frames()
    """

    def __enter__(self) -> Self:
        """Enter context manager, starting all camera sources."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, stopping all camera sources."""
        self.stop()

    def start(self) -> None:
        """Start all camera sources."""
        if self._running:
            return

        for source in self.sources.values():
            source.start()
        self._running = True

    def stop(self) -> None:
        """Stop all camera sources."""
        if not self._running:
            return

        for source in self.sources.values():
            source.stop()
        self._running = False

        # Clear queues
        with self._lock:
            for queue in self._frame_queues.values():
                queue.clear()

    def is_running(self) -> bool:
        """Check if the rig is running."""
        return self._running

    def _build_calibration(self, rig_extrinsics: dict[str, Extrinsics]) -> RigCalibration:
        """Build calibration data from all sources.

        Args:
            rig_extrinsics: Dict mapping source names to their pose in the rig frame.
        """
        intrinsics: dict[str, list[Intrinsics]] = {}
        extrinsics: dict[str, list[Extrinsics]] = {}

        for name, source in self.sources.items():
            intrinsics[name] = source.get_intrinsics()
            extrinsics[name] = source.get_extrinsics()

        return RigCalibration(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            rig_extrinsics=rig_extrinsics,
        )

    @property
    def calibration(self) -> RigCalibration:
        """Get the rig calibration data."""
        return self._calibration

    def load_rig_extrinsics(self, rig_extrinsics: dict[str, Extrinsics]) -> None:
        """Load rig extrinsics for multiple sources.

        Args:
            rig_extrinsics: Dict mapping source names to their pose in the rig frame.
        """
        for name in rig_extrinsics:
            if name not in self.sources:
                raise ValueError(f"Unknown source: {name}")

        # Rebuild calibration with updated rig extrinsics
        new_rig_extrinsics = self._calibration.rig_extrinsics.copy()
        new_rig_extrinsics.update(rig_extrinsics)
        self._calibration = self._build_calibration(new_rig_extrinsics)

    def get_rig_extrinsics(self, source_name: str) -> Extrinsics | None:
        """Get the rig extrinsics for a camera source.

        Args:
            source_name: Name of the camera source.

        Returns:
            The source's pose in the rig frame, or None if not set.
        """
        return self._calibration.rig_extrinsics.get(source_name)

    def get_world_extrinsics(self, source_name: str) -> list[Extrinsics] | None:
        """Get camera extrinsics transformed to rig/world coordinate frame.

        Combines camera extrinsics with rig extrinsics:
            world_T_camera = rig_T_source @ source_T_camera

        Args:
            source_name: Name of the camera source.

        Returns:
            List of extrinsics in rig coordinate frame, or None if source not found.
        """
        return self._calibration.get_world_extrinsics(source_name)

    def _poll_cameras(self) -> None:
        """Poll all cameras for new frames and add to queues."""
        for name, source in self.sources.items():
            if name == self._imu_source:
                sensor_data = source.try_get_timestamped_sensor_data()
                if sensor_data is not None and sensor_data[0] is not None and sensor_data[1] is not None:
                    self._imu_queue.append((sensor_data[1], sensor_data[0]))

            frames = source.try_get_latest_frames()
            if frames:
                frame_set = FrameSet.from_frames(frames, source_name=name)
                # if source.has_sensor_data:
                #     # Try to get sensor data (non-blocking)
                #     sensor_data = source.try_get_sensor_data()
                #     if sensor_data:
                #         frame_set.sensor_data = sensor_data.get("imu")
                #         frame_set.sensor_timestamp = frame_set.sensor_data.timestamp

                with self._lock:
                    self._frame_queues[name].append(frame_set)

    def _find_closest_frame_set(
        self,
        queue: deque[FrameSet],
        target_timestamp: float,
    ) -> FrameSet | None:
        """Find the frame set in a queue closest to the target timestamp.

        Args:
            queue: The queue to search.
            target_timestamp: The target timestamp to match.

        Returns:
            The frame set closest to the target timestamp, or None if queue is empty.
        """
        if not queue:
            return None

        return min(queue, key=lambda fs: abs(fs.timestamp - target_timestamp))

    def _find_closest_imu_data(
        self, queue: deque[tuple[float, dict]], target_timestamp: float
    ) -> tuple[float | None, dict | None]:
        """Find the IMU data closest to the target timestamp from a queue.

        Args:
            queue: The queue to search (contains tuples of (timestamp, IMUData)).
            target_timestamp: The target timestamp to match.

        Returns:
            Tuple of (IMUData object closest to target, timestamp), or (None, None) if no IMU data available.
        """
        if not queue:
            return None, None

        closest = min(queue, key=lambda data: abs(data[0] - target_timestamp))
        return closest[0], closest[1]

    def _get_reference_timestamp(self) -> float | None:
        """Get the reference timestamp (from the slowest camera).

        The reference timestamp is the minimum of the maximum timestamps
        across all queues. This ensures all cameras have frames at or after
        this timestamp.

        Returns:
            The reference timestamp, or None if any queue is empty.
        """
        latest_timestamps = {}

        with self._lock:
            for name, queue in self._frame_queues.items():
                if not queue:
                    return None  # A queue is empty, can't synchronize yet
                latest_timestamps[name] = queue[-1].timestamp

        # Return the minimum of all latest timestamps
        # This is the timestamp of the camera that's furthest behind
        return min(latest_timestamps.values())

    def get_synchronized_frames(self, max_wait_ms: float = 100.0) -> SynchronizedFrameSet | None:
        """Get a synchronized set of frames from all camera sources.

        This method:
        1. Polls all cameras for new frames
        2. Finds the camera that's furthest behind (slowest)
        3. Uses that camera's timestamp as the reference
        4. For each camera, finds the frame set closest to the reference timestamp
        5. Aggregates IMU data from all sources, finding the closest IMU reading to the reference timestamp
        6. Returns the synchronized frame set

        Args:
            max_wait_ms: Maximum time to wait for frames (not currently used).

        Returns:
            SynchronizedFrameSet if synchronization is successful, None otherwise.
        """
        if not self._running:
            return None

        # Poll cameras for new frames
        self._poll_cameras()

        # Get the reference timestamp (from the slowest camera)
        reference_timestamp = self._get_reference_timestamp()
        if reference_timestamp is None:
            logger.warning("No reference timestamp found, not all cameras have frames yet")
            return None  # Not all cameras have frames yet

        # Find closest frame set for each camera
        synchronized_frame_sets: dict[str, FrameSet] = {}
        max_time_delta = 0.0
        with self._lock:
            for name, queue in self._frame_queues.items():
                closest = self._find_closest_frame_set(queue, reference_timestamp)
                if closest is None:
                    return None  # Should not happen if reference_timestamp is valid

                synchronized_frame_sets[name] = closest
                time_delta = abs(closest.timestamp - reference_timestamp)
                max_time_delta = max(max_time_delta, time_delta)

        sensor_data: dict | None = None
        sensor_timestamp: float | None = None

        if self._imu_source is not None:
            imu_timestamp, closest_imu_data = self._find_closest_imu_data(self._imu_queue, reference_timestamp)
            if closest_imu_data is not None:
                sensor_data = closest_imu_data
                sensor_timestamp = imu_timestamp

        return SynchronizedFrameSet(
            timestamp=reference_timestamp,
            frame_sets=synchronized_frame_sets,
            max_time_delta=max_time_delta,
            sensor_data=sensor_data,
            sensor_timestamp=sensor_timestamp,
        )

    def get_latest_frames(self) -> dict[str, FrameSet] | None:
        """Get the latest frames from all cameras without synchronization.

        This returns the most recent FrameSet from each camera's queue,
        without any timestamp matching. Each FrameSet preserves the individual
        frame timestamps.

        Returns:
            Dictionary mapping source_name -> latest FrameSet, or None if any camera has no frames.
        """
        if not self._running:
            return None

        # Poll cameras for new frames first
        self._poll_cameras()

        result: dict[str, FrameSet] = {}
        with self._lock:
            for name, queue in self._frame_queues.items():
                if not queue:
                    logger.warning("Camera %s has no frames yet", name)
                    return None  # A camera has no frames yet
                result[name] = queue[-1]  # Get the most recent frame set

        return result

    def get_source_names(self) -> list[str]:
        """Get the names of all camera sources."""
        return list(self.sources.keys())

    def get_source(self, name: str) -> CameraSource | None:
        """Get a camera source by name."""
        return self.sources.get(name)

    def clear_queues(self) -> None:
        """Clear all frame queues."""
        with self._lock:
            for queue in self._frame_queues.values():
                queue.clear()

    def get_queue_depths(self) -> dict[str, int]:
        """Get the current depth of each frame queue."""
        with self._lock:
            return {name: len(queue) for name, queue in self._frame_queues.items()}

    def prune_old_frames(self, max_age_seconds: float = 1.0) -> int:
        """Remove frames older than max_age_seconds from all queues.

        Args:
            max_age_seconds: Maximum age of frames to keep.

        Returns:
            Number of frames pruned.
        """
        pruned_count = 0

        # Get current reference time (newest frame across all cameras)
        newest_timestamp = None
        with self._lock:
            for queue in self._frame_queues.values():
                if queue:
                    if newest_timestamp is None or queue[-1].timestamp > newest_timestamp:
                        newest_timestamp = queue[-1].timestamp

        if newest_timestamp is None:
            return 0

        cutoff_timestamp = newest_timestamp - max_age_seconds

        with self._lock:
            for queue in self._frame_queues.values():
                while queue and queue[0].timestamp < cutoff_timestamp:
                    queue.popleft()
                    pruned_count += 1

        return pruned_count
