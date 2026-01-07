"""Coordinating system for multiple cameras with frame synchronization."""

import logging
from collections import deque
from dataclasses import dataclass
from threading import Lock

from thor_slam.camera.types import CameraSource, Extrinsics, FrameSet, Intrinsics, SynchronizedFrameSet

logger = logging.getLogger(__name__)


@dataclass
class RigCalibration:
    """Calibration data for the entire camera rig."""

    intrinsics: dict[str, list[Intrinsics]]  # source_name -> [intrinsics per camera]
    extrinsics: dict[str, list[Extrinsics]]  # source_name -> [extrinsics per camera]


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
    _calibration: RigCalibration | None

    def __init__(
        self,
        sources: list[CameraSource],
        queue_size: int = 30,
    ) -> None:
        """Initialize the camera rig.

        Args:
            sources: List of camera sources to synchronize.
            queue_size: Maximum number of frame sets to keep in each queue.
        """
        self.sources = {src.name: src for src in sources}
        self.queue_size = queue_size
        self._frame_queues = {name: deque(maxlen=queue_size) for name in self.sources}
        self._lock = Lock()
        self._running = False
        self._calibration = None

    def start(self) -> None:
        """Start all camera sources."""
        if self._running:
            return

        for source in self.sources.values():
            source.start()
        self._running = True

        # Cache calibration data
        self._calibration = self._load_calibration()

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

    def _load_calibration(self) -> RigCalibration:
        """Load calibration data from all sources."""
        intrinsics: dict[str, list[Intrinsics]] = {}
        extrinsics: dict[str, list[Extrinsics]] = {}

        for name, source in self.sources.items():
            intrinsics[name] = source.get_intrinsics()
            extrinsics[name] = source.get_extrinsics()

        return RigCalibration(intrinsics=intrinsics, extrinsics=extrinsics)

    def get_calibration(self) -> RigCalibration:
        """Get the rig calibration data."""
        if self._calibration is None:
            self._calibration = self._load_calibration()
        return self._calibration

    def _poll_cameras(self) -> None:
        """Poll all cameras for new frames and add to queues."""
        for name, source in self.sources.items():
            frames = source.try_get_latest_frames()
            if frames:
                frame_set = FrameSet.from_frames(frames, source_name=name)
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
        5. Returns the synchronized frame set

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

        return SynchronizedFrameSet(
            timestamp=reference_timestamp,
            frame_sets=synchronized_frame_sets,
            max_time_delta=max_time_delta,
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
                    logger.warning(f"Camera {name} has no frames yet")
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
