"""Various utility functions for the SLAM pipeline."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import depthai as dai
import numpy as np
from scipy.spatial.transform import Rotation as R

from thor_slam.camera.types import Extrinsics, IPv4

logger = logging.getLogger(__name__)


def get_luxonis_devices_info() -> list[dai.DeviceInfo]:
    """Get all devices on the network."""
    return dai.Device.getAllAvailableDevices()


def get_luxonis_device(ip: IPv4) -> dai.Device | None:
    """Get a device by its IP address."""
    device_info = get_luxonis_devices_info()
    for info in device_info:
        if info.name == ip.ip:
            return dai.Device(info)
    logger.error(
        "Device with IP address %s not found. Possible IP addresses: %s",
        ip,
        ", ".join([info.name for info in device_info]),
    )
    return None


def get_luxonis_camera_valid_modes(device: dai.Device, socket: dai.CameraBoardSocket) -> list[dai.CameraSensorType]:
    """Get the valid modes for a camera."""
    features = device.getConnectedCameraFeatures()
    for feature in features:
        if feature.socket == socket:
            return [mode for mode in feature.supportedTypes]
    logger.warning("No valid modes found for device %s with socket %s", device.getMxId(), socket)
    return []


def get_luxonis_camera_valid_resolutions(device: dai.Device, socket: dai.CameraBoardSocket) -> list[tuple[int, int]]:
    """Get the valid resolutions for a camera."""
    features = device.getConnectedCameraFeatures()
    for feature in features:
        if feature.socket == socket:
            return [(config.width, config.height) for config in feature.configs]
    logger.warning("No valid resolutions found for device %s with socket %s", device.getMxId(), socket)
    return []


# TODO: write tests to make sure this is correct.
# e.g. make an onshape with something 1m in x, 0.5m in y, 0.25m in z and check the transform. Also check roll pitch yaw.
def parse_urdf_transform(joint_elem: ET.Element) -> np.ndarray:
    """Parses a 4x4 transform matrix from a URDF fixed joint origin."""
    origin = joint_elem.find("origin")
    if origin is None:
        logger.warning("Joint %s has no origin tag, assuming identity.", joint_elem.get("name"))
        return np.eye(4)

    # Parse XYZ translation
    xyz_str = origin.get("xyz", "0 0 0")
    xyz = np.array([float(x) for x in xyz_str.split()])

    # Parse RPY rotation (URDF standard: Fixed Axis XYZ)
    rpy_str = origin.get("rpy", "0 0 0")
    rpy = [float(x) for x in rpy_str.split()]

    # Convert Euler to Rotation Matrix
    # URDF uses extrinsic XYZ rotation (Roll around X, then Pitch around Y, then Yaw around Z)
    rotation = R.from_euler("xyz", rpy, degrees=False)
    rot_matrix = rotation.as_matrix()

    # Build 4x4 Homogeneous Matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = xyz

    return transform


def load_rig_extrinsics_from_urdf(urdf_path: str | Path, camera_map: dict[str, str]) -> dict[str, Extrinsics]:
    """Loads camera extrinsics from a URDF file based on link name matching.

    Args:
        urdf_path: Path to the .urdf with star topology.
        camera_map: A dictionary mapping Source Name to the URDF Link Name.
                    Example: { "192.168.1.101": "link_oak-d-pro_bracket_1" }

    Returns:
        Dictionary mapping Source Name -> Extrinsics object.
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found at {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    extrinsics_out: dict[str, Extrinsics] = {}

    for source_name, link_name in camera_map.items():
        found = False

        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is None:
                continue

            child_link_name = child.get("link")

            if link_name == child_link_name:
                # Verify parent is base_link (sanity check for star topology)
                parent = joint.find("parent")
                if parent is None or parent.get("link") != "base_link":
                    logger.warning("Skipping joint %s: parent is not base_link", joint.get("name"))
                    continue

                # Get the Transform (Base -> Source)
                t_world_source = parse_urdf_transform(joint)

                extrinsics_out[source_name] = Extrinsics.from_4x4_matrix(t_world_source)

                logger.info("Loaded extrinsics for %s (found link: %s)", source_name, child_link_name)
                found = True
                break

        if not found:
            logger.warning("Could not find URDF link matching '%s' for source %s", link_name, source_name)

    return extrinsics_out
