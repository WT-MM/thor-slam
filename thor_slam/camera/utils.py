"""Package for discovering cameras on the network."""

import logging

import depthai as dai

from thor_slam.camera.types import IPv4

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
