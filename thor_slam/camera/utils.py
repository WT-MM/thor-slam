"""Package for discovering cameras on the network."""

import logging

import depthai as dai

from thor_slam.camera.types import IPv4

logger = logging.getLogger(__name__)


def get_devices_info() -> list[dai.DeviceInfo]:
    """Get all devices on the network."""
    return dai.Device.getAllAvailableDevices()


def get_device(ip: IPv4) -> dai.Device | None:
    """Get a device by its IP address."""
    device_info = get_devices_info()
    for info in device_info:
        if info.name == ip.ip:
            return dai.Device(info)
    logger.error(
        "Device with IP address %s not found. Possible IP addresses: %s",
        ip,
        ", ".join([info.name for info in device_info]),
    )
    return None
