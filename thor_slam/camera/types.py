"""Types for the camera package."""

import re


class IPv4(str):
    """A class for representing an IPv4 address."""

    _ip: str

    def __init__(self, ip: str) -> None:
        if not re.match(r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$", ip):
            raise ValueError(f"Invalid IPv4 address: {ip}")
        self._ip = ip

    def __str__(self) -> str:
        return self.ip

    @property
    def ip(self) -> str:
        return self._ip
