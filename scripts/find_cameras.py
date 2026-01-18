"""Script to find cameras on the network."""

import depthai as dai

"""
Workbench ip

DLR
- 192.168.2.91
D Pro
- 192.168.2.190
- 192.168.2.129
- 192.168.2.72

"""

infos = dai.Device.getAllAvailableDevices()
print(f"Found {len(infos)} devices")
for info in infos:
    print(info)
