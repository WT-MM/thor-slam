from pathlib import Path

from thor_slam.camera.drivers.luxonis import LuxonisCameraConfig, LuxonisCameraSource, LuxonisResolution
from thor_slam.camera.rig import CameraRig
from thor_slam.camera.utils import load_rig_extrinsics_from_urdf

camera_map = {
    "192.168.2.25": "link_Camera_1_centroid",  # Front low cam
    "192.168.2.21": "link_Camera_2_centroid",  # Right cam
    "192.168.2.23": "link_Camera_3_centroid",  # Up cam
    "192.168.2.22": "link_Camera_4_centroid",  # Left cam
}

# Front, right, top, left
# camera1 = front, camera2 = right, camera3 = top, camera4 = left

urdf_path = Path(__file__).parent / "assets" / "brackets.urdf"

rig_extrinsics = load_rig_extrinsics_from_urdf(
    urdf_path=urdf_path,
    camera_map=camera_map
)

my_sources = [
    LuxonisCameraSource(cfg=LuxonisCameraConfig(ip="192.168.2.21", resolution=LuxonisResolution.from_name("720"), stereo=True, fps=10)),
    LuxonisCameraSource(cfg=LuxonisCameraConfig(ip="192.168.2.22", resolution=LuxonisResolution.from_name("720"), stereo=True, fps=10)),
    LuxonisCameraSource(cfg=LuxonisCameraConfig(ip="192.168.2.23", resolution=LuxonisResolution.from_name("1200"), stereo=True, fps=10, camera_mode="COLOR")),
    LuxonisCameraSource(cfg=LuxonisCameraConfig(ip="192.168.2.25", resolution=LuxonisResolution.from_name("720"), stereo=True, fps=10)),
]

# 3. Initialize Rig
rig = CameraRig(
    sources=my_sources,
    rig_extrinsics=rig_extrinsics
)

# Verify
world_ext = rig.get_world_extrinsics("192.168.2.21")
print(f"Right Camera World Pose:\n{world_ext[0].to_4x4_matrix()}")
