# thor-slam

Multi-camera SLAM system for Jetson Thor with 4× Luxonis OAK cameras.

## Installation

### System Dependencies

For the OAK viewer
```bash
sudo apt-get -y install \
    libclang-dev \
    libatk-bridge2.0 \
    libfontconfig1-dev \
    libfreetype6-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    libssl-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    libxkbcommon-dev \
    patchelf
```

### Python Dependencies

Install the package:
```bash
pip install -e .
```

### Useful tips

- Get a urdf from Onshape with [onshnap](https://github.com/WT-MM/onshnap)

To rviz the isaac_ros_visual_slam node, you can use the following command:
```bash
rviz2 -d $(ros2 pkg prefix
isaac_ros_visual_slam
--share)/rviz
/default.cfg.rviz
```


### Setting up isaac ros visual slam

[cuvslam repo](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam)

follow the isaac ros getting started [guide](https://nvidia-isaac-ros.github.io/getting_started/index.html)

then follow the vslam specific [guide](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_visual_slam/isaac_ros_visual_slam/index.html#quickstart)

then run `isaac-ros activate` to enter the isaac ros environment 

and run `make isaac-ros-launch` to launch the isaac ros visual slam node

then run `make slam-run` to start the camera bridge

you can run `make rviz` to launch the rviz2 visualizer

### Luxonis coordinate convention

Luxonis cameras use the following coordinate convention:

- X: Right
- Y: Down
- Z: Forward (direction of camera outwards)

However the IMU convention is dependent on camera model.

The OAK D Pro uses the following:
- X up
- Y left
- Z back

The Oak D Long Range uses:

- X right
- Y down
- Z forward

isaac ros uses:
 +x = forward
 +y = left
 +z = up # check this (https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_mapping_and_localization/isaac_ros_visual_global_localization/index.html)

 Camera optical frames (for cuVSLAM) need to be
 +z forward (out through lens)
 +y down
 +x right

 So alignment is same for camera to base_link, but depending on camera type need to rotate the imu data to match the camera optical frame.

 Furthermore, need to make sure urdf coordinates are aligned with the RDF convention.


### Scratch
```python

# luxonis in RDF, cuvslam in FLU
rdf_to_flu_matrix = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
])

# usage
point_in_rdf = np.array([1, 0, 0, 1])
point_in_flu = rdf_to_flu_matrix @ point_in_rdf
print(point_in_flu)

```
