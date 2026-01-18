# thor-slam

Multi-camera SLAM system for Jetson Thor with 4× Luxonis OAK cameras.

### Installation

Make a virtual environment and just `pip install -e .`.


### Python Dependencies

Install the package:
```bash
pip install -e .
```

### Code structure

```
thor-slam/
├── thor_slam/                    # Main package
│   ├── camera/                   # Camera handling and synchronization
│   │   ├── drivers/              # Camera driver implementations
│   │   │   └── luxonis.py       # Luxonis OAK camera driver (DepthAI)
│   │   ├── rig.py               # CameraRig: multi-camera synchronization
│   │   ├── types.py             # Camera types (Intrinsics, Extrinsics, CameraFrame, etc.)
│   │   └── utils.py             # Camera utilities (finding devices, parsing URDF, etc.)
│   └── slam/                    # SLAM interface and adapters
│       ├── adapters/
│       │   └── isaac_ros.py     # Isaac ROS Visual SLAM adapter
│       └── interface.py         # SLAM interface (abstract base class)
│
├── scripts/                      # deployment scripts
│   ├── run_pipeline.py          # Main pipeline: SLAM + RGB-D publishing for nvblox
│   ├── run_slam.py              # SLAM-only script (Isaac ROS Visual SLAM)
│   ├── find_cameras.py          # Find available Luxonis cameras
│   ├── set_ip.py                # Set IP address of Luxonis camera
│   └── publish_odom_tf.py      # Publish odometry as TF transforms
│
├── examples/                     # Example scripts for testing
│   ├── rgbd_stream.py           # Stream and visualize RGB-D data
│   ├── test_camera_driver.py     # Test Luxonis camera driver
│   ├── test_camera_rig.py        # Test multi-camera synchronization
│   ├── test_imu.py               # Test IMU data (single or synchronized)
│   ├── test_stream_resolutions.py # Test different camera resolutions
│   └── pull_extrinsics.py       # Extract camera extrinsics from URDF
│
├── launch/                       # ROS 2 launch files
│   ├── thor_visual_slam.launch.py  # Launch Isaac ROS Visual SLAM node
│   └── thor_nvblox.launch.py    # Launch nvblox mapping node
│
└── config/                       # Configuration files
    ├── slam_config.yaml          # Camera and SLAM configuration
    ├── thor_visual_slam.rviz     # RViz config for Visual SLAM
    └── thor_nvblox.rviz         # RViz config for nvblox
```

Core idea here is to abstract out different SLAM engines and camera sources so that it can be easily swapped out. It got really messy in trying to get cuVSLAM and nvblox to work with this system, but the core ideas are still there. 

In camera, the important things to look at are `rig.py` which shouldn't need to change for different camera sources (e.g. receiving frames over ROS) and `types.py` which defines all of the abstraction for camera sources.

The rest of the system is just configs and ai-generated scripts to run the system. 

**Key Components:**

- **`thor_slam/camera/drivers/luxonis.py`**: Main camera driver implementing the `CameraSource` interface. Handles:
  - Stereo and mono camera modes
  - RGB-D stream generation (RGB + depth from stereo)
  - IMU data collection
  - Camera intrinsics and extrinsics extraction
  - Independent resolution control for SLAM and RGB-D outputs

- **`thor_slam/camera/rig.py`**: `CameraRig` class for synchronizing multiple cameras:
  - Frame synchronization across cameras
  - Rig calibration management (intrinsics, extrinsics, rig poses)
  - Synchronized frame set generation for SLAM

- **`thor_slam/slam/interface.py`**: Abstract SLAM interface defining:
  - `SlamEngine` base class
  - `SlamPose` and `TrackingState` types
  - Methods for processing synchronized frame sets

- **`thor_slam/slam/adapters/isaac_ros.py`**: Isaac ROS Visual SLAM implementation:
  - ROS 2 node wrapper for cuVSLAM
  - Pose estimation and tracking state management
  - Coordinate frame transformations (RDF to FLU)

- **`scripts/run_pipeline.py`**: Main production script that:
  - Initializes multiple cameras from config
  - Publishes stereo frames for SLAM
  - Publishes RGB-D frames for nvblox (if enabled)
  - Manages camera lifecycle and error handling


### Useful tips

- Get a urdf from Onshape with [onshnap](https://github.com/WT-MM/onshnap)

Config used for test stand:
```json
    {
    "url": "https://cad.onshape.com/documents/e3ab336faaf474d905c762d1/w/f935c8f9134246c5618007f3/e/354246bd5503e2a6863246a1",
    "filetype": "mjcf",
    "create_centroid_links": true
    }
```

See `examples/pull_extrinsics.py` for an example of pulling extrinsics from the urdf.


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
- X down
- Y right
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


### Next Steps

- [ ] Tune camera intrinsics and rig extrinsics (properly fix all the cameras, represent it accurately in Onshape)
- [ ] (optional) experiment with masking the input camera streams to cuVSLAM to mask out pallets, people, and other vehicles. Otherwise, there's not really a way to finetune the feature extraction so will probably need to use a different VSLAM system. 
- [ ] Add nvblox to the pipeline
    - This will allow for 3D mapping of the environment. (cuVSLAM's mapping is not good enough for navigation)
    - nvblox builds on top of cuVSLAM's odometry.
    - Possibly this will require segmentation (people, forklifts, etc). There are examples of this on NVIDIA's documentation.
- [ ] Add navigation to the pipeline. Once nvblox is working, we can pull the occupancy grid and use it for 2d navigation


### Scratch
Download from `sudo apt-get install -y ros-jazzy-isaac-ros-visual-slam`


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
