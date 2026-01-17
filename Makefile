# Makefile

.PHONY: all install install-dev format static-checks test \
        isaac-ros-launch slam-run pipeline-run ros2-topics clean \
        nvblox-install nvblox-launch nvblox-rviz

# ============================================ #
#                    Help                      #
# ============================================ #

all:
	@echo "thor-slam - Visual SLAM for Thor robot"
	@echo ""
	@echo "Quick Start (SLAM only):"
	@echo "  Terminal 1:  make isaac-ros-launch              # Start Isaac ROS (2 cams)"
	@echo "  Terminal 2:  make slam-run                     # Start camera bridge"
	@echo ""
	@echo "Quick Start (SLAM + nvblox):"
	@echo "  Terminal 1:  make isaac-ros-launch              # Start Isaac ROS"
	@echo "  Terminal 2:  make pipeline-run                 # Start camera bridge + RGB-D"
	@echo "  Terminal 3:  make nvblox-launch NVBLOX_NUM_CAMERAS=1  # Start nvblox"
	@echo "  Terminal 4:  make nvblox-rviz                  # Visualize nvblox"
	@echo ""
	@echo "Note: Configure nvblox_cameras in config/slam_config.yaml to specify"
	@echo "      which cameras publish RGB-D data for nvblox"
	@echo ""
	@echo "Multi-camera:"
	@echo "  make isaac-ros-launch NUM_CAMERAS=4"
	@echo "  make pipeline-run NUM_CAMERAS=4"
	@echo ""
	@echo "Commands:"
	@echo "  make isaac-ros-launch  - Launch Isaac ROS Visual SLAM"
	@echo "  make slam-run          - Run camera bridge (SLAM only)"
	@echo "  make pipeline-run     - Run camera bridge + RGB-D (for nvblox)"
	@echo "  make odom-tf           - Publish map->odom TF transform"
	@echo "  make nvblox-install   - Install nvblox (Isaac ROS)"
	@echo "  make nvblox-launch    - Launch nvblox"
	@echo "  make nvblox-rviz      - Launch RViz for nvblox"
	@echo "  make ros2-topics       - List ROS2 topics"
	@echo "  make format            - Format code"
	@echo "  make static-checks     - Run linters"
	@echo "  make test              - Run tests"

# ============================================ #
#                Configuration                 #
# ============================================ #

# Number of cameras (2 for stereo, 4 for two stereo pairs, etc.)
NUM_CAMERAS ?= 2

# Camera IP(s) - comma separated for multiple
CAMERA ?= 192.168.2.21

# FPS
FPS ?= 30

STEREO ?= true

STEREO_FLAG :=
ifeq ($(STEREO),true)
  STEREO_FLAG := --stereo
else
  STEREO_FLAG := --no-stereo
endif

# ============================================ #
#              Isaac ROS Launch                #
# ============================================ #

# Launch Isaac ROS Visual SLAM node
# Topics: /visual_slam/image_0..N, /visual_slam/camera_info_0..N
isaac-ros-launch:
	@echo "Launching Isaac ROS Visual SLAM ($(NUM_CAMERAS) cameras)..."
	@echo "Expected topics:"
	@for i in $$(seq 0 $$(($(NUM_CAMERAS) - 1))); do echo "  /visual_slam/image_$$i"; done
	@echo ""
	ros2 launch ./launch/thor_visual_slam.launch.py \
		num_cameras:=$(NUM_CAMERAS) \
		enable_slam_visualization:=true \
		rectified_images:=false \
		enable_imu_fusion:=true \
		enable_landmarks_view:=true \
		enable_observations_view:=true \
		enable_localization_n_mapping:=true \
		enable_debug_mode:=false \
		verbosity:=1 \
		image_sync_threshold_ms:=100.0 \

# ============================================ #
#              Thor SLAM Bridge                #
# ============================================ #

install-force:
	pip install --break-system-packages .

# Run the camera bridge (SLAM only)
slam-run:
	python -m scripts.run_slam

# Run the camera bridge with RGB-D publishing (for nvblox)
pipeline-run:
	python -m scripts.run_pipeline

# Publish odom TF transform (run in separate terminal)
odom-tf:
	@echo "Publishing map->odom transform from visual SLAM odometry..."
	python -m scripts.publish_odom_tf

# ============================================ #
#                 ROS2 Utils                   #
# ============================================ #

ros2-topics:
	@echo "Visual SLAM topics:"
	@ros2 topic list 2>/dev/null | grep visual_slam || echo "  (none - is Isaac ROS running?)"
	@echo ""
	@echo "RGB-D topics (for nvblox):"
	@ros2 topic list 2>/dev/null | grep -E "(camera_[0-9]+/(rgb|depth)|nvblox)" || echo "  (none - is pipeline running?)"

rviz:
	@echo "Launching RViz with Visual SLAM visualization..."
	@if [ -f ./config/thor_visual_slam.rviz ]; then \
		ros2 run rviz2 rviz2 -d ./config/thor_visual_slam.rviz --ros-args -r /visual_slam/camera_info:=/visual_slam/camera_info_6; \
	else \
		echo "Warning: Config file not found at ./config/thor_visual_slam.rviz"; \
		ros2 run rviz2 rviz2 --ros-args -r /visual_slam/camera_info:=/visual_slam/camera_info_6; \
	fi

ros2-hz:
	@echo "Checking image_0 rate..."
	@timeout 3 ros2 topic hz /visual_slam/image_0 2>/dev/null || echo "No data"

ros2-odom:
	ros2 topic echo /visual_slam/tracking/odometry

# ============================================ #
#                  nvblox                      #
# ============================================ #

# Install nvblox (requires Isaac ROS environment)
nvblox-install:
	@echo "Installing nvblox system package using apt and rosdep..."
	@echo "This will install isaac_ros_nvblox and nvblox_examples_bringup packages..."
	@sudo apt update && \
	sudo apt-get install -y ros-jazzy-isaac-ros-nvblox ros-jazzy-nvblox-examples-bringup && \
	rosdep update && \
	rosdep install -y -r --from-paths /opt/ros/jazzy/share/isaac_ros_nvblox --ignore-src || true
	@if ! command -v ros2 >/dev/null 2>&1; then \
		echo ""; \
		echo "Warning: ROS2 not found. Please source your ROS2 installation."; \
		exit 1; \
	fi
	@if ! ros2 pkg list 2>/dev/null | grep -q nvblox_examples_bringup && ! ros2 pkg list 2>/dev/null | grep -q isaac_ros_nvblox; then \
		echo ""; \
		echo "Warning: nvblox packages not found. Please check installation steps above."; \
		echo "You may need to build from source in your Isaac ROS workspace."; \
		exit 1; \
	fi
	@echo "✓ nvblox installation check complete"

# Number of RGB-D cameras for nvblox (can be different from SLAM cameras)
NVBLOX_NUM_CAMERAS ?= 1

# Launch nvblox
# Topics expected: /camera_0/rgb/image_raw, /camera_0/depth/image_raw, etc.
# Note: NUM_CAMERAS here refers to RGB-D cameras, not SLAM cameras
nvblox-launch:
	@echo "Launching nvblox ($(NVBLOX_NUM_CAMERAS) RGB-D camera(s))..."
	@echo "Expected topics:"
	@for i in $$(seq 0 $$(($(NVBLOX_NUM_CAMERAS) - 1))); do \
		echo "  /camera_$$i/rgb/image_raw, /camera_$$i/depth/image_raw"; \
	done
	@echo ""
	@echo "Note: Make sure pipeline-run is publishing RGB-D data for these cameras"
	@echo "      (configure nvblox_cameras in config/slam_config.yaml)"
	@echo ""
	@if ! command -v ros2 >/dev/null 2>&1; then \
		echo "Error: ROS2 not found. Please source your ROS2 installation."; \
		exit 1; \
	fi
	@if ! ros2 pkg list 2>/dev/null | grep -q isaac_ros_nvblox; then \
		echo "Error: isaac_ros_nvblox package not found."; \
		echo "Run 'make nvblox-install' for installation instructions."; \
		exit 1; \
	fi
	@echo "Launching nvblox using our custom launch file..."
	ros2 launch ./launch/thor_nvblox.launch.py \
		num_cameras:=$(NVBLOX_NUM_CAMERAS) \
		map_frame:=map \
		global_frame:=odom \
		voxel_size:=0.05 \
		tsdf_integrator_max_integration_distance_m:=10.0 \
		tsdf_integrator_truncation_distance_vox:=4.0

# Launch RViz for nvblox visualization
nvblox-rviz:
	@echo "Launching RViz with nvblox visualization..."
	@if [ -f ./config/thor_nvblox.rviz ]; then \
		ros2 run rviz2 rviz2 -d ./config/thor_nvblox.rviz; \
	else \
		echo "Warning: Config file not found at ./config/thor_nvblox.rviz"; \
		ros2 run rviz2 rviz2; \
	fi

# ============================================ #
#               Development                    #
# ============================================ #

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

py-files := $(shell find scripts thor_slam -name '*.py' 2>/dev/null)

format:
	@black $(py-files)
	@ruff format $(py-files)

static-checks:
	@black --diff --check $(py-files)
	@ruff check $(py-files)
	@mypy --install-types --non-interactive $(py-files)

test:
	python -m pytest

clean:
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
