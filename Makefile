# Makefile

.PHONY: all install install-dev format static-checks test \
        isaac-ros-launch slam-run ros2-topics clean

# ============================================ #
#                    Help                      #
# ============================================ #

all:
	@echo "thor-slam - Visual SLAM for Thor robot"
	@echo ""
	@echo "Quick Start:"
	@echo "  Terminal 1:  make isaac-ros-launch              # Start Isaac ROS (2 cams)"
	@echo "  Terminal 2:  make slam-run CAMERA=192.168.2.21  # Start camera bridge"
	@echo ""
	@echo "Multi-camera:"
	@echo "  make isaac-ros-launch NUM_CAMERAS=4"
	@echo "  make slam-run CAMERA=192.168.2.21,192.168.2.22 NUM_CAMERAS=4"
	@echo ""
	@echo "Commands:"
	@echo "  make isaac-ros-launch  - Launch Isaac ROS Visual SLAM"
	@echo "  make slam-run          - Run camera bridge"
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

# ============================================ #
#              Thor SLAM Bridge                #
# ============================================ #

# Run the camera bridge
slam-run:
	python -m scripts.run_slam --camera-ips $(CAMERA) --num-cameras $(NUM_CAMERAS) --fps $(FPS)

slam-run-headless:
	python -m scripts.run_slam --camera-ips $(CAMERA) --num-cameras $(NUM_CAMERAS) --fps $(FPS) --no-display

# ============================================ #
#                 ROS2 Utils                   #
# ============================================ #

ros2-topics:
	@echo "Visual SLAM topics:"
	@ros2 topic list 2>/dev/null | grep visual_slam || echo "  (none - is Isaac ROS running?)"

rviz:
	@echo "Launching RViz with Visual SLAM visualization..."
	@if [ -f ./config/thor_visual_slam.rviz ]; then \
		rviz2 -d ./config/thor_visual_slam.rviz; \
	else \
		echo "Warning: Config file not found at ./config/thor_visual_slam.rviz"; \
		rviz2; \
	fi

ros2-hz:
	@echo "Checking image_0 rate..."
	@timeout 3 ros2 topic hz /visual_slam/image_0 2>/dev/null || echo "No data"

ros2-odom:
	ros2 topic echo /visual_slam/tracking/odometry

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
