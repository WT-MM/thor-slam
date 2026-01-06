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
