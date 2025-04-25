# Vision-Based F1TENTH Navigation

This repository contains all code and assets for our autonomous F1TENTH vehicle project. The system enables real-time traffic sign detection and responsive control using a combination of ROS2, a custom-trained YOLOv11s model, and Jetson CUDA acceleration.

---

## Project Structure

```
project/
├── scripts/
│   ├── auto_drive_node.py           # Handles navigation, sign-triggered behaviors (e.g., U-turn, stop)
│   ├── object_detection_node.py     # Runs YOLO11s for real-time traffic sign detection
│   ├── parallel_park.py             # (Optional) Executes a parking maneuver in front of detected box
├── yolo11s_model/
│   └── model.pt                     # Custom-trained YOLO model
```

---

## Setup & Installation

1. **Create virtual environment:**
   ```bash
   sudo apt-get install virtualenv
   python3 -m virtualenv -p python3 team-a3
   source team-a3/bin/activate
   ```

2. **Environment Requirements (Jetson only):**
   - JetPack 5.1.4
   - CUDA 11.4
   - Python 3.8

   Confirm:
   ```bash
   nvcc --version
   python --version
   ls /usr/local | grep cuda
   ```

3. **Install dependencies:**
   ```bash
   export TORCH_INSTALL=<torch_url>
   python3 -m pip install --no-cache $TORCH_INSTALL

   export TORCH_VISION_INSTALL=<torchvision_url>
   python3 -m pip install --no-cache $TORCH_VISION_INSTALL

   pip install ultralytics onnx onnxslim

   export ONNX_INSTALL=<onnxruntime_url>
   python3 -m pip install --no-cache $ONNX_INSTALL

   sudo apt-get install libcudnn8 libcudnn8-dev
   ```
   The following URLs may be different depending on your environment:

   torch_url = https://pypi.jetson-ai-lab.dev/jp5/cu114/+f/4c1/d7a5d0ba92527/torch-2.2.0-cp38-cp38-linux_aarch64.whl#sha256=4c1d7a5d0ba92527c163ce9da74a2bdccce47541ef09a14d186e413a47337385

   torchvision_url = https://pypi.jetson-ai-lab.dev/jp5/cu114/+f/12c/2173bcd5255bd/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl#sha256=12c2173bcd5255bddad13047c573de24e0ce2ea47374c48ee8fb88466e021d2a

   onnxruntime_url = https://pypi.jetson-ai-lab.dev/jp5/cu114/+f/ed5/8a16480d70d91/onnxruntime_gpu-1.16.3-cp38-cp38-linux_aarch64.whl#sha256=ed58a16480d70d917494de4d99ad7d6f0855a5241751aa67f477f2061875022e

4. **Run nodes:**
   ```bash
   # Detection node (with LD_PRELOAD for OpenMP compatibility on Jetson)
   LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python3 object_detection_node.py

   # Driving node
   python3 auto_drive_node.py
   ```

---

## System Architecture

### Goal
Detect and react to traffic signs in real-time, using RGB and depth images for perception and LiDAR for local navigation.

### Modules

| Component                | Description |
|--------------------------|-------------|
| **YOLO11s**     | Custom-trained on 500+ images per class (Stop, Yield, U-Turn, Limit-20, Limit-100). |
| **Object Detection Node**| Subscribes to RGB and depth topics, runs inference on GPU, and publishes detected sign + depth. |
| **Auto Drive Node**      | Follows the largest LiDAR gap. Handles STOP, YIELD, U-TURN, and SPEED-LIMIT responses. |
| **Timer-Based Triggers** | Uses average speed and sign distance to calculate time-to-reach; action is triggered after cooldown. |
| **U-turn State Machine** | Multi-phase turning logic (approach, turn, 3-point-turn, wall follow). Disables detection during maneuver. |

---

## Known Bugs & Limitations

- **Speed Limit Confusion:** Model occasionally confuses "Limit-20" and "Limit-100" signs due to 0 in number. A known fix is keeping the sign on the right side of the track so the camera cannot see just the 0.
- **No temporal smoothing**: Detection is frame-by-frame; adding history could reduce flicker.
- **Yaw drift**: U-turn logic relies on odometry yaw which may drift slightly; future IMU fusion planned.
- **Emergency stops**: LiDAR-based emergency stop can be overly sensitive in narrow tracks.
- **Parallel park node** exists but is not yet integrated with the full driving pipeline.

---

## Future Work

- Add semantic segmentation and more sign types
- Use Model Predictive Control (MPC) for smoother speed adaptation
- Improve data variety (e.g., low-light, rain)
- Add ROS bag logging and performance dashboards
- Integrate ROS2 nav stack for global path planning