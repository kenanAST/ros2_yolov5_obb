# ROS2 YOLOv5_OBB

This repository contains a ROS2 implementation of the YOLOv5_OBB object detection algorithm. YOLOv5_OBB is an extension of the YOLO (You Only Look Once) algorithm that specifically focuses on detecting objects with oriented bounding boxes (OBB). The implementation leverages the power of the Robot Operating System 2 (ROS2) framework to enable seamless integration with robotic systems.

## Features

- Real-time object detection using YOLOv5_OBB algorithm
- Supports detection of objects with oriented bounding boxes
- ROS2 node implementation for easy integration with ROS2-based robotic systems

## Installation

### Prerequisites

- Linux **(Recommend)**, Windows **(not Recommend, Please refer to this [issue](https://github.com/hukaixuan19970627/yolov5_obb/issues/224) if you have difficulty in generating utils/nms_rotated_ext.cpython-XX-XX-XX-XX.so)**
- Python 3.7+
- PyTorch ≥ 1.7
- CUDA 11.7
- ROS2 (tested with Galactic)

### Usage

a. Install PyTorch.

```
pip3 install torch torchvision torchaudio
```

b. Clone the ros2-yolov5-obb repository.

```
git clone https://github.com/kenanAST/ros2_yolov5_obb.git
cd ros2_yolov5_obb
```

c. Install ros2_yolov5_obb dependencies.

```python
pip install -r requirements.txt
cd utils/nms_rotated
python setup.py develop  #or "pip install -v -e ."
```

d. Run Inference on Image Topic

```
python ros_recognition_yolo_obb.py
```
