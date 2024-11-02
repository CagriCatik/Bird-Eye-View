# Bird Eye View

## Overview

**Bird Eye View (BEV)** is a pivotal concept in computer vision and autonomous driving, enabling systems to perceive and interpret the environment from a top-down perspective. This project delves into various methodologies for generating and utilizing BEV representations, ranging from traditional geometric techniques to cutting-edge deep learning approaches. Whether you're interested in depth-based transformations, geometric algorithms, or integrating BEV into autonomous driving pipelines, this repository offers comprehensive resources, implementations, and insights to advance your projects.

## Table of Contents

### 1. Depth-Based & Geometric-Based Bird Eye View
- [Bird Eye View on the Camera](#bird-eye-view-on-the-camera)
- [Bird Eye View from Depth Maps & Feature Lifting](#bird-eye-view-from-depth-maps-feature-lifting)
- [Bird Eye View from Geometry](#bird-eye-view-from-geometry)
- [IPM from the 4-Point Algorithm](#ipm-from-the-4-point-algorithm)
- [IPM from Parameters](#ipm-from-parameters-part-1)
- [Traditional BEV Depth](#traditional-bev-depth)

### 2. Bird Eye View with Deep Learning
- [Deep Learning Approaches (Geometric)](#deep-learning-approaches-geometric)
- [Pure Neural Network Approaches & Summary](#pure-neural-network-approaches-summary)
- [Cam2BEV Project Starter & Dataset](#cam2bev-project-starter-dataset)
- [Custom Dataset & PyTorch](#custom-dataset-pytorch)
- [Encoder: Deep Eagle View](#encoder-deep-eagle-view)
- [Model Assembly & Training](#model-assembly-training)
- [Inference & Spatial Visualization](#inference-spatial-visualization)
- [Feature Lifting](#feature-lifting)

### 3. From Bird Eye View to Autonomous Driving
- [Bird Eye View Techniques](#game-bird-eye-view-techniques)
- [Lane Line Detection](#lane-line-detection)
- [BEV Object Detection](#bev-object-detection)
- [From Bird Eye View to HD Maps](#from-bird-eye-view-to-hd-maps)
- [Motion Planning on Bird Eye Views](#motion-planning-on-bird-eye-views)
- [Sensor Fusion in the BEV Space](#sensor-fusion-in-the-bev-space)
- [360 Vision & Summary](#360-vision-summary)

## Getting Started

### Prerequisites

- Additional dependencies listed in `requirements.txt`

### Installation
```bash
git clone https://github.com/CagriCatik/bird-eye-view.git
cd bird-eye-view
pip install -r requirements.txt
```

### Usage
Detailed instructions on how to utilize the various modules and scripts provided in the project.

```bash
# Example: Running a training script
python train.py --config config/train_config.yaml
```
