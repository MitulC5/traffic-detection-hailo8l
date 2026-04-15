<!-- Project Badges -->
<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-blue?style=for-the-badge&logo=yolo" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/Hailo--8L-NPU-00C853?style=for-the-badge" alt="Hailo-8L"/>
  <img src="https://img.shields.io/badge/Raspberry%20Pi%205-A22846?style=for-the-badge&logo=raspberrypi&logoColor=white" alt="Raspberry Pi 5"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX"/>
  <img src="https://img.shields.io/badge/GStreamer-Pipeline-orange?style=for-the-badge" alt="GStreamer"/>
  <img src="https://img.shields.io/badge/Edge%20AI-Inference-ff6f00?style=for-the-badge" alt="Edge AI"/>
</p>

---

# 🚦 Traffic Detection — Hailo-8L

**Custom 8-class traffic object detection** using YOLOv8n, compiled and deployed on **Raspberry Pi 5 + Hailo-8L NPU** accelerator.

Achieve real-time traffic detection at the edge — pedestrians, vehicles, bicycles, and more — with hardware-accelerated inference powered by the Hailo-8L neural processing unit.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Detection Classes](#detection-classes)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Folder Structure](#folder-structure)
- [Setup & Usage](#setup--usage)
  - [Step 1: ONNX Export](#step-1-onnx-export)
  - [Step 2: Parse](#step-2-parse)
  - [Step 3: Optimize](#step-3-optimize)
  - [Step 4: Compile](#step-4-compile)
  - [Step 5: Inference](#step-5-inference)
- [Current Status](#current-status)
- [Known Issues](#known-issues)
- [License](#license)

---

## Overview

This project implements an end-to-end pipeline for deploying a custom-trained YOLOv8n model on edge hardware:

1. **Train** YOLOv8n on a custom 8-class traffic dataset
2. **Export** to ONNX format
3. **Compile** using Hailo Dataflow Compiler (DFC) with NMS postprocessing
4. **Deploy** on Raspberry Pi 5 with Hailo-8L NPU for real-time inference

The entire inference runs on the Hailo-8L accelerator with CPU-based NMS postprocessing, delivering efficient real-time detection suitable for traffic monitoring, smart city, and ADAS prototyping applications.

---

## Detection Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | `pedestrian` | People walking or standing |
| 1 | `bicycle` | Bicycles with or without riders |
| 2 | `car` | Passenger cars and sedans |
| 3 | `motorcycle` | Two-wheeled motorized vehicles |
| 4 | `auto_rickshaw` | Three-wheeled auto rickshaws |
| 5 | `bus` | Public and private buses |
| 6 | `commercial_vehicle` | Vans, delivery vehicles, utility trucks |
| 7 | `truck` | Heavy commercial trucks |

---

## Hardware & Software Requirements

### Hardware
| Component | Specification |
|-----------|---------------|
| **SBC** | Raspberry Pi 5 (8GB RAM) |
| **NPU** | Hailo-8L M.2 AI Accelerator |
| **M.2 HAT** | Raspberry Pi AI HAT / M.2 HAT+ |
| **Storage** | 32GB+ microSD (Class 10/U3) |
| **Camera** | USB webcam or Raspberry Pi Camera Module |

### Software
| Tool | Version | Purpose |
|------|---------|---------|
| **Raspberry Pi OS** | Bookworm (64-bit) | Operating system |
| **Python** | 3.10+ | Runtime |
| **HailoRT** | 4.18+ | Hailo runtime |
| **Hailo DFC** | 3.28+ | Model compiler (x86 host) |
| **GStreamer** | 1.0 | Real-time pipeline |
| **OpenCV** | 4.x | Image processing |
| **Ultralytics** | 8.x | YOLOv8 training & export |

> **Note:** The Hailo Dataflow Compiler (DFC) runs on an **x86 Linux host** (Ubuntu 20.04/22.04). The compiled `.hef` file is then transferred to the Raspberry Pi for inference.

---

## Folder Structure

```
traffic-detection-hailo8l/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── inference/                   # Inference scripts
│   ├── detection.py             # GStreamer real-time pipeline
│   └── labels.json              # Class labels and config
│
├── compile/                     # Hailo DFC compilation scripts
│   ├── parse.sh                 # ONNX → HAR conversion
│   ├── optimize.sh              # Quantization + calibration
│   ├── compile_nms.sh           # HAR → HEF compilation
│   └── nms.alls                 # NMS model script
│
├── docs/                        # Documentation
│   ├── pipeline.md              # Full pipeline explanation
│   └── troubleshooting.md       # Known issues & solutions
│
└── assets/                      # Images, diagrams, demo outputs
```

---

## Setup & Usage

### Prerequisites

1. **Train your YOLOv8n model** on your custom 8-class traffic dataset using Ultralytics.
2. **Install Hailo DFC** on an x86 Linux machine (compilation host).
3. **Install HailoRT** on your Raspberry Pi 5.
4. **Set up Hailo-8L** hardware (M.2 HAT, driver, firmware).

### Step 1: ONNX Export

Export your trained YOLOv8n model to ONNX:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx", opset=11, imgsz=640)
```

### Step 2: Parse

Convert the ONNX model to Hailo Archive (HAR) format with explicit end-node selection:

```bash
cd compile/
bash parse.sh
```

This strips YOLOv8's built-in Detect layer and selects the 6 detection head convolution outputs.

### Step 3: Optimize

Quantize the model using calibration data:

```bash
# First, prepare calibration images as .npy files
# (see docs/troubleshooting.md for conversion script)

bash optimize.sh
```

### Step 4: Compile

Compile the optimized HAR into a deployable HEF binary:

```bash
bash compile_nms.sh
```

Transfer the output `.hef` file to your Raspberry Pi 5.

### Step 5: Inference

Run real-time detection using the GStreamer pipeline:

```bash
python inference/detection.py
```

---

## Current Status

| Milestone | Status |
|-----------|--------|
| Custom dataset collection | ✅ Complete |
| YOLOv8n training (8 classes) | ✅ Complete |
| ONNX export | ✅ Complete |
| Hailo DFC parse | ✅ Complete |
| Hailo DFC optimize | ✅ Complete |
| Hailo DFC compile (HEF) | ✅ Complete |
| NMS configuration | ✅ Configured |
| GStreamer inference testing | 🔄 In Progress |
| GStreamer pipeline testing | 🔄 In Progress |
| Real-time performance benchmarking | ⬚ Planned |

---

## Known Issues

| Issue | Details | Workaround |
|-------|---------|------------|
| **End-node selection** | Must specify 6 YOLOv8 detection head nodes in parser | Use exact node names from `model.22` |
| **NMS engine** | `engine=npu` unsupported on Hailo-8L for YOLOv8 | Use `engine=cpu` |
| **HEF output naming** | Filename derived from internal network name | Check compiler logs, rename after build |
| **Calibration format** | `hailo optimize` requires `.npy`, not `.jpg` | Convert using cv2 + numpy script |
| **Calibration path** | Relative path resolved from CWD | Use absolute path or run from correct dir |

> 📖 See [docs/troubleshooting.md](docs/troubleshooting.md) for detailed solutions.

---

## License

This project is provided as-is for educational and research purposes.

---

## ⚠️ Dataset Label Bug (Fixed)

The DriveIndia dataset contains a serialization bug where YOLO annotation
files use literal `\n` escape sequences instead of real newlines. This caused
Ultralytics to silently discard **85% of training labels** (20,217 of 23,718
train images, 2,035 of 2,323 val images).

Root cause: the dataset export script used `"\\n".join(rows)` instead of
`"\n".join(rows)`, writing 2-character escape sequences instead of actual
newline bytes.

Run before training:
```bash
python dataset/fix_labels.py
python dataset/fix_image_symlinks.py
```

Note: image directories must be real directories (not symlinks) or Ultralytics
resolves the symlink back to the original corrupt label path.

---

## Training

Trained on Kaggle (2× Tesla T4 GPU, DDP) on the fixed 8-class dataset.

| Setting | Value |
|---|---|
| Model | YOLOv8n |
| Train images | 23,718 (after fix) |
| Val images | 2,323 (after fix) |
| Epochs | 100 |
| Batch | 64 (32 per GPU) |
| Image size | 640 |
| Optimizer | SGD |
| mAP@0.5 | 76%+ |

```bash
python training/train.py
```

---

<p align="center">
  <b>Built with ❤️ for Edge AI</b><br/>
  Raspberry Pi 5 • Hailo-8L • YOLOv8 • GStreamer
</p>
