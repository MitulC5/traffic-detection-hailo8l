# Hailo-8L Inference Pipeline

This document details the full model compilation and inference pipeline for deploying a custom YOLOv8n traffic detection model on **Raspberry Pi 5 + Hailo-8L NPU**.

---

## Pipeline Overview

```
YOLOv8n (PyTorch)
    │
    ▼
ONNX Export (Ultralytics)
    │
    ▼
Hailo Parser  ──►  HAR (Hailo Archive)
    │
    ▼
Hailo Optimize  ──►  Quantized HAR
    │
    ▼
Hailo Compiler  ──►  HEF (Hailo Executable Format)
    │
    ▼
HailoRT Inference on Raspberry Pi 5
```

---

## 1. ONNX Export from YOLOv8n

Export the custom-trained YOLOv8n model to ONNX format using the Ultralytics library:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx", opset=11, imgsz=640)
```

**Key points:**
- Use `opset=11` for maximum Hailo DFC compatibility.
- The exported file will be named `best.onnx`.
- Ensure the model was trained with exactly **8 classes**: `pedestrian`, `bicycle`, `car`, `motorcycle`, `auto_rickshaw`, `bus`, `commercial_vehicle`, `truck`.

---

## 2. Hailo Parser — ONNX to HAR

The Hailo parser converts the ONNX model into Hailo's internal HAR (Hailo Archive) format.

```bash
hailo parser onnx best.onnx \
  --hw-arch hailo8l \
  --end-node-names \
    "/model/model.22/cv2.0/cv2.0.2/Conv" \
    "/model/model.22/cv3.0/cv3.0.2/Conv" \
    "/model/model.22/cv2.1/cv2.1.2/Conv" \
    "/model/model.22/cv3.1/cv3.1.2/Conv" \
    "/model/model.22/cv2.2/cv2.2.2/Conv" \
    "/model/model.22/cv3.2/cv3.2.2/Conv" \
  --har-path best_clean.har
```

### Why end-node selection matters

YOLOv8's detection head (`model.22`) contains 6 output convolution layers:
- **3 `cv2` branches** — bounding box regression heads (one per feature scale)
- **3 `cv3` branches** — classification heads (one per feature scale)

By explicitly selecting these end-nodes, we:
1. Strip out YOLOv8's built-in postprocessing (Detect layer)
2. Allow Hailo's own NMS postprocessing to be configured via the `.alls` script
3. Ensure the model graph is clean for quantization

---

## 3. Hailo Optimize — Quantization

The optimization step quantizes the model from FP32 to INT8 using calibration data.

```bash
hailo optimize best_clean.har \
  --hw-arch hailo8l \
  --calib-set-path calib_npy \
  --model-script nms_compile.alls \
  --output-har-path best_optimized.har
```

### Calibration Data Preparation

1. Select **50–100 representative images** from your training/validation set.
2. Convert them to NumPy arrays matching the model input shape:

```python
import cv2
import numpy as np
from pathlib import Path

input_dir = Path("calib_images")
output_dir = Path("calib_npy")
output_dir.mkdir(exist_ok=True)

for img_path in sorted(input_dir.glob("*.jpg")):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    np.save(output_dir / f"{img_path.stem}.npy", img)
```

### NMS Model Script (`nms.alls`)

```
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
quantization_param([conv42, conv53, conv63], force_range_out=[0.0, 1.0])
nms_postprocess(meta_arch=yolov8, engine=cpu, classes=8, nms_scores_th=0.7, nms_iou_th=0.45)
```

- **`normalization`**: Input pixel values are divided by 255 (0–1 range).
- **`quantization_param`**: Forces detection head output range to [0, 1] for accurate quantization.
- **`nms_postprocess`**: Configures CPU-based NMS with `meta_arch=yolov8`, 8 classes, and confidence/IOU thresholds.

---

## 4. Hailo Compiler — HAR to HEF

The compiler converts the optimized HAR into a HEF (Hailo Executable Format) binary that runs on the Hailo-8L hardware.

```bash
hailo compiler best_optimized.har \
  --hw-arch hailo8l \
  --output-dir .
```

The output `.hef` file is the deployable binary for the Hailo-8L accelerator.

---

## 5. HailoRT Inference on Raspberry Pi 5

### GStreamer Real-Time Pipeline

Use `inference/detection.py` for real-time camera/video inference via GStreamer:

```bash
python inference/detection.py
```

This uses the Hailo GStreamer detection app framework which handles:
- Camera/video source via GStreamer pipeline
- Hardware-accelerated inference on Hailo-8L
- Real-time detection overlay
- Frame-by-frame callback with detection results

### Runtime Dependencies

Ensure these are installed on the Raspberry Pi 5:
- **HailoRT** — Hailo runtime library
- **hailo-platform** — Python bindings for HailoRT
- **GStreamer 1.0** — For real-time pipeline
- **OpenCV** — For image processing and visualization

---

## Data Flow Summary

| Step | Tool | Input | Output |
|------|------|-------|--------|
| Export | Ultralytics | `best.pt` | `best.onnx` |
| Parse | `hailo parser` | `best.onnx` | `best_clean.har` |
| Optimize | `hailo optimize` | `best_clean.har` + `calib_npy/` | `best_optimized.har` |
| Compile | `hailo compiler` | `best_optimized.har` | `*.hef` |
| Infer | HailoRT / GStreamer | `*.hef` + image/video | Detections |
