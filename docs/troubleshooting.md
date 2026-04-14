# Troubleshooting Guide

Common issues encountered during development of the Hailo-8L YOLOv8n traffic detection pipeline, along with solutions.

---

## 1. End Node Selection for YOLOv8n in Hailo Parser

### Problem
Running `hailo parser` without specifying `--end-node-names` includes YOLOv8's built-in `Detect` postprocessing layer in the graph. This causes conflicts with Hailo's own NMS postprocessing and may result in compilation errors or incorrect output tensor shapes.

### Solution
Explicitly specify the 6 detection head output convolution layers:

```bash
--end-node-names \
  "/model/model.22/cv2.0/cv2.0.2/Conv" \
  "/model/model.22/cv3.0/cv3.0.2/Conv" \
  "/model/model.22/cv2.1/cv2.1.2/Conv" \
  "/model/model.22/cv3.1/cv3.1.2/Conv" \
  "/model/model.22/cv2.2/cv2.2.2/Conv" \
  "/model/model.22/cv3.2/cv3.2.2/Conv"
```

### How to find end-node names
Use Netron (https://netron.app) to visualize your ONNX model and identify the last Conv nodes before the Detect layer. For standard YOLOv8n, these are always under `model.22`.

---

## 2. Why `meta_arch=yolov8` (Not `yolov8n`)

### Problem
When configuring NMS in the `.alls` script, you might expect to use `meta_arch=yolov8n` since the model is YOLOv8**n**. However, this value is not recognized by the Hailo DFC.

### Explanation
In Hailo's NMS configuration, `meta_arch` refers to the **architecture family**, not the specific model variant:
- `yolov8` covers **all** YOLOv8 variants: `n`, `s`, `m`, `l`, `x`
- The `n` in YOLOv8n only denotes the model **size** (nano), not a different architecture

### Correct usage
```
nms_postprocess(meta_arch=yolov8, engine=cpu, classes=8, nms_scores_th=0.7, nms_iou_th=0.45)
```

---

## 3. Why `engine=cpu` for YOLOv8 NMS on Hailo-8L

### Problem
You might expect NMS to run on the Hailo-8L NPU for maximum performance, but using `engine=npu` causes errors or incorrect results.

### Explanation
The Hailo-8L (a lower-power variant of Hailo-8) has limited support for on-chip NMS operations, particularly for architectures like YOLOv8. The `engine=cpu` setting offloads the NMS postprocessing to the host CPU (Raspberry Pi 5's ARM Cortex-A76), which:

1. **Is fully supported** — CPU NMS works reliably across all Hailo hardware variants
2. **Has negligible overhead** — NMS is a lightweight operation compared to the neural network inference itself
3. **Provides flexibility** — Easier to tune thresholds and debug detection outputs

### When to use `engine=npu`
Only for architectures that have verified NPU-based NMS support on Hailo-8L (check the Hailo Model Zoo for compatibility).

---

## 4. HEF Output Path and Naming Issues During Compilation

### Problem
After running `hailo compiler`, the output `.hef` file may not appear where expected, or may have an unexpected filename derived from the internal network name rather than the input file.

### Solution
1. Always specify `--output-dir .` to control the output location:
   ```bash
   hailo compiler best_optimized.har --hw-arch hailo8l --output-dir .
   ```
2. The output filename is derived from the **network name inside the HAR**, not the HAR filename. Check the compiler output log for the actual `.hef` filename.
3. If you need a specific name, rename the file after compilation:
   ```bash
   mv *.hef traffic_detection_yolov8n.hef
   ```

---

## 5. Converting JPGs to NPY for Optimization (Calibration Data)

### Problem
The `hailo optimize` command requires calibration data in NumPy `.npy` format, but your dataset images are in `.jpg`/`.png` format.

### Solution
Use this Python script to convert calibration images:

```python
import cv2
import numpy as np
from pathlib import Path

input_dir = Path("calib_images")
output_dir = Path("calib_npy")
output_dir.mkdir(exist_ok=True)

for img_path in sorted(input_dir.glob("*.jpg")):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (640, 640))          # Match model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR → RGB
    np.save(output_dir / f"{img_path.stem}.npy", img)

print(f"Converted {len(list(output_dir.glob('*.npy')))} images to .npy format")
```

### Key requirements
- **Image size must be 640×640** (or whatever your model's input resolution is)
- **Color space must be RGB** (OpenCV loads as BGR by default)
- Use **50–100 representative images** from your training/validation set
- Images should cover all 8 detection classes for balanced calibration

---

## 6. `calib_images` Path Error During Optimize Step

### Problem
Running `hailo optimize` fails with a path-related error when specifying the calibration set:

```
Error: calib_set_path 'calib_npy' does not exist or is not a valid directory
```

### Causes and Solutions

**Cause 1: Wrong working directory**
The `--calib-set-path` is resolved relative to where you run the command.
```bash
# Run from the compile/ directory
cd compile/
hailo optimize best_clean.har --calib-set-path ../calib_npy ...

# OR use absolute path
hailo optimize best_clean.har --calib-set-path /home/pi/project/calib_npy ...
```

**Cause 2: Empty directory**
The calibration directory exists but contains no `.npy` files.
```bash
ls calib_npy/
# Should show: image_001.npy  image_002.npy  ...
```

**Cause 3: Wrong file format**
Files in the directory are not `.npy` files. Ensure you ran the conversion script (see Issue 5 above).

**Cause 4: Permission issues**
```bash
chmod -R 755 calib_npy/
```

---

## Quick Reference

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| Parser includes Detect layer | Missing `--end-node-names` | Specify 6 conv end-nodes |
| `meta_arch=yolov8n` not found | Wrong arch name | Use `meta_arch=yolov8` |
| NMS errors on NPU | Hailo-8L NMS limitation | Use `engine=cpu` |
| HEF file not found | Unexpected output name | Check compiler logs, use `--output-dir` |
| Calibration format error | JPG instead of NPY | Convert with cv2 + numpy |
| Calibration path error | Wrong CWD or empty dir | Use absolute path, verify contents |
