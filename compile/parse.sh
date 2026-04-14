#!/bin/bash
# Parse ONNX model to Hailo Archive (HAR) format
# Selects the 6 YOLOv8n detection head end-nodes for proper NMS integration

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
