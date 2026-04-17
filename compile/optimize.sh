#!/bin/bash
# Optimize the parsed HAR model with calibration data and NMS configuration
# Requires: calib_npy/ directory with calibration .npy files

hailo optimize best_clean.har \
  --hw-arch hailo8l \
  --calib-set-path calib_npy \
  --model-script nms.alls \
  --output-har-path best_optimized.har
