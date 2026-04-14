#!/bin/bash
# Compile the optimized HAR model into a HEF binary for Hailo-8L deployment
# Output: .hef file in the current directory

hailo compiler best_optimized.har \
  --hw-arch hailo8l \
  --output-dir .
