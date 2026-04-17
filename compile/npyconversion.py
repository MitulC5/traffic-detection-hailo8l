import cv2
import numpy as np
import os

# ─── CONFIG ────────────────────────────────────────────────────────
CALIB_DIR   = "/home/mitul/hailo_workspace/calib_images"
OUTPUT_NPY  = "/home/mitul/hailo_workspace/calib_data.npy"
INPUT_SIZE  = (640, 640)
MAX_IMAGES  = 64       #keeping 64 helps since Hailo DFC has a hard time detecting GPUs on WSL so it will run on CPU
# ───────────────────────────────────────────────────────────────────

image_files = sorted([
    os.path.join(CALIB_DIR, f)
    for f in os.listdir(CALIB_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])[:MAX_IMAGES]

if len(image_files) == 0:
    print("ERROR: No images found in calib_images folder!")
    exit(1)

calib_data = []

for img_path in image_files:
    img = cv2.imread(img_path)               # BGR uint8
    if img is None:
        print(f"Skipping: {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # Fix 1: BGR → RGB
    img = cv2.resize(img, INPUT_SIZE,
                     interpolation=cv2.INTER_LINEAR)      # Fix 2: resize
    img = img.astype(np.uint8)                            # Fix 3: keep uint8
    # DO NOT divide by 255 — Hailo needs raw 0-255 values

    calib_data.append(img)

# Fix 4: stack into 4D batch array [N, H, W, 3]
calib_array = np.stack(calib_data, axis=0)

np.save(OUTPUT_NPY, calib_array)

print("=== Fixed NPY Metadata ===")
print(f"Shape       : {calib_array.shape}")    # Should be (N, 640, 640, 3)
print(f"Dtype       : {calib_array.dtype}")    # Should be uint8
print(f"Min value   : {calib_array.min()}")    # Should be 0
print(f"Max value   : {calib_array.max()}")    # Should be 255
print(f"Total images: {calib_array.shape[0]}") # Should be your image count
print(f"Saved to    : {OUTPUT_NPY}")