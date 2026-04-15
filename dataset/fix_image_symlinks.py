import os

SRC_BASE = "/kaggle/input/datasets/champanerimitul/road-traffic-detection-dataset"
DST_BASE = "/kaggle/working/dataset_fixed"

for split in ['train', 'val', 'test']:
    src_img_dir = f"{SRC_BASE}/{split}/images"
    dst_img_dir = f"{DST_BASE}/{split}/images"

    if os.path.islink(dst_img_dir):
        os.unlink(dst_img_dir)

    os.makedirs(dst_img_dir, exist_ok=True)

    count = 0
    for img_file in os.listdir(src_img_dir):
        src = os.path.join(src_img_dir, img_file)
        dst = os.path.join(dst_img_dir, img_file)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            count += 1
    print(f"{split}: {count} image symlinks created ✅")
