import os
import glob

def fix_label_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(os.path.join(src_dir, "*.txt"))
    fixed, clean, skipped = 0, 0, 0

    for src_path in files:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, filename)

        with open(src_path, 'r') as f:
            content = f.read()

        if '\\n' in content:
            content = content.replace('\\n', '\n')
            fixed += 1
        else:
            clean += 1

        lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
        if all(len(l.split()) == 5 for l in lines):
            with open(dst_path, 'w') as f:
                f.write(content)
        else:
            skipped += 1

    print(f"{dst_dir}: {len(files)} files → {fixed} fixed, {clean} already clean, {skipped} skipped")

if __name__ == "__main__":
    SRC = "/kaggle/input/datasets/champanerimitul/road-traffic-detection-dataset"
    DST = "/kaggle/working/dataset_fixed"
    for split in ['train', 'val', 'test']:
        fix_label_files(f"{SRC}/{split}/labels", f"{DST}/{split}/labels")
