"""
Create train/val split for YOLOv8 classification.

Input structure:
asl_dataset/asl_dataset/
    a/
    b/
    c/
    ...

Output structure:
asl_dataset/asl_dataset/
    train/a/
    val/a/
    ...
"""

import os
import random
import shutil

ROOT_DIR = "asl_dataset/asl_dataset"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
VAL_DIR = os.path.join(ROOT_DIR, "val")

VAL_RATIO = 0.1  # 10% validation
IMG_EXTS = (".jpg", ".jpeg", ".png")

random.seed(42)

def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    for cls in sorted(os.listdir(ROOT_DIR)):
        cls_path = os.path.join(ROOT_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        if cls in ["train", "val"]:
            continue

        train_cls = os.path.join(TRAIN_DIR, cls)
        val_cls = os.path.join(VAL_DIR, cls)
        os.makedirs(train_cls, exist_ok=True)
        os.makedirs(val_cls, exist_ok=True)

        images = [f for f in os.listdir(cls_path) if f.lower().endswith(IMG_EXTS)]
        random.shuffle(images)

        n_val = max(1, int(len(images) * VAL_RATIO))
        val_imgs = images[:n_val]
        train_imgs = images[n_val:]

        for img in train_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(train_cls, img))

        for img in val_imgs:
            shutil.copy(os.path.join(cls_path, img), os.path.join(val_cls, img))

        print(f"{cls}: {len(train_imgs)} train, {len(val_imgs)} val")

    print("YOLO train/val split created successfully.")

if __name__ == "__main__":
    main()
