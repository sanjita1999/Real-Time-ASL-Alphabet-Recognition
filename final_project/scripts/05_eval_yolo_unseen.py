"""
Evaluate trained YOLOv8 classification model on unseen dataset.
"""

from ultralytics import YOLO

MODEL_PATH = "runs/asl_yolo_cls/weights/best.pt"
DATA_DIR = "data"

def main():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATA_DIR,
        imgsz=96,
        batch=64,
        device="cpu"
    )

    print("YOLOv8 Evaluation on Unseen Data")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")

if __name__ == "__main__":
    main()
