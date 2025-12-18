"""
Train YOLOv8 classification model on ASL dataset.
"""

from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data="asl_dataset/asl_dataset",
        imgsz=96,
        epochs=100,
        batch=32,
        device="cpu",
        project="runs",
        name="asl_yolo_cls"
    )

    print("YOLOv8 training completed.")

if __name__ == "__main__":
    main()
