"""
YOLOv8 inference pipeline for video file or live camera.
Annotates predictions and saves output video.
"""

import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/asl_yolo_cls/weights/best.pt"
INPUT_SOURCE = 0  # 0 for camera, or path to video file
OUTPUT_VIDEO = "outputs/yolo/inference_output.mp4"
IMG_SIZE = 96

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_SOURCE)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = model.predict(frame, imgsz=IMG_SIZE, device="cpu", verbose=False)[0]
        probs = res.probs
        pred = res.names[int(probs.top1)]
        conf = float(probs.top1conf)

        cv2.putText(frame, f"{pred.upper()} ({conf:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

        writer.write(frame)
        cv2.imshow("ASL Inference", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
