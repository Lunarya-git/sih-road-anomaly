"""
Fine-tune YOLOv8n-seg (COCO-pretrained) on the pothole segmentation dataset.

Usage:
    python src/train.py
"""

from ultralytics import YOLO

DATA_CONFIG = "configs/pothole-seg.yaml"
BASE_WEIGHTS = "yolov8n-seg.pt"   # auto-downloads on first run
EPOCHS = 3
IMG_SIZE = 640
PROJECT_DIR = "weights/runs"
RUN_NAME = "pothole_seg_v1"


def main():
    model = YOLO(BASE_WEIGHTS)

    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=20,       # stop early if val metrics stall for 20 epochs
        plots=True,
    )

    # Best weights end up at: weights/runs/pothole_seg_v1/weights/best.pt
    print("Training complete. Best weights saved under:")
    print(f"  {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()