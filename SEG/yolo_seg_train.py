import torch
from ultralytics import YOLO


def train_yolo():
    model = YOLO(r"..\PRETRAINED\Yolo11_Classification\yolo11n-seg.pt")


    results = model.train(
        data=r"seg.yaml",
        epochs=50,
        imgsz=224,
        batch=256,
        device="cpu",
        workers=8,
        name="yolo11_cls_dog_cat_fox",
        patience=50,
    )

    print("Training completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    train_yolo()
