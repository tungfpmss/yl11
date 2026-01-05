from ultralytics import YOLO

DATASET = r"D:\9.Source_Code\SVSC\yolo_dataset"


def train_yolo():
    model = YOLO(r"D:\9.Source_Code\Yolo11\yolo11n-cls.pt")

    results = model.train(
        data=DATASET,
        epochs=20,
        imgsz=224,
        batch=256,
        device="cuda",
        workers=8,
        project=r"..\runs\classify",
        name="yolo11_cls_dog_cat_fox",
        plots=True,
        augment=True,
    )

    print("Training completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    train_yolo()
