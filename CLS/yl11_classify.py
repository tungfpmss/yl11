import torch
from ultralytics import YOLO

def train_yolo():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Đang sử dụng thiết bị: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO('yolo11n-cls.pt')

    results = model.train(
        data=r'D:\9.Source_Code\SVSC\yolo_dataset',  # Path to dataset directory
        epochs=50,      # Number of epochs
        imgsz=224,      # Image size (224 is common for classification)
        batch=256,       # Batch size
        device='cuda',  # Use GPU for training
        workers=8,              # Số luồng CPU xử lý ảnh đầu vào
        name='yolo11_cls_dog_cat_fox'  # Name for the training run
    )

    print("Training completed!")
    print(f"Results: {results}")

if __name__ == "__main__":
    train_yolo()