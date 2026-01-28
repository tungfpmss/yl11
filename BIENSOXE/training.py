import cv2
from ultralytics import YOLO


# labelme2yolo --json_dir D:\PythonProject\Dataset_2\Data_Train --val_size=0.2

def training():
    model = YOLO(r'..\PreTrained\Yolo11_Object_Detection\yolo11n.pt')
    data_path = r'..\DATASET\Video_BienSoXe\Data_Known\YOLODataset\dataset.yaml'
    model.train(
        data=data_path,
        epochs=1000,
        batch=256,
        imgsz=320,
        device="cpu",
        workers=8,
        hsv_v=0.6,
        scale=0.3,
    )


if __name__ == '__main__':
    training()
