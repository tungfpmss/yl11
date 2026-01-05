import os
from ultralytics import YOLO


DATATEST = r"D:\9.ATC_AI_Core\SVSC2026\DATASET\Animal3_yolo_dataset"


if __name__ == "__main__":
    test_dir = os.path.join(DATATEST, "test", "cat", "cat010.jpg")
    model = YOLO(r"..\runs\classify\yolo11_cls_dog_cat_fox\weights\best.pt")

    # for f in os.listdir(test_dir):
    #     results = model.predict(source=f, save=True, conf=0.5)

    results = model.predict(source=test_dir, conf=0.25)

    for result in results:
        # Lấy tên các class
        names = result.names

        # Lấy ID của class có xác suất cao nhất
        top1_idx = result.probs.top1
        top1_conf = result.probs.top1conf.item()

        print(f"Kết quả: {names[top1_idx]} với độ tự tin: {top1_conf:.2f}")

        # Hiển thị hoặc lưu ảnh đã vẽ kết quả
        result.show()
