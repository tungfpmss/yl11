import os
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.5
WEIGHT_DIR = r"\weights\best.pt"
model = YOLO(WEIGHT_DIR)
results = model.predict(
    source=test_images_path,
    save=True,  # Bật lưu kết quả
    project=absolute_path,  # Gán thư mục lưu dự án (thư mục gốc)
    name="test_prediction",  # Tên lần chạy dự đoán (tạo thư mục con)
    conf=CONFIDENCE_THRESHOLD,  # Ngưỡng độ tin cậy
    exist_ok=True,  # Cho phép ghi đè/tiếp tục nếu thư mục đã tồn tại
)
