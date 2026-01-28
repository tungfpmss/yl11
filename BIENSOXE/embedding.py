import numpy as np
from scipy.stats import cosine
from ultralytics import YOLO
from pathlib import Path

# Load model pre-trained (sử dụng bản -cls để lấy đặc trưng tốt hơn)
model = YOLO(r'..\PRETRAINED\Yolo11_Classification\yolo11n-cls.pt')

def get_embedding(img_path):
    # Dự đoán và lấy ra vector đặc trưng ở lớp cuối cùng (lớp trước khi phân loại)
    results = model.predict(img_path, embed=[10, 11]) # Lấy từ layer sâu
    vector = results[0].probs.data.cpu().numpy() # Đây là embedding
    return vector

def known_embedding():
    known_embeddings = []
    known_dir = Path('path/to/known_plates')

    # Tạo kho dữ liệu đặc trưng
    for img_file in known_dir.glob('*.jpg'):
        emb = get_embedding(str(img_file))
        known_embeddings.append(emb)

    # Tính vector trung bình (Centroid) đại diện cho tập "Đã biết"
    centroid = np.mean(known_embeddings, axis=0)
    print("Đã xây dựng xong profile cho biển số đã biết!")


def check_anomaly(test_img_path, threshold=0.1):
    """
    threshold: Ngưỡng bất thường.
    Số càng nhỏ càng khắt khe (chỉ giống hệt mới cho qua).
    """
    test_emb = get_embedding(test_img_path)

    # Tính khoảng cách Cosine giữa biển số mới và vector trung bình
    distance = cosine(centroid, test_emb)

    if distance < threshold:
        print(f"✅ BIỂN SỐ ĐÃ BIẾT (Distance: {distance:.4f})")
    else:
        print(f"❌ CẢNH BÁO: BIỂN SỐ LẠ (Distance: {distance:.4f})")


# Chạy thử nghiệm
check_anomaly('new_plate_coming.jpg', threshold=0.15)