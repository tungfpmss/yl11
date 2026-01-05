from ultralytics import YOLO
import cv2
import os

# 1. Cấu hình đường dẫn và Model
video_path = r'F:\9.Source_Code\DATASET\Video\car-uhd_3840_2160_60fps.mp4'
model_pt = r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n.pt"
# TensorRT model sẽ có đuôi .engine
model_trt = r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n.engine"

# 1. Kiểm tra và Export sang TensorRT (chạy 1 lần duy nhất)
if not os.path.exists(model_trt):
    print("Đang tối ưu mô hình cho GPU (TensorRT)... Vui lòng đợi.")
    model = YOLO(model_pt)
    # half=True sử dụng FP16 giúp tăng tốc gấp đôi trên GPU
    model.export(format='engine', device=0, half=True)

# Lấy thông tin độ phân giải gốc của video
cap_info = cv2.VideoCapture(video_path)
orig_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_info.release()

# Tính toán kích thước hiển thị tương thích
# Giả sử bạn muốn chiều cao cửa sổ là 800px, chiều rộng sẽ tự tính theo tỷ lệ
DISPLAY_HEIGHT = 800
aspect_ratio = orig_w / orig_h
DISPLAY_WIDTH = int(DISPLAY_HEIGHT * aspect_ratio)

# Load model
model = YOLO(model_trt, task='detect')

vehicle_classes = [2, 3, 5, 7]
counted_ids = set()
total_vehicles = 0

# 2. Chạy Tracking
results = model.track(
    source=video_path,
    stream=True,
    conf=0.4,
    device=0,
    persist=True,
    classes=vehicle_classes
)

for r in results:
    frame = r.plot()

    if r.boxes.id is not None:
        ids = r.boxes.id.int().cpu().tolist()
        for obj_id in ids:
            if obj_id not in counted_ids:
                counted_ids.add(obj_id)
                total_vehicles += 1

    # Hiển thị số lượng (Tùy chỉnh font scale dựa trên độ phân giải gốc)
    font_scale = orig_h / 1000  # Scale font theo độ cao video
    cv2.putText(
        frame, f"Total Vehicles: {total_vehicles}", (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 3
    )

    # 3. Resize và hiển thị theo tỷ lệ đã tính
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("YOLO12 Vehicle Counting - Auto Resized", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print(f"--- Kết quả cuối cùng: Đã đếm được {total_vehicles} phương tiện ---")