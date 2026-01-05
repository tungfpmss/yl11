from ultralytics import YOLO
import cv2
import os

# 1. Cấu hình đường dẫn
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

# Load Model
model = YOLO(model_trt, task='detect')

# Lấy thông tin video gốc
cap = cv2.VideoCapture(video_path)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# TỰ ĐỘNG RESIZE CỬA SỔ HIỂN THỊ
DISPLAY_HEIGHT = 800
scale_factor = DISPLAY_HEIGHT / orig_h
DISPLAY_WIDTH = int(orig_w * scale_factor)

# --- ĐỊNH NGHĨA VÙNG RECT (ROI) ---
# Tọa độ tính theo tỷ lệ (0.0 đến 1.0) để tương thích mọi độ phân giải
# Rect: [x_start, y_start, x_end, y_end]
roi_ratio = [0.1, 0.6, 0.9, 0.75]  # Vùng đếm nằm từ 60% đến 75% chiều cao video

roi_x1 = int(orig_w * roi_ratio[0])
roi_y1 = int(orig_h * roi_ratio[1])
roi_x2 = int(orig_w * roi_ratio[2])
roi_y2 = int(orig_h * roi_ratio[3])

# Biến lưu trữ
counted_ids = set()
total_vehicles = 0
vehicle_classes = [2, 3, 5, 7]

# 2. Chạy Tracking
results = model.track(
    source=video_path,
    stream=True,
    conf=0.4,
    device=0,
    persist=True,
    classes=vehicle_classes,
    imgsz=640
)

for r in results:
    frame = r.orig_img

    # Vẽ vùng RECT đếm xe (Màu xanh lam nhạt)
    overlay = frame.copy()
    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), -1)
    # Tạo hiệu ứng trong suốt cho vùng đếm
    alpha = 0.3
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 5)

    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()

        for box, obj_id in zip(boxes, ids):
            # Tính tâm vật thể
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

            # KIỂM TRA TÂM VẬT THỂ NẰM TRONG RECT
            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_vehicles += 1

            # Vẽ tâm (dot) và box của xe
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Hiển thị thông tin đếm
    cv2.putText(frame, f"TOTAL COUNT: {total_vehicles}", (70, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 10)

    # 3. Resize hiển thị
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("YOLO12 ROI Counting", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()