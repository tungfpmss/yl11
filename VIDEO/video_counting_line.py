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

# Lấy thông tin video gốc để tính toán tỷ lệ
cap = cv2.VideoCapture(video_path)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()  # Giải phóng ngay sau khi lấy thông tin

# TỰ ĐỘNG RESIZE CỬA SỔ HIỂN THỊ (Ví dụ: đặt chiều cao cố định là 800, chiều rộng tự nhảy theo tỷ lệ)
DISPLAY_HEIGHT = 800
scale_factor = DISPLAY_HEIGHT / orig_h
DISPLAY_WIDTH = int(orig_w * scale_factor)

# Vị trí vạch kẻ (tính theo % chiều cao để tương thích mọi độ phân giải)
LINE_RATIO = 0.7  # Vạch nằm ở 70% chiều cao khung hình
line_y_orig = int(orig_h * LINE_RATIO)
OFFSET = int(orig_h * 0.02)  # Sai số 2% chiều cao để đếm chính xác hơn

# Biến lưu trữ
counted_ids = set()
total_vehicles = 0
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

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
    frame = r.orig_img  # Lấy ảnh gốc để xử lý tọa độ chuẩn

    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()

        for box, obj_id in zip(boxes, ids):
            # Tính tâm vật thể
            y_center = int((box[1] + box[3]) / 2)

            # Logic đếm khi cắt vạch
            if (line_y_orig - OFFSET) < y_center < (line_y_orig + OFFSET):
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_vehicles += 1

            # Vẽ Box và ID lên frame gốc
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"ID:{obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    # Vẽ vạch kẻ và thông tin đếm
    cv2.line(frame, (0, line_y_orig), (orig_w, line_y_orig), (0, 0, 255), 8)
    cv2.putText(frame, f"COUNT: {total_vehicles}", (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 10)

    # 3. RESIZE TƯƠNG THÍCH ĐỂ HIỂN THỊ
    # frame_resized sẽ giữ đúng tỷ lệ của video gốc (4K, 2K, Full HD đều đẹp)
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    cv2.imshow("YOLO12 Auto-Resized Window", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()