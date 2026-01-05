from ultralytics import YOLO
import cv2
import os

# 1. Cấu hình đường dẫn
video_path = r'F:\9.Source_Code\DATASET\Video\car-uhd_3840_2160_60fps.mp4'
model_pt = r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n.pt"
# Chuyển đổi sang format openvino
model_ov = r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n_openvino_model"

# 1. Kiểm tra và Export sang OpenVINO (Tối ưu cho CPU)
if not os.path.exists(model_ov):
    print("Đang tối ưu mô hình cho CPU (OpenVINO)... Vui lòng đợi.")
    model = YOLO(model_pt)
    # Export sang OpenVINO để chạy nhanh nhất trên CPU
    model.export(format='openvino')

# Load Model OpenVINO
model = YOLO(model_ov, task='detect')

# Lấy thông tin video gốc
cap = cv2.VideoCapture(video_path)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# TỰ ĐỘNG RESIZE CỬA SỔ HIỂN THỊ
DISPLAY_HEIGHT = 600 # Giảm xuống 600 để CPU đỡ tải phần hiển thị
scale_factor = DISPLAY_HEIGHT / orig_h
DISPLAY_WIDTH = int(orig_w * scale_factor)

# --- ĐỊNH NGHĨA VÙNG RECT (ROI) ---
roi_ratio = [0.1, 0.6, 0.9, 0.75]
roi_x1, roi_y1 = int(orig_w * roi_ratio[0]), int(orig_h * roi_ratio[1])
roi_x2, roi_y2 = int(orig_w * roi_ratio[2]), int(orig_h * roi_ratio[3])

# Biến lưu trữ
counted_ids = set()
total_vehicles = 0
vehicle_classes = [2, 3, 5, 7]

# 2. Chạy Tracking với cấu hình tối ưu cho CPU
results = model.track(
    source=video_path,
    stream=True,
    conf=0.3,       # Giảm nhẹ conf để bắt vật thể nhạy hơn trên CPU
    device='cpu',   # CHUYỂN SANG CPU
    persist=True,
    classes=vehicle_classes,
    imgsz=320,      # QUAN TRỌNG: Giảm imgsz xuống 320 giúp CPU chạy mượt hơn rất nhiều
    vid_stride=2    # QUAN TRỌNG: Nhảy cách 1 frame để tăng tốc độ xử lý thực tế
)

for r in results:
    frame = r.orig_img

    # Hiệu ứng vùng ROI
    overlay = frame.copy()
    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 5)

    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()

        for box, obj_id in zip(boxes, ids):
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_vehicles += 1

            # Vẽ Box đơn giản (bỏ r.plot để tiết kiệm CPU)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Hiển thị thông tin (giảm kích thước font để cân đối)
    cv2.putText(frame, f"TOTAL: {total_vehicles}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

    # 3. Resize hiển thị
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("YOLO12 CPU Optimized (OpenVINO)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()