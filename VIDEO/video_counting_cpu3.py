from ultralytics import YOLO
import cv2
import os

# 1. Cấu hình đường dẫn
video_path = r'F:\9.Source_Code\DATASET\Video\car-uhd_3840_2160_60fps.mp4'
model_pt = r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n.pt"
# Thư mục chứa model OpenVINO sau khi tối ưu
model_ov = r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n_openvino_model"

# 2. Kiểm tra và Export sang INT8 (Nhanh nhất cho CPU)
if not os.path.exists(model_ov):
    print("Đang thực hiện lượng tử hóa INT8 (Quantization) cho CPU... Vui lòng đợi.")
    model = YOLO(model_pt)
    # format='openvino', int8=True sẽ nén mô hình từ FP32 xuống INT8
    # Điều này giúp giảm dung lượng và tăng tốc độ xử lý gấp nhiều lần trên CPU
    model.export(format='openvino', int8=True)

# Load Model đã tối ưu
model = YOLO(model_ov, task='detect')

# Lấy thông tin video
cap = cv2.VideoCapture(video_path)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Resize hiển thị (Giảm kích thước hiển thị để giảm tải cho CPU)
DISPLAY_HEIGHT = 600
scale_factor = DISPLAY_HEIGHT / orig_h
DISPLAY_WIDTH = int(orig_w * scale_factor)

# Vùng ROI
roi_ratio = [0.1, 0.6, 0.9, 0.75]
roi_x1, roi_y1 = int(orig_w * roi_ratio[0]), int(orig_h * roi_ratio[1])
roi_x2, roi_y2 = int(orig_w * roi_ratio[2]), int(orig_h * roi_ratio[3])

counted_ids = set()
total_vehicles = 0
vehicle_classes = [2, 3, 5, 7]

# 3. Tracking với cấu hình ép tốc độ
results = model.track(
    source=video_path,
    stream=True,
    conf=0.25,  # Giảm nhẹ conf để bù đắp sai số sau khi nén INT8
    device='cpu',
    persist=True,
    classes=vehicle_classes,
    imgsz=320,  # Rất quan trọng để đạt real-time trên CPU
    vid_stride=3,  # Nhảy 3 frame (Xử lý 20fps từ video 60fps) giúp chạy mượt mà không bị lag
    augment=False  # Tắt augment để tối đa tốc độ
)

for r in results:
    frame = r.orig_img

    # Vẽ ROI trong suốt
    overlay = frame.copy()
    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()

        for box, obj_id, conf in zip(boxes, ids, confs):
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

            # Kiểm tra đếm xe
            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_vehicles += 1

            # Vẽ trực tiếp lên frame
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            # Hiển thị ID và Conf (rút gọn để tránh rối mắt khi chạy nhanh)
            cv2.putText(frame, f"ID:{obj_id} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Hiển thị tổng số
    cv2.putText(frame, f"TOTAL: {total_vehicles}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 6)

    # 4. Resize và hiển thị
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("YOLO12 Real-time CPU (INT8)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()