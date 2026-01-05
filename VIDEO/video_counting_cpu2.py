from ultralytics import YOLO
import cv2
import os

# 1. Cấu hình đường dẫn
video_path = r'F:\9.Source_Code\DATASET\Video\car-uhd_3840_2160_60fps.mp4'
model_pt = r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n.pt"
model_ov = r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n_openvino_model"

# Kiểm tra và Export OpenVINO
if not os.path.exists(model_ov):
    print("Đang tối ưu mô hình cho CPU (OpenVINO)...")
    model = YOLO(model_pt)
    model.export(format='openvino')

model = YOLO(model_ov, task='detect')

# Lấy thông tin video gốc
cap = cv2.VideoCapture(video_path)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Resize cửa sổ hiển thị
DISPLAY_HEIGHT = 600
scale_factor = DISPLAY_HEIGHT / orig_h
DISPLAY_WIDTH = int(orig_w * scale_factor)

# Định nghĩa vùng ROI
roi_ratio = [0.1, 0.6, 0.9, 0.75]
roi_x1, roi_y1 = int(orig_w * roi_ratio[0]), int(orig_h * roi_ratio[1])
roi_x2, roi_y2 = int(orig_w * roi_ratio[2]), int(orig_h * roi_ratio[3])

counted_ids = set()
total_vehicles = 0
vehicle_classes = [2, 3, 5, 7]

# 2. Chạy Tracking
results = model.track(
    source=video_path,
    stream=True,
    conf=0.3,
    device='cpu',
    persist=True,
    classes=vehicle_classes,
    imgsz=320,
    vid_stride=2
)

for r in results:
    frame = r.orig_img

    # Vẽ vùng ROI mờ
    overlay = frame.copy()
    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.cpu().numpy()  # Tọa độ [x1, y1, x2, y2]
        ids = r.boxes.id.int().cpu().tolist()  # Object ID
        confs = r.boxes.conf.cpu().tolist()  # Confidence score

        for box, obj_id, conf in zip(boxes, ids, confs):
            # 1. Tính tâm đối tượng
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)

            # 2. Logic đếm xe
            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_vehicles += 1

            # 3. Vẽ thủ công để tối ưu hiển thị
            x1, y1, x2, y2 = map(int, box)

            # Vẽ Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ Tâm đối tượng (Centroid)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            # Vẽ ID và Confidence Score
            label = f"ID: {obj_id} | {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Hiển thị tổng số lượng
    cv2.putText(frame, f"TOTAL: {total_vehicles}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8)

    # Resize và hiển thị
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("YOLO12 CPU - Centroid & ID", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()