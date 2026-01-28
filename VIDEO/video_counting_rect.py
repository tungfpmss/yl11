import shutil
import sys

from ultralytics import YOLO
import cv2
import os

import time
from pathlib import Path

COLOR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0)
}


def run_predict(video_path, model_path):
    model_ov = os.path.join(Path(model_path).parent, Path(model_path).stem + '_openvino_model')

    # if os.path.exists(model_ov):
    #     shutil.rmtree(model_ov)

    if not os.path.exists(model_ov):
        print(f"Exporting model to OpenVINO...{model_ov}")
        model = YOLO(model_path)
        model.export(format='openvino', half=True)

    model = YOLO(model_ov, task='detect')

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    DISPLAY_HEIGHT = 600
    scale_factor = DISPLAY_HEIGHT / orig_h
    DISPLAY_WIDTH = int(orig_w * scale_factor)

    roi_ratio = [0.1, 0.6, 0.9, 0.75]
    roi_x1, roi_y1 = int(orig_w * roi_ratio[0]), int(orig_h * roi_ratio[1])
    roi_x2, roi_y2 = int(orig_w * roi_ratio[2]), int(orig_h * roi_ratio[3])

    # vehicle_classes = [2, 3, 5, 7]

    results = model.track(
        source=video_path,  # Đường dẫn đến file video đầu vào hoặc link stream
        stream=True,  # Sử dụng generator (luồng). Giúp tiết kiệm RAM bằng cách chỉ nạp frame đang xử lý
        conf=0.5,  # Ngưỡng tin cậy (Confidence). Chỉ giữ lại các vật thể có độ chắc chắn trên 50%
        persist=True,  # Giữ lại ID của đối tượng qua từng frame. Cần thiết để thực hiện việc "theo dõi" (tracking)
        # classes=vehicle_classes,  # Chỉ lọc và xử lý các lớp đối tượng đã định nghĩa ở trên
        imgsz=320,  # Kích thước ảnh đầu vào của model. 320 là khá nhỏ, giúp tăng tốc độ xử lý (trade-off độ chính xác)
        tracker='bytetrack.yaml',  # Sử dụng thuật toán ByteTrack để gán ID và theo dõi đối tượng
        device='cpu',  # Chạy tính toán trên CPU (nếu có card đồ họa bạn nên đổi thành '0' để dùng GPU)
        verbose=False,  # Tắt thông báo chi tiết của model ở cửa sổ terminal để màn hình sạch hơn
        # vid_stride=3,  # Nhảy cách frame (xử lý frame 1, bỏ frame 2-3, xử lý frame 4...). Giảm tải cho CPU
        augment=False  # Tắt kỹ thuật tăng cường dữ liệu khi dự đoán để ưu tiên tốc độ thực thi nhanh nhất
    )

    count_ids = set()
    total_ids = 0

    prev_time = 0
    for r in results:
        curr_time = time.time()
        tact = curr_time - prev_time
        fps = 1 / tact
        prev_time = curr_time

        frame = r.orig_img
        # roi
        overlay = frame.copy()
        cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (0, 0), (500, 220), (255, 255, 255), -1)  # background

        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            for box, id, conf, cls in zip(boxes, ids, confs, clss):
                class_name = model.names[cls]
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                    if id not in count_ids:
                        count_ids.add(id)
                        total_ids += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                cv2.putText(frame, f"ID:{str(id)}_{class_name}_Conf:{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)

        inference_time = r.speed['inference']
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR['blue'], 5)
        cv2.putText(frame, f"Tact: {tact * 1000:.1f} ms", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR['blue'], 5)
        cv2.putText(frame, f"Inference: {inference_time:.1f} ms", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR['blue'], 5)
        cv2.putText(frame, f"Total: {total_ids} ", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR['blue'], 5)

        frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("YOLO11 OpenVINO CPU Counting", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = r'../DATASET/Video_Car_Traffic\2103099-uhd_3840_2160_30fps.mp4'
    # video_path = r'../DATASET/Video_Car_Traffic\car-uhd_3840_2160_60fps.mp4'
    model_path = r"..\PRETRAINED/Yolo11_Object_Detection/yolo11n.pt"
    model_path = r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n.pt"

    run_predict(video_path, model_path)
