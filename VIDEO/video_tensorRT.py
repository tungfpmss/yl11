from ultralytics import YOLO
import cv2
import os

# Đường dẫn file
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

# 2. Load mô hình đã tối ưu với GPU
model = YOLO(model_trt, task='detect')

WIDTH, HEIGHT = 432, 768

# 3. Chạy Tracking với các tham số tăng tốc
# device=0: Sử dụng GPU
# vid_stride=2: Bỏ qua 1 frame (nhảy cách) để tăng FPS nếu video quá nặng
results = model.track(
    source=video_path,
    stream=True,
    conf=0.4,
    device=0,
    imgsz=640  # Resize đầu vào về 640 để xử lý cực nhanh
)

for r in results:
    # r.plot() trả về mảng BGR chuẩn OpenCV
    frame = r.plot()

    # Resize nhanh để hiển thị
    frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))

    cv2.imshow("YOLO12 GPU Optimized", frame_resized)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()