from torch.cuda import device
from ultralytics import YOLO
import cv2

video_path = r'F:\9.Source_Code\DATASET\Cat_Dog_Human\8944044-uhd_2160_3840_25fps.mp4'
model = YOLO(r"..\PRETRAINED/Yolo12_Object_Detection/yolo12n.pt")
model.export(format='openvino')

model = YOLO(r'..\PRETRAINED/Yolo12_Object_Detection/yolo12n_openvino_model')

# model.predict(source=video_path, save=True, conf=0.5)
# model.track(source=video_path, show=True, conf=0.5, imgsz=640)

WIDTH = 432
HEIGHT = 768
results = model.track(source=video_path, stream=True, conf=0.5, device='cpu')

for r in results:
    frame = r.plot()
    frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))

    cv2.imshow("YOLO Tracking Custom Size", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()