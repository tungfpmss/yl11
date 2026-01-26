import sys

from ultralytics import YOLO
import cv2
import os

import time
from pathlib import Path


def run_predict(video_path, model_path):
    model_ov = os.path.join(Path(model_path).parent, Path(model_path).stem + '_openvino_model')

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

    results = model.track(
        source=video_path,
        stream=True,
        conf=0.5,
        persist=True,
        imgsz=320,
        tracker='bytetrack.yaml',
        device='cpu',
        verbose=False,
        augment=False
    )

    print('Init_Done')
    sys.stdout.flush()

    prev_time = 0
    for r in results:
        curr_time = time.time()
        tact = curr_time - prev_time
        fps = 1 / tact
        prev_time = curr_time

        frame = r.plot()
        frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        inference_time = r.speed['inference']
        cv2.putText(frame_resized, f"FPS: {int(fps)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_resized, f"Tact: {tact * 1000:.1f} ms", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame_resized, f"Inference: {inference_time:.1f} ms", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("YOLO11 OpenVINO CPU Tracking", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = r'..\DATASET\Video_Car_Traffic\2103099-uhd_3840_2160_30fps.mp4'
    model_path = r"..\PRETRAINED/Yolo11_Object_Detection/yolo11n.pt"

    run_predict(video_path, model_path)
