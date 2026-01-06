import sys

from ultralytics import YOLO
import cv2
import os

import time
from pathlib import Path


def run_predict(video_path, model_path):
    model_ov = os.path.join(Path(model_path).parent, 'yolo11n_openvino_model')

    if not os.path.exists(model_ov):
        print("Exporting model to OpenVINO...")
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

    results = model.track(
        source=video_path,
        stream=True,
        conf=0.5,
        persist=True,
        imgsz=320,
        tracker='bytetrack.yaml',
        device='cpu',
        verbose=False,
        augment=False)

    print('Init_Done')
    sys.stdout.flush()

    count_ids = set()
    total_ids = 0

    prev_time = 0
    for r in results:
        curr_time = time.time()
        tact = curr_time - prev_time
        fps = 1 / tact
        prev_time = curr_time

        frame = r.orig_img

        overlay = frame.copy()
        cv2.rectangle(overlay, (roi_x1, roi_y1),(roi_x2, roi_y2), (0, 255, 0), -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, id, conf in zip(boxes, ids, confs):
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                    if id not in count_ids:
                        count_ids.add(id)
                        total_ids += 1

                x1, y1, x2,y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                cv2.putText(frame,f"ID:{str(id)}_Conf:{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)

        inference_time = r.speed['inference']
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Tact: {tact * 1000:.1f} ms", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Inference: {inference_time:.1f} ms", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Total: {total_ids} ", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)


        frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("YOLO11 OpenVINO CPU Tracking", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = r'..\DATASET\VIDEO\2103099-uhd_3840_2160_30fps.mp4'
    model_path = r"..\PRETRAINED/Yolo11_Object_Detection/yolo11n.pt"

    # video_path = sys.argv[1]
    # model_path = sys.argv[2]

    run_predict(video_path, model_path)
