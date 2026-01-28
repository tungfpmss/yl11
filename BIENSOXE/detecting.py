import os
import sys
import time
from pathlib import Path

import cv2
from torch.xpu import device
from ultralytics import YOLO

output_path = r'Output'
os.makedirs(output_path, exist_ok=True)

path_Known = os.path.join(output_path, 'Known')
os.makedirs(path_Known, exist_ok=True)

path_Unknown = os.path.join(output_path, 'Unknown')
os.makedirs(path_Unknown, exist_ok=True)


def detecting(video_path, model_path):
    model_ov = os.path.join(Path(model_path).parent, 'best_openvino_model')
    if not os.path.exists(model_ov):
        model = YOLO(model_path)
        model.export(format='OpenVINO', half=True, device='cpu')

    model = YOLO(model_ov, task='detect')

    results = model.track(
        source=video_path,
        stream=True,
        conf=0.6,
        device='cpu',
        persist=False,
        tracker='bytetrack.yaml'
    )

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    class_name = model.names

    DISPLAY_HEIGHT = 600
    scale_display = DISPLAY_HEIGHT / orig_h
    DISPLAY_WIDTH = int(scale_display * orig_w)

    prev_time = time.perf_counter()
    count_unk = 0
    for r in results:
        curr_time = time.perf_counter()
        tact = curr_time - prev_time
        fps = 1 / tact
        prev_time = curr_time

        frame = r.orig_img

        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.int().cpu().numpy()

            for box, id, conf, cls in zip(boxes, ids, confs, cls):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Bien So: {str(class_name[cls])}_Conf:{conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                x_1, y_1, x_2, y_2 = int(x1), int(y1), int(x2), int(y2)
                bien_so = frame[y_1: y_2, x_1:x_2, :]
                cv2.imshow('Bien so', bien_so)

                if (cls < 30):
                    cv2.imwrite(os.path.join(path_Known, f'{str(class_name[cls])}.png'), bien_so)
        else:
            count_unk+=1
            cv2.imwrite(os.path.join(path_Unknown, f'Unknown_{count_unk}.png'), frame)

        frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow('Dang xu ly video', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = r'..\DATASET\Video_BienSoXe\New video.mp4'
    model_path = r'runs\detect\train_riengbienso\weights\best.pt'

    detecting(video_path, model_path)
