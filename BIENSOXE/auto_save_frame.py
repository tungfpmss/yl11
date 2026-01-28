import os
from pathlib import Path

import cv2

video_path = r'D:\PythonProject\Dataset_2\New video.mp4'
output_path = os.path.join(Path(video_path).parent, 'frames')
os.makedirs(output_path, exist_ok=True)

count = 0
count_save = 0
frame_save = 10

cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_save == 0:
        count_save += 1
        cv2.imwrite(os.path.join(output_path, f'frame_{count_save}.png'), frame)

    count += 1

cap.release()
