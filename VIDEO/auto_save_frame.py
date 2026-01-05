from pathlib import Path

import cv2
import os

video_path = r'D:\9.ATC_AI_Core\YOLO_Trainers\DATASET\Video\15042568_2160_3840_30fps.mp4'
output_dir = os.path.join(Path(video_path).parent, 'dataset')
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 10
count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_rate == 0:
        file_path = f"{output_dir}/img_{saved_count}.jpg"
        cv2.imwrite(file_path, frame)
        saved_count += 1

    count += 1

cap.release()
print(f"Hoàn thành! Đã lưu {saved_count} ảnh.")