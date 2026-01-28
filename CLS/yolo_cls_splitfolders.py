import os
import splitfolders

source_data = r'D:\9.ATC_AI_Core\YOLO_Trainers\DATASET\Animal3'
output_data = os.path.join(
    os.path.dirname(source_data), "Animal3_yolo_dataset"
)

# Chia theo tỉ lệ: Train (80%), Validation (10%), Test (10%)
splitfolders.ratio(source_data, output=output_data,
                   seed=42, ratio=(.8, .1, .1),
                   group_prefix=None, move=False)  # move=True nếu muốn di chuyển thay vì copy
