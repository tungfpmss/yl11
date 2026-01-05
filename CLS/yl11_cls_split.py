import os
import shutil
import random

def split_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Thiết lập danh sách các class (cat, dog, fox)
    classes = ['cat', 'dog', 'fox']
    
    for cls in classes:
        # Đường dẫn đến thư mục gốc của từng loài
        src_path = os.path.join(source_dir, cls)
        if not os.path.exists(src_path):
            print(f"Cảnh báo: Không tìm thấy thư mục {src_path}")
            continue

        # Lấy danh sách tất cả file ảnh
        images = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        random.shuffle(images)

        # Tính toán số lượng file cho từng tập
        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        # Chia danh sách file
        train_files = images[:train_count]
        val_files = images[train_count:train_count + val_count]
        test_files = images[train_count + val_count:]

        # Hàm copy file vào thư mục đích
        def copy_files(files, split_name):
            target_path = os.path.join(output_dir, split_name, cls)
            os.makedirs(target_path, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(src_path, f), os.path.join(target_path, f))

        # Thực hiện copy
        copy_files(train_files, 'train')
        copy_files(val_files, 'val')
        copy_files(test_files, 'test')

        print(f"Hoàn thành class {cls}: Train({len(train_files)}), Val({len(val_files)}), Test({len(test_files)})")

# Cấu hình đường dẫn của bạn ở đây
source_data = r'D:\9.Source_Code\SVSC\Animal3'  # Thư mục chứa 3 folder cat, dog, fox
output_data = os.path.join(os.path.dirname(source_data), 'yolo_dataset')          # Thư mục đích sẽ dùng để train

split_dataset(source_data, output_data)