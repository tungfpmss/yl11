import os
import random
import shutil
from pathlib import Path


def split_data_yolo(input_folder):
    # --- CẤU HÌNH ---
    p = Path(input_folder)
    # Thư mục đầu ra nơi các tập train/val/test sẽ được tạo
    output_folder = os.path.join(p.parent, p.name + "_yolo")

    # Tỷ lệ phân chia (tổng phải bằng 1.0)
    train_ratio = 0.8  # 80% cho tập huấn luyện
    val_ratio = 0.15  # 15% cho tập xác thực
    test_ratio = 0.05  # 5% cho tập kiểm tra

    # --- CHẠY SCRIPT ---
    # Lấy danh sách tất cả các file ảnh
    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    # Lặp qua các file ảnh và đảm bảo file label tương ứng tồn tại
    valid_files = []
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        if os.path.exists(os.path.join(input_folder, label_file)):
            valid_files.append(
                os.path.splitext(img_file)[0]
            )  # Lưu tên gốc không có đuôi

    # Trộn ngẫu nhiên danh sách
    random.shuffle(valid_files)

    # Phân chia danh sách theo tỷ lệ
    total_count = len(valid_files)
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count

    train_list = valid_files[:train_count]
    val_list = valid_files[train_count : train_count + val_count]
    test_list = valid_files[train_count + val_count :]

    # Tạo cấu trúc thư mục đầu ra
    subsets = ["train", "val", "test"]
    for subset in subsets:
        os.makedirs(os.path.join(output_folder, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, subset, "labels"), exist_ok=True)

    # Bắt đầu di chuyển
    move_files(input_folder, output_folder, train_list, "train")
    move_files(input_folder, output_folder, val_list, "val")
    move_files(input_folder, output_folder, test_list, "test")

    print("\n--- HOÀN TẤT ---")
    print(f"Tổng số file được xử lý: {total_count}")
    print(f"Số file train: {len(train_list)}")
    print(f"Số file val: {len(val_list)}")
    print(f"Số file test: {len(test_list)}")


# Hàm copy file
def move_files(input_folder, output_folder, file_list, subset_name):
    print(f"Đang sao chép {len(file_list)} file tới tập {subset_name}...")
    for filename in file_list:
        # Copy ảnh
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            src_img_path = os.path.join(input_folder, filename + ext)
            if os.path.exists(src_img_path):
                dest_img_path = os.path.join(
                    output_folder, subset_name, "images", filename + ext
                )
                shutil.copy(src_img_path, dest_img_path)
                break

        # Copy file label
        src_label_path = os.path.join(input_folder, filename + ".txt")
        dest_label_path = os.path.join(
            output_folder, subset_name, "labels", filename + ".txt"
        )
        shutil.copy(src_label_path, dest_label_path)


if __name__ == "__main__":
    # Thư mục chứa tất cả ảnh và file label (ví dụ: data/all_images)
    input_folder = r"D:\9.ATC_AI_Core\SVSC2026\DATASET\LHC"
    split_data_yolo(input_folder)
