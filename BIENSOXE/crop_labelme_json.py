import json, cv2, os
import numpy as np
from pathlib import Path


def crop_labelme(input_dir, output_dir):
    path_in, path_out = Path(input_dir), Path(output_dir)
    path_out.mkdir(parents=True, exist_ok=True)

    for json_file in path_in.glob("*.json"):
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)

        img = cv2.imread(str(path_in / os.path.basename(data['imagePath'])))
        if img is None: continue

        for i, shape in enumerate(data['shapes']):
            # Lấy tọa độ và crop nhanh với boundingRect
            x, y, w, h = cv2.boundingRect(np.array(shape['points'], dtype=np.int32))
            crop = img[max(0, y):y + h, max(0, x):x + w]

            # Lưu ảnh: label_tên-gốc_số-thứ-tự.jpg
            cv2.imwrite(str(path_out / f"{shape['label']}_{json_file.stem}_{i}.jpg"), crop)


def rename_labels(folder_path, new_name):
    path = Path(folder_path)

    # Tìm tất cả file .json trong thư mục
    json_files = list(path.glob('*.json'))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Thay đổi label
        if 'shapes' in data:
            for shape in data['shapes']:
                shape['label'] = new_name

        # Ghi lại file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Xong! Đã xử lý {len(json_files)} file.")


if __name__ == "__main__":
    crop_labelme(r"..\DATASET\Video_BienSoXe\Data_Origin",
                 r"..\DATASET\Video_BienSoXe\Data_BienSo")

    rename_labels(r"..\DATASET\Video_BienSoXe\Data_Known", 'bienso')
