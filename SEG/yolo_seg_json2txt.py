import json
import os
import cv2


# Hàm để chuẩn hóa tọa độ đa giác (dùng cho Segmentation)
def normalize_polygon_coordinates(points, img_width, img_height):
    """
    Chuẩn hóa các điểm đa giác cho định dạng YOLO Segmentation.
    [x1_norm y1_norm x2_norm y2_norm ...]
    """
    normalized_points = []
    for x, y in points:
        normalized_points.append(f"{x / img_width:.6f}")
        normalized_points.append(f"{y / img_height:.6f}")
    return " ".join(normalized_points)


# Hàm chuyển đổi tọa độ hình chữ nhật sang định dạng YOLO Detection
def convert_rectangle_to_yolo(points, img_width, img_height):
    """
    Chuyển đổi 2 điểm (min/max) của hình chữ nhật sang định dạng YOLO Detection:
    x_center_norm y_center_norm width_norm height_norm
    """
    # points[0] = [xmin, ymin], points[1] = [xmax, ymax]

    xmin, ymin = points[0]
    xmax, ymax = points[1]

    # Tính toán kích thước và tâm (pixel)
    width_px = abs(xmax - xmin)
    height_px = abs(ymax - ymin)
    x_center_px = (xmin + xmax) / 2
    y_center_px = (ymin + ymax) / 2

    # Chuẩn hóa
    x_center_norm = x_center_px / img_width
    y_center_norm = y_center_px / img_height
    width_norm = width_px / img_width
    height_norm = height_px / img_height

    # Trả về chuỗi định dạng YOLO Detection
    return (
        f"{x_center_norm:.6f} {y_center_norm:.6f} "
        f"{width_norm:.6f} {height_norm:.6f}"
    )


# ----------------------------------------------------------------------
## Hàm Chuyển đổi Chính
# ----------------------------------------------------------------------


def convert_json2txt(json_folder, output_folder, is_segmentation=True):
    """
    Chuyển đổi nhãn từ JSON sang TXT chuẩn YOLO.

    :param is_segmentation: Nếu True, nhãn Rectangle sẽ được chuyển thành
                            định dạng Segmentation (để mô hình Seg có thể xử lý).
                            Nếu False, nhãn Rectangle sẽ được chuyển thành định dạng Detection
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ĐỊNH NGHĨA ÁNH XẠ LỚP (BẮT BUỘC PHẢI THAY THẾ)
    class_map = {"mark": 0, "car": 1, "person": 2}  # ⚠️ THAY THẾ BẰNG TỪ ĐIỂN CỦA BẠN!

    # Lặp qua tất cả các file trong thư mục JSON
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            json_path = os.path.join(json_folder, filename)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Lỗi đọc file JSON {filename}: {e}")
                continue

            # Lấy kích thước ảnh để chuẩn hóa tọa độ
            img_width = data.get("imageWidth")
            img_height = data.get("imageHeight")

            if not img_width or not img_height:
                print(f"Bỏ qua {filename}: Thiếu imageWidth hoặc imageHeight.")
                continue

            # Tên file đầu ra (ví dụ: image1.json -> image1.txt)
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, "w", encoding="utf-8") as out_f:
                # Lặp qua từng shape (đối tượng) trong file JSON
                for shape in data.get("shapes", []):
                    label = shape.get("label")
                    points = shape.get("points")
                    shape_type = shape.get("shape_type")

                    if label not in class_map:
                        # print(f"Cảnh báo: Bỏ qua nhãn '{label}' không có trong class_map.")
                        continue

                    class_id = class_map[label]

                    # 1. Xử lý nhãn Đa giác (Polygon)
                    if shape_type == "polygon":
                        normalized_coords = normalize_polygon_coordinates(
                            points, img_width, img_height
                        )
                        # Định dạng Segmentation: [class_id] [x1 y1 x2 y2 ...]
                        out_f.write(f"{class_id} {normalized_coords}\n")
                    # 2. Xử lý nhãn Hình chữ nhật (Rectangle)
                    elif shape_type == "rectangle":
                        if is_segmentation:
                            # Nếu bạn đang huấn luyện mô hình Segmentation,
                            # chúng ta chuyển hình chữ nhật thành 4 điểm đa giác
                            # points là [[xmin, ymin], [xmax, ymax]]
                            xmin, ymin = points[0]
                            xmax, ymax = points[1]

                            # Chuyển thành 4 điểm: [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
                            # Lưu ý: Hàm normalize_polygon_coordinates sẽ xử lý tọa độ này
                            polygon_points = [
                                [xmin, ymin],
                                [xmax, ymin],
                                [xmax, ymax],
                                [xmin, ymax],
                            ]

                            normalized_coords = normalize_polygon_coordinates(
                                polygon_points, img_width, img_height
                            )
                            # Định dạng Segmentation: [class_id] [x1 y1 x2 y2 ...]
                            out_f.write(f"{class_id} {normalized_coords}\n")

                        else:
                            # Nếu bạn đang huấn luyện mô hình Detection, dùng format Bounding Box
                            normalized_coords = convert_rectangle_to_yolo(
                                points, img_width, img_height
                            )
                            # Định dạng Detection: [class_id] [x_c y_c w h]
                            out_f.write(f"{class_id} {normalized_coords}\n")

    print("Chuyển đổi hoàn tất! (Đã xử lý cả Polygon và Rectangle)")


if __name__ == "__main__":
    # Đường dẫn đến thư mục chứa các file JSON và ảnh
    json_folder = r"D:\0.TRAIN\0.MainAlign_LHC_PL_EO_V3\V3_LHC_NG_EO_221111"  # Thay thế bằng thư mục của bạn
    convert_json2txt(json_folder=json_folder, output_folder=json_folder)
