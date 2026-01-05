import labelme2yolo

# Đường dẫn đến thư mục chứa các file .json của Labelme
input_dir = r"D:\9.Source_Code\DATASET\Cat"
output_dir = r"D:\9.Source_Code\DATASET\Cat_Yolo11_Seg"

# Khởi tạo converter
# Nếu dùng cho Keypoints, bạn cần cấu hình thêm labels
converter = labelme2yolo.Labelme2YOLO(input_dir, output_dir, label_list=["eyes", "nose", "ears"]) 

# Thực hiện chuyển đổi
converter.convert()