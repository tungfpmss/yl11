import shutil
import sys
import os
from pathlib import Path

import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QUrl, pyqtSignal, QThread
from ultralytics import YOLO

# Import class từ file bạn đã convert
from main_window import Ui_MainWindow


class YOLOWorker(QThread):
    # Tín hiệu gửi frame đã vẽ kết quả về UI
    frame_ready = pyqtSignal(QImage)
    # bienso_ready = pyqtSignal(QImage)
    # Tín hiệu báo khi video kết thúc
    video_finished = pyqtSignal()

    def __init__(self, model_path):
        super().__init__()
        self.video_path = ""
        self.is_running = False
        # Load model với nhiệm vụ detect/track
        self.model = YOLO(r'..\PRETRAINED\Yolo11_Object_Detection\yolo11n.pt')
        # self.model_cls = YOLO(r'..\PRETRAINED\Yolo11_Classification\yolo11n-cls.pt')
        model_ov = os.path.join(Path(model_path).parent, Path(model_path).stem + '_openvino_model')
        # if os.path.exists(model_ov):
        #     shutil.rmtree(model_ov)

        if not os.path.exists(model_ov):
            self.model = YOLO(model_path)
            self.model.export(format='OpenVINO', half=True, device='cpu')

        self.model = YOLO(model_ov, task='detect')

    def set_video(self, path):
        self.video_path = path

    def run(self):
        # Mở luồng video bằng OpenCV
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        self.is_running = True

        # Sử dụng model.track dưới dạng Generator để xử lý luồng stream video
        # stream=True giúp tiết kiệm bộ nhớ khi xử lý video dài
        results = self.model.track(
            source=self.video_path,
            persist=True,
            stream=True,
            conf=0.25,
            device='cpu',
            tracker='bytetrack.yaml'
        )

        for r in results:
            if not self.is_running:
                break

            # Lấy frame đã được vẽ (annotated frame)
            annotated_frame = r.plot()

            # Chuyển đổi màu từ BGR sang RGB cho PyQt5
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w

            # Tạo QImage từ dữ liệu ảnh
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Gửi frame về Main UI để hiển thị (tạo bản sao để tránh lỗi bộ nhớ)
            self.frame_ready.emit(qt_img.copy())

            # frame = r.orig_img
            # if r.boxes.id is not None:
            #     boxes = r.boxes.xyxy.cpu().numpy()
            #     classes = r.boxes.cls.cpu().numpy()
            #     for box, cls in boxes, classes:
            #         x1, y1, x2, y2 = map(int, box)
            #         cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            #         bienso = frame[y1:y2, x1:x2]
            #         # self.bienso_ready.emit(bienso.copy())

        # Giải phóng tài nguyên
        cap.release()
        self.is_running = False
        self.video_finished.emit()

    def stop(self):
        self.is_running = False
        self.wait()  # Đợi luồng kết thúc hẳn


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 1. Khai báo biến quản lý
        self.worker = None
        self.video_list = []
        self.current_index = -1
        self.model_path = self.ui.led_model.text()

        # 2. Kết nối Button
        self.ui.btn_init.clicked.connect(self.init_yolo)
        self.ui.btn_load.clicked.connect(self.load_folder)
        self.ui.btn_next.clicked.connect(self.next_video)
        self.ui.btn_play.clicked.connect(self.start_prediction)

    def init_yolo(self):
        self.model_path = self.ui.led_model.text()
        # Khởi tạo Worker (Luồng phụ)
        self.worker = YOLOWorker(self.model_path)
        self.worker.frame_ready.connect(self.display_frame)
        # self.worker.bienso_ready.connect(self.display_bienso)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục")
        path = Path(folder)
        if folder:
            # self.video_list = [os.path.join(folder, f) for f in os.listdir(folder)
            #                    if f.lower().endswith(('.mp4', '.avi'))]
            self.video_list = list(path.glob('*.mp4'))
            if self.video_list:
                self.current_idx = 0
                print(f"Đã load {len(self.video_list)} video")

    def start_prediction(self):
        if self.current_idx == -1: return

        if self.worker.isRunning():
            self.worker.stop()
            self.ui.btn_play.setText("Play")
        else:
            self.worker.set_video(self.video_list[self.current_idx])
            self.worker.start()  # Bắt đầu chạy luồng run()
            self.ui.btn_play.setText("Stop")

    def display_frame(self, qt_img):
        # Hiển thị ảnh lên Label mà không lo bị treo giao diện
        pixmap = QPixmap.fromImage(qt_img)
        self.ui.lbl_video.setPixmap(pixmap.scaled(
            self.ui.lbl_video.width(),
            self.ui.lbl_video.height(),
            aspectRatioMode=1
        ))

    def display_bienso(self, qt_img):
        # Hiển thị ảnh lên Label mà không lo bị treo giao diện
        pixmap = QPixmap.fromImage(qt_img)
        self.ui.lbl_bienso.setPixmap(pixmap.scaled(
            self.ui.lbl_bienso.width(),
            self.ui.lbl_bienso.height(),
            aspectRatioMode=1
        ))

    def next_video(self):
        self.worker.stop()
        if self.video_list:
            self.current_idx = (self.current_idx + 1) % len(self.video_list)
            self.start_prediction()

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
