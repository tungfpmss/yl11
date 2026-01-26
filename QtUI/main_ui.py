import sys
from PyQt5 import QtWidgets, uic


class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        uic.loadUi('main_window.ui', self)
        self.btn_Load.clicked.connect(self.update_label)

    def update_label(self):
        self.label.setText("Dữ liệu lải lơ!")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())