import gxipy as gx
from PIL import Image
import sys
import datetime
import numpy as np

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QTextEdit
from PyQt5.QtCore import QTimer, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap

from ultralytics import YOLO

class MainWindow(QMainWindow):
    def __init__(self, cam, parent=None):
        super(MainWindow, self).__init__(parent)
        self.cam = cam
        self.setWindowTitle('完美复刻的demo')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.prediction_label = QLabel(self)
        self.prediction_label.setAlignment(Qt.AlignCenter)

        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_acquisition)

        self.pause_button = QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.pause_acquisition)
        self.pause_button.setEnabled(False)

        self.resume_button = QPushButton('Resume', self)
        self.resume_button.clicked.connect(self.resume_acquisition)
        self.resume_button.setEnabled(False)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)

        self.close_button = QPushButton('Close the Work', self)
        self.close_button.clicked.connect(self.close_work)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.resume_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.close_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.prediction_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.log_text)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)

        self.model = YOLO("/home/zhenlin/chenrongyu/python_Prj/yolov8n.pt")  # Load YOLOv8 model

    def log(self, message):
        self.log_text.append(f"{datetime.datetime.now()}: {message}")

    @pyqtSlot()
    def start_acquisition(self):
        self.cam.stream_on()
        self.timer.start(1000 // 30)  # 30 fps
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log("Acquisition started")

    @pyqtSlot()
    def pause_acquisition(self):
        self.timer.stop()
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(True)
        self.log("Acquisition paused")

    @pyqtSlot()
    def resume_acquisition(self):
        self.timer.start(1000 // 30)  # 30 fps
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(False)
        self.log("Acquisition resumed")

    @pyqtSlot()
    def stop_acquisition(self):
        self.timer.stop()
        self.cam.stream_off()
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.image_label.clear()
        self.prediction_label.clear()
        self.log("Acquisition stopped")

    @pyqtSlot()
    def close_work(self):
        self.stop_acquisition()
        self.cam.close_device()
        self.log("采集工作完美完成")
        self.close()

    @pyqtSlot()
    def update_image(self):
        raw_image = self.cam.data_stream[0].get_image()
        if raw_image is None:
            return
        rgb_image = raw_image.convert("RGB")
        rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)
        numpy_image = rgb_image.get_numpy_array()

        if numpy_image is None:
            return

        img = Image.fromarray(numpy_image, 'RGB')
        results = self.model([img] , stream=True)

        # 显示原始图像
        qimg = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

        # 处理并显示图像
        for result in results:
            result_img = result.plot()
            result_qimg = QImage(result_img.tobytes(), result_img.width, result_img.height, QImage.Format_RGB888)
            result_pixmap = QPixmap.fromImage(result_qimg).scaled(self.prediction_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.prediction_label.setPixmap(result_pixmap)

    def resizeEvent(self, event):
        if self.image_label.pixmap():
            self.image_label.setPixmap(self.image_label.pixmap().scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if self.prediction_label.pixmap():
            self.prediction_label.setPixmap(self.prediction_label.pixmap().scaled(self.prediction_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

def main():
    framerate_set = 1  # 设置帧率
    print("Initializing...")

    # 创建设备
    device_manager = gx.DeviceManager()  # 创建设备对象
    dev_num, dev_info_list = device_manager.update_device_list()  # 枚举所有可用设备
    if dev_num == 0:
        print("Number of enumerated devices is 0")
        return
    else:
        print(f"创建设备成功，设备号为：{dev_num}")

    # 打开设备
    strSN = dev_info_list[0].get("sn")  # 获取设备基本信息列表
    cam = device_manager.open_device_by_sn(strSN)  # 通过序列号打开设备

    # exit when the camera is a mono camera
    if not cam.PixelColorFilter.is_implemented():
        print("This sample does not support mono camera.")
        cam.close_device()
        return

    # 设置连续采集
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

    # 设置曝光时间
    cam.ExposureTime.set(10000.0)

    # 设置通道流包长
    cam.GevSCPSPacketSize.set(8164)

    # 设置图像增益
    cam.Gain.set(10.0)

    # 设置参数图像增强
    global gamma_lut, contrast_lut, color_correction_param
    gamma_lut = gx.Utility.get_gamma_lut(cam.GammaParam.get()) if cam.GammaParam.is_readable() else None
    contrast_lut = gx.Utility.get_contrast_lut(cam.ContrastParam.get()) if cam.ContrastParam.is_readable() else None
    color_correction_param = cam.ColorCorrectionParam.get() if cam.ColorCorrectionParam.is_readable() else 0

    cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
    cam.AcquisitionFrameRate.set(framerate_set)
    print(f"用户设置的帧率为: {framerate_set} fps")
    print(f"当前采集的帧率为: {cam.CurrentAcquisitionFrameRate.get()} fps")

    
    
    app = QApplication(sys.argv)
    mainWindow = MainWindow(cam)
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
