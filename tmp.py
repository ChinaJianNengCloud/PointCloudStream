import sys
import requests
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap

class MJPEGStreamReader(QThread):
    new_frame = Signal(np.ndarray)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = True

    def run(self):
        stream = requests.get(self.url, stream=True)
        bytes_buffer = b""

        for chunk in stream.iter_content(chunk_size=1024):
            if not self.running:
                break
            bytes_buffer += chunk
            a = bytes_buffer.find(b'\xff\xd8')  # Start of JPEG
            b = bytes_buffer.find(b'\xff\xd9')  # End of JPEG
            if a != -1 and b != -1 and b > a:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                img_array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.new_frame.emit(frame)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()


class MJPEGViewer(QWidget):
    def __init__(self, stream_url):
        super().__init__()
        self.setWindowTitle("MJPEG IP Camera Viewer")

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.thread = MJPEGStreamReader(stream_url)
        self.thread.new_frame.connect(self.update_image)
        self.thread.start()

    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    stream_url = "http://192.168.1.123:81/stream"  # Your MJPEG stream
    app = QApplication(sys.argv)
    viewer = MJPEGViewer(stream_url)
    viewer.resize(800, 600)
    viewer.show()
    sys.exit(app.exec())
