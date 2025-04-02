import sys
import subprocess
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QWidget, QComboBox, QPushButton
from PySide6.QtGui import QImage, QPixmap

class VideoManager:
    """Manages video devices and operations."""

    def find_video_devices(self):
        """Get a list of available video devices."""
        try:
            # Run the v4l2-ctl command to list devices
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                    capture_output=True, text=True)
            
            # Parse the output to extract device names and paths
            devices = []
            lines = result.stdout.strip().split('\n')
            
            current_device_name = None
            for line in lines:
                if not line.startswith('\t'):
                    # This is a device name line
                    current_device_name = line.strip().rstrip(':')
                elif line.strip().startswith('/dev/video'):
                    # This is a device path line
                    device_path = line.strip()
                    has_formats = self.check_device_formats(device_path)
                    
                    if has_formats:
                        devices.append({
                            'name': current_device_name.split(':')[0].strip(),
                            'path': device_path
                        })
            
            return devices
        except Exception as e:
            print(f"Error getting video devices: {e}")
            return []

    def check_device_formats(self, device_path):
        """Check if a device has any supported formats."""
        try:
            # Run v4l2-ctl to check formats
            result = subprocess.run(
                ['v4l2-ctl', '-d', device_path, '--list-formats-ext'],
                capture_output=True, text=True
            )
            
            # A valid device with formats will have lines containing "Size:"
            output = result.stdout
            return "Size:" in output
        except Exception as e:
            print(f"Error checking formats for {device_path}: {e}")
            return False

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera Display")
        self.setGeometry(100, 100, 1200, 600)

        # Create VideoManager instance
        self.video_manager = VideoManager()
        self.devices = self.video_manager.find_video_devices()

        # Check if there are at least two valid cameras
        if len(self.devices) < 2:
            print("Not enough valid cameras found.")
            sys.exit()

        # Create a QWidget to hold everything
        self.widget = QWidget(self)
        self.setCentralWidget(self.widget)

        # Create the layout for the cameras
        self.layout = QHBoxLayout()
        self.widget.setLayout(self.layout)

        # Create a combo box to select the left camera
        self.camera_selector_left = QComboBox(self)
        self.camera_selector_left.addItem("Select Left Camera", None)
        for device in self.devices:
            self.camera_selector_left.addItem(device['name'], device['path'])
        self.layout.addWidget(self.camera_selector_left)

        # Create a combo box to select the right camera
        self.camera_selector_right = QComboBox(self)
        self.camera_selector_right.addItem("Select Right Camera", None)
        for device in self.devices:
            self.camera_selector_right.addItem(device['name'], device['path'])
        self.layout.addWidget(self.camera_selector_right)

        # Start button
        self.start_button = QPushButton("Start Cameras", self)
        self.start_button.clicked.connect(self.start_cameras)
        self.layout.addWidget(self.start_button)

        # Create two labels to display the camera feeds
        self.left_label = QLabel(self)
        self.left_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.left_label)

        self.right_label = QLabel(self)
        self.right_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.right_label)

        # Timer to refresh the frames from the cameras
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_cameras)

        # OpenCV video capture objects for the selected cameras
        self.cap_left = None
        self.cap_right = None

    def start_cameras(self):
        """Start the camera feeds"""
        left_camera_path = self.camera_selector_left.currentData()
        right_camera_path = self.camera_selector_right.currentData()

        if not left_camera_path or not right_camera_path:
            print("Please select both cameras.")
            return

        # Open video capture for the selected cameras
        self.cap_left = cv2.VideoCapture(left_camera_path)
        self.cap_right = cv2.VideoCapture(right_camera_path)

        self.timer.start(30)  # Refresh rate (30 ms)
        self.update_cameras()

    def update_cameras(self):
        """Update the camera frames"""
        if self.cap_left is None or self.cap_right is None:
            return

        # Read frames from both cameras
        ret1, frame1 = self.cap_left.read()
        ret2, frame2 = self.cap_right.read()

        if ret1 and ret2:
            # Convert frames to RGB
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # Convert numpy arrays to QImage
            h, w, ch = frame1.shape
            bytes_per_line = ch * w
            image1 = QImage(frame1.data, w, h, bytes_per_line, QImage.Format_RGB888)

            h, w, ch = frame2.shape
            bytes_per_line = ch * w
            image2 = QImage(frame2.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Update the labels with the new images
            self.left_label.setPixmap(QPixmap.fromImage(image1))
            self.right_label.setPixmap(QPixmap.fromImage(image2))

    def closeEvent(self, event):
        """Close the camera streams when the app closes"""
        if self.cap_left:
            self.cap_left.release()
        if self.cap_right:
            self.cap_right.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
