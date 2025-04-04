from app.utils.camera import HTTPCamera
from typing import Union
from PySide6.QtCore import QThread, QMutex
import cv2
import numpy as np

class CameraReader(QThread):
    """Thread for reading frames from cameras continuously"""
    
    def __init__(self, camera_name, camera_capture: Union[cv2.VideoCapture, HTTPCamera]):
        super().__init__()
        self.camera_name = camera_name
        self.camera = camera_capture
        self.running = True
        self.frame: np.ndarray = None
        self.frame_lock = QMutex()
        self.frame_ready = False
    
    def run(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self.frame_lock.lock()
                self.frame = frame
                self.frame_ready = True
                self.frame_lock.unlock()
            else:
                self.frame_lock.lock()
                self.frame_ready = False
                self.frame_lock.unlock()
                
            # Small delay to prevent CPU overuse
            QThread.msleep(10)
    
    def get_frame(self):
        self.frame_lock.lock()
        if not self.frame_ready:
            self.frame_lock.unlock()
            return False, None
        
        frame_copy = self.frame.copy() if self.frame is not None else None
        self.frame_lock.unlock()
        return True, frame_copy
    
    def stop(self):
        self.running = False
        self.wait()  # Wait for the thread to finish