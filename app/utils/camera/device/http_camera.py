import numpy as np
import cv2
import requests
from PySide6.QtCore import QThread, QMutex
import logging
logger = logging.getLogger(__name__)


class MJPEGStreamReader(QThread):
    """Thread for reading MJPEG stream from HTTP cameras using PyQt QThread"""
    
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = True
        self.frame = None
        self.frame_lock = QMutex()
        
    def run(self):
        try:
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
                        self.frame_lock.lock()
                        self.frame = frame
                        self.frame_lock.unlock()
        except Exception as e:
            logger.error(f"Error in HTTP camera stream: {str(e)}")
            self.running = False

    def read(self):
        self.frame_lock.lock()
        if self.frame is None:
            result = (False, None)
        else:
            result = (True, self.frame.copy())
        self.frame_lock.unlock()
        return result

    def stop(self):
        self.running = False
        self.wait()  # Wait for the thread to finish


class HTTPCamera:
    """Represents an HTTP camera using MJPEG streaming"""
    
    def __init__(self, url):
        self.url = url
        self.stream_reader = MJPEGStreamReader(url)
        self.stream_reader.start()
        
    def read(self):
        return self.stream_reader.read()
    
    def release(self):
        self.stream_reader.stop()
        
    def isOpened(self):
        return self.stream_reader.isRunning()