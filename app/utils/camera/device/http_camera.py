import numpy as np
import cv2
import requests
from PySide6.QtCore import QThread, QMutex
import logging
import time
from threading import Event
logger = logging.getLogger(__name__)


class MJPEGStreamReader(QThread):
    """Thread for reading MJPEG stream from HTTP cameras using PyQt QThread"""
    
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.running = True
        self.frame = None
        self.frame_lock = QMutex()
        self.stream = None
        self.stop_event = Event()
        
    def run(self):
        try:
            # Use a short timeout to ensure we can break the connection if needed
            self.stream = requests.get(self.url, stream=True, timeout=5)
            bytes_buffer = b""

            for chunk in self.stream.iter_content(chunk_size=1024):
                if self.stop_event.is_set() or not self.running:
                    logger.info(f"MJPEG stream reader for {self.url} stopping")
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
                QThread.msleep(10)
        except requests.exceptions.RequestException as e:
            if not self.stop_event.is_set():  # Only log errors if not intentionally stopping
                logger.error(f"Request error in HTTP camera stream: {str(e)}")
        except Exception as e:
            if not self.stop_event.is_set():  # Only log errors if not intentionally stopping
                logger.error(f"Error in HTTP camera stream: {str(e)}")
        finally:
            self.running = False
            if self.stream and hasattr(self.stream, 'close'):
                try:
                    self.stream.close()
                except Exception as e:
                    logger.debug(f"Error while closing stream: {str(e)}")
                    pass
            logger.info(f"MJPEG stream reader for {self.url} exited")

    def read(self):
        self.frame_lock.lock()
        if self.frame is None:
            result = (False, None)
        else:
            result = (True, self.frame.copy())
        self.frame_lock.unlock()
        return result

    def stop(self):
        """Stop the stream reader thread with timeout"""
        logger.info(f"Stopping MJPEG stream reader for {self.url}")
        self.running = False
        self.stop_event.set()
        
        # Close the stream connection if possible to interrupt any blocking I/O
        if self.stream and hasattr(self.stream, 'close'):
            try:
                self.stream.close()
            except Exception as e:
                logger.debug(f"Error while closing stream in stop(): {str(e)}")
                pass
        
        if not self.wait(150):  # 150ms timeout
            logger.warning(f"Thread for {self.url} did not exit gracefully, terminating")
            self.terminate()
            self.wait(150)  # Give a little extra time after termination
        

class HTTPCamera:
    """Represents an HTTP camera using MJPEG streaming"""
    
    def __init__(self, url):
        self.url = url
        self.stream_reader = MJPEGStreamReader(url)
        self.stream_reader.start()
        
    def read(self):
        return self.stream_reader.read()
    
    def release(self):
        """Release the camera resources"""
        logger.info(f"Releasing HTTP camera: {self.url}")
        start_time = time.time()
        self.stream_reader.stop()
        duration = time.time() - start_time
        logger.info(f"HTTP camera released in {duration:.2f} seconds")
        
    def isOpened(self):
        return self.stream_reader.isRunning()


if __name__ == "__main__":
    pass