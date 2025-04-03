import numpy as np
import cv2
import requests
from typing import Union
from PySide6.QtCore import QThread, QMutex
from app.utils.logger import setup_logger
logger = setup_logger(__name__)

class FakeCamera:
    """Fake camera that generates synthetic RGBD frames for debugging."""
    def __init__(self):
        self.width = 640
        self.height = 480
        self.frame_idx = 0

    def connect(self, index):
        """Fake connect method, always returns True."""
        return True

    def release(self):
        pass

    def disconnect(self):
        """Fake disconnect method."""
        pass

    def read(self, both: bool = False):
        color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        center_x = int((self.frame_idx * 5) % self.width)
        center_y = self.height // 2
        cv2.circle(color_image, (center_x, center_y), 50, (0, 255, 0), -1)

        # Generate a dynamic depth image
        x = np.linspace(0, 2 * np.pi, self.width)
        y = np.linspace(0, 2 * np.pi, self.height)
        xx, yy = np.meshgrid(x, y)

        # Use a sine wave to create dynamic depth variations
        base_depth = 1000  # Base depth value in mm
        amplitude = 500  # Amplitude of the depth variation
        depth_image = base_depth + amplitude * np.sin(xx + self.frame_idx * 0.1)

        # Convert to uint16
        depth_image = depth_image.astype(np.uint16)

        # Randomly zero out some regions to simulate missing depth
        num_missing_regions = np.random.randint(5, 15)  # Random number of missing regions
        for _ in range(num_missing_regions):
            # Randomly choose the size and position of the missing region
            start_x = np.random.randint(0, self.width - 50)
            start_y = np.random.randint(0, self.height - 50)
            width = np.random.randint(20, 100)
            height = np.random.randint(20, 100)

            # Zero out the region
            depth_image[start_y:start_y + height, start_x:start_x + width] = 0
        self.frame_idx += 1
        if not both:
            return True, color_image
        return True, FakeRGBDFrame(depth_image, color_image)

    def capture_frame(self, enable_align_depth_to_color=True):
        """Generate synthetic depth and color images with dynamic depth regions."""
        return self.read(both=True)[1]


class FakeRGBDFrame:
    """Fake RGBD frame containing synthetic depth and color images."""
    def __init__(self, depth_image, color_image):
        self.depth = depth_image
        self.color = color_image


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


class CameraReader(QThread):
    """Thread for reading frames from cameras continuously"""
    
    def __init__(self, camera_name, camera_capture: Union[cv2.VideoCapture, HTTPCamera]):
        super().__init__()
        self.camera_name = camera_name
        self.camera = camera_capture
        self.running = True
        self.frame = None
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


class MultiCamStreamer:
    def __init__(self, params: dict=None):
        self.params = params
        self.cameras = {}  # Dictionary to store cameras with user-defined names and indices
        self.camera_threads = {}  # Dictionary to store camera reader threads
    
    def get_frame(self, **kwargs):
        frames = {}
        for camera_name, thread in self.camera_threads.items():
            ret, frame = thread.get_frame()
            if ret:
                frames[camera_name] = frame
            else:
                frames[camera_name] = None
        return frames

    def disconnect(self):
        # Stop all camera reader threads
        for thread in self.camera_threads.values():
            thread.stop()
        self.camera_threads.clear()
        
        # Release all camera resources
        for camera_info in self.cameras.values():
            cap = camera_info.get('capture')
            if cap is not None:
                cap.release()
        self.cameras.clear()

    def init_by_cam_id(self, idx: int, http_url: str = None):
        """Initialize camera by ID or HTTP URL
        
        Args:
            idx: Camera ID, use 99 for HTTP camera
            http_url: HTTP URL for MJPEG stream if idx is 99
            
        Returns:
            Tuple[bool, capture]: Success flag and capture object
        """
        if idx == -1:
            cap = FakeCamera()
            return True, cap
        elif idx == 99 and http_url:  # Special ID for HTTP cameras
            try:
                cap = HTTPCamera(http_url)
                logger.info(f"HTTP Camera connected: {http_url}")
                return True, cap
            except Exception as e:
                logger.error(f"Failed to connect to HTTP camera: {str(e)}")
                return False, None
        else:
            cap = cv2.VideoCapture(idx)
            logger.info(f"Camera RES: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            if not cap.isOpened():
                return False, None
            return True, cap

    def add_camera(self, idx: int, name: str, http_url: str = None):
        """Add a camera to the streamer with a custom name"""
        success, cap = self.init_by_cam_id(idx, http_url)
        if success:
            # Add camera to cameras dictionary
            self.cameras[name] = {
                'capture': cap,
                'id': idx,
                'name': name,
                'http_url': http_url if idx == 99 else None
            }
            
            # Create and start a camera reader thread
            thread = CameraReader(name, cap)
            thread.start()
            self.camera_threads[name] = thread
            
            return True
        return False

    def remove_camera(self, name: str):
        """Remove a camera from the streamer by name"""
        # Stop and remove camera thread if exists
        if name in self.camera_threads:
            self.camera_threads[name].stop()
            del self.camera_threads[name]
            
        # Release and remove camera if exists
        if name in self.cameras:
            cap = self.cameras[name].get('capture')
            if cap is not None:
                cap.release()
            del self.cameras[name]
            return True
        return False

    def camera_mode_init(self):
        """Initialize cameras from params"""
        camera_list = self.params.get('camera_list', [])
    
        # Initialize all cameras in the list
        success = True
        for camera in camera_list:
            cam_id = camera.get('id', -1)
            cam_name = camera.get('name', f'cam_{len(self.cameras)}')
            http_url = camera.get('http_url') if cam_id == 99 else None
            
            if not self.add_camera(cam_id, cam_name, http_url):
                success = False
                
        return success and len(self.cameras) > 0
