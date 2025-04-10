import cv2
from app.utils.logger import setup_logger
from app.threads.cam_reader import CameraReader
from app.utils.camera import HTTPCamera, FakeCamera

logger = setup_logger(__name__)


class MultiCamStreamer:
    def __init__(self, params: dict=None):
        self.params = params
        self.cameras = {}  # Dictionary to store cameras with user-defined names and indices
        self.camera_threads = {}  # Dictionary to store camera reader threads
    
    def get_frame(self, **kwargs):
        frames = {}
        thread: CameraReader
        for camera_name, thread in self.camera_threads.items():
            ret, frame = thread.get_frame()
            if ret:
                frames[camera_name] = frame
            else:
                frames[camera_name] = None
        return frames

    def disconnect(self):
        # Stop all camera reader threads
        print("Start to disconnect cameras...")
        for thread in self.camera_threads.values():
            thread.stop()
        self.camera_threads.clear()
        print("Camera reader threads stopped.")
        # Release all camera resources
        for camera_info in self.cameras.values():
            cap = camera_info.get('capture')
            if cap is not None:
                print(f"Releasing camera resources: {cap}")
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
            logger.info(f"Camera RES: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, ID: {idx}")
            if not cap.isOpened():
                return False, None
            return True, cap

    def add_camera(self, idx: int, name: str, http_url: str = None):
        """Add a camera to the streamer with a custom name"""
        success, cap = self.init_by_cam_id(idx, http_url)
        if success and 'control' in name:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 )
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
        if success:
            # Add camera to cameras dictionary
            self.cameras[name] = {
                'capture': cap,
                'id': idx,
                'name': name,
                'http_url': http_url if idx == 99 else None
            }
            # print("Camera added:", name)
            # Create and start a camera reader thread
            logger.info(f"Cam name: {name}, ID: {idx} success: {success}")
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
