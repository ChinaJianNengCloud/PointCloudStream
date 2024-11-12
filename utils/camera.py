import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import logging
import threading
import open3d as o3d

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)



class CameraInterface:
    def __init__(self, camera, charuco_dict: cv2.aruco.Dictionary, charuco_board: cv2.aruco.CharucoBoard):
        self.camera = camera
        self.charuco_dict = charuco_dict
        self.charuco_board = charuco_board
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)

        self.live_feedback_thread = None
        self.live_feedback_running = False
        self.captured_images = []
        self.camera_matrix = None
        self.dist_coeffs = None

    def start_live_feedback(self):
        self.live_feedback_running = True
        logger.info("Starting live feedback...")
        self.live_feedback_thread = threading.Thread(target=self._live_feedback_loop)
        self.live_feedback_thread.start()

    def stop_live_feedback(self):
        self.live_feedback_running = False
        if self.live_feedback_thread is not None:
            self.live_feedback_thread.join()
            self.live_feedback_thread = None
        logger.info("Live feedback stopped.")
        cv2.destroyAllWindows()

    def _live_feedback_loop(self):
        cv2.namedWindow('Live Feedback')
        while self.live_feedback_running:
            frame = self.capture_frame()
            if frame is None:
                continue
            self._process_and_display_frame(frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.live_feedback_running = False
                break
            elif key == ord(' '):  # Press space to capture image
                self.capture_image(frame)

    def capture_frame(self):
        rgbd = self.camera.capture_frame(True)
        if rgbd is None:
            return None
        color = np.asarray(rgbd.color)
        return cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)

    def capture_image(self, image: np.ndarray):
        self.captured_images.append(image)
        logger.info(f"Image captured. Total images: {len(self.captured_images)}")
        if len(self.captured_images) >= 3:
            self.calibrate_camera()

    def calibrate_camera(self):
        logger.info("Starting camera calibration...")
        all_image_points = []
        all_object_points = []
        image_size = None

        for img in self.captured_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
            
            if charuco_ids is not None and charuco_corners is not None:
                # Use matchImagePoints to get object points and image points
                object_points, image_points = self.charuco_board.matchImagePoints(charuco_corners, charuco_ids)
                
                # Append the object and image points for calibration
                all_object_points.append(object_points)
                all_image_points.append(image_points)
                
                if image_size is None:
                    image_size = gray.shape[::-1]

        if len(all_image_points) > 0:
            # Use cv2.calibrateCamera for camera calibration
            ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
                all_object_points, all_image_points, image_size, None, None
            )
            if ret:
                logger.info("Camera calibration successful")
            else:
                logger.warning("Camera calibration failed")
        else:
            logger.warning("No valid Charuco corners detected for calibration")

    def _process_and_display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)
        
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame_rgb, marker_corners, marker_ids)
            if charuco_corners is not None and charuco_ids is not None:
                # logger.info(f"Detected Charuco IDs: {charuco_ids.ravel()}")
                cv2.aruco.drawDetectedCornersCharuco(frame_rgb, charuco_corners, charuco_ids)
                
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    # Use matchImagePoints to get object and image points for pose estimation
                    object_points, image_points = self.charuco_board.matchImagePoints(charuco_corners, charuco_ids)
                    
                    # Estimate pose using solvePnP
                    try:
                        success, rvec, tvec = cv2.solvePnP(
                            object_points, image_points, self.camera_matrix, self.dist_coeffs
                        )
                    except:
                        success = False
                    if success:
                        cv2.drawFrameAxes(frame_rgb, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
            else:
                logger.warning("No valid Charuco corners detected for pose estimation")

        cv2.imshow('Live Feedback', frame_rgb)

    def clear(self):
        self.captured_images.clear()


def camera_test(params):
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    charuco_board = cv2.aruco.CharucoBoard(
        params['board_shape'],
        squareLength=params['board_square_size'] / 1000,
        markerLength=params['board_marker_size'] / 1000,
        dictionary=charuco_dict
    )

    camera_config = './default_config.json'
    sensor_config = o3d.io.read_azure_kinect_sensor_config(camera_config)
    camera = o3d.io.AzureKinectSensor(sensor_config)
    if not camera.connect(0):
        raise RuntimeError('Failed to connect to Azure Kinect sensor')
    camera_interface = CameraInterface(camera, charuco_dict, charuco_board)
    camera_interface.start_live_feedback()

if __name__ == '__main__':
    params = {
        'directory': '.',  # Change to your directory if needed
        'ImageAmount': 13,
        'board_shape': (11, 6),
        'board_square_size': 23, # mm
        'board_marker_size': 17.5, # mm
        'input_method': 'auto_calibrated_mode',  # 'capture', 'load_from_folder', or 'auto_calibrated_mode'
        'folder_path': '_tmp',  # Specify the folder path if using 'load_from_folder'
        'pose_file_path': './poses.txt',  # Specify the pose file path for 'auto_calibrated_mode'
        'load_intrinsic': True,  # Set to True or False
        'intrinsic_path': './Calibration_results/calibration_results.json',  # Path to the intrinsic JSON file
        'cv_realtime_stream': False
    }


    camera_test(params)
