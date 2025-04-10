import numpy as np
import cv2
import os
import time
import sys
import json
import shutil
from pathlib import Path
import logging
from scipy.spatial.transform import Rotation as R
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QPushButton
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt, Signal, QObject
import signal


# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# Define available ArUco dictionary names
MARKER_SIZE = [50, 100, 250, 1000]
MARKER_GRID = [4, 5, 6]
ARUCO_BOARD = {
    f'DICT_{grid}X{grid}_{size}': getattr(cv2.aruco, f'DICT_{grid}X{grid}_{size}')
    for size in MARKER_SIZE
    for grid in MARKER_GRID
    if hasattr(cv2.aruco, f'DICT_{grid}X{grid}_{size}')
}

########################################################################
# CalibrationData: Camera-only calibration class using cv2.solvePnP.
########################################################################
class CalibrationData(QObject):
    data_changed = Signal()
    
    def __init__(self, board: cv2.aruco.CharucoBoard, save_dir: str = None):
        super().__init__()
        self.board = board
        self.detector = cv2.aruco.CharucoDetector(board)
        self.images: list[np.ndarray] = []
        self.robot_poses: list[np.ndarray] = []  # Not used in camera-only mode.
        self.objpoints: list[np.ndarray] = []
        self.imgpoints: list[np.ndarray] = []
        self.camera_to_board_rvecs: list[np.ndarray] = []
        self.camera_to_board_tvecs: list[np.ndarray] = []
        self.__manage_list: list[str] = []
        self.__save_dir: Path = Path(save_dir) if save_dir is not None else None
        self.calibration_results = {}
        self.image_size = None
        self.camera_matrix = None
        self.dist_coeffs = None

    def board_dectect(self, img):
        # Convert to grayscale and update image size.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_size = gray.shape[::-1]  # (width, height)
        # Detect the board (returns charuco corners/ids plus marker corners/ids).
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        if charuco_ids is not None and charuco_corners is not None:
            if len(charuco_ids) > 5:
                cur_object_points, cur_image_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
                ret = True
            else:
                cur_object_points, cur_image_points = None, None
                ret = False
                logger.warning("Not enough markers detected in image")
        else:
            logger.warning("No valid Charuco corners detected in image")
            cur_object_points, cur_image_points = None, None
            ret = False
        # Return extra information for drawing purposes.
        return ret, cur_object_points, cur_image_points, charuco_corners, charuco_ids

    def append(self, image: np.ndarray, robot_pose: np.ndarray = None, recalib=False):
        ret, cur_object_points, cur_image_points, _, _ = self.board_dectect(image)
        if ret:
            self.images.append(image.copy())
            self.imgpoints.append(cur_image_points)
            self.objpoints.append(cur_object_points)
            # In camera-only mode, robot poses are not used.
            self.robot_poses.append(robot_pose)
            logger.info(f"Board detected in image, image added. Count: {len(self.images)}")
            if recalib:
                self.calibrate_all()
            self.data_changed.emit()
        else:
            logger.warning("Failed to detect board in image, image not added.")

    def __len__(self):
        return len(self.images)

    def pop(self, index: int):
        if len(self.images) > index:
            self.images.pop(index)
            self.robot_poses.pop(index)
            self.objpoints.pop(index)
            self.imgpoints.pop(index)
            self.data_changed.emit()

    def calibrate_camera(self):
        logger.info(f"Number of image points: {len(self.imgpoints)}")
        if len(self.imgpoints) >= 3:
            ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, self.image_size, None, None
            )
            if ret:
                self.data_changed.emit()
                logger.info("Camera calibration successful")
            else:
                logger.warning("Camera calibration failed")
        else:
            logger.warning("Not enough object points and image points for calibration")

    def board_pose_calculation(self):
        if hasattr(self, 'camera_matrix') and hasattr(self, 'dist_coeffs'):
            invalid_index = []
            self.camera_to_board_rvecs = []
            self.camera_to_board_tvecs = []
            for idx, (cur_object_points, cur_image_points) in enumerate(zip(self.objpoints, self.imgpoints)):
                ret, rvec, tvec = cv2.solvePnP(
                    cur_object_points, cur_image_points, self.camera_matrix, self.dist_coeffs
                )
                if ret:
                    self.camera_to_board_rvecs.append(rvec)
                    self.camera_to_board_tvecs.append(tvec)
                else:
                    invalid_index.append(idx)
                    logger.warning(f"Could not solvePnP for image points {cur_image_points}")
            invalid_index = invalid_index[::-1]
            for idx in invalid_index:
                self.pop(idx)
        else:
            logger.warning("Camera matrix and distortion coefficients are not available.")

    def calibrate_all(self):
        if len(self.images) >= 3:
            self.calibrate_camera()
            self.board_pose_calculation()
            self.data_changed.emit()
            return True
        return False

    def save_calibration_data(self, path: str):
        if self.camera_matrix is None or self.dist_coeffs is None:
            logger.warning("Camera matrix and distortion coefficients are not available.")
            return
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
        }
        path = Path(path)
        with open(path, 'w') as json_file:
            json.dump(calibration_data, json_file, indent=2)
        logger.info(f"Calibration results saved to {path}")

    def compute_reprojection_error(self):
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.camera_to_board_rvecs[i], self.camera_to_board_tvecs[i],
                self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(self.objpoints)
        logger.info(f"Total reprojection error: {mean_error}")
        return mean_error
        
    def save_img_and_pose(self):
        if self.__save_dir:
            if self.__save_dir.exists():
                shutil.rmtree(str(self.__save_dir))
            self.__save_dir.mkdir(parents=True, exist_ok=True)
            images_dir = self.__save_dir / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)
            pose_file_path = self.__save_dir / 'pose.txt'
            with open(pose_file_path, 'w') as pose_file:
                for idx, (image, robot_pose) in enumerate(zip(self.images, self.robot_poses)):
                    img_path = images_dir / f'{idx}.png'
                    cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # For camera-only calibration, we simply write the image index.
                    pose_file.write(f"{idx}\n")
            logger.info(f"Saved images and poses to {self.__save_dir}")

########################################################################
# PySideCameraInterface: Manual capture UI using PySide6.
########################################################################
class PySideCameraInterface:
    def __init__(self, camera: cv2.VideoCapture, calibration_data: CalibrationData, desired_image_count: int = None):
        self.camera = camera
        self.calibration_data = calibration_data
        self.desired_image_count = desired_image_count
        self.live = False
        self.app = None
        self.window = None
        self.timer = None

    def start_live_feedback(self):
        self.live = True
        self.app = QApplication.instance() or QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("Live Feed - Manual Capture")
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_current_frame)
        layout.addWidget(self.capture_button)
        self.window.resize(800, 600)
        self.window.show()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self, signum, frame):
        print("SIGINT received, quitting application...")
        self.app.quit()

    def update_frame(self):
        if self.camera is None:
            return
        ret, frame = self.camera.read()
        if ret:
            # Detect board and obtain corner information.
            board_detected, obj_pts, img_pts, charuco_corners, charuco_ids = self.calibration_data.board_dectect(frame)
            if board_detected:
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
                if self.calibration_data.camera_matrix is not None and self.calibration_data.dist_coeffs is not None:
                    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts,
                                                       self.calibration_data.camera_matrix,
                                                       self.calibration_data.dist_coeffs)
                    if success:
                        rotation = R.from_rotvec(rvec.ravel())
                        logger.info(f"rvec: {rvec.ravel()}, tvec: {tvec.ravel()}, rxyz: {rotation.as_euler('xyz', degrees=False)}")
                        cv2.drawFrameAxes(frame, self.calibration_data.camera_matrix,
                                          self.calibration_data.dist_coeffs, rvec, tvec, 0.05)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))
        if self.app:
            self.app.processEvents()

    def capture_current_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("Failed to capture frame from camera.")
            return
        prev_count = len(self.calibration_data)
        # Trigger calibration automatically if at least 3 images will be available.
        recalib = (prev_count + 1 >= 3)
        self.calibration_data.append(frame, recalib=recalib)
        new_count = len(self.calibration_data)
        if new_count > prev_count:
            logger.info(f"Image captured successfully. Total valid images: {new_count}")
        else:
            logger.info("No valid Charuco corners detected in image. Image not added.")
        self.window.setWindowTitle(f"Live Feed - {new_count}/{self.desired_image_count} images captured")
        if self.desired_image_count is not None and new_count >= self.desired_image_count:
            logger.info("Desired image count reached. Closing live feed.")
            self.stop_live_feedback()

    def capture_frame(self):
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None

    def stop_live_feedback(self):
        self.live = False
        if self.timer:
            self.timer.stop()
        if self.window:
            self.window.close()
        if self.app:
            self.app.processEvents()

########################################################################
# CameraCalibrationProcess: Orchestrates the calibration workflow.
########################################################################
class CameraCalibrationProcess:
    def __init__(self, params: dict, camera_interface: PySideCameraInterface, calibration_data: CalibrationData):
        self.params = params
        self.directory = params.get('directory', os.getcwd())
        self.image_amount = params.get('ImageAmount', None)
        self.input_method = params.get('input_method', 'capture')
        self.folder_path = params.get('folder_path', None)
        self.cv_realtime_stream = params.get('cv_realtime_stream', False)
        self.camera = camera_interface
        self.calibration_dir = os.path.join(self.directory, 'Calibration_results')
        os.makedirs(self.calibration_dir, exist_ok=True)
        self.calibration_data = calibration_data

    def capture_images(self):
        if self.input_method == 'capture':
            logger.info("Starting manual image capture...")
            self.camera.start_live_feedback()
            # The PySide6 event loop will block until the window is closed.
            self.camera.app.exec()
            logger.info("Manual image capture finished.")
        elif self.input_method == 'load_from_folder':
            if not self.folder_path:
                raise ValueError("Folder path must be specified for 'load_from_folder' input method.")
            logger.info(f"Loading images from folder: {self.folder_path}")
            image_files = sorted([f for f in os.listdir(self.folder_path) if f.endswith('.png')])
            for image_file in image_files:
                image_path = os.path.join(self.folder_path, image_file)
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Failed to load image at path: {image_path}")
                    continue
                logger.info(f"Loaded image: {image_file}")
                self.calibration_data.append(img)
            logger.info("Finished loading images from folder")
        else:
            raise ValueError(f"Unknown input method: {self.input_method}")

    def run(self):
        self.capture_images()
        if len(self.calibration_data) == 0:
            logger.info("No valid images were captured or loaded. Exiting calibration process.")
            return
        if len(self.calibration_data.objpoints) == 0 or len(self.calibration_data.imgpoints) == 0:
            logger.info("Charuco corners were not found in any image. Exiting calibration process.")
            return
        self.calibration_data.calibrate_camera()
        logger.info("Camera calibration completed.")
        self.calibration_data.board_pose_calculation()
        self.calibration_data.compute_reprojection_error()
        json_path = os.path.join(self.calibration_dir, 'calibration_results.json')
        self.calibration_data.save_calibration_data(json_path)
        self.calibration_data.save_img_and_pose()

########################################################################
# Main entry point.
########################################################################
def main(params):
    input_method = params.get('input_method', 'capture')
    # Create the Charuco board using the selected dictionary.
    charuco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_BOARD[params['board_type']])
    charuco_board = cv2.aruco.CharucoBoard(
        params['board_shape'],
        squareLength=params['board_square_size'] / 1000,
        markerLength=params['board_marker_size'] / 1000,
        dictionary=charuco_dict
    )
    calibration_data = CalibrationData(charuco_board, save_dir='./Calibration_results')
    if input_method == 'capture':
        camera = cv2.VideoCapture(6)
        # Set MJPEG format
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Set resolution and FPS
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 )
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.set(cv2.CAP_PROP_FPS, 30)
        if not camera.isOpened():
            raise RuntimeError("Failed to open cv2 camera with id 0")
        camera_interface = PySideCameraInterface(camera, calibration_data, desired_image_count=params['ImageAmount'])
    elif input_method == 'load_from_folder':
        camera_interface = PySideCameraInterface(None, calibration_data)
    else:
        raise ValueError(f"Unknown input method: {input_method}")

    calibration_process = CameraCalibrationProcess(params, camera_interface, calibration_data)
    calibration_process.run()

if __name__ == '__main__':
    params = {
        'directory': '.',
        'ImageAmount': 50,            # Desired number of valid calibration images
        'board_shape': (3, 5),
        'board_square_size': 23,      # in mm
        'board_marker_size': 17.5,    # in mm
        'input_method': 'capture',    # 'capture' or 'load_from_folder'
        'folder_path': '_tmp',        # Used if input_method is 'load_from_folder'
        'cv_realtime_stream': False,
        'board_type': 'DICT_4X4_100'
    }
    main(params)
