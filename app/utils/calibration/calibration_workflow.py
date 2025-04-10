import numpy as np
import cv2
import os
import time
import logging
try:
    from app.utils.robot.robot_utils import RobotInterface
    from app.utils.camera.camera_utils import CameraInterface
    from app.utils.calibration.calibration_data import CalibrationData
except ImportError:
    from app.utils.robot.robot_utils import RobotInterface
    from app.utils.camera.camera_utils import CameraInterface
    from app.utils.calibration.calibration_data import CalibrationData

# Configure logging
from app.utils.logger import setup_logger
logger = setup_logger(__name__)

MARKER_SIZE = [50, 100, 250, 1000]
MARKER_GRID = [4, 5, 6]

ARUCO_BOARD = {
    f'DICT_{grid}X{grid}_{size}': getattr(cv2.aruco, f'DICT_{grid}X{grid}_{size}')
    for size in MARKER_SIZE
    for grid in MARKER_GRID
    if hasattr(cv2.aruco, f'DICT_{grid}X{grid}_{size}')
}


class CalibrationProcess:
    def __init__(self, params: dict, camera_interface: CameraInterface, robot_interface: RobotInterface, calibration_data: CalibrationData):
        self.params = params
        self.directory = params.get('directory', os.getcwd())
        self.image_amount = params.get('ImageAmount', None)
        self.input_method = params.get('input_method', 'capture')
        self.folder_path = params.get('folder_path', None)
        self.pose_file_path = params.get('pose_file_path', None)
        self.load_intrinsic = params.get('load_intrinsic', False)
        self.intrinsic_path = params.get('intrinsic_path', None)
        self.cv_realtime_stream = params.get('cv_realtime_stream', False)
        self.camera = camera_interface
        self.robot = robot_interface
        self.calibration_dir = os.path.join(self.directory, 'Calibration_results')
        os.makedirs(self.calibration_dir, exist_ok=True)
        # Initialize CalibrationData
        self.calibration_data = calibration_data

    def capture_images(self):
        if self.input_method == 'capture':
            logger.info("Starting image capture...")
            self.camera.start_live_feedback()
            last_num_images = 0
            while len(self.calibration_data) < self.image_amount:
                current_num_images = len(self.calibration_data)
                if current_num_images > last_num_images:
                    logger.info(f"Captured image {current_num_images} of {self.image_amount}")
                    # Capture robot pose
                    robot_pose = self.robot.get_state('tcp')
                    # Capture the frame from the camera
                    img = self.camera.capture_frame()
                    self.calibration_data.append(img, robot_pose)
                    last_num_images = current_num_images
                time.sleep(0.1)
            self.camera.stop_live_feedback()
            logger.info("Image capture completed.")

        elif self.input_method == 'auto_calibrated_mode':
            if not self.pose_file_path:
                raise ValueError("Pose file path must be specified for 'auto_calibrated_mode'")

            # Read the pose file
            with open(self.pose_file_path, 'r') as f:
                lines = f.readlines()
            cartesian_poses_list = [line.strip().split() for line in lines]
            cartesian_poses_list = [np.array([float(x) for x in line], dtype=np.float32) for line in cartesian_poses_list]
            cartesian_poses_list = [np.hstack((line[0:3] / 1000, np.deg2rad(line[3:6]))) for line in cartesian_poses_list]

            for idx, cartesian_pose in enumerate(cartesian_poses_list):
                # Move robot to pose
                cartesian_pose_dict = self.robot.pose_array_to_euler_dict(cartesian_pose)
                joint_pose = self.robot.lebai.kinematics_inverse(cartesian_pose_dict)
                self.robot.lebai.movej(joint_pose, self.robot.acceleration, self.robot.velocity, self.robot.time_running, self.robot.radius)
                self.robot.lebai.wait_move()
                time.sleep(0.5)
                # Capture image
                img = self.camera.capture_frame()
                if img is None:
                    logger.warning(f"Failed to capture image at position {idx}")
                    continue
                robot_pose = self.robot.get_state('tcp')
                self.calibration_data.append(img, robot_pose)

            logger.info("Image capture completed.")

        elif self.input_method == 'load_from_folder':
            if not self.folder_path:
                raise ValueError("Folder path must be specified for 'load_from_folder' input method.")

            logger.info(f"Loading images from folder: {self.folder_path}")
            pose_file_path = os.path.join(self.folder_path, 'pose.txt')
            if not os.path.exists(pose_file_path):
                raise FileNotFoundError(f"Pose file not found at: {pose_file_path}")

            with open(pose_file_path, 'r') as pose_file:
                lines = pose_file.readlines()

            for idx, line in enumerate(lines):
                image_path = os.path.join(self.folder_path, f'{idx}.png')
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found at path: {image_path}")
                    continue
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Failed to load image at path: {image_path}")
                    continue
                logger.info(f"Loaded image {idx}.png")
                parts = line.strip().split()
                if len(parts) != 6:
                    logger.warning(f"Invalid pose format in line: {line}")
                    continue
                x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = map(float, parts)
                x_m = x_mm / 1000.0
                y_m = y_mm / 1000.0
                z_m = z_mm / 1000.0
                rx_rad = np.deg2rad(rx_deg)
                ry_rad = np.deg2rad(ry_deg)
                rz_rad = np.deg2rad(rz_deg)
                robot_pose = np.array([x_m, y_m, z_m, rx_rad, ry_rad, rz_rad])
                self.calibration_data.append(img, robot_pose)

            logger.info("Finished loading images and poses from folder")
        else:
            raise ValueError(f"Unknown input method: {self.input_method}")

    def run(self):
        self.capture_images()
        if len(self.calibration_data) == 0:
            logger.info("No valid images were captured or loaded. Exiting calibration process.")
            return

        if not self.calibration_data.objpoints or not self.calibration_data.imgpoints:
            logger.info("ChArUco corners were not found in any image. Exiting calibration process.")
            return

        # Do not load intrinsic parameters; directly perform calibration
        self.calibration_data.calibrate_camera()
        logger.info("Camera calibration completed.")

        self.calibration_data.board_pose_calculation()
        logger.info("Board pose calculated.")
        self.calibration_data.compute_reprojection_error()
        logger.info("Reprojection error computed.")
        self.calibration_data.calibrate_hand_eye()
        logger.info("Hand-eye calibration completed.")
        json_path = os.path.join(self.calibration_dir, 'calibration_results.json')
        self.calibration_data.save_calibration_data(json_path)
        self.calibration_data.save_img_and_pose()


def main(params):
    input_method = params.get('input_method', 'capture')

    # Create ChArUco dictionary and board
    charuco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_BOARD[params['board_type']])
    charuco_board = cv2.aruco.CharucoBoard(
        params['board_shape'],
        squareLength=params['board_square_size'] / 1000,
        markerLength=params['board_marker_size'] / 1000,
        dictionary=charuco_dict
    )
    calibration_data = CalibrationData(charuco_board)
    
    if input_method in ['capture', 'auto_calibrated_mode']:
        # Use cv2 VideoCapture with camera id 0
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Failed to open cv2 camera with id 0")
        camera_interface = CameraInterface(camera, calibration_data)
        robot_interface = RobotInterface()
        robot_interface.find_device()
        robot_interface.connect()
    elif input_method == 'load_from_folder':
        camera_interface = CameraInterface(None, calibration_data)
        robot_interface = None
    else:
        raise ValueError(f"Unknown input method: {input_method}")

    calibration_process = CalibrationProcess(params, camera_interface, robot_interface, calibration_data)
    calibration_process.run()


if __name__ == '__main__':
    params = {
        'directory': '.',  # Change to your directory if needed
        'ImageAmount': 13,
        'board_shape': (11, 6),
        'board_square_size': 23,  # mm
        'board_marker_size': 17.5,  # mm
        'input_method': 'auto_calibrated_mode',  # 'capture', 'load_from_folder', or 'auto_calibrated_mode'
        'folder_path': '_tmp',  # Specify the folder path if using 'load_from_folder'
        'pose_file_path': './poses.txt',  # Specify the pose file path for 'auto_calibrated_mode'
        'load_intrinsic': False,  # Intrinsic parameters will not be loaded
        'intrinsic_path': './Calibration_results/calibration_results.json',  # Not used since load_intrinsic is False
        'cv_realtime_stream': False,
        'board_type': 'DICT_4X4_100'
    }
    main(params)
