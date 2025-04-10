import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from PySide6.QtCore import Signal, QObject
import logging
import json
from pathlib import Path
import shutil
logger = logging.getLogger(__name__)

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

    @property
    def display_str_list(self):
        return [f"Image {i}" for i in range(len(self.images))]

    def draw_board(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Convert to grayscale and update image size.
        ret, cur_object_points, cur_image_points, charuco_corners, charuco_ids = self.board_dectect(img)
        if ret:
            img = cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids, (0, 255, 0))
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                
                success, rvec, tvec = cv2.solvePnP(cur_object_points, cur_image_points,
                                                   self.camera_matrix,
                                                   self.dist_coeffs)
                if success:
                    rotation = R.from_rotvec(rvec.ravel())
                    logger.info(f"rvec: {rvec.ravel()}, tvec: {tvec.ravel()}, rxyz: {rotation.as_euler('xyz', degrees=False)}")
                    img =cv2.drawFrameAxes(img, self.camera_matrix,
                                      self.dist_coeffs, rvec, tvec, 0.05)
                return rvec.ravel(), tvec.ravel(), img
        return None, None, img

    def board_dectect(self, img):
        # Convert to grayscale and update image size.
        logger.info(f"Image shape: {img.shape}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        self.image_size = gray.shape[::-1]  # (width, height)
        # Detect the board (returns charuco corners/ids plus marker corners/ids).
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        if charuco_ids is not None and charuco_corners is not None:
            if len(charuco_ids) > 4:
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

    @property
    def save_dir(self):
        return self.__save_dir

    def reset(self):
        self.images.clear()
        self.robot_poses.clear()
        self.objpoints.clear()
        self.imgpoints.clear()
        self.camera_to_board_rvecs.clear()
        self.camera_to_board_tvecs.clear()
        self.__manage_list.clear()
        self.__save_dir = None
        self.calibration_results = {}
        self.image_size = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.data_changed.emit()

    def load_camera_parameters(self, intrinsic, dist_coeffs):
        self.camera_matrix = intrinsic
        self.dist_coeffs = dist_coeffs

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