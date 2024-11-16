import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import logging
import json
from pathlib import Path
import shutil

# Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class CalibrationData:
    def __init__(self, board: cv2.aruco.CharucoBoard, save_dir: str = None):
        self.board = board
        self.detector = cv2.aruco.CharucoDetector(board)
        self.images: list[np.ndarray] = []
        self.robot_poses: list[np.ndarray] = []
        self.objpoints: list[np.ndarray] = []
        self.imgpoints: list[np.ndarray] = []
        self.camera_to_board_rvecs: list[np.ndarray] = []
        self.camera_to_board_tvecs: list[np.ndarray] = []
        self.__manage_list: list[str] = []
        self.save_dir = Path(save_dir) if save_dir else None

        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_results = {}
        self.image_size = None

    @property
    def display_str_list(self) -> list[str]:
        if self.camera_matrix is None:
            if any(x is None for x in self.robot_poses):
                return ["p_cam:"+str(i) for i in range(len(self.images))]
            else:
                return ["p_cam:"+str(i) + " p_arm:"+str(i) for i in range(len(self.images))]
        if len(self.camera_to_board_tvecs) > 0:
            cam_tvec_string =  [np.array2string(value.ravel(), 
                                        formatter={'float_kind': lambda x: f"{x:.2f}"}) 
                                        for value in self.camera_to_board_tvecs ] 
            if any(x is None for x in self.robot_poses):
                robot_tvec_string = [''] * len(self.robot_poses)
            else:
                robot_tvec_string = [np.array2string(value[0:3], 
                                        formatter={'float_kind': lambda x: f"{x:.2f}"}) 
                                        for value in self.robot_poses]
                
            return ["p_cam:" + cam + " p_arm:" + robot for cam, robot in zip(cam_tvec_string, robot_tvec_string)]

        return []

    def reset(self):
        self.images = []
        self.robot_poses = []
        self.objpoints = []
        self.imgpoints = []
        self.camera_to_board_rvecs = []
        self.camera_to_board_tvecs = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_results = {}

    def __len__(self):
        return len(self.images)

    def append(self, image: np.ndarray, robot_pose: np.ndarray = None, recalib=False):
        ret, cur_object_points, cur_image_points = self.board_dectect(image)
        if ret:
            self.images.append(image)
            self.imgpoints.append(cur_image_points)
            self.objpoints.append(cur_object_points)
            self.robot_poses.append(robot_pose)

            if recalib:
                self.calibrate_all()
        else:
            logger.warning("Failed to detect board in image, image not added.")

    def modify(self, index: int, image: np.ndarray, robot_pose: np.ndarray = None):
        ret, cur_object_points, cur_image_points = self.board_dectect(image)
        if ret:
            self.images[index] = image
            self.robot_poses[index] = robot_pose
            self.imgpoints[index] = cur_image_points
            self.objpoints[index] = cur_object_points

        else:
            logger.warning("Failed to detect board in image, image not modified.")

    def board_dectect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image_size = gray.shape[::-1]
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        if charuco_ids is not None and charuco_corners is not None:
            if len(charuco_ids) > 5:
                cur_object_points, cur_image_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
                ret = True
            else:
                cur_object_points, cur_image_points = None, None
                ret = False
                logger.warning(f"Not enough markers detected in image")
        else:
            logger.warning(f"No valid Charuco corners detected in image")
            cur_object_points, cur_image_points = None, None
            ret = False

        return ret, cur_object_points, cur_image_points

    def calibrate_camera(self):
        print(len(self.imgpoints))
        if len(self.imgpoints) >= 3:
            # Use cv2.calibrateCamera for camera calibration
            ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, self.image_size, None, None
            )
            if ret:
                logger.info("Camera calibration successful")
                # logger.info(f"Camera matrix:\n{self.camera_matrix}")
                # logger.info(f"Distortion coefficients:\n{self.dist_coeffs}")
            else:
                logger.warning("Camera calibration failed")
        else:
            logger.warning("Not enough object points and image points for calibration")

    def board_pose_calculation(self):
        if self.camera_matrix is not None and self.dist_coeffs is not None:
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
            [self.pop(idx) for idx in invalid_index]
        else:
            logger.warning("Camera matrix and distortion coefficients are not available.")

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

    def inv_vec(self, R_matrices: list[np.ndarray], T_vecs: list[np.ndarray]):
        new_rmatrices = []
        new_tvecs = []
        for r_matrix, tvec in zip(R_matrices, T_vecs):
            R_b2e = r_matrix.T
            t_b2e = -R_b2e @ tvec
            new_rmatrices.append(R_b2e)
            new_tvecs.append(t_b2e)
        return new_rmatrices, new_tvecs

    def calibrate_hand_eye(self):
        if any(x is None for x in self.robot_poses):
            logger.warning("No robot poses available for hand-eye calibration.")
            return
        
        base_to_end_rmatrices: list[np.ndarray] = []
        base_to_end_tvecs: list[np.ndarray] = []
        for robot_pose in self.robot_poses:
            # Assuming robot_pose is an array of shape (6,), [x, y, z, rx, ry, rz], rotations in radians
            base_to_end_rmatrices.append(cv2.Rodrigues(robot_pose[3:6])[0])
                # R.from_euler('xyz', robot_pose[3:6], degrees=False).as_matrix())
            base_to_end_tvecs.append(robot_pose[0:3])

        # Convert to end_to_base
        end_to_base_rmatrices, end_to_base_tvecs = self.inv_vec(base_to_end_rmatrices, base_to_end_tvecs)

        methods = {
            'Tsai': cv2.CALIB_HAND_EYE_TSAI,
            'Park': cv2.CALIB_HAND_EYE_PARK,
            'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
            'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
            'Andreff': cv2.CALIB_HAND_EYE_ANDREFF
        }

        cam_to_board_rmatrices = [cv2.Rodrigues(rvec)[0] for rvec in self.camera_to_board_rvecs]
        cam_to_board_tvecs = [tvec.flatten() for tvec in self.camera_to_board_tvecs]

        calibration_results = {}
        for method_name, method in methods.items():
            try:
                R_cam2base, t_cam2base = cv2.calibrateHandEye(
                    end_to_base_rmatrices,
                    end_to_base_tvecs,
                    cam_to_board_rmatrices,
                    cam_to_board_tvecs, method=method)
            except Exception as e:
                logger.error(e)
                continue
            if np.nan in R_cam2base:
                logger.warning(f"Hand-eye calibration failed for method {method_name}")
                continue
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R_cam2base
            transformation_matrix[:3, 3] = t_cam2base.flatten()
            calibration_results[method_name] = {
                'transformation_matrix': transformation_matrix.tolist(),
            }
            logger.info(f"{method_name} Calibration Matrix:\n{transformation_matrix}\n")
            rvecs, _ = cv2.Rodrigues(R_cam2base)
            logger.info(f"{method_name} Calibration Pose (x, y, z, rx, ry, rz):\n {t_cam2base.ravel()} {rvecs.ravel()}\n")

        # Save calibration results in self
        self.calibration_results = calibration_results

    def save_calibration_data(self, path: str):
        if self.camera_matrix is None or self.dist_coeffs is None or len(self.calibration_results) == 0:
            logger.warning("Camera matrix and distortion coefficients are not available.")
            return
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'calibration_results': self.calibration_results
        }
        path = Path(path)
        with open(path, 'w') as json_file:
            json.dump(calibration_data, json_file, indent=2)
        logger.info(f"Calibration results saved to {path}")

    def save_img_and_pose(self):
        # Save images and robot poses to self.save_dir
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(str(self.save_dir))
            images_dir = self.save_dir / 'images'
            images_dir.mkdir(parents=True, exist_ok=True)
            pose_file_path = self.save_dir / 'pose.txt'
            with open(pose_file_path, 'w') as pose_file:
                for idx, (image, robot_pose) in enumerate(zip(self.images, self.robot_poses)):
                    img_path = images_dir / f'{idx}.png'
                    cv2.imwrite(str(img_path), image)
                    # Save robot pose
                    tvecs = robot_pose[0:3]
                    rvecs = robot_pose[3:6]
                    pose_file.write(f"{tvecs[0]*1000:.3f} {tvecs[1]*1000:.3f} {tvecs[2]*1000:.3f} "
                                    f"{np.rad2deg(rvecs[0]):.3f} {np.rad2deg(rvecs[1]):.3f} {np.rad2deg(rvecs[2]):.3f}\n")
            logger.info(f"Saved images and poses to {self.save_dir}")

    def load_img_and_pose(self):
        # Load images and robot poses from self.save_dir
        if self.save_dir:
            images_dir = self.save_dir / 'images'
            pose_file_path = self.save_dir / 'pose.txt'
            if not images_dir.exists() or not pose_file_path.exists():
                logger.error(f"Images or poses not found in {self.save_dir}")
                return
            robot_poses = []
            with open(pose_file_path, 'r') as pose_file:
                lines = pose_file.readlines()
                for line in lines:
                    tvecs_str, rvecs_str = line.strip().split()
                    tvecs = np.array([float(t) / 1000 for t in tvecs_str.split()])
                    rvecs = np.array([np.deg2rad(float(r)) for r in rvecs_str.split()])
                    robot_poses.append(np.hstack((tvecs, rvecs)))
            images = sorted(images_dir.glob('*.png'))
            images = [cv2.imread(str(image)) for image in images]
            self.reset()
            for image, robot_pose in zip(images, robot_poses):
                self.append(image, robot_pose)

    def pop(self, index: int):
        self.images.pop(index)
        self.robot_poses.pop(index)
        self.objpoints.pop(index)
        self.imgpoints.pop(index)
        self.camera_to_board_rvecs.pop(index)
        self.camera_to_board_tvecs.pop(index)
        self.calibrate_all()


    def calibrate_all(self):
        if len(self.images) > 3:
            self.calibrate_camera()
            self.board_pose_calculation()
            self.compute_reprojection_error()
            self.calibrate_hand_eye()

    def load_camera_intrinsics(self, intrinsic_path):
        intrinsic_path = Path(intrinsic_path)
        with open(intrinsic_path, 'r') as json_file:
            data = json.load(json_file)
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['dist_coeffs'])
        logger.info(f"Loaded camera intrinsic parameters from {intrinsic_path}")
        logger.info(f"Camera matrix:\n{self.camera_matrix}")
        logger.info(f"Distortion coefficients:\n{self.dist_coeffs}")
