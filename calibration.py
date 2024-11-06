import os
import sys
import time
import cv2
import numpy as np
from datetime import datetime
import open3d as o3d
import json
from scipy.spatial.transform import Rotation as R
from robot import RobotInterface
import logging
import numpy as np
from scipy.linalg import logm, expm

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import lstsq

def skew(vector):
    """Create a skew-symmetric matrix from a 3D vector."""
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def transl(translation):
    """Create a 4x4 transformation matrix for a translation vector."""
    T = np.eye(4)
    T[:3, 3] = translation
    return T

def handEye(bHg, cHw):
    """Python implementation of the hand-eye calibration function using scipy utilities."""
    M = len(bHg)
    K = (M * (M - 1)) // 2
    
    # Matrices A and B for the linear system
    A = np.zeros((3 * K, 3))
    B = np.zeros((3 * K, 1))
    k = 0

    for i in range(M):
        for j in range(i + 1, M):
            # Transformation from i-th to j-th gripper pose
            Hgij = np.linalg.solve(bHg[j], bHg[i])
            Pgij = 2 * R.from_matrix(Hgij[:3, :3]).as_rotvec()

            # Transformation from i-th to j-th camera pose
            Hcij = np.dot(cHw[j], np.linalg.inv(cHw[i]))
            Pcij = 2 * R.from_matrix(Hcij[:3, :3]).as_rotvec()

            # Formulate the linear equations
            A[3 * k:3 * k + 3, :] = skew(Pgij + Pcij)
            B[3 * k:3 * k + 3, 0] = Pcij - Pgij
            k += 1

    # Solve for rotation using least squares
    Pcg_ = lstsq(A, B)[0].flatten()
    Pcg = 2 * Pcg_ / np.sqrt(1 + np.dot(Pcg_.T, Pcg_))
    Rcg = R.from_rotvec(Pcg / 2).as_matrix()

    # Solve for translation
    A = np.zeros((3 * K, 3))
    B = np.zeros((3 * K, 1))
    k = 0
    for i in range(M):
        for j in range(i + 1, M):
            # Transformations between poses
            Hgij = np.linalg.solve(bHg[j], bHg[i])
            Hcij = np.dot(cHw[j], np.linalg.inv(cHw[i]))

            # Formulate the linear equations for translation
            A[3 * k:3 * k + 3, :] = Hgij[:3, :3] - np.eye(3)
            B[3 * k:3 * k + 3, 0] = Rcg @ Hcij[:3, 3] - Hgij[:3, 3]
            k += 1

    # Solve for translation using least squares
    Tcg = lstsq(A, B)[0].flatten()

    # Combine rotation and translation into the 4x4 transformation matrix gHc
    gHc = np.eye(4)
    gHc[:3, :3] = Rcg
    gHc[:3, 3] = Tcg

    return gHc


class CameraInterface:
    def __init__(self, camera, checkerboard_dims):
        self.camera = camera
        self.checkerboard_dims = tuple(checkerboard_dims)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def live_feedback(self):
        cv2.namedWindow('Live Feedback')
        while True:
            rgbd = self.camera.capture_frame(True)
            if rgbd is None:
                continue
            color = np.asarray(rgbd.color)
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                color_display = cv2.drawChessboardCorners(color.copy(), self.checkerboard_dims, corners2, ret)
            else:
                color_display = color.copy()
            cv2.imshow('Live Feedback', color_display)
            key = cv2.waitKey(1)
            if key == ord('q'):
                return None  # User chose to quit
            elif key == ord(' '):  # Press space to capture image
                logger.info("Image captured.")
                return color  # Return the captured image

    def capture_frame(self):
        rgbd = self.camera.capture_frame(True)
        if rgbd is None:
            raise RuntimeError('Failed to capture frame')
        color = np.asarray(rgbd.color)
        color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        return color

    def show_image(self, image, window_name='Image', time_ms=1500):
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, image)
        cv2.waitKey(time_ms)
        # Do not destroy the window

    def clear(self):
        pass  # No cleanup necessary for Open3D Azure Kinect sensor


class CalibrationProcess:
    def __init__(self, params, camera_interface: CameraInterface, robot_interface: RobotInterface):
        self.directory = params.get('directory', os.getcwd())
        self.image_amount = params.get('ImageAmount', None)
        self.square_size = params.get('CheckerboardSquareSize', None)
        self.checkerboard_dims = tuple(params.get('Checkerboard', None))
        self.input_method = params.get('input_method', 'capture')
        self.folder_path = params.get('folder_path', None)
        self.camera = camera_interface
        self.robot = robot_interface
        self.img_frames = []
        self.position_list = []
        self.useful_positions = []
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None

        self.calibration_dir = os.path.join(self.directory, f'Calibration_results')
        os.makedirs(self.calibration_dir, exist_ok=True)

    def capture_images(self):
        if self.input_method == 'capture':
            logger.info("Starting image capture...")
            for iteration in range(self.image_amount):
                logger.info("Press space to take picture or 'q' to quit...")
                img = None
                while img is None:
                    img = self.camera.live_feedback()
                    if img is None:
                        logger.info("User opted to quit capturing images.")
                        return  # Exit the method if user quits
                self.img_frames.append(img)
                logger.info(f"Captured image {iteration + 1} of {self.image_amount}")

                R_g2b, t_g2b = self.robot.capture_gripper_to_base()
                rvecs, _ = cv2.Rodrigues(R_g2b)
                tvecs = t_g2b.flatten()
                logger.info(f"Robot translation vector: {tvecs}")
                position = np.hstack((tvecs.reshape(1, 3), rvecs.reshape(1, 3)))
                self.position_list.append(position)
                time.sleep(0.2)
            logger.info("Image capture completed.")
            # No need to destroy windows here

            with open(os.path.join(self.calibration_dir, "robot_positions.txt"), "w") as file:
                for position in self.position_list:
                    file.write(f"{position.tolist()}\n")
            self.camera.clear()
        elif self.input_method == 'load_from_folder':
            self.load_images_from_folder()
        else:
            raise ValueError(f"Unknown input method: {self.input_method}")

    def load_images_from_folder(self):
        if not self.folder_path:
            raise ValueError("Folder path must be specified for 'load_from_folder' input method.")

        logger.info(f"Loading images from folder: {self.folder_path}")
        pose_file_path = os.path.join(self.folder_path, 'pose.txt')
        if not os.path.exists(pose_file_path):
            raise FileNotFoundError(f"Pose file not found at: {pose_file_path}")

        # Read poses from pose.txt
        with open(pose_file_path, 'r') as pose_file:
            lines = pose_file.readlines()

        # Now load images named idx.png, where idx corresponds to the pose index
        for idx, line in enumerate(lines):
            image_path = os.path.join(self.folder_path, f'{idx}.png')
            if not os.path.exists(image_path):
                logger.warning(f"Image not found at path: {image_path}")
                continue
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Failed to load image at path: {image_path}")
                continue
            self.img_frames.append(img)
            logger.info(f"Loaded image {idx}.png")

            # Process pose line
            parts = line.strip().split()
            if len(parts) != 6:
                logger.warning(f"Invalid pose format in line: {line}")
                continue
            x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = map(float, parts)
            x_m = x_mm / 1000.0  # Convert cm to meters
            y_m = y_mm / 1000.0
            z_m = z_mm / 1000.0
            rx_rad = np.deg2rad(rx_deg)  # Convert degrees to radians
            ry_rad = np.deg2rad(ry_deg)
            rz_rad = np.deg2rad(rz_deg)
            position = np.array([x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]).reshape(1, 6)
            self.position_list.append(position)
            logger.debug(f"Loaded pose: {position}")

        # Check that the number of images matches the number of poses
        if len(self.img_frames) != len(self.position_list):
            raise ValueError("Number of images does not match number of poses")

        logger.info("Finished loading images and poses from folder")

    def find_corners(self):
        objp = np.zeros((self.checkerboard_dims[1] * self.checkerboard_dims[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_dims[0], 0:self.checkerboard_dims[1]].T.reshape(-1, 2) * self.square_size

        for idx, img in enumerate(self.img_frames):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, None)
            if ret:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.camera.criteria)
                self.imgpoints.append(corners2)
                img_with_corners = cv2.drawChessboardCorners(img.copy(), self.checkerboard_dims, corners2, ret)
                window_name = f'Corners Image'
                self.camera.show_image(img_with_corners, window_name=window_name, time_ms=150)
                self.useful_positions.append(self.position_list[idx])

    def calibrate_camera(self):
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.img_frames[0].shape[1::-1], None, None)
        
        logger.info("Camera calibration results:")
        logger.info(f"Camera matrix:\n{self.camera_matrix}")
        logger.info(f"Distortion coefficients:\n{self.dist_coeffs}")

    # def compute_collected_poses(self):
    #     self.rvecs = []
    #     self.tvecs = []
    #     for 

    def compute_reprojection_error(self):
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.rvecs[i], self.tvecs[i], self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(self.objpoints)
        logger.info(f"Total reprojection error: {mean_error}")
        self.camera.clear()

    def compute_camera_pose(self, R_cam2gripper, t_cam2gripper):
        # Compute camera pose in the robot's coordinate system
        camera_pose = np.zeros(6)
        
        # Compute position
        camera_position = -R_cam2gripper.T @ t_cam2gripper
        camera_pose[:3] = camera_position.flatten()
        
        # Compute rotation (roll, pitch, yaw)
        r = R.from_matrix(R_cam2gripper.T)
        camera_pose[3:6] = r.as_euler('xyz')
        
        return camera_pose
    
    def calibrate_hand_eye(self, robot_rvecs, robot_tvecs, cam_rvecs, cam_tvecs):
        """
        Python implementation of hand-eye calibration.
        
        Parameters:
        - robot_rvecs: List of rotation vectors for each robot pose
        - robot_tvecs: List of translation vectors for each robot pose
        - cam_rvecs: List of rotation vectors for each camera pose
        - cam_tvecs: List of translation vectors for each camera pose
        
        Returns:
        - r_vec, t_vec: The computed hand-eye calibration matrix
        """
        assert len(robot_rvecs) == len(cam_rvecs)
        robot_pose = []
        chessboard_pose = []
        
        # Reading robot poses and converting to transformation matrices
        for rvec, tvec in zip(robot_rvecs, robot_tvecs):
            rot_temp = R.from_rotvec(rvec.ravel()).as_matrix()
            pose_temp = np.eye(4)
            pose_temp[:3, :3] = rot_temp
            pose_temp[:3, 3] = tvec.ravel()
            pose_temp = np.linalg.inv(pose_temp)
            robot_pose.append(pose_temp)

        for rvec, tvec in zip(cam_rvecs, cam_tvecs):
            rot_temp = R.from_rotvec(rvec.ravel()).as_matrix()
            pose_temp = np.eye(4)
            pose_temp[:3, :3] = rot_temp
            pose_temp[:3, 3] = tvec.ravel()
            
            chessboard_pose.append(pose_temp)

        gHc = handEye(robot_pose, chessboard_pose)

        cam_rvecs = gHc[:3, :3]
        cam_tvecs = gHc[:3, 3]
        return cam_rvecs, cam_tvecs


    def inv_vec(self, R_vecs, T_vecs):
        new_rvecs = []
        new_tvecs = []
        for rvec, tvec in zip(R_vecs, T_vecs):
            rot_temp = R.from_rotvec(rvec.ravel()).as_matrix()
            pose_temp = np.eye(4)
            pose_temp[:3, :3] = rot_temp
            pose_temp[:3, 3] = tvec.ravel()
            pose_temp = np.linalg.inv(pose_temp)

            new_rvecs.append(cv2.Rodrigues(pose_temp[:3, :3])[0])
            new_tvecs.append(pose_temp[:3, 3].ravel())
            
        return new_rvecs, new_tvecs

    def hand_eye_calibration(self):
        robot_rvecs = []
        robot_tvecs = []
        for position in self.useful_positions:
            robot_rvecs.append(position[0][3:6])
            robot_tvecs.append(position[0][0:3])

        methods = {
            'Tsai': cv2.CALIB_HAND_EYE_TSAI,
            'Park': cv2.CALIB_HAND_EYE_PARK,
            'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
            'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
            'Andreff': cv2.CALIB_HAND_EYE_ANDREFF
        }

        calibration_results = {}
        for method_name, method in methods.items():
            # robot_rvecs, robot_tvecs = self.inv_vec(robot_rvecs, robot_tvecs)

            robot_R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in robot_rvecs]
            cam_R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in self.rvecs]
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                cam_R_matrices, self.tvecs, robot_R_matrices, robot_tvecs, method=method)

            # R_cam2gripper, t_cam2gripper = self.calibrate_hand_eye(
            #     robot_rvecs, robot_tvecs, self.rvecs, self.tvecs)
            # Calculate the camera pose in the robot's coordinate system
            camera_pose = self.compute_camera_pose(R_cam2base, t_cam2base)
            
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R_cam2base
            transformation_matrix[:3, 3] = t_cam2base.flatten()
            calibration_results[method_name] = {
                'transformation_matrix': transformation_matrix.tolist(),
                'calibrated_camera_pose': camera_pose.tolist()
            }
            logger.info(f"{method_name} Calibration Matrix:\n{transformation_matrix}\n")
            logger.info(f"{method_name} Camera Pose: {camera_pose}\n")
            # break

        # Save the intrinsic parameters and poses to JSON
        calibration_data = {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'calibration_results': calibration_results
        }
        json_path = os.path.join(self.calibration_dir, 'calibration_results.json')
        with open(json_path, 'w') as json_file:
            json.dump(calibration_data, json_file, indent=4)
        logger.info(f"Calibration results saved to {json_path}")

    def run(self):
        self.capture_images()
        if not self.img_frames:
            logger.info("No images were captured or loaded. Exiting calibration process.")
            return
        self.find_corners()
        if not self.objpoints or not self.imgpoints:
            logger.info("Chessboard corners were not found in any image. Exiting calibration process.")
            return
        self.calibrate_camera()
        self.compute_reprojection_error()
        self.hand_eye_calibration()
        cv2.destroyAllWindows()  # Destroy all windows at the end of the process

def main():
    params = {
        'directory': '.',  # Change to your directory if needed
        'ImageAmount': 20,
        'CheckerboardSquareSize': 0.02,
        'Checkerboard': [10, 7],
        'input_method': 'capture',  # 'capture' or 'load_from_folder'
        'folder_path': '/home/capre/data/calibration_test'  # Specify the folder path if using 'load_from_folder'
    }

    input_method = params.get('input_method', 'capture')

    if input_method == 'capture':
        camera_config = './default_config.json'
        sensor_config = o3d.io.read_azure_kinect_sensor_config(camera_config)
        camera = o3d.io.AzureKinectSensor(sensor_config)
        if not camera.connect(0):
            raise RuntimeError('Failed to connect to Azure Kinect sensor')
        camera_interface = CameraInterface(camera, params['Checkerboard'])
        robot_interface = RobotInterface()
        robot_interface.find_device()
        robot_interface.connect()
    elif input_method == 'load_from_folder':
        camera_interface = CameraInterface(None, params['Checkerboard'])  # No camera needed
        robot_interface = None  # No robot interface needed
    else:
        raise ValueError(f"Unknown input method: {input_method}")

    calibration_process = CalibrationProcess(params, camera_interface, robot_interface)
    calibration_process.run()

if __name__ == '__main__':
    main()
