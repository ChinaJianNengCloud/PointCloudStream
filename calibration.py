import lebai_sdk
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import logging
import threading
import time
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import open3d as o3d
from robot import RobotInterface

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import lstsq

def calibrate_camera_optimize_based_on_reprojection_error(object_points,
                                                          image_points,
                                                          camera_matrix,
                                                          distortion_coefficients,
                                                          rvecs,tvecs,
                                                          max_error=3.0):

    # Calculate reprojection errors
    mean_error = 0
    for i in range(len(object_points)):
        image_points2, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, distortion_coefficients)
        error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2) / len(image_points2)
        mean_error += error
    mean_error /= len(object_points)

    # Discard points with high reprojection errors
    object_points_refined = []
    image_points_refined = []
    tot_error = 0
    for i in range(len(object_points)):
        image_points2, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, distortion_coefficients)
        error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2) / len(image_points2)
        tot_error += error
        if error < max_error:
            print(f'error:{error} accepted since it is less than {max_error}')
            object_points_refined.append(object_points[i])
            image_points_refined.append(image_points[i])
    print("total error: ", tot_error / len(object_points))
    return object_points_refined,image_points_refined



class CameraInterface:
    def __init__(self, camera, checkerboard_dims):
        self.camera = camera
        self.checkerboard_dims = tuple(checkerboard_dims)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.live_feedback_thread = None
        self.live_feedback_running = False

    def start_live_feedback(self):
        self.live_feedback_running = True
        self.live_feedback_thread = threading.Thread(target=self._live_feedback_loop)
        self.live_feedback_thread.start()

    def stop_live_feedback(self):
        self.live_feedback_running = False
        if self.live_feedback_thread is not None:
            self.live_feedback_thread.join()
            # self.live_feedback_thread.
            self.live_feedback_thread = None
        cv2.destroyAllWindows()

    def _live_feedback_loop(self):
        cv2.namedWindow('Live Feedback')
        while self.live_feedback_running:
            rgbd = self.camera.capture_frame(True)
            if rgbd is None:
                continue
            color = np.asarray(rgbd.color)
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, None, 
                                                     flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                color_display = cv2.drawChessboardCorners(color.copy(), self.checkerboard_dims, corners2, ret)
            else:
                color_display = color.copy()
            cv2.imshow('Live Feedback', color_display)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.live_feedback_running = False
                break
        cv2.destroyWindow('Live Feedback')

    def live_feedback(self):
        cv2.namedWindow('Live Feedback')
        while True:
            rgbd = self.camera.capture_frame(True)
            if rgbd is None:
                continue
            color = np.asarray(rgbd.color)
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, None, 
                                                     flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
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

    def show_image(self, image, window_name='Image', time_ms=150):
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, image)
        cv2.waitKey(time_ms)
        # Do not destroy the window

    def clear(self):
        pass  # No cleanup necessary for Open3D Azure Kinect sensor


class CalibrationProcess:
    def __init__(self, params:dict, camera_interface: CameraInterface, robot_interface: RobotInterface):
        self.params = params
        self.directory = params.get('directory', os.getcwd())
        self.image_amount = params.get('ImageAmount', None)
        self.square_size = params.get('CheckerboardSquareSize', None)
        self.checkerboard_dims = tuple(params.get('Checkerboard', None))
        self.input_method = params.get('input_method', 'capture')
        self.folder_path = params.get('folder_path', None)
        self.pose_file_path = params.get('pose_file_path', None)
        self.load_intrinsic = params.get('load_intrinsic', False)
        self.intrinsic_path = params.get('intrinsic_path', None)
        self.camera = camera_interface
        self.robot = robot_interface
        self.img_frames = []
        self.robot_pose_list = []
        self.useful_robot_pose = []
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_to_board_rvecs = None
        self.robot_exec = ThreadPoolExecutor(max_workers=3,
                                             thread_name_prefix='robot_exec')
        self.camera_to_board_tvecs = None
        self.chessboard_flags = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        self.calibration_dir = os.path.join(self.directory, f'Calibration_results')
        os.makedirs(self.calibration_dir, exist_ok=True)

    def capture_images(self):
        if self.input_method == 'capture':
            tmp_dir = Path(self.directory) / '_tmp'
    
            # Clear tmp directory before starting
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            
            pose_file_path = tmp_dir / "pose.txt"
            
            logger.info("Starting image capture...")
            with pose_file_path.open("w") as pose_file:
                for iteration in range(self.image_amount):
                    logger.info("Press space to take picture or 'q' to quit...")
                    img = None
                    while img is None:
                        img = self.camera.live_feedback()
                        if img is None:
                            logger.info("User opted to quit capturing images.")
                            return  # Exit the method if user quits
                    
                    # Save each captured frame to tmp folder
                    img_filename = tmp_dir / f"{iteration}.png"
                    cv2.imwrite(str(img_filename), img)
                    logger.info(f"Captured and saved image {iteration + 1} of {self.image_amount}")
                    self.img_frames.append(img)
                    # Capture robot pose and save to pose file
                    rvecs, t_g2b = self.robot.capture_gripper_to_base()
                    flat_rvecs = rvecs.flatten()
                    tvecs = t_g2b.flatten()
                    robot_pose = np.hstack((tvecs.reshape(1, 3), rvecs.reshape(1, 3)))
                    pose_file.write(f"{tvecs[0]*1000:.3f} {tvecs[1]*1000:.3f} {tvecs[2]*1000:.3f} " # m to mm
                                    f"{np.rad2deg(flat_rvecs[0]):.3f} {np.rad2deg(flat_rvecs[1]):.3f} {np.rad2deg(flat_rvecs[2]):.3f}\n")
                    self.robot_pose_list.append(robot_pose)
                    time.sleep(0.2)
            
            logger.info("Image capture completed and saved to tmp folder.")
            self.camera.clear()
        elif self.input_method == 'load_from_folder':
            self.load_images_from_folder()
        elif self.input_method == 'auto_calibrated_mode':
            pose_file_path = self.pose_file_path
            if not pose_file_path:
                raise ValueError("Pose file path must be specified for 'auto_calibrated_mode'")
            tmp_dir = Path(self.directory) / '_tmp'

            # Clear tmp directory before starting
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)

            img_save_path = tmp_dir
            pose_save_path = tmp_dir / 'pose.txt'
            # Start live feedback
            self.camera.start_live_feedback()

            # Read the pose file
            with open(pose_file_path, 'r') as f:
                lines = f.readlines()
            cartesian_poses_list = [line.strip().split() for line in lines]
            cartesian_poses_list = [np.array([float(x) for x in line], dtype=np.float32) for line in cartesian_poses_list]
            cartesian_poses_list =[np.hstack((line[0:3] / 1000, np.deg2rad(line[3:6]))) for line in cartesian_poses_list]

            # Loop over each pose
            with pose_save_path.open('w') as pose_file:
                for idx, cartesian_pose in enumerate(cartesian_poses_list):
                    # Move robot to pose
                    cartesian_pose_dict = self.robot.pose_array_to_dict(cartesian_pose)
                    joint_pose = self.robot.lebai.kinematics_inverse(cartesian_pose_dict)
                    self.robot.lebai.movej(joint_pose, self.robot.acceleration, self.robot.velocity, self.robot.time_running, self.robot.radius)
                    self.robot.lebai.wait_move()

                    # Wait briefly
                    time.sleep(0.5)

                    # Capture image
                    img = self.camera.capture_frame()
                    if img is None:
                        logger.warning(f"Failed to capture image at position {idx}")
                        continue

                    # Save image
                    img_filename = img_save_path / f"{idx}.png"
                    cv2.imwrite(str(img_filename), img)
                    logger.info(f"Captured and saved image {idx + 1}")

                    # Append image to img_frames
                    self.img_frames.append(img)

                    # Capture robot pose and save to pose file
                    rvecs, t_g2b = self.robot.capture_gripper_to_base()
                    flat_rvecs = rvecs.flatten()
                    tvecs = t_g2b.flatten()
                    robot_pose = np.hstack((tvecs.reshape(1, 3), rvecs.reshape(1, 3)))
                    pose_file.write(f"{tvecs[0]*1000:.3f} {tvecs[1]*1000:.3f} {tvecs[2]*1000:.3f} " # m to mm
                                    f"{np.rad2deg(flat_rvecs[0]):.3f} {np.rad2deg(flat_rvecs[1]):.3f} {np.rad2deg(flat_rvecs[2]):.3f}\n")
                    # Append robot pose to robot_pose_list
                    self.robot_pose_list.append(robot_pose)

            logger.info("Image capture completed and saved to tmp folder.")
            # Stop live feedback
            self.camera.stop_live_feedback()
            # logger.info("Live feedback stopped.")
            # self.camera.clear()
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
            x_m = x_mm / 1000.0  # Convert mm to meters
            y_m = y_mm / 1000.0
            z_m = z_mm / 1000.0
            rx_rad = np.deg2rad(rx_deg)  # Convert degrees to radians
            ry_rad = np.deg2rad(ry_deg)
            rz_rad = np.deg2rad(rz_deg)
            position = np.array([x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]).reshape(1, 6)
            self.robot_pose_list.append(position)
            logger.debug(f"Loaded pose: {position}")

        # Check that the number of images matches the number of poses
        if len(self.img_frames) != len(self.robot_pose_list):
            raise ValueError("Number of images does not match number of poses")

        logger.info("Finished loading images and poses from folder")

    def find_corners(self):
        logger.info("Finding corners...")
        objp = np.zeros((self.checkerboard_dims[1] * self.checkerboard_dims[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_dims[0], 0:self.checkerboard_dims[1]].T.reshape(-1, 2) * self.square_size

        for idx, img in enumerate(self.img_frames):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, None, 
                                                     flags=self.chessboard_flags)
            # logger.debug(f"Found corners: {ret}")
            if ret:
                logger.info(f"Found corners in image {idx}")
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.camera.criteria)
                # logger.info(f"Found sub corners in image {idx}")
                self.imgpoints.append(corners2)
                img_with_corners = cv2.drawChessboardCorners(img.copy(), self.checkerboard_dims, corners2, ret)
                window_name = f'Corners Image'
                # self.camera.show_image(img_with_corners, window_name=window_name, time_ms=150)
                self.useful_robot_pose.append(self.robot_pose_list[idx])
            

    def calibrate_camera(self):
        [ret, 
         self.camera_matrix, 
         self.dist_coeffs, 
         self.camera_to_board_rvecs, 
         self.camera_to_board_tvecs] = cv2.calibrateCamera(
            self.objpoints, 
            self.imgpoints, 
            self.img_frames[0].shape[1::-1], 
            None, None,
            # flags=cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3
            )
        
        logger.info("Camera calibration results:")
        logger.info(f"Camera matrix:\n{self.camera_matrix}")
        logger.info(f"Distortion coefficients:\n{self.dist_coeffs}")

        # corrected_objpoints, corrected_imgpoints = calibrate_camera_optimize_based_on_reprojection_error(
        #     self.objpoints,
        #     self.imgpoints,
        #     self.camera_matrix,
        #     self.dist_coeffs,
        #     self.camera_to_board_rvecs,
        #     self.camera_to_board_tvecs
        # )

        [ret, 
         self.camera_matrix, 
         self.dist_coeffs, 
         self.camera_to_board_rvecs, 
         self.camera_to_board_tvecs] = cv2.calibrateCamera(
            self.objpoints, 
            self.imgpoints, 
            self.img_frames[0].shape[1::-1], 
            self.camera_matrix, 
            self.dist_coeffs,
            self.camera_to_board_rvecs,
            self.camera_to_board_tvecs,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL 
            )
        logger.info("Optimized camera calibration results:")
        logger.info(f"Camera matrix:\n{self.camera_matrix}")
        # logger.info(f"Distortion coefficients:\n{self.dist_coeffs}")
        # self.camera_matrix, roi=cv2.getOptimalNewCameraMatrix(self.camera_matrix, 
        #                                           self.dist_coeffs, 
        #                                           self.img_frames[0].shape[1::-1], 0, self.img_frames[0].shape[1::-1])
        
        # logger.info(f"Optimized camera matrix:\n{self.camera_matrix}")

    def get_board_pose(self):
        objp = np.zeros((self.checkerboard_dims[1] * self.checkerboard_dims[0], 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_dims[0], 0:self.checkerboard_dims[1]].T.reshape(-1, 2) * self.square_size
        self.useful_robot_pose:list[np.ndarray] = []
        self.objpoints = []
        self.imgpoints = []
        self.camera_to_board_rvecs:list[np.ndarray] = []
        self.camera_to_board_tvecs:list[np.ndarray] = []
        for idx, img in enumerate(self.img_frames):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, None, 
                                                     flags=self.chessboard_flags)
            if ret:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.camera.criteria)
                self.imgpoints.append(corners2)
                img_with_corners = cv2.drawChessboardCorners(img.copy(), self.checkerboard_dims, corners2, ret)
                window_name = f'get board pose'
                # self.camera.show_image(img_with_corners, window_name=window_name, time_ms=50)
                self.useful_robot_pose.append(self.robot_pose_list[idx])

                ret, camera_to_board_rvecs, camera_to_board_tvecs = cv2.solvePnP(
                    objp, corners2, self.camera_matrix, self.dist_coeffs)
                self.camera_to_board_rvecs.append(camera_to_board_rvecs)
                self.camera_to_board_tvecs.append(camera_to_board_tvecs)

    def load_camera_intrinsics(self):
        if not self.intrinsic_path:
            raise ValueError("Intrinsic path must be specified when load_intrinsic is True")
        with open(self.intrinsic_path, 'r') as json_file:
            data = json.load(json_file)
        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['dist_coeffs'])
        logger.info(f"Loaded camera intrinsic parameters from {self.intrinsic_path}")
        logger.info(f"Camera matrix:\n{self.camera_matrix}")
        logger.info(f"Distortion coefficients:\n{self.dist_coeffs}")

    def compute_reprojection_error(self):
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self.objpoints[i], self.camera_to_board_rvecs[i], self.camera_to_board_tvecs[i], self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(self.objpoints)
        logger.info(f"Total reprojection error: {mean_error}")
        self.camera.clear()

    def inv_vec(self, R_matrices: list[np.ndarray], T_vecs:list[np.ndarray]):

        new_rmatrices = []
        new_tvecs = []
        for r_matrix, tvec in zip(R_matrices, T_vecs):
            R_b2e = r_matrix.T
            t_b2e = -R_b2e @ tvec
            new_rmatrices.append(R_b2e)
            new_tvecs.append(t_b2e)
        return new_rmatrices, new_tvecs

    def hand_eye_calibration(self):
        base_to_end_rmatrices:list[np.ndarray] = []
        base_to_end_tvecs:list[np.ndarray] = []
        for robot_pose in self.useful_robot_pose:
            base_to_end_rmatrices.append(
                R.from_euler('xyz', robot_pose[0][3:6], degrees=False).as_matrix().reshape(3, 3))
            base_to_end_tvecs.append(robot_pose[0][0:3])

        end_to_base_rmatrices, end_to_base_tvecs = self.inv_vec(base_to_end_rmatrices, base_to_end_tvecs)

        methods = {
            'Tsai': cv2.CALIB_HAND_EYE_TSAI,
            'Park': cv2.CALIB_HAND_EYE_PARK,
            'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
            'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
            'Andreff': cv2.CALIB_HAND_EYE_ANDREFF
        }
        cam_to_board_rmatrices = [R.from_euler('xyz', rpose.reshape(1, 3), degrees=False).as_matrix().reshape(3, 3)
                                  for rpose in self.camera_to_board_rvecs]
        cam_to_board_tvecs = self.camera_to_board_tvecs

        board_to_cam_rmatrices, board_to_cam_tvecs = cam_to_board_rmatrices, cam_to_board_tvecs
        calibration_results = {}
        for method_name, method in methods.items():
            # rvecs, tvecs = self.inv_vec(self.rvecs, self.tvecs)
            

            # robot_R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in robot_rvecs]
            # cam_R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in self.rvecs]
            # R_cam2base, t_cam2base = cv2.calibrateHandEye(
            #     cam_R_matrices, self.tvecs, robot_R_matrices, robot_tvecs, method=method)
            try:
                R_cam2base, t_cam2base = cv2.calibrateHandEye(
                    end_to_base_rmatrices, 
                    end_to_base_tvecs, 
                    cam_to_board_rmatrices, 
                    cam_to_board_tvecs, method=method)

            except Exception as e:
                logger.error(e)
                continue
            # R_cam2base, t_cam2base = self.calibrate_hand_eye(
            #     robot_rvecs, robot_tvecs, self.rvecs, self.tvecs)
            # Calculate the camera pose in the robot's coordinate system
            # camera_pose = self.compute_camera_pose(R_cam2base, t_cam2base)
            
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R_cam2base
            transformation_matrix[:3, 3] = t_cam2base.flatten()
            calibration_results[method_name] = {
                'transformation_matrix': transformation_matrix.tolist(),
                # 'calibrated_camera_pose': camera_pose.tolist()
            }
            # logger.info(f"base pose: {t_cam2base.ravel()}, {cv2.Rodrigues(R_cam2base)[0].ravel()}")
            logger.info(f"{method_name} Calibration Matrix:\n{transformation_matrix}\n")
            logger.info(f"{method_name} Calibration Pose (x, y, z, rx, ry, rz):\n \x1b[33;20m{t_cam2base.ravel()} {cv2.Rodrigues(R_cam2base)[0].ravel()}\x1b[0m\n")
            # logger.info(f"{method_name} Camera Pose: {camera_pose}\n")
            

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

        if self.load_intrinsic:
            logger.info("Loading camera intrinsic parameters from file...")
            self.load_camera_intrinsics()
        else:
            logger.info("Camera intrinsic parameters not loaded. Calibrating camera...")
            self.calibrate_camera()

        logger.info("Camera calibration completed.")
        self.get_board_pose()
        logger.info("Board pose calculated.")
        self.compute_reprojection_error()
        logger.info("Reprojection error computed.")
        self.hand_eye_calibration()
        logger.info("Hand-eye calibration completed.")
        # cv2.destroyAllWindows()  # Destroy all windows at the end of the process

def main():
    params = {
        'directory': '.',  # Change to your directory if needed
        'ImageAmount': 13,
        'CheckerboardSquareSize': 0.015,
        'Checkerboard': [11, 8],
        'input_method': 'auto_calibrated_mode',  # 'capture', 'load_from_folder', or 'auto_calibrated_mode'
        # 'folder_path': '/path/to/folder',  # Specify the folder path if using 'load_from_folder'
        'pose_file_path': './poses.txt',  # Specify the pose file path for 'auto_calibrated_mode'
        'load_intrinsic': True,  # Set to True or False
        'intrinsic_path': './Calibration_results/calibration_results.json'  # Path to the intrinsic JSON file
    }

    input_method = params.get('input_method', 'capture')

    if input_method in ['capture', 'auto_calibrated_mode']:
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