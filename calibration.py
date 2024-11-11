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


class CalibrationProcess:
    def __init__(self, params: dict, camera_interface: CameraInterface, robot_interface: RobotInterface):
        self.params = params
        self.directory = params.get('directory', os.getcwd())
        self.image_amount = params.get('ImageAmount', None)
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
        self.calibration_dir = os.path.join(self.directory, f'Calibration_results')
        os.makedirs(self.calibration_dir, exist_ok=True)

    def _save_image_and_pose(self, img, robot_pose, index, img_save_path, pose_file=None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_filename = img_save_path / f"{index}.png"
        cv2.imwrite(str(img_filename), img)
        tvecs = robot_pose[0][:3]
        rvecs = robot_pose[0][3:]
        if pose_file is not None:
            pose_file.write(f"{tvecs[0]*1000:.3f} {tvecs[1]*1000:.3f} {tvecs[2]*1000:.3f} "
                        f"{np.rad2deg(rvecs[0]):.3f} {np.rad2deg(rvecs[1]):.3f} {np.rad2deg(rvecs[2]):.3f}\n")
        logger.info(f"Captured and saved image {index + 1}")

    def capture_images(self):
        img_save_path = Path(self.directory) / '_tmp'
        img_save_path.mkdir(parents=True, exist_ok=True)
        pose_save_path = img_save_path / 'pose.txt'

        if self.input_method == 'capture':
            logger.info("Starting image capture...")
            with pose_save_path.open('w') as pose_file:
                self.camera.start_live_feedback()
                last_num_images = 0
                while len(self.camera.captured_images) < self.image_amount:
                    current_num_images = len(self.camera.captured_images)
                    if current_num_images > last_num_images:
                        idx = current_num_images - 1
                        logger.info(f"Captured image {current_num_images} of {self.image_amount}")
                        # Capture robot pose
                        rvecs, t_g2b = self.robot.capture_gripper_to_base()
                        flat_rvecs = rvecs.flatten()
                        tvecs = t_g2b.flatten()
                        robot_pose = np.hstack((tvecs.reshape(1, 3), rvecs.reshape(1, 3)))
                        self.robot_pose_list.append(robot_pose)
                        # Save image and pose
                        img = self.camera.captured_images[-1]
                        self._save_image_and_pose(img, robot_pose.reshape(1, 6), idx, img_save_path, pose_file)
                        last_num_images = current_num_images
                    time.sleep(0.1)
                self.camera.stop_live_feedback()
                # Set self.img_frames to the captured images
                self.img_frames = self.camera.captured_images
                logger.info("Image capture completed and saved to _tmp folder.")

        elif self.input_method == 'auto_calibrated_mode':
            if not self.pose_file_path:
                raise ValueError("Pose file path must be specified for 'auto_calibrated_mode'")

            img_save_path = Path(self.directory) / '_tmp'
            img_save_path.mkdir(parents=True, exist_ok=True)
            pose_save_path = img_save_path / 'pose.txt'
            # Read the pose file
            with open(self.pose_file_path, 'r') as f:
                lines = f.readlines()
            cartesian_poses_list = [line.strip().split() for line in lines]
            cartesian_poses_list = [np.array([float(x) for x in line], dtype=np.float32) for line in cartesian_poses_list]
            cartesian_poses_list = [np.hstack((line[0:3] / 1000, np.deg2rad(line[3:6]))) for line in cartesian_poses_list]

            for idx, cartesian_pose in enumerate(cartesian_poses_list):
                # Move robot to pose
                cartesian_pose_dict = self.robot.pose_array_to_dict(cartesian_pose)
                joint_pose = self.robot.lebai.kinematics_inverse(cartesian_pose_dict)
                self.robot.lebai.movej(joint_pose, self.robot.acceleration, self.robot.velocity, self.robot.time_running, self.robot.radius)
                self.robot.lebai.wait_move()
                time.sleep(0.5)
                # Capture image
                img = self.camera.capture_frame()
                if img is None:
                    logger.warning(f"Failed to capture image at position {idx}")
                    continue
                # Append image to img_frames
                self.img_frames.append(img)
                # Capture robot pose and save to pose file
                rvecs, t_g2b = self.robot.capture_gripper_to_base()
                flat_rvecs = rvecs.flatten()
                tvecs = t_g2b.flatten()
                robot_pose = np.hstack((tvecs.reshape(1, 3), rvecs.reshape(1, 3)))
                self.robot_pose_list.append(robot_pose)
                # Save image and pose
                self._save_image_and_pose(img, robot_pose.reshape(1, 6), idx, img_save_path)

            logger.info("Image capture completed and saved to _tmp folder.")

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
        self.useful_robot_pose = []
        for idx, img in enumerate(self.img_frames):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.camera.charuco_detector.detectBoard(gray)
            if charuco_ids is not None and charuco_corners is not None:
                # Use matchImagePoints to get object points and image points
                object_points, image_points = self.camera.charuco_board.matchImagePoints(charuco_corners, charuco_ids)
                # Append the object and image points for calibration
                self.objpoints.append(object_points)
                self.imgpoints.append(image_points)
                self.useful_robot_pose.append(self.robot_pose_list[idx])
                # For visualization
                img_with_corners = img.copy()
                cv2.aruco.drawDetectedMarkers(img_with_corners, marker_corners, marker_ids)
                cv2.aruco.drawDetectedCornersCharuco(img_with_corners, charuco_corners, charuco_ids)
            else:
                logger.warning(f"No valid Charuco corners detected in image {idx}")

    def calibrate_camera(self):
        logger.info("Starting camera calibration...")
        if self.camera.camera_matrix is not None and self.camera.dist_coeffs is not None:
            self.camera_matrix = self.camera.camera_matrix
            self.dist_coeffs = self.camera.dist_coeffs
            logger.info("Using camera intrinsic parameters from camera interface.")
            logger.info(f"Camera matrix:\n{self.camera_matrix}")
            logger.info(f"Distortion coefficients:\n{self.dist_coeffs}")
        else:
            if len(self.objpoints) > 0:
                ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
                    self.objpoints, self.imgpoints, self.img_frames[0].shape[1::-1], None, None
                )
                if ret:
                    logger.info("Camera calibration successful")
                    logger.info(f"Camera matrix:\n{self.camera_matrix}")
                    logger.info(f"Distortion coefficients:\n{self.dist_coeffs}")
                else:
                    logger.warning("Camera calibration failed")
            else:
                logger.warning("No valid object points and image points for calibration")

    def get_board_pose(self):
        self.camera_to_board_rvecs = []
        self.camera_to_board_tvecs = []
        for object_points, image_points in zip(self.objpoints, self.imgpoints):
            ret, rvec, tvec = cv2.solvePnP(
                object_points, image_points, self.camera_matrix, self.dist_coeffs
            )
            if ret:
                self.camera_to_board_rvecs.append(rvec)
                self.camera_to_board_tvecs.append(tvec)
            else:
                logger.warning(f"Could not solvePnP for image points {image_points}")

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

    def inv_vec(self, R_matrices: list[np.ndarray], T_vecs: list[np.ndarray]):

        new_rmatrices = []
        new_tvecs = []
        for r_matrix, tvec in zip(R_matrices, T_vecs):
            R_b2e = r_matrix.T
            t_b2e = -R_b2e @ tvec
            new_rmatrices.append(R_b2e)
            new_tvecs.append(t_b2e)
        return new_rmatrices, new_tvecs

    def hand_eye_calibration(self):
        base_to_end_rmatrices: list[np.ndarray] = []
        base_to_end_tvecs: list[np.ndarray] = []
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
        cam_to_board_rmatrices = [R.from_rotvec(rvec.flatten()).as_matrix() for rvec in self.camera_to_board_rvecs]
        cam_to_board_tvecs = [tvec.flatten() for tvec in self.camera_to_board_tvecs]

        board_to_cam_rmatrices, board_to_cam_tvecs = cam_to_board_rmatrices, cam_to_board_tvecs
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
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R_cam2base
            transformation_matrix[:3, 3] = t_cam2base.flatten()
            calibration_results[method_name] = {
                'transformation_matrix': transformation_matrix.tolist(),
            }
            logger.info(f"{method_name} Calibration Matrix:\n{transformation_matrix}\n")
            logger.info(f"{method_name} Calibration Pose (x, y, z, rx, ry, rz):\n {t_cam2base.ravel()} {cv2.Rodrigues(R_cam2base)[0].ravel()}\n")

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
            logger.info("ChArUco corners were not found in any image. Exiting calibration process.")
            return

        self.calibrate_camera()
        logger.info("Camera calibration completed.")
        self.get_board_pose()
        logger.info("Board pose calculated.")
        self.compute_reprojection_error()
        logger.info("Reprojection error computed.")
        self.hand_eye_calibration()
        logger.info("Hand-eye calibration completed.")


def main():
    params = {
        'directory': '.',  # Change to your directory if needed
        'ImageAmount': 13,
        'board_shape': (11, 6),
        'board_square_size': 0.023,
        'board_marker_size': 0.0175,
        'input_method': 'auto_calibrated_mode',  # 'capture', 'load_from_folder', or 'auto_calibrated_mode'
        'folder_path': '_tmp',  # Specify the folder path if using 'load_from_folder'
        'pose_file_path': './poses.txt',  # Specify the pose file path for 'auto_calibrated_mode'
        'load_intrinsic': True,  # Set to True or False
        'intrinsic_path': './Calibration_results/calibration_results.json'  # Path to the intrinsic JSON file
    }

    input_method = params.get('input_method', 'capture')

    # Create ChArUco dictionary and board
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    charuco_board = cv2.aruco.CharucoBoard(
        params['board_shape'],
        squareLength=params['board_square_size'],
        markerLength=params['board_marker_size'],
        dictionary=charuco_dict
    )

    if input_method in ['capture', 'auto_calibrated_mode']:
        camera_config = './default_config.json'
        sensor_config = o3d.io.read_azure_kinect_sensor_config(camera_config)
        camera = o3d.io.AzureKinectSensor(sensor_config)
        if not camera.connect(0):
            raise RuntimeError('Failed to connect to Azure Kinect sensor')
        camera_interface = CameraInterface(camera, charuco_dict, charuco_board)
        robot_interface = RobotInterface()
        robot_interface.find_device()
        robot_interface.connect()
    elif input_method == 'load_from_folder':
        camera_interface = CameraInterface(None, charuco_dict, charuco_board)  # No camera needed
        robot_interface = None  # No robot interface needed
    else:
        raise ValueError(f"Unknown input method: {input_method}")

    calibration_process = CalibrationProcess(params, camera_interface, robot_interface)
    calibration_process.run()


if __name__ == '__main__':
    main()
