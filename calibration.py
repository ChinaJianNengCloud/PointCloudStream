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

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class CameraInterface:
    def __init__(self, camera):
        self.camera = camera

    def live_feedback(self):
        while True:
            rgbd = self.camera.capture_frame(True)
            if rgbd is None:
                continue
            color = np.asarray(rgbd.color)
            color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
            cv2.imshow('Live Feedback', color)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None  # User chose to quit
            elif key == ord(' '):  # Press space to capture image
                cv2.destroyAllWindows()
                return color  # Return the captured image

    def capture_frame(self):
        rgbd = self.camera.capture_frame(True)
        if rgbd is None:
            raise RuntimeError('Failed to capture frame')
        color = np.asarray(rgbd.color)
        color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        return color

    def show_image(self, image, time_ms=1500):
        cv2.imshow('Image', image)
        cv2.waitKey(time_ms)
        cv2.destroyAllWindows()

    def clear(self):
        pass  # No cleanup necessary for Open3D Azure Kinect sensor


class CalibrationProcess:
    def __init__(self, params, camera_interface: CameraInterface, robot_interface: RobotInterface):
        self.directory = params.get('directory', os.getcwd())
        self.image_amount = params.get('ImageAmount', 25)
        self.square_size = params.get('CheckerboardSquareSize', 0.025)
        self.checkerboard_dims = params.get('Checkerboard', (9, 6))
        self.camera = camera_interface
        self.robot = robot_interface
        self.img_frames = []
        self.position_list = []
        self.useful_positions = []
        self.objpoints = []
        self.imgpoints = []
        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_mtx = None
        self.roi = None
        self.rvecs = None
        self.tvecs = None

        timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        self.picture_folder = os.path.join(self.directory, f'Calibration_{timestamp}')
        os.makedirs(self.picture_folder, exist_ok=True)

    def capture_images(self):
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
        cv2.destroyAllWindows()

        with open(os.path.join(self.picture_folder, "robot_positions.txt"), "w") as file:
            for position in self.position_list:
                file.write(f"{position.tolist()}\n")
        self.camera.clear()

    def find_corners(self):
        objp = np.zeros((self.checkerboard_dims[1] * self.checkerboard_dims[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_dims[0], 0:self.checkerboard_dims[1]].T.reshape(-1, 2) * self.square_size

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for idx, img in enumerate(self.img_frames):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, None)
            if ret:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners2)
                img_with_corners = cv2.drawChessboardCorners(img, self.checkerboard_dims, corners2, ret)
                self.camera.show_image(img_with_corners)
                self.useful_positions.append(self.position_list[idx])

    def calibrate_camera(self):
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.img_frames[0].shape[1::-1], None, None)
        logger.info("Camera calibration results:")
        logger.info(f"Camera matrix:\n{self.camera_matrix}")
        logger.info(f"Distortion coefficients:\n{self.dist_coeffs}")

    def undistort_images(self):
        h, w = self.img_frames[0].shape[:2]
        self.new_camera_mtx, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        for idx, img in enumerate(self.img_frames):
            undistorted_img = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.new_camera_mtx)
            x, y, w, h = self.roi
            undistorted_img = undistorted_img[y:y + h, x:x + w]
            self.camera.show_image(undistorted_img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def compute_reprojection_error(self):
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(self.objpoints)
        logger.info(f"Total reprojection error: {mean_error}")
        self.camera.clear()

    def hand_eye_calibration(self):
        robot_rvecs = []
        robot_tvecs = []
        for position in self.useful_positions:
            robot_rvecs.append(position[0][3:6])
            robot_tvecs.append(position[0][0:3])

        methods = {
            1: cv2.CALIB_HAND_EYE_PARK,
            2: cv2.CALIB_HAND_EYE_TSAI,
            3: cv2.CALIB_HAND_EYE_HORAUD,
            4: cv2.CALIB_HAND_EYE_DANIILIDIS
        }
        results = {}
        for idx, (key, method) in enumerate(methods.items(), 1):
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(robot_rvecs, robot_tvecs, self.rvecs, self.tvecs, method=method)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R_cam2gripper
            transformation_matrix[:3, 3] = t_cam2gripper.flatten()
            results[key] = transformation_matrix
            method_name = ["Park-Martin", "Tsai", "Horaud", "Daniilidis"][idx - 1]
            logger.info(f"{method_name} Calibration Matrix ({idx}):\n{transformation_matrix}\n")

        calibration_choice = int(input("Choose Calibration method (1-4): "))
        chosen_matrix = results.get(calibration_choice, results[1])
        logger.info(f"Selected Calibration Matrix:\n{chosen_matrix}\n")
        logger.info(f"Camera Intrinsic Matrix:\n{self.camera_matrix}\n")
        logger.info(f"Optimal Camera Intrinsic Matrix:\n{self.new_camera_mtx}\n")

    def run(self):
        self.capture_images()
        if not self.img_frames:
            logger.info("No images were captured. Exiting calibration process.")
            return
        self.find_corners()
        if not self.objpoints or not self.imgpoints:
            logger.info("Chessboard corners were not found in any image. Exiting calibration process.")
            return
        self.calibrate_camera()
        self.undistort_images()
        self.compute_reprojection_error()
        self.hand_eye_calibration()

def main():
    params = {
        'directory': '.',  # Change to your directory if needed
        'ImageAmount': 20,
        'CheckerboardSquareSize': 0.02,
        'Checkerboard': [10, 7],
    }

    camera_config = './default_config.json'
    sensor_config = o3d.io.read_azure_kinect_sensor_config(camera_config)
    camera = o3d.io.AzureKinectSensor(sensor_config)
    if not camera.connect(0):
        raise RuntimeError('Failed to connect to Azure Kinect sensor')

    camera_interface = CameraInterface(camera)
    robot_interface = RobotInterface()
    robot_interface.find_device()
    robot_interface.connect()
    calibration_process = CalibrationProcess(params, camera_interface, robot_interface)
    calibration_process.run()

if __name__ == '__main__':
    main()
