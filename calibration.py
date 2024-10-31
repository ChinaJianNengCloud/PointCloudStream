import numpy as np
import cv2
import open3d as o3d
import time
from robot import RoboticArm
import logging as log

# Configure logging
log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EyeHandCalibration:
    def __init__(self, robotic_arm, camera, num_samples=10, eye_to_hand=True, intrinsic=None, dist_coeffs=None):
        self.robotic_arm: RoboticArm = robotic_arm
        self.camera = camera
        self.num_samples = num_samples
        self.intrinsic = intrinsic if intrinsic is not None else np.eye(3)
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1))
        self.eye_to_hand = eye_to_hand
        self.chessboard_size = (11, 8)  # Adjust as needed (number of corners per row and column)
        self.square_size = 0.01  # Chessboard square size in meters, adjust as needed
        self.previous_rvecs = []  # For checking pose differences during hand-eye calibration

    def is_blurry(self, image, threshold=100):
        """
        Check if the image is blurry using the variance of the Laplacian.

        Args:
            image: The image to check.
            threshold: The variance threshold below which the image is considered blurry.

        Returns:
            True if the image is blurry, False otherwise.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    def rotationMatrixToEulerAngles(self, R):
        """
        Convert rotation matrix to Euler angles.

        Args:
            R: Rotation matrix.

        Returns:
            Euler angles (theta_x, theta_y, theta_z).
        """
        sy = np.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1] , R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def is_pose_different(self, current_rvec, previous_rvecs, angle_threshold):
        """
        Check if the current pose is sufficiently different from previous poses.

        Args:
            current_rvec: Rotation vector of the current pose.
            previous_rvecs: List of previous rotation vectors.
            angle_threshold: Minimum angular difference in degrees.

        Returns:
            True if the pose is different enough, False otherwise.
        """
        current_rot = cv2.Rodrigues(current_rvec)[0]
        current_euler = self.rotationMatrixToEulerAngles(current_rot)

        for prev_rvec in previous_rvecs:
            prev_rot = cv2.Rodrigues(prev_rvec)[0]
            prev_euler = self.rotationMatrixToEulerAngles(prev_rot)

            angle_diff = np.abs(current_euler - prev_euler)
            angle_diff = np.degrees(angle_diff)  # Convert radians to degrees

            if np.all(angle_diff < angle_threshold):
                return False  # Pose is too similar

        return True  # Pose is sufficiently different

    def get_valid_corner_images(self):
        rgbd = None
        while not rgbd:
            rgbd = self.camera.capture_frame(True)
        image = np.asarray(rgbd.color)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if not ret:
            cv2.imshow('Calibration Image', image_rgb)
            cv2.waitKey(1)
            log.warning("Chessboard not detected. Please adjust the chessboard and try again.")
            return False, image_rgb, gray, None

        # Refine the corners
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # Draw and display the corners

        cv2.drawChessboardCorners(image_rgb, self.chessboard_size, corners2, ret)
        cv2.imshow('Calibration Image', image_rgb)
        cv2.waitKey(1)
        if self.is_blurry(image):
            # Compute the pose of the chessboard
            log.warning("Image is blurry. Please stabilize the camera or chessboard.")
            return False, image_rgb, gray, None
        
        return True, image_rgb, gray, corners2
        

    def calibrate_camera(self, num_images=20, angle_threshold=10):
        """
        Calibrate the camera using multiple images of a chessboard pattern.

        Args:
            num_images: Number of images to capture for calibration.
            angle_threshold: Minimum angular difference between poses in degrees.

        Updates:
            self.intrinsic: Camera intrinsic matrix.
            self.dist_coeffs: Distortion coefficients.
        """
        n_img = 0
        objpoints = []  # 3D points in real-world space
        imgpoints = []  # 2D points in image plane
        previous_rvecs = []  # List to store previous rotation vectors
        self.intrinsic = np.eye(3)
        self.dist_coeffs = np.zeros((5, 1))

        # Prepare object points based on the real chessboard dimensions
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                               0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        log.info("Starting camera calibration...")
        while n_img < num_images:
            # Capture an image from the camera
            ret, image_rgb, gray, corners2 = self.get_valid_corner_images()

            if not ret:
                continue

            ret_pnp, rvec, tvec = cv2.solvePnP(
                objp, corners2, self.intrinsic, self.dist_coeffs)
            if not ret_pnp:
                log.warning("Could not compute pose. Please adjust the chessboard and try again.")
                continue

            if not self.is_pose_different(rvec, previous_rvecs, angle_threshold):
                log.warning("Pose is too similar to previous images. Please change the angle.")
                continue

            # Append object points and image points
            n_img += 1
            objpoints.append(objp)
            imgpoints.append(corners2)
            previous_rvecs.append(rvec)
            log.info(f"Captured image {n_img}/{num_images} for calibration.")


        cv2.destroyAllWindows()

        if n_img < num_images:
            log.error("Calibration was not completed.")
            return

        # Get image size
        image_size = gray.shape[::-1]

        # Calibrate the camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None)

        if ret:
            # Store the intrinsic parameters and distortion coefficients
            self.intrinsic = camera_matrix
            self.dist_coeffs = dist_coeffs

            # Log the results
            log.info("Camera calibration successful.")
            log.info("Camera matrix:\n%s", camera_matrix)
            log.info("Distortion coefficients:\n%s", dist_coeffs)
        else:
            log.error("Camera calibration failed.")

    def capture_transformations(self, angle_threshold=10):
        """
        Capture transformations from robot to base and from target to camera.

        Args:
            angle_threshold: Minimum angular difference between poses in degrees.

        Returns:
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam
        """
        R_gripper2base, t_gripper2base = [], []
        R_target2cam, t_target2cam = [], []
        previous_rvecs = []  # For checking pose differences

        log.info("Starting to capture transformations for hand-eye calibration...")
        while len(R_gripper2base) < self.num_samples:
            # Capture gripper-to-base transformation
            R_g2b, t_g2b = self.robotic_arm.capture_gripper_to_base()
            if R_g2b is None or t_g2b is None:
                log.warning("Failed to get robot transformation. Retrying...")
                continue

            # Capture target-to-camera transformation
            ret, R_t2c, t_t2c, rvec = self.detect_chessboard_pose(return_rvec=True)
            if ret:
                if self.is_pose_different(rvec, previous_rvecs, angle_threshold):
                    if not self.is_blurry_image:  # Check if the last image was not blurry
                        R_gripper2base.append(R_g2b)
                        t_gripper2base.append(t_g2b)
                        R_target2cam.append(R_t2c)
                        t_target2cam.append(t_t2c)
                        previous_rvecs.append(rvec)
                        log.info(f"Captured transformation {len(R_gripper2base)}/{self.num_samples}.")
                    else:
                        log.warning("Image is blurry. Please stabilize the camera or chessboard.")
                else:
                    log.warning("Pose is too similar to previous images. Please change the angle.")
            else:
                log.warning("Chessboard not detected. Ensure proper target positioning and retry.")

        log.info("Finished capturing transformations.")
        return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam

    def detect_chessboard_pose(self, return_rvec=False):
        """
        Detect the chessboard in the image and compute its pose relative to the camera.

        Args:
            return_rvec: If True, also return the rotation vector.

        Returns:
            ret: Boolean indicating success of chessboard detection.
            R_t2c: Rotation matrix from target (chessboard) to camera.
            t_t2c: Translation vector from target (chessboard) to camera.
            rvec (optional): Rotation vector from target to camera.
        """
        if self.intrinsic is None or self.dist_coeffs is None:
            raise ValueError("Camera intrinsic parameters and distortion coefficients are not set. "
                             "Please calibrate the camera first.")

        # Define object points (3D points of the chessboard corners in the chessboard frame)
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                               0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size


        ret, image_rgb, gray, corners2= self.get_valid_corner_images()
        if not ret:
            return False, None, None
        
        # SolvePnP to find the rotation and translation of the chessboard relative to the camera
        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, self.intrinsic, self.dist_coeffs)

        if ret_pnp:
            R_t2c = cv2.Rodrigues(rvec)[0]
            t_t2c = tvec
            if return_rvec:
                return True, R_t2c, t_t2c, rvec
            else:
                return True, R_t2c, t_t2c

        if return_rvec:
            return False, None, None, None
        else:
            return False, None, None

    def calibrate(self, angle_threshold=10):
        """
        Perform eye-hand calibration.

        Args:
            angle_threshold: Minimum angular difference between poses in degrees.

        Returns:
            Calibrated rotation matrix and translation vector.
        """
        if self.intrinsic is None or self.dist_coeffs is None:
            log.info("Camera Intrinsic Parameters and Distortion Coefficients are not set.")
            log.info("Starting Camera Calibration.")
            self.calibrate_camera()

        # Capture the required transformations
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = self.capture_transformations(angle_threshold)

        # Perform the calibration
        R_calibrated, t_calibrated = self.calibrate_eye_hand(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, self.eye_to_hand
        )

        log.info("Hand-eye calibration completed.")
        log.info("Calibrated Rotation Matrix:\n%s", R_calibrated)
        log.info("Calibrated Translation Vector:\n%s", t_calibrated)
        return R_calibrated, t_calibrated

    @staticmethod
    def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):
        """
        Perform eye-hand calibration using OpenCV.

        Args:
            R_gripper2base: List of rotation matrices from gripper to base.
            t_gripper2base: List of translation vectors from gripper to base.
            R_target2cam: List of rotation matrices from target to camera.
            t_target2cam: List of translation vectors from target to camera.
            eye_to_hand: Boolean indicating eye-to-hand calibration.

        Returns:
            Calibrated rotation matrix and translation vector.
        """
        if eye_to_hand:
            R_base2gripper, t_base2gripper = [], []
            for R, t in zip(R_gripper2base, t_gripper2base):
                R_b2g = R.T
                t_b2g = -R_b2g @ t
                R_base2gripper.append(R_b2g)
                t_base2gripper.append(t_b2g)

            R_gripper2base = R_base2gripper
            t_gripper2base = t_base2gripper

        R, t = cv2.calibrateHandEye(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
        )
        return R, t

if __name__ == "__main__":
    # Create a RoboticArm object
    # robot = RoboticArm()  # Replace with your robotic arm initialization
    robot = None  # Replace with your robotic arm initialization
    import json
    # Create a Camera object
    camera_config = './default_config.json'
    config = json.load(open(camera_config, 'r'))
    camera = o3d.io.AzureKinectSensor(
        o3d.io.read_azure_kinect_sensor_config(camera_config))
    camera.connect(0)
    print(config['intrinsic_matrix'])
    print(config['distortion_coeffs'])
    # Initialize EyeHandCalibration without intrinsic parameters
    eye_hand_calib = EyeHandCalibration(robot, camera, 
                                        num_samples=10, 
                                        eye_to_hand=True,
                                        intrinsic=config['intrinsic_matrix'],
                                        dist_coeffs=config['distortion_coeffs'],
                                        )

    # Calibrate the camera
    eye_hand_calib.calibrate_camera(num_images=20)

    # Perform eye-hand calibration
    # eye_hand_calib.calibrate()
