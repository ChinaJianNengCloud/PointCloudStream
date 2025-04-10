import sys
import cv2
import vtk
import numpy as np
import logging
import shutil
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from robot.remote_control import BoardRobotSyncManager
from robot.pose import Pose 
# PySide6 imports
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QTabWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QImage

# Import the Qt interactor widget for VTK
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkActor, vtkPolyDataMapper

# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ------------------------------------------------------------
# CalibrationData class (as before, with minimal modifications)
# ------------------------------------------------------------
class CalibrationData(QWidget):
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
        self.__save_dir: Path = Path(save_dir) if save_dir is not None else None
        self.calibration_results = {}
        self.image_size = None
        self.camera_matrix = None
        self.dist_coeffs = None

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
                logger.warning("Not enough markers detected in image")
        else:
            logger.warning("No valid Charuco corners detected in image")
            cur_object_points, cur_image_points = None, None
            ret = False
        return ret, cur_object_points, cur_image_points, charuco_corners, charuco_ids

    def append(self, image: np.ndarray, robot_pose: np.ndarray = None, recalib=False):
        ret, cur_object_points, cur_image_points, _, _ = self.board_dectect(image)
        if ret:
            self.images.append(image.copy())
            self.imgpoints.append(cur_image_points)
            self.objpoints.append(cur_object_points)
            self.robot_poses.append(robot_pose)
            logger.info(f"Board detected in image, image added. Count: {len(self.images)}")
            if recalib:
                self.calibrate_all()
        else:
            logger.warning("Failed to detect board in image, image not added.")

    def calibrate_camera(self):
        logger.info(f"Number of image points: {len(self.imgpoints)}")
        if len(self.imgpoints) >= 3:
            ret, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, self.image_size, None, None
            )
            if ret:
                logger.info("Camera calibration successful")
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
                    logger.warning("Could not solvePnP for image points")
            for idx in reversed(invalid_index):
                self.images.pop(idx)
                self.robot_poses.pop(idx)
                self.objpoints.pop(idx)
                self.imgpoints.pop(idx)
        else:
            logger.warning("Camera matrix and distortion coefficients are not available.")

    def calibrate_all(self):
        if len(self.images) >= 3:
            self.calibrate_camera()
            self.board_pose_calculation()
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
                    pose_file.write(f"{idx}\n")
            logger.info(f"Saved images and poses to {self.__save_dir}")

# ------------------------------------------------------------
# DummyRobot and update_actor_transform (as before)
# ------------------------------------------------------------
class DummyRobot:
    def __init__(self):
        self.current_pose = np.zeros(6)  # [x, y, z, rx, ry, rz]

    def get_state(self, state_type='tcp'):
        return self.current_pose

    def step(self, pose_vector, action_type='default', wait=True):
        self.current_pose = np.array(pose_vector)
        logger.info(f"DummyRobot: commanded to move to: {self.current_pose}")
        return Pose.from_1d_array(self.current_pose, vector_type="rotvec")

def update_actor_transform(actor, pose_vec):
    pos = pose_vec[0:3]
    rvec = pose_vec[3:6]
    R_mat, _ = cv2.Rodrigues(rvec)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_mat
    transform_matrix[:3, 3] = pos

    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, transform_matrix[i, j])

    vtk_transform = vtk.vtkTransform()
    vtk_transform.SetMatrix(vtk_matrix)
    actor.SetUserTransform(vtk_transform)

# ------------------------------------------------------------
# CameraWidget: shows live feed with drawn detections and capture button.
# ------------------------------------------------------------
class CameraWidget(QWidget):
    def __init__(self, calibration_data, parent=None):
        super().__init__(parent)
        self.calibration_data = calibration_data
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera.")
        self.video_label = QLabel("Waiting for camera frames...")
        self.video_label.setAlignment(Qt.AlignCenter)
        # (Optional) Let the label scale its contents.
        self.video_label.setScaledContents(True)
        self.capture_button = QPushButton("Capture Board Image")
        self.capture_button.clicked.connect(self.capture_image)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.capture_button)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Approximately 33 FPS
        self.last_frame = None

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.last_frame = frame.copy()
            board_detected, obj_pts, img_pts, charuco_corners, charuco_ids = self.calibration_data.board_dectect(frame)
            if board_detected:
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
                if (self.calibration_data.camera_matrix is not None and 
                    self.calibration_data.dist_coeffs is not None):
                    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts,
                                                       self.calibration_data.camera_matrix,
                                                       self.calibration_data.dist_coeffs)
                    if success:
                        rotation = R.from_rotvec(rvec.ravel())
                        logger.info(f"rvec: {rvec.ravel()}, tvec: {tvec.ravel()}, "
                                    f"rxyz: {rotation.as_euler('xyz', degrees=False)}")
                        cv2.drawFrameAxes(frame, self.calibration_data.camera_matrix,
                                          self.calibration_data.dist_coeffs,
                                          rvec, tvec, 0.05)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def capture_image(self):
        if self.last_frame is not None:
            board_detected, obj_pts, img_pts, _, _ = self.calibration_data.board_dectect(self.last_frame)
            if board_detected:
                self.calibration_data.append(self.last_frame)
                logger.info(f"Captured board image. Total captured: {len(self.calibration_data.images)}")
                if len(self.calibration_data.images) >= 3:
                    if self.calibration_data.calibrate_all():
                        logger.info("Calibration successful after image capture")
                    else:
                        logger.warning("Calibration failed during capture")
            else:
                logger.warning("Board not detected in captured image.")

    def closeEvent(self, event):
        self.camera.release()
        event.accept()

# ------------------------------------------------------------
# VTKViewerWidget: embeds a VTK render window in a Qt widget.
# ------------------------------------------------------------
class VTKViewerWidget(QWidget):
    def __init__(self, calibration_data, dummy_robot, sync_manager, parent=None):
        super().__init__(parent)
        self.calibration_data = calibration_data
        self.dummy_robot = dummy_robot
        self.sync_manager = sync_manager
        self.latest_board_pose = None

        # Create the QVTKRenderWindowInteractor as a child widget.
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        # Do not force a native window; comment out or remove the next line:
        # self.vtk_widget.setAttribute(Qt.WA_NativeWindow)
        # (Optionally, ensure it acts strictly as a widget.)
        # self.vtk_widget.setWindowFlags(Qt.Widget)

        layout = QVBoxLayout(self)
        layout.addWidget(self.vtk_widget)
        self.setLayout(layout)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Set up your actors (axes, board, and robot)
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        self.renderer.AddActor(axes)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(0.05)
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        self.board_actor = vtk.vtkActor()
        self.board_actor.SetMapper(sphere_mapper)
        self.renderer.AddActor(self.board_actor)

        cube_source = vtk.vtkCubeSource()
        cube_source.SetXLength(0.1)
        cube_source.SetYLength(0.1)
        cube_source.SetZLength(0.1)
        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputConnection(cube_source.GetOutputPort())
        self.robot_actor = vtk.vtkActor()
        self.robot_actor.SetMapper(cube_mapper)
        initial_robot_vector = [3, 2, 1, 0.3, 0.2, 0.1]
        update_actor_transform(self.robot_actor, np.array(initial_robot_vector))
        self.renderer.AddActor(self.robot_actor)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_vtk)
        self.timer.start(30)

        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)

    def showEvent(self, event):
        super().showEvent(event)
        # Properly initialize the VTK interactor after the widget is shown.
        # self.vtk_widget.Initialize()
        # You can also call self.vtk_widget.Start() if needed,
        # but using the timer to call ProcessEvents() works well for integration.

    def update_vtk(self):
        if not self.vtk_widget.isVisible():
            return
        self.interactor.ProcessEvents()
        self.vtk_widget.GetRenderWindow().Render()

    def on_key_press(self, caller, event):
        key = caller.GetKeySym()
        if key.lower() == "s":
            if self.latest_board_pose is not None:
                board_pose = Pose.from_1d_array(self.latest_board_pose, vector_type="rotvec")
                current_robot_pose = Pose.from_1d_array(self.dummy_robot.get_state(), vector_type="rotvec")
                self.sync_manager.sync(board_pose, current_robot_pose)
            else:
                logger.info("No board pose available yet.")
        elif key.lower() == "c":
            if self.latest_board_pose is not None:
                current_board_pose = Pose.from_1d_array(self.latest_board_pose, vector_type="rotvec")
                target_robot_pose = self.sync_manager.step(current_board_pose)
                update_actor_transform(self.robot_actor, target_robot_pose.to_1d_array("rotvec"))
            else:
                logger.info("No board pose available yet.")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)

    def update_board_pose(self, board_pose):
        self.latest_board_pose = board_pose
        update_actor_transform(self.board_actor, board_pose)

# ------------------------------------------------------------
# MainWindow: contains tabs for the Camera feed and the 3D VTK viewer.
# ------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, calibration_data, dummy_robot, sync_manager):
        super().__init__()
        self.setWindowTitle("Calibration and 3D Viewer")
        self.resize(1200, 800)
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        self.camera_widget = CameraWidget(calibration_data)
        self.vtk_viewer_widget = VTKViewerWidget(calibration_data, dummy_robot, sync_manager, self)
        
        self.tab_widget.addTab(self.camera_widget, "Camera")
        self.tab_widget.addTab(self.vtk_viewer_widget, "3D Viewer")

# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    
    # Set up the Charuco board parameters.
    MARKER_SIZE = [50, 100, 250, 1000]
    MARKER_GRID = [4, 5, 6]
    ARUCO_BOARD = {
        f'DICT_{grid}X{grid}_{size}': getattr(cv2.aruco, f'DICT_{grid}X{grid}_{size}')
        for size in MARKER_SIZE
        for grid in MARKER_GRID
        if hasattr(cv2.aruco, f'DICT_{grid}X{grid}_{size}')
    }
    params = {
        'board_shape': (3, 5),
        'board_square_size': 23,      # in mm
        'board_marker_size': 17.5,      # in mm
        'board_type': 'DICT_4X4_100'
    }
    charuco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_BOARD[params['board_type']])
    charuco_board = cv2.aruco.CharucoBoard(
        params['board_shape'],
        squareLength=params['board_square_size'] / 1000,
        markerLength=params['board_marker_size'] / 1000,
        dictionary=charuco_dict
    )
    calibration_data = CalibrationData(charuco_board, save_dir='./Calibration_results')
    # If calibration parameters are not set, use dummy intrinsic parameters.
    if calibration_data.camera_matrix is None:
        calibration_data.camera_matrix = np.array([[1000, 0, 960],
                                                     [0, 1000, 540],
                                                     [0, 0, 1]], dtype=float)
        calibration_data.dist_coeffs = np.zeros((5, 1), dtype=float)
    
    dummy_robot = DummyRobot()
    sync_manager = BoardRobotSyncManager(dummy_robot)
    
    main_window = MainWindow(calibration_data, dummy_robot, sync_manager)
    main_window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
