import sys
import time
import json
import logging
import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
import cv2
import copy
from typing import Callable, Union, List
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, QObject, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap
from functools import wraps

# Import specific modules from vtkmodules
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkActor,
    vtkPolyDataMapper,
    vtkFollower
)
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkLine
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList, vtkUnsignedCharArray, vtkDataArray
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkRenderingFreeType import vtkVectorText
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget 
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkFiltersSources import vtkCubeSource
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from utils import RobotInterface, CameraInterface, ARUCO_BOARD
from utils import CalibrationData
from .pcd_streamer import PCDStreamerFromCamera, PCDUpdater
from .app_ui import PCDStreamerUI
# Import constants
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR



logger = logging.getLogger(__name__)
_callback_names = []

@staticmethod
def callback(func):
    """Decorator to collect callback methods."""
    _callback_names.append(func.__name__)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return func(self, *args, **kwargs)
    # setattr(PipelineController, func.__name__, wrapper)
    return wrapper


class PCDStreamer(PCDStreamerUI):
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing  and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """
    def __init__(self, params:dict=None):
        super().__init__()
        """Initialize.

        Args:
            camera_config_file (str): Camera configuration json file.
            rgbd_video (str): RS bag file containing the RGBD video. If this is
                provided, connected cameras are ignored.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
        """
        self.params = params
        self.rectangle_geometry = None
        self.prev_frame_time = time.time()
        self.frame_num = 0
        self.real_fps = 0
        self.frame = None
        self.callback_bindings()
        self.palettes = self.get_num_of_palette(80)
        self.__init_scene_objects()
        self.__init_bbox()
        self.__init_signals()
        self.set_disable_before_stream_init()
        self.init_ui_values_from_params()
        self.streamer = PCDStreamerFromCamera(params=params)
        self.pcd_updater = PCDUpdater(self.renderer)
        self.streamer.camera_frustrum.register_renderer(self.renderer)
        self.callbacks = {name: getattr(self, name) for name in _callback_names}

    def init_ui_values_from_params(self):
        self.calib_save_text.setText(self.params.get('calib_path', "Please_set_calibration_path"))
        self.board_col_num_edit.setValue(self.params.get('board_shape', (11, 6))[0])
        self.board_row_num_edit.setValue(self.params.get('board_shape', (11, 6))[1])
        self.board_square_size_num_edit.setValue(self.params.get('board_square_size', 0.023))
        self.board_marker_size_num_edit.setValue(self.params.get('board_marker_size', 0.0175))
        self.board_type_combobox.setCurrentText(self.params.get('board_type', "DICT_4X4_100"))

    def __init_signals(self):
        self.streaming = False
        self.show_axis = False

    def init_bbox_controls(self, layout:QtWidgets.QVBoxLayout):
        self.bbox_groupbox = QtWidgets.QGroupBox("Bounding Box Controls")
        group_layout = QtWidgets.QVBoxLayout()
        self.bbox_groupbox.setLayout(group_layout)
        layout.addWidget(self.bbox_groupbox)

        self.bbox_sliders = {}
        self.bbox_edits = {}
        bbox_params = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for param in bbox_params:
            h_layout = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(param)
            h_layout.addWidget(label)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(-500, 500)
            h_layout.addWidget(slider)
            spin_box = QtWidgets.QDoubleSpinBox()
            spin_box.setRange(-5.0, 5.0)
            h_layout.addWidget(spin_box)
            group_layout.addLayout(h_layout)
            self.bbox_sliders[param] = slider
            self.bbox_edits[param] = spin_box

        h_layout = QtWidgets.QHBoxLayout()
        group_layout.addLayout(h_layout)
        self.save_bbox_button = QtWidgets.QPushButton("Save")
        h_layout.addWidget(self.save_bbox_button)
        self.load_bbox_button = QtWidgets.QPushButton("Load")
        h_layout.addWidget(self.load_bbox_button)

    def set_disable_before_stream_init(self):
        self.capture_toggle.setEnabled(False)
        self.seg_model_init_toggle.setEnabled(False)
        self.data_collect_button.setEnabled(False)
        self.calib_collect_button.setEnabled(False)
        self.calib_button.setEnabled(False)
        self.calib_op_save_button.setEnabled(False)
        self.calib_op_load_button.setEnabled(False)
        self.calib_op_run_button.setEnabled(False)
        self.detect_board_toggle.setEnabled(False)
        self.show_axis_in_scene_button.setEnabled(False)
        self.calib_list_remove_button.setEnabled(False)
        self.robot_move_button.setEnabled(False)

    def set_enable_after_stream_init(self):
        self.capture_toggle.setEnabled(True)
        self.seg_model_init_toggle.setEnabled(True)
        self.data_collect_button.setEnabled(True)
        self.calib_collect_button.setEnabled(True)


    def set_enable_after_calib_init(self):
        self.calib_button.setEnabled(True)
        self.detect_board_toggle.setEnabled(True)
        self.show_axis_in_scene_button.setEnabled(True)
        self.calib_list_remove_button.setEnabled(True)
        self.robot_move_button.setEnabled(True)
        self.calib_op_save_button.setEnabled(True)
        self.calib_op_load_button.setEnabled(True)
        self.calib_op_run_button.setEnabled(True)

    def callback_bindings(self):
        # Binding callbacks to GUI elements
        # General Tab
        self.stream_init_button.clicked.connect(self.on_stream_init_button_clicked)

        # View Tab
        self.capture_toggle.stateChanged.connect(self.on_capture_toggle_state_changed)
        self.seg_model_init_toggle.stateChanged.connect(self.on_seg_model_init_toggle_state_changed)
        self.acq_mode_toggle.stateChanged.connect(self.on_acq_mode_toggle_state_changed)
        self.display_mode_combobox.currentTextChanged.connect(self.on_display_mode_combobox_changed)

        # Calibration Tab
        self.cam_calib_init_button.clicked.connect(self.on_cam_calib_init_button_clicked)
        self.robot_init_button.clicked.connect(self.on_robot_init_button_clicked)
        self.calib_collect_button.clicked.connect(self.on_calib_collect_button_clicked)
        self.calib_button.clicked.connect(self.on_calib_button_clicked)
        self.detect_board_toggle.stateChanged.connect(self.on_detect_board_toggle_state_changed)
        self.show_axis_in_scene_button.clicked.connect(self.on_show_axis_in_scene_button_clicked)
        self.calib_list_remove_button.clicked.connect(self.on_calib_list_remove_button_clicked)
        self.robot_move_button.clicked.connect(self.on_robot_move_button_clicked)
        self.calib_op_load_button.clicked.connect(self.on_calib_op_load_button_clicked)
        self.calib_op_save_button.clicked.connect(self.on_calib_op_save_button_clicked)
        self.calib_op_run_button.clicked.connect(self.on_calib_op_run_button_clicked)
        self.calib_save_button.clicked.connect(self.on_calib_save_button_clicked)
        self.calib_check_button.clicked.connect(self.on_calib_check_button_clicked)
        # BBox sliders and edits are connected in init_bbox()

        # Data Tab
        self.data_collect_button.clicked.connect(self.on_data_collect_button_clicked)
        self.data_save_button.clicked.connect(self.on_data_save_button_clicked)

        # Key press events
        self.vtk_widget.AddObserver("KeyPressEvent", self.on_key_press)

    # Define the callback methods
    def on_stream_init_button_clicked(self):
        if self.streaming:
            # Stop streaming
            self.streaming = False
            self.stream_init_button.setText("Start")
            if hasattr(self, 'timer'):
                self.timer.stop()
            if hasattr(self.streamer, 'camera'):
                self.streamer.camera.disconnect()
            self.status_message.setText("System: Stream Stopped")
        else:
            # Start streaming
            self.streaming = True
            self.stream_init_button.setText("Stop")
            connected = self.streamer.camera_mode_init()
            if connected:
                self.status_message.setText("System: Streaming from Camera")
                self.timer = QTimer()
                self.timer.timeout.connect(self.frame_calling)
                self.timer.start()  # Update at ~33 FPS
            else:
                self.status_message.setText("System: Failed to connect to Camera")
            self.set_vtk_camera_from_intrinsics(self.streamer.intrinsic_matrix, self.streamer.extrinsics)
        self.set_enable_after_stream_init()

    def set_vtk_camera_from_intrinsics(self, intrinsic_matrix, extrinsics):
        """
        Configure the VTK camera using intrinsic and extrinsic matrices.
        
        :param renderer: vtkRenderer instance.
        :param intrinsic_matrix: 3x3 NumPy array (camera intrinsics).
        :param extrinsics: 4x4 NumPy array (camera extrinsics).
        """
        # Extract parameters from intrinsic matrix
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        
        # Set up the VTK camera
        camera = self.renderer.GetActiveCamera()
        
        position = extrinsics[:3, 3]
        rotation = extrinsics[:3, :3]

        focal_point = position + rotation[:, 2]
        view_up = -rotation[:, 1]
        
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point)
        camera.SetViewUp(*view_up)
        
        image_width = 1280
        image_height = 720 
        aspect_ratio = image_width / image_height
        fov_y = 2 * np.arctan(image_height / (2 * fy)) * 180 / np.pi
        camera.SetViewAngle(fov_y)
        
        self.renderer.ResetCamera()

    def frame_calling(self):
        current_frame_time = time.time()
        if self.frame_num % 10 == 0:  # Avoid calculation on the first frame
            time_diff = current_frame_time - self.prev_frame_time
            if time_diff > 0:  # Avoid division by zero
                self.real_fps = 10*(1.0 / time_diff)
            self.prev_frame_time = current_frame_time
        # Capture a frame from the fake camera
        frame = self.streamer.get_frame(take_pcd=True)
        
        frame_elements = {
            'fps': round(self.real_fps, 2),  # Assuming 30 FPS for fake camera
        }
        frame_elements.update(frame)
        self.current_frame = frame_elements
        self.update(frame_elements)

    def update(self, frame_elements: dict):
        """Update visualization with point cloud and images."""
        if 'pcd' in frame_elements:
            self.update_pcd_geometry(frame_elements['pcd'])

        if 'color' in frame_elements:
            self.update_color_image(frame_elements['color'])

        if 'depth' in frame_elements:
            self.update_depth_image(frame_elements['depth'])

        if 'status_message' in frame_elements:
            self.status_message.setText(frame_elements['status_message'])

        if 'fps' in frame_elements:
            fps = frame_elements["fps"]
            self.fps_label.setText(f"FPS: {int(fps)}")

        # if hasattr(self, 'robot'):
        #     if 'robot_pose' in frame_elements:
        #         self.robot_end_frame.SetUserMatrix(frame_elements['robot_pose'])

        self.frame_num += 1
        logger.debug(f"Frame: {self.frame_num}")

    def update_pcd_geometry(self, pcd):
        """Update the point cloud visualization with Open3D point cloud data."""
        if not isinstance(pcd, o3d.geometry.PointCloud):
            logger.error("Input to update_pcd_geometry is not a valid Open3D PointCloud")
            return
        self.pcd_updater.update_pcd(pcd)
        self.vtk_widget.GetRenderWindow().Render()

        logger.debug("Point cloud visualization updated.")

    # Implement other callback methods as needed (even if they are empty for now)
    def on_toggle_record_state_changed(self):
        logger.debug("Record state changed")

    def on_save_pcd_button_clicked(self):
        logger.debug("Saving PCD")

    def on_save_rgbd_button_clicked(self):
        logger.debug("Saving RGBD")

    def on_capture_toggle_state_changed(self):
        logger.debug("Capture state changed")

    def on_seg_model_init_toggle_state_changed(self):
        if not hasattr(self, 'pcd_seg_model'):
            from ultralytics import YOLO, SAM
            self.pcd_seg_model = YOLO("yolo11x-seg.pt")

        logger.debug(f"Segmentation model state changed:{self.seg_model_init_toggle.isChecked()}")

    def on_acq_mode_toggle_state_changed(self):
        logger.debug("Acquisition mode state changed")
    
    @callback
    def on_display_mode_combobox_changed(self, text):
        self.display_mode = text
        logger.debug(f"Display mode changed to: {text}")

    def on_robot_init_button_clicked(self):
        from utils import RobotInterface
        self.robot =  RobotInterface()
        try:
            self.robot.find_device()
            self.robot.connect()
            ip = self.robot.ip_address
            msg = f'Robot: Connected [{ip}]'
            self.robot_init_button.setStyleSheet("background-color: green;")
            if hasattr(self, 'calibration_data'):
                self.calibration_data.reset()
        except Exception as e:
            msg = f'Robot: Connection failed'
            del self.robot
            logger.error(msg+f' [{e}]')
            self.robot_init_button.setStyleSheet("background-color: red;")
            self.flag_robot_init = False
        # logger.debug("Robot init button clicked")

    def on_calib_data_list_changed(self):
        self.calib_data_list.clear()
        self.calib_data_list.addItems(self.calibration_data.display_str_list)
        logger.debug("Calibration data list changed")

    def on_calib_op_load_button_clicked(self):
        self.calibration_data.load_img_and_pose()
        logger.debug("Calibration operation load button clicked")

    def on_calib_op_save_button_clicked(self):
        self.calibration_data.save_img_and_pose()
        logger.debug("Calibration operation save button clicked")

    def update_progress(self, value):
        pose = self.robot.capture_gripper_to_base(sep=False)
        img = self.img_to_array(self.current_frame['color'])
        self.calibration_data.modify(value, img, pose)
        logger.debug(f"Robot Move Progress: {value} and update calibration data")
        # self.label.setText(f"Progress: {value}")

    def calibration_finished(self):
        self.calibration_data.calibrate_all()
        logger.info("Calibration operation completed.")

    def on_calib_op_run_button_clicked(self):
        self.calib_thread = CalibrationThread(self.robot, self.calibration_data, self.current_frame)
        self.calib_thread.progress.connect(self.update_progress)
        self.calib_thread.finished.connect(self.calibration_finished)
        self.calib_thread.start()
        logger.debug("Calibration operation run button clicked")

    def on_calib_save_button_clicked(self):
        path = self.calib_save_text.text()
        self.calibration_data.save_calibration_data(path)
        logger.debug(f"Calibration saved: {path}")

    def on_calib_check_button_clicked(self):
        try:
            self.on_cam_calib_init_button_clicked()
            path = self.params['calib_path']
            with open(path, 'r') as f:
                self.calib: dict = json.load(f)
            intrinsic = np.array(self.calib.get('camera_matrix'))
            dist_coeffs = np.array(self.calib.get('dist_coeffs'))
            self.streamer.intrinsic_matrix = intrinsic
            self.streamer.dist_coeffs = dist_coeffs
            self.calibration_data.load_camera_parameters(intrinsic, dist_coeffs)
            self.calib_combobox.clear()
            for name in self.calib.get('calibration_results').keys():
                self.calib_combobox.addItem(name)
                # self.pipeline_view.scene_widgets.calib_combobox.add_item(name)
            self.calib_combobox.setEnabled(True)
            self.calib_combobox.setCurrentIndex(0)
            curent_selected = self.calib_combobox.currentText()
            self.T_CamToBase = np.array(self.calib.get('calibration_results').get(curent_selected).get('transformation_matrix'))
            self.center_to_base_toggle.setEnabled(True)
            
        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")
        logger.debug("Calibration check button clicked")

    def on_cam_calib_init_button_clicked(self):
        try:
            if not hasattr(self, 'calibration_data'):
                from utils import CalibrationData
                charuco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_BOARD[self.params['board_type']])
                charuco_board = cv2.aruco.CharucoBoard(
                    self.params['board_shape'],
                    squareLength=self.params['board_square_size'] / 1000,
                    markerLength=self.params['board_marker_size'] / 1000,
                    dictionary=charuco_dict
                )
                self.calibration_data = CalibrationData(charuco_board, save_dir=self.params['folder_path'])
                self.calibration_data.data_changed.connect(self.on_calib_data_list_changed)
                logger.debug("Camera calibration init button clicked")
            else:
                square_size = self.board_square_size_num_edit.value()
                marker_size = self.board_marker_size_num_edit.value()
                board_col = self.board_col_num_edit.value()
                board_row = self.board_row_num_edit.value()
                board_type = self.board_type_combobox.currentText()
                board_shape = (board_col, board_row)
                logger.debug(f"Reinit camera calibration with Board type: {board_type}, shape: {board_shape}, square size: {square_size}, marker size: {marker_size}")
                self.params['board_type'] = board_type
                self.params['board_shape'] = board_shape
                self.params['board_square_size'] = square_size
                self.params['board_marker_size'] = marker_size
                charuco_board = cv2.aruco.CharucoBoard(
                    self.params['board_shape'],
                    squareLength=self.params['board_square_size'] / 1000, # to meter
                    markerLength=self.params['board_marker_size'] / 1000, # to meter
                    dictionary=cv2.aruco.getPredefinedDictionary(ARUCO_BOARD[self.params['board_type']])
                )
                self.calibration_data = CalibrationData(charuco_board, save_dir=self.params['folder_path'])

            if not hasattr(self, 'camera_interface'):
                from utils import CameraInterface
                self.camera_interface: CameraInterface = CameraInterface(self.streamer.camera, self.calibration_data)
            
            self.cam_calib_init_button.setStyleSheet("background-color: green;")
            self.set_enable_after_calib_init()
        except Exception as e:
            self.cam_calib_init_button.setStyleSheet("background-color: red;")
            logger.error(f"Failed to init camera calibration: {e}")
        
    def on_calib_collect_button_clicked(self):
        if hasattr(self, 'timer'):
            self.timer.stop()

        if hasattr(self, 'calibration_data'):
            robot_pose = None
            if hasattr(self, 'robot'):
                robot_pose = self.robot.capture_gripper_to_base(sep=False)
            color = self.img_to_array(self.current_frame['color'])
            self.calibration_data.append(color, robot_pose=robot_pose)

        if not hasattr(self, 'timer'):
            self.timer.start()
        logger.debug("Calibration collect button clicked")
    
    def on_calib_list_remove_button_clicked(self):
        self.calibration_data.pop(self.calib_data_list.currentIndex().row())
        logger.debug(f"Calibration list remove button clicked")

    def on_robot_move_button_clicked(self):
        idx = self.calib_data_list.currentIndex().row()
        try:
            self.robot.move_to_pose(
                self.calibration_data.robot_poses[idx])
            
            if hasattr(self, 'timer'):
                self.timer.stop()
            self.calibration_data.modify(idx, self.img_to_array(self.current_frame['color']),
                                        self.robot.capture_gripper_to_base(sep=False),
                                        copy.deepcopy(self.bbox_params))
            if hasattr(self, 'timer'):
                self.timer.start()
            logger.debug("Moving robot and collecting data")
        except:
            logger.error("Failed to move robot")


        logger.debug("Robot move button clicked")

    def on_calib_button_clicked(self):
        logger.debug("Calibration button clicked")

    def on_detect_board_toggle_state_changed(self):
        logger.debug("Detect board state changed to:")

    def on_data_collect_button_clicked(self):
        logger.debug("Data collect button clicked")

    def on_data_save_button_clicked(self):
        logger.debug("Data save button clicked")

    def on_key_press(self, obj, event):
        key = obj.GetKeySym()
        if key == 'space':
            logger.info("Space key pressed")
            pass  # Handle space key press if needed
    
    def img_to_array(self, image: Union[np.ndarray, o3d.geometry.Image, o3d.t.geometry.Image]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, o3d.geometry.Image):
            return np.asarray(image)
        elif isinstance(image, o3d.t.geometry.Image):
            return np.asarray(image.cpu())

    def update_color_image(self, color_image):
        """Update the color image display."""
        if self.color_groupbox.isChecked():
            image = self.img_to_array(color_image)
            # Convert color_image to QImage and display in QLabel
            if image.shape[2] == 3:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                
                q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.color_video.setPixmap(pixmap.scaled(self.color_video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


    def update_depth_image(self, depth_image):
        """Update the depth image display."""
        if self.depth_groupbox.isChecked():
            image = self.img_to_array(depth_image)
            # Convert depth_image to QImage and display in QLabel
            if image.shape[2] == 3:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                self.depth_video.setPixmap(pixmap.scaled(self.depth_video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


    def __init_scene_objects(self):
        """Initialize scene objects in the VTK renderer."""
        # Robot base frame
        size = [0.06]*3
        colors = vtkNamedColors()
        self.robot_base_frame = vtkAxesActor()
        self.robot_base_frame.AxisLabelsOff()
        self.robot_base_frame.SetTotalLength(*size)
        # self.robot_base_frame.SetYAxisLabelText('')
        # self.renderer
        self.renderer.AddActor(self.robot_base_frame)
        # Robot end frame
        self.robot_end_frame = vtkAxesActor()
        self.robot_end_frame.AxisLabelsOff()
        self.robot_end_frame.SetTotalLength(*size)
        self.renderer.AddActor(self.robot_end_frame)
        # Board pose
        self.board_pose_frame = vtkAxesActor()
        self.board_pose_frame.AxisLabelsOff()
        self.board_pose_frame.SetTotalLength(*size)
        self.renderer.AddActor(self.board_pose_frame)

        self.robot_base_frame.SetVisibility(0)
        self.robot_end_frame.SetVisibility(0)
        self.board_pose_frame.SetVisibility(0)

    def __init_bbox(self):
        """Initialize bounding box visualization."""
        # Initialize bounding box parameters
        self.bbox_params = {'xmin': -0.5, 'xmax': 0.5,
                            'ymin': -0.5, 'ymax': 0.5,
                            'zmin': 0.0, 'zmax': 1.0}

        # Create bounding box actor
        self.update_bounding_box()

        # Set up callbacks for bbox sliders and edits
        bbox_params = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for param in bbox_params:
            # Assuming sliders and spin boxes are named accordingly
            slider: QtWidgets.QSlider = self.bbox_sliders[param]
            spin_box: QtWidgets.QDoubleSpinBox = self.bbox_edits[param]
            slider.setValue(int(self.bbox_params[param] * 100))
            spin_box.setValue(int(self.bbox_params[param]))

            slider.valueChanged.connect(lambda value, p=param: self._on_bbox_slider_changed(value, p))
            spin_box.valueChanged.connect(lambda value, p=param: self._on_bbox_edit_changed(value, p))

    def update_bounding_box(self):
        """Update the bounding box actor."""
        # Create a cube source with the bounding box dimensions
        cube = vtkCubeSource()
        cube.SetBounds(
            self.bbox_params['xmin'], self.bbox_params['xmax'],
            self.bbox_params['ymin'], self.bbox_params['ymax'],
            self.bbox_params['zmin'], self.bbox_params['zmax']
        )

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # Red color
        actor.GetProperty().SetRepresentationToWireframe()

        # Remove old bounding box actor if it exists
        if hasattr(self, 'bbox_actor'):
            self.renderer.RemoveActor(self.bbox_actor)

        # Add new bounding box actor
        self.bbox_actor = actor
        self.renderer.AddActor(self.bbox_actor)
        self.vtk_widget.GetRenderWindow().Render()

    def _on_bbox_slider_changed(self, value, param):
        self.bbox_params[param] = value / 100.0  # Assuming slider range is scaled
        # Update the corresponding spin box
        self.bbox_edits[param].setValue(self.bbox_params[param])
        self.update_bounding_box()

    def _on_bbox_edit_changed(self, value, param):
        self.bbox_params[param] = value
        # Update the corresponding slider
        self.bbox_sliders[param].setValue(int(self.bbox_params[param] * 100))
        self.update_bounding_box()

    def transform_geometry(self, name, transform=None):
        """Apply transformation to the specified actor."""
        actor = None
        if name == 'robot_base_frame':
            actor = self.robot_base_frame
        elif name == 'robot_end_frame':
            actor = self.robot_end_frame
        elif name == 'board_pose':
            actor = self.board_pose_frame

        if actor and transform is not None:
            # Assuming transform is a 4x4 numpy array
            matrix = vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    matrix.SetElement(i, j, transform[i, j])
            actor.SetUserMatrix(matrix)
            self.vtk_widget.GetRenderWindow().Render()

    def get_num_of_palette(self, num_colors):
        """Generate a color palette."""
        # For simplicity, generate random colors
        np.random.seed(0)
        palettes = np.random.randint(0, 256, size=(num_colors, 3))
        return palettes


    def init_settinngs_values(self):
        # self.chessboard_dims = self.params.get('board_shape', (11, 6))
        # self.pipeline_view.scene_widgets.board_col_num_edit.int_value = self.chessboard_dims[0]
        # self.pipeline_view.scene_widgets.board_row_num_edit.int_value = self.chessboard_dims[1]
        # self.pipeline_view.scene_widgets.board_square_size_num_edit.double_value = self.params.get('board_square_size', 0.023)
        # self.pipeline_view.scene_widgets.board_marker_size_num_edit.double_value = self.params.get('board_marker_size', 0.0175)
        # self.pipeline_view.scene_widgets.board_type_combobox.selected_text = self.params.get('board_type', "DICT_4X4_100")
        # self.pipeline_view.scene_widgets.calib_save_text.text_value = self.params.get('calib_path', "")
        pass
        # self.pipeline_view.scene_widgets.data_folder_text.text_value = ""
        

    def load_in_startup(self):
        startup_params = self.params.get('load_in_startup', {})
        if 'camera_init' in startup_params and startup_params['camera_init']:
            self.on_stream_init_button()
        if 'camera_calib_init' in startup_params and startup_params['camera_calib_init']:
            self.on_cam_calib_init_button()
        if 'robot_init' in startup_params and startup_params['robot_init']:
            self.on_robot_init_button()
        if 'calib_check' in startup_params and startup_params['calib_check']:
            self.on_calib_check_button()
        if 'collect_data_viewer' in startup_params and startup_params['collect_data_viewer']:
            self.collected_data.start_display_thread()

    def transform_element(self, elements:dict, element_name: str):
        pass
        

    def on_show_axis_in_scene_button_clicked(self):
        logger.debug(f"on_show_axis {self.show_axis}")
        if self.show_axis:
            # Stop streaming
            self.show_axis = False
            self.show_axis_in_scene_button.setText("Show Axis in Scene")
            self.robot_base_frame.SetVisibility(0)
            self.robot_end_frame.SetVisibility(0)
            self.board_pose_frame.SetVisibility(0)
        else:
            # Start streaming
            self.show_axis = True
            self.show_axis_in_scene_button.setText("Hide Axis in Scene")
            self.robot_base_frame.SetVisibility(1)
            self.robot_end_frame.SetVisibility(1)
            self.board_pose_frame.SetVisibility(1)
        self.renderer.GetRenderWindow().Render()

    def closeEvent(self, event):
        """Ensure the popup window is closed when the main window exits."""
        if hasattr(self, 'popup_window'):
            self.popup_window.close()
            self.popup_window = None
        logger.debug("Exiting main window")
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.streamer = None
        self.current_frame = None
        super().closeEvent(event) 



class CalibrationThread(QThread):
    progress = pyqtSignal(int)  # Signal to communicate progress updates
    def __init__(self, robot: RobotInterface, robot_poses: List[np.ndarray]):
        self.robot = robot
        self.robot_poses = robot_poses
        super().__init__()

    def run(self):
        self.robot.set_teach_mode(False)
        for idx, each_pose in enumerate(self.robot_poses):
            logger.info(f"Moving to pose {idx}")
            self.robot.move_to_pose(each_pose)
            self.progress.emit(idx)
        self.robot.set_teach_mode(True)
