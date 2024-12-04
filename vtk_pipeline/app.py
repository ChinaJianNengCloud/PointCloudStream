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
import socket
import pickle
from typing import Callable, Union, List
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, QObject, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap
from scipy.spatial.transform import Rotation as R
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
from utils import CalibrationData, CollectedData, ConversationData
from utils.segmentation import segment_pcd_from_2d
from utils.net.client import send_message, discover_server

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
        self.streamer = PCDStreamerFromCamera(params=params)
        self.pcd_updater = PCDUpdater(self.renderer)
        self.collected_data = CollectedData(self.params.get('data_path', './data'))
        self.streamer.camera_frustrum.register_renderer(self.renderer)
        self.callbacks = {name: getattr(self, name) for name in _callback_names}
        self.palettes = self.get_num_of_palette(80)
        self.conversation_data = ConversationData()
        self.__init_scene_objects()
        self.__init_bbox()
        self.__init_signals()
        self.__init_ui_values_from_params()
        self.callback_bindings()
        self.set_disable_before_stream_init()


    def __init_ui_values_from_params(self):
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
        self.center_to_robot_base_toggle.stateChanged.connect(self.on_center_to_robot_base_toggle_state_changed)

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
        self.calib_combobox.currentTextChanged.connect(self.on_calib_combobox_changed)
        # BBox sliders and edits are connected in init_bbox()

        # Data Tab
        self.data_collect_button.clicked.connect(self.on_data_collect_button_clicked)
        self.data_save_button.clicked.connect(self.on_data_save_button_clicked)
        self.data_tree_view_load_button.clicked.connect(self.on_data_tree_view_load_button_clicked)
        self.data_folder_select_button.clicked.connect(self.on_data_folder_select_button_clicked)
        self.data_tree_view_remove_button.clicked.connect(self.on_data_tree_view_remove_button_clicked)
        self.data_tree_view.set_on_selection_changed(self.on_tree_selection_changed)
        self.collected_data.data_changed.connect(self._data_tree_view_update)

        # Agent Tab
        self.scan_button.clicked.connect(self.on_scan_button_clicked)
        self.send_button.clicked.connect(self.on_send_button_clicked)
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
        
        self.board_pose_update(frame)
        self.robot_pose_update(frame)
        self.segment_pcd_from_yolo(frame)


        frame_elements.update(frame)
        self.current_frame = frame_elements
        self.point_cloud_update(frame_elements)

    def robot_pose_update(self, frame):
        if self.show_axis:
            ret, _, pose_matrix = self.get_robot_pose()
            if ret:
                self.robot_end_frame.SetUserMatrix(self._numpy_to_vtk_matrix_4x4(pose_matrix))
                if not self.center_to_robot_base_toggle.isChecked():
                    self.robot_base_frame.SetUserMatrix(self._numpy_to_vtk_matrix_4x4(self.T_BaseToCam))
                else:
                    self.robot_base_frame.SetUserMatrix(self._numpy_to_vtk_matrix_4x4(np.eye(4)))
            


    def get_robot_pose(self):
        """
        Get the current pose of the robot's end-effector.
        
        Returns:
            bool: Whether the robot pose was successfully retrieved.
            list or None: The pose as a list of length 6, containing the translation (x, y, z) and rotation (rx, ry, rz) components in the camera frame if robot calibration data is available.
            np.ndarray or None: The pose as a 4x4 numpy array in the camera frame if robot calibration data is available.
        """
        if not hasattr(self, 'robot'):
            logger.error("Robot not initialized.")
            return False, None, None
        
        robot_pose = self.robot.capture_gripper_to_base(sep=False)
        t_xyz, r_xyz = robot_pose[0:3], robot_pose[3:6]
        base_to_end = np.eye(4)
        base_to_end[:3, :3] = R.from_euler('xyz', r_xyz, degrees=False).as_matrix()
        base_to_end[:3, 3] = t_xyz
        if not self.center_to_robot_base_toggle.isChecked():
            if hasattr(self, 'T_CamToBase'):
                cam_to_end = self.T_BaseToCam @ base_to_end
                xyzrxryrz = np.hstack((cam_to_end[:3, 3],
                                    R.from_matrix(cam_to_end[:3, :3]).as_euler('xyz', degrees=False)))
                return True, xyzrxryrz, cam_to_end
            else:
                logger.error("No robot calibration data detected.")
                return False, None, None
        else:
            return True, robot_pose, base_to_end

    def board_pose_update(self, frame):
        if hasattr(self, 'camera_interface') and self.detect_board_toggle.isChecked():
            rgb_with_pose, rvec, tvec = self.camera_interface._process_and_display_frame(
                self.img_to_array(frame['color']), ret_vecs=True)
            if rvec is None or tvec is None:
                logger.warning("Failed to detect board.")
            else:
                cam_to_board = np.eye(4)
                cam_to_board[:3, :3] = cv2.Rodrigues(rvec)[0]
                cam_to_board[:3, 3] = tvec.ravel()
                if self.center_to_robot_base_toggle.isChecked():
                    if hasattr(self, 'T_CamToBase'):
                        base_to_board = self.T_CamToBase @ cam_to_board
                        self.board_pose_frame.SetUserMatrix(self._numpy_to_vtk_matrix_4x4(base_to_board))
                else:
                    self.board_pose_frame.SetUserMatrix(self._numpy_to_vtk_matrix_4x4(cam_to_board))
    
    def point_cloud_update(self, frame_elements: dict):
        """Update visualization with point cloud and images."""
        if 'seg_labels' in frame_elements:
            if self.display_mode_combobox.currentText() == "Segmentation":
                self.palettes
                seg_labels = frame_elements['seg_labels']
                num_points = seg_labels.shape[0]
                colors = np.zeros((num_points, 3))
                valid_idx = seg_labels >= 0
                colors[valid_idx] = np.array(self.palettes)[seg_labels[valid_idx]] / 255.0


        if 'pcd' in frame_elements:
            if 'seg_labels' in frame_elements and self.display_mode_combobox.currentText() == "Segmentation":
                seg_labels = frame_elements['seg_labels']
                num_points = seg_labels.shape[0]
                colors = np.zeros((num_points, 3))
                valid_idx = seg_labels >= 0
                colors[valid_idx] = np.array(self.palettes)[seg_labels[valid_idx]] / 255.0
                self.update_pcd_geometry(frame_elements['pcd'], colors)
            else:
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
        # logger.debug(f"Frame: {self.frame_num}")

    def update_pcd_geometry(self, pcd, lb_colors: np.ndarray = None):
        """Update the point cloud visualization with Open3D point cloud data."""
        if not isinstance(pcd, o3d.geometry.PointCloud):
            logger.error("Input to update_pcd_geometry is not a valid Open3D PointCloud")
            return
        self.pcd_updater.update_pcd(pcd, lb_colors)
        self.vtk_widget.GetRenderWindow().Render()

        # logger.debug("Point cloud visualization updated.")

    def segment_pcd_from_yolo(self, frame: dict):
        if self.seg_model_init_toggle.isChecked():
            if not hasattr(self, 'pcd_seg_model'):
                logger.error("Segmentation model not initialized")
                return
            try:
                labels = segment_pcd_from_2d(self.pcd_seg_model, 
                                    frame['pcd'], frame['color'], 
                                    self.streamer.intrinsic_matrix, 
                                    self.streamer.extrinsics)
            except Exception as e:
                logger.error(f"Segmentation failed: {e}")
                frame['seg_labels'] = np.zeros((frame['pcd'].points.shape[0], 1), dtype=np.int64)
                return
            frame['seg_labels'] = labels

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
            self.pcd_seg_model = YOLO(self.params['yolo_model_path'])
        logger.debug(f"Segmentation model state changed:{self.seg_model_init_toggle.isChecked()}")

    def on_acq_mode_toggle_state_changed(self):
        logger.debug("Acquisition mode state changed")
    
    @callback
    def on_display_mode_combobox_changed(self, text):
        self.display_mode = text
        logger.debug(f"Display mode changed to: {text}")

    def on_center_to_robot_base_toggle_state_changed(self):
        logger.debug("Center to robot base state changed")
        if self.center_to_robot_base_toggle.isChecked():
            self.streamer.extrinsics = self.T_BaseToCam
        else:
            self.streamer.extrinsics = np.eye(4)

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
        if not hasattr(self, 'robot'):
            logger.error("Robot not initialized")
            return
        
        if not hasattr(self, 'current_frame'):
            logger.error("No current frame, please start streaming first.")
            return
        
        self.calib_thread = CalibrationThread(self.robot, self.calibration_data.robot_poses)
        self.calib_thread.progress.connect(self.update_progress)
        self.calib_thread.finished.connect(self.calibration_finished)
        self.calib_thread.start()
        logger.debug("Calibration operation run button clicked")

    def on_calib_save_button_clicked(self):
        path = self.calib_save_text.text()
        self.calibration_data.save_calibration_data(path)
        logger.debug(f"Calibration saved: {path}")

    @property
    def T_CamToBase(self):
        return self._T_CamToBase
    
    @T_CamToBase.setter
    def T_CamToBase(self, value: np.ndarray):
        assert value.shape == (4, 4)
        self._T_CamToBase = value
        self.T_BaseToCam = np.linalg.inv(value)

    def on_calib_combobox_changed(self, text):
        if text != "":
            self.T_CamToBase = np.array(self.calib.get('calibration_results').get(text).get('transformation_matrix'))
        logger.debug(f"Calibration combobox changed: {text}")

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
            self.calib_combobox.addItems(self.calib.get('calibration_results').keys())
            self.calib_combobox.setEnabled(True)
            self.calib_combobox.setCurrentIndex(0)
            curent_selected = self.calib_combobox.currentText()
            self.T_CamToBase = np.array(self.calib.get('calibration_results').get(curent_selected).get('transformation_matrix'))
            self.center_to_robot_base_toggle.setEnabled(True)
            
        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")
        logger.debug("Calibration check button clicked")

    def on_cam_calib_init_button_clicked(self):
        try:
            if not hasattr(self, 'calibration_data'):
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
                self.calibration_data.reset()
                self.calibration_data.board = charuco_board
                self.calibration_data.save_dir = self.params['folder_path']


            if not hasattr(self, 'camera_interface'):
                from utils import CameraInterface
                self.camera_interface: CameraInterface = CameraInterface(self.streamer.camera, self.calibration_data)
            
            self.cam_calib_init_button.setStyleSheet("background-color: green;")
            self.set_enable_after_calib_init()
        except Exception as e:
            self.cam_calib_init_button.setStyleSheet("background-color: red;")
            logger.error(f"Failed to init camera calibration: {e}")
        
    def on_calib_collect_button_clicked(self):
        if hasattr(self, 'calibration_data'):
            ret, robot_pose, _ = self.get_robot_pose()
            if ret:
                color = self.img_to_array(self.current_frame['color'])
                self.calibration_data.append(color, robot_pose=robot_pose)
            else:
                logger.error("Failed to get robot pose")
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
        self.calibration_data.calibrate_all()
        logger.debug("Calibration button clicked")

    def on_detect_board_toggle_state_changed(self):
        logger.debug("Detect board state changed to:")

    def on_data_collect_button_clicked(self):
        ret, robot_pose, _ = self.get_robot_pose()
        if ret:
            color = self.img_to_array(self.current_frame['color'])
            depth = self.img_to_array(self.current_frame['depth'])
            pcd =  self.current_frame['pcd']
            
            xyz = np.asarray(pcd.points)
            rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            seg = self.current_frame.get('seg', np.zeros(xyz.shape[0]))
            pcd_with_labels = np.hstack((xyz, rgb, seg.reshape(-1, 1)))

            self.collected_data.append(prompt=self.prompt_text.text(),
                                    color=color,
                                    depth=depth,
                                    point_cloud=pcd_with_labels,
                                    pose=robot_pose,
                                    bbox_dict=self.bbox_params)
            logger.debug("Data collected")
        else:
            logger.error("Failed to get robot pose")

        logger.debug("Data collect button clicked")

    def on_send_button_clicked(self):
        try:
            ret, robot_pose, _ = self.get_robot_pose()
            if ret:
                prompt = self.agent_prompt_editor.toPlainText()
                frame = copy.deepcopy(self.current_frame)
                colors = np.asarray(frame['pcd'].colors)
                points = np.asarray(frame['pcd'].points)
                labels = np.asarray(frame['seg_labels'])
                pcd_with_labels = np.hstack((points, colors, labels.reshape(-1, 1)))
                image = np.asarray(frame['color'].cpu())
                # pose = frame['robot_pose']
                past_actions = []
                msg_dict = {'prompt': prompt, 
                            'pcd': pcd_with_labels, 
                            'image': image, 
                            'pose': robot_pose, 
                            'past_actions': past_actions, 
                            'command': "process_pcd"}
                
                self.sendingThread  = DataSendToServerThread(ip=self.ip_editor.text(), 
                                                        port=int(self.port_editor.text()), 
                                                        msg_dict=msg_dict)
                self.conversation_data.append('User', prompt)
                self.sendingThread.progress.connect(self.on_send_progress)
                self.sendingThread.finished.connect(self.on_finish_sending_thread)
                self.send_button.setEnabled(False)
                self.sendingThread.start()
                logger.debug("Send button clicked")
            else:
                logger.error("Failed to get robot pose")
        except Exception as e:
            logger.error(f"Failed to send data: {e}")

    def on_finish_sending_thread(self):
        self.send_button.setEnabled(True)
        response = self.sendingThread.get_response()
        if response['status'] == 'action':
            self.conversation_data.append('Agent', str(response['message']))
        elif response['status'] == 'no_action':
            self.conversation_data.append('Agent', str(response['message']))
        logger.info(self.conversation_data.get_terminal_conversation())
        logger.debug("Sending thread finished")


    def refresh_line_progress(self, progress):
        """
        Refresh the terminal line with the current progress.
        :param progress: Tuple of (status, percentage)
        """
        status, percentage = progress
        bar_length = 40  # Length of the progress bar
        filled_length = int(bar_length * percentage / 100)
        bar = "#" * filled_length + "-" * (bar_length - filled_length)
        if percentage < 100:
            sys.stdout.write(f"\r[{bar}] {percentage:.2f}% - {status}  ")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r[{bar}] {percentage:.2f}% - {status}  \n")
            sys.stdout.flush()

    def on_send_progress(self, progress):
        # logger.debug(f"Send progress: {progress}")  # Log progress for debugging
        self.refresh_line_progress(progress)  # Update the line progress bar

    def on_scan_button_clicked(self):
        try:
            ip, port = discover_server(self.params)
            self.ip_editor.setText(ip)
            self.port_editor.setText(str(port))
            self.scan_button.setStyleSheet("background-color: green;")
        except Exception as e:
            self.scan_button.setStyleSheet("background-color: red;")
            logger.error(f"Failed to discover server: {e}")
        logger.debug("Scan button clicked")

    def _data_tree_view_update(self):
        """
        Updates the tree view with data from `shown_data_json`.
        """

        self.data_tree_view.clear()

        for key, value in self.collected_data.shown_data_json.items():
            root_id = self.data_tree_view.add_item(parent_item=None, text=key, level=1)
            prompt_id = self.data_tree_view.add_item(parent_item=root_id, text="Prompt", level=2, root_text=key)
            self.data_tree_view.add_item(parent_item=prompt_id, text=value["prompt"], level=3, root_text=key)
            bbox_id = self.data_tree_view.add_item(parent_item=root_id, text="Bbox", level=2, root_text=key)
            bbox_text = f"[{','.join(f'{v:.2f}' for v in value['bboxes'])}]"
            self.data_tree_view.add_item(parent_item=bbox_id, text=bbox_text, level=3, root_text=key)
            pose_id = self.data_tree_view.add_item(parent_item=root_id, text="Pose", level=2, root_text=key)
            
            for i, pose in enumerate(value["pose"]):
                pose_text = f"{i + 1}: [{','.join(f'{v:.2f}' for v in pose)}]"
                self.data_tree_view.add_item(
                    parent_item=pose_id,
                    text=pose_text,
                    level=3,
                    root_text=key
                )
        self.data_tree_view.expandAll()

    def on_data_tree_view_load_button_clicked(self):
        try:
            self.collected_data.load(self.data_folder_text.text())
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
        logger.debug("Data tree view load button clicked")

    def on_data_tree_view_remove_button_clicked(self):
        select_item = self.data_tree_view.selected_item
        if select_item != None:
            match select_item.level:
                case 1:
                    logger.debug(f"Removing {select_item.root_text}")
                    self.collected_data.pop(self.collected_data.dataids.index(select_item.root_text))
                case 2:
                    pass
                case 3:
                    logger.debug(f"Removing pose{select_item.root_text}-{select_item.index_in_level}")
                    if select_item.parent_text == "Pose":
                        self.collected_data.pop_pose(self.collected_data.dataids.index(select_item.root_text), 
                                                     select_item.index_in_level)
                        pass
        logger.debug("Data tree view remove button clicked")

    def on_tree_selection_changed(self, item, level, index_in_level, parent_text, root_text):
        """
        Callback for when the selection changes.
        """
        logger.debug(f"Selected Item: {item.text(0)}, Level: {level}, Index in Level: {index_in_level}, Parent Text: {parent_text}, Root Text: {root_text}")
        select_item = self.data_tree_view.selected_item
        self.prompt_text.setText(self.collected_data.shown_data_json.get(
                                select_item.root_text
                                ).get('prompt'))


    def on_data_folder_select_button_clicked(self):
        from PyQt5.QtWidgets import QFileDialog
        start_dir = self.params.get('data_path', './data')
        dir_text = QFileDialog.getExistingDirectory(
            directory=start_dir,
            options=QFileDialog.ShowDirsOnly
        )
        if not dir_text == "":
            self.data_folder_text.setText(dir_text)
        logger.debug("Data folder select button clicked")

    def on_data_save_button_clicked(self):
        self.collected_data.save(self.data_folder_text.text())
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
        self.T_base = np.eye(4)
        self.renderer.AddActor(self.robot_base_frame)

        self.robot_end_frame = vtkAxesActor()
        self.robot_end_frame.AxisLabelsOff()
        self.robot_end_frame.SetTotalLength(*size)
        self.T_End = np.eye(4)
        self.renderer.AddActor(self.robot_end_frame)

        self.board_pose_frame = vtkAxesActor()
        self.board_pose_frame.AxisLabelsOff()
        self.board_pose_frame.SetTotalLength(*size)
        self.T_Board = np.eye(4)
        self.renderer.AddActor(self.board_pose_frame)

        self.robot_base_frame.SetVisibility(0)
        self.robot_end_frame.SetVisibility(0)
        self.board_pose_frame.SetVisibility(0)

    def axis_set_to_matrix(self, axis:vtkAxesActor, matrix: np.ndarray):
        """Convert axis pose to matrix."""
        # Assuming axis pose is a 4x4 numpy array
        axis.SetUserMatrix(self._numpy_to_vtk_matrix_4x4(matrix, axis))

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

    def _numpy_to_vtk_matrix_4x4(self, matrix: np.ndarray):
        assert matrix.shape == (4, 4)
        vtk_matrix = vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_matrix.SetElement(i, j, matrix[i, j])
        return vtk_matrix


    def get_num_of_palette(self, num_colors):
        """Generate a color palette."""
        # For simplicity, generate random colors
        np.random.seed(0)
        palettes = np.random.randint(0, 256, size=(num_colors, 3))
        return palettes


    def load_in_startup(self):
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
        if hasattr(self, 'pcd_seg_model'):
            self.pcd_seg_model = None
        
        if hasattr(self, 'sendingThread'):
            self.sendingThread = None
        if hasattr(self, 'calibration_data'):
            self.calibration_data = None
        if hasattr(self, 'collected_data'):
            self.collected_data = None
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


class DataSendToServerThread(QThread):
    progress = pyqtSignal(tuple)  # Signal to communicate progress updates (step, progress)

    def __init__(self, ip, port, msg_dict: dict):
        self.ip = ip
        self.port = port
        self.msg_dict = msg_dict
        super().__init__()

    def run(self):
        try:
            logger.debug("Sending data to server")
            # Serialize the message dictionary
            data = pickle.dumps(self.msg_dict)
            data_length = len(data)

            with socket.create_connection((self.ip, int(self.port))) as client_socket:
                start_time = time.time()
                client_socket.sendall(data_length.to_bytes(4, byteorder='big'))
                self.progress.emit(("Sending Length", 100))  # Mark step as complete

                bytes_sent = 0
                chunk_size = 4096 
                while bytes_sent < data_length:
                    chunk = data[bytes_sent:bytes_sent + chunk_size]
                    sent = client_socket.send(chunk)
                    bytes_sent += sent
                    progress = (bytes_sent / data_length) * 100
                    self.progress.emit(("Sending Data", progress))

                # Ensure sending step reaches 100%
                self.progress.emit(("Sending Data", 100))

                # Step 3: Receiving the response length
                response_length = int.from_bytes(client_socket.recv(4), byteorder='big')
                self.progress.emit(("Receiving Length", 100))  # Mark step as complete

                # Step 4: Receiving the response data
                response_data = b""
                bytes_received = 0
                while len(response_data) < response_length:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    response_data += packet
                    bytes_received = len(response_data)
                    progress = (bytes_received / response_length) * 100
                    self.progress.emit(("Receiving Data", progress))

                # Ensure receiving step reaches 100%
                self.progress.emit(("Receiving Data", 100))
                # Deserialize the response
                self.__response = pickle.loads(response_data)
                end_time = time.time() - start_time
                logger.debug(f"Response from server: {self.__response}")
                logger.debug(f"Total time: {end_time:.2f} s")
        except Exception as e:
            print(f"An error occurred: {e}")
            self.progress.emit(("Error", 0))  # Emit an error state if something goes wrong
    
    def get_response(self):
        return self.__response
