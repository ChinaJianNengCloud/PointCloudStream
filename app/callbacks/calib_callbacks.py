import logging
import cv2
import json
import copy
import numpy as np

from typing import TYPE_CHECKING
from app.utils import RobotInterface, CameraInterface, ARUCO_BOARD
from app.utils import CalibrationData
from app.utils.pose import Pose
from app.threads.op_thread import RobotTcpOpThread

if TYPE_CHECKING:
    from app.entry import SceneStreamer

from app.utils.logger import setup_logger
logger = setup_logger(__name__)

def on_calib_combobox_changed(self: "SceneStreamer", text):
    if text != "":
        self.T_CamToBase = Pose.from_matrix(np.array(self.calib.get('calibration_results').get(text).get('transformation_matrix')))
    logger.debug(f"Calibration combobox changed: {text}")
    
def on_calib_data_list_changed(self: "SceneStreamer"):
        self.calib_data_list.clear()
        self.calib_data_list.addItems(self.calibration_data.display_str_list)
        logger.debug("Calibration data list changed")

def on_calib_check_button_clicked(self: "SceneStreamer"):
    try:
        on_cam_calib_init_button_clicked(self)
        path = self.params['calib_path']
        with open(path, 'r') as f:
            self.calib = json.load(f)
        intrinsic = np.array(self.calib.get('camera_matrix'))
        dist_coeffs = np.array(self.calib.get('dist_coeffs'))
        self.streamer.intrinsic_matrix = intrinsic
        self.streamer.dist_coeffs = dist_coeffs
        self.calibration_data.load_camera_parameters(intrinsic, dist_coeffs)
        self.calib_combobox.clear()
        self.calib_combobox.addItems(
            self.calib.get('calibration_results').keys())
        self.calib_combobox.setEnabled(True)
        self.calib_combobox.setCurrentIndex(0)
        curent_selected = self.calib_combobox.currentText()
        
        self.T_CamToBase = Pose.from_matrix(
            np.array(self.calib\
                     .get('calibration_results')\
                        .get(curent_selected)\
                            .get('transformation_matrix')))
        self.center_to_robot_base_toggle.setEnabled(True)
        
    except Exception as e:
        logger.error(f"Failed to load calibration data: {e}")
    logger.debug("Calibration check button clicked")

def on_cam_calib_init_button_clicked(self: "SceneStreamer"):
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
            self.calibration_data.data_changed.connect(lambda: on_calib_data_list_changed(self))
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


        if self.camera_interface is None:
            self.camera_interface = CameraInterface(self.streamer.camera, self.calibration_data)
        
        self.cam_calib_init_button.setStyleSheet("background-color: green;")
        self.set_enable_after_calib_init()
    except Exception as e:
        self.cam_calib_init_button.setStyleSheet("background-color: red;")
        logger.error(f"Failed to init camera calibration: {e}")
    
def on_calib_collect_button_clicked(self: "SceneStreamer"):
    if hasattr(self, 'calibration_data'):
        robot_pose = self.robot.get_state('tcp')
        color = self.current_frame['color']
        self.calibration_data.append(color, robot_pose=robot_pose)
    logger.debug("Calibration collect button clicked")

def on_calib_list_remove_button_clicked(self: "SceneStreamer"): 
    self.calibration_data.pop(self.calib_data_list.currentIndex().row())
    logger.debug(f"Calibration list remove button clicked")

def on_robot_move_button_clicked(self: "SceneStreamer"):
    idx = self.calib_data_list.currentIndex().row()
    try:
        self.robot.step(
            self.calibration_data.robot_poses[idx], action_type='tcp', wait=True)
        
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.calibration_data.modify(idx, self.current_frame['color'],
                                    self.robot.get_state('tcp'),
                                    )
        if hasattr(self, 'timer'):
            self.timer.start()
        logger.debug("Moving robot and collecting data")
    except:
        logger.error("Failed to move robot")


    logger.debug("Robot move button clicked")

def on_calib_button_clicked(self: "SceneStreamer"):
    self.calibration_data.calibrate_all()
    logger.debug("Calibration button clicked")

    def on_detect_board_toggle_state_changed(self):
        logger.debug("Detect board state changed to:")

def on_robot_init_button_clicked(self: "SceneStreamer"):
    self.robot =  RobotInterface(sim=True)
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


def on_detect_board_toggle_state_changed(self: "SceneStreamer"):
    logger.debug("Detect board state changed to:")


def on_show_axis_in_scene_button_clicked(self: "SceneStreamer"):
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


def on_calib_op_load_button_clicked(self: "SceneStreamer"):
    self.calibration_data.load_img_and_pose()
    logger.debug("Calibration operation load button clicked")


def on_calib_op_save_button_clicked(self: "SceneStreamer"):
    self.calibration_data.save_img_and_pose()
    logger.debug("Calibration operation save button clicked")


def on_calib_op_run_button_clicked(self: "SceneStreamer"):
    if self.robot is None:
        logger.error("Robot not initialized")
        return
    
    if not hasattr(self, 'current_frame'):
        logger.error("No current frame, please start streaming first.")
        return
    
    self.calib_thread = RobotTcpOpThread(self.robot, self.calibration_data.robot_poses)
    self.calib_thread.progress.connect(lambda value: update_progress(self, value))
    self.calib_thread.finished.connect(lambda: calibration_finished(self))
    self.calib_thread.start()
    logger.debug("Calibration operation run button clicked")

def on_calib_save_button_clicked(self: "SceneStreamer"):
    path = self.calib_save_text.text()
    self.calibration_data.save_calibration_data(path)
    logger.debug(f"Calibration saved: {path}")


def update_progress(self: "SceneStreamer", value):
    pose = self.robot.get_state('tcp')
    img = self.current_frame['color']
    self.calibration_data.modify(value, img, pose)
    logger.debug(f"Robot Move Progress: {value} and update calibration data")
    # self.label.setText(f"Progress: {value}")

def calibration_finished(self: "SceneStreamer"): 
    self.calibration_data.calibrate_all()
    logger.info("Calibration operation completed.")