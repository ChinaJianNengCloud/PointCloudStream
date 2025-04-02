import time
import numpy as np
import open3d as o3d
import cv2
from functools import partial
from typing import *

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QLabel, QDoubleSpinBox, 
                             QVBoxLayout, QHBoxLayout, QPushButton, 
                             QGroupBox, QSlider)

from scipy.spatial.transform import Rotation as R


from app.ui import PCDStreamerUI
from app.utils import CalibrationData, CollectedData, ConversationData
from app.utils.camera import segment_pcd_from_2d, CameraInterface
from app.utils.robot import RobotInterface
from app.viewers.pcd_viewer import Streamer, PCDUpdater, DualCamStreamer
from app.threads.op_thread import DataSendToServerThread, RobotTcpOpThread, RobotJointOpThread
from app.utils.camera.video_manager import VideoManager
from app.callbacks import (
    on_stream_init_button_clicked,        
    on_capture_toggle_state_changed,
    on_seg_model_init_toggle_state_changed,
    on_acq_mode_toggle_state_changed,
    on_display_mode_combobox_changed,
    on_center_to_robot_base_toggle_state_changed,

    # Calibration Tab
    on_cam_calib_init_button_clicked,
    on_robot_init_button_clicked,
    on_calib_collect_button_clicked,
    on_calib_button_clicked,
    on_detect_board_toggle_state_changed,
    on_show_axis_in_scene_button_clicked,
    on_calib_list_remove_button_clicked,
    on_robot_move_button_clicked,
    on_calib_op_load_button_clicked,
    on_calib_op_save_button_clicked,
    on_calib_op_run_button_clicked,
    on_calib_save_button_clicked,
    on_calib_check_button_clicked,
    on_calib_combobox_changed,

    # Data Tab
    on_data_collect_button_clicked,
    on_data_save_button_clicked,
    on_data_tree_view_load_button_clicked,
    on_data_folder_select_button_clicked,
    on_data_tree_view_remove_button_clicked,
    on_tree_selection_changed,
    on_data_tree_changed,
    on_data_replay_and_save_button_clicked,

    # Agent Tab
    on_scan_button_clicked,
    on_send_button_clicked,
)
from app.utils.pose import Pose
# from app.utils.robot.matrix_pose_op import *

from app.utils.logger import setup_logger
logger = setup_logger(__name__)

class PCDStreamer(PCDStreamerUI):
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing  and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """
    def __init__(self, params:Dict=None):
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
        self.timer = QTimer()
        self.frame_num = 0
        self.real_fps = 0
        self.frame = None
        self.streamer = DualCamStreamer(params=params)
        # self.pcd_updater = PCDUpdater(self.renderer)
        self.video_manager = VideoManager()
        self.robot: RobotInterface = None
        self.calib_thread: RobotTcpOpThread = None
        self.robot_joint_thread: "RobotJointOpThread" = None
        self.collected_data = CollectedData(self.params.get('data_path', './data'))
        self.calibration_data: CalibrationData = None
        self.robot: RobotInterface = None
        self.camera_interface: CameraInterface = None
        self.T_CamToBase: Pose = None
        self.T_BaseToCam: Pose = None
        self.pcd_seg_model = None
        self.calib: Dict = None

        self.palettes = self.get_num_of_palette(80)
        self.conversation_data = ConversationData()
        self.sendingThread: DataSendToServerThread = None
        self.__init_signals()
        self.__init_ui_values_from_params()
        self.__callback_bindings()
        self.set_disable_before_stream_init()

    def update_cam_ids(self):
        logger.info(f"Main camera: {self.main_camera_combobox.currentText()}")
        logger.info(f"Sub camera: {self.sub_camera_combobox.currentText()}")
        self.params['main_camera_id'] = self.video_manager.get_id_by_name(self.main_camera_combobox.currentText())
        self.params['sub_camera_id'] = self.video_manager.get_id_by_name(self.sub_camera_combobox.currentText())


    @property
    def T_CamToBase(self) -> Pose:
        return self._T_CamToBase
    
    @T_CamToBase.setter
    def T_CamToBase(self, value: Pose):
        if value is not None:
            self._T_CamToBase = value
            self.T_BaseToCam = self._T_CamToBase.inv()


    def __init_ui_values_from_params(self):
        cams = self.video_manager.devices + [{'name': 'Fake Camera', 'index': -1}]
        self.main_camera_combobox.addItems([d['name'] for d in cams])
        self.main_camera_combobox.currentTextChanged.connect(self.update_sub_camera_options)
        self.update_sub_camera_options()
        self.update_cam_ids()
        self.calib_save_text.setText(self.params.get('calib_path', "Please_set_calibration_path"))
        self.board_col_num_edit.setValue(self.params.get('board_shape', (11, 6))[0])
        self.board_row_num_edit.setValue(self.params.get('board_shape', (11, 6))[1])
        self.board_square_size_num_edit.setValue(self.params.get('board_square_size', 0.023))
        self.board_marker_size_num_edit.setValue(self.params.get('board_marker_size', 0.0175))
        self.board_type_combobox.setCurrentText(self.params.get('board_type', "DICT_4X4_100"))
        del self.robot, self.calib_thread, self.calibration_data

    def __init_signals(self):
        self.streaming = False
        self.show_axis = False

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

    def __callback_bindings(self):
        # Binding callbacks to GUI elements
        # General Tab
        self.main_init_button.clicked.connect(partial(on_stream_init_button_clicked, self))
        self.sub_camera_combobox.currentTextChanged.connect(self.update_cam_ids)
        # View Tab
        self.capture_toggle.stateChanged.connect(partial(on_capture_toggle_state_changed, self))
        self.seg_model_init_toggle.stateChanged.connect(partial(on_seg_model_init_toggle_state_changed, self))
        self.acq_mode_toggle.stateChanged.connect(partial(on_acq_mode_toggle_state_changed, self))
        self.display_mode_combobox.currentTextChanged.connect(partial(on_display_mode_combobox_changed, self))
        self.center_to_robot_base_toggle.stateChanged.connect(partial(on_center_to_robot_base_toggle_state_changed, self))

        # Calibration Tab
        self.cam_calib_init_button.clicked.connect(partial(on_cam_calib_init_button_clicked, self))
        self.robot_init_button.clicked.connect(partial(on_robot_init_button_clicked, self))
        self.calib_collect_button.clicked.connect(partial(on_calib_collect_button_clicked, self))
        self.calib_button.clicked.connect(partial(on_calib_button_clicked, self))
        self.detect_board_toggle.stateChanged.connect(partial(on_detect_board_toggle_state_changed, self))
        self.show_axis_in_scene_button.clicked.connect(partial(on_show_axis_in_scene_button_clicked, self))
        self.calib_list_remove_button.clicked.connect(partial(on_calib_list_remove_button_clicked, self))
        self.robot_move_button.clicked.connect(partial(on_robot_move_button_clicked, self))
        self.calib_op_load_button.clicked.connect(partial(on_calib_op_load_button_clicked, self))
        self.calib_op_save_button.clicked.connect(partial(on_calib_op_save_button_clicked, self))
        self.calib_op_run_button.clicked.connect(partial(on_calib_op_run_button_clicked, self))
        self.calib_save_button.clicked.connect(partial(on_calib_save_button_clicked, self))
        self.calib_check_button.clicked.connect(partial(on_calib_check_button_clicked, self))
        self.calib_combobox.currentTextChanged.connect(partial(on_calib_combobox_changed, self))

        # Data Tab
        self.data_collect_button.clicked.connect(partial(on_data_collect_button_clicked, self))
        self.data_save_button.clicked.connect(partial(on_data_save_button_clicked, self))
        self.data_tree_view_load_button.clicked.connect(partial(on_data_tree_view_load_button_clicked, self))
        self.data_folder_select_button.clicked.connect(partial(on_data_folder_select_button_clicked, self))
        self.data_tree_view_remove_button.clicked.connect(partial(on_data_tree_view_remove_button_clicked, self))
        self.data_tree_view.set_on_selection_changed(partial(on_tree_selection_changed, self))
        self.collected_data.data_changed.connect(partial(on_data_tree_changed, self))
        self.data_replay_and_save_button.clicked.connect(partial(on_data_replay_and_save_button_clicked, self))

        # Agent Tab
        self.scan_button.clicked.connect(partial(on_scan_button_clicked, self))
        self.send_button.clicked.connect(partial(on_send_button_clicked, self))


    def frame_calling(self):
        current_frame_time = time.time()
        if self.frame_num % 10 == 0:  # Avoid calculation on the first frame
            time_diff = current_frame_time - self.prev_frame_time
            if time_diff > 0:  # Avoid division by zero
                self.real_fps = 10*(1.0 / time_diff)
            self.prev_frame_time = current_frame_time     
        frame_elements = {
            'fps': round(self.real_fps, 2),  # Assuming 30 FPS for fake camera
        }
        frame_elements.update(self.streamer.get_frame(take_pcd=True))

        self.board_pose_update(frame_elements)
        self.segment_pcd_from_yolo(frame_elements)
        self.elements_update(frame_elements)
        
        self.current_frame = frame_elements
            
    

    def get_robot_pose(self) -> Tuple[bool, Pose]:
        """
        Get the current pose of the robot's end-effector.
        
        Returns:
            bool: Whether the robot pose was successfully retrieved.
            list or None: The pose as a list of length 6, containing the translation (x, y, z) and rotation (rx, ry, rz) components in the camera frame if robot calibration data is available.
            np.ndarray or None: The pose as a 4x4 numpy array in the camera frame if robot calibration data is available.
        """
        if self.robot is None:
            logger.error("Robot not initialized.")
            return False, None
        
        robot_pose = self.robot.capture_gripper_to_base(sep=False)
        base_to_end = Pose.from_1d_array(vector=robot_pose, vector_type="euler", degrees=False)

        if not self.center_to_robot_base_toggle.isChecked():
            if self.T_CamToBase is not None:
                cam_to_end = base_to_end.apply_delta_pose(self.T_BaseToCam, on="base")
                return True, cam_to_end
            else:
                logger.error("No robot calibration data detected.")
                return False, None
        else:
            return True, base_to_end

    def board_pose_update(self, frame):
        pass
        # if self.camera_interface is not None and self.detect_board_toggle.isChecked():
        #     rgb_with_pose, rvec, tvec = self.camera_interface._process_and_display_frame(
        #         self._img_to_array(frame['color']), ret_vecs=True)
        #     if rvec is None or tvec is None:
        #         logger.warning("Failed to detect board.")
        #     else:
        #         cam_to_board = Pose.from_1d_array(np.hstack([tvec.ravel(), rvec.ravel()]), 
        #                                           vector_type="rotvec", degrees=False)
        #         if self.center_to_robot_base_toggle.isChecked():
        #             if self.T_CamToBase is not None:
        #                 base_to_board = cam_to_board.apply_delta_pose(self.T_CamToBase, on="base")
        #                 self.board_pose_frame.SetUserMatrix(base_to_board.vtk_matrix)
        #         else:
        #             self.board_pose_frame.SetUserMatrix(cam_to_board.vtk_matrix)
    
    

    def elements_update(self, frame_elements: dict):
        """Update visualization with point cloud and images."""

        if 'color' in frame_elements:
            self.update_color_image(frame_elements['color'])

        if 'depth' in frame_elements:
            self.update_depth_image(frame_elements['depth'])

        if 'main' in frame_elements:
            self.update_main_cam_image(frame_elements['main'])

        if 'sub' in frame_elements:
            self.update_sub_cam_image(frame_elements['sub'])

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


        # logger.debug("Point cloud visualization updated.")

    def segment_pcd_from_yolo(self, frame: dict):
        if self.seg_model_init_toggle.isChecked():
            if self.pcd_seg_model is None:
                logger.error("Segmentation model not initialized")
                return
            try:
                labels = segment_pcd_from_2d(self.pcd_seg_model, 
                                    frame['pcd'], frame['color'], 
                                    self.streamer.intrinsic_matrix, 
                                    self.streamer.extrinsics)
            except Exception as e:
                logger.error(f"Segmentation failed: {e}")
                return
            frame['seg_labels'] = labels



    def on_key_press(self, obj, event):
        key = obj.GetKeySym()
        if key == 'space':
            logger.info("Space key pressed")
            pass  # Handle space key press if needed

    @staticmethod
    def _img_to_array(image: Union[np.ndarray, o3d.geometry.Image, o3d.t.geometry.Image]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, o3d.geometry.Image):
            return np.asarray(image)
        elif isinstance(image, o3d.t.geometry.Image):
            return np.asarray(image.cpu())
    
    def _update_image(self, image, label: QLabel):
        """Update the image display in a QLabel."""
        image = self._img_to_array(image)
        # Convert image to QImage and display in QLabel
        if image.shape[2] == 3:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            label.setPixmap(pixmap.scaled(label.size(), 
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))


    def update_color_image(self, color_image):
        """Update the color image display."""
        if self.color_groupbox.isChecked():
            self._update_image(color_image, self.color_video)

    def update_depth_image(self, depth_image):
        """Update the depth image display."""
        if self.depth_groupbox.isChecked():
            self._update_image(depth_image, self.depth_video)

    def update_main_cam_image(self, image):
        self._update_image(image, self.viewer.main_cam)

    def update_sub_cam_image(self, image):
        self._update_image(image, self.viewer.sub_cam)

    def set_predict_pose(self, pose: np.ndarray):
        pose_p = Pose.from_1d_array(pose[:6], vector_type="euler", degrees=False)
        print(f"custom pose: {pose_p}")


    def view_predicted_poses(self, poses:np.ndarray):
        ret, robot_pose, _ = self.get_robot_pose()
        if not ret:
            logger.error("Failed to get robot pose.")
            return
        if len(poses) < 1:
            logger.error("Warning: At least one relative pose is required for visualization.")
            return

        base_pose = np.array(robot_pose[:6])  # Extract (x, y, z) from current_pose
        previous_pose = Pose.from_1d_array(base_pose, vector_type="euler", degrees=False)
        realpose = []
        # Add relative poses incrementally
        for i, pose in enumerate(poses):
            if pose[6] == 1:
                break
            # dx, dy, dz, drx, dry, drz = pose[:6]  # Extract relative (x, y, z) changes
            # current_pose = previous_pose + np.array([dx, dy, dz, drx, dry, drz])
            delta_pose = Pose.from_1d_array(pose[:6], vector_type="euler", degrees=False)
            current_pose = previous_pose.apply_delta_pose(delta_pose, on="align").to_1d_array(vector_type="euler", degrees=False)
            realpose.append(current_pose)
            previous_pose = current_pose

        return realpose

    def transform_poses(self, poses:np.ndarray, transform_pose: Pose):
        transformed_poses = []
        for pose in poses:
            pose_p = Pose.from_1d_array(pose[:6], vector_type="euler", degrees=False)
            transformed_pose = pose_p.apply_delta_pose(transform_pose, on="base")
            transformed_poses.append(transformed_pose.to_1d_array(vector_type="euler", degrees=False))
        return transformed_poses

    @staticmethod
    def get_num_of_palette(num_colors):
        """Generate a color palette."""
        # For simplicity, generate random colors
        np.random.seed(0)
        palettes = np.random.randint(0, 256, size=(num_colors, 3))
        return palettes

        
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
        if hasattr(self, 'image_dialog'):
            self.image_dialog = None
        self.streamer = None
        self.current_frame = None
        super().closeEvent(event) 

    def update_sub_camera_options(self):
        # Get the currently selected main camera
        main_camera = self.main_camera_combobox.currentText()
        
        # Clear current items in sub camera combobox
        self.sub_camera_combobox.clear()
        
        # Get all available cameras
        cams = self.video_manager.devices
        
        # Add all cameras except the one selected in main combobox
        sub_cameras = [d['name'] for d in cams if d['name'] != main_camera] + ['Fake Camera']
        self.sub_camera_combobox.addItems(sub_cameras)
        
        # Update the camera IDs in params
        self.update_cam_ids()
