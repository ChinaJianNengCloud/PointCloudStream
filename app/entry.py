import time
import numpy as np
from functools import partial
from typing import List, Dict, Any

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QLabel, QListWidgetItem)
from app.ui import SceneStreamerUI
from app.utils import CalibrationData, CollectedData, ConversationData
from app.utils.camera import CameraInterface
from app.utils.robot import RobotInterface
from app.viewers.scene_viewer import MultiCamStreamer
from app.threads.op_thread import DataSendToServerThread, RobotTcpOpThread, RobotJointOpThread
from app.utils.camera.device.usb_camera_parser import USBVideoManager
from app.callbacks import (
    on_stream_init_button_clicked,        
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
import logging
logger = logging.getLogger(__name__)

class SceneStreamer(SceneStreamerUI):
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing  and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """
    def __init__(self, params:Dict=None):
        """Initialize the SceneStreamer application.

        Args:
            params (Dict): Parameters for the app.
        """
        super().__init__()
        # Core parameters and state
        self.params = params
        self.frame_num = 0
        self.real_fps = 0
        self.prev_frame_time = time.time()
        self.streaming = False
        self.current_frame = None
        self.frame = None
        
        # Core components
        self.streamer = MultiCamStreamer(params=params)
        self.video_manager = USBVideoManager()
        self.timer = QTimer()
        
        # Data management
        self.collected_data = CollectedData(self.params.get('data_path', './data'))
        self.conversation_data = ConversationData()
        
        # Display components
        self.rectangle_geometry = None
        self.palettes = self.get_num_of_palette(80)
        
        # Robot and calibration components
        self.robot = None
        self.calib_thread = None
        self.robot_joint_thread = None
        self.calibration_data = None
        self.camera_interface = None
        self.pcd_seg_model = None
        self.calib = None
        
        # Network components
        self.sendingThread = None
        
        # Initialize UI and callbacks
        self.__init_ui_values_from_params()
        self.__callback_bindings()
        self.set_disable_before_stream_init()

    #-------------------------------------------------------------------------
    # Camera Management Methods
    #-------------------------------------------------------------------------
    
    def update_cam_ids(self):
        """Update camera list in params from the UI list widget"""
        camera_list = []
        for i in range(self.camera_list_widget.count()):
            item = self.camera_list_widget.item(i)
            camera_data = item.data(Qt.ItemDataRole.UserRole)
            if camera_data:
                # Ensure camera data has all necessary fields
                camera_entry = {
                    'id': camera_data.get('id', -1),
                    'name': camera_data.get('name', f'camera_{i}'),
                    'device_name': camera_data.get('device_name', '')
                }
                
                # Add HTTP URL for HTTP cameras (ID 99)
                if camera_entry['id'] == 99:
                    camera_entry['http_url'] = camera_data.get('http_url', '')
                    
                camera_list.append(camera_entry)
        
        self.params['camera_list'] = camera_list
        return True

    def update_sub_camera_options(self):
        """Update the sub-camera dropdown options based on the main camera selection."""
        main_camera = self.main_camera_combobox.currentText()
        self.sub_camera_combobox.clear()
        cams = self.video_manager.devices
        sub_cameras = [d['name'] for d in cams if d['name'] != main_camera] + ['Fake Camera']
        self.sub_camera_combobox.addItems(sub_cameras)
        self.update_cam_ids()

    def add_camera_to_list(self):
        """Add a camera to the camera list widget based on selected camera type."""
        # Get camera type from dropdown
        camera_type = self.camera_type_combobox.currentText()
        
        # Set camera ID and details based on camera type
        if camera_type == "HTTP Camera":
            camera_id = 99  # Special ID for HTTP cameras
            device_name = "HTTP Camera"
            http_url = self.http_camera_url.text().strip()
            
            if not http_url:
                logger.error("HTTP camera URL cannot be empty")
                return
        else:  # USB Camera
            device_name = self.camera_combobox.currentText()
            camera_id = self.video_manager.get_id_by_name(device_name)
            http_url = None
        name_by_user = self.camera_name_input.text().strip()
        if not name_by_user:
            name_by_user = "Unnamed"

        existing_names = []
        for i in range(self.camera_list_widget.count()):
            item = self.camera_list_widget.item(i)
            camera_data = item.data(Qt.ItemDataRole.UserRole)
            if camera_data:
                existing_names.append(camera_data['name'])

        original_name = name_by_user
        counter = 1
        while name_by_user in existing_names:
            name_by_user = f"{original_name}_{counter}"
            counter += 1

        camera_item = {
            'id': camera_id, 
            'name': name_by_user, 
            'device_name': device_name,
            'http_url': http_url
        }

        item = QListWidgetItem(f"{self.camera_list_widget.count()}: {camera_item['device_name']} -- {camera_item['name']}")
        item.setData(Qt.ItemDataRole.UserRole, camera_item)
        self.camera_list_widget.addItem(item)
        
        # Clear the input fields after adding
        self.camera_name_input.clear()
        if camera_type == "HTTP Camera":
            self.http_camera_url.clear()
        
        # Update the camera list in params
        self.update_cam_ids()

    def remove_camera_from_list(self):
        """Remove the selected camera from the camera list widget."""
        current_row = self.camera_list_widget.currentRow()
        if current_row != -1:
            self.camera_list_widget.takeItem(current_row)

    def clear_camera_list(self):
        """Clear all cameras from the camera list widget."""
        self.camera_list_widget.clear()

    #-------------------------------------------------------------------------
    # Frame Processing Methods
    #-------------------------------------------------------------------------
    
    def frame_calling(self):
        """Process frames and update UI elements with new frame data."""
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
        self.elements_update(frame_elements)
        
        self.current_frame = frame_elements

    def elements_update(self, frame_elements: dict):
        """Update visualization with point cloud and images."""
        scene_viewer = self.viewer
        if scene_viewer:
            for camera_name, frame in frame_elements.items():
                # Skip non-camera keys
                if camera_name in ['status_message', 'fps', 'robot_pose']:
                    continue
                
                # Get the camera widget and update it
                if camera_name in scene_viewer.camera_widgets:
                    camera_info = scene_viewer.camera_widgets[camera_name]
                    camera_widget = camera_info['widget']
                    self.update_widget_with_image(camera_widget, frame)

        if 'status_message' in frame_elements:
            self.status_message.setText(frame_elements['status_message'])

        if 'fps' in frame_elements:
            fps = frame_elements["fps"]
            self.fps_label.setText(f"FPS: {int(fps)}")

        self.frame_num += 1
    
    def _update_image(self, image: np.ndarray, label: QLabel):
        """Update the image display in a QLabel."""
        # Convert image to QImage and display in QLabel
        if image.shape[2] == 3:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            label.setPixmap(pixmap.scaled(label.size(), 
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))

    def update_widget_with_image(self, widget, image):
        """Update any widget with an image"""
        if image is None:
            return
        
        h, w, c = image.shape
        bytes_per_line = c * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        widget.setPixmap(QPixmap.fromImage(q_img))

    def __init_ui_values_from_params(self):
        """Initialize UI values from parameters."""
        # Get available cameras
        cams = self.video_manager.devices + [{'name': 'Fake Camera', 'id': -1}]
        
        # Add cameras to the dropdown
        self.camera_combobox.clear()
        self.camera_combobox.addItems([d['name'] for d in cams])
        
        # Initialize camera list from params
        camera_list:List[Dict[str, Any]] = self.params.get('camera_list', [])

        # Populate camera list widget
        self.camera_list_widget.clear()
        for i, camera in enumerate(camera_list):
            # Get camera details
            cam_id = camera.get('id', -1)
            
            # Set device name based on camera type
            if cam_id == 99:  # HTTP camera
                camera_list[i]['device_name'] = "HTTP Camera"
                camera_list[i]['http_url'] = camera.get('http_url', '')
            elif cam_id == -1:
                camera_list[i]['device_name'] = 'Fake Camera'
            else:
                camera_list[i]['device_name'] = self.video_manager.get_name_by_id(cam_id)

            cam_name = camera.get('name', 'Unknown')
            device_name = camera.get('device_name', '')
            
            # Format item text using the new format
            item_text = f"{i}: {device_name} -- {cam_name}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, camera)
            self.camera_list_widget.addItem(item)

        self.params['camera_list'] = camera_list
        # Connect camera management signals
        self.add_camera_button.clicked.connect(self.add_camera_to_list)
        self.remove_camera_button.clicked.connect(self.remove_camera_from_list)
        self.clear_cameras_button.clicked.connect(self.clear_camera_list)
        
        # For backward compatibility with old code
        self.update_sub_camera_options()
        self.update_cam_ids()
        
        # Initialize other UI values
        self.calib_save_text.setText(self.params.get('calib_path', "Please_set_calibration_path"))
        self.board_col_num_edit.setValue(self.params.get('board_shape', (11, 6))[0])
        self.board_row_num_edit.setValue(self.params.get('board_shape', (11, 6))[1])
        self.board_square_size_num_edit.setValue(self.params.get('board_square_size', 0.023))
        self.board_marker_size_num_edit.setValue(self.params.get('board_marker_size', 0.0175))
        self.board_type_combobox.setCurrentText(self.params.get('board_type', "DICT_4X4_100"))

    def __callback_bindings(self):
        """Bind callbacks to UI elements."""
        # General Tab
        self.main_init_button.clicked.connect(partial(on_stream_init_button_clicked, self))
        self.sub_camera_combobox.currentTextChanged.connect(self.update_cam_ids)

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

    def on_key_press(self, obj, event):
        """Handle key press events in the viewer."""
        key = obj.GetKeySym()
        if key == 'space':
            logger.info("Space key pressed")
            pass  # Handle space key press if needed

    def set_predict_pose(self, pose: np.ndarray):
        """Set a predicted pose."""
        pose_p = Pose.from_1d_array(pose[:6], vector_type="euler", degrees=False)
        print(f"custom pose: {pose_p}")

    def view_predicted_poses(self, poses:np.ndarray):
        """Transform predicted poses relative to current robot position."""
        robot_pose = self.robot.get_state('tcp')
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
        """Apply a transformation to a series of poses."""
        transformed_poses = []
        for pose in poses:
            pose_p = Pose.from_1d_array(pose[:6], vector_type="euler", degrees=False)
            transformed_pose = pose_p.apply_delta_pose(transform_pose, on="base")
            transformed_poses.append(transformed_pose.to_1d_array(vector_type="euler", degrees=False))
        return transformed_poses

    #-------------------------------------------------------------------------
    # UI State Management Methods
    #-------------------------------------------------------------------------
    
    def set_disable_before_stream_init(self):
        """Disable UI elements before stream initialization."""
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
        """Enable UI elements after stream initialization."""
        self.data_collect_button.setEnabled(True)
        self.calib_collect_button.setEnabled(True)

    def set_enable_after_calib_init(self):
        """Enable UI elements after calibration initialization."""
        self.calib_button.setEnabled(True)
        self.detect_board_toggle.setEnabled(True)
        self.show_axis_in_scene_button.setEnabled(True)
        self.calib_list_remove_button.setEnabled(True)
        self.robot_move_button.setEnabled(True)
        self.calib_op_save_button.setEnabled(True)
        self.calib_op_load_button.setEnabled(True)
        self.calib_op_run_button.setEnabled(True)
    
    @staticmethod
    def get_num_of_palette(num_colors):
        """Generate a color palette."""
        # For simplicity, generate random colors
        np.random.seed(0)
        palettes = np.random.randint(0, 256, size=(num_colors, 3))
        return palettes
        
    def closeEvent(self, event):
        """Ensure cleanup when the main window exits."""
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
