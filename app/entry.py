
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


from vtkmodules.vtkRenderingCore import vtkActor
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkLine, vtkCellArray
from vtkmodules.vtkCommonCore import vtkEventDataAction

from app.ui import PCDStreamerUI
from app.utils import CalibrationData, CollectedData, ConversationData
from app.utils.camera import segment_pcd_from_2d, CameraInterface
from app.utils.robot import RobotInterface
from app.viewers.pcd_viewer import PCDStreamerFromCamera, PCDUpdater
from app.threads.op_thread import DataSendToServerThread, RobotTcpOpThread
from app.callbacks import *
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
        self.streamer = PCDStreamerFromCamera(params=params)
        self.pcd_updater = PCDUpdater(self.renderer)
        self.robot: RobotInterface = None
        self.calib_thread: RobotTcpOpThread = None
        self.robot_joint_thread: RobotJointOpThread = None
        self.collected_data = CollectedData(self.params.get('data_path', './data'))
        self.calibration_data: CalibrationData = None
        self.robot: RobotInterface = None
        self.camera_interface: CameraInterface = None
        self.T_CamToBase: Pose = None
        self.T_BaseToCam: Pose = None
        self.pcd_seg_model = None
        self.calib: Dict = None
        self.streamer.camera_frustrum.register_renderer(self.renderer)
        self.palettes = self.get_num_of_palette(80)
        self.conversation_data = ConversationData()
        self.bbox_actor: vtkActor = None
        self.sendingThread: DataSendToServerThread = None
        self.__init_scene_objects()
        self.__init_bbox()
        self.__init_signals()
        self.__init_ui_values_from_params()
        self.__callback_bindings()
        self.set_disable_before_stream_init()

    @property
    def T_CamToBase(self) -> Pose:
        return self._T_CamToBase
    
    @T_CamToBase.setter
    def T_CamToBase(self, value: Pose):
        if value is not None:
            self._T_CamToBase = value
            self.T_BaseToCam = self._T_CamToBase.inv()

    def __init_ui_values_from_params(self):
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

    def init_bbox_controls(self, layout:QVBoxLayout):
        self.bbox_groupbox = QGroupBox("Bounding Box Controls")
        group_layout = QVBoxLayout()
        self.bbox_groupbox.setLayout(group_layout)
        layout.addWidget(self.bbox_groupbox)

        self.bbox_sliders = {}
        self.bbox_edits = {}
        bbox_params = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for param in bbox_params:
            h_layout = QHBoxLayout()
            label = QLabel(param)
            h_layout.addWidget(label)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-500, 500)
            h_layout.addWidget(slider)
            spin_box = QDoubleSpinBox()
            spin_box.setRange(-5.0, 5.0)
            h_layout.addWidget(spin_box)
            group_layout.addLayout(h_layout)
            self.bbox_sliders[param] = slider
            self.bbox_edits[param] = spin_box

        h_layout = QHBoxLayout()
        group_layout.addLayout(h_layout)
        self.save_bbox_button = QPushButton("Save")
        h_layout.addWidget(self.save_bbox_button)
        self.load_bbox_button = QPushButton("Load")
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

    def __callback_bindings(self):
        # Binding callbacks to GUI elements
        # General Tab
        self.stream_init_button.clicked.connect(partial(on_stream_init_button_clicked, self))

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

        # Key press events
        self.vtk_widget.AddObserver("KeyPressEvent", self.on_key_press)


    def set_vtk_camera_from_intrinsics(self, intrinsic_matrix:np.ndarray, extrinsics):
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
        frame_elements = {
            'fps': round(self.real_fps, 2),  # Assuming 30 FPS for fake camera
        }
        frame_elements.update(self.streamer.get_frame(take_pcd=True))

        self.board_pose_update(frame_elements)
        self.robot_pose_update(frame_elements)
        self.segment_pcd_from_yolo(frame_elements)
        self.point_cloud_update(frame_elements)
        
        self.current_frame = frame_elements

    def robot_pose_update(self, frame):
        if self.show_axis:
            ret, pose = self.get_robot_pose()
            if ret:
                self.robot_end_frame.SetUserMatrix(pose.vtk_matrix)
                if not self.center_to_robot_base_toggle.isChecked():
                    self.robot_base_frame.SetUserMatrix(self.T_BaseToCam.vtk_matrix)
                else:
                    self.robot_base_frame.SetUserMatrix(Pose.from_1d_array(np.zeros(6)).vtk_matrix)
            

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
        if self.camera_interface is not None and self.detect_board_toggle.isChecked():
            rgb_with_pose, rvec, tvec = self.camera_interface._process_and_display_frame(
                self._img_to_array(frame['color']), ret_vecs=True)
            if rvec is None or tvec is None:
                logger.warning("Failed to detect board.")
            else:
                cam_to_board = Pose.from_1d_array(np.hstack([tvec.ravel(), rvec.ravel()]), 
                                                  vector_type="rotvec", degrees=False)
                if self.center_to_robot_base_toggle.isChecked():
                    if self.T_CamToBase is not None:
                        base_to_board = cam_to_board.apply_delta_pose(self.T_CamToBase, on="base")
                        self.board_pose_frame.SetUserMatrix(base_to_board.vtk_matrix)
                else:
                    self.board_pose_frame.SetUserMatrix(cam_to_board.vtk_matrix)
    
    def point_cloud_update(self, frame_elements: dict):
        """Update visualization with point cloud and images."""

        if 'pcd' in frame_elements:
            if 'seg_labels' in frame_elements and self.display_mode_combobox.currentText() == "Segmentation":
                seg_labels = frame_elements['seg_labels']
                num_points = seg_labels.shape[0]
                colors = np.zeros((num_points, 3))
                valid_idx = seg_labels >= 0
                colors[valid_idx] = np.array(self.palettes)[seg_labels[valid_idx].reshape(-1)] / 255.0
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

    def update_color_image(self, color_image):
        """Update the color image display."""
        if self.color_groupbox.isChecked():
            image = self._img_to_array(color_image)
            # Convert color_image to QImage and display in QLabel
            if image.shape[2] == 3:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                
                q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.color_video.setPixmap(pixmap.scaled(self.color_video.size(), 
                                                         Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation))

    def update_depth_image(self, depth_image):
        """Update the depth image display."""
        if self.depth_groupbox.isChecked():
            image = self._img_to_array(depth_image)
            # Convert depth_image to QImage and display in QLabel
            if image.shape[2] == 3:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                self.depth_video.setPixmap(pixmap.scaled(self.depth_video.size(), 
                                                         Qt.AspectRatioMode.KeepAspectRatio,
                                                         Qt.TransformationMode.SmoothTransformation))


    def __init_scene_objects(self):
        """Initialize scene objects in the VTK renderer."""
        # Robot base frame
        size = [0.06] * 3
        colors = vtkNamedColors()
        self.robot_base_frame = vtkAxesActor()
        self.robot_base_frame.AxisLabelsOff()
        self.robot_base_frame.SetTotalLength(*size)
        self.renderer.AddActor(self.robot_base_frame)

        self.robot_end_frame = vtkAxesActor()
        self.robot_end_frame.AxisLabelsOff()
        self.robot_end_frame.SetTotalLength(*size)
        self.renderer.AddActor(self.robot_end_frame)

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
        update_bounding_box(self)

        # Set up callbacks for bbox sliders and edits
        bbox_params = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for param in bbox_params:
            # Assuming sliders and spin boxes are named accordingly
            slider: QSlider = self.bbox_sliders[param]
            spin_box: QDoubleSpinBox = self.bbox_edits[param]
            slider.setValue(int(self.bbox_params[param] * 100))
            spin_box.setValue(int(self.bbox_params[param]))

            slider.valueChanged.connect(lambda value, p=param: on_bbox_slider_changed(self, value, p))
            spin_box.valueChanged.connect(lambda value, p=param: on_bbox_edit_changed(self, value, p))

    def set_predict_pose(self, pose: np.ndarray):
        pose_p = Pose.from_1d_array(pose[:6], vector_type="euler", degrees=False)
        if hasattr(self, 'predicted_pose_axes'):
            self.predicted_pose_axes = vtkAxesActor()
            self.predicted_pose_axes.SetXAxisLabelText("")
            self.predicted_pose_axes.SetYAxisLabelText("Predicted")
            self.predicted_pose_axes.SetZAxisLabelText("")
            self.predicted_pose_axes.GetYAxisCaptionActor2D().SetWidth(0.1)
            self.predicted_pose_axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 1)  # Green color
            self.predicted_pose_axes.SetTotalLength(0, 0, 0.3)

        self.predicted_pose_axes.SetUserMatrix(pose_p.vtk_matrix)
        self.renderer.AddActor(self.predicted_pose_axes)
        self.vtk_widget.GetRenderWindow().Render()
        print(f"Added custom pose: {pose}")


    def view_predicted_poses(self, poses:np.ndarray):
        """
        Visualize a sequence of predicted relative poses starting from self.current_pose using VTK.
        
        Args:
            poses (list of tuples): List of (dx, dy, dz) relative poses representing incremental offsets.
        """
        ret, robot_pose, _ = self.get_robot_pose()
        if not ret:
            logger.error("Failed to get robot pose.")
            return
        if len(poses) < 1:
            logger.error("Warning: At least one relative pose is required for visualization.")
            return

        if not hasattr(self, 'renderer') or self.renderer is None:
            raise ValueError("Renderer is not initialized. Please initialize 'self.renderer' before calling this method.")
        
        # Clear previous pose actors if they exist
        if not hasattr(self, 'pose_actors'):
            self.pose_actors = []
        
        for actor in self.pose_actors:
            self.renderer.RemoveActor(actor)
        self.pose_actors.clear()

        # Start with the current pose
        base_pose = np.array(robot_pose[:6])  # Extract (x, y, z) from current_pose

        points = vtkPoints()
        lines = vtkCellArray()
        points.InsertNextPoint(*base_pose[:3])

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
            points.InsertNextPoint(*current_pose[:3])
            
            # Add a line between the previous and current pose
            line = vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)
            previous_pose = current_pose

        poly_data = vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        # Create an actor
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # Red color for the path
        actor.GetProperty().SetLineWidth(2)    # Make lines thicker for better visibility

        # Add new actor to the renderer and track it
        self.renderer.AddActor(actor)
        self.pose_actors.append(actor)

        # Render the updated scene
        self.renderer.GetRenderWindow().Render()
        if not self.center_to_robot_base_toggle.isChecked():
            realpose = self.transform_poses(realpose, self.T_CamToBase)
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

