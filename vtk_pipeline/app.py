import sys
import time
import json
import logging
import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c

import cv2


from typing import Callable
from PyQt5 import QtWidgets, QtCore, QtGui
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

from .pcd_streamer import PCDStreamerFromCamera, PCDUpdater
from utils import RobotInterface, CameraInterface, ARUCO_BOARD
# Import constants
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR

from .app_ui import PCDStreamerUI

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

    # Class-level dictionary to store callbacks
    

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
        self.capturing = False  # Initialize capturing flag
        self.acq_mode = False  # Initialize acquisition mode flag
        self.frame_num = 0
        self.real_fps = 0
        self.streaming = False  # Streaming flag
        self.frame = None
        self.callback_bindings()
        self.palettes = self.get_num_of_palette(80)
        self.init_scene_objects()
        self.init_bbox()
        self.streamer = PCDStreamerFromCamera(params=params)
        self.pcd_updater = PCDUpdater(self.renderer)
        self.streamer.camera_frustrum.register_renderer(self.renderer)
        self.callbacks = {name: getattr(self, name) for name in _callback_names}

        
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
        self.handeye_calib_init_button.setEnabled(False)
        # self.save_pcd_button.setEnabled(False)
        # self.save_rgbd_button.setEnabled(False)
        self.detect_board_toggle.setEnabled(False)
        self.data_collect_button.setEnabled(False)

    def set_enable_after_stream_init(self):
        self.capture_toggle.setEnabled(True)
        self.seg_model_init_toggle.setEnabled(True)
        self.handeye_calib_init_button.setEnabled(True)
        # self.save_pcd_button.setEnabled(True)
        # self.save_rgbd_button.setEnabled(True)
        self.detect_board_toggle.setEnabled(True)
        self.data_collect_button.setEnabled(True)
        
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
        self.robot_init_button.clicked.connect(self.on_robot_init_button_clicked)
        self.calib_collect_button.clicked.connect(self.on_calib_collect_button_clicked)
        self.calib_button.clicked.connect(self.on_calib_button_clicked)
        self.detect_board_toggle.stateChanged.connect(self.on_detect_board_toggle_state_changed)

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
            # Create an instance of FakeCamera
            connected = self.streamer.camera_mode_init()
            # self.fake_camera = FakeCamera()
            # connected = self.fake_camera.connect(0)  # Assuming index 0
            if connected:
                self.status_message.setText("System: Streaming from Camera")
                # Start a QTimer to update frames
                self.timer = QtCore.QTimer()
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
        view_up = rotation[:, 1]
        
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
        # Update the GUI with the frame
        
        frame_elements = {
            'fps': round(self.real_fps, 2),  # Assuming 30 FPS for fake camera
        }

        frame_elements.update(frame)
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
        logger.debug("Robot init button clicked")

    def on_calib_collect_button_clicked(self):
        logger.debug("Calibration collect button clicked")

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

    def update_color_image(self, color_image):
        """Update the color image display."""
        if self.color_groupbox.isChecked():
            # Convert color_image to QImage and display in QLabel
            if isinstance(color_image, np.ndarray):
                # Ensure the image is in RGB format
                if color_image.shape[2] == 3:
                    height, width, channel = color_image.shape
                    bytes_per_line = 3 * width
                    q_image = QtGui.QImage(color_image.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(q_image)
                    self.color_video.setPixmap(pixmap.scaled(self.color_video.size(), QtCore.Qt.KeepAspectRatio))
            elif isinstance(color_image, o3d.geometry.Image):
                # Handle other types of images if necessary
                self.update_color_image(np.array(color_image))
                
            elif isinstance(color_image, o3d.t.geometry.Image):
                # Handle other types of images if necessary
                self.update_color_image(np.array(color_image.cpu()))
            else:
                # Handle other types of images if necessary
                pass

    def update_depth_image(self, depth_image):
        """Update the depth image display."""
        if self.depth_groupbox.isChecked():
            # Convert depth_image to QImage and display in QLabel
            if isinstance(depth_image, np.ndarray):
                # Normalize depth image for display
                # depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                # depth_image_normalized = depth_image_normalized.astype(np.uint8)
                height, width, channel = depth_image.shape
                bytes_per_line = 3 * width
                q_image = QtGui.QImage(depth_image.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.depth_video.setPixmap(pixmap.scaled(self.depth_video.size(), QtCore.Qt.KeepAspectRatio))
            elif isinstance(depth_image, o3d.geometry.Image):
                # Handle other types of images if necessary
                self.update_depth_image(np.array(depth_image))
            elif isinstance(depth_image, o3d.t.geometry.Image):
                self.update_depth_image(np.array(depth_image.cpu()))

    def init_scene_objects(self):
        """Initialize scene objects in the VTK renderer."""
        # Robot base frame
        size = [0.02]*3
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

    def init_bbox(self):
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
        if 'handeye_calib_init' in startup_params and startup_params['handeye_calib_init']:
            self.on_handeye_calib_init_button()
        if 'calib_check' in startup_params and startup_params['calib_check']:
            self.on_calib_check_button()
        if 'collect_data_viewer' in startup_params and startup_params['collect_data_viewer']:
            self.collected_data.start_display_thread()

    def transform_element(self, elements:dict, element_name: str):
        # if self.pipeline_model.T_cam_to_base is None:
        #     matrix = np.eye(4)
        # else:
        #     matrix = self.pipeline_model.T_cam_to_base

        # if element_name in elements:
        #     match element_name:
        #         case 'pcd':
        #             elements[element_name].transform(matrix)
        #         case 'robot_end_frame':
        #             elements[element_name] @= matrix
        #         case 'robot_base_frame':
        #             elements[element_name] @= matrix
        #         case 'board_pose':
        #             elements[element_name] @= matrix
        #         case _:
        #             logger.warning(f"No transform for {element_name}")
        pass

    def _render_update(self, frame_elements):
        """Helper to handle render update and signal completion."""
        
        # Signal rendering is done
        # with self.pipeline_model.cv_render:
        #     self.pipeline_view.update(frame_elements)
        #     self.pipeline_model.render_done = True
        #     self.pipeline_model.cv_render.notify_all()
        pass

    @callback
    def on_capture_toggle(self, is_on):
        logger.debug("on_capture_toggle")
        """Callback to toggle capture."""
        # self.pipeline_view.capturing = is_on
        # self.pipeline_view.vfov =  1.25 * self.pipeline_model.vfov
        # if not self.pipeline_view.capturing:
        #     # Set the mouse callback when not capturing
        #     self.pipeline_view.pcdview.set_on_mouse(self.on_mouse_widget3d)
        # else:
        #     # Unset the mouse callback when capturing
        #     self.pipeline_view.pcdview.set_on_mouse(None)

        # # Update model
        # self.pipeline_model.flag_capture = is_on
        # if not is_on:
        #     self.on_toggle_record(False)

        # else:
        #     with self.pipeline_model.cv_capture:
        #         self.pipeline_model.cv_capture.notify()
        pass

    @callback
    def on_center_to_base_toggle(self, is_on):
        logger.debug("on_center_to_base_toggle")
        # self.pipeline_model.flag_center_to_base = is_on
        pass

    @callback
    def on_toggle_record(self, is_enabled):
        logger.debug("on_toggle_record")
        """Callback to toggle recording RGBD video."""
        # self.pipeline_model.flag_record = is_enabled
        pass


    @callback
    def on_window_close(self):
        logger.debug("on_window_close")
        """Callback when the user closes the application window."""
        # self.pipeline_model.flag_exit = True
        # with self.pipeline_model.cv_capture:
        #     self.pipeline_model.cv_capture.notify_all()
        # return True  # OK to close window
        pass

    @callback
    def on_save_pcd_button(self):
        logger.debug("on_save_pcd_button")
        """Callback to save current point cloud."""
        # self.pipeline_model.flag_save_pcd = True
        pass

    @callback
    def on_seg_model_init_toggle(self, is_enabled):
        logger.debug("on_seg_model_init_toggle")
        # self.pipeline_model.seg_model_intialization()
        # self.pipeline_model.flag_segemtation_mode = is_enabled
        pass

    @callback
    def on_save_rgbd_button(self):
        logger.debug("on_save_rgbd_button")
        """Callback to save current RGBD image pair."""
        # self.pipeline_model.flag_save_rgbd = True
        # logger.debug('Saving RGBD image pair')
        pass

    @callback
    def on_mouse_widget3d(self, event):
        logger.debug("on_mouse_widget")
        # if self.pipeline_view.capturing:
        #     return gui.Widget.EventCallbackResult.IGNORED  # Do nothing if capturing

        # if not self.pipeline_view.acq_mode:
        #     return gui.Widget.EventCallbackResult.IGNORED

        # # Handle left button down with Ctrl key to start drawing
        # if (event.type == gui.MouseEvent.Type.BUTTON_DOWN and
        #     event.is_modifier_down(gui.KeyModifier.CTRL) and
        #     event.is_button_down(gui.MouseButton.LEFT)):
        #     x = event.x - self.pipeline_view.pcdview.frame.x
        #     y = event.y - self.pipeline_view.pcdview.frame.y
        #     if 0 <= x < self.pipeline_view.pcdview.frame.width and 0 <= y < self.pipeline_view.pcdview.frame.height:

        #         def depth_callback(depth_image):
        #             depth_array = np.asarray(depth_image)
        #             # Check if (x, y) are valid coordinates inside the depth image
        #             if y < depth_array.shape[0] and x < depth_array.shape[1]:
        #                 depth = depth_array[y, x]
        #             else:
        #                 depth = 1.0  # Assign far plane depth if out of bounds

        #             if depth == 1.0:  # clicked on nothing (far plane)
        #                 text = "Mouse Coord: Clicked on nothing"
        #             else:
        #                 # Compute world coordinates from screen (x, y) and depth
        #                 world = self.pipeline_view.pcdview.scene.camera.unproject(
        #                     x, y, depth, self.pipeline_view.pcdview.frame.width, self.pipeline_view.pcdview.frame.height)
        #                 text = "Mouse Coord: ({:.3f}, {:.3f}, {:.3f})".format(
        #                     world[0], world[1], world[2])

        #             # Update label in the main UI thread
        #             def update_label():
        #                 self.pipeline_view.scene_widgets.mouse_coord.text = text
        #                 self.pipeline_view.window.set_needs_layout()

        #             gui.Application.instance.post_to_main_thread(self.pipeline_view.window, update_label)

        #         # Perform the depth rendering asynchronously
        #         self.pipeline_view.pcdview.scene.scene.render_to_depth_image(depth_callback)
        #     return gui.Widget.EventCallbackResult.HANDLED

        # # Handle dragging to update rectangle
        # elif event.type == gui.MouseEvent.Type.DRAG and self.drawing_rectangle:
        #     pass
        #     return gui.Widget.EventCallbackResult.HANDLED

        # # Handle left button up to finish drawing
        # elif (event.type == gui.MouseEvent.Type.BUTTON_UP and
        #     self.drawing_rectangle):
        #     # Finalize rectangle
        #     self.drawing_rectangle = False
        #     return gui.Widget.EventCallbackResult.HANDLED

        # return gui.Widget.EventCallbackResult.IGNORED
        pass

    @callback
    def on_robot_init_button(self):
        logger.debug("on_robot_init_button")
        # ret, msg, msg_color = self.pipeline_model.robot_init()

        # if self.pipeline_model.flag_robot_init and self.pipeline_model.flag_camera_init:
        #     self.pipeline_view.scene_widgets.handeye_calib_init_button.enabled = True
        
        # self.pipeline_view.scene_widgets.robot_msg.text = msg
        # self.pipeline_view.scene_widgets.robot_msg.text_color = msg_color
        # self.calibration_data.reset()
        # self.pipeline_view.scene_widgets.frame_list_view.set_items(['Click "Collect Current Frame" to start'])
        pass

    @callback
    def on_cam_calib_init_button(self):
        logger.debug("on_cam_calib_init_button")
        # ret, msg, msg_color = self.pipeline_model.camera_calibration_init()
        
        # if self.pipeline_model.flag_robot_init and self.pipeline_model.flag_camera_init:
        #     self.pipeline_view.scene_widgets.handeye_calib_init_button.enabled = True
        
        # self.pipeline_view.scene_widgets.calibration_msg.text = msg
        # self.pipeline_view.scene_widgets.calibration_msg.text_color = msg_color
        # self.pipeline_view.scene_widgets.detect_board_toggle.enabled = True
        # self.calibration_data = self.pipeline_model.calibration_data_init()
        # self.pipeline_view.scene_widgets.frame_list_view.set_items(['Click "Collect Current Frame" to start'])
        pass

    @callback
    def on_handeye_calib_init_button(self):
        logger.debug("on_handeye_calib_init_button")
        # ret, msg, msg_color = self.pipeline_model.handeye_calibration_init()
        # if ret:
        #     self.pipeline_model.flag_handeye_calib_init = True
        #     self.pipeline_view.scene_widgets.calibration_msg.text = msg
        #     self.pipeline_view.scene_widgets.calibration_msg.text_color = msg_color
        #     self.calibration_data = self.pipeline_model.calibration_data_init()
        #     self.pipeline_view.scene_widgets.frame_list_view.set_items(['Click "Collect Current Frame" to start'])
            # self.pipeline_view.scene_widgets.he_calibreate_button.enabled = False
        pass


    @callback
    def on_camera_view_button(self):
        logger.debug("on_camera_view_button")
        # self.pipeline_view.pcdview.setup_camera(self.pipeline_view.vfov, 
        #                                         self.pipeline_view.pcd_bounds, [0, 0, 0])
        # lookat = [0, 0, 0]
        # placeat = [-0.139, -0.356, -0.923]
        # pointat = [-0.037, -0.93, 0.3649]
        # self.pipeline_view.pcdview.scene.camera.look_at(lookat, placeat, pointat)
        pass

    @callback
    def on_birds_eye_view_button(self):
        logger.debug("on_birds_eye_view_button")
        """Callback to reset point cloud view to birds eye (overhead) view"""
        # self.pipeline_view.pcdview.setup_camera(self.pipeline_view.vfov, self.pipeline_view.pcd_bounds, [0, 0, 0])
        # self.pipeline_view.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])
        pass

    @callback
    def on_stream_init_button(self):
        logger.debug("on_stream_init_button")
        # log.debug('Stream init start')
        # match self.pipeline_view.scene_widgets.stream_combbox.selected_text:
        #     case 'Camera':
        #         try:
        #             self.pipeline_model.camera_mode_init()
        #             self.pipeline_model.flag_stream_init = True
        #             self.pipeline_view.scene_widgets.status_message.text = "Azure Kinect camera connected."
        #             self.pipeline_view.scene_widgets.after_stream_init()
        #             self.pipeline_model.flag_camera_init = True
        #             if self.pipeline_model.flag_robot_init and self.pipeline_model.flag_camera_init:
        #                 self.pipeline_view.scene_widgets.handeye_calib_init_button.enabled = True
        #             self.on_camera_view_button()
        #         except Exception as e:
        #             self.pipeline_view.scene_widgets.status_message.text = "Camera initialization failed!"
        #     case 'Video':
        #         pass
        pass

    @callback
    def on_acq_mode_toggle(self, is_on):
        logger.debug("on_acq_mode_toggle")
        # self.pipeline_view.acq_mode = is_on
        # if is_on:
        #     self.pipeline_view.pcdview.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)
        #     if self.pipeline_view.plane is None:
        #         plane = o3d.geometry.TriangleMesh.create_box(width=10, height=0.01, depth=10)
        #         plane.translate([-5, 1, -5])  # Position the plane at y=1
        #         plane.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray color
        #         plane_material = rendering.MaterialRecord()
        #         plane_material.shader = "defaultUnlit"
        #         plane_material.base_color = [0.8, 0.8, 0.8, 0.5]  # Semi-transparent
        #         self.pipeline_view.pcdview.scene.add_geometry("edit_plane", plane, plane_material)
        #         self.pipeline_view.plane = plane
        #         self.pipeline_view.pcdview.force_redraw()
        # else:
        #     self.pipeline_view.pcdview.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        #     if self.pipeline_view.plane is not None:
        #         self.pipeline_view.pcdview.scene.remove_geometry("edit_plane")
        #         self.pipeline_view.plane = None
        #     self.pipeline_view.pcdview.force_redraw()
        pass

    @callback
    def on_calib_check_button(self):
        logger.debug("on_calib_check_button")
        # try:
        #     path = self.params['calib_path']
        #     with open(path, 'r') as f:
        #         self.calib:dict = json.load(f)
        #     intrinsic = np.array(self.calib.get('camera_matrix'))
        #     dist_coeffs = np.array(self.calib.get('dist_coeffs'))
        #     self.pipeline_model.update_camera_matrix(intrinsic, dist_coeffs)
        #     self.pipeline_view.scene_widgets.calib_combobox.clear_items()

        #     for name in self.calib.get('calibration_results').keys():
        #         self.pipeline_view.scene_widgets.calib_combobox.add_item(name)
            
        #     self.pipeline_view.scene_widgets.center_to_base_toggle.enabled = True
        #     self.pipeline_view.scene_widgets.calib_combobox.enabled = True
        #     self.pipeline_model.flag_handeye_calib_success = True
        #     self.pipeline_view.scene_widgets.center_to_base_toggle.enabled = True
        #     self.pipeline_view.scene_widgets.calib_combobox.selected_index = 0
        #     self.pipeline_view.scene_widgets.data_tab.visible = True
        #     method = self.pipeline_view.scene_widgets.calib_combobox.selected_text
        #     self.pipeline_model.T_cam_to_base = np.array(self.calib.get('calibration_results').get(method).get('transformation_matrix'))
            
        # except Exception as e:
        #     self.pipeline_view.scene_widgets.calib_combobox.enabled = False
        #     log.error(e)
        pass

    @callback
    def on_calib_combobox_change(self, text, index):
        logger.debug("on_calib_combobox_change")
        # self.pipeline_model.T_cam_to_base = np.array(self.calib.get('calibration_results').get(text).get('transformation_matrix'))
        # self.pipeline_model.T_cam_to_base
        pass
        
    @callback
    def on_board_col_num_edit_change(self, value):
        logger.debug("on_board_col_num_edit_change")
        # self.calibration.chessboard_size[0] = int(value)
        # self.params['board_shape'] = (int(value), self.params['board_shape'][1])
        # log.debug(f"Chessboard type: {self.params.get('board_shape')}")    
        pass

    @callback
    def on_board_row_num_edit_change(self, value):
        logger.debug("on_board_row_num_edit_change")
        # self.calibration.chessboard_size[1] = int(value)
        # self.params['board_shape'] = (self.params.get('board_shape')[0], int(value))
        # log.debug(f"Chessboard type: {self.params.get('board_shape')}")
        pass

    @callback
    def on_board_square_size_num_edit_change(self, value):
        logger.debug("on_board_square_size_num_edit_change")
        # self.params['board_square_size'] = value
        # logger.debug(f'board_square_size changed: {value} mm')
        pass
    
    @callback
    def on_board_marker_size_num_edit_change(self, value):
        logger.debug("on_board_marker_size_num_edit_change")
        # self.params['board_marker_size'] = value
        # logger.debug(f'board_marker_size changed: {value} mm')
        pass

    @callback
    def on_data_folder_select_button(self):
        logger.debug("on_data_folder_select_button")
        # filedlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, 
        #                          "Select Folder",
        #                          self.pipeline_view.window.theme)
        # # filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")
        # # filedlg.add_filter("", "All files")
        # filedlg.set_on_cancel(self._on_data_folder_cancel)
        # filedlg.set_on_done(self._on_data_folder_selcted)
        # self.pipeline_view.window.show_dialog(filedlg)
        pass

    def _on_data_folder_selcted(self, path):
        # self.pipeline_view.scene_widgets.data_folder_text.text_value = path+"/data.recorder"
        # self.pipeline_view.window.close_dialog()
        pass

    def _on_data_folder_cancel(self):
        # self.pipeline_view.scene_widgets.data_folder_text.text = ""
        # self.pipeline_view.window.close_dialog()
        pass
    
    @callback
    def on_prompt_text_change(self, text):
        logger.debug("on_prompt_text_change")
        # if text == "":
        #     self.pipeline_view.scene_widgets.data_collect_button.enabled = False
        # else:
        #     self.pipeline_view.scene_widgets.data_collect_button.enabled = True
        # logger.debug(f"Prompt text changed: {text}")
        # select_item = self.pipeline_view.scene_widgets.data_tree_view.selected_item
        # if select_item.parent_text == "Prompt":
        #     self.collected_data.data_list[self.collected_data.dataids.index(select_item.root_text)]['prompt'] = text
        #     self._data_tree_view_update()
        pass
            
    def _data_tree_view_update(self):
        # self.pipeline_view.scene_widgets.data_tree_view.tree.clear()
        # for key, value in self.collected_data.shown_data_json.items():
        #     root_id = self.pipeline_view.scene_widgets.data_tree_view.add_item(
        #         self.pipeline_view.scene_widgets.data_tree_view.tree.get_root_item(), key, level=1)

        #     # Add 'prompt' field
        #     prompt_id = self.pipeline_view.scene_widgets.data_tree_view.add_item(root_id, "Prompt", level=2, root_text=key)
        #     self.pipeline_view.scene_widgets.data_tree_view.add_item(prompt_id, value["prompt"], level=3, root_text=key)

        #     # Add 'bbox' field
        #     bbox_id = self.pipeline_view.scene_widgets.data_tree_view.add_item(root_id, "Bbox", level=2, root_text=key)
        #     bbox_text = f"[{','.join(f'{v:.2f}' for v in value['bboxes'])}]"
        #     self.pipeline_view.scene_widgets.data_tree_view.add_item(bbox_id, bbox_text, level=3, root_text=key)

        #     # Add 'pose' field
        #     pose_id = self.pipeline_view.scene_widgets.data_tree_view.add_item(root_id, "Pose", level=2, root_text=key)
        #     for i, pose in enumerate(value["pose"]):
        #         pose_text = f"{i + 1}: [{','.join(f'{v:.2f}' for v in pose)}]"
        #         self.pipeline_view.scene_widgets.data_tree_view.add_item(pose_id, pose_text, level=3, root_text=key)

        # self.pipeline_view.scene_widgets.data_tree_view.selected_item.reset()
        pass

    @callback
    def on_data_collect_button(self):
        logger.debug("on_data_collect_button")
        # tmp_pose = np.array([1,2,3,4,5,6])
        # try:
        #     if not self.pipeline_view.scene_widgets.capture_toggle.is_on:
        #         logger.warning("Stream not started")
        #         return
                
        #     if self.pipeline_model.flag_center_to_base:
        #         tmp_pose = self.pipeline_model.robot_interface.capture_gripper_to_base(sep=False)
        #     else:
        #         tmp_pose = self.pipeline_model.get_cam_space_gripper_pose()

        #     frame = self.frame
        #     if frame is None:
        #         raise Exception("No frame")

        #     if 'color' in frame:
        #         color = np.asarray(self.pipeline_model.rgbd_frame.color)
        #     if 'depth' in frame:
        #         depth = np.asarray(self.pipeline_model.rgbd_frame.depth)
        #     if 'pcd' in frame:
        #         if 'seg' in frame:
        #             seg = frame['seg']
        #         else:
        #             seg = np.zeros(frame['pcd'].point.positions.shape[0])
        #         pcd = frame['pcd'].to_legacy()
        #         xyz = np.asarray(pcd.points)
        #         rgb = np.asarray(pcd.colors)
        #         rgb = (rgb * 255).astype(np.uint8)
        #         pcd_with_labels = np.hstack((xyz, rgb, seg.reshape(-1, 1)))

        #     self.collected_data.append(prompt=self.pipeline_view.scene_widgets.prompt_text.text_value, 
        #                                pose=tmp_pose, 
        #                                bbox_dict=copy.deepcopy(self.pipeline_view.bbox_params),
        #                                color=color, 
        #                                depth=depth, 
        #                                point_cloud=pcd_with_labels)
            
        #     self._data_tree_view_update()
        #     logger.debug(f"On data collect Click")
        # except Exception as e: 
        #     logger.error(e)
        pass
        # self.pipeline_view.scene_widgets.data_list_view.set_items(self.collected_data)

    @callback
    def on_data_tree_view_changed(self, item):
        logger.debug("on_data_tree_view_changed")
        # logger.debug(
        #     f"Root Parent Text: {item.root_text}, "
        #     f"Custom Callback -> Selected Item ID: {item.item_id}, "
        #     f"Level: {item.level}, Index in Level: {item.index_in_level}, "
        #     f"Parent Text: {item.parent_text}"
        # )
        # select_item = self.pipeline_view.scene_widgets.data_tree_view.selected_item
        # self.pipeline_view.scene_widgets.prompt_text.text_value = self.collected_data.shown_data_json.get(
        #         select_item.root_text
        #         ).get('prompt')
        # match select_item.level:
        #     case 1:
        #         pass
        #     case 2:
        #         pass
        #     case 3:
        #         if select_item.parent_text == "Pose":
        #             prompt_idx = self.collected_data.dataids.index(select_item.root_text)
        #             pose_idx = select_item.index_in_level
        #             self.collected_data.show_image(prompt_idx, pose_idx)
        pass

    @callback
    def on_data_tree_view_remove_button(self):
        logger.debug("on_data_tree_view_remove_button")
        # select_item = self.pipeline_view.scene_widgets.data_tree_view.selected_item
        # # self.on_data_tree_view_changed(select_item)
        # if select_item != None:
        #     match select_item.level:
        #         case 1:
        #             logger.debug(f"Removing {select_item.root_text}")
        #             self.collected_data.pop(self.collected_data.dataids.index(select_item.root_text))
        #             self._data_tree_view_update()
        #         case 2:
        #             pass
        #         case 3:
        #             logger.debug(f"Removing pose")
        #             if select_item.parent_text == "Pose":
        #                 self.collected_data.pop_pose(self.collected_data.dataids.index(select_item.root_text), 
        #                                              select_item.index_in_level)
        #             self._data_tree_view_update()
        pass
    

    @callback
    def on_data_tree_view_load_button(self):
        logger.debug("on_data_tree_view_load_button")
        # path = self.params.get('data_path', './data')
        # path += "/" + self.pipeline_view.scene_widgets.data_folder_text.text_value
        # if self.pipeline_view.scene_widgets.data_folder_text.text_value == "":
        #     logger.warning("No data folder selected")
        # else:
        #     try:
        #         self.collected_data.path = path
        #         self.collected_data.load()
        #         self._data_tree_view_update()
        #     except Exception as e:
        #         logger.error(f"Failed to load data: {e}")
            
        # logger.debug("Loading data")
        pass

    @callback
    def on_data_save_button(self):
        logger.debug("on_data_save_button")
        # path = self.params.get('data_path', './data')
        # data_folder = self.pipeline_view.scene_widgets.data_folder_text.text_value
        # if data_folder == "":
        #     logger.warning("No data folder selected, use tmp folder")
        #     data_folder = "tmp"

        # path += "/" + data_folder
        # if len(self.collected_data) == 0:
        #     logger.warning("No data to save")
        # else:
        #     self.collected_data.path = path
        #     self.collected_data.save()
        # logger.debug("Saving data")
        pass


    @callback
    def on_board_type_combobox_change(self, text, index):
        logger.debug("on_board_type_combobox_change")
        # self.params['board_type'] = self.pipeline_view.scene_widgets.board_type_combobox.selected_text
        # logger.debug(f"Board type: {self.params.get('board_type')}")
        pass
    
    @callback
    def on_calib_collect_button(self):
        logger.debug("on_calib_collect_button")
        # logger.debug(self.calibration_data.display_str_list)
        # self.pipeline_model.flag_calib_collect = True
        # self.pipeline_view.scene_widgets.frame_list_view.set_items(
        #     self.calibration_data.display_str_list)
        # logger.debug("Collecting calibration data")
        pass

    @callback
    def on_calib_list_remove_button(self):
        logger.debug("on_calib_list_remove_button")
        # self.calibration_data.pop(self.pipeline_view.scene_widgets.frame_list_view.selected_index)
        # self.pipeline_view.scene_widgets.frame_list_view.set_items(
        #     self.calibration_data.display_str_list)
        # logger.debug("Removing calibration data")
        pass

    @callback
    def on_robot_move_button(self):
        logger.debug("on_robot_move_button")
        # idx = self.pipeline_view.scene_widgets.frame_list_view.selected_index
        # try:
        #     self.pipeline_model.robot_interface.move_to_pose(
        #         self.calibration_data.robot_poses[idx])
        #     self.calibration_data.modify(idx, np.asarray(self.pipeline_model.rgbd_frame.color),
        #                                 self.pipeline_model.robot_interface.capture_gripper_to_base(sep=False),
        #                                 copy.deepcopy(self.pipeline_view.bbox_params))
        #     logger.debug("Moving robot and collecting data")
        # except:
        #     logger.error("Failed to move robot")
        pass
        
    @callback
    def on_calib_save_button(self):
        logger.debug("on_calib_save_button")
        # self.calibration_data.save_calibration_data(
        #     self.pipeline_view.scene_widgets.calib_save_text.text_value)
        # self.on_calib_check_button()
        # logger.debug("Saving calibration data and check data")
        pass
    
    @callback
    def on_calib_op_save_button(self):
        logger.debug("on_calib_op_save_button")
        # self.calibration_data.save_img_and_pose()
        # logger.debug("Saving images and poses")
        pass

    @callback
    def on_calib_op_load_button(self):
        logger.debug("on_calib_op_load_button")
        # self.calibration_data.load_img_and_pose()
        # self.pipeline_view.scene_widgets.frame_list_view.set_items(
        #     self.calibration_data.display_str_list)
        # logger.debug("Checking calibration data")
        pass
    
    @callback
    def on_calib_op_run_button(self):
        logger.debug("on_calib_op_run_button")
        # logger.debug("Running calibration data")
        # self.pipeline_model.robot_interface.set_teach_mode(False)
        # self.pipeline_model.calib_exec.submit(self.pipeline_model.auto_calibration)
        pass


    @callback
    def on_calib_button(self):
        logger.debug("on_calib_button")
        # self.calibration_data.calibrate_all()
        # self.pipeline_view.scene_widgets.frame_list_view.set_items(
        #     self.calibration_data.display_str_list)
        # self.pipeline_model.update_camera_matrix(self.calibration_data.camera_matrix, 
        #                                          self.calibration_data.dist_coeffs)
        # logger.debug("calibration button")
        pass

    @callback 
    def on_detect_board_toggle(self, is_on):
        logger.debug("on_detect_board_toggle")
        # self.pipeline_model.flag_tracking_board = is_on
        # logger.debug(f"Detecting board: {is_on}")
        pass

    @callback 
    def on_show_axis_in_scene_toggle(self, is_on):
        logger.debug("on_show_axis_in_scene_toggle")
        # self.pipeline_model.flag_calib_axis_to_scene = is_on
        # self.pipeline_view.pcdview.scene.show_geometry('board_pose', is_on)
        # self.pipeline_view.pcdview.scene.show_geometry('robot_base_frame', is_on)
        # self.pipeline_view.pcdview.scene.show_geometry('robot_end_frame', is_on)

        # logger.debug(f"Show axis in scene: {is_on}")
        pass

    @callback
    def on_frame_list_view_changed(self, new_val, is_dbl_click):
        logger.debug("on_frame_list_view_changed")
        # # TODO: update still buggy, need to be fixed
        # logger.debug(new_val)
        # self.pipeline_model.flag_tracking_board = False
        # self.pipeline_view.scene_widgets.detect_board_toggle.is_on = False
        # if len(self.calibration_data) > 0:
        #     img = self.calibration_data.images[self.pipeline_view.scene_widgets.frame_list_view.selected_index]
        #     # img = np.random.randint(0, 255, (img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #     img = o3d.t.geometry.Image(img)
        #     if self.pipeline_view.scene_widgets.show_calib.get_is_open():
        #         sampling_ratio = self.pipeline_view.video_size[1] / img.columns
        #         self.pipeline_view.scene_widgets.calib_video.update_image(img.resize(sampling_ratio))
        #         logger.debug("Showing calibration pic")
        pass

    @callback
    def on_calib_save_text_changed(self, text):
        logger.debug("on_calib_save_text_changed")
        # self.params['calib_path'] = text
        # logger.debug(f"calib_path changed: {text}")
        pass

    @callback
    def on_key_pressed(self, event):
        logger.debug("on_key_pressed")
        # if self.pipeline_view.scene_widgets.tab_view.selected_tab_index == 2 and \
        #             self.pipeline_model.flag_camera_init:
        #     if event.type == gui.KeyEvent.Type.DOWN:
        #         if event.key == gui.KeyName.SPACE:
        #         # self.pipeline_model.flag_tracking_board = not self.pipeline_model.flag_tracking_board
        #             self.on_calib_collect_button()
        #         elif event.key == gui.KeyName.C:
        #             self.on_calib_button()
        #         logger.info(f"key pressed {event.key}")
        #     return True
        # # if event.type == gui.KeyEvent.Type.DOWN:
        # #     print(event.key)
        # return False
        pass