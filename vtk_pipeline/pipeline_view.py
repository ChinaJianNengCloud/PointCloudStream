import sys, cv2
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
from typing import Callable
import logging
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import json
import time

logger = logging.getLogger(__name__)

class FakeCamera:
    """Fake camera that generates synthetic RGBD frames for debugging."""
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.frame_idx = 0

    def connect(self, index):
        """Fake connect method, always returns True."""
        return True

    def disconnect(self):
        """Fake disconnect method."""
        pass

    def capture_frame(self, enable_align_depth_to_color=True):
        """Generate synthetic depth and color images with missing depth regions."""
        # Create a color image with a moving circle
        color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        center_x = int((self.frame_idx * 5) % self.width)
        center_y = self.height // 2
        cv2.circle(color_image, (center_x, center_y), 50, (0, 255, 0), -1)

        # Generate a depth image as a gradient
        depth_image = np.tile(np.linspace(500, 2000, self.width, dtype=np.uint16), (self.height, 1))

        # Randomly zero out some regions to simulate missing depth
        num_missing_regions = np.random.randint(5, 15)  # Random number of missing regions
        for _ in range(num_missing_regions):
            # Randomly choose the size and position of the missing region
            start_x = np.random.randint(0, self.width - 50)
            start_y = np.random.randint(0, self.height - 50)
            width = np.random.randint(20, 100)
            height = np.random.randint(20, 100)
            
            # Zero out the region
            depth_image[start_y:start_y + height, start_x:start_x + width] = 0

        self.frame_idx += 1

        # Return a fake RGBD frame
        return FakeRGBDFrame(depth_image, color_image)


class FakeRGBDFrame:
    """Fake RGBD frame containing synthetic depth and color images."""
    def __init__(self, depth_image, color_image):
        self.depth = depth_image
        self.color = color_image

class PCDStreamerFromCamera:
    """Fake RGBD streamer that generates synthetic RGBD frames."""
    def __init__(self, align_depth_to_color: bool = True, params: dict=None):
        self.align_depth_to_color = align_depth_to_color
        self.use_fake_camera = params.get('use_fake_camera', False)
        self.o3d_device = o3d.core.Device(params.get('device', 'cuda:0'))
        self.camera_config_file = params.get('camera_config', None)
        self.depth_max = 3.0 
        self.pcd_stride = 2  # downsample point cloud, may increase frame rate
        self.next_frame_func = None
        self.square_size = 0.015
        self.camera = None
        self.flag_normals = False
        self.extrinsics = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float32,
                                              device=self.o3d_device)

    def get_frame(self, take_pcd: bool = True):
        if self.camera is None:
            logger.warning("No camera connected")
        if take_pcd:
            rgbd_frame = self.camera.capture_frame(True)
            if rgbd_frame is None:
                return {}
            
            depth = o3d.t.geometry.Image(o3c.Tensor(np.asarray(rgbd_frame.depth), 
                                                    device=self.o3d_device))
            color = o3d.t.geometry.Image(o3c.Tensor(np.asarray(rgbd_frame.color), 
                                                    device=self.o3d_device))
            # logger.debug("Stream Debug Point 2.0")
            rgbd_image = o3d.t.geometry.RGBDImage(color, depth)
            pcd_frame = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self.intrinsic_matrix, self.extrinsics,
                self.depth_scale, self.depth_max,
                self.pcd_stride, self.flag_normals)
            
            depth_in_color = depth.colorize_depth(
                    self.depth_scale, 0, self.depth_max)
            
            return {
                'pcd': pcd_frame.to_legacy(), 
                'color': color, 
                'depth': depth_in_color
                    }

    
    def camera_mode_init(self):
        try:
            if self.camera_config_file:
                config = o3d.io.read_azure_kinect_sensor_config(self.camera_config_file)
                if self.camera is None:
                    if self.use_fake_camera:
                        self.camera = FakeCamera()
                    else:
                        self.camera = o3d.io.AzureKinectSensor(config)
                    self.camera_json = json.load(open(self.camera_config_file, 'r'))
                intrinsic = o3d.io.read_pinhole_camera_intrinsic(self.camera_config_file)
            else:
                if self.camera is None:
                    if self.use_fake_camera:
                        self.camera = FakeCamera()
                    else:
                        self.camera = o3d.io.AzureKinectSensor()
                # Use default intrinsics
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
                
            if not self.camera.connect(0):
                raise RuntimeError('Failed to connect to sensor')
            
            self.intrinsic_matrix = o3d.core.Tensor(
                intrinsic.intrinsic_matrix,
                dtype=o3d.core.Dtype.Float32,
                device=self.o3d_device)
            self.depth_scale = 1000.0  # Azure Kinect depth scale
            logger.info("Intrinsic matrix:")
            logger.info(self.intrinsic_matrix)
            return True
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False


class ResizableImageLabel(QtWidgets.QLabel):
    """Custom QLabel that automatically resizes the image to fit the widget while maintaining aspect ratio."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(100, 100)  # Prevent the label from becoming too small
        self.original_pixmap = None
        self.setScaledContents(False)  # We'll handle scaling manually

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        if self.original_pixmap:
            self.update_scaled_pixmap()

    def resizeEvent(self, event):
        if self.original_pixmap:
            self.update_scaled_pixmap()
        super().resizeEvent(event)

    def update_scaled_pixmap(self):
        scaled_pixmap = self.original_pixmap.scaled(
            self.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)
        
          
class PipelineView(QtWidgets.QMainWindow):
    """Controls display and user interface using VTK and PyQt5."""

    def __init__(self, callbacks: dict[str, Callable] = None, params:dict=None):
        super().__init__()
        self.setWindowTitle("VTK GUI Application")
        self.resize(1200, 800)
        self.callbacks = callbacks  # Store the callbacks dictionary
        self.capturing = False  # Initialize capturing flag
        self.acq_mode = False  # Initialize acquisition mode flag
        self.frame_num = 0
        self.real_fps = 0
        self.streaming = False  # Streaming flag
        self.streamer = PCDStreamerFromCamera(params=params)
        # Initialize other variables
        self.display_mode = 'Colors'  # Initialize display mode to 'Colors'
        self.initUI()
        self.callback_bindings()
        # Palettes for segmentation coloring
        self.palettes = self.get_num_of_palette(80)
        # Initialize scene objects
        self.init_scene_objects()
        self.init_bbox()
        self.prev_frame_time = time.time()

    def initUI(self):
        # Create a splitter to divide the window
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        # Left side: VTK render window
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.splitter.addWidget(self.vtk_widget)

        # Right side: Panel
        self.panel = QtWidgets.QWidget()
        self.panel_layout = QtWidgets.QVBoxLayout(self.panel)
        self.splitter.addWidget(self.panel)

        # Set up VTK renderer
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.5, 0.5, 0.5)  # White background
        self.init_pcd_polydata()
        # Add axes
        self.axes = vtk.vtkAxesActor()
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(self.axes)
        self.axes_widget.SetInteractor(self.vtk_widget)
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOff()

        # Add labels
        self.add_3d_label((0, 0, 0), "Camera")
        self.add_3d_label((1, 0, 0), "X")
        self.add_3d_label((0, 1, 0), "Y")
        self.add_3d_label((0, 0, 1), "Z")

        # Initialize other GUI elements
        self.init_widgets()

        # Start the interactor
        self.vtk_widget.Initialize()
        # self.vtk_widget.Start()

    def init_pcd_polydata(self):
        vtk_points = vtk.vtkPoints()
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")

        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(vtk_points)
        self.polydata.GetPointData().SetScalars(vtk_colors)

        vertices = vtk.vtkCellArray()
        self.polydata.SetVerts(vertices)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)
        self.renderer.AddActor(actor)

    def init_widgets(self):
        # Initialize fps label
        self.init_fps_label(self.panel_layout)

        # Initialize status message
        self.init_status_message(self.panel_layout)

        # Initialize tab view
        self.init_tab_view(self.panel_layout)

        # Initialize widgets in tabs
        self.init_general_tab()
        self.init_view_tab()
        self.init_calibration_tab()
        self.init_bbox_tab()
        self.init_data_tab()

        # Set initial states
        self.set_disable_before_stream_init()

    # Include all the init_* methods from your VTKWidgets class here
    # For example:
    def init_fps_label(self, layout: QtWidgets.QVBoxLayout):
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        layout.addWidget(self.fps_label)

    def init_status_message(self, layout: QtWidgets.QVBoxLayout):
        self.status_message = QtWidgets.QLabel("System: No Stream is Initialized")
        layout.addWidget(self.status_message)
        self.robot_msg = QtWidgets.QLabel("Robot: Not Connected")
        layout.addWidget(self.robot_msg)
        self.calibration_msg = QtWidgets.QLabel("Calibration: None")
        layout.addWidget(self.calibration_msg)

    def add_3d_label(self, position, text):
        text_source = vtk.vtkVectorText()
        text_source.SetText(text)

        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())

        text_actor = vtk.vtkFollower()
        text_actor.SetMapper(text_mapper)
        text_actor.SetPosition(position)
        text_actor.SetScale(0.1, 0.1, 0.1)
        text_actor.SetCamera(self.renderer.GetActiveCamera())
        self.renderer.AddActor(text_actor)
    def init_tab_view(self, layout:QtWidgets.QVBoxLayout):
        self.tab_view = QtWidgets.QTabWidget()
        layout.addWidget(self.tab_view)

    def init_general_tab(self):
        self.general_tab = QtWidgets.QWidget()
        self.tab_view.addTab(self.general_tab, "General")
        general_layout = QtWidgets.QVBoxLayout(self.general_tab)

        # Initialize widgets in general tab
        self.init_stream_set(general_layout)
        self.init_record_save(general_layout)
        self.init_save_layout(general_layout)
        self.init_video_displays(general_layout)
        self.init_scene_info(general_layout)

    def init_view_tab(self):
        self.view_tab = QtWidgets.QWidget()
        self.tab_view.addTab(self.view_tab, "View")
        view_layout = QtWidgets.QVBoxLayout(self.view_tab)

        # Initialize widgets in view tab
        self.init_toggle_view_set(view_layout)
        self.init_aqui_mode(view_layout)
        self.init_center_to_base_mode(view_layout)
        self.init_display_mode(view_layout)
        self.init_view_layout(view_layout)
        self.init_operate_info(view_layout)

    def init_calibration_tab(self):
        self.calibration_tab = QtWidgets.QWidget()
        self.tab_view.addTab(self.calibration_tab, "Calib")
        calibration_layout = QtWidgets.QVBoxLayout(self.calibration_tab)

        self.init_calibrate_layout(calibration_layout)
        self.init_calibration_settings(calibration_layout)
        self.init_calib_color_image_display(calibration_layout)
        self.init_calibration_collect_layout(calibration_layout)
        self.init_calibrate_set(calibration_layout)

    def init_bbox_tab(self):
        self.bbox_tab = QtWidgets.QWidget()
        self.tab_view.addTab(self.bbox_tab, "Bbox")
        bbox_layout = QtWidgets.QVBoxLayout(self.bbox_tab)

        self.init_bbox_controls(bbox_layout)

    def init_data_tab(self):
        self.data_tab = QtWidgets.QWidget()
        self.tab_view.addTab(self.data_tab, "Data")
        data_layout = QtWidgets.QVBoxLayout(self.data_tab)

        self.data_collect_layout(data_layout)

    def init_stream_set(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        stream_label = QtWidgets.QLabel("Stream Init: ")
        h_layout.addWidget(stream_label)

        self.stream_combobox = QtWidgets.QComboBox()
        self.stream_combobox.addItem("Camera")
        self.stream_combobox.addItem("Video")
        h_layout.addWidget(self.stream_combobox)

        self.stream_init_button = QtWidgets.QPushButton("Start")
        h_layout.addWidget(self.stream_init_button)

    def init_record_save(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        record_label = QtWidgets.QLabel("Record / Save")
        h_layout.addWidget(record_label)

        self.toggle_record = QtWidgets.QCheckBox("Video")
        h_layout.addWidget(self.toggle_record)

    def init_save_layout(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        self.save_pcd_button = QtWidgets.QPushButton("Save Point cloud")
        h_layout.addWidget(self.save_pcd_button)

        self.save_rgbd_button = QtWidgets.QPushButton("Save RGBD frame")
        h_layout.addWidget(self.save_rgbd_button)

    def init_video_displays(self, layout:QtWidgets.QVBoxLayout):
        self.init_color_image_display(layout)
        self.init_depth_image_display(layout)

    def init_color_image_display(self, layout: QtWidgets.QVBoxLayout):
        self.color_groupbox = QtWidgets.QGroupBox("Color image")
        self.color_groupbox.setCheckable(True)
        self.color_groupbox.setChecked(True)
        group_layout = QtWidgets.QVBoxLayout()
        self.color_groupbox.setLayout(group_layout)
        layout.addWidget(self.color_groupbox)

        self.color_video = ResizableImageLabel()
        self.color_video.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.color_video.setAlignment(QtCore.Qt.AlignCenter)
        group_layout.addWidget(self.color_video)

    def init_depth_image_display(self, layout: QtWidgets.QVBoxLayout):
        self.depth_groupbox = QtWidgets.QGroupBox("Depth image")
        self.depth_groupbox.setCheckable(True)
        self.depth_groupbox.setChecked(True)
        group_layout = QtWidgets.QVBoxLayout()
        self.depth_groupbox.setLayout(group_layout)
        layout.addWidget(self.depth_groupbox)

        self.depth_video = ResizableImageLabel()
        self.depth_video.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.depth_video.setAlignment(QtCore.Qt.AlignCenter)
        group_layout.addWidget(self.depth_video)

    def init_scene_info(self, layout:QtWidgets.QVBoxLayout):
        self.scene_info = QtWidgets.QGroupBox("Scene Info")
        self.scene_info.setCheckable(True)
        self.scene_info.setChecked(False)
        group_layout = QtWidgets.QVBoxLayout()
        self.scene_info.setLayout(group_layout)
        layout.addWidget(self.scene_info)

        self.view_status = QtWidgets.QLabel("")
        group_layout.addWidget(self.view_status)

    def init_toggle_view_set(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        self.capture_toggle = QtWidgets.QCheckBox("Capture / Play")
        h_layout.addWidget(self.capture_toggle)

        self.seg_model_init_toggle = QtWidgets.QCheckBox("Seg Mode")
        h_layout.addWidget(self.seg_model_init_toggle)

    def init_aqui_mode(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        self.acq_mode_toggle = QtWidgets.QCheckBox("Acquisition Mode")
        h_layout.addWidget(self.acq_mode_toggle)

    def init_center_to_base_mode(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        self.center_to_base_toggle = QtWidgets.QCheckBox("Center to Base")
        self.center_to_base_toggle.setEnabled(False)
        h_layout.addWidget(self.center_to_base_toggle)

    def init_display_mode(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        display_mode_label = QtWidgets.QLabel("Display Mode:")
        h_layout.addWidget(display_mode_label)

        self.display_mode_combobox = QtWidgets.QComboBox()
        self.display_mode_combobox.addItem("Colors")
        self.display_mode_combobox.addItem("Normals")
        self.display_mode_combobox.addItem("Segmentation")
        h_layout.addWidget(self.display_mode_combobox)

    def init_view_layout(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        self.camera_view_button = QtWidgets.QPushButton("Camera view")
        h_layout.addWidget(self.camera_view_button)

        self.birds_eye_view_button = QtWidgets.QPushButton("Bird's eye view")
        h_layout.addWidget(self.birds_eye_view_button)

    def init_operate_info(self, layout:QtWidgets.QVBoxLayout):
        self.info_groupbox = QtWidgets.QGroupBox("Operate Info")
        self.info_groupbox.setCheckable(True)
        self.info_groupbox.setChecked(True)
        group_layout = QtWidgets.QVBoxLayout()
        self.info_groupbox.setLayout(group_layout)
        layout.addWidget(self.info_groupbox)

        self.mouse_coord = QtWidgets.QLabel("mouse coord: ")
        group_layout.addWidget(self.mouse_coord)

    def init_calibrate_layout(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        self.robot_init_button = QtWidgets.QPushButton("Robot Init")
        h_layout.addWidget(self.robot_init_button)

        self.cam_calib_init_button = QtWidgets.QPushButton("Cam Calib Init")
        h_layout.addWidget(self.cam_calib_init_button)

        self.handeye_calib_init_button = QtWidgets.QPushButton("HandEye Calib Init")
        h_layout.addWidget(self.handeye_calib_init_button)

    def init_calibration_settings(self, layout:QtWidgets.QVBoxLayout):
        groupbox = QtWidgets.QGroupBox("Calibration Settings")
        group_layout = QtWidgets.QVBoxLayout()
        groupbox.setLayout(group_layout)
        layout.addWidget(groupbox)

        # Aruco type
        h_layout = QtWidgets.QHBoxLayout()
        group_layout.addLayout(h_layout)
        aruco_label = QtWidgets.QLabel("Aruco type: ")
        h_layout.addWidget(aruco_label)
        self.board_type_combobox = QtWidgets.QComboBox()
        self.board_type_combobox.addItems(["Type1", "Type2", "Type3"])
        h_layout.addWidget(self.board_type_combobox)

        # Square Size and Marker Size
        h_layout = QtWidgets.QHBoxLayout()
        group_layout.addLayout(h_layout)
        square_size_label = QtWidgets.QLabel("Square Size:")
        h_layout.addWidget(square_size_label)
        self.board_square_size_num_edit = QtWidgets.QDoubleSpinBox()
        self.board_square_size_num_edit.setRange(0.01, 30.0)
        self.board_square_size_num_edit.setValue(23.0)
        h_layout.addWidget(self.board_square_size_num_edit)
        h_layout.addWidget(QtWidgets.QLabel("mm"))

        marker_size_label = QtWidgets.QLabel("Marker Size:")
        h_layout.addWidget(marker_size_label)
        self.board_marker_size_num_edit = QtWidgets.QDoubleSpinBox()
        self.board_marker_size_num_edit.setRange(0.01, 30.0)
        self.board_marker_size_num_edit.setValue(17.5)
        h_layout.addWidget(self.board_marker_size_num_edit)
        h_layout.addWidget(QtWidgets.QLabel("mm"))

        # Col and Row
        h_layout = QtWidgets.QHBoxLayout()
        group_layout.addLayout(h_layout)
        col_label = QtWidgets.QLabel("Col")
        h_layout.addWidget(col_label)
        self.board_col_num_edit = QtWidgets.QSpinBox()
        self.board_col_num_edit.setRange(5, 15)
        self.board_col_num_edit.setValue(11)
        h_layout.addWidget(self.board_col_num_edit)

        row_label = QtWidgets.QLabel("Row")
        h_layout.addWidget(row_label)
        self.board_row_num_edit = QtWidgets.QSpinBox()
        self.board_row_num_edit.setRange(5, 15)
        self.board_row_num_edit.setValue(6)
        h_layout.addWidget(self.board_row_num_edit)

    def init_calib_color_image_display(self, layout:QtWidgets.QVBoxLayout):
        self.calib_groupbox = QtWidgets.QGroupBox("Calibration image")
        self.calib_groupbox.setCheckable(True)
        self.calib_groupbox.setChecked(True)
        group_layout = QtWidgets.QVBoxLayout()
        self.calib_groupbox.setLayout(group_layout)
        layout.addWidget(self.calib_groupbox)

        self.calib_video = QtWidgets.QLabel()
        group_layout.addWidget(self.calib_video)

        h_layout = QtWidgets.QHBoxLayout()
        group_layout.addLayout(h_layout)
        self.detect_board_toggle = QtWidgets.QCheckBox("Detect Board")
        h_layout.addWidget(self.detect_board_toggle)
        self.show_axis_in_scene_toggle = QtWidgets.QCheckBox("Show Axis in Scene")
        h_layout.addWidget(self.show_axis_in_scene_toggle)

    def init_calibration_collect_layout(self, layout:QtWidgets.QVBoxLayout):
        self.calib_collect_button = QtWidgets.QPushButton("Collect Current Frame (Space)")
        layout.addWidget(self.calib_collect_button)

        self.frame_list_view = QtWidgets.QListWidget()
        self.frame_list_view.addItem('Click "Collect Current Frame" to start')
        layout.addWidget(self.frame_list_view)

        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)
        self.calib_list_remove_button = QtWidgets.QPushButton("Remove")
        h_layout.addWidget(self.calib_list_remove_button)
        self.robot_move_button = QtWidgets.QPushButton("Move Robot")
        h_layout.addWidget(self.robot_move_button)
        self.calib_button = QtWidgets.QPushButton("Calib (C)")
        h_layout.addWidget(self.calib_button)

        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)
        self.calib_op_save_button = QtWidgets.QPushButton("List Save")
        h_layout.addWidget(self.calib_op_save_button)
        self.calib_op_load_button = QtWidgets.QPushButton("List Load")
        h_layout.addWidget(self.calib_op_load_button)
        self.calib_op_run_button = QtWidgets.QPushButton("List Rerun")
        h_layout.addWidget(self.calib_op_run_button)

        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)
        frame_folder_label = QtWidgets.QLabel("Calib Path:")
        h_layout.addWidget(frame_folder_label)
        self.calib_save_text = QtWidgets.QLineEdit()
        h_layout.addWidget(self.calib_save_text)

        self.calib_save_button = QtWidgets.QPushButton("Save Calibration and Check")
        layout.addWidget(self.calib_save_button)

    def init_calibrate_set(self, layout:QtWidgets.QVBoxLayout):
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)
        stream_label = QtWidgets.QLabel("Load calib: ")
        h_layout.addWidget(stream_label)
        self.calib_combobox = QtWidgets.QComboBox()
        self.calib_combobox.addItem("None")
        self.calib_combobox.setEnabled(False)
        h_layout.addWidget(self.calib_combobox)
        self.calib_check_button = QtWidgets.QPushButton("check")
        h_layout.addWidget(self.calib_check_button)

    def data_collect_layout(self, layout:QtWidgets.QVBoxLayout):
        self.data_collect_button = QtWidgets.QPushButton("Collect Current Data")
        layout.addWidget(self.data_collect_button)

        prompt_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(prompt_layout)
        prompt_label = QtWidgets.QLabel("Prompt:")
        prompt_layout.addWidget(prompt_label)
        self.prompt_text = QtWidgets.QLineEdit()
        self.prompt_text.setPlaceholderText("Input prompt here...")
        prompt_layout.addWidget(self.prompt_text)

        self.data_tree_view = QtWidgets.QTreeWidget()
        self.data_tree_view.setHeaderLabel("Data Items")
        layout.addWidget(self.data_tree_view)

        list_operation_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(list_operation_layout)
        self.data_tree_view_remove_button = QtWidgets.QPushButton("Remove")
        list_operation_layout.addWidget(self.data_tree_view_remove_button)
        self.data_tree_view_load_button = QtWidgets.QPushButton("Load")
        list_operation_layout.addWidget(self.data_tree_view_load_button)

        data_folder_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(data_folder_layout)
        data_folder_label = QtWidgets.QLabel("Save to:")
        data_folder_layout.addWidget(data_folder_label)
        self.data_folder_text = QtWidgets.QLineEdit()
        self.data_folder_text.setPlaceholderText("Input data path here...")
        data_folder_layout.addWidget(self.data_folder_text)
        self.data_folder_select_button = QtWidgets.QPushButton("...")
        data_folder_layout.addWidget(self.data_folder_select_button)

        self.data_save_button = QtWidgets.QPushButton("Save Data")
        layout.addWidget(self.data_save_button)

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
        self.save_pcd_button.setEnabled(False)
        self.save_rgbd_button.setEnabled(False)
        self.detect_board_toggle.setEnabled(False)
        self.data_collect_button.setEnabled(False)
        
    def callback_bindings(self):
        # Binding callbacks to GUI elements
        # General Tab
        self.stream_init_button.clicked.connect(self.on_stream_init_button_clicked)
        self.toggle_record.stateChanged.connect(self.on_toggle_record_state_changed)
        self.save_pcd_button.clicked.connect(self.on_save_pcd_button_clicked)
        self.save_rgbd_button.clicked.connect(self.on_save_rgbd_button_clicked)

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
            if hasattr(self, 'fake_camera'):
                self.fake_camera.disconnect()
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
                self.timer.timeout.connect(self.update_frame)
                self.timer.start(10)  # Update at ~33 FPS
            else:
                self.status_message.setText("System: Failed to connect to Camera")

    def update_frame(self):
        current_frame_time = time.time()
        if self.frame_num > 0:  # Avoid calculation on the first frame
            time_diff = current_frame_time - self.prev_frame_time
            if time_diff > 0:  # Avoid division by zero
                self.real_fps = 1.0 / time_diff
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

        # Convert Open3D point cloud to numpy arrays
        points = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        vtk_points_array = vtk.vtkPoints()
        vtk_points_array.SetData(numpy_to_vtk(points))

        vtk_colors_array = numpy_to_vtk(colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_colors_array.SetName("Colors")

        # Create vertices
        num_points = len(points)
        vtk_cells = vtk.vtkCellArray()
        vtk_cells_array = np.hstack((np.ones((num_points, 1), dtype=np.int64),
                                    np.arange(num_points, dtype=np.int64).reshape(-1, 1))).flatten()
        vtk_cells_id_array = numpy_to_vtkIdTypeArray(vtk_cells_array, deep=True)
        vtk_cells.SetCells(num_points, vtk_cells_id_array)

        # Update the polydata
        self.polydata.SetPoints(vtk_points_array)
        self.polydata.GetPointData().SetScalars(vtk_colors_array)
        self.polydata.SetVerts(vtk_cells)
        self.polydata.Modified()

        # Clear the renderer and add the new actor
        # self.renderer.RemoveAllViewProps()

        # self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

        logger.debug("Point cloud visualization updated.")

    # Implement other callback methods as needed (even if they are empty for now)
    def on_toggle_record_state_changed(self, state):
        pass

    def on_save_pcd_button_clicked(self):
        pass

    def on_save_rgbd_button_clicked(self):
        pass

    def on_capture_toggle_state_changed(self, state):
        pass

    def on_seg_model_init_toggle_state_changed(self, state):
        pass

    def on_acq_mode_toggle_state_changed(self, state):
        pass

    def on_display_mode_combobox_changed(self, text):
        self.display_mode = text

    def on_robot_init_button_clicked(self):
        pass

    def on_calib_collect_button_clicked(self):
        pass

    def on_calib_button_clicked(self):
        pass

    def on_detect_board_toggle_state_changed(self, state):
        pass

    def on_data_collect_button_clicked(self):
        pass

    def on_data_save_button_clicked(self):
        pass

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
        self.robot_base_frame = vtk.vtkAxesActor()
        self.robot_base_frame.SetTotalLength(0.1, 0.1, 0.1)
        self.renderer.AddActor(self.robot_base_frame)
        # Robot end frame
        self.robot_end_frame = vtk.vtkAxesActor()
        self.robot_end_frame.SetTotalLength(0.1, 0.1, 0.1)
        self.renderer.AddActor(self.robot_end_frame)
        # Board pose
        self.board_pose_frame = vtk.vtkAxesActor()
        self.board_pose_frame.SetTotalLength(0.1, 0.1, 0.1)
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
        cube = vtk.vtkCubeSource()
        cube.SetBounds(
            self.bbox_params['xmin'], self.bbox_params['xmax'],
            self.bbox_params['ymin'], self.bbox_params['ymax'],
            self.bbox_params['zmin'], self.bbox_params['zmax']
        )

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())

        actor = vtk.vtkActor()
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
            matrix = vtk.vtkMatrix4x4()
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

    def set_disable_before_stream_init(self):
        """Disable UI elements before stream initialization."""
        self.capture_toggle.setEnabled(False)
        self.seg_model_init_toggle.setEnabled(False)
        self.handeye_calib_init_button.setEnabled(False)
        self.save_pcd_button.setEnabled(False)
        self.save_rgbd_button.setEnabled(False)
        self.detect_board_toggle.setEnabled(False)
        self.data_collect_button.setEnabled(False)

    # Include any other methods you need from your VTKWidgets class

def main():
    params = {
        'directory': '.',
        'Image_Amount': 13,
        'board_shape': (7, 10),
        'board_square_size': 23.5,
        'board_marker_size': 19,
        'input_method': 'auto_calibrated_mode',
        'folder_path': '_tmp',
        'pose_file_path': './poses.txt',
        'load_intrinsic': True,
        'calib_path': './Calibration_results/calibration_results.json',
        'device': 'cuda:0',
        'camera_config': './camera_config.json',
        'rgbd_video': None,
        'board_type': 'DICT_4X4_100',
        'data_path': './data',
        'load_in_startup': {
            'camera_init': True,
            'camera_calib_init': True,
            'robot_init': True,
            'handeye_calib_init': True,
            'calib_check': True,
            'collect_data_viewer': True
        },
        'use_fake_camera': True
    }
    app = QtWidgets.QApplication(sys.argv)
    callbacks = {
        # Define your callbacks here
        # 'stream_init_button': your_callback_function,
    }
    window = PipelineView(callbacks=callbacks, params=params)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
