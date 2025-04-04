import sys
import math
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QLabel, QListWidget,
                             QApplication,QDoubleSpinBox, QCheckBox,
                             QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QSizePolicy, QTabWidget, 
                             QGroupBox, QMainWindow, QComboBox, 
                             QSplitter, QLineEdit, QSpinBox,
                             QTextEdit, QScrollArea, QGridLayout)


from app.utils import ARUCO_BOARD
from app.viewers.image_viewer import ResizableImageLabel
from .chat_ui_widget import ChatHistoryWidget
from .data_ui_widget import DataTreeWidget
from .robot_graph_widget import RobotPoseGraphWidget

import logging
logger = logging.getLogger(__name__)


class SceneViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.camera_widgets = {}  # Dictionary to store camera widgets by camera name
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Create a grid layout for cameras
        self.cam_grid_layout = QGridLayout()
        layout.addLayout(self.cam_grid_layout)
        
        # Add robot pose graph after camera grid
        self.robot_pose_graph = RobotPoseGraphWidget()
        layout.addWidget(self.robot_pose_graph)
        
        # For backward compatibility, keep the main and sub cam references
        main_cam = ResizableImageLabel()
        main_cam.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        sub_cam = ResizableImageLabel()
        sub_cam.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sub_cam.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add the default cameras to the grid
        self.add_camera_to_grid("main", main_cam, 0, 0)
        self.add_camera_to_grid("sub", sub_cam, 0, 1)
        
        self.main_cam = main_cam
        self.sub_cam = sub_cam
        
        self.setLayout(layout)
    
    def add_camera_to_grid(self, camera_name, camera_widget=None, row=None, col=None):
        """Add a camera to the grid at the specified position or auto-position"""
        if camera_widget is None:
            camera_widget = ResizableImageLabel(camera_name=camera_name)
            camera_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            camera_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        elif isinstance(camera_widget, ResizableImageLabel):
            camera_widget.set_camera_name(camera_name)
        
        # Create container for the camera widget
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(2, 2, 2, 2)
        container_layout.addWidget(camera_widget)
        
        # Determine grid position if not specified
        if row is None or col is None:
            count = len(self.camera_widgets)
            # Calculate optimal grid size
            grid_size = max(1, math.ceil(math.sqrt(count + 1)))
            row = count // grid_size
            col = count % grid_size
        
        self.cam_grid_layout.addWidget(container, row, col)
        self.camera_widgets[camera_name] = {
            "widget": camera_widget,
            "container": container,
            "row": row,
            "col": col
        }
        
        return camera_widget
    
    def remove_camera(self, camera_name):
        """Remove a camera from the grid"""
        if camera_name in self.camera_widgets:
            camera_info = self.camera_widgets[camera_name]
            self.cam_grid_layout.removeWidget(camera_info["container"])
            camera_info["container"].deleteLater()
            del self.camera_widgets[camera_name]
            return True
        return False
    
    def clear_all_cameras(self):
        """Remove all cameras from the grid"""
        for camera_name in list(self.camera_widgets.keys()):
            self.remove_camera(camera_name)
    
    def reorganize_grid(self):
        """Reorganize the grid to make optimal use of space"""
        cameras = list(self.camera_widgets.items())
        if not cameras:
            return
        
        # Calculate optimal grid size
        count = len(cameras)
        grid_size = max(1, math.ceil(math.sqrt(count)))
        
        for i, (camera_name, camera_info) in enumerate(cameras):
            row = i // grid_size
            col = i % grid_size
            
            # If position changed, update layout
            if row != camera_info["row"] or col != camera_info["col"]:
                self.cam_grid_layout.removeWidget(camera_info["container"])
                self.cam_grid_layout.addWidget(camera_info["container"], row, col)
                camera_info["row"] = row
                camera_info["col"] = col



class SceneStreamerUI(QMainWindow):
    """Controls display and user interface using VTK and PySide6."""

    def __init__(self):
        super().__init__()
        # self.vtk_widget: QVTKRenderWindowInteractor = None
        self.setWindowTitle("Scene Streamer")
        self.resize(1200, 640)
        self.display_mode = 'Colors'  # Initialize display mode to 'Colors'
        self.initUI()
        

    def initUI(self):
        # Create a splitter to divide the window
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)
        self.viewer = SceneViewer()
        self.splitter.addWidget(self.viewer)

        # Right side: Panel
        self.panel = QWidget()
        self.panel_layout = QVBoxLayout(self.panel)
        self.splitter.addWidget(self.panel)
        self.splitter.setStretchFactor(0, 1)
        self.init_widgets()


    def init_widgets(self):
        # Initialize tab view
        self.init_tab_view(self.panel_layout)

        # Initialize widgets in tabs
        self.init_general_tab()
        self.init_calibration_tab()
        self.init_data_tab()
        self.init_agent_tab()  # Add the new Robot tab

        # Create System Info group
        system_info_group = QGroupBox("System Info")
        system_info_layout = QVBoxLayout()
        system_info_group.setLayout(system_info_layout)
        self.panel_layout.addWidget(system_info_group)

        # Initialize fps label and status message within the group
        self.init_fps_label(system_info_layout)
        self.init_status_message(system_info_layout)

        # Set initial states

    # Include all the init_* methods from your VTKWidgets class here
    # For example:
    def init_fps_label(self, layout: QVBoxLayout):
        self.fps_label = QLabel("FPS: 0")
        layout.addWidget(self.fps_label)

    def init_status_message(self, layout: QVBoxLayout):
        self.status_message = QLabel("System: No Stream is Initialized")
        layout.addWidget(self.status_message)
        self.robot_msg = QLabel("Robot: Not Connected")
        layout.addWidget(self.robot_msg)
        self.calibration_msg = QLabel("Calibration: None")
        layout.addWidget(self.calibration_msg)



    def init_tab_view(self, layout:QVBoxLayout):
        self.tab_view = QTabWidget()
        layout.addWidget(self.tab_view)

    def init_general_tab(self):
        self.general_tab = QWidget()
        self.tab_view.addTab(self.general_tab, "General")
        general_layout = QVBoxLayout(self.general_tab)

        self.init_stream_set(general_layout)
        # self.init_video_displays(general_layout)
        self.init_scene_info(general_layout)

    def init_calibration_tab(self):
        self.calibration_tab = QWidget()
        self.tab_view.addTab(self.calibration_tab, "Calib")
        calibration_layout = QVBoxLayout(self.calibration_tab)

        self.init_calibrate_layout(calibration_layout)
        self.init_calibration_settings(calibration_layout)
        self.init_calib_color_image_display(calibration_layout)
        self.init_calibration_collect_layout(calibration_layout)
        self.init_calibrate_set(calibration_layout)

    def init_data_tab(self):
        self.data_tab = QWidget()
        self.tab_view.addTab(self.data_tab, "Data")
        data_layout = QVBoxLayout(self.data_tab)

        self.data_collect_layout(data_layout)

    def init_stream_set(self, layout:QVBoxLayout):
        v_layout = QVBoxLayout()
        layout.addLayout(v_layout)

        # Camera selection group
        camera_group = QGroupBox("Camera Management")
        camera_group_layout = QVBoxLayout()
        camera_group.setLayout(camera_group_layout)
        v_layout.addWidget(camera_group)

        # Camera selection controls
        selection_layout = QHBoxLayout()
        camera_group_layout.addLayout(selection_layout)
        
        # Camera type selection
        camera_type_label = QLabel("Camera Type: ")
        selection_layout.addWidget(camera_type_label)
        
        self.camera_type_combobox = QComboBox()
        self.camera_type_combobox.addItems(["USB Camera", "HTTP Camera"])
        selection_layout.addWidget(self.camera_type_combobox)
        
        # Connect camera type change to show/hide appropriate controls
        self.camera_type_combobox.currentTextChanged.connect(self.on_camera_type_changed)
        
        # Camera dropdown for USB cameras (in a container)
        self.usb_camera_container = QWidget()
        usb_camera_layout = QHBoxLayout(self.usb_camera_container)
        usb_camera_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.addWidget(self.usb_camera_container)
        
        camera_label = QLabel("Camera: ")
        usb_camera_layout.addWidget(camera_label)
        
        self.camera_combobox = QComboBox()
        usb_camera_layout.addWidget(self.camera_combobox)
        
        # HTTP camera URL input (hidden by default)
        self.http_camera_container = QWidget()
        self.http_camera_layout = QHBoxLayout(self.http_camera_container)
        self.http_camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_group_layout.addWidget(self.http_camera_container)
        
        http_url_label = QLabel("HTTP URL: ")
        self.http_camera_layout.addWidget(http_url_label)
        
        self.http_camera_url = QLineEdit()
        self.http_camera_url.setPlaceholderText("http://camera-ip:port/stream")
        self.http_camera_layout.addWidget(self.http_camera_url)
        
        # Hide HTTP URL input by default
        self.http_camera_container.setVisible(False)
        
        # Camera name input
        name_layout = QHBoxLayout()
        camera_group_layout.addLayout(name_layout)
        name_label = QLabel("Camera Name: ")
        name_layout.addWidget(name_label)
        
        self.camera_name_input = QLineEdit()
        self.camera_name_input.setPlaceholderText("Enter camera name")
        name_layout.addWidget(self.camera_name_input)
        
        # Add camera button
        self.add_camera_button = QPushButton("Add Camera")
        name_layout.addWidget(self.add_camera_button)
        
        # Camera list widget
        self.camera_list_widget = QListWidget()
        camera_group_layout.addWidget(self.camera_list_widget)
        
        # Camera list controls
        list_controls_layout = QHBoxLayout()
        camera_group_layout.addLayout(list_controls_layout)
        
        self.remove_camera_button = QPushButton("Remove Selected")
        list_controls_layout.addWidget(self.remove_camera_button)
        
        self.clear_cameras_button = QPushButton("Clear All")
        list_controls_layout.addWidget(self.clear_cameras_button)
        
        # Initialize button
        self.main_init_button = QPushButton("Initialize Cameras")
        camera_group_layout.addWidget(self.main_init_button)
        
        # For backward compatibility
        self.main_camera_combobox = self.camera_combobox
        self.sub_camera_combobox = QComboBox()  # Hidden, just for compatibility
        
        # Robot group
        robot_group = QGroupBox("Robot Management")
        robot_group_layout = QVBoxLayout()
        robot_group.setLayout(robot_group_layout)
        v_layout.addWidget(robot_group)
        
        # Robot init
        self.robot_init_button = QPushButton("Robot Init")
        robot_group_layout.addWidget(self.robot_init_button)
        
        # Robot pose plot controls
        robot_plot_layout = QHBoxLayout()
        robot_group_layout.addLayout(robot_plot_layout)
        
        # Robot pose plot toggle
        self.plot_robot_pose_checkbox = QCheckBox("Plot Robot Pose")
        self.plot_robot_pose_checkbox.setChecked(True)  # Enable by default
        robot_plot_layout.addWidget(self.plot_robot_pose_checkbox)
        
        # Robot state type selection
        robot_plot_layout.addWidget(QLabel("State Type:"))
        self.robot_state_type_combobox = QComboBox()
        self.robot_state_type_combobox.addItems(["tcp", "joint"])
        robot_plot_layout.addWidget(self.robot_state_type_combobox)

    def on_camera_type_changed(self, camera_type):
        """Handle camera type combobox changes"""
        if camera_type == "HTTP Camera":
            self.usb_camera_container.setVisible(False)
            self.http_camera_container.setVisible(True)
        else:  # USB Camera
            self.usb_camera_container.setVisible(True)
            self.http_camera_container.setVisible(False)

    def init_record_save(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        record_label = QLabel("Record / Save")
        h_layout.addWidget(record_label)

        self.toggle_record = QCheckBox("Video")
        h_layout.addWidget(self.toggle_record)

    def init_save_layout(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        self.save_pcd_button = QPushButton("Save Point cloud")
        h_layout.addWidget(self.save_pcd_button)

        self.save_rgbd_button = QPushButton("Save RGBD frame")
        h_layout.addWidget(self.save_rgbd_button)

    def init_video_displays(self, layout:QVBoxLayout):
        self.init_color_image_display(layout)
        self.init_depth_image_display(layout)

    def init_color_image_display(self, layout: QVBoxLayout):
        self.color_groupbox = QGroupBox("Color image")
        self.color_groupbox.setCheckable(True)
        self.color_groupbox.setChecked(True)
        group_layout = QVBoxLayout()
        self.color_groupbox.setLayout(group_layout)
        layout.addWidget(self.color_groupbox)

        self.color_video = ResizableImageLabel()
        self.color_video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.color_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group_layout.addWidget(self.color_video)

    def init_depth_image_display(self, layout: QVBoxLayout):
        self.depth_groupbox = QGroupBox("Depth image")
        self.depth_groupbox.setCheckable(True)
        self.depth_groupbox.setChecked(True)
        group_layout = QVBoxLayout()
        self.depth_groupbox.setLayout(group_layout)
        layout.addWidget(self.depth_groupbox)

        self.depth_video = ResizableImageLabel()
        self.depth_video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.depth_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        group_layout.addWidget(self.depth_video)
        
    def init_calib_color_image_display(self, layout:QVBoxLayout):
        self.calib_groupbox = QGroupBox("Calibration image")
        self.calib_groupbox.setCheckable(True)
        self.calib_groupbox.setChecked(True)
        group_layout = QVBoxLayout()
        self.calib_groupbox.setLayout(group_layout)
        layout.addWidget(self.calib_groupbox)

        self.calib_video = QLabel()
        group_layout.addWidget(self.calib_video)

        h_layout = QHBoxLayout()
        group_layout.addLayout(h_layout)
        self.detect_board_toggle = QCheckBox("Detect Board")
        h_layout.addWidget(self.detect_board_toggle)
        self.show_axis_in_scene_button = QPushButton("Show Axis in Scene")
        h_layout.addWidget(self.show_axis_in_scene_button)

    def init_scene_info(self, layout:QVBoxLayout):
        self.scene_info = QGroupBox("Scene Info")
        self.scene_info.setCheckable(True)
        self.scene_info.setChecked(False)
        group_layout = QVBoxLayout()
        self.scene_info.setLayout(group_layout)
        layout.addWidget(self.scene_info)

        self.view_status = QLabel("")
        group_layout.addWidget(self.view_status)


    def init_calibrate_layout(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        self.cam_calib_init_button = QPushButton("Cam Calib Init")
        h_layout.addWidget(self.cam_calib_init_button)

        # self.handeye_calib_init_button = QPushButton("HandEye Calib Init")
        # h_layout.addWidget(self.handeye_calib_init_button)

    def init_calibration_settings(self, layout:QVBoxLayout):
        groupbox = QGroupBox("Calibration Settings")
        group_layout = QVBoxLayout()
        groupbox.setLayout(group_layout)
        layout.addWidget(groupbox)

        # Aruco type
        h_layout = QHBoxLayout()
        group_layout.addLayout(h_layout)
        aruco_label = QLabel("Aruco type: ")
        h_layout.addWidget(aruco_label)
        self.board_type_combobox = QComboBox()
        for item in ARUCO_BOARD.keys():
            self.board_type_combobox.addItem(item)
        # self.board_type_combobox.addItems(["Type1", "Type2", "Type3"])
        h_layout.addWidget(self.board_type_combobox)

        # Square Size and Marker Size
        h_layout = QHBoxLayout()
        group_layout.addLayout(h_layout)
        square_size_label = QLabel("Square Size:")
        h_layout.addWidget(square_size_label)
        self.board_square_size_num_edit = QDoubleSpinBox()
        self.board_square_size_num_edit.setRange(0.01, 30.0)
        self.board_square_size_num_edit.setValue(23.0)
        h_layout.addWidget(self.board_square_size_num_edit)
        h_layout.addWidget(QLabel("mm"))

        marker_size_label = QLabel("Marker Size:")
        h_layout.addWidget(marker_size_label)
        self.board_marker_size_num_edit = QDoubleSpinBox()
        self.board_marker_size_num_edit.setRange(0.01, 30.0)
        self.board_marker_size_num_edit.setValue(17.5)
        h_layout.addWidget(self.board_marker_size_num_edit)
        h_layout.addWidget(QLabel("mm"))

        # Col and Row
        h_layout = QHBoxLayout()
        group_layout.addLayout(h_layout)
        col_label = QLabel("Col")
        h_layout.addWidget(col_label)
        self.board_col_num_edit = QSpinBox()
        self.board_col_num_edit.setRange(5, 15)
        self.board_col_num_edit.setValue(11)
        h_layout.addWidget(self.board_col_num_edit)

        row_label = QLabel("Row")
        h_layout.addWidget(row_label)
        self.board_row_num_edit = QSpinBox()
        self.board_row_num_edit.setRange(5, 15)
        self.board_row_num_edit.setValue(6)
        h_layout.addWidget(self.board_row_num_edit)

    def init_calibration_collect_layout(self, layout:QVBoxLayout):
        self.calib_collect_button = QPushButton("Collect Current Frame (Space)")
        layout.addWidget(self.calib_collect_button)

        self.calib_data_list = QListWidget()
        self.calib_data_list.addItem('Click "Collect Current Frame" to start')
        layout.addWidget(self.calib_data_list)

        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)
        self.calib_list_remove_button = QPushButton("Remove")
        h_layout.addWidget(self.calib_list_remove_button)
        self.robot_move_button = QPushButton("Move Robot")
        h_layout.addWidget(self.robot_move_button)
        self.calib_button = QPushButton("Calib (C)")
        h_layout.addWidget(self.calib_button)

        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)
        self.calib_op_save_button = QPushButton("List Save")
        h_layout.addWidget(self.calib_op_save_button)
        self.calib_op_load_button = QPushButton("List Load")
        h_layout.addWidget(self.calib_op_load_button)
        self.calib_op_run_button = QPushButton("List Rerun")
        h_layout.addWidget(self.calib_op_run_button)

        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)
        frame_folder_label = QLabel("Calib Path:")
        h_layout.addWidget(frame_folder_label)
        self.calib_save_text = QLineEdit()
        h_layout.addWidget(self.calib_save_text)

        self.calib_save_button = QPushButton("Save Calibration and Check")
        layout.addWidget(self.calib_save_button)

    def init_calibrate_set(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)
        stream_label = QLabel("Load calib: ")
        h_layout.addWidget(stream_label)
        self.calib_combobox = QComboBox()
        self.calib_combobox.addItem("None")
        self.calib_combobox.setEnabled(False)
        h_layout.addWidget(self.calib_combobox)
        self.calib_check_button = QPushButton("check")
        h_layout.addWidget(self.calib_check_button)

    def data_collect_layout(self, layout:QVBoxLayout):
        self.data_collect_button = QPushButton("Collect Current Data")
        layout.addWidget(self.data_collect_button)

        prompt_layout = QHBoxLayout()
        layout.addLayout(prompt_layout)
        prompt_label = QLabel("Prompt:")
        prompt_layout.addWidget(prompt_label)
        self.prompt_text = QLineEdit()
        self.prompt_text.setPlaceholderText("Input prompt here...")
        prompt_layout.addWidget(self.prompt_text)

        self.data_tree_view = DataTreeWidget()
        self.data_tree_view.setHeaderLabel("Episodes:")
        layout.addWidget(self.data_tree_view)

        list_operation_layout = QHBoxLayout()
        layout.addLayout(list_operation_layout)
        self.data_tree_view_remove_button = QPushButton("Remove")
        list_operation_layout.addWidget(self.data_tree_view_remove_button)
        self.data_tree_view_load_button = QPushButton("Load")
        list_operation_layout.addWidget(self.data_tree_view_load_button)
        self.data_replay_and_save_button = QPushButton("Replay and Save")
        list_operation_layout.addWidget(self.data_replay_and_save_button)

        data_folder_layout = QHBoxLayout()
        layout.addLayout(data_folder_layout)
        data_folder_label = QLabel("Save to:")
        data_folder_layout.addWidget(data_folder_label)
        self.data_folder_text = QLineEdit()
        self.data_folder_text.setPlaceholderText("Input data path here...")
        data_folder_layout.addWidget(self.data_folder_text)
        self.data_folder_select_button = QPushButton("...")
        data_folder_layout.addWidget(self.data_folder_select_button)

        self.data_save_button = QPushButton("Save Data")
        layout.addWidget(self.data_save_button)

    def init_agent_tab(self):
        self.agent_tab = QWidget()
        self.tab_view.addTab(self.agent_tab, "Agent")
        robot_layout = QVBoxLayout(self.agent_tab)

        # Use font size as the base unit (em)
        em = self.font().pointSize()

        # Agent server connection group
        llm_server_group = QGroupBox("Agent Server")
        connection_layout = QHBoxLayout()
        llm_server_group.setLayout(connection_layout)

        ip_label = QLabel("IP:")
        self.ip_editor = QLineEdit("localhost")
        self.ip_editor.setMinimumWidth(em * 10)

        port_label = QLabel("Port:")
        self.port_editor = QLineEdit("65432")
        self.port_editor.setMinimumWidth(em * 5)

        self.scan_button = QPushButton("Scan")
        self.scan_button.setMinimumWidth(em * 6)
        self.scan_button.setMinimumHeight(em * 2)

        connection_layout.addWidget(ip_label)
        connection_layout.addWidget(self.ip_editor)
        connection_layout.addWidget(port_label)
        connection_layout.addWidget(self.port_editor)
        connection_layout.addWidget(self.scan_button)

        robot_layout.addWidget(llm_server_group)

        # Create a QSplitter for the chat history and prompt editor sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        robot_layout.addWidget(splitter)

        # Scroll area for chat history
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.chat_history = ChatHistoryWidget(self.scroll_area, em=em)
        self.chat_history.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area.setWidget(self.chat_history)

        # Add chat history to the splitter
        splitter.addWidget(self.scroll_area)

        # Group box for prompt editor and send button
        prompt_group = QGroupBox()
        prompt_group_layout = QVBoxLayout(prompt_group)
        prompt_group_layout.setContentsMargins(em, em, em, em)

        # Add User QLabel
        prompt_label = QLabel("User:")
        prompt_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        prompt_group_layout.addWidget(prompt_label)

        # Prompt editor
        self.agent_prompt_editor = QTextEdit()
        single_line_height = em * 2
        self.agent_prompt_editor.setMinimumHeight(single_line_height)
        self.agent_prompt_editor.setMaximumHeight(em * 8)
        self.agent_prompt_editor.document().documentLayout().documentSizeChanged.connect(
            lambda size: self.agent_prompt_editor.setFixedHeight(
                max(single_line_height, int(min(size.height() + em, em * 8)))
            )
        )
        prompt_group_layout.addWidget(self.agent_prompt_editor)

        # Send button layout
        send_button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.reset_button.setMaximumWidth(int(em * 10))
        self.reset_button.setMinimumHeight(int(em * 2.5))
        send_button_layout.addWidget(self.reset_button)
        
        self.send_button = QPushButton("Send")
        self.send_button.setMaximumWidth(int(em * 10))
        self.send_button.setMinimumHeight(int(em * 2.5))
        send_button_layout.addStretch()
        send_button_layout.addWidget(self.send_button)
        prompt_group_layout.addLayout(send_button_layout)

        # Add prompt group to the splitter
        splitter.addWidget(prompt_group)

        # Set initial splitter sizes
        splitter.setStretchFactor(0, 9)  # Chat history gets 90% of space
        splitter.setStretchFactor(1, 1)  # Prompt editor gets 10% of space
        splitter.setSizes([em * 50, em * 10])  # Set a default size distribution




def main():
    app = QApplication(sys.argv)
    window = SceneStreamerUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
