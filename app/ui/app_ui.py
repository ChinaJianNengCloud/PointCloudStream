import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QLabel, QListWidget,
                             QApplication,QDoubleSpinBox, QCheckBox,
                             QVBoxLayout, QHBoxLayout, QWidget, 
                             QPushButton, QSizePolicy, QTabWidget, 
                             QGroupBox, QMainWindow, QComboBox, 
                             QSplitter, QLineEdit, QSpinBox,
                             QTextEdit, QScrollArea)


from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkActor,
    vtkPolyDataMapper,
    vtkFollower
)
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkRenderingFreeType import vtkVectorText
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget 
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from app.utils import ARUCO_BOARD
from app.viewers.image_viewer import ResizableImageLabel
from .chat_ui_widget import ChatHistoryWidget
from .data_ui_widget import DataTreeWidget

from app.utils.logger import setup_logger
logger = setup_logger(__name__)


class PCDStreamerUI(QMainWindow):
    """Controls display and user interface using VTK and PyQt5."""

    def __init__(self):
        super().__init__()
        self.vtk_widget: QVTKRenderWindowInteractor = None
        self.setWindowTitle("VTK GUI Application")
        self.resize(1200, 640)
        self.display_mode = 'Colors'  # Initialize display mode to 'Colors'
        self.initUI()
        

    def initUI(self):
        # Create a splitter to divide the window
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)
        
        # Left side: VTK render window
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.splitter.addWidget(self.vtk_widget)

        # Right side: Panel
        self.panel = QWidget()
        self.panel_layout = QVBoxLayout(self.panel)
        self.splitter.addWidget(self.panel)
        self.splitter.setStretchFactor(0, 1)
        # Set up VTK renderer
        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.5, 0.5, 0.5)  # White background

        camera = self.renderer.GetActiveCamera()
        camera.Roll(45)  # Rotate the camera by 45 degrees
        # camera.eye_transform_matrix
        # Create an interactor style
        style = vtkInteractorStyleTrackballCamera()
        self.vtk_widget.SetInteractorStyle(style)
        self.init_pcd_polydata()
        # Add axes at corner
        self.axes = vtkAxesActor()
        self.axes_widget = vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(self.axes)
        self.axes_widget.SetInteractor(self.vtk_widget)
        self.axes_widget.SetEnabled(1)
        self.axes_widget.InteractiveOn()

        self.init_widgets()

        # Start the interactor
        self.vtk_widget.Initialize()
        # self.vtk_widget.Start()

    def init_pcd_polydata(self):
        vtk_points = vtkPoints()
        vtk_colors = vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")

        self.polydata = vtkPolyData()
        self.polydata.SetPoints(vtk_points)
        self.polydata.GetPointData().SetScalars(vtk_colors)

        vertices = vtkCellArray()
        self.polydata.SetVerts(vertices)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(self.polydata)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)
        self.renderer.AddActor(actor)

    def init_widgets(self):
        # Initialize tab view
        self.init_tab_view(self.panel_layout)

        # Initialize widgets in tabs
        self.init_general_tab()
        self.init_view_tab()
        self.init_calibration_tab()
        self.init_bbox_tab()
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
        self.set_disable_before_stream_init()

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

    def add_3d_label(self, position, text):
        text_source = vtkVectorText()
        text_source.SetText(text)

        text_mapper = vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())

        text_actor = vtkFollower()
        text_actor.SetMapper(text_mapper)
        text_actor.SetPosition(position)
        text_actor.SetScale(0.1, 0.1, 0.1)
        text_actor.SetCamera(self.renderer.GetActiveCamera())
        self.renderer.AddActor(text_actor)

    def init_tab_view(self, layout:QVBoxLayout):
        self.tab_view = QTabWidget()
        layout.addWidget(self.tab_view)

    def init_general_tab(self):
        self.general_tab = QWidget()
        self.tab_view.addTab(self.general_tab, "General")
        general_layout = QVBoxLayout(self.general_tab)

        self.init_stream_set(general_layout)
        self.init_video_displays(general_layout)
        self.init_scene_info(general_layout)

    def init_view_tab(self):
        self.view_tab = QWidget()
        self.tab_view.addTab(self.view_tab, "View")
        view_layout = QVBoxLayout(self.view_tab)

        # Initialize widgets in view tab
        self.init_toggle_view_set(view_layout)
        self.init_aqui_mode(view_layout)
        self.init_center_to_base_mode(view_layout)
        self.init_display_mode(view_layout)
        self.init_view_layout(view_layout)
        self.init_operate_info(view_layout)

    def init_calibration_tab(self):
        self.calibration_tab = QWidget()
        self.tab_view.addTab(self.calibration_tab, "Calib")
        calibration_layout = QVBoxLayout(self.calibration_tab)

        self.init_calibrate_layout(calibration_layout)
        self.init_calibration_settings(calibration_layout)
        self.init_calib_color_image_display(calibration_layout)
        self.init_calibration_collect_layout(calibration_layout)
        self.init_calibrate_set(calibration_layout)

    def init_bbox_tab(self):
        self.bbox_tab = QWidget()
        self.tab_view.addTab(self.bbox_tab, "Bbox")
        bbox_layout = QVBoxLayout(self.bbox_tab)

        self.init_bbox_controls(bbox_layout)

    def init_data_tab(self):
        self.data_tab = QWidget()
        self.tab_view.addTab(self.data_tab, "Data")
        data_layout = QVBoxLayout(self.data_tab)

        self.data_collect_layout(data_layout)

    def init_stream_set(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        stream_label = QLabel("Stream Init: ")
        h_layout.addWidget(stream_label)

        self.stream_combobox = QComboBox()
        self.stream_combobox.addItem("Camera")
        self.stream_combobox.addItem("Video")
        h_layout.addWidget(self.stream_combobox)

        self.stream_init_button = QPushButton("Start")
        h_layout.addWidget(self.stream_init_button)

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

    def init_toggle_view_set(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        self.capture_toggle = QCheckBox("Capture / Play")
        h_layout.addWidget(self.capture_toggle)

        self.seg_model_init_toggle = QCheckBox("Seg Mode")
        h_layout.addWidget(self.seg_model_init_toggle)

    def init_aqui_mode(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        self.acq_mode_toggle = QCheckBox("Acquisition Mode")
        h_layout.addWidget(self.acq_mode_toggle)

    def init_center_to_base_mode(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        self.center_to_robot_base_toggle = QCheckBox("Center to Base")
        self.center_to_robot_base_toggle.setEnabled(False)
        h_layout.addWidget(self.center_to_robot_base_toggle)

    def init_display_mode(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        display_mode_label = QLabel("Display Mode:")
        h_layout.addWidget(display_mode_label)

        self.display_mode_combobox = QComboBox()
        self.display_mode_combobox.addItem("Colors")
        # self.display_mode_combobox.addItem("Normals")
        self.display_mode_combobox.addItem("Segmentation")
        h_layout.addWidget(self.display_mode_combobox)

    def init_view_layout(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        self.camera_view_button = QPushButton("Camera view")
        h_layout.addWidget(self.camera_view_button)

        self.birds_eye_view_button = QPushButton("Bird's eye view")
        h_layout.addWidget(self.birds_eye_view_button)

    def init_operate_info(self, layout:QVBoxLayout):
        self.info_groupbox = QGroupBox("Operate Info")
        self.info_groupbox.setCheckable(True)
        self.info_groupbox.setChecked(True)
        group_layout = QVBoxLayout()
        self.info_groupbox.setLayout(group_layout)
        layout.addWidget(self.info_groupbox)

        self.mouse_coord = QLabel("mouse coord: ")
        group_layout.addWidget(self.mouse_coord)

    def init_calibrate_layout(self, layout:QVBoxLayout):
        h_layout = QHBoxLayout()
        layout.addLayout(h_layout)

        self.robot_init_button = QPushButton("Robot Init")
        h_layout.addWidget(self.robot_init_button)

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
    window = PCDStreamerUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
