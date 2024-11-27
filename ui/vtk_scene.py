import sys
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtWidgets, QtCore


class VTKWidgets(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(VTKWidgets, self).__init__(parent)
        self.setWindowTitle("VTK GUI Application")
        self.resize(1200, 800)
        self.initUI()

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
        self.renderer.SetBackground(1.0, 1.0, 1.0)  # White background

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
        self.vtk_widget.Start()

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

    def init_fps_label(self, layout:QtWidgets.QVBoxLayout):
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        layout.addWidget(self.fps_label)

    def init_status_message(self, layout:QtWidgets.QVBoxLayout):
        self.status_message = QtWidgets.QLabel("System: No Stream is Initialized")
        layout.addWidget(self.status_message)
        self.robot_msg = QtWidgets.QLabel("Robot: Not Connected")
        layout.addWidget(self.robot_msg)
        self.calibration_msg = QtWidgets.QLabel("Calibration: None")
        layout.addWidget(self.calibration_msg)

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

    def init_color_image_display(self, layout:QtWidgets.QVBoxLayout):
        self.color_groupbox = QtWidgets.QGroupBox("Color image")
        self.color_groupbox.setCheckable(True)
        self.color_groupbox.setChecked(True)
        group_layout = QtWidgets.QVBoxLayout()
        self.color_groupbox.setLayout(group_layout)
        layout.addWidget(self.color_groupbox)

        self.color_video = QtWidgets.QLabel()
        group_layout.addWidget(self.color_video)

    def init_depth_image_display(self, layout:QtWidgets.QVBoxLayout):
        self.depth_groupbox = QtWidgets.QGroupBox("Depth image")
        self.depth_groupbox.setCheckable(True)
        self.depth_groupbox.setChecked(True)
        group_layout = QtWidgets.QVBoxLayout()
        self.depth_groupbox.setLayout(group_layout)
        layout.addWidget(self.depth_groupbox)

        self.depth_video = QtWidgets.QLabel()
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



def main():
    app = QtWidgets.QApplication(sys.argv)
    window = VTKWidgets()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
