import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import logging
from utils import ARUCO_BOARD

logger = logging.getLogger(__name__)


class SceneWidgets:
    def __init__(self, window, callbacks):
        self.window = window
        self.callbacks = callbacks
        self.em = self.window.theme.font_size
        self.__init_pcd_view()
        self.__init_panel()
        self.__init_widgets()

    def __init_pcd_view(self):
        self.pcdview = gui.SceneWidget()
        self.pcdview.enable_scene_caching(True)  # makes UI _much_ more responsive
        self.pcdview.scene = rendering.Open3DScene(self.window.renderer)
        self.pcdview.scene.set_background([1, 1, 1, 1])  # White background
        self.pcdview.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])
        self.pcdview.scene.show_axes(True)
        self.pcdview.scene.show_skybox(enable=False)
        self.pcdview.add_3d_label([0, 0, 0], "Camera")
        self.pcdview.add_3d_label([1, 0, 0], "X")
        self.pcdview.add_3d_label([0, 1, 0], "Y")
        self.pcdview.add_3d_label([0, 0, 1], "Z")
        self.window.add_child(self.pcdview)


    def __init_panel(self):
        self.panel = gui.Vert(self.em, gui.Margins(self.em, self.em, self.em, self.em))
        self.window.add_child(self.panel)

    def __init_widgets(self):
        # Initialize all widgets by calling their respective initialization functions
        self.__init_fps_label(parent_layout=self.window)

        # panel
        self.__init_status_message(parent_layout=self.panel)
        self.__init_tab_view(parent_layout=self.panel)

        # view tab
        self.__init_toggle_view_set(parent_layout=self.view_tab)
        self.__init_toggle_aqui_mode(parent_layout=self.view_tab)
        self.__init_calibration_mode(parent_layout=self.view_tab)
        self.__init_display_mode_combobox(parent_layout=self.view_tab)
        self.__init_calibrate_set(parent_layout=self.view_tab)
        self.__init_view_buttons(parent_layout=self.view_tab)
        self.__init_operate_info(parent_layout=self.view_tab)

        # general tab
        self.__init_stream_set(parent_layout=self.general_tab)
        self.__init_save_toggles(parent_layout=self.general_tab)
        self.__init_save_buttons(parent_layout=self.general_tab)
        self.__init_video_displays(parent_layout=self.general_tab)
        self.__init_scene_info(parent_layout=self.general_tab)

        # hardware tab
        self.__init_robot_button(parent_layout=self.calibration_tab)
        self.__init_calibration_settings(parent_layout=self.calibration_tab)
        self.__init_calibrate_button(parent_layout=self.calibration_tab)
        self.__calib_color_image_display(parent_layout=self.calibration_tab)
        # bbox tab
        self.__init_bbox_controls(parent_layout=self.bbox_tab)

        # data tab
        self.__data_collect_layout(parent_layout=self.data_tab)

        self.set_disable_before_stream_init()

    def __init_tab_view(self, parent_layout=None):
        self.tab_view = gui.TabControl()
        parent_layout.add_child(self.tab_view)
        self.general_tab = gui.Vert(self.em, gui.Margins(0, self.em, self.em // 2, self.em))
        self.tab_view.add_tab("General", self.general_tab)

        self.view_tab = gui.Vert(self.em, gui.Margins(0, self.em, self.em // 2, self.em))
        self.tab_view.add_tab("View", self.view_tab)

        self.bbox_tab = gui.Vert(self.em, gui.Margins(0, self.em, self.em // 2, self.em))
        self.tab_view.add_tab("Bbox", self.bbox_tab)

        self.calibration_tab = gui.Vert(self.em, gui.Margins(0, self.em, self.em // 2, self.em))
        self.tab_view.add_tab("Calib", self.calibration_tab)

        self.data_tab = gui.Vert(self.em, gui.Margins(0, self.em, self.em // 2, self.em))
        self.tab_view.add_tab("Data", self.data_tab)

    def __data_collect_layout(self, parent_layout=None):
        layout = gui.Vert(self.em, gui.Margins(0, self.em, self.em // 2, self.em))
        parent_layout.add_child(layout)
        self.data_collect_button = gui.Button("Collect Current Data")
        self.data_collect_button.horizontal_padding_em = 1
        self.data_collect_button.vertical_padding_em = 0
        layout.add_child(self.data_collect_button)
        
        # self.record_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

        self.data_list_view = gui.ListView()
        # self.data_list_view.set_items(self.record_list)
        self.data_list_view.set_max_visible_items(5)
        layout.add_child(self.data_list_view)
        list_operation_layout = gui.Horiz()
        layout.add_child(list_operation_layout)
        self.data_list_remove_button = gui.Button("Remove")
        self.data_list_remove_button.horizontal_padding_em = 0.5
        self.data_list_remove_button.vertical_padding_em = 0
        list_operation_layout.add_child(self.data_list_remove_button)

        data_folder_layout = gui.Horiz()
        data_folder_layout.add_child(gui.Label("Save to:"))
        self.data_folder_text = gui.TextEdit()
        data_folder_layout.add_child(self.data_folder_text)
        data_folder_layout.add_fixed(0.25 * self.em)
        self.data_folder_select = gui.Button("...")
        self.data_folder_select.horizontal_padding_em = 0.5
        self.data_folder_select.vertical_padding_em = 0
        data_folder_layout.add_child(self.data_folder_select)
        layout.add_child(data_folder_layout)


        self.data_save_button = gui.Button("Save Data")
        self.data_save_button.horizontal_padding_em = 1
        self.data_save_button.vertical_padding_em = 0
        layout.add_child(self.data_save_button)


    def __init_fps_label(self, parent_layout=None):
        self.fps_label = gui.Label("FPS: 99")
        parent_layout.add_child(self.fps_label)

    def __init_stream_set(self, parent_layout=None):
        layout = gui.Horiz(self.em)
        parent_layout.add_child(layout)
        stream_label = gui.Label("Stream Init: ")
        layout.add_child(stream_label)
        self.stream_combbox = gui.Combobox()
        self.stream_combbox.add_item("Camera")
        self.stream_combbox.add_item("Video")
        layout.add_child(self.stream_combbox)
        self.stream_combbox.selected_text = "Camera"
        self.stream_init_start = gui.Button("Start")
        self.stream_init_start.horizontal_padding_em = 1
        self.stream_init_start.vertical_padding_em = 0
        layout.add_child(self.stream_init_start)


    def __init_calibrate_set(self, parent_layout=None):
        layout = gui.Horiz(self.em)
        parent_layout.add_child(layout)
        stream_label = gui.Label("Check calib: ")
        layout.add_child(stream_label)
        self.calib_combobox = gui.Combobox()
        self.calib_combobox.add_item("None      ")
        self.calib_combobox.enabled = False
        # self.stream_combbox.add_item("Video")
        layout.add_child(self.calib_combobox)

        # self.stream_combbox.selected_text = "Camera"
        self.calib_check = gui.Button("check")
        self.calib_check.horizontal_padding_em = 1
        self.calib_check.vertical_padding_em = 0
        layout.add_child(self.calib_check)


    def __init_robot_button(self, parent_layout=None):
        button_layout = gui.Horiz(self.em)
        parent_layout.add_child(button_layout)
        # button_layout.add_stretch()
        self.robot_button = gui.Button("Robot Init")
        self.robot_button.horizontal_padding_em = 0.5
        self.robot_button.vertical_padding_em = 0
    
        button_layout.add_child(self.robot_button)
        # button_layout.add_stretch()
        
    def __init_status_message(self, parent_layout=None):
        self.status_message = gui.Label("System: No Stream is Initialized                 ")
        parent_layout.add_child(self.status_message)

        self.robot_msg = gui.Label("Robot: Not Connected                             ")
        parent_layout.add_child(self.robot_msg)

        self.calibration_msg = gui.Label("Calibration: None                           ")
        parent_layout.add_child(self.calibration_msg)


    def __init_calibration_settings(self, parent_layout=None):

        setting_layout = gui.Vert(self.em)
        
        board_type_layout = gui.Horiz(self.em)
        board_type_text = gui.Label("Aruco type: ")
        self.board_type_combobox = gui.Combobox()
        [self.board_type_combobox.add_item(b_type) 
            for b_type in 
            sorted(ARUCO_BOARD.keys(), key=lambda x: int(x.split('_')[1][0]))]
        
        board_type_layout.add_child(board_type_text)
        board_type_layout.add_child(self.board_type_combobox)

        setting_layout.add_child(board_type_layout)

        
        self.board_square_size = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.board_square_size.double_value = 23
        self.board_square_size.set_limits(0.01, 30)
        self.board_marker_size = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.board_marker_size.double_value = 17.5
        self.board_marker_size.set_limits(0.01, 30)

        numlayout = gui.Horiz(self.em // 2)
        numlayout.add_child(gui.Label("Square Size:"))
        numlayout.add_child(self.board_square_size)
        numlayout.add_child(gui.Label("mm"))
        setting_layout.add_child(numlayout)
        numlayout = gui.Horiz(self.em // 2)
        numlayout.add_child(gui.Label("Marker Size:"))
        numlayout.add_child(self.board_marker_size)
        numlayout.add_child(gui.Label("mm"))
        numlayout.add_fixed(self.em)
        setting_layout.add_child(numlayout)

        setting_layout.add_fixed(self.em)

        self.chessboard_col = gui.NumberEdit(gui.NumberEdit.INT)
        self.chessboard_col.int_value = 11
        self.chessboard_col.set_limits(5, 15)  # value coerced to 1
        numlayout = gui.Horiz()
        numlayout.add_child(gui.Label("Col"))
        numlayout.add_child(self.chessboard_col)
        numlayout.add_fixed(self.em)
    
        self.chessboard_row = gui.NumberEdit(gui.NumberEdit.INT)
        self.chessboard_row.int_value = 6
        self.chessboard_row.set_limits(5, 15)  # value coerced to 1
        numlayout.add_child(gui.Label("Row"))
        numlayout.add_child(self.chessboard_row)
        numlayout.add_fixed(self.em)
        
        setting_layout.add_child(numlayout)

        parent_layout.add_child(setting_layout)

    def __init_calibrate_button(self, parent_layout=None):
        button_layout = gui.Horiz(self.em)
        parent_layout.add_child(button_layout)
        # button_layout.add_stretch()
        self.cam_calibreate_button = gui.Button("Camera Calibration")
        self.cam_calibreate_button.tooltip = "Start Camera Calibration"
        self.cam_calibreate_button.horizontal_padding_em = 0.5
        self.cam_calibreate_button.vertical_padding_em = 0
        button_layout.add_child(self.cam_calibreate_button)

        self.he_calibreate_button = gui.Button("H-E Calibration")
        self.he_calibreate_button.tooltip = "Start Hand-Eye Calibration"
        self.he_calibreate_button.horizontal_padding_em = 0.5
        self.he_calibreate_button.vertical_padding_em = 0
        button_layout.add_child(self.he_calibreate_button)
        # button_layout.add_stretch()
        

    def __init_toggle_view_set(self, parent_layout=None):
        toggle_layout = gui.Horiz(self.em)
        parent_layout.add_child(toggle_layout)
        # toggle_layout.add_stretch()

        self.toggle_capture = gui.ToggleSwitch("Capture / Play")
        self.toggle_capture.is_on = False
        toggle_layout.add_child(self.toggle_capture)

        self.toggle_model_init = gui.ToggleSwitch("Seg Mode")
        self.toggle_model_init.is_on = False
        toggle_layout.add_child(self.toggle_model_init)

        # toggle_layout.add_stretch()


    def __init_toggle_aqui_mode(self, parent_layout=None):
        edit_mode_toggle_2 = gui.Horiz(self.em)
        parent_layout.add_child(edit_mode_toggle_2)

        self.toggle_acq_mode = gui.ToggleSwitch("Acquisition Mode")
        self.toggle_acq_mode.is_on = False
        edit_mode_toggle_2.add_child(self.toggle_acq_mode)

    def __init_calibration_mode(self, parent_layout=None):
        edit_mode_toggle_2 = gui.Horiz(self.em)
        parent_layout.add_child(edit_mode_toggle_2)

        self.calibration_mode = gui.ToggleSwitch("Calibration Mode")
        self.calibration_mode.is_on = False
        self.calibration_mode.enabled = False
        edit_mode_toggle_2.add_child(self.calibration_mode)


    def __init_display_mode_combobox(self, parent_layout=None):
        display_mode_layout = gui.Horiz(self.em)
        parent_layout.add_child(display_mode_layout)

        display_mode_label = gui.Label("Display Mode:")
        display_mode_layout.add_child(display_mode_label)

        self.display_mode_combobox = gui.Combobox()
        self.display_mode_combobox.add_item("Colors")
        self.display_mode_combobox.add_item("Normals")
        self.display_mode_combobox.add_item("Segmentation")
        self.display_mode_combobox.selected_text = "Colors"
        # Callback to be set later in PipelineView
        display_mode_layout.add_child(self.display_mode_combobox)

    def __init_view_buttons(self, parent_layout=None):
        view_buttons = gui.Horiz(self.em)
        parent_layout.add_child(view_buttons)
        view_buttons.add_stretch()  # for centering
        self.__init_camera_view_button(view_buttons)
        self.__init_birds_eye_view_button(view_buttons)
        view_buttons.add_stretch()  # for centering

    def __init_camera_view_button(self, parent_layout):
        self.camera_view_button = gui.Button("Camera view")
        self.camera_view_button.horizontal_padding_em = 0.5
        self.camera_view_button.vertical_padding_em = 0
        # Callback to be set later in PipelineView
        parent_layout.add_child(self.camera_view_button)

    def __init_birds_eye_view_button(self, parent_layout):
        self.birds_eye_view_button = gui.Button("Bird's eye view")
        self.birds_eye_view_button.horizontal_padding_em = 0.5
        self.birds_eye_view_button.vertical_padding_em = 0
        # Callback to be set later in PipelineView
        parent_layout.add_child(self.birds_eye_view_button)

    def __init_save_toggles(self, parent_layout=None):
        save_toggle = gui.Horiz(self.em)
        parent_layout.add_child(save_toggle)
        save_toggle.add_child(gui.Label("Record / Save"))
        self.toggle_record = None
        if self.callbacks['on_toggle_record'] is not None:
            save_toggle.add_fixed(1.5 * self.em)
            self.toggle_record = gui.ToggleSwitch("Video")
            self.toggle_record.is_on = False
            self.toggle_record.set_on_clicked(self.callbacks['on_toggle_record'])
            save_toggle.add_child(self.toggle_record)

    def __init_save_buttons(self, parent_layout=None):
        save_buttons = gui.Horiz(self.em)
        parent_layout.add_child(save_buttons)
        # save_buttons.add_stretch()  # for centering
        self.__init_save_pcd_button(save_buttons)
        self.__init_save_rgbd_button(save_buttons)

        # save_buttons.add_stretch()  # for centering

    def __init_save_pcd_button(self, parent_layout):
        self.save_pcd_button = gui.Button("Save Point cloud")
        self.save_pcd_button.horizontal_padding_em = 0.5
        self.save_pcd_button.vertical_padding_em = 0
        parent_layout.add_child(self.save_pcd_button)

    def __init_save_rgbd_button(self, parent_layout):
        self.save_rgbd_button = gui.Button("Save RGBD frame")
        self.save_rgbd_button.horizontal_padding_em = 0.5
        self.save_rgbd_button.vertical_padding_em = 0
        parent_layout.add_child(self.save_rgbd_button)

    def __init_video_displays(self, parent_layout=None):
        # Video size
        self.video_size = (int(180 * self.window.scaling),
                           int(320 * self.window.scaling), 3)

        # Color image display
        self.__init_color_image_display(parent_layout)

        # Depth image display
        self.__init_depth_image_display(parent_layout)

    def __init_color_image_display(self, parent_layout=None):
        self.show_color = gui.CollapsableVert("Color image")
        self.show_color.set_is_open(True)
        parent_layout.add_child(self.show_color)
        self.color_video = gui.ImageWidget()
        self.show_color.add_child(self.color_video)
    
    def __calib_color_image_display(self, parent_layout=None):
        self.show_calib = gui.CollapsableVert("Calibration image")
        self.show_calib.set_is_open(True)
        parent_layout.add_child(self.show_calib)
        self.calib_video = gui.ImageWidget()
        self.show_calib.add_child(self.calib_video)

    def __init_depth_image_display(self, parent_layout=None):
        self.show_depth = gui.CollapsableVert("Depth image")
        self.show_depth.set_is_open(True)
        parent_layout.add_child(self.show_depth)
        self.depth_video = gui.ImageWidget()
        self.show_depth.add_child(self.depth_video)

    def __init_operate_info(self, parent_layout=None):
        self.info_show = gui.CollapsableVert("Operate Info")
        self.info_show.set_is_open(True)
        parent_layout.add_child(self.info_show)
        self.mouse_coord = gui.Label("mouse coord: ")
        self.info_show.add_child(self.mouse_coord)

    def __init_scene_info(self, parent_layout=None):
        self.scene_info = gui.CollapsableVert("Scene Info")
        self.scene_info.set_is_open(False)
        parent_layout.add_child(self.scene_info)
        self.view_status = gui.Label("")
        self.scene_info.add_child(self.view_status)

    def __init_bbox_controls(self, parent_layout=None):
        self.bbox_controls = gui.CollapsableVert("Bounding Box Controls", 
                                                 self.em, gui.Margins(self.em, 0, 0, 0))
        self.bbox_controls.set_is_open(True)
        parent_layout.add_child(self.bbox_controls)

        # Create sliders and NumberEdit widgets for xmin, xmax, ymin, ymax, zmin, zmax
        self.bbox_sliders = {}
        self.bbox_edits = {}
        bbox_params = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for param in bbox_params:
            slider = gui.Slider(gui.Slider.DOUBLE)
            slider.set_limits(-5.0, 5.0)  # Adjust the limits as needed
            # Callbacks to be set later in PipelineView
            number_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            number_edit.set_limits(-5.0, 5.0)
            # Callbacks to be set later in PipelineView

            label = gui.Label(param)
            layout = gui.Horiz()
            layout.add_child(label)
            layout.add_child(slider)
            layout.add_child(number_edit)
            self.bbox_controls.add_child(layout)
            self.bbox_sliders[param] = slider
            self.bbox_edits[param] = number_edit

        control_buttons = gui.Horiz(self.em)
        control_buttons.add_stretch()
        self.bbox_controls.add_child(control_buttons)
        self.save_bbox_button = gui.Button("Save")
        # self.save_bbox_button.set_on_clicked(self.callbacks['on_save_bbox'])
        self.save_bbox_button.horizontal_padding_em = 0.5
        self.save_bbox_button.vertical_padding_em = 0
        control_buttons.add_child(self.save_bbox_button)

        self.load_bbox_button = gui.Button("Load")
        # self.load_bbox_button.set_on_clicked(self.callbacks['on_load_bbox'])
        self.load_bbox_button.horizontal_padding_em = 0.5
        self.load_bbox_button.vertical_padding_em = 0
        control_buttons.add_child(self.load_bbox_button)
        control_buttons.add_stretch()

    def set_disable_before_stream_init(self):
        self.toggle_capture.enabled = False
        self.toggle_model_init.enabled = False
        self.cam_calibreate_button.enabled = False
        self.he_calibreate_button.enabled = False
        self.save_pcd_button.enabled = False
        self.save_rgbd_button.enabled = False
    
    def after_stream_init(self):
        self.toggle_capture.enabled = True
        self.toggle_model_init.enabled = True
        self.cam_calibreate_button.enabled = True
        self.save_pcd_button.enabled = True
        self.save_rgbd_button.enabled = True

    def get_pcd_view(self):
        return self.pcdview