import threading
import logging as log
import open3d.visualization.gui as gui
import numpy as np
import open3d.visualization.rendering as rendering
import open3d as o3d
from pipeline.pipeline_model import PipelineModel
from pipeline.pipeline_view import PipelineView
from utils.calibration_process import CalibrationProcess
from utils.robot import RobotInterface
from functools import wraps
import logging
import json
import open3d.core as o3c
from utils import ARUCO_BOARD


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


class PipelineController:
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing  and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """

    # Class-level dictionary to store callbacks
    

    def __init__(self, params:dict):
        """Initialize.

        Args:
            camera_config_file (str): Camera configuration json file.
            rgbd_video (str): RS bag file containing the RGBD video. If this is
                provided, connected cameras are ignored.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
        """
        self.params = params
        self.pipeline_model = PipelineModel(self.update_view, params=params)
        self.calibration_data = self.pipeline_model.calibration_data
        self.collected_data = self.pipeline_model.collected_data
        self.calibration = None
        self.drawing_rectangle = False
        self.initial_point = None
        self.rectangle_geometry = None
        self.robot = RobotInterface()

        # Collect bound methods into callbacks dictionary
        self.callbacks = {name: getattr(self, name) for name in _callback_names}
        
        self.pipeline_view = PipelineView(
            self.pipeline_model.max_points,
            callbacks=self.callbacks
        )

        
        self.init_settinngs_values()
        threading.Thread(name='PipelineModel',
                         target=self.pipeline_model.run).start()
        gui.Application.instance.run()

    def init_settinngs_values(self):
        self.chessboard_dims = self.params.get('board_shape', (11, 6))
        self.pipeline_view.scene_widgets.board_col_num_edit.int_value = self.chessboard_dims[0]
        self.pipeline_view.scene_widgets.board_row_num_edit.int_value = self.chessboard_dims[1]
        self.pipeline_view.scene_widgets.board_square_size_num_edit.double_value = self.params.get('board_square_size', 0.023)
        self.pipeline_view.scene_widgets.board_marker_size_num_edit.double_value = self.params.get('board_marker_size', 0.0175)
        self.pipeline_view.scene_widgets.board_type_combobox.selected_text = self.params.get('board_type', "DICT_4X4_100")
        self.pipeline_view.scene_widgets.calib_save_text.text_value = self.params.get('calib_path', "")
        

    def update_view(self, frame_elements):
        """Updates view with new data. May be called from any thread.

        Args:
            frame_elements (dict): Display elements (point cloud and images)
                from the new frame to be shown.
        """
        gui.Application.instance.post_to_main_thread(
            self.pipeline_view.window,
            lambda: self.pipeline_view.update(frame_elements))

    @callback
    def on_capture_toggle(self, is_on):
        """Callback to toggle capture."""
        self.pipeline_view.capturing = is_on
        self.pipeline_view.vfov =  1.25 * self.pipeline_model.vfov
        if not self.pipeline_view.capturing:
            # Set the mouse callback when not capturing
            self.pipeline_view.pcdview.set_on_mouse(self.on_mouse_widget3d)
        else:
            # Unset the mouse callback when capturing
            self.pipeline_view.pcdview.set_on_mouse(None)

        # Update model
        self.pipeline_model.flag_capture = is_on
        if not is_on:
            self.on_toggle_record(False)
            if self.pipeline_view.toggle_record is not None:
                self.pipeline_view.toggle_record.is_on = False
        else:
            with self.pipeline_model.cv_capture:
                self.pipeline_model.cv_capture.notify()

    @callback
    def on_toggle_record(self, is_enabled):
        """Callback to toggle recording RGBD video."""
        self.pipeline_model.flag_record = is_enabled

    @callback
    def on_display_mode_combobox_changed(self, text, index):
        """Callback to change display mode."""
        self.pipeline_model.flag_normals = False
        self.pipeline_view.display_mode = text
        match text:
            case 'Colors':
                log.debug('Display mode: Colors')
                pass
            case 'Normals':
                log.debug('Display mode: Normals')
                self.pipeline_model.flag_normals = True
                self.pipeline_view.flag_gui_init = False
            case 'Segmentation':
                log.debug('Display mode: Segmentation')
                self.pipeline_model.flag_segemtation_mode = True
                self.pipeline_view.update_pcd_geometry()

    # @callback
    # def on_toggle_normals(self, is_enabled):
    #     """Callback to toggle display of normals"""
    #     self.pipeline_model.flag_normals = is_enabled
    #     self.pipeline_view.flag_gui_init = False
    

    @callback
    def on_window_close(self):
        """Callback when the user closes the application window."""
        self.pipeline_model.flag_exit = True
        with self.pipeline_model.cv_capture:
            self.pipeline_model.cv_capture.notify_all()
        return True  # OK to close window

    @callback
    def on_save_pcd_button(self):
        """Callback to save current point cloud."""
        self.pipeline_model.flag_save_pcd = True

    @callback
    def on_seg_model_init_toggle(self, is_enabled):
        self.pipeline_model.seg_model_intialization()
        self.pipeline_model.flag_segemtation_mode = is_enabled

    @callback
    def on_save_rgbd_button(self):
        """Callback to save current RGBD image pair."""
        self.pipeline_model.flag_save_rgbd = True
        logger.debug('Saving RGBD image pair')

    @callback
    def on_mouse_widget3d(self, event):
        if self.pipeline_view.capturing:
            return gui.Widget.EventCallbackResult.IGNORED  # Do nothing if capturing

        if not self.pipeline_view.acq_mode:
            return gui.Widget.EventCallbackResult.IGNORED

        # Handle left button down with Ctrl key to start drawing
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN and
            event.is_modifier_down(gui.KeyModifier.CTRL) and
            event.is_button_down(gui.MouseButton.LEFT)):
            x = event.x - self.pipeline_view.pcdview.frame.x
            y = event.y - self.pipeline_view.pcdview.frame.y
            if 0 <= x < self.pipeline_view.pcdview.frame.width and 0 <= y < self.pipeline_view.pcdview.frame.height:

                def depth_callback(depth_image):
                    depth_array = np.asarray(depth_image)
                    # Check if (x, y) are valid coordinates inside the depth image
                    if y < depth_array.shape[0] and x < depth_array.shape[1]:
                        depth = depth_array[y, x]
                    else:
                        depth = 1.0  # Assign far plane depth if out of bounds

                    if depth == 1.0:  # clicked on nothing (far plane)
                        text = "Mouse Coord: Clicked on nothing"
                    else:
                        # Compute world coordinates from screen (x, y) and depth
                        world = self.pipeline_view.pcdview.scene.camera.unproject(
                            x, y, depth, self.pipeline_view.pcdview.frame.width, self.pipeline_view.pcdview.frame.height)
                        text = "Mouse Coord: ({:.3f}, {:.3f}, {:.3f})".format(
                            world[0], world[1], world[2])

                    # Update label in the main UI thread
                    def update_label():
                        self.pipeline_view.scene_widgets.mouse_coord.text = text
                        self.pipeline_view.window.set_needs_layout()

                    gui.Application.instance.post_to_main_thread(self.pipeline_view.window, update_label)

                # Perform the depth rendering asynchronously
                self.pipeline_view.pcdview.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED

        # Handle dragging to update rectangle
        elif event.type == gui.MouseEvent.Type.DRAG and self.drawing_rectangle:
            pass
            return gui.Widget.EventCallbackResult.HANDLED

        # Handle left button up to finish drawing
        elif (event.type == gui.MouseEvent.Type.BUTTON_UP and
            self.drawing_rectangle):
            # Finalize rectangle
            self.drawing_rectangle = False
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    @callback
    def on_robot_init_button(self):
        ret, msg, msg_color = self.pipeline_model.robot_init()

        if self.pipeline_model.flag_robot_init and self.pipeline_model.flag_camera_init:
            self.pipeline_view.scene_widgets.handeye_calib_init_button.enabled = True
        
        self.pipeline_view.scene_widgets.robot_msg.text = msg
        self.pipeline_view.scene_widgets.robot_msg.text_color = msg_color
        self.calibration_data.reset()
        self.pipeline_view.scene_widgets.frame_list_view.set_items(['Click "Collect Current Frame" to start'])

    @callback
    def on_cam_calib_init_button(self):
        ret, msg, msg_color = self.pipeline_model.camera_calibration_init()
        
        if self.pipeline_model.flag_robot_init and self.pipeline_model.flag_camera_init:
            self.pipeline_view.scene_widgets.handeye_calib_init_button.enabled = True
        
        self.pipeline_view.scene_widgets.calibration_msg.text = msg
        self.pipeline_view.scene_widgets.calibration_msg.text_color = msg_color
        self.pipeline_view.scene_widgets.detect_board_toggle.enabled = True
        self.pipeline_model.calibration_data_init()
        self.pipeline_view.scene_widgets.frame_list_view.set_items(['Click "Collect Current Frame" to start'])

    @callback
    def on_handeye_calib_init_button(self):
        ret, msg, msg_color = self.pipeline_model.handeye_calibration_init()
        if ret:
            self.pipeline_model.flag_handeye_calib_init = True
            self.pipeline_view.scene_widgets.calibration_msg.text = msg
            self.pipeline_view.scene_widgets.calibration_msg.text_color = msg_color
            self.pipeline_model.calibration_data_init()
            self.pipeline_view.scene_widgets.frame_list_view.set_items(['Click "Collect Current Frame" to start'])
            # self.pipeline_view.scene_widgets.he_calibreate_button.enabled = False


    @callback
    def on_camera_view_button(self):
        self.pipeline_view.pcdview.setup_camera(self.pipeline_view.vfov, 
                                                self.pipeline_view.pcd_bounds, [0, 0, 0])
        lookat = [0, 0, 0]
        placeat = [-0.139, -0.356, -0.923]
        pointat = [-0.037, -0.93, 0.3649]
        self.pipeline_view.pcdview.scene.camera.look_at(lookat, placeat, pointat)

    @callback
    def on_birds_eye_view_button(self):
        """Callback to reset point cloud view to birds eye (overhead) view"""
        self.pipeline_view.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        self.pipeline_view.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])

    @callback
    def on_stream_init_button(self):
        log.debug('Stream init start')
        match self.pipeline_view.scene_widgets.stream_combbox.selected_text:
            case 'Camera':
                try:
                    self.pipeline_model.camera_mode_init()
                    self.pipeline_model.flag_stream_init = True
                    self.pipeline_view.scene_widgets.status_message.text = "Azure Kinect camera connected."
                    self.pipeline_view.scene_widgets.after_stream_init()
                    self.pipeline_model.flag_camera_init = True
                    if self.pipeline_model.flag_robot_init and self.pipeline_model.flag_camera_init:
                        self.pipeline_view.scene_widgets.handeye_calib_init_button.enabled = True
                    self.on_camera_view_button()
                except Exception as e:
                    self.pipeline_view.scene_widgets.status_message.text = "Camera initialization failed!"
            case 'Video':
                pass

    @callback
    def on_acq_mode_toggle(self, is_on):
        self.pipeline_view.acq_mode = is_on
        if is_on:
            self.pipeline_view.pcdview.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)
            if self.pipeline_view.plane is None:
                plane = o3d.geometry.TriangleMesh.create_box(width=10, height=0.01, depth=10)
                plane.translate([-5, 1, -5])  # Position the plane at y=1
                plane.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray color
                plane_material = rendering.MaterialRecord()
                plane_material.shader = "defaultUnlit"
                plane_material.base_color = [0.8, 0.8, 0.8, 0.5]  # Semi-transparent
                self.pipeline_view.pcdview.scene.add_geometry("edit_plane", plane, plane_material)
                self.pipeline_view.plane = plane
                self.pipeline_view.pcdview.force_redraw()
        else:
            self.pipeline_view.pcdview.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            if self.pipeline_view.plane is not None:
                self.pipeline_view.pcdview.scene.remove_geometry("edit_plane")
                self.pipeline_view.plane = None
            self.pipeline_view.pcdview.force_redraw()

    @callback
    def on_calib_check_button(self):
        try:
            path = self.params['calib_path']
            with open(path, 'r') as f:
                self.calib:dict = json.load(f)
            intrinsic = np.array(self.calib.get('camera_matrix'))
            dist_coeffs = np.array(self.calib.get('dist_coeffs'))
            self.pipeline_model.update_camera_matrix(intrinsic, dist_coeffs)
            self.pipeline_view.scene_widgets.calib_combobox.clear_items()

            for name in self.calib.get('calibration_results').keys():
                self.pipeline_view.scene_widgets.calib_combobox.add_item(name)
            
            self.pipeline_view.scene_widgets.calibration_mode_toggle.enabled = True
            self.pipeline_view.scene_widgets.calib_combobox.enabled = True
            self.pipeline_model.flag_handeye_calib_success = True
            self.pipeline_view.scene_widgets.calib_combobox.selected_index = 0
            self.pipeline_view.scene_widgets.data_tab.visible = True
            
        except Exception as e:
            self.pipeline_view.scene_widgets.calib_combobox.enabled = False
            log.error(e)

    @callback
    def on_calib_combobox_change(self, text, index):
        self.pipeline_model.T_cam_to_base = np.array(self.calib.get('calibration_results').get(text).get('transformation_matrix'))
        # self.pipeline_model.T_cam_to_base
        
    @callback
    def on_board_col_num_edit_change(self, value):
        # self.calibration.chessboard_size[0] = int(value)
        self.params['board_shape'] = (int(value), self.params['board_shape'][1])
        log.debug(f"Chessboard type: {self.params.get('board_shape')}")

    @callback
    def on_board_row_num_edit_change(self, value):
        # self.calibration.chessboard_size[1] = int(value)
        self.params['board_shape'] = (self.params.get('board_shape')[0], int(value))
        log.debug(f"Chessboard type: {self.params.get('board_shape')}")

    @callback
    def on_board_square_size_num_edit_change(self, value):
        self.params['board_square_size'] = value
        logger.debug(f'board_square_size changed: {value} mm')
    
    @callback
    def on_board_marker_size_num_edit_change(self, value):
        self.params['board_marker_size'] = value
        logger.debug(f'board_marker_size changed: {value} mm')

    @callback
    def on_data_folder_select_button(self):
        filedlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, 
                                 "Select Folder",
                                 self.pipeline_view.window.theme)
        # filedlg.add_filter(".obj .ply .stl", "Triangle mesh (.obj, .ply, .stl)")
        # filedlg.add_filter("", "All files")
        filedlg.set_on_cancel(self._on_data_folder_cancel)
        filedlg.set_on_done(self._on_data_folder_selcted)
        self.pipeline_view.window.show_dialog(filedlg)

    def _on_data_folder_selcted(self, path):
        self.pipeline_view.scene_widgets.data_folder_text.text_value = path+"/data.recorder"
        self.pipeline_view.window.close_dialog()

    def _on_data_folder_cancel(self):
        # self.pipeline_view.scene_widgets.data_folder_text.text = ""
        self.pipeline_view.window.close_dialog()
    
    @callback
    def on_prompt_text_change(self, text):
        if text == "":
            self.pipeline_view.scene_widgets.data_collect_button.enabled = False
        else:
            self.pipeline_view.scene_widgets.data_collect_button.enabled = True
        logger.debug(f"Prompt text changed: {text}")

    def _data_tree_view_update(self):
        self.pipeline_view.scene_widgets.data_tree_view.tree.clear()
        for key, value in self.collected_data.data_json.items():
            root_id = self.pipeline_view.scene_widgets.data_tree_view.add_item(
                self.pipeline_view.scene_widgets.data_tree_view.tree.get_root_item(), key, level=1)

            # Add 'prompt' field
            prompt_id = self.pipeline_view.scene_widgets.data_tree_view.add_item(root_id, "Prompt", level=2, root_text=key)
            self.pipeline_view.scene_widgets.data_tree_view.add_item(prompt_id, value["prompt"], level=3, root_text=key)

            # Add 'bbox' field
            prompt_id = self.pipeline_view.scene_widgets.data_tree_view.add_item(root_id, "Bbox", level=2, root_text=key)
            self.pipeline_view.scene_widgets.data_tree_view.add_item(prompt_id, value["bboxes"], level=3, root_text=key)

            # Add 'pose' field
            pose_id = self.pipeline_view.scene_widgets.data_tree_view.add_item(root_id, "Pose", level=2, root_text=key)
            for i, pose in enumerate(value["pose"]):
                pose_text = f"{i + 1}: [{','.join(f'{v:.2f}' for v in pose)}]"
                self.pipeline_view.scene_widgets.data_tree_view.add_item(pose_id, pose_text, level=3, root_text=key)

        self.pipeline_view.scene_widgets.data_tree_view.selected_item.reset()

    @callback
    def on_data_collect_button(self):
        # tmp_pose = np.array([1,2,3,4,5,6])
        try:
            tmp_pose = self.pipeline_model.robot_interface.capture_gripper_to_base(sep=False)
            self.collected_data.append(self.pipeline_view.scene_widgets.prompt_text.text_value, tmp_pose)
            self._data_tree_view_update()
            logger.debug(f"On data collect Click")
        except:
            logger.error("Failed to capture gripper pose")

        # self.pipeline_view.scene_widgets.data_list_view.set_items(self.collected_data)

    @callback
    def on_data_tree_view_changed(self, item):
        logger.debug(
            f"Root Parent Text: {item.root_text}, "
            f"Custom Callback -> Selected Item ID: {item.item_id}, "
            f"Level: {item.level}, Index in Level: {item.index_in_level}, "
            f"Parent Text: {item.parent_text}"
        )
        self.pipeline_view.scene_widgets.prompt_text.text_value = self.collected_data.data_json.get(
                self.pipeline_view.scene_widgets.data_tree_view.selected_item.root_text
                ).get('prompt')

    @callback
    def on_data_tree_view_remove_button(self):
        select_item = self.pipeline_view.scene_widgets.data_tree_view.selected_item
        # self.on_data_tree_view_changed(select_item)
        if select_item != None:
            match select_item.level:
                case 1:
                    logger.debug(f"Removing {select_item.root_text}")
                    self.collected_data.pop(self.collected_data.dataids.index(select_item.root_text))
                    self._data_tree_view_update()
                case 2:
                    pass
                case 3:
                    logger.debug(f"Removing pose")
                    if select_item.parent_text == "Pose":
                        self.collected_data.pop_pose(self.collected_data.dataids.index(select_item.root_text), 
                                                     select_item.index_in_level)
                    self._data_tree_view_update()
        # logger.debug("Removing data")


    @callback
    def on_data_save_button(self):
        if self.collected_data == '':
            pass
        else:
            self.collected_data.save(self.pipeline_view.scene_widgets.data_folder_text.text_value)
        logger.debug("Saving data")
        pass

    @callback
    def on_board_type_combobox_change(self, text, index):
        self.params['board_type'] = self.pipeline_view.scene_widgets.board_type_combobox.selected_text
        logger.debug(f"Board type: {self.params.get('board_type')}")
    
    @callback
    def on_calib_collect_button(self):
        self.pipeline_model.flag_calib_collect = True
        self.pipeline_view.scene_widgets.frame_list_view.set_items(
            self.calibration_data.display_str_list)
        logger.debug("Collecting calibration data")

    @callback
    def on_calib_list_remove_button(self):
        self.calibration_data.pop(self.pipeline_view.scene_widgets.frame_list_view.selected_index)
        self.pipeline_view.scene_widgets.frame_list_view.set_items(
            self.calibration_data.display_str_list)
        logger.debug("Removing calibration data")

    @callback
    def on_robot_move_button(self):
        idx = self.pipeline_view.scene_widgets.frame_list_view.selected_index
        try:
            self.pipeline_model.robot_interface.move_to_pose(
                self.calibration_data.robot_poses[idx])
            self.calibration_data.modify(idx, np.asarray(self.pipeline_model.rgbd_frame.color),
                                        self.pipeline_model.robot_interface.capture_gripper_to_base(sep=False))
            logger.debug("Moving robot and collecting data")
        except:
            logger.error("Failed to move robot")
        
    @callback
    def on_calib_save_button(self):
        self.calibration_data.save_calibration_data(
            self.pipeline_view.scene_widgets.calib_save_text.text_value)
        self.on_calib_check_button()
        logger.debug("Saving calibration data and check data")
    
    @callback
    def on_calib_op_save_button(self):
        self.calibration_data.save_img_and_pose()
        logger.debug("Saving images and poses")

    @callback
    def on_calib_op_load_button(self):
        self.calibration_data.load_img_and_pose()
        self.pipeline_view.scene_widgets.frame_list_view.set_items(
            self.calibration_data.display_str_list)
        logger.debug("Checking calibration data")
    
    @callback
    def on_calib_op_run_button(self):
        logger.debug("Running calibration data")
        for each_pose in self.calibration_data.robot_poses:
            self.pipeline_model.robot_interface.move_to_pose(each_pose)
            self.calibration_data.modify(self.calibration_data.robot_poses.index(each_pose), 
                                        np.asarray(self.pipeline_model.rgbd_frame.color),
                                        self.pipeline_model.robot_interface.capture_gripper_to_base(sep=False))
        

    @callback
    def on_calib_button(self):
        self.calibration_data.calibrate_all()
        self.pipeline_view.scene_widgets.frame_list_view.set_items(
            self.calibration_data.display_str_list)
        self.pipeline_model.update_camera_matrix(self.calibration_data.camera_matrix, 
                                                 self.calibration_data.dist_coeffs)
        logger.debug("calibration button")

    @callback 
    def on_detect_board_toggle(self, is_on):
        self.pipeline_model.flag_tracking_board = is_on
        logger.debug(f"Detecting board: {is_on}")

    @callback 
    def on_show_axis_in_scene_toggle(self, is_on):
        self.pipeline_model.flag_calib_axis_to_scene = is_on
        self.pipeline_view.pcdview.scene.show_geometry('board_pose', is_on)
        self.pipeline_view.pcdview.scene.show_geometry('robot_base_frame', is_on)
        self.pipeline_view.pcdview.scene.show_geometry('robot_end_frame', is_on)

        logger.debug(f"Show axis in scene: {is_on}")

    @callback
    def on_frame_list_view_changed(self, new_val, is_dbl_click):
        # TODO: update still buggy, need to be fixed
        self.pipeline_model.flag_tracking_board = False
        self.pipeline_view.scene_widgets.detect_board_toggle.is_on = False
        if len(self.calibration_data) > 0:
            img = self.calibration_data.images[self.pipeline_view.scene_widgets.frame_list_view.selected_index]
            img = o3d.t.geometry.Image(img).cpu()
            if self.pipeline_view.scene_widgets.show_calib.get_is_open():
                sampling_ratio = self.pipeline_view.video_size[1] / img.columns
                self.pipeline_view.scene_widgets.calib_video.update_image(img.resize(sampling_ratio))
                logger.debug("Showing calibration pic")

    @callback
    def on_calib_save_text_changed(self, text):
        self.params['calib_path'] = text
        logger.debug(f"calib_path changed: {text}")

    @callback
    def on_key_pressed(self, event):
        if self.pipeline_view.scene_widgets.tab_view.selected_tab_index == 3 and \
                    self.pipeline_model.flag_camera_init:
            if event.type == gui.KeyEvent.Type.DOWN:
                if event.key == gui.KeyName.SPACE:
                # self.pipeline_model.flag_tracking_board = not self.pipeline_model.flag_tracking_board
                    self.on_calib_collect_button()
                elif event.key == gui.KeyName.C:
                    self.on_calib_button()
                logger.info(f"key pressed {event.key}")
            return True
        return False
    