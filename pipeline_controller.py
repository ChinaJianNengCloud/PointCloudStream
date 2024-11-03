import threading
import logging as log
import open3d.visualization.gui as gui
import numpy as np
import open3d.visualization.rendering as rendering
import open3d as o3d
from pipeline_model import PipelineModel
from pipeline_view import PipelineView
from calibration import CalibrationProcess
from robot import RobotInterface
from functools import wraps
import logging

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
    

    def __init__(self, camera_config_file=None, rgbd_video=None, device=None):
        """Initialize.

        Args:
            camera_config_file (str): Camera configuration json file.
            rgbd_video (str): RS bag file containing the RGBD video. If this is
                provided, connected cameras are ignored.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
        """
        self.pipeline_model = PipelineModel(self.update_view,
                                            camera_config_file, rgbd_video,
                                            device)
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

        self.chessboard_type = [11, 8]
        threading.Thread(name='PipelineModel',
                         target=self.pipeline_model.run).start()
        gui.Application.instance.run()


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
    def on_toggle_capture(self, is_on):
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
    def on_display_mode_changed(self, text, index):
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
                self.pipeline_view.show_segmentation = True
                self.pipeline_view.update_pcd_geometry()

    @callback
    def on_toggle_normals(self, is_enabled):
        """Callback to toggle display of normals"""
        self.pipeline_model.flag_normals = is_enabled
        self.pipeline_view.flag_gui_init = False

    @callback
    def on_window_close(self):
        """Callback when the user closes the application window."""
        self.pipeline_model.flag_exit = True
        with self.pipeline_model.cv_capture:
            self.pipeline_model.cv_capture.notify_all()
        return True  # OK to close window

    @callback
    def on_save_pcd(self):
        """Callback to save current point cloud."""
        self.pipeline_model.flag_save_pcd = True

    @callback
    def on_toggle_model_init(self, is_enabled):
        self.pipeline_model.model_intialization()
        self.pipeline_model.flag_model_init = is_enabled

    @callback
    def on_save_rgbd(self):
        """Callback to save current RGBD image pair."""
        self.pipeline_model.flag_save_rgbd = True

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
                        self.pipeline_view.widget_all.mouse_coord.text = text
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
    def on_robot_button(self):
        try:
            self.robot.find_device()
            self.robot.connect()
            ip = self.robot.ip_address
            msg = f'Robot: Connected [{ip}]'
            self.pipeline_view.widget_all.robot_msg.text_color = gui.Color(0, 1, 0)
            self.pipeline_model.robot_init = True
            if self.pipeline_model.robot_init and self.pipeline_model.camera_init:
                self.pipeline_view.widget_all.he_calibreate_button.enabled = True
            self.pipeline_model.robot_init = True
            self.pipeline_model.robot = self.robot

        except Exception as e:
            msg = f'Robot: Connection failed [{e}]'
            self.pipeline_view.widget_all.robot_msg.text_color = gui.Color(1, 0, 0)

        self.pipeline_view.widget_all.robot_msg.text = msg

    @callback
    def on_camera_calibration(self):
        distortion  = self.pipeline_model.camera_json.get('distortion_coeffs', None)
        if self.calibration is None:
            self.calibration = CalibrationProcess(self.robot, 
                                        self.pipeline_model.camera,
                                        intrinsic=self.pipeline_model.intrinsic_matrix, 
                                        dist_coeffs=distortion)
        self.calibration.chessboard_size = self.chessboard_type
        self.pipeline_model.calib_exec.submit(self.calibration.calibrate_camera)
        self.pipeline_view.widget_all.calibration_msg.text = "CalibrationProcess: Camera calibration..."
        # self.pipeline_model.calib_exec.shutdown()
        self.on_camera_view()

    @callback
    def on_he_calibration(self):
        distortion  = self.pipeline_model.camera_json.get('distortion_coeffs', None)
        if self.calibration is None:
            self.calibration = CalibrationProcess(self.robot, 
                                        self.pipeline_model.camera,
                                        intrinsic=self.pipeline_model.intrinsic_matrix, 
                                        dist_coeffs=distortion)
        self.calibration.chessboard_size = self.chessboard_type
        self.pipeline_model.calib_exec.submit(self.calibration.calibrate_eye_hand_from_camera)
        self.pipeline_view.widget_all.calibration_msg.text = "CalibrationProcess: HandEye calibration..."
        self.pipeline_view.widget_all.he_calibreate_button.enabled = False


    @callback
    def on_camera_view(self):
        self.pipeline_view.pcdview.setup_camera(self.pipeline_view.vfov, 
                                                self.pipeline_view.pcd_bounds, [0, 0, 0])
        lookat = [0, 0, 0]
        placeat = [-0.139, -0.356, -0.923]
        pointat = [-0.037, -0.93, 0.3649]
        self.pipeline_view.pcdview.scene.camera.look_at(lookat, placeat, pointat)

    @callback
    def on_birds_eye_view(self):
        """Callback to reset point cloud view to birds eye (overhead) view"""
        self.pipeline_view.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        self.pipeline_view.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])

    @callback
    def on_stream_init_start(self):
        log.debug('Stream init start')
        match self.pipeline_view.widget_all.stream_combbox.selected_text:
            case 'Camera':
                try:
                    self.pipeline_model.camera_mode_init()
                    self.pipeline_model.flag_stream_init = True
                    self.pipeline_view.widget_all.status_message.text = "Azure Kinect camera connected."
                    self.pipeline_view.widget_all.after_stream_init()
                    self.pipeline_model.camera_init = True
                    if self.pipeline_model.robot_init and self.pipeline_model.camera_init:
                        self.pipeline_view.widget_all.he_calibreate_button.enabled = True
                    self.on_camera_view()
                except Exception as e:
                    self.pipeline_view.widget_all.status_message.text = "Camera initialization failed!"
            case 'Video':
                pass

    @callback
    def on_toggle_acq_mode(self, is_on):
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
    def on_chessboard_row_change(self, value):
        self.calibration.chessboard_size[1] = int(value)
        log.debug(f'Chessboard type: {self.chessboard_type}')

    @callback
    def on_chessboard_col_change(self, value):
        self.calibration.chessboard_size[0] = int(value)
        log.debug(f'Chessboard type: {self.chessboard_type}')
