import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c
import numpy as np
import torch
from utils import get_num_of_palette

class Widget_Init:
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
        self.__init_fps_label()
        self.__init_toggles()
        self.__init_edit_mode_toggle()
        self.__init_model_init_toggle()
        self.__init_view_buttons()
        self.__init_save_toggles()
        self.__init_save_buttons()
        self.__init_video_displays()
        self.__init_operate_info()
        self.__init_scene_info()
        self.__init_status_message()
        self.__init_bbox_controls()

    def __init_fps_label(self):
        self.fps_label = gui.Label("FPS: 00")
        self.window.add_child(self.fps_label)

    def __init_toggles(self):
        toggles = gui.Horiz(self.em)
        self.panel.add_child(toggles)
        self.__init_toggle_capture(toggles)
        self.__init_toggle_normals(toggles)

    def __init_toggle_capture(self, parent_layout):
        self.toggle_capture = gui.ToggleSwitch("Capture / Play")
        self.toggle_capture.is_on = False
        self.toggle_capture.set_on_clicked(self.callbacks['on_toggle_capture'])
        parent_layout.add_child(self.toggle_capture)

    def __init_toggle_normals(self, parent_layout):
        self.toggle_normals = gui.ToggleSwitch("Colors / Normals")
        self.toggle_normals.is_on = False
        self.toggle_normals.set_on_clicked(self.callbacks['on_toggle_normals'])
        parent_layout.add_child(self.toggle_normals)

    def __init_edit_mode_toggle(self):
        edit_mode_toggle = gui.Horiz(self.em)
        self.panel.add_child(edit_mode_toggle)
        self.toggle_edit_mode = gui.ToggleSwitch("Edit Mode")
        self.toggle_edit_mode.is_on = False
        # Callback to be set later in PipelineView
        edit_mode_toggle.add_child(self.toggle_edit_mode)

        self.__init_toggle_segmentation(edit_mode_toggle)

    def __init_toggle_segmentation(self, parent_layout):
        self.toggle_segmentation = gui.ToggleSwitch("Show Segmentation")
        self.toggle_segmentation.is_on = False
        # Callback to be set later in PipelineView
        parent_layout.add_child(self.toggle_segmentation)

    def __init_model_init_toggle(self):
        edit_mode_toggle_2 = gui.Horiz(self.em)
        self.panel.add_child(edit_mode_toggle_2)
        self.toggle_model_init = gui.ToggleSwitch("Model Initialization")
        self.toggle_model_init.is_on = False
        self.toggle_model_init.set_on_clicked(self.callbacks['on_toggle_model_init'])
        edit_mode_toggle_2.add_child(self.toggle_model_init)

    def __init_view_buttons(self):
        view_buttons = gui.Horiz(self.em)
        self.panel.add_child(view_buttons)
        view_buttons.add_stretch()  # for centering

        self.__init_camera_view_button(view_buttons)
        self.__init_birds_eye_view_button(view_buttons)

        view_buttons.add_stretch()  # for centering

    def __init_camera_view_button(self, parent_layout):
        self.camera_view_button = gui.Button("Camera view")
        # Callback to be set later in PipelineView
        parent_layout.add_child(self.camera_view_button)

    def __init_birds_eye_view_button(self, parent_layout):
        self.birds_eye_view_button = gui.Button("Bird's eye view")
        # Callback to be set later in PipelineView
        parent_layout.add_child(self.birds_eye_view_button)

    def __init_save_toggles(self):
        save_toggle = gui.Horiz(self.em)
        self.panel.add_child(save_toggle)
        save_toggle.add_child(gui.Label("Record / Save"))
        self.toggle_record = None
        if self.callbacks['on_toggle_record'] is not None:
            save_toggle.add_fixed(1.5 * self.em)
            self.toggle_record = gui.ToggleSwitch("Video")
            self.toggle_record.is_on = False
            self.toggle_record.set_on_clicked(self.callbacks['on_toggle_record'])
            save_toggle.add_child(self.toggle_record)

    def __init_save_buttons(self):
        save_buttons = gui.Horiz(self.em)
        self.panel.add_child(save_buttons)
        save_buttons.add_stretch()  # for centering

        self.__init_save_pcd_button(save_buttons)
        self.__init_save_rgbd_button(save_buttons)

        save_buttons.add_stretch()  # for centering

    def __init_save_pcd_button(self, parent_layout):
        self.save_pcd_button = gui.Button("Save Point cloud")
        self.save_pcd_button.set_on_clicked(self.callbacks['on_save_pcd'])
        parent_layout.add_child(self.save_pcd_button)

    def __init_save_rgbd_button(self, parent_layout):
        self.save_rgbd_button = gui.Button("Save RGBD frame")
        self.save_rgbd_button.set_on_clicked(self.callbacks['on_save_rgbd'])
        parent_layout.add_child(self.save_rgbd_button)

    def __init_video_displays(self):
        # Video size
        self.video_size = (int(180 * self.window.scaling),
                           int(320 * self.window.scaling), 3)

        # Color image display
        self.__init_color_image_display()

        # Depth image display
        self.__init_depth_image_display()

    def __init_color_image_display(self):
        self.show_color = gui.CollapsableVert("Color image")
        self.show_color.set_is_open(True)
        self.panel.add_child(self.show_color)
        self.color_video = gui.ImageWidget()
        self.show_color.add_child(self.color_video)

    def __init_depth_image_display(self):
        self.show_depth = gui.CollapsableVert("Depth image")
        self.show_depth.set_is_open(True)
        self.panel.add_child(self.show_depth)
        self.depth_video = gui.ImageWidget()
        self.show_depth.add_child(self.depth_video)

    def __init_operate_info(self):
        self.info_show = gui.CollapsableVert("Operate Info")
        self.info_show.set_is_open(True)
        self.panel.add_child(self.info_show)
        self.mouse_coord = gui.Label("mouse coord: ")
        self.info_show.add_child(self.mouse_coord)

    def __init_scene_info(self):
        self.scene_info = gui.CollapsableVert("Scene Info")
        self.scene_info.set_is_open(False)
        self.panel.add_child(self.scene_info)
        self.view_status = gui.Label("")
        self.scene_info.add_child(self.view_status)

    def __init_status_message(self):
        self.status_message = gui.Label("")
        self.panel.add_child(self.status_message)

    def __init_bbox_controls(self):
        self.bbox_controls = gui.CollapsableVert("Bounding Box Controls", self.em, gui.Margins(self.em, 0, 0, 0))
        self.bbox_controls.set_is_open(True)
        self.panel.add_child(self.bbox_controls)

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

    def get_pcd_view(self):
        return self.pcdview

class PipelineView:
    """Controls display and user interface. All methods must run in the main thread."""

    def __init__(self, vfov=60, max_pcd_vertices=1 << 20, **callbacks):
        """Initialize."""
        self.vfov = vfov
        self.max_pcd_vertices = max_pcd_vertices
        self.callbacks = callbacks  # Store the callbacks dictionary
        self.capturing = False  # Initialize capturing flag
        self.edit_mode = False  # Initialize edit mode flag

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Real time RGBD camera and PCD rendering", 1280, 720)
        # Called on window layout (e.g., resize)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(callbacks['on_window_close'])

        self.widget_all = Widget_Init(self.window, callbacks)

        # Set the callbacks for widgets that require methods of PipelineView
        self.widget_all.toggle_edit_mode.set_on_clicked(self._on_toggle_edit_mode)
        self.widget_all.toggle_segmentation.set_on_clicked(self._on_toggle_segmentation)
        self.widget_all.camera_view_button.set_on_clicked(self.camera_view)
        self.widget_all.birds_eye_view_button.set_on_clicked(self.birds_eye_view)
        self.widget_all.pcdview.set_on_mouse(callbacks['on_mouse_widget3d'])  # Set initial mouse callback

        # Now, we can access the widgets via self.widget_all
        self.pcdview = self.widget_all.get_pcd_view()
        self.em = self.widget_all.em

        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = "defaultLit"
        # Set n_pixels displayed for each 3D point, accounting for HiDPI scaling
        self.pcd_material.point_size = int(4 * self.window.scaling)

        # Point cloud bounds, depends on the sensor range
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0],
                                                              [0, 0, 0.5])
        self.camera_view()  # Initially look from the camera

        # Initialize other variables
        self.flag_normals = False
        self.show_segmentation = False
        self.num_segments = 20
        self.segcmap = plt.get_cmap('tab20', self.num_segments)

        self.video_size = self.widget_all.video_size

        self.flag_exit = False
        self.flag_gui_init = False
        self.camera_material = rendering.MaterialRecord()
        self.camera_material.shader = "unlitLine"
        self.camera_material.line_width = 5

        # Initialize plane and rectangle
        self.plane = None
        self.rectangle = None
        self.rectangle_material = rendering.MaterialRecord()
        self.rectangle_material.shader = "unlitLine"
        self.rectangle_material.line_width = 2  # Adjust the line width as needed

        # Initialize bounding box parameters
        self.bbox_params = {'xmin': -0.5, 'xmax': 0.5,
                            'ymin': -0.5, 'ymax': 0.5,
                            'zmin': -0, 'zmax': 1}

        # Initialize the bounding box lineset
        self.bbox_lineset = None
        self.bbox_lineset_name = "bounding_box"
        self.bbox_material = rendering.MaterialRecord()
        self.bbox_material.shader = "unlitLine"
        self.bbox_material.line_width = 2  # Adjust as needed

        # Initialize the bounding box in the scene
        self.update_bounding_box()

        # Initialize palettes
        self.palettes = get_num_of_palette(80)

        # Set up callbacks for bbox sliders and edits
        bbox_params = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for param in bbox_params:
            self.widget_all.bbox_sliders[param].double_value = self.bbox_params[param]
            self.widget_all.bbox_sliders[param].set_on_value_changed(lambda value, p=param: self._on_bbox_slider_changed(value, p))
            self.widget_all.bbox_edits[param].double_value = self.bbox_params[param]
            self.widget_all.bbox_edits[param].set_on_value_changed(lambda value, p=param: self._on_bbox_edit_changed(value, p))

    def update(self, frame_elements):
        """Update visualization with point cloud and images. Must run in main
        thread since this makes GUI calls.

        Args:
            frame_elements: dict {element_type: geometry element}.
                Dictionary of element types to geometry elements to be updated
                in the GUI:
                    'pcd': point cloud,
                    'color': rgb image (3 channel, uint8),
                    'depth': depth image (uint8),
                    'status_message': message
        """
        # Store the current point cloud and segmentation data
        if 'pcd' in frame_elements:
            self.current_pcd = frame_elements['pcd']
        else:
            self.current_pcd = None

        if 'seg' in frame_elements:
            self.current_seg = frame_elements['seg']
        else:
            self.current_seg = None

        # Update the point cloud visualization
        self.update_pcd_geometry()

        # Update color and depth images
        if self.widget_all.show_color.get_is_open() and 'color' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['color'].columns
            self.widget_all.color_video.update_image(
                frame_elements['color'].resize(sampling_ratio).to_legacy())

        if self.widget_all.show_depth.get_is_open() and 'depth' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['depth'].columns
            self.widget_all.depth_video.update_image(
                frame_elements['depth'].resize(sampling_ratio).to_legacy())

        if 'camera' in frame_elements:
            self.pcdview.scene.remove_geometry("camera")
            self.pcdview.scene.add_geometry("camera", frame_elements['camera'], self.camera_material)

        if 'status_message' in frame_elements:
            self.widget_all.status_message.text = frame_elements["status_message"]

        if 'fps' in frame_elements:
            fps = frame_elements["fps"]
            self.widget_all.fps_label.text = f"FPS: {int(fps)}"

        self.widget_all.view_status.text = str(self.pcdview.scene.camera.get_view_matrix())

    def _on_toggle_segmentation(self, is_on):
        self.show_segmentation = is_on
        # Update the point cloud visualization
        self.update_pcd_geometry()

    def update_pcd_geometry(self):
        if not self.flag_gui_init:
            # Initialize the point cloud geometry in the scene
            dummy_pcd = o3d.t.geometry.PointCloud({
                'positions':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'colors':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'normals':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32)
            })
            if self.pcdview.scene.has_geometry('pcd'):
                self.pcdview.scene.remove_geometry('pcd')

            self.pcd_material.shader = "normals" if self.flag_normals else "defaultLit"
            self.pcdview.scene.add_geometry('pcd', dummy_pcd, self.pcd_material)
            self.flag_gui_init = True

        pcd = self.current_pcd
        if pcd is None:
            return

        if self.show_segmentation and self.current_seg is not None:
            labels = self.current_seg  # Assuming labels is either a numpy array or a torch tensor

            # If labels is a torch tensor (possibly on GPU), handle accordingly
            if isinstance(labels, torch.Tensor):
                device = labels.device  # Get the device (CPU or GPU)

                # Convert GOLIATH_PALETTE to a torch tensor on the same device
                GOLIATH_PALETTE_TENSOR = torch.tensor(self.palettes, device=device, dtype=torch.float32) / 255.0

                num_points = labels.shape[0]

                # Initialize colors tensor with zeros (black color)
                colors = torch.zeros((num_points, 3), device=device, dtype=torch.float32)

                # Get valid indices where labels are non-negative
                valid_indices = torch.nonzero(labels >= 0, as_tuple=False).squeeze(1)

                # Assign colors to valid points
                colors[valid_indices] = GOLIATH_PALETTE_TENSOR[labels[valid_indices]]

                # Convert colors to Open3D tensor and assign to point cloud
                pcd.point.colors = o3c.Tensor(colors, dtype=o3c.Dtype.Float32, device=device)
            else:
                # If labels is a numpy array
                labels = labels.astype(np.int32)
                num_points = len(labels)

                # Initialize colors array with zeros (black color)
                colors = np.zeros((num_points, 3), dtype=np.float32)

                # Get valid indices where labels are non-negative
                valid_idx = labels >= 0

                # Assign colors to valid points
                colors[valid_idx] = np.array(self.palettes)[labels[valid_idx]] / 255.0  # Normalize to [0, 1]

                # Convert colors to Open3D tensor and assign to point cloud
                pcd.point.colors = o3c.Tensor(colors, dtype=o3c.Dtype.Float32)

            pcd_to_display = pcd
        else:
            pcd_to_display = pcd

        # Update the geometry in the scene
        if self.pcdview.scene.has_geometry('pcd'):
            self.pcdview.scene.remove_geometry('pcd')
        self.pcdview.scene.add_geometry('pcd', pcd_to_display, self.pcd_material)
        self.pcdview.force_redraw()

    def camera_view(self):
        """Callback to reset point cloud view to the camera"""
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        lookat = [0, 0, 0]
        placeat = [-0.139, -0.356, -0.923]
        pointat = [-0.037, -0.93, 0.3649]
        self.pcdview.scene.camera.look_at(lookat, placeat, pointat)

    def birds_eye_view(self):
        """Callback to reset point cloud view to birds eye (overhead) view"""
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        self.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])

    def on_layout(self, layout_context):
        """Callback on window initialize / resize"""
        frame = self.window.content_rect
        panel_width = self.em * 20
        # Set the frame for the panel on the right side
        self.widget_all.panel.frame = gui.Rect(frame.get_right() - panel_width,
                                               frame.y, panel_width,
                                               frame.height)

        # Set the frame for the scene on the left side
        self.pcdview.frame = gui.Rect(frame.x, frame.y,
                                      frame.width - panel_width,
                                      frame.height)

        pref = self.widget_all.fps_label.calc_preferred_size(layout_context,
                                                             gui.Widget.Constraints())
        self.widget_all.fps_label.frame = gui.Rect(frame.x,
                                                   frame.get_bottom() - pref.height,
                                                   pref.width, pref.height)

    def _on_toggle_edit_mode(self, is_on):
        self.edit_mode = is_on
        if is_on:
            # Disable camera controls by setting to PICK_POINTS mode
            self.pcdview.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)
            # Add plane at y=1 to the scene
            if self.plane is None:
                # Create a plane geometry at y=1
                plane = o3d.geometry.TriangleMesh.create_box(width=10, height=0.01, depth=10)
                plane.translate([-5, 1, -5])  # Position the plane at y=1
                plane.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray color
                plane_material = rendering.MaterialRecord()
                plane_material.shader = "defaultUnlit"
                plane_material.base_color = [0.8, 0.8, 0.8, 0.5]  # Semi-transparent
                self.pcdview.scene.add_geometry("edit_plane", plane, plane_material)
                self.plane = plane
                self.pcdview.force_redraw()
        else:
            # Enable normal camera controls
            self.pcdview.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
            # Remove the plane from the scene
            if self.plane is not None:
                self.pcdview.scene.remove_geometry("edit_plane")
                self.plane = None
            # Remove the rectangle if it exists
            if self.rectangle is not None:
                self.pcdview.scene.remove_geometry("rectangle")
                self.rectangle = None
            self.pcdview.force_redraw()

    def _on_bbox_slider_changed(self, value, param):
        self.bbox_params[param] = value
        # Update the corresponding NumberEdit widget
        self.widget_all.bbox_edits[param].double_value = value
        self.update_bounding_box()

    def _on_bbox_edit_changed(self, value, param):
        self.bbox_params[param] = value
        # Update the corresponding slider
        self.widget_all.bbox_sliders[param].double_value = value
        self.update_bounding_box()

    def update_bounding_box(self):
        # Create the lineset for the bounding box
        points = [
            [self.bbox_params['xmin'], self.bbox_params['ymin'], self.bbox_params['zmin']],
            [self.bbox_params['xmax'], self.bbox_params['ymin'], self.bbox_params['zmin']],
            [self.bbox_params['xmax'], self.bbox_params['ymax'], self.bbox_params['zmin']],
            [self.bbox_params['xmin'], self.bbox_params['ymax'], self.bbox_params['zmin']],
            [self.bbox_params['xmin'], self.bbox_params['ymin'], self.bbox_params['zmax']],
            [self.bbox_params['xmax'], self.bbox_params['ymin'], self.bbox_params['zmax']],
            [self.bbox_params['xmax'], self.bbox_params['ymax'], self.bbox_params['zmax']],
            [self.bbox_params['xmin'], self.bbox_params['ymax'], self.bbox_params['zmax']]
        ]

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
        ]

        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for all lines

        bbox_lineset = o3d.geometry.LineSet()
        bbox_lineset.points = o3d.utility.Vector3dVector(points)
        bbox_lineset.lines = o3d.utility.Vector2iVector(lines)
        bbox_lineset.colors = o3d.utility.Vector3dVector(colors)

        # Remove the old lineset if it exists
        if self.bbox_lineset is not None:
            self.pcdview.scene.remove_geometry(self.bbox_lineset_name)
        # Add the new lineset
        self.pcdview.scene.add_geometry(self.bbox_lineset_name, bbox_lineset, self.bbox_material)
        self.bbox_lineset = bbox_lineset
        self.pcdview.force_redraw()
