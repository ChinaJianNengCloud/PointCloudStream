import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c
import numpy as np
import torch
from utils.palette import get_num_of_palette
from typing import Callable
import logging
from ui import SceneWidgets

logger = logging.getLogger(__name__)



class PipelineView:
    """Controls display and user interface. All methods must run in the main thread."""

    def __init__(self, max_pcd_vertices=1 << 20, callbacks:dict[str, Callable]=None):
        # def __init__(self, max_pcd_vertices=1 << 20, **callbacks):
        """Initialize."""
        self.vfov = 60
        self.max_pcd_vertices = max_pcd_vertices
        self.callbacks = callbacks  # Store the callbacks dictionary
        self.capturing = False  # Initialize capturing flag
        self.acq_mode = False  # Initialize acquisition mode flag

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Real time RGBD camera and PCD rendering", 1280, 720)
        # Called on window layout (e.g., resize)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(callbacks['on_window_close'])
        # self.window.show_settings = True

        self.scene_widgets = SceneWidgets(self.window, callbacks)
        self.callback_bindings()
        # Set the callbacks for widgets that require methods of PipelineView
        
        
        # Now, we can access the widgets via self.widget_all
        self.pcdview = self.scene_widgets.get_pcd_view()
        self.em = self.scene_widgets.em

        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = "defaultLit"
        # Set n_pixels displayed for each 3D point, accounting for HiDPI scaling
        self.pcd_material.point_size = int(4 * self.window.scaling)

        # Point cloud bounds, depends on the sensor range
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([0, 0, 0],
                                                              [0, 0, 0.5])
        # self.camera_view()  # Initially look from the camera

        # Initialize other variables
        self.display_mode = 'Colors'  # Initialize display mode to 'Colors'

        self.video_size = self.scene_widgets.video_size

        self.flag_gui_init = False
        self.line_material = rendering.MaterialRecord()
        self.line_material.shader = "unlitLine"
        self.line_material.line_width = 5

        # Initialize plane and rectangle
        self.plane = None

        self.palettes = get_num_of_palette(80)

        self.__init_bbox()

    def callback_bindings(self):
        for each in self.callbacks.keys():
            
            if "button" in each or 'toggle' in each:
                getattr(self.scene_widgets, "_".join(each.split('_')[1:])).set_on_clicked(self.callbacks[each])
                logger.info(f"Registered callback for {each}")
            elif "num_edit" in each:
                getattr(self.scene_widgets, "_".join(each.split('_')[1:-1])).set_on_value_changed(self.callbacks[each])
                logger.info(f"Registered callback for {each}")
            elif "combobox" in each:
                getattr(self.scene_widgets, "_".join(each.split('_')[1:-1])).set_on_selection_changed(self.callbacks[each])
                logger.info(f"Registered callback for {each}")
            elif "list_view" in each:
                getattr(self.scene_widgets, "_".join(each.split('_')[1:-1])).set_on_selection_changed(self.callbacks[each])
                logger.info(f"Registered callback for {each}")
            elif "text" in each:
                getattr(self.scene_widgets, "_".join(each.split('_')[1:-1])).set_on_text_changed(self.callbacks[each])
                # set_on_text_changed
                logger.info(f"Registered callback for {each}")
            elif "tree_view" in each:
                getattr(self.scene_widgets, "_".join(each.split('_')[1:-1])).set_on_selection_changed(self.callbacks[each])
                logger.info(f"Registered callback for {each}")

        # print(getattr(self.scene_widgets))
        self.scene_widgets.pcdview.set_on_mouse(
            self.callbacks['on_mouse_widget3d'])  # Set initial mouse callback

        self.window.set_on_key(self.callbacks['on_key_pressed'])


    def __init_bbox(self):
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

        # Set up callbacks for bbox sliders and edits
        bbox_params = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for param in bbox_params:
            self.scene_widgets.bbox_sliders[param].double_value = self.bbox_params[param]
            self.scene_widgets.bbox_sliders[param].set_on_value_changed(lambda value, 
                                                                        p=param: self._on_bbox_slider_changed(value, p))
            self.scene_widgets.bbox_edits[param].double_value = self.bbox_params[param]
            self.scene_widgets.bbox_edits[param].set_on_value_changed(lambda value, 
                                                                      p=param: self._on_bbox_edit_changed(value, p))

    def update(self, frame_elements: dict):
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
        self.current_pcd = frame_elements.get('pcd', None)
        self.current_seg = frame_elements.get('seg', None)
        self.current_robot_frame = frame_elements.get('robot_frame', None)

        # Update the point cloud visualization
        self.update_pcd_geometry()

        if self.scene_widgets.tab_view.selected_tab_index == 0: # general tab
        # Update color and depth images
            if self.scene_widgets.show_color.get_is_open() and 'color' in frame_elements:
                sampling_ratio = self.video_size[1] / frame_elements['color'].columns
                self.scene_widgets.color_video.update_image(
                    frame_elements['color'].resize(sampling_ratio).to_legacy())

            if self.scene_widgets.show_depth.get_is_open() and 'depth' in frame_elements:
                sampling_ratio = self.video_size[1] / frame_elements['depth'].columns
                self.scene_widgets.depth_video.update_image(
                    frame_elements['depth'].resize(sampling_ratio).to_legacy())
        
        if self.scene_widgets.tab_view.selected_tab_index == 3: # calib tab
            if self.scene_widgets.show_calib.get_is_open() and 'calib_color' in frame_elements:
                sampling_ratio = self.video_size[1] / frame_elements['calib_color'].columns
                self.scene_widgets.calib_video.update_image(
                    frame_elements['calib_color'].resize(sampling_ratio).to_legacy())


        self.geometry_registry("camera", frame_elements, self.line_material)
        self.geometry_registry("robot_base_frame", frame_elements, self.pcd_material)
        self.geometry_registry("robot_end_frame", frame_elements, self.pcd_material)
        self.geometry_registry("board_pose", frame_elements, self.pcd_material)


        if 'status_message' in frame_elements:
            self.scene_widgets.status_message.text = frame_elements["status_message"]

        if 'fps' in frame_elements:
            fps = frame_elements["fps"]
            self.scene_widgets.fps_label.text = f"FPS: {int(fps)}"

        self.scene_widgets.view_status.text = str(self.pcdview.scene.camera.get_view_matrix())

    def geometry_registry(self, name, frame_elements, material):
        if name in frame_elements:
            self.pcdview.scene.remove_geometry(name)
            self.pcdview.scene.add_geometry(name, frame_elements[name], material)

    def update_pcd_geometry(self):
        if not self.flag_gui_init:
            # Initialize the point cloud geometry in the scene
            dummy_pcd = o3d.t.geometry.PointCloud({
                'positions':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'colors':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3c.Dtype.Float32),
                'normals':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3c.Dtype.Float32)
            })
            if self.pcdview.scene.has_geometry('pcd'):
                self.pcdview.scene.remove_geometry('pcd')

            # Set shader based on display mode
            if self.display_mode == 'Normals':
                self.pcd_material.shader = 'normals'
            else:
                self.pcd_material.shader = 'defaultLit'

            self.pcdview.scene.add_geometry('pcd', dummy_pcd, self.pcd_material)
            self.flag_gui_init = True

        pcd = self.current_pcd
        if pcd is None:
            return

        if self.display_mode == 'Segmentation' and self.current_seg is not None:
            labels = self.current_seg  # Assuming labels is either a numpy array or a torch tensor

            # If labels is a torch tensor (possibly on GPU), handle accordingly
            if isinstance(labels, torch.Tensor):
                device = labels.device  # Get the device (CPU or GPU)

                # Convert palettes to a torch tensor on the same device
                PALETTE_TENSOR = torch.tensor(self.palettes, device=device, dtype=torch.float32) / 255.0

                num_points = labels.shape[0]

                # Initialize colors tensor with zeros (black color)
                colors = torch.zeros((num_points, 3), device=device, dtype=torch.float32)

                # Get valid indices where labels are non-negative
                valid_indices = torch.nonzero(labels >= 0, as_tuple=False).squeeze(1)

                # Assign colors to valid points
                colors[valid_indices] = PALETTE_TENSOR[labels[valid_indices]]

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

            # Set shader to defaultLit for segmentation coloring
            self.pcd_material.shader = 'defaultLit'
            pcd_to_display = pcd

        elif self.display_mode == 'Normals':
            # Use normals for coloring; shader will use normals
            self.pcd_material.shader = 'normals'
            pcd_to_display = pcd

        else:  # 'Colors' mode
            # Use colors as usual
            self.pcd_material.shader = 'defaultLit'
            pcd_to_display = pcd

        # Update the geometry in the scene
        if self.pcdview.scene.has_geometry('pcd'):
            self.pcdview.scene.remove_geometry('pcd')
        self.pcdview.scene.add_geometry('pcd', pcd_to_display, self.pcd_material)
        self.pcdview.force_redraw()


    def on_layout(self, layout_context):
        """Callback on window initialize / resize"""
        frame = self.window.content_rect
        panel_width = self.em * 22
        # Set the frame for the panel on the right side
        self.scene_widgets.panel.frame = gui.Rect(frame.get_right() - panel_width,
                                               frame.y, panel_width,
                                               frame.height)

        # Set the frame for the scene on the left side
        self.pcdview.frame = gui.Rect(frame.x, frame.y,
                                      frame.width - panel_width,
                                      frame.height)

        pref = self.scene_widgets.fps_label.calc_preferred_size(layout_context,
                                                             gui.Widget.Constraints())
        self.scene_widgets.fps_label.frame = gui.Rect(frame.x,
                                                   frame.get_bottom() - pref.height,
                                                   pref.width, pref.height)


    def _on_bbox_slider_changed(self, value, param):
        self.bbox_params[param] = value
        # Update the corresponding NumberEdit widget
        self.scene_widgets.bbox_edits[param].double_value = value
        self.update_bounding_box()

    def _on_bbox_edit_changed(self, value, param):
        self.bbox_params[param] = value
        # Update the corresponding slider
        self.scene_widgets.bbox_sliders[param].double_value = value
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
