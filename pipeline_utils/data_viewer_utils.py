import vtk
import numpy as np
import json
import atexit
from pathlib import Path

class SceneViewerVTK:
    def __init__(self, data, resource_path: Path, validity_path: str = 'validity.json'):
        self.data = data
        self.resource_path = resource_path
        self.current_scene_index = 0

        # Instead of only a valid flag, store both valid and inverse in scene_flags
        # Default: valid=True, inverse=False for all scenes
        self.scene_flags = {
            key: {"valid": True, "inverse": False}
            for key in data.keys()
        }

        self.validity_path = validity_path
        self.load_validity()  # Merge any saved flags with defaults

        # Show segmentation color toggle
        self.show_seg_color = False
        
        # Track which index of the point_cloud_files to use (0 or -1)
        self.pcd_file_index = 0  # Start using the first file, 0

        # Keep track of point size for the point cloud
        self.point_size = 2

        # VTK setup
        self.renderer = vtk.vtkRenderer()
        # We'll set background color later in update_scene depending on validity
        self.renderer.SetBackground(0.2, 0.2, 0.2)  # fallback
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(1280, 720)
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.interactor.SetRenderWindow(self.render_window)

        # Primary text actor (scene info)
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput("Loading...")
        self.text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        self.text_actor.GetTextProperty().SetFontSize(20)
        # Initial (fallback) display position
        self.text_actor.SetDisplayPosition(10, 10)
        self.renderer.AddActor2D(self.text_actor)

        # Manual text actor for key bindings
        self.manual_text_actor = vtk.vtkTextActor()
        self.manual_text_actor.SetInput(
            "Key Binds:\n"
            "  A/D : Previous/Next Scene\n"
            "  Z/C : Previous/Next Valid Scene\n"
            "  X   : Toggle Validity\n"
            "  S   : Toggle Inverse\n"
            "  V   : Toggle Seg Color\n"
            "  N/M : Smaller/Bigger Points\n"
            "  F   : Toggle PCD File\n"
            "  Q   : Save & Quit\n"
        )
        self.manual_text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        self.manual_text_actor.GetTextProperty().SetFontSize(18)
        # Initial (fallback) display position
        self.manual_text_actor.SetDisplayPosition(10, 130)
        self.renderer.AddActor2D(self.manual_text_actor)

        # Keep text in top-left corner on window resize (using relative positions)
        self.render_window.AddObserver("ModifiedEvent", self.update_text_position)

    def load_validity(self):
        """Load scene_flags from an existing JSON file if available."""
        validity_file = Path(self.validity_path)
        if validity_file.exists():
            try:
                with open(validity_file, 'r') as f:
                    loaded_flags = json.load(f)
                # Merge loaded flags into scene_flags
                for key, flags_dict in loaded_flags.items():
                    if key in self.scene_flags and isinstance(flags_dict, dict):
                        if "valid" in flags_dict:
                            self.scene_flags[key]["valid"] = flags_dict["valid"]
                        if "inverse" in flags_dict:
                            self.scene_flags[key]["inverse"] = flags_dict["inverse"]
                print(f"Loaded existing flags from {self.validity_path}")
            except Exception as e:
                print(f"Warning: Could not load {self.validity_path}. Error: {e}")
        else:
            print(f"No existing {self.validity_path} found. Using default flags.")

    def save_validity(self):
        """Save current scene_flags to JSON file."""
        with open(self.validity_path, 'w') as f:
            json.dump(self.scene_flags, f, indent=2)
        print(f"Scene flags saved to {self.validity_path}")

    def update_text_position(self, obj, event):
        """
        Keep text actors in a relative position when the window is resized.
        Adjust the percentages as needed to change the relative position.
        """
        window_width, window_height = self.render_window.GetSize()
        
        # For example, place the main text actor ~2% from the top, 2% from the left
        margin_left = int(window_width * 0.02)
        margin_top_main = int(window_height * 0.02)
        self.text_actor.SetDisplayPosition(margin_left, margin_top_main)

        # Place the manual text actor ~2% from the left, 15% from the top
        margin_top_manual = int(window_height * 0.15)
        self.manual_text_actor.SetDisplayPosition(margin_left, margin_top_manual)

        self.render_window.Render()

    def load_point_cloud(self, file_path):
        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetName("OriginalColors")
        colors.SetNumberOfComponents(3)

        seg_colors = vtk.vtkUnsignedCharArray()
        seg_colors.SetName("SegmentColors")
        seg_colors.SetNumberOfComponents(3)

        with open(file_path, 'r') as f:
            ply_data = f.readlines()

        header_end = [
            i for i, line in enumerate(ply_data)
            if line.strip() == "end_header"
        ]
        start_idx = (header_end[0] + 1) if header_end else 0

        # skipping 7 out of 8 lines for performance
        data = np.array([line.split() for line in ply_data[start_idx::8]], dtype=object)
        xyz = data[:, :3].astype(float)
        rgb = data[:, 3:6].astype(int)
        seg_ids = data[:, 6].astype(int)

        # Group data by seg_ids
        unique_seg_ids = np.unique(seg_ids)
        seg_groups = {seg_id: {"points": [], "colors": []} for seg_id in unique_seg_ids}
        skin_color = np.array([213, 144, 106])
        min_dist = np.inf
        best_group = None

        for i, (x, y, z) in enumerate(xyz):
            r, g, b = rgb[i]
            seg_id = seg_ids[i]
            seg_groups[seg_id]["points"].append((x, y, z))
            seg_groups[seg_id]["colors"].append((r, g, b))

        # Find segment whose average color is closest to skin_color
        for seg_id, group in seg_groups.items():
            avg_color = np.mean(group["colors"], axis=0)
            dist = np.linalg.norm(avg_color - skin_color)
            if dist < min_dist:
                min_dist = dist
                best_group = seg_id

        sphere_center = None
        if best_group is not None:
            best_points = np.array(seg_groups[best_group]["points"])
            sphere_center = best_points.mean(axis=0)

        # Insert points and colors
        for x, y, z in xyz:
            points.InsertNextPoint(x, y, z)
        for r, g, b in rgb:
            colors.InsertNextTuple3(r, g, b)

        # Generate pseudo-random segment colors
        for seg_id in seg_ids:
            np.random.seed(seg_id)
            seg_color = np.random.randint(0, 256, size=3)
            seg_colors.InsertNextTuple3(*seg_color)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.GetPointData().SetScalars(colors)
        poly_data.GetPointData().AddArray(seg_colors)

        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(poly_data)
        vertex_filter.Update()
        poly_data = vertex_filter.GetOutput()

        # Optionally add a sphere at the center
        if sphere_center is not None:
            print(f"Adding sphere at center of skin-like group: {sphere_center}")
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*sphere_center)
            sphere.SetRadius(0.05)

            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere.GetOutputPort())

            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.GetProperty().SetColor(1, 0.8, 0.6)
            self.renderer.AddActor(sphere_actor)
            print(f"Added sphere at center of skin-like group: {sphere_center}")

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 1)

        # Attach sphere_center to the actor so we can reference it later
        actor.sphere_center = sphere_center

        return actor

    def add_pose_lines(self, poses):
        """Draw lines between consecutive poses."""
        if len(poses) < 2:
            return

        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        for i, pose in enumerate(poses):
            x, y, z, *_ = pose
            points.InsertNextPoint(x, y, z)
            if i > 0:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i - 1)
                line.GetPointIds().SetId(1, i)
                lines.InsertNextCell(line)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # red lines
        self.renderer.AddActor(actor)

    def update_scene(self):
        # Save current camera parameters
        camera = self.renderer.GetActiveCamera()
        position = camera.GetPosition()
        focal_point = camera.GetFocalPoint()
        view_up = camera.GetViewUp()

        # Clear old props, re-add text actors
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor2D(self.text_actor)
        self.renderer.AddActor2D(self.manual_text_actor)

        # Identify scene
        scene_key = list(self.data.keys())[self.current_scene_index]
        # Use pcd_file_index to pick file 0 or -1
        pcd_file = self.resource_path / self.data[scene_key]['point_cloud_files'][self.pcd_file_index]

        # Load point cloud
        pcd_actor = self.load_point_cloud(pcd_file)

        # Apply color mode
        if self.show_seg_color:
            pcd_actor.GetMapper().ScalarVisibilityOn()
            pcd_actor.GetMapper().SetScalarModeToUsePointFieldData()
            pcd_actor.GetMapper().SelectColorArray("SegmentColors")
        else:
            pcd_actor.GetMapper().ScalarVisibilityOn()
            pcd_actor.GetMapper().SetScalarModeToUsePointFieldData()
            pcd_actor.GetMapper().SelectColorArray("OriginalColors")

        # Apply point size
        pcd_actor.GetProperty().SetPointSize(self.point_size)

        self.renderer.AddActor(pcd_actor)

        # Pose data
        poses = self.data[scene_key].get('pose', [])
        self.add_pose_lines(poses)

        # Add axis to last pose
        distance_str = "not available"
        sphere_center = getattr(pcd_actor, 'sphere_center', None)  # may be None or np.array

        if poses:
            last_pose = poses[-1]
            x, y, z, rx, ry, rz = last_pose

            last_axes = vtk.vtkAxesActor()
            last_axes.SetTotalLength(0.1, 0.1, 0.1)
            last_axes.SetShaftTypeToCylinder()
            last_axes.SetCylinderRadius(0.02)
            last_axes.GetXAxisShaftProperty().SetColor(1, 0, 0)
            last_axes.GetYAxisShaftProperty().SetColor(0, 1, 0)
            last_axes.GetZAxisShaftProperty().SetColor(0, 0, 1)
            last_axes.SetXAxisLabelText("")
            last_axes.SetYAxisLabelText("")
            last_axes.SetZAxisLabelText("")

            transform = vtk.vtkTransform()
            transform.Translate(x, y, z)
            transform.RotateX(np.degrees(rx))
            transform.RotateY(np.degrees(ry))
            transform.RotateZ(np.degrees(rz))
            last_axes.SetUserTransform(transform)
            self.renderer.AddActor(last_axes)

            # If we have a sphere center, calculate distance from last pose to center
            if sphere_center is not None:
                dist = np.linalg.norm(np.array([x, y, z]) - sphere_center)
                distance_str = f"{dist:.3f}"

        # Get scene validity & inverse flags
        valid_state = self.scene_flags[scene_key]["valid"]
        inverse_state = self.scene_flags[scene_key]["inverse"]

        # -----------  BACKGROUND COLOR BASED ON VALIDITY  ----------- #
        if valid_state:
            # Slight green for valid scenes
            self.renderer.SetBackground(0.16, 0.2, 0.16)
        else:
            # Slight red for invalid scenes
            self.renderer.SetBackground(0.2, 0.16, 0.16)

        # Build multiline text (scene info)
        total_scenes = len(self.data)
        current_index = self.current_scene_index + 1
        prompt_str = self.data[scene_key].get('prompt', "No prompt available.")

        multiline_text = (
            f"prompt [{current_index}/{total_scenes}] : {prompt_str}\n"
            f"valid: {valid_state}\n"
            f"inverse: {inverse_state}\n"
            f"distance: {distance_str}"
        )

        self.text_actor.SetInput(multiline_text)

        # Restore camera
        camera.SetPosition(position)
        camera.SetFocalPoint(focal_point)
        camera.SetViewUp(view_up)

        self.render_window.Render()
        print(f"Scene: {scene_key}, file_index={self.pcd_file_index} flags: {self.scene_flags[scene_key]}")
        print(multiline_text)

    #
    # Toggling Methods
    #
    def toggle_validity(self):
        """Toggle the 'valid' state of the current scene."""
        scene_key = list(self.data.keys())[self.current_scene_index]
        self.scene_flags[scene_key]["valid"] = not self.scene_flags[scene_key]["valid"]
        print(f"Scene {scene_key} valid toggled -> {self.scene_flags[scene_key]['valid']}")
        self.update_scene()

    def toggle_inverse(self):
        """Toggle the 'inverse' state of the current scene."""
        scene_key = list(self.data.keys())[self.current_scene_index]
        self.scene_flags[scene_key]["inverse"] = not self.scene_flags[scene_key]["inverse"]
        print(f"Scene {scene_key} inverse toggled -> {self.scene_flags[scene_key]['inverse']}")
        self.update_scene()

    #
    # Navigation Among Scenes
    #
    def go_previous_valid_scene(self):
        """Go to the previous scene whose 'valid' is True."""
        new_index = self.current_scene_index - 1
        while new_index >= 0:
            scene_key = list(self.data.keys())[new_index]
            if self.scene_flags[scene_key]["valid"]:
                self.current_scene_index = new_index
                self.update_scene()
                return
            new_index -= 1
        print("No previous valid scene found.")

    def go_next_valid_scene(self):
        """Go to the next scene whose 'valid' is True."""
        new_index = self.current_scene_index + 1
        total_scenes = len(self.data)
        while new_index < total_scenes:
            scene_key = list(self.data.keys())[new_index]
            if self.scene_flags[scene_key]["valid"]:
                self.current_scene_index = new_index
                self.update_scene()
                return
            new_index += 1
        print("No next valid scene found.")

    def toggle_seg_color(self):
        """Toggle segmentation color."""
        self.show_seg_color = not self.show_seg_color
        print(f"Toggled segmentation color mode -> {self.show_seg_color}")
        self.update_scene()

    #
    # Methods to Adjust Point Size
    #
    def bigger_points(self):
        """Increase point size (up to a limit)."""
        self.point_size = min(20, self.point_size + 1)
        print(f"Point size increased to {self.point_size}")
        self.update_scene()

    def smaller_points(self):
        """Decrease point size (down to a limit)."""
        self.point_size = max(1, self.point_size - 1)
        print(f"Point size decreased to {self.point_size}")
        self.update_scene()

    #
    # Toggle PCD File (0 or -1)
    #
    def toggle_pcd_file(self):
        """Press 'f' to switch pcd_file_index between 0 and -1."""
        if self.pcd_file_index == 0:
            self.pcd_file_index = -1
        else:
            self.pcd_file_index = 0
        print(f"Toggled pcd_file_index to {self.pcd_file_index}")
        self.update_scene()

    #
    # Main Entry
    #
    def run(self):
        self.update_scene()

        def keypress_callback(obj, event):
            key = obj.GetKeySym().lower()
            if key == 'a':
                # Previous scene
                self.current_scene_index = max(0, self.current_scene_index - 1)
                self.update_scene()
            elif key == 'd':
                # Next scene
                self.current_scene_index = min(len(self.data) - 1, self.current_scene_index + 1)
                self.update_scene()
            elif key == 'x':
                # Toggle 'valid'
                self.toggle_validity()
            elif key == 's':
                # Toggle 'inverse'
                self.toggle_inverse()
            elif key == 'z':
                # Go to previous valid
                self.go_previous_valid_scene()
            elif key == 'c':
                # Go to next valid
                self.go_next_valid_scene()
            elif key == 'v':
                # Toggle segmentation color
                self.toggle_seg_color()
            elif key == 'n':
                # Smaller points
                self.smaller_points()
            elif key == 'm':
                # Bigger points
                self.bigger_points()
            elif key == 'f':
                # Toggle pcd file index
                self.toggle_pcd_file()
            elif key == 'q':
                # Save validity and quit
                self.save_validity()
                self.interactor.TerminateApp()

        self.interactor.AddObserver("KeyPressEvent", keypress_callback)
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()

if __name__ == '__main__':
    # Example usage:
    main_path = '/home/capre/disk_4/yutao/leo_data/merged_data'  # Adjust path
    path = Path(main_path)
    data_json = path / 'all_data.json'
    with open(data_json, 'r') as f:
        data = json.load(f)

    # Reverse the data dictionary (optional)
    reversed_data = {key: data[key] for key in reversed(list(data.keys()))}
    resource_path = path / 'resources'

    viewer = SceneViewerVTK(reversed_data, resource_path, validity_path='validity.json')
    atexit.register(viewer.save_validity)
    viewer.run()
