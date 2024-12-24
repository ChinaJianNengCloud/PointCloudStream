import vtk
import numpy as np
import json
from pathlib import Path
# vtk.vtkPLYReader
class SceneViewerVTK:
    def __init__(self, data, resource_path: Path):
        self.data = data
        self.resource_path = resource_path
        self.current_scene_index = 0
        self.valid_scenes = {key: True for key in data.keys()}
        self.show_seg_color = False  # New toggle state

        # VTK Renderer and Window setup
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(1280, 720)
        self.render_window.AddRenderer(self.renderer)

        self.interactor = vtk.vtkRenderWindowInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.interactor.SetRenderWindow(self.render_window)

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

        # Process lines efficiently using numpy
        header_end = [i for i, line in enumerate(ply_data) if line.strip() == "end_header"]
        if header_end:
            start_idx = header_end[0] + 1
        else:
            start_idx = 0

        data = np.array([line.split() for line in ply_data[start_idx:]], dtype=object)
        xyz = data[:, :3].astype(float)
        rgb = data[:, 3:6].astype(int)
        seg_ids = data[:, 6].astype(int)
        ########################################
        # Group data by seg_ids
        unique_seg_ids = np.unique(seg_ids)
        seg_groups = {seg_id: {"points": [], "colors": []} for seg_id in unique_seg_ids}
        skin_color = np.array([213, 144, 106])  # Approximate skin tone in RGB
        min_dist = np.inf
        best_group = None

        for i, (x, y, z) in enumerate(xyz):
            r, g, b = rgb[i]
            seg_id = seg_ids[i]
            seg_groups[seg_id]["points"].append((x, y, z))
            seg_groups[seg_id]["colors"].append((r, g, b))

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

        ########################################
        # Insert points and colors into vtkPoints and vtkUnsignedCharArray
        for x, y, z in xyz:
            points.InsertNextPoint(x, y, z)
        for r, g, b in rgb:
            colors.InsertNextTuple3(r, g, b)
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
        # Add a ball at the center of the skin-like group###################
        if sphere_center is not None:
            print(f"Adding sphere at center of skin-like group: {sphere_center}")
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(*sphere_center)
            sphere.SetRadius(0.05)  # Adjust size as needed

            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere.GetOutputPort())

            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.GetProperty().SetColor(1, 0.8, 0.6)  # Skin-like color for the ball
            self.renderer.AddActor(sphere_actor)
            print(f"Added sphere at center of skin-like group: {sphere_center}")
        ######################################################
    
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 1, 1)  # Default white
        actor.poly_data = poly_data  # Attach poly_data for toggling

        return actor

    def add_pose_axes(self, poses):
        for pose in poses:
            x, y, z, rx, ry, rz = pose

            # Create axes
            axes = vtk.vtkAxesActor()
            axes.SetPosition(x, y, z)
            self.renderer.AddActor(axes)

    def add_pose_lines(self, poses):
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
        actor.GetProperty().SetColor(1, 0, 0)  # Red lines
        self.renderer.AddActor(actor)

    def add_custom_pose(self, pose):
        x, y, z, rx, ry, rz = pose
        # Create a custom axis
        axes = vtk.vtkAxesActor()
        axes.SetPosition(x, y, z)
        axes.SetXAxisLabelText("")
        axes.SetYAxisLabelText("Predicted")
        axes.SetZAxisLabelText("")
        axes.GetYAxisCaptionActor2D().SetWidth(0.1)
        axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 1)  # Green color
        axes.SetTotalLength(0, 0, 0.3)
        # axes.SetScale(1, 1, 1.5)
        # Apply rotation (rx, ry, rz in radians)
        transform = vtk.vtkTransform()
        transform.Translate(x, y, z)
        transform.RotateX(np.degrees(rx))
        transform.RotateY(np.degrees(ry))
        transform.RotateZ(np.degrees(rz))

        axes.SetUserTransform(transform)
        self.renderer.AddActor(axes)
        self.render_window.Render()
        print(f"Added custom pose: {pose}")

    def update_scene(self):
        # Save current camera parameters
        camera = self.renderer.GetActiveCamera()
        position = camera.GetPosition()
        focal_point = camera.GetFocalPoint()
        view_up = camera.GetViewUp()

        self.renderer.RemoveAllViewProps()

        scene_key = list(self.data.keys())[self.current_scene_index]
        pcd_file = self.resource_path / self.data[scene_key]['point_cloud_files'][0]
        pcd_actor = self.load_point_cloud(pcd_file)

        # Apply color mode based on toggle
        if self.show_seg_color:
            pcd_actor.GetMapper().ScalarVisibilityOn()
            pcd_actor.GetMapper().SetScalarModeToUsePointFieldData()
            pcd_actor.GetMapper().SelectColorArray("SegmentColors")
        else:
            pcd_actor.GetMapper().ScalarVisibilityOn()
            pcd_actor.GetMapper().SetScalarModeToUsePointFieldData()
            pcd_actor.GetMapper().SelectColorArray("OriginalColors")

        self.renderer.AddActor(pcd_actor)

        custom_pose = [-0.22745098, -0.07843137,  0.43921569,  0.88703793,  0.6406385,   2.41471435]
        self.add_custom_pose(custom_pose)

        poses = self.data[scene_key].get('pose', [])
        self.add_pose_axes(poses)
        self.add_pose_lines(poses)

        # Reapply saved camera parameters
        camera.SetPosition(position)
        camera.SetFocalPoint(focal_point)
        camera.SetViewUp(view_up)

        self.render_window.Render()
        print(f"Showing scene: {scene_key}, valid: {self.valid_scenes[scene_key]}")

    def mark_invalid(self):
        scene_key = list(self.data.keys())[self.current_scene_index]
        self.valid_scenes[scene_key] = False
        print(f"Marked scene {scene_key} as invalid.")

    def toggle_seg_color(self):
        self.show_seg_color = not self.show_seg_color
        print(f"Toggled segmentation color mode to {self.show_seg_color}")
        self.update_scene()

    def run(self):
        self.update_scene()

        def keypress_callback(obj: vtk.vtkEvent, event):
            key = obj.GetKeySym().lower()
            if key == 'a':
                self.current_scene_index = max(0, self.current_scene_index - 1)
                self.update_scene()
            elif key == 'd':
                self.current_scene_index = min(len(self.data) - 1, self.current_scene_index + 1)
                self.update_scene()
            elif key == 'x':
                self.mark_invalid()
            elif key == 's':
                self.toggle_seg_color()
            elif key == 'q':
                self.save_validity()
                self.interactor.TerminateApp()

        self.interactor.AddObserver("KeyPressEvent", keypress_callback)
        self.interactor.Initialize()
        self.render_window.Render()
        self.interactor.Start()

    def save_validity(self):
        with open('validity.json', 'w') as f:
            json.dump(self.valid_scenes, f)
        print("Validity saved to validity.json")

if __name__ == '__main__':
    ## CONFIG DATA PATH HERE
    main_path = '/home/capre/disk_4/yutao/leo_data/merged_data'

    path = Path(main_path)
    data_json = path / 'all_data.json'
    data = json.load(open(data_json))

    # Reverse the index of the data dictionary
    reversed_data = {key: data[key] for key in reversed(list(data.keys()))}

    resource_path = path / 'resources'
    viewer = SceneViewerVTK(reversed_data, resource_path)
    viewer.run()