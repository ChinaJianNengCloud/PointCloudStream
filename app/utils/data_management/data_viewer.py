import vtk
import numpy as np
import json
from pathlib import Path

class SceneViewerVTK:
    def __init__(self, data, resource_path: Path):
        self.data = data
        self.resource_path = resource_path
        self.current_scene_index = 0
        self.valid_scenes = {key: True for key in data.keys()}
        # VTK Renderer and Window setup
        self.renderer = vtk.vtkRenderer()
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
        colors.SetNumberOfComponents(3)

        with open(file_path, 'r') as f:
            ply_data = f.readlines()
        header = True
        for line in ply_data:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue
            values = line.split()
            # print(f"RGB Range: {min(values[3:6])}, {max(values[3:6])}")
            x, y, z = map(float, values[:3])
            r, g, b = [int(c) for c in values[3:6]]
            points.InsertNextPoint(x, y, z)
            colors.InsertNextTuple3(r, g, b)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)

        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(poly_data)
        vertex_filter.Update()

        poly_data = vertex_filter.GetOutput()
        poly_data.GetPointData().SetScalars(colors)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
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

    def update_scene(self):
        # Save current camera parameters
        camera = self.renderer.GetActiveCamera()
        position = camera.GetPosition()
        focal_point = camera.GetFocalPoint()
        view_up = camera.GetViewUp()

        self.renderer.RemoveAllViewProps()

        scene_key = list(self.data.keys())[self.current_scene_index]
        pcd_file = self.resource_path / self.data[scene_key]['point_cloud_files'][-1]
        pcd_actor = self.load_point_cloud(pcd_file)

        self.renderer.AddActor(pcd_actor)

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

    def run(self):
        self.update_scene()

        def keypress_callback(obj, event):
            key = obj.GetKeySym().lower()
            if key == 'a':
                self.current_scene_index = max(0, self.current_scene_index - 1)
                self.update_scene()
            elif key == 'd':
                self.current_scene_index = min(len(self.data) - 1, self.current_scene_index + 1)
                self.update_scene()
            elif key == 'x':
                self.mark_invalid()
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
    main_path = '/home/capre/disk_4/yutao/leo_data/data_2nd'

    path = Path(main_path)
    data_json = path / 'all_data.json'
    data = json.load(open(data_json))
    resource_path = path / 'resources'
    viewer = SceneViewerVTK(data, resource_path)
    viewer.run()
