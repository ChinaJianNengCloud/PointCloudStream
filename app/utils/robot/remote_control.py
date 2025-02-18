import numpy as np
import vtk
from scipy.spatial.transform import Rotation as R
from pose import Pose

class CalibrationManager:
    def __init__(self, controller_pose, target_pose, coord_system='left'):
        self.controller_pose = Pose.from_1d_array(controller_pose, vector_type='euler')
        self.target_poses = Pose.from_1d_array(target_pose, vector_type='euler')
        self.calibration_pose: Pose = None  # This will store the calibration transformation
        self.coord_system = coord_system

    def convert_coordinate_system(self, pose: Pose):
        # Assuming Z-axis inversion for coordinate system transformation
        if self.coord_system == 'left':
            return pose  # No transformation needed if using left-handed coordinate system
        elif self.coord_system == 'right':
            pose_in_right = pose.copy()
            pose_in_right.z *= -1  # Invert the Z-axis for right-hand coordinate system
            return pose_in_right
        else:
            raise ValueError("Invalid coordinate system. Use 'left' or 'right'.")

    def calculate_calibration(self):
        # Convert poses to a unified coordinate system (e.g., right-handed)
        controller_pose_converted = self.convert_coordinate_system(self.controller_pose)
        target_pose_converted = self.convert_coordinate_system(self.target_poses)
        self.calibration_pose = controller_pose_converted.cal_delta_pose(target_pose_converted, on="base")
    
    def apply_calibration(self, controller_pose: Pose):
        if self.calibration_pose is None:
            raise ValueError("Calibration matrix not calculated. Press 'X' to calculate it.")
        controller_pose_converted = self.convert_coordinate_system(controller_pose)
        return controller_pose_converted.apply_delta_pose(self.calibration_pose, on="base")

    def update_target_poses(self, controller_pose: Pose):
        calibrated_pose = self.apply_calibration(controller_pose)
        return calibrated_pose.to_1d_array(vector_type='euler')

def numpy_to_vtk4x4(matrix: np.ndarray):
    assert matrix.shape == (4, 4)
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, matrix[i, j])
    return vtk_matrix

# --- Helper function: Update a VTK actor's transform from a Pose ---
def update_actor_transform(actor: vtk.vtkAxesActor, pose: Pose):
    """
    Update the VTK actor's transform based on a Pose object.
    Assumes the Pose.to_1d_array returns [x, y, z, roll, pitch, yaw]
    with rotations in radians.
    """
    arr = pose.to_1d_array(vector_type='euler')
    x, y, z, roll, pitch, yaw = arr
    transform = vtk.vtkTransform()
    array4x4 = np.eye(4)
    array4x4[:3, 3] = [x, y, z]
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
    array4x4[:3, :3] = rotation
    vtk_matrix = numpy_to_vtk4x4(array4x4)
    transform.SetMatrix(vtk_matrix)

    actor.SetUserTransform(transform)


# --- Initialize Calibration Manager ---
# Define initial poses (position: x,y,z and Euler angles: roll, pitch, yaw in radians)
controller_pose_initial = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
target_pose_initial     = [1.2, 2.2, 3.2, 0.2, 0.3, 0.4]

cal_manager = CalibrationManager(controller_pose_initial, target_pose_initial)
# "Press X" equivalent: calculate the calibration transformation.
cal_manager.calculate_calibration()


# --- VTK Setup: Create two axes actors for controller and target poses ---
controller_axes = vtk.vtkAxesActor()
target_axes     = vtk.vtkAxesActor()

# Optionally, scale or position the axes for better visibility.
controller_axes.SetTotalLength(0.5, 0.5, 0.5)
target_axes.SetTotalLength(0.5, 0.5, 0.5)

# Create a renderer, render window, and interactor.
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

style = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(style)
renderer.AddActor(controller_axes)
renderer.AddActor(target_axes)
renderer.SetBackground(0.1, 0.2, 0.4)  # Dark blue background

renderWindow.SetSize(800, 600)

# Initialize the actors with the initial poses.
initial_controller_pose = Pose.from_1d_array(controller_pose_initial, vector_type='euler')

update_actor_transform(controller_axes, initial_controller_pose)
initial_target_pose = cal_manager.apply_calibration(initial_controller_pose)
update_actor_transform(target_axes, initial_target_pose)

from joyconrobotics import JoyconRobotics
joyconrobotics_right = JoyconRobotics("right")

# --- Timer Callback for Animation ---
class TimerCallback:
    def __init__(self, cal_manager, controller_actor, target_actor):
        self.cal_manager = cal_manager
        self.controller_actor = controller_actor
        self.target_actor = target_actor
        self.t = 0.0  # time counter

    def execute(self, obj, event):
        # Increase time
        self.t += 0.05

        # Smoothly update the controller pose:
        # - Oscillate the x-position.
        # - Modify the yaw angle.
        new_controller_pose,_,_ = joyconrobotics_right.get_control()
        new_controller_pose = Pose.from_1d_array(new_controller_pose, vector_type='euler')
        update_actor_transform(self.controller_actor, new_controller_pose)
        
        # Update the target pose using the calibration manager.
        updated_target_array = self.cal_manager.update_target_poses(new_controller_pose)
        new_target_pose = Pose.from_1d_array(updated_target_array, vector_type='euler')
        update_actor_transform(self.target_actor, new_target_pose)
        
        # Render the updated scene.
        obj.GetRenderWindow().Render()


# Create and add the timer callback.
timer_callback = TimerCallback(cal_manager, controller_axes, target_axes)
interactor.AddObserver('TimerEvent', timer_callback.execute)
interactor.CreateRepeatingTimer(50)  # update every 50 ms

# --- Start the VTK render loop ---
renderWindow.Render()
interactor.Start()
