import math
import time
import matplotlib.pyplot as plt
from joyconrobotics import JoyconRobotics

import numpy as np
from scipy.spatial.transform import Rotation as R

from remote_control import CalibrationManager
from pose import Pose
from robot_utils import RobotInterface


class RobotController:
    def __init__(self):
        self.joyconrobotics_right = JoyconRobotics("right")
        self.robot = RobotInterface()
        self.robot.find_device()
        # self.robot.connect()
        self.max_time = 10
        self.control_active = False 
        self.initial_controller_pose = None 
        self.calib_manager: CalibrationManager = None
        self.calibrate()

    def calibrate(self):
        target_pose = self.robot.euler_dict_to_array(self.robot.get_tcp_position())
        controller_pose,_,_ = self.joyconrobotics_right.get_control()
        self.calib_manager = CalibrationManager(controller_pose, target_pose)
        self.calib_manager.calculate_calibration()

    def toggle_control(self):
        # Toggle the control state when 'X' button is pressed
        if self.joyconrobotics_right.button.get_button_x() == 1:
            self.calibrate()
            self.control_active = not self.control_active
            state = "started" if self.control_active else "stopped"
            print(f'Control {state}')

            if self.control_active:
                controller_pose, _, _ = self.joyconrobotics_right.get_control()
                self.initial_controller_pose = np.array(controller_pose)
                print(f"Initial controller pose for calibration: {self.initial_controller_pose}")


    def run(self):
        while True:  # Keep running the loop
            self.toggle_control()  # Check for button press to toggle control
            if self.control_active:
                pose, gripper, control_button = self.joyconrobotics_right.get_control()

                if self.initial_controller_pose is not None:
                    # Apply the calibration matrix to the controller pose
                    calibrated_pose = self.calib_manager.update_target_poses(Pose.from_1d_array(pose, vector_type='euler'))
                    try:
                        self.robot.move_to_pose(calibrated_pose)
                    except TypeError:
                        print("Invalid pose. Skipping...")

            time.sleep(0.01)

controller = RobotController()
controller.run()
