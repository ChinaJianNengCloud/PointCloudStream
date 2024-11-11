import lebai_sdk
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import logging
import threading
import time
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import open3d as o3d

logger = logging.getLogger(__name__)
class RobotInterface:
    def __init__(self, ip_address=None, port=None):
        # Initialize SDK and network parameters
        self.ip_address = ip_address
        self.port = port
        self.lebai = None
        # Initialize motion parameters
        self.acceleration = 0.8
        self.velocity = 0.4
        self.time_running = 0
        self.radius = 0
        self.motion_flag = False

    def find_device(self):
        """Find the device and connect to it."""
        lebai_sdk.init()
        if not self.ip_address:
            self.ip_address = lebai_sdk.discover_devices(1)[0]['ip']
        else:
            self.ip_address = self.ip_address
        self.lebai = lebai_sdk.connect(self.ip_address, False)

    def connect(self):
        """Establish network connection to the robotic arm."""
        try:
            self.lebai = lebai_sdk.connect(self.ip_address, False)
            self.lebai.start_sys()
            logging.info(f"Connected to robotic arm at: {self.ip_address}")
        except Exception as e:
            logging.info(f"Failed to connect to robotic arm: {e}")

    def disconnect(self):
        """Disconnect from the robotic arm."""
        self.lebai.stop_sys()
        logging.info("Disconnected from robotic arm.")

    def update_motion_parameters(self, acceleration, velocity, time_running, radius):
        self.acceleration = acceleration
        self.velocity = velocity
        self.time_running = time_running
        self.radius = radius

    def get_position(self):
        """Retrieve the current position of the robot's end-effector."""
        position = self.lebai.get_kin_data()
        return position['actual_tcp_pose']
    
    def move_command(self, joint_pose):
        """Send movement command to the robotic arm."""
        try:
            self.lebai.movej(joint_pose, self.acceleration, self.velocity, self.time_running, self.radius)
            logging.info(f"Robot moving to joint position: {joint_pose}")
            self.lebai.wait_move()
        except Exception as e:
            logging.info(f"Failed to send command: {e}")

    def capture_gripper_to_base(self):
        """Capture transformation from gripper to base using forward kinematics."""
        cpose = self.pose_dict_to_array(self.lebai.get_kin_data()['actual_tcp_pose'])
        # x, y, z = cpose['x'], cpose['y'], cpose['z']
        # Rz, Ry, Rx = cpose['rz'], cpose['ry'], cpose['rx']
        # rotation = R.from_euler('xyz', [Rx, Ry, Rz], degrees=False)
        # R_g2b = rotation.as_matrix()
        R_g2b = cpose[3:6]
        t_g2b = cpose[0:3]
        return R_g2b, t_g2b
    
    def pose_dict_to_array(self, pose_quaternion_dict):
        # logger.debug("original pose: ", pose_quaternion_dict)
        # logger.debug("array pose: ", trs)
        return np.array([pose_quaternion_dict[key] for key in ['x', 'y', 'z', 'rx', 'ry', 'rz']])
    
    def pose_array_to_dict(self, pose_array):
        pose_array = pose_array.tolist()
        pose_array = [float(x) for x in pose_array]
        keys = ['x', 'y', 'z', 'rx', 'ry', 'rz']
        return dict(zip(keys, pose_array))
    
    def pose_unit_change_to_machine(self, pose_array):
        pose_array[0:3] = pose_array[0:3] / 1000
        pose_array[3:6] = np.deg2rad(pose_array[3:6])
        return pose_array
    
    def pose_unit_change_to_store(self, pose_array):
        pose_array[0:3] = pose_array[0:3] * 1000
        pose_array[3:6] = np.rad2deg(pose_array[3:6])
        return pose_array
    
    
    def move_with_pose_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        cartesian_poses_dict = [line.strip().split() for line in lines]
        cartesian_poses_dict = [np.array([float(x) for x in line], dtype=np.float32) for line in cartesian_poses_dict]
        cartesian_poses_dict =[np.hstack((line[0:3] / 1000, np.deg2rad(line[3:6]))) for line in cartesian_poses_dict] #[line[0:3] / 1000 + line[3:6] * np.pi / 180 for line in joint_poses]

        cartesian_poses_dict_list = [self.pose_array_to_dict(pose) for pose in cartesian_poses_dict]

        for cartesian_poses_dict in cartesian_poses_dict_list:
            self.motion_flag = False
            cartesian_poses_array = self.pose_dict_to_array(cartesian_poses_dict)
            print("move pose", cartesian_poses_array.tolist())

            # print("pose", np.degrees(pose[3:6]))
            # joint_pose = self.lebai.kinematics_inverse(cartesian_poses_array.tolist())
            # print("joint_pose", joint_pose)

            joint_pose = self.lebai.kinematics_inverse(cartesian_poses_dict)
            print("move joint_pose", joint_pose)
            self.lebai.movej(joint_pose, self.acceleration, self.velocity, self.time_running, self.radius)
            self.lebai.wait_move()
            # motion_id = self.lebai.movel(cartesian_poses_dict, self.acceleration, self.velocity, self.time_running, self.radius)
            # self.lebai.wait_move()
            print("current pose", self.pose_unit_change_to_store(
                self.pose_dict_to_array(self.lebai.get_kin_data()['actual_tcp_pose'])))
            self.motion_flag = True
            time.sleep(0.5)



if __name__ == "__main__":
    arm = RobotInterface()
    import time
    arm.find_device()
    arm.connect()
    path = "pose.txt"
    test_pose = np.array([-644, 30, 81, -36, -12, 137], dtype=np.float32)
    test_joint_pose = np.array([-19, -4, 27, -118, -46, 114], dtype=np.float32)
    test_machine_joint_pose = np.deg2rad(test_joint_pose).tolist()
    print("test_forward_pose", arm.lebai.kinematics_forward(test_machine_joint_pose))
    test_machine_pose = arm.pose_unit_change_to_machine(test_pose)
    print("test_machine_pose", test_machine_pose)
    # arm.move_command(np.deg2rad(test_joint_pose).tolist())
    print("dict_pose", arm.pose_array_to_dict(test_machine_pose))
    # inversed_test_joint_pose = arm.lebai.kinematics_inverse(arm.pose_array_to_dict(test_machine_pose))
    # print("inversed_test_joint_pose", inversed_test_joint_pose)
    print("arm", arm.capture_gripper_to_base())
    print("curr joint pose", arm.lebai.get_kin_data()['actual_joint_pose'])
    # print("curr joint pose", np.rad2deg(np.array(arm.lebai.get_kin_data()['actual_joint_pose'])))
    # print(test_pose)
    # test_machine_pose = arm.pose_unit_change_to_machine(test_pose)
    # print(test_machine_pose)
    # curr_pose = arm.get_position()
    # # print("kinematics_inverse", arm.lebai.kinematics_forward(test_joint_pose.tolist()))
    # test_machine_joint_pose = np.deg2rad(test_joint_pose).tolist()
    # # arm.move_command(test_machine_joint_pose)
    # test_pose_dict = arm.pose_array_to_dict(test_machine_pose)
    # print(test_pose_dict)
    # print(test_machine_pose)
    # arm.lebai.movej(test_pose_dict, arm.acceleration, arm.velocity, arm.time_running, arm.radius)
    arm.move_with_pose_file(path)
    # print(arm.get_position())
    # # print(arm.lebai.get_kin_data().keys())
    # print(arm.lebai.get_kin_data()['actual_tcp_pose'])
    # # print(arm.lebai.get_kin_data()['actual_flange_pose'])
    # print("rx ry rz:", arm.capture_gripper_to_base()[0].ravel())
    # print(cv2.Rodrigues(cv2.Rodrigues(arm.capture_gripper_to_base()[0])[0])[0].ravel())
    # arm.lebai.start_record_trajectory()
    # time.sleep(10)
    # arm.lebai.end_record_trajectory("calib")
    # motion_id = arm.lebai.move_trajectory("calib")
    # arm.move_command([0, 0, 0, 0, 0, 0])
    # arm.disconnect()