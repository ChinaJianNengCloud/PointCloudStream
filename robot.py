import lebai_sdk
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import logging as log
class RoboticArm:
    def __init__(self, ip_address=None, port=None):
        # Initialize SDK and network parameters
        self.ip_address = ip_address
        self.port = port
        self.lebai = None
        # Initialize motion parameters
        self.acceleration = 0.5
        self.velocity = 0.2
        self.time_running = 0
        self.radius = 0

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
            log.info(f"Connected to robotic arm at: {self.ip_address}")
        except Exception as e:
            log.info(f"Failed to connect to robotic arm: {e}")

    def disconnect(self):
        """Disconnect from the robotic arm."""
        self.lebai.stop_sys()
        log.info("Disconnected from robotic arm.")

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
            log.info(f"Robot moving to joint position: {joint_pose}")
            self.lebai.wait_move()
        except Exception as e:
            log.info(f"Failed to send command: {e}")

    def capture_gripper_to_base(self):
        """Capture transformation from gripper to base using forward kinematics."""
        cpose = self.lebai.get_kin_data()['actual_tcp_pose']
        x, y, z = cpose['x'], cpose['y'], cpose['z']
        Rz, Ry, Rx = cpose['rz'], cpose['ry'], cpose['rx']
        rotation = R.from_euler('zyx', [Rz, Ry, Rx], degrees=False)
        R_g2b = rotation.as_matrix()
        t_g2b = np.array([[x], [y], [z]])
        return R_g2b, t_g2b


if __name__ == "__main__":
    arm = RoboticArm()
    arm.connect()
    # arm.move_command([0, 0, 0, 0, 0, 0])
    # arm.disconnect()