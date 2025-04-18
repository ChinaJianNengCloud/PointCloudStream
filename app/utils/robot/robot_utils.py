import lebai_sdk
import numpy as np

import logging
logger = logging.getLogger(__name__)

class SimLebai:
    """Simulation class that mimics the lebai_sdk interface for testing without hardware."""
    
    def __init__(self):
        self._joint_position = np.zeros(6)
        self._tcp_pose = {
            'x': 0.0, 'y': 0.0, 'z': 0.5, 
            'rx': 0.0, 'ry': 0.0, 'rz': 0.0
        }
        self._robot_state = "IDLE"
        self._teach_mode = False
        self._joint_limits_enabled = True
        logger.info("Simulation lebai SDK initialized")
    
    def start_sys(self):
        """Start the robot system."""
        logger.info("Simulation: Robot system started")
        return True
    
    def stop_sys(self):
        """Stop the robot system."""
        logger.info("Simulation: Robot system stopped")
        return True
    
    def get_robot_state(self):
        """Get the current robot state."""
        return self._robot_state
    
    def get_kin_data(self):
        """Get kinematic data including TCP and joint positions."""
        # Generate random TCP pose for simulation
        self._tcp_pose = {
            'x': np.random.uniform(-0.5, 0.5),
            'y': np.random.uniform(-0.5, 0.5),
            'z': np.random.uniform(0.1, 0.8),
            'rx': np.random.uniform(-3.14, 3.14),
            'ry': np.random.uniform(-3.14, 3.14),
            'rz': np.random.uniform(-3.14, 3.14)
        }
        
        # Generate random joint positions
        self._joint_position = np.random.uniform(-3.14, 3.14, 6).tolist()
        
        return {
            'actual_tcp_pose': self._tcp_pose,
            'actual_joint_pose': self._joint_position
        }
    
    def movej(self, joint_position, acceleration, velocity, time_running, radius):
        """Move robot to joint position."""
        self._robot_state = "MOVING"
        self._joint_position = joint_position
        logger.info(f"Simulation: Moving to joint position {joint_position}")
        return True
    
    def wait_move(self):
        """Wait for movement to complete."""
        time.sleep(np.random.uniform(0.5, 1.5))
        self._robot_state = "IDLE"
        logger.info("Simulation: Movement completed")
        return True
    
    def kinematics_inverse(self, tcp_pose):
        """Inverse kinematics to convert TCP pose to joint position."""
        # Generate plausible joint positions for the given TCP pose
        joint_position = np.random.uniform(-3.14, 3.14, 6).tolist()
        logger.info(f"Simulation: Inverse kinematics calculated for {tcp_pose}")
        return joint_position
    
    def teach_mode(self):
        """Enter teach mode."""
        self._teach_mode = True
        self._robot_state = "TEACHING"
        logger.info("Simulation: Entered teach mode")
        return True
    
    def end_teach_mode(self):
        """Exit teach mode."""
        self._teach_mode = False
        self._robot_state = "IDLE"
        logger.info("Simulation: Exited teach mode")
        return True
    
    def enable_joint_limits(self):
        """Enable joint limits."""
        self._joint_limits_enabled = True
        logger.info("Simulation: Joint limits enabled")
        return True
    
    def disable_joint_limits(self):
        """Disable joint limits."""
        self._joint_limits_enabled = False
        logger.info("Simulation: Joint limits disabled")
        return True

class RobotInterface:
    def __init__(self, ip_address=None, port=None, sim: bool = False):
        # Initialize SDK and network parameters
        self.ip_address = ip_address
        self.port = port
        self.sim = sim
        self.lebai = None
        # Initialize motion parameters
        self.acceleration = 2
        self.velocity = 2
        self.time_running = 0
        self.radius = 0
        self.motion_flag = False
        self.recording_flag = False

    def find_device(self):
        """Find the device and connect to it."""
        if self.sim:
            self.ip_address = "127.0.0.1"  # Dummy IP for simulation
            logger.info(f"Simulation mode: Using dummy IP {self.ip_address}")
            self.lebai = SimLebai()
            return
            
        lebai_sdk.init()
        if not self.ip_address:
            self.ip_address = lebai_sdk.discover_devices(1)[0]['ip']
        else:
            self.ip_address = self.ip_address
        self.lebai = lebai_sdk.connect(self.ip_address, False)

    @property
    def is_moving(self):
        # IDEL, TEACHING, MOVING
        return self.lebai.get_robot_state() == 'MOVING'

    def connect(self):
        """Establish network connection to the robotic arm."""
        try:
            if self.sim:
                self.lebai = SimLebai()
            else:
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

    def high_speed_mode(self):
        self.update_motion_parameters(10, 10, 0, 0)
    
    def low_speed_mode(self):
        self.update_motion_parameters(2, 2, 0, 0)

    def get_joint_position(self):
        """Retrieve the current position of the robot's end-effector."""
        position = self.lebai.get_kin_data()
        return position['actual_joint_pose']
    
    def _set_joint_position(self, joint_posistion: np.ndarray, wait=True):
        """Send movement command to the robotic arm."""
        assert joint_posistion.shape == (6,), f"Invalid joint position shape: {joint_posistion.shape}"
        if isinstance(joint_posistion, np.ndarray):
            joint_posistion = joint_posistion.tolist()
        try:
            self.lebai.movej(joint_posistion, self.acceleration, self.velocity, self.time_running, self.radius)
            logging.info(f"Robot moving to joint position: {joint_posistion}")
            if wait:
                self.lebai.wait_move()
        except Exception as e:
            logging.info(f"Failed to send command: {e}")

    def get_state(self, state_type='joint') -> np.ndarray:
        match state_type:
            case 'joint':
                return np.array(self.lebai.get_kin_data()['actual_joint_pose'])
            case 'tcp':
                return self.euler_dict_to_array(self.lebai.get_kin_data()['actual_tcp_pose'])
            case _:
                logging.info(f"Invalid state type: {state_type}") 
                return None

    def step(self, action: np.ndarray, 
             action_type: str, wait=True) -> np.ndarray:
        if action_type == "joint":
            if isinstance(action, list):
                action = np.array(action)
            assert action.shape == (6,), f"Invalid action shape: {action.shape}"
            self._set_joint_position(action, wait=wait)
            return self.get_state('joint')
        elif action_type == "tcp":
            assert action.shape == (6,), f"Invalid action shape: {action.shape}"
            self._set_tcp_position(action, wait=wait)
            return self.get_state('tcp')

    def euler_dict_to_array(self, pose_quaternion_dict):
        return np.array([pose_quaternion_dict[key] for key in ['x', 'y', 'z', 'rx', 'ry', 'rz']])
    
    def pose_array_to_euler_dict(self, pose_array: np.ndarray):
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
        cartesian_poses_array_list = [line.strip().split() for line in lines]
        cartesian_poses_array_list = [np.array([float(x) for x in line], dtype=np.float32) for line in cartesian_poses_array_list]
        cartesian_poses_array_list =[np.hstack((line[0:3] / 1000, np.deg2rad(line[3:6]))) for line in cartesian_poses_array_list]

        for cartesian_poses_array in cartesian_poses_array_list:
            self.motion_flag = False
            self.step(cartesian_poses_array, action_type='tcp', wait=True)
            
            print("current pose", self.pose_unit_change_to_store(
                self.euler_dict_to_array(self.lebai.get_kin_data()['actual_tcp_pose'])))
            self.motion_flag = True
            # time.sleep(0.5)

    def _set_tcp_position(self, pose_array: np.ndarray, wait=True):
        try:
            joint_position = self.lebai.kinematics_inverse(
                self.pose_array_to_euler_dict(pose_array))
            logger.info(f"Moving to joint position: {joint_position}")
            self._set_joint_position(joint_position, wait=wait)
        except Exception as e:
            logger.error(f"Kinematics inverse failed: {e}")
    
    def set_joint_limits(self, limit):
        if not limit:
            self.lebai.disable_joint_limits()
        else:
            self.lebai.enable_joint_limits()

    def on_robot_state(self, robot_state):
        print("robot_state", robot_state)
        

    def set_teach_mode(self, teach_mode):
        if teach_mode:
            try:
                self.lebai.teach_mode()
            except Exception as e:
                logging.info(f"Already in teach mode: {e}")
        else:
            try:
                self.lebai.end_teach_mode()
            except Exception as e:
                logging.info(f"Failed to end teach mode: {e}")

if __name__ == "__main__":
    arm = RobotInterface(sim=True)
    import time
    arm.find_device()
    print(arm.ip_address)
    arm.connect()
    # path = "pose.txt"
    # test_pose = np.array([-644, 30, 81, -36, -12, 137], dtype=np.float32)
    test_joint_pose = np.array([-19, -4, 27, -118, -46, 114], dtype=np.float32)
    test_pose = np.hstack([[-0.057, -0.165, 0.558], np.deg2rad([-51, -35, -53])])
    # arm.lebai.movel([0 , 0, 0, 0, 0, 0 ], 1, 1, 0, 0)
    print(arm.lebai.get_kin_data())
    print(arm.get_joint_position())
    print(np.rad2deg(arm.get_joint_position()))

    # arm.set_teach_mode(True)
    # test_machine_joint_pose = np.deg2rad(test_joint_pose).tolist()
    # print("test_forward_pose", arm.lebai.kinematics_forward(test_machine_joint_pose))
    # test_machine_pose = arm.pose_unit_change_to_machine(test_pose)
    # print("test_machine_pose", test_machine_pose)
    # # arm.move_command(np.deg2rad(test_joint_pose).tolist())
    # print("dict_pose", arm.pose_array_to_dict(test_machine_pose))
    # # inversed_test_joint_pose = arm.lebai.kinematics_inverse(arm.pose_array_to_dict(test_machine_pose))
    # # print("inversed_test_joint_pose", inversed_test_joint_pose)
    # print("arm", arm.capture_gripper_to_base())
    # print("curr joint pose", arm.lebai.get_kin_data()['actual_joint_pose'])
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
    # arm.move_with_pose_file(path)
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