import time
import logging
import numpy as np
import socket
import pickle
from app.utils.robot.openpi_client import websocket_client_policy
from typing import List, Tuple
from PySide6.QtCore import QThread, Signal
# Import specific modules from vtkmodules


from app.utils import RobotInterface

logger = logging.getLogger(__name__)

class RobotTcpOpThread(QThread):
    progress = Signal(int)  # Signal to communicate progress updates
    def __init__(self, robot: RobotInterface, robot_poses: List[np.ndarray]):
        self.robot = robot
        self.robot_poses = robot_poses
        self.robot.update_motion_parameters(
            acceleration=10, velocity=10, time_running=0, radius=0
        )
        super().__init__()

    def run(self):
        self.robot.set_teach_mode(False)
        for idx, action in enumerate(self.robot_poses):
            logger.info(f"Moving to pose {idx}")
            try:
                self.robot.step(action, action_type='tcp', wait=True)
            except Exception as e:
                logger.error(f"Failed to move to pose {idx}: {e}")
                logger.info(f"Skipping pose {idx}: {action}")
            self.progress.emit(idx)
        # self.robot.set_teach_mode(True)

class RobotJointOpThread(QThread):
    progress = Signal(int)  # Signal to communicate progress updates
    action_start = Signal()  # Signal to start recording
    action_finished = Signal()  # Signal to communicate that the action has finished
    def __init__(self, robot: RobotInterface, joint_positions: List[np.ndarray], wait=True):
        self.robot = robot
        self.robot.update_motion_parameters(
            acceleration=10, velocity=10, time_running=0, radius=0
        )
        self.joint_positions = joint_positions
        self._wait = wait
        super().__init__()

    def run(self):
        self.robot.set_teach_mode(False)
        self.start_to_move = False
        for idx, joint_position in enumerate(self.joint_positions):
            print(f"Progress: {idx}")
            logger.info(f"Moving to pose {idx}")
            try:
                self.robot.step(joint_position, action_type='joint', wait=self._wait)
            except Exception as e:
                logger.error(f"Failed to move to pose {idx}: {e}")
                logger.info(f"Skipping pose {idx}: {joint_position}")

            if idx == 0:
                self.action_start.emit()
                while not self.robot.recording_flag:
                    time.sleep(0.1)
            self.progress.emit(idx)
        time.sleep(0.5)
        self.action_finished.emit()
        self.robot.set_teach_mode(True)

class DataSendToServerThread(QThread):
    progress = Signal(tuple)  # Signal to communicate progress updates (step, progress)
    def __init__(self, msg_dict: dict, server:websocket_client_policy.WebsocketClientPolicy):
        self.server = server
        self.msg_dict = msg_dict
        super().__init__()

    def run(self):
        try:
            response = self.server.infer(self.msg_dict)
            self.__response = {"status": "action", "message": response}
            self.progress.emit(("Success", 100))
        except Exception as e:
            self.__response = {"status": "error", "message": str(e)}
            print(f"An error occurred: {e}")
            self.progress.emit(("Error", 0))  # Emit an error state if something goes wrong
    
    def get_response(self):
        return self.__response
