import time
import logging
import numpy as np
import socket
import pickle

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
        self.robot.set_teach_mode(True)

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
                self.robot._set_joint_position(joint_position, self._wait)
                
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
    def __init__(self, ip, port, msg_dict: dict):
        self.ip = ip
        self.port = port
        self.msg_dict = msg_dict
        super().__init__()

    def run(self):
        try:
            logger.debug("Sending data to server")
            # Serialize the message dictionary
            data = pickle.dumps(self.msg_dict)
            data_length = len(data)

            with socket.create_connection((self.ip, int(self.port))) as client_socket:
                start_time = time.time()
                client_socket.sendall(data_length.to_bytes(4, byteorder='big'))
                self.progress.emit(("Sending Length", 100))  # Mark step as complete

                bytes_sent = 0
                chunk_size = 4096 
                while bytes_sent < data_length:
                    chunk = data[bytes_sent:bytes_sent + chunk_size]
                    sent = client_socket.send(chunk)
                    bytes_sent += sent
                    progress = (bytes_sent / data_length) * 100
                    self.progress.emit(("Sending Data", progress))

                # Ensure sending step reaches 100%
                self.progress.emit(("Sending Data", 100))

                # Step 3: Receiving the response length
                response_length = int.from_bytes(client_socket.recv(4), byteorder='big')
                self.progress.emit(("Receiving Length", 100))  # Mark step as complete

                # Step 4: Receiving the response data
                response_data = b""
                bytes_received = 0
                while len(response_data) < response_length:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    response_data += packet
                    bytes_received = len(response_data)
                    progress = (bytes_received / response_length) * 100
                    self.progress.emit(("Receiving Data", progress))

                # Ensure receiving step reaches 100%
                self.progress.emit(("Receiving Data", 100))
                # Deserialize the response
                self.__response = pickle.loads(response_data)
                end_time = time.time() - start_time
                logger.debug(f"Response from server: {self.__response}")
                logger.debug(f"Total time: {end_time:.2f} s")
        except Exception as e:
            self.__response = {"status": "error", "message": str(e)}
            print(f"An error occurred: {e}")
            self.progress.emit(("Error", 0))  # Emit an error state if something goes wrong
    
    def get_response(self):
        return self.__response
