import numpy as np
from .pose import Pose
from .robot_utils import RobotInterface
from PySide6.QtCore import QThread, QObject, Signal, Slot
from typing import TYPE_CHECKING
import time
if TYPE_CHECKING:
    from app.entry import SceneStreamer
    

class StepWorker(QObject):
    finished = Signal(object)  # Emits the target robot pose when finished.
    error = Signal(Exception)  # Emits any exception encountered.

    def __init__(self, board_origin: "Pose", robot_origin: "Pose", 
                 current_board_pose: "Pose", robot: "RobotInterface", use_swap: bool = False):
        """
        :param board_origin: The Pose representing board sync origin.
        :param robot_origin: The Pose representing robot sync origin.
        :param current_board_pose: The current board pose.
        :param robot: A RobotInterface object.
        :param use_swap: Whether to swap the axes.
        """
        super().__init__()
        self.board_origin = board_origin
        self.robot_origin = robot_origin
        self.current_board_pose = current_board_pose
        self.robot = robot
        self.use_swap = use_swap

    def run(self):
        try:
            # Calculate the board delta pose.
            board_delta = self.board_origin.cal_delta_pose(self.current_board_pose, on="ee")
            print(f"Board delta pose: {board_delta}")
            if self.use_swap:
                vec = board_delta.to_1d_array(vector_type="rotvec")
                swapped_position = np.array([vec[2], vec[1], vec[0]])
                swapped_rot = np.array([vec[5], vec[4], vec[3]])
                swapped_vec = np.hstack((swapped_position, swapped_rot))
                robot_delta = board_delta.__class__.from_1d_array(swapped_vec, vector_type="rotvec")
                print(f"Robot delta (after swapping axes): {robot_delta}")
            else:
                robot_delta = board_delta
                print(f"Robot delta (without axis swap): {robot_delta}")
            target_robot_pose = self.robot_origin.apply_delta_pose(robot_delta, on="ee")
            print(f"Robot origin: {self.robot_origin}")
            print(f"Target robot pose: {target_robot_pose}")
            robot_cmd = target_robot_pose.to_1d_array("euler")
            self.robot.step(robot_cmd, action_type="tcp", wait=True)
            self.finished.emit(target_robot_pose)
        except Exception as e:
            self.error.emit(e)


class BoardRobotSyncManager(QObject):
    def __init__(self, robot: "RobotInterface", parent:"SceneStreamer"):
        """
        Initialize the manager with a robot instance.
        The robot instance must have a .step(6-tuple) method that accepts a 6-element vector [x, y, z, rx, ry, rz].
        """
        super().__init__()
        self.streamer = parent
        self.robot = robot
        self.board_origin = None  # Pose in board frame.
        self.robot_origin = None  # Pose in robot frame.
        self.is_moving = False

        # Store the board pose associated with the latest step command.
        self._last_board_pose = None

        # Keep references to thread and worker so that they don't get garbage collected.
        self._thread = None
        self._worker = None

    def sync(self, board_pose: "Pose", robot_pose: "Pose"):
        self.board_origin = board_pose
        self.robot_origin = robot_pose
        print(f"Sync complete:\n  Board origin: {self.board_origin}\n  Robot origin: {self.robot_origin}")

    def swap_axes(self, pose: "Pose") -> "Pose":
        vec = pose.to_1d_array(vector_type="rotvec")
        swapped_position = np.array([vec[2], vec[1], vec[0]])
        swapped_rot = np.array([vec[5], vec[4], vec[3]])
        swapped_vec = np.hstack((swapped_position, swapped_rot))
        return pose.__class__.from_1d_array(swapped_vec, vector_type="rotvec")

    def step_async(self, current_board_pose: "Pose", use_swap: bool = False):
        if self.is_moving:
            print("Step in progress, ignoring new step call.")
            return

        if self.board_origin is None or self.robot_origin is None:
            raise ValueError("Please call sync() first to initialize origins.")

        self._last_board_pose = current_board_pose
        self.is_moving = True
        self._thread = QThread(self)
        self._worker = StepWorker(self.board_origin, self.robot_origin,
                                  current_board_pose, self.robot, use_swap=use_swap)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.on_step_finished)
        self._worker.error.connect(self.on_step_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot(object)
    def on_step_finished(self, target_robot_pose: "Pose"):
        self.is_moving = False
        self.board_origin = self.streamer.board_and_robot_pose_list[0]
        self.robot_origin = self.streamer.board_and_robot_pose_list[1]
        print(f"Step finished: Target robot pose: {target_robot_pose}")
        # self.sync(self._last_board_pose, target_robot_pose)

    @Slot(Exception)
    def on_step_error(self, error: Exception):
        self.is_moving = False
        print(f"Step encountered an error: {error}")

# =========================
# Example usage with a dummy robot
# =========================
if __name__ == "__main__":
    # A dummy robot class for demonstration. It expects a 6-element vector command.
    class DummyRobot:
        def __init__(self):
            self.current_pose = np.zeros(6)  # [x, y, z, rx, ry, rz]
            
        def get_state(self, state_type='tcp'):
            return self.current_pose

        def step(self, pose_vector, action_type, wait=True):
            self.current_pose = np.array(pose_vector)
            print(f"Robot commanded to move to: {self.current_pose}")
    
    # Create an instance of the dummy robot.
    robot = DummyRobot()
    
    # Initialize the BoardRobotSyncManager with the dummy robot.
    sync_manager = BoardRobotSyncManager(robot)
    
    # Define initial poses (using 6-element vectors in rotvec representation).
    initial_board_vector = [1, 2, 3, 0.1, 0.2, 0.3]  # board: [x, y, z, rx, ry, rz]
    initial_robot_vector = [3, 2, 1, 0.3, 0.2, 0.1]  # robot: [x, y, z, rx, ry, rz]
    
    board_origin = Pose.from_1d_array(initial_board_vector, vector_type="rotvec")
    robot_origin = Pose.from_1d_array(initial_robot_vector, vector_type="rotvec")
    
    # Sync the board and robot origins.
    sync_manager.sync(board_origin, robot_origin)
    
    # Simulate an update: the board moves to a new pose.
    new_board_vector = [1.5, 2.0, 3.2, 0.15, 0.2, 0.35]
    current_board_pose = Pose.from_1d_array(new_board_vector, vector_type="rotvec")
    
    # Calculate and command the corresponding robot movement.
    target_robot_pose = sync_manager.step(current_board_pose)