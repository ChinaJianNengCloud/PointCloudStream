from PySide6.QtCore import QTimer
from typing import TYPE_CHECKING
from app.utils.robot.board_sync import BoardRobotSyncManager
from app.utils import RobotInterface
import logging 

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.entry import SceneStreamer

def on_stream_init_button_clicked(self: "SceneStreamer"):
    if self.streaming:
        # Stop streaming
        self.main_init_button.setText("Initialize Cameras")
        if hasattr(self, 'timer'):
            self.timer.stop()

        self.streamer.disconnect()
        scene_viewer = self.viewer
        if scene_viewer:
            scene_viewer.clear_all_cameras()
        self.status_message.setText("System: Stream Stopped")
        self.streaming = False
    else:
        # Start streaming
        self.streaming = True
        self.main_init_button.setText("Stop")
        
        # Update camera list in params before initialization
        self.update_cam_ids()
        
        # Set up camera views in the grid
        scene_viewer = self.viewer
        if scene_viewer:
            scene_viewer.clear_all_cameras()
            
            # Add each camera from the list to the grid
            camera_list = self.params.get('camera_list', [])
            for camera in camera_list:
                camera_name = camera.get('name', 'unnamed')
                camera_widget = scene_viewer.add_camera_to_grid(camera_name)
            
            # Reorganize the grid
            scene_viewer.reorganize_grid()
        
        # Initialize cameras
        connected = self.streamer.camera_mode_init()
        if connected:
            self.status_message.setText("System: Streaming from Cameras")
            # self.timer = QTimer(self)
            self.timer.timeout.connect(self.frame_calling)
            self.timer.start(30)  # Update at ~30 FPS
        else:
            # Ensure all cameras are properly released when initialization fails
            self.streamer.disconnect()
            self.streaming = False
            self.main_init_button.setText("Initialize Cameras")
            self.status_message.setText("System: Failed to connect to Cameras")
    
    self.set_enable_after_stream_init()


def on_robot_init_button_clicked(self: "SceneStreamer"):
    self.robot =  RobotInterface(sim=False)
    try:
        self.robot.find_device()
        self.robot.connect()
        ip = self.robot.ip_address
        msg = f'Robot: Connected [{ip}]'
        self.board_sync_manager = BoardRobotSyncManager(self.robot, self)
        self.robot_init_button.setStyleSheet("background-color: green;")
        self.flag_robot_init = True
    except Exception as e:
        msg = 'Robot: Connection failed'
        del self.robot
        logger.error(msg+f' [{e}]')
        self.robot_init_button.setStyleSheet("background-color: red;")
        self.flag_robot_init = False


def on_teach_mode_button_clicked(self: "SceneStreamer"):
    if hasattr(self, 'robot') and self.flag_robot_init:
        logger.info("Setting robot to teach mode")
        self.robot.set_teach_mode(True)

def on_end_teach_mode_button_clicked(self: "SceneStreamer"):
    if hasattr(self, 'robot') and self.flag_robot_init:
        logger.info("Setting robot to end teach mode")
        self.robot.set_teach_mode(False)

def on_sync_board_button_clicked(self: "SceneStreamer"):
    if hasattr(self, 'robot') and self.flag_robot_init:
        logger.info("Syncing board")
        self.board_sync_manager.sync(self.board_and_robot_pose_list[0], self.board_and_robot_pose_list[1])
