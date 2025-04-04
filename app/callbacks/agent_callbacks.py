import logging
import sys
import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING
from app.utils.networking import send_message, discover_server
from PIL import Image
if TYPE_CHECKING:
    from app.entry import SceneStreamer


from app.threads.op_thread import DataSendToServerThread, RobotTcpOpThread

logger = logging.getLogger(__name__)

def on_scan_button_clicked(self: 'SceneStreamer'):
    try:
        ip, port = discover_server(self.params)
        self.ip_editor.setText(ip)
        self.port_editor.setText(str(port))
        self.scan_button.setStyleSheet("background-color: green;")
    except Exception as e:
        self.scan_button.setStyleSheet("background-color: red;")
        logger.error(f"Failed to discover server: {e}")
    logger.debug("Scan button clicked")

def on_reset_button_clicked(self: 'SceneStreamer'):
    msg_dict = {
        'command': "reset"
    }
    self.sendingThread = DataSendToServerThread(ip=self.ip_editor.text(), 
                                                    port=int(self.port_editor.text()), 
                                                    msg_dict=msg_dict)
    self.sendingThread.start()
    

def on_send_button_clicked(self: 'SceneStreamer'):
    
    text = self.agent_prompt_editor.toPlainText().strip()
    intruction ='''The robot should only interact with the breast closest to its arm. Directions are defined relative to the human body:
                    Upper/Bottom: The top and bottom of the breast.
                    Inner/Outer: Inner is closer to the body's center, outer is farther away.
                    Current Instruction:'''
    if not hasattr(self, 'robot'):
        if not hasattr(self, 'current_frame'):
            logger.error("No current frame, please start streaming first.")
            return
        frame = deepcopy(self.current_frame)
        image = np.asarray(frame['color'].cpu())
    else:
        image = Image.open("/home/capre/disk_4/yutao/data/resources/00af133d3fe24ee1a6b16a8859b08c49_color_4.png")
        image = np.asarray(image)

    msg_dict = {
        'prompt': intruction + text,
        'image': image,
        'command': "process_pcd"
        }
    self.sendingThread = DataSendToServerThread(ip=self.ip_editor.text(), 
                                                    port=int(self.port_editor.text()), 
                                                    msg_dict=msg_dict)

    self.sendingThread.progress.connect(lambda progress: on_send_progress(progress))
    self.sendingThread.finished.connect(lambda: on_finish_sending_thread(self))
    self.send_button.setEnabled(False)
    self.sendingThread.start()
    
    if text:
        self.chat_history.add_message(text, is_user=True)


def on_finish_sending_thread(self: "SceneStreamer"):
    self.send_button.setEnabled(True)
    
    response = self.sendingThread.get_response()
    if response['status'] == 'action':
        self.chat_history.add_message(f"{response['message']}", is_user=False)
        real_pose = self.view_predicted_poses(response['message'])
        thread = RobotTcpOpThread(self.robot, real_pose)
        thread.start()

    elif response['status'] == 'no_action':
        self.chat_history.add_message("Something went wrong", is_user=False)
    elif response['status'] == 'error':
        self.chat_history.add_message("Something went wrong", is_user=False)

    logger.info(f"Received response: {response}")
    logger.debug("Sending thread finished")



def refresh_line_progress(progress):
    """
    Refresh the terminal line with the current progress.
    :param progress: Tuple of (status, percentage)
    """
    status, percentage = progress
    bar_length = 40  # Length of the progress bar
    filled_length = int(bar_length * percentage / 100)
    bar = "#" * filled_length + "-" * (bar_length - filled_length)
    if percentage < 100:
        sys.stdout.write(f"\r[{bar}] {percentage:.2f}% - {status}  ")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r[{bar}] {percentage:.2f}% - {status}  \n")
        sys.stdout.flush()

def on_send_progress(progress):
    # logger.debug(f"Send progress: {progress}")  # Log progress for debugging
    refresh_line_progress(progress)  # Update the line progress bar