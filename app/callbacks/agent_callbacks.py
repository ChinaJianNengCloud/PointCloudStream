import logging
import sys

from typing import TYPE_CHECKING
from app.utils.networking import send_message, discover_server

if TYPE_CHECKING:
    from app.main_app import PCDStreamer

from app.utils.logger import setup_logger
logger = setup_logger(__name__)

def on_scan_button_clicked(self: 'PCDStreamer'):
    try:
        ip, port = discover_server(self.params)
        self.ip_editor.setText(ip)
        self.port_editor.setText(str(port))
        self.scan_button.setStyleSheet("background-color: green;")
    except Exception as e:
        self.scan_button.setStyleSheet("background-color: red;")
        logger.error(f"Failed to discover server: {e}")
    logger.debug("Scan button clicked")


def on_send_button_clicked(self: 'PCDStreamer'):
    
    text = self.agent_prompt_editor.toPlainText().strip()
    if text:
        self.chat_history.add_message(text, is_user=True)
        
    #     self.chat_history.add_message("This is a response.", is_user=False)
    #     self.conversation_widget.add_message("User", text, is_user=True)
    #     self.agent_prompt_editor.clear()
    #     # Simulate a response
    #     QTimer.singleShot(1000, lambda: self.conversation_widget.add_message("Agent", "This is a response.", is_user=False))

    # try:
    #     ret, robot_pose, _ = self.get_robot_pose()
    #     if ret:
    #         prompt = self.agent_prompt_editor.toPlainText()
    #         frame = copy.deepcopy(self.current_frame)
    #         colors = np.asarray(frame['pcd'].colors)
    #         points = np.asarray(frame['pcd'].points)
    #         labels = np.asarray(frame['seg_labels'])
    #         pcd_with_labels = np.hstack((points, colors, labels.reshape(-1, 1)))
    #         image = np.asarray(frame['color'].cpu())
    #         # pose = frame['robot_pose']
    #         past_actions = []
    #         msg_dict = {'prompt': prompt, 
    #                     'pcd': pcd_with_labels, 
    #                     'image': image, 
    #                     'pose': robot_pose, 
    #                     'past_actions': past_actions, 
    #                     'command': "process_pcd"}
            
    #         self.sendingThread = DataSendToServerThread(ip=self.ip_editor.text(), 
    #                                                 port=int(self.port_editor.text()), 
    #                                                 msg_dict=msg_dict)
    #         self.conversation_data.append('User', prompt)
    #         self.sendingThread.progress.connect(lambda progress: on_send_progress(progress))
    #         self.sendingThread.finished.connect(lambda: on_finish_sending_thread(self))
    #         self.send_button.setEnabled(False)
    #         self.sendingThread.start()
    #         logger.debug("Send button clicked")
    #     else:
    #         logger.error("Failed to get robot pose")
    # except Exception as e:
    #     logger.error(f"Failed to send data: {e}")


def on_finish_sending_thread(self: "PCDStreamer"):
    self.send_button.setEnabled(True)
    response = self.sendingThread.get_response()
    if response['status'] == 'action':
        self.conversation_data.append('Agent', str(response['message']))
    elif response['status'] == 'no_action':
        self.conversation_data.append('Agent', str(response['message']))
    self.conversation_editor.setText(self.conversation_data.get_qt_format_conversation())
    logger.info(self.conversation_data.get_terminal_conversation())
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