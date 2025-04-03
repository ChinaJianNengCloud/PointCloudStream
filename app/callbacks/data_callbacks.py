import logging
import numpy as np
import time
from functools import partial
from typing import TYPE_CHECKING
from PySide6.QtWidgets import QLabel, QDialog, QVBoxLayout, QFileDialog
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from threading import Thread
if TYPE_CHECKING:
    from app.entry import PCDStreamer
from app.viewers.image_viewer import ImageConfirmationDialog
from app.threads.op_thread import RobotJointOpThread
from app.utils.pose import interpolate_joint_positions_equal_distance
from app.utils.logger import setup_logger
logger = setup_logger(__name__)

def on_data_replay_and_save_button_clicked(self: "PCDStreamer"):
    select_item = self.data_tree_view.selected_item
    data_index = self.collected_data.dataids.index(select_item.root_text)
    self.collected_data.reset_record(data_index)
    if select_item.item is not None:
        joint_posistions = self.collected_data.shown_data_json.get(
                            select_item.root_text
                            ).get('joint_positions')
        try:
            interpolate_positions = interpolate_joint_positions_equal_distance(joint_posistions,
                                                                            target_length=7, method='quadratic')
        except ValueError as e:
            logger.error(f"Failed to interpolate joint positions: {e}")
            return False
        self.robot_joint_thread = RobotJointOpThread(self.robot, 
                                                     interpolate_positions,
                                                     wait=True)
        

        fps = 15  # desired frequency in FPS
        data_collection_thread = Thread(target=collect_data_at_fps, args=(self, fps,))
        self.robot_joint_thread.action_start.connect(
            lambda: on_record_start(self, data_collection_thread))
        self.robot_joint_thread.action_finished.connect(
            lambda: on_record_finished(self, data_collection_thread))
        self.robot_joint_thread.start()

    logger.debug("Data replay and save button clicked")

def on_record_start(self:"PCDStreamer", thread: Thread):

    notice = "Please confirm the Robot Arm is in the correct position before recording."
    # timer_was_active = False
    # if hasattr(self, 'timer') and self.timer.isActive():
    #     timer_was_active = True
    #     self.timer.stop()
        
    dialog = ImageConfirmationDialog(None, notice)
    result = dialog.exec()
    # if timer_was_active:
    #     self.timer.start()
    if result == QDialog.DialogCode.Accepted:
        self.robot.high_speed_mode()
        self.robot.recording_flag = True
        thread.start()
    else:
        self.robot.recording_flag = False
        self.robot.low_speed_mode()
        thread.join()


def on_record_finished(self:"PCDStreamer", thread: Thread):
    self.robot.recording_flag = False
    self.robot.low_speed_mode()
    thread.join()
    
    notice = "Data collection finished. Please Check the robot arm if it's end at correct position."
    image_file = self.collected_data.resource_path + '/' + self.collected_data.saved_data_json.get(
                            self.data_tree_view.selected_item.root_text
                            ).get('main_cam_files')[-1]
    dialog = ImageConfirmationDialog(image_file, notice)
    result = dialog.exec()
    if result == QDialog.DialogCode.Accepted:
        logger.info("Data collection confirmed")
    else:
        logger.info("Data collection not confirmed")

def on_progress_update(self: "PCDStreamer", thread:Thread, progress):
    if progress == 0:
        self.robot.high_speed_mode()
        self.robot.recording_flag = True
        thread.start()
    if progress == len(self.robot_joint_thread.joint_positions) - 1:
        self.robot.recording_flag = False
        self.robot.low_speed_mode()
        thread.join()

def collect_data_at_fps(self: "PCDStreamer", fps):
    interval = 1 / fps
    idx = 0
    logger.info("Data collection started")
    while self.robot.recording_flag:
        robot_pose = self.robot.get_state('tcp')
        joint_position = self.robot.get_joint_position()
        main = self._img_to_array(self.current_frame['main'])
        sub = self._img_to_array(self.current_frame['sub'])

        self.collected_data.add_record(
            prompt=self.prompt_text.text(),
            main=main,
            sub=sub,
            base_poses=robot_pose,
            joint_position=joint_position,
            record_stage=True
        )
        idx += 1
        time.sleep(interval)
        if idx > 300:
            logger.error("Failed to collect data, idx is greater than 300, please check!")
            break

def on_data_collect_button_clicked(self: "PCDStreamer"):
    if self.robot is not None:
        robot_pose = self.robot.get_state('tcp')
        joint_posistion = self.robot.get_joint_position()
        main = self._img_to_array(self.current_frame['main'])
        sub = self._img_to_array(self.current_frame['sub'])

        self.collected_data.add_record(prompt=self.prompt_text.text(),
                                main=main,
                                sub=sub,
                                joint_position=joint_posistion,
                                base_poses=robot_pose,
                                record_stage=False)
        
        logger.debug("Data collected")
    else:
        logger.error("Failed to get robot pose")

    logger.debug("Data collect button clicked")

def on_data_save_button_clicked(self: "PCDStreamer"):
    self.collected_data.save(self.data_folder_text.text())
    logger.debug("Data save button clicked")

def on_data_tree_view_load_button_clicked(self: "PCDStreamer"):
    try:
        self.collected_data.load(self.data_folder_text.text())
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
    logger.debug("Data tree view load button clicked")

def on_data_tree_view_remove_button_clicked(self: "PCDStreamer"):
    select_item = self.data_tree_view.selected_item
    if select_item != None:
        match select_item.level:
            case 1:
                logger.debug(f"Removing {select_item.root_text}")
                self.collected_data.pop(self.collected_data.dataids.index(select_item.root_text))
            case 2:
                pass
            case 3:
                logger.debug(f"Removing pose {select_item.root_text}-{select_item.index_in_level}")
                if select_item.parent_text == "Pose" or select_item.parent_text == "Joint Position":
                    self.collected_data.pop_pose(self.collected_data.dataids.index(select_item.root_text), 
                                                    select_item.index_in_level)
                    pass
    logger.debug("Data tree view remove button clicked")

def on_tree_selection_changed(self: "PCDStreamer", item, level, index_in_level, parent_text, root_text):
    """
    Callback for when the selection changes.
    """
    print("Clicked")
    logger.debug(f"Selected Item: {item.text(0)}, Level: {level}, Index in Level: {index_in_level}, Parent Text: {parent_text}, Root Text: {root_text}")
    select_item = self.data_tree_view.selected_item
    self.prompt_text.setText(self.collected_data.shown_data_json.get(
                            select_item.root_text
                            ).get('prompt'))
    if select_item.level == 3 and select_item.parent_text == "Pose":
        pass # No image to show now
        # show_image_popup(self, self.collected_data.resource_path + '/' + self.collected_data.saved_data_json.get(
        #                     select_item.root_text
        #                     ).get('color_files')[index_in_level])


def on_data_folder_select_button_clicked(self: "PCDStreamer"):
    
    start_dir = self.params.get('data_path', './data')
    
    # Temporarily pause the timer if it's running
    timer_was_active = False
    if hasattr(self, 'timer') and self.timer.isActive():
        timer_was_active = True
        self.timer.stop()
    
    # Show the dialog
    dir_text = QFileDialog.getExistingDirectory(
        self,
        "Select Folder",
        start_dir,
        QFileDialog.Option.ShowDirsOnly
    )
    
    # Resume the timer if it was active before
    if timer_was_active:
        self.timer.start()
    
    if dir_text:
        self.data_folder_text.setText(dir_text)
    logger.debug("Data folder select button clicked")

def on_data_tree_changed(self: "PCDStreamer"): 
    """
    Updates the tree view with data from `shown_data_json`.
    """

    self.data_tree_view.clear()

    for key, value in self.collected_data.shown_data_json.items():
        root_id = self.data_tree_view.add_item(parent_item=None, text=key, level=1)
        prompt_id = self.data_tree_view.add_item(parent_item=root_id, text="Prompt", level=2, root_text=key)
        self.data_tree_view.add_item(parent_item=prompt_id, text=value["prompt"], level=3, root_text=key)
        record_len_id = self.data_tree_view.add_item(parent_item=root_id, text="record_len", level=2, root_text=key)
        record_len_text = str(value['record_len'])
        self.data_tree_view.add_item(parent_item=record_len_id, text=record_len_text, level=3, root_text=key)
        pose_id = self.data_tree_view.add_item(parent_item=root_id, text="Joint Position", level=2, root_text=key)
        
        for i, pose in enumerate(value["joint_positions"]):

            pose_text = f"{i + 1}: [{','.join(f'{v:.2f}' for v in np.rad2deg(pose))}]"
            self.data_tree_view.add_item(
                parent_item=pose_id,
                text=pose_text,
                level=3,
                root_text=key
            )
    self.data_tree_view.expandAll()
