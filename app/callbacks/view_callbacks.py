import logging
import numpy as np
from ultralytics import YOLO, SAM
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.entry import PCDStreamer

from app.utils.logger import setup_logger
logger = setup_logger(__name__)

def on_capture_toggle_state_changed(self: "PCDStreamer"):
    logger.debug("Capture state changed")

def on_capture_toggle_state_changed(self: "PCDStreamer"):
    logger.debug("Capture state changed")

def on_seg_model_init_toggle_state_changed(self: "PCDStreamer"):
    if self.pcd_seg_model is None:
        self.pcd_seg_model = YOLO(self.params['yolo_model_path'])
    logger.debug(f"Segmentation model state changed:{self.seg_model_init_toggle.isChecked()}")

def on_acq_mode_toggle_state_changed(self: "PCDStreamer"):
    logger.debug("Acquisition mode state changed")

def on_display_mode_combobox_changed(self: "PCDStreamer", text):
    self.display_mode = text
    logger.debug(f"Display mode changed to: {text}")

def on_center_to_robot_base_toggle_state_changed(self: "PCDStreamer"):
    logger.debug("Center to robot base state changed")
    if self.center_to_robot_base_toggle.isChecked():
        self.streamer.extrinsics = self.T_BaseToCam.matrix
    else:
        self.streamer.extrinsics = np.eye(4)