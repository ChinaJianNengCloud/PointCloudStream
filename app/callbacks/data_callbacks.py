import logging
import numpy as np
from typing import TYPE_CHECKING
from PyQt5.QtWidgets import QLabel,QDialog, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import  Qt

if TYPE_CHECKING:
    from app.main_app import PCDStreamer

from app.utils.logger import setup_logger
logger = setup_logger(__name__)

def on_data_collect_button_clicked(self: "PCDStreamer"):
    ret, robot_pose, _ = self.get_robot_pose()
    if ret:
        color = self.img_to_array(self.current_frame['color'])
        depth = self.img_to_array(self.current_frame['depth'])
        pcd =  self.current_frame['pcd']
        
        xyz = np.asarray(pcd.points)
        rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        seg = self.current_frame.get('seg', np.zeros(xyz.shape[0]))
        pcd_with_labels = np.hstack((xyz, rgb, seg.reshape(-1, 1)))

        self.collected_data.append(prompt=self.prompt_text.text(),
                                color=color,
                                depth=depth,
                                point_cloud=pcd_with_labels,
                                pose=robot_pose,
                                bbox_dict=self.bbox_params)
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
                logger.debug(f"Removing pose{select_item.root_text}-{select_item.index_in_level}")
                if select_item.parent_text == "Pose":
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
        show_image_popup(self, self.collected_data.resource_path + '/' + self.collected_data.saved_data_json.get(
                            select_item.root_text
                            ).get('color_files')[index_in_level])

def show_image_popup(self: "PCDStreamer", image_path):
    """
    Show a pop-up window with an image.
    """
    # Create a QDialog for the image
    self.image_dialog = QDialog()  # Store dialog as an instance variable
    self.image_dialog.setWindowTitle("Selected Item Image")
    self.image_dialog.setWindowFlags(self.image_dialog.windowFlags() | Qt.Window)  # Make it a standalone, draggable window

    # Create a layout and QLabel to display the image
    layout = QVBoxLayout()
    image_label = QLabel(self.image_dialog)
    image_label.setFixedSize(400, 300)  # Set the label size
    image_label.setStyleSheet("border: 1px solid black;")  # Optional: Add a border for clarity

    pixmap = QPixmap(image_path)
    if not pixmap.isNull():
        scaled_pixmap = pixmap.scaled(image_label.size(), 
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)

    # Add QLabel to the layout and set the layout to the dialog
    layout.addWidget(image_label)
    self.image_dialog.setLayout(layout)

    # Set a fixed size for the dialog
    self.image_dialog.resize(420, 320)  # Slightly larger to account for padding

    # Show the dialog as a non-modal window
    self.image_dialog.show()

def on_data_folder_select_button_clicked(self: "PCDStreamer"):
    
    start_dir = self.params.get('data_path', './data')
    dir_text = QFileDialog.getExistingDirectory(
        directory=start_dir,
        options=QFileDialog.Option.ShowDirsOnly
    )
    if not dir_text == "":
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
        bbox_id = self.data_tree_view.add_item(parent_item=root_id, text="Bbox", level=2, root_text=key)
        bbox_text = f"[{','.join(f'{v:.2f}' for v in value['bboxes'])}]"
        self.data_tree_view.add_item(parent_item=bbox_id, text=bbox_text, level=3, root_text=key)
        pose_id = self.data_tree_view.add_item(parent_item=root_id, text="Pose", level=2, root_text=key)
        
        for i, pose in enumerate(value["pose"]):
            pose_text = f"{i + 1}: [{','.join(f'{v:.2f}' for v in pose)}]"
            self.data_tree_view.add_item(
                parent_item=pose_id,
                text=pose_text,
                level=3,
                root_text=key
            )
    self.data_tree_view.expandAll()
