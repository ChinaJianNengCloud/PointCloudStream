import json
import uuid
from typing import Dict, List, AnyStr, Union
import numpy as np
import os
from PIL import Image
from plyfile import PlyData, PlyElement
import logging
import threading
import cv2
from PySide6.QtCore import Signal, QObject

from app.utils.logger import setup_logger
logger = setup_logger(__name__)

class CollectedData(QObject):
    data_changed = Signal()
    def __init__(self, path='data'):
        super().__init__()
        self.dataids: List[AnyStr] = []
        self.resource_path = os.path.join(path, "resources")
        self.episode_list: List[Dict[AnyStr, Dict[AnyStr, List | AnyStr]]] = []
        self.bboxes: Dict[AnyStr, float] = {}
        self.__path = path
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.resource_path, exist_ok=True)
        self.display_thread = None
        self.stop_display = threading.Event()
        self.current_image_path = None
        self.lock = threading.Lock()

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        os.makedirs(path, exist_ok=True)
        self.__path = path

    def __len__(self):
        return len(self.episode_list)

    @property
    def img_list(self):
        return [id + ".png" for id in self.dataids]

    @property
    def prompts(self):
        return [data['prompt'] for data in self.episode_list]

    @property
    def shown_data_json(self):
        ext_key_for_show = ('main_cam_files', 
                            'sub_cam_files', 
                            'joint_position_records',
                            'pose_record',
                            )
        
        # Creating a dictionary with updated records and adding 'record_len'
        return {
            key: {**{k: v for k, v in value.items() if k not in ext_key_for_show},
                'record_len': len(value.get('joint_position_records', []))}
            for key, value in zip(reversed(self.dataids), reversed(self.episode_list))
        }

    @property
    def saved_data_json(self):
        data = {}
        for id, episode in zip(self.dataids, self.episode_list):
            entry = {k: v for k, v in episode.items() if k not in ('main_cam_files', 'sub_cam_files')}
            poses = episode['pose']
            entry['poses'] = [{"pose0": poses[i], "pose1": poses[i + 1] if i + 1 < len(poses) else None} for i in
                              range(len(poses))]
            entry['main_cam_files'] = episode['main_cam_files']
            entry['sub_cam_files'] = episode['sub_cam_files']
            data[id] = entry
        return data

    def pop(self, idx):
        data_entry = self.episode_list.pop(idx)
        id = self.dataids.pop(idx)
        for file_list in [data_entry.get('main_cam_files', []), data_entry.get('sub_cam_files', [])]:
            for file_name in file_list:
                file_path = os.path.join(self.resource_path, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
        self.data_changed.emit()

    def pop_pose(self, prompt_idx, pose_idx=-1):
        data_entry = self.episode_list[prompt_idx]
        id = self.dataids[prompt_idx]
        data_entry['pose'].pop(pose_idx)
        data_entry['joint_positions'].pop(pose_idx)
        if len(data_entry['pose']) == 0 or len(data_entry['joint_positions']) == 0:
            # Remove the entire entry if no poses are left
            self.episode_list.pop(prompt_idx)
            self.dataids.pop(prompt_idx)

        self.data_changed.emit()

    def reset_record(self, prompt_idx):
        self.episode_list[prompt_idx]['joint_position_records'] = []
        self.episode_list[prompt_idx]['pose_record'] = []
        self.episode_list[prompt_idx]['main_cam_files'] = []
        self.episode_list[prompt_idx]['sub_cam_files'] = []

        self.data_changed.emit()

    def add_record(self, prompt, base_poses, 
                   joint_position, 
                   main: np.ndarray = None,
                   sub: np.ndarray = None, 
                   record_stage=False):
        id = uuid.uuid4().hex
        if isinstance(base_poses, np.ndarray):
            base_poses = base_poses.tolist()
        if isinstance(joint_position, np.ndarray):
            joint_position = joint_position.tolist()

        degree_joints = np.rad2deg(joint_position)
        if np.max(degree_joints) > 360 or np.min(degree_joints) < -360:
            out_of_range_indices = np.where((degree_joints > 360) | (degree_joints < -360))[0]
            logger.warning(f"Joint position is out of range at indices: {out_of_range_indices}")
            return False
        
        if prompt not in self.prompts:
            self.dataids.append(id)
            current_episode: Dict[str, Union[List, str]] = {"prompt": prompt,
                          'pose': [base_poses],
                          "pose_record": [],
                          'joint_positions': [joint_position],
                          'joint_position_records': [],
                          'main_cam_files': [],
                          'sub_cam_files': []
                          }
            self.episode_list.append(current_episode)
            data_entry_idx = len(self.episode_list) - 1
            pose_idx = 0
        else:
            data_entry_idx = self.prompts.index(prompt)
            current_episode = self.episode_list[data_entry_idx]
            id = self.dataids[data_entry_idx]

            if record_stage:
                pose_idx = len(current_episode['pose_record'])
            else:
                pose_idx = len(current_episode['pose'])

            if record_stage:
                current_episode['joint_position_records'].append(joint_position)
                current_episode['pose_record'].append(base_poses)
            else:
                current_episode['joint_positions'].append(joint_position)
                current_episode['pose'].append(base_poses)
                # current_episode['bboxes'] = self.box_from_dict(bbox_dict)

        # Generate file names
        main_cam = f"{id}_main_{pose_idx}.png"
        sub_cam = f"{id}_sub_{pose_idx}.png"


        def save_files(main, sub, main_file, sub_file):
            if main is not None:
                Image.fromarray(main).save(os.path.join(self.resource_path, main_file))
            if sub is not None:
                Image.fromarray(sub).save(os.path.join(self.resource_path, sub_file))
        
        if record_stage:
            current_episode['main_cam_files'].append(main_cam)
            current_episode['sub_cam_files'].append(sub_cam)
            thread = threading.Thread(target=save_files, args=(
                main, sub, main_cam, sub_cam))
            thread.start()
        self.data_changed.emit()
        return True

    def save(self, path=None):
        if path is not None:
            self.path = path
        with open(os.path.join(self.path, "shown_data.json"), "w") as f:
            json.dump(self.shown_data_json, f, indent=2)

        with open(os.path.join(self.path, "saved_data.json"), "w") as f:
            json.dump(self.saved_data_json, f, indent=2)

    def load(self, path=None):
        """
        Load saved data from the specified path and populate the CollectedData instance.
        """
        # Load shown_data.json and saved_data.json
        # shown_data_path = os.path.join(self.path, "shown_data.json")
        if path is not None:
            self.path = path
        saved_data_path = os.path.join(self.path, "saved_data.json")

        if not os.path.exists(saved_data_path):
            raise FileNotFoundError("Required JSON files not found in the specified directory.")

        # with open(shown_data_path, "r") as f:
        #     shown_data = json.load(f)

        
        with open(saved_data_path, "r") as f:
            saved_data = json.load(f)

        # Populate dataids and data_list
        self.dataids = []
        self.episode_list = []

        for data_id in saved_data.keys():
            entry = saved_data[data_id]
            # Check if all files exist
            files_exist = True
            for file_list in [entry['main_cam_files'], entry['sub_cam_files']]:
                for file_name in file_list:
                    file_path = os.path.join(self.resource_path, file_name)
                    if not os.path.exists(file_path):
                        files_exist = False
                        logger.warning(f"File {file_path} does not exist. Skipping data_id {data_id}.")
                        break
                if not files_exist:
                    break
            if files_exist:
                # All files exist, proceed to load data_entry
                data_entry = {
                    'prompt': entry['prompt'],
                    'pose': [pose['pose0'] for pose in entry['poses']],
                    # 'bboxes': self.box_from_dict(bbox_dict_or_list=entry["bboxes"]),
                    'main_cam_files': entry['main_cam_files'],
                    'sub_cam_files': entry['sub_cam_files'],
                    'joint_positions': entry.get('joint_positions', []),
                    'joint_position_records': entry.get('joint_position_records', []),
                    'pose_record': entry.get('pose_record', []),
                }
                self.dataids.append(data_id)
                self.episode_list.append(data_entry)
            else:
                # Files missing, skip this data_id
                logger.info(f"Data entry {data_id} skipped due to missing files.")
                continue
        self.data_changed.emit()
        logger.info("Data loaded successfully.")

    def box_from_dict(self, bbox_dict_or_list):
        key_set = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        if isinstance(bbox_dict_or_list, dict):
            # Convert dictionary to list
            return [bbox_dict_or_list[key] for key in key_set]
        elif isinstance(bbox_dict_or_list, list):
            # If it's already a list, return as is
            return bbox_dict_or_list
        else:
            raise TypeError("Expected a dictionary or list for bounding boxes.")

    def bbox_to_dict(self):
        key_set = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        return {key: self.bboxes[idx] for idx, key in enumerate(key_set)}


if __name__ == "__main__":
    # Test the updated class
    data_collection = CollectedData()
    import time
    # Load existing data
    try:
        data_collection.path = './data/data_pose_r_cam_u_lin'
        data_collection.load()
    except FileNotFoundError:
        print("No existing data found. Starting fresh.")

    data_collection.start_display_thread()

    data_collection.show_image(prompt_idx=0, pose_idx=0)
    time.sleep(1)
    data_collection.show_image(prompt_idx=0, pose_idx=1)
    time.sleep(1)
    data_collection.show_image(prompt_idx=0, pose_idx=2)
    data_collection.stop_display_thread()
