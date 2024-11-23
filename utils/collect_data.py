import json
import uuid
from typing import Dict, List, AnyStr
import numpy as np
import os
from PIL import Image
from plyfile import PlyData, PlyElement
import logging
import threading
import cv2


logger = logging.getLogger(__name__)


class CollectedData:
    def __init__(self, path='data'):
        self.dataids: List[AnyStr] = []
        self.resource_path = os.path.join(path, "resources")
        self.data_list: List[Dict[AnyStr, Dict[AnyStr, List | AnyStr]]] = []
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
        return len(self.data_list)

    @property
    def img_list(self):
        return [id + ".png" for id in self.dataids]

    @property
    def prompts(self):
        return [data['prompt'] for data in self.data_list]

    @property
    def shown_data_json(self):
        return {key: {k: v for k, v in value.items() if k not in ('color_files', 'depth_files', 'point_cloud_files')}
                for key, value in zip(reversed(self.dataids), reversed(self.data_list))}

    @property
    def saved_data_json(self):
        data = {}
        for key, value in zip(self.dataids, self.data_list):
            entry = {k: v for k, v in value.items() if k not in ('color_files', 'depth_files', 'point_cloud_files')}
            poses = value['pose']
            entry['poses'] = [{"pose0": poses[i], "pose1": poses[i + 1] if i + 1 < len(poses) else None} for i in
                              range(len(poses))]
            entry['color_files'] = value['color_files']
            entry['depth_files'] = value['depth_files']
            entry['point_cloud_files'] = value['point_cloud_files']
            data[key] = entry
        return data

    def pop(self, idx):
        data_entry = self.data_list.pop(idx)
        id = self.dataids.pop(idx)

        # Delete files
        for file_list in [data_entry.get('color_files', []), data_entry.get('depth_files', []),
                          data_entry.get('point_cloud_files', [])]:
            for file_name in file_list:
                file_path = os.path.join(self.resource_path, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)

    def pop_pose(self, prompt_idx, pose_idx=-1):
        data_entry = self.data_list[prompt_idx]
        id = self.dataids[prompt_idx]

        # Remove files
        color_file = data_entry['color_files'].pop(pose_idx)
        depth_file = data_entry['depth_files'].pop(pose_idx)
        point_cloud_file = data_entry['point_cloud_files'].pop(pose_idx)

        for file_name in [color_file, depth_file, point_cloud_file]:
            file_path = os.path.join(self.resource_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Remove pose
        data_entry['pose'].pop(pose_idx)

        # If no poses left, remove the data_entry entirely
        if not data_entry['pose']:
            self.data_list.pop(prompt_idx)
            self.dataids.pop(prompt_idx)

    def append(self, prompt, pose, bbox_dict: Dict[AnyStr, float] = None, color: np.ndarray = None,
               depth: np.ndarray = None, point_cloud: np.ndarray = None):
        id = uuid.uuid4().hex
        if isinstance(pose, np.ndarray):
            pose = pose.tolist()

        if prompt in self.prompts:
            data_entry_idx = self.prompts.index(prompt)
            data_entry = self.data_list[data_entry_idx]
            id = self.dataids[data_entry_idx]
            pose_idx = len(data_entry['pose'])
            data_entry['pose'].append(pose)
            data_entry['bboxes'] = self.box_from_dict(bbox_dict)
        else:
            self.dataids.append(id)
            data_entry = {"prompt": prompt,
                          "pose": [pose],
                          'bboxes': self.box_from_dict(bbox_dict),
                          'color_files': [],
                          'depth_files': [],
                          'point_cloud_files': []}
            self.data_list.append(data_entry)
            data_entry_idx = len(self.data_list) - 1
            pose_idx = 0

        # Generate file names
        color_file = f"{id}_color_{pose_idx}.png"
        depth_file = f"{id}_depth_{pose_idx}.npy"
        point_cloud_file = f"{id}_point_cloud_{pose_idx}.ply"

        # Update data_entry with file names before starting the thread
        data_entry['color_files'].append(color_file)
        data_entry['depth_files'].append(depth_file)
        data_entry['point_cloud_files'].append(point_cloud_file)

        # Save files in a separate thread
        def save_files(color, depth, point_cloud, color_file, depth_file, point_cloud_file):
            if color is not None:
                Image.fromarray(color).save(os.path.join(self.resource_path, color_file))
            if depth is not None:
                np.save(os.path.join(self.resource_path, depth_file), depth)
            if point_cloud is not None:
                vertex = np.array(
                    [(x, y, z, r, g, b, s) for x, y, z, r, g, b, s in point_cloud],
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                           ('segment_id', 'i4')]
                )
                ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
                ply.write(os.path.join(self.resource_path, point_cloud_file))

        # Start the thread
        thread = threading.Thread(target=save_files, args=(
            color, depth, point_cloud, color_file, depth_file, point_cloud_file))
        thread.start()

        return True

    def save(self, path=None):
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
        saved_data_path = os.path.join(self.path, "saved_data.json")

        if not os.path.exists(saved_data_path):
            raise FileNotFoundError("Required JSON files not found in the specified directory.")

        # with open(shown_data_path, "r") as f:
        #     shown_data = json.load(f)

        
        with open(saved_data_path, "r") as f:
            saved_data = json.load(f)

        # Populate dataids and data_list
        self.dataids = []
        self.data_list = []

        for data_id in saved_data.keys():
            entry = saved_data[data_id]
            # Check if all files exist
            files_exist = True
            for file_list in [entry['color_files'], entry['depth_files'], entry['point_cloud_files']]:
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
                    'bboxes': self.box_from_dict(entry["bboxes"]),
                    'color_files': entry['color_files'],
                    'depth_files': entry['depth_files'],
                    'point_cloud_files': entry['point_cloud_files'],
                }
                self.dataids.append(data_id)
                self.data_list.append(data_entry)
            else:
                # Files missing, skip this data_id
                logger.info(f"Data entry {data_id} skipped due to missing files.")
                continue

        print("Data loaded successfully.")

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

    def _image_display_loop(self):
        """
        Continuously display the image, updating whenever `current_image_path` is changed.
        """
        while not self.stop_display.is_set():
            with self.lock:
                image_path = self.current_image_path

            if image_path and os.path.exists(image_path):
                color_image = cv2.imread(image_path)
                if color_image is not None:
                    height, width = color_image.shape[:2]
                    resized_image = cv2.resize(color_image, (width // 2, height // 2))
                    cv2.imshow("Image Viewer", resized_image)
                else:
                    print(f"Failed to load image: {image_path}")
            else:
                # If no valid image path, show a blank window
                blank_image = np.zeros((500, 500, 3), dtype=np.uint8)
                cv2.imshow("Image Viewer", blank_image)

            if cv2.waitKey(500) == ord('q'):  # Refresh every 500ms, press 'q' to quit
                self.stop_display.set()
                break

        cv2.destroyAllWindows()

    def show_image(self, prompt_idx, pose_idx):
        """
        Update the displayed image to the one specified by `prompt_idx` and `pose_idx`.
        """
        try:
            data_entry = self.data_list[prompt_idx]
            color_file = data_entry['color_files'][pose_idx]
            color_file_path = os.path.join(self.resource_path, color_file)

            if not os.path.exists(color_file_path):
                print(f"File not found: {color_file_path}")
                return
            if not self.display_thread is None:
                with self.lock:
                    self.current_image_path = color_file_path

        except IndexError:
            print("Invalid prompt_idx or pose_idx.")

    def start_display_thread(self):
        """
        Start the image display thread if it's not already running.
        """
        if self.display_thread is None or not self.display_thread.is_alive():
            self.stop_display.clear()
            self.display_thread = threading.Thread(target=self._image_display_loop, daemon=True)
            self.display_thread.start()

    def stop_display_thread(self):
        """
        Stop the display thread and close the OpenCV window.
        """
        self.stop_display.set()
        if self.display_thread:
            self.display_thread.join()
        cv2.destroyAllWindows()


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
    # data_collection.stop_showing()
    # # Test data
    # test_color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # test_depth = np.random.rand(100, 100).astype(np.float32)
    # test_bbox = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'zmin': 0, 'zmax': 1}
    # test_point_cloud = np.random.rand(100, 7).astype(np.float32)
    # test_point_cloud[:, 3:6] = (test_point_cloud[:, 3:6] * 255).astype(np.uint8)

    # # Append new data
    # data_collection.append("Prompt A", [1, 2, 3], bbox_dict=test_bbox, color=test_color, depth=test_depth,
    #                        point_cloud=test_point_cloud)
    # data_collection.append("Prompt A", [13, 2, 3], bbox_dict=test_bbox, color=test_color, depth=test_depth,
    #                        point_cloud=test_point_cloud)
    # data_collection.append("Prompt A", [1, 2, 33], bbox_dict=test_bbox, color=test_color, depth=test_depth,
    #                        point_cloud=test_point_cloud)
    # data_collection.append("Prompt B", [1, 23, 3], bbox_dict=test_bbox, color=test_color, depth=test_depth,
    #                        point_cloud=test_point_cloud)
    # data_collection.append("Prompt V", [1, 2, 23], bbox_dict=test_bbox, color=test_color, depth=test_depth,
    #                        point_cloud=test_point_cloud)

    # # Save JSON data
    # data_collection.save()
    # print("Data saved.")

    # # Test popping data
    # data_collection.pop_pose(0, 1)  # Remove second pose of first prompt
    # data_collection.pop(1)          # Remove second prompt entirely

    # # Save changes
    # data_collection.save()
    # print("Data updated and saved.")
