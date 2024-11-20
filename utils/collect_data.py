import json
import uuid
from typing import Dict, List, AnyStr
import numpy as np
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from plyfile import PlyData, PlyElement

class CollectedData:
    def __init__(self):
        self.dataids: List[AnyStr] = []
        self.data_list: List[Dict[AnyStr, Dict[AnyStr, List | AnyStr]]] = None
        self.bboxes: Dict[AnyStr, float] = {}
        self.save_exec = ThreadPoolExecutor(max_workers=3, thread_name_prefix='SaveImage')

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
        return {key: {k: v for k, v in value.items() if k not in ('color', 'depth', 'point_cloud')}
                for key, value in zip(self.dataids, self.data_list)}

    @property
    def saved_data_json(self):
        data = {}
        for key, value in zip(self.dataids, self.data_list):
            entry = {k: v for k, v in value.items() if k not in ('color', 'depth', 'point_cloud')}
            poses = value['pose']
            entry['poses'] = [{"pose0": poses[i], "pose1": poses[i + 1] if i + 1 < len(poses) else None} for i in range(len(poses))]
            entry['color_files'] = [f"{key}_color_{i}.png" for i in range(len(value['color']))]
            entry['depth_files'] = [f"{key}_depth_{i}.npy" for i in range(len(value['depth']))]
            entry['point_cloud_files'] = [f"{key}_point_cloud_{i}.ply" for i in range(len(value['point_cloud']))]
            data[key] = entry
        return data

    def pop(self, idx):
        self.dataids.pop(idx)
        self.data_list.pop(idx)

    def pop_pose(self, prompt_idx, pose_idx=-1):
        self.data_list[prompt_idx]["pose"].pop(pose_idx)
        self.data_list[prompt_idx]["color"].pop(pose_idx)
        self.data_list[prompt_idx]["depth"].pop(pose_idx)
        self.data_list[prompt_idx]["point_cloud"].pop(pose_idx)

    def append(self, prompt, pose, bbox_dict: Dict[AnyStr, float] = None, color: np.ndarray = None, depth: np.ndarray = None, point_cloud: np.ndarray = None):
        id = uuid.uuid4().hex
        if isinstance(pose, np.ndarray):
            pose = pose.tolist()
        if self.data_list is None:
            self.dataids.append(id)
            self.data_list = [{"prompt": prompt, "pose": [pose],
                               'bboxes': self.box_from_dict(bbox_dict),
                               'color': [color], 'depth': [depth], 'point_cloud': [point_cloud]}]
            return True
        if prompt not in self.prompts:
            self.dataids.append(id)
            self.data_list.append({"prompt": prompt,
                                   "pose": [pose],
                                   'bboxes': self.box_from_dict(bbox_dict),
                                   'color': [color], 'depth': [depth], 'point_cloud': [point_cloud]})
        else:
            idx = self.prompts.index(prompt)
            self.data_list[idx]["pose"].append(pose)
            self.data_list[idx]["color"].append(color)
            self.data_list[idx]["depth"].append(depth)
            self.data_list[idx]["point_cloud"].append(point_cloud)
        return True

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        def save_image_and_cloud(data_id, index, color, depth, point_cloud):
            color_path = os.path.join(path, f"{data_id}_color_{index}.png")
            depth_path = os.path.join(path, f"{data_id}_depth_{index}.npy")
            point_cloud_path = os.path.join(path, f"{data_id}_point_cloud_{index}.ply")

            if color is not None:
                Image.fromarray(color).save(color_path)
            if depth is not None:
                np.save(depth_path, depth)
            if point_cloud is not None:
                vertex = np.array(
                    [(x, y, z, r, g, b, s) for x, y, z, r, g, b, s in point_cloud],
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('segment_id', 'i4')]
                )
                ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
                ply.write(point_cloud_path)

        for idx, data_id in enumerate(self.dataids):
            data_entry = self.data_list[idx]
            for i, (color, depth, point_cloud) in enumerate(zip(data_entry['color'], data_entry['depth'], data_entry['point_cloud'])):
                self.save_exec.submit(save_image_and_cloud, data_id, i, color, depth, point_cloud)

        # Save JSON files synchronously
        with open(os.path.join(path, "shown_data.json"), "w") as f:
            json.dump(self.shown_data_json, f, indent=2)

        with open(os.path.join(path, "saved_data.json"), "w") as f:
            json.dump(self.saved_data_json, f, indent=2)

    def box_from_dict(self, bbox_dict: Dict[AnyStr, float]):
        key_set = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        return [bbox_dict[key] for key in key_set]

    def bbox_to_dict(self):
        key_set = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        return {key: self.bboxes[idx] for idx, key in enumerate(key_set)}


if __name__ == "__main__":
    # Test the updated class
    data_collection = CollectedData()
    test_color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_depth = np.random.rand(100, 100).astype(np.float32)
    test_bbox = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1, 'zmin': 0, 'zmax': 1}
    test_point_cloud = np.random.rand(100, 7).astype(np.float32)
    test_point_cloud[:, 3:6] = (test_point_cloud[:, 3:6] * 255).astype(np.uint8)

    data_collection.append("Prompt A", [1, 2, 3], bbox_dict=test_bbox, color=test_color, depth=test_depth, point_cloud=test_point_cloud)
    data_collection.append("Prompt A", [13, 2, 3], bbox_dict=test_bbox, color=test_color, depth=test_depth, point_cloud=test_point_cloud)
    data_collection.append("Prompt A", [1, 2, 33], bbox_dict=test_bbox, color=test_color, depth=test_depth, point_cloud=test_point_cloud)
    data_collection.append("Prompt B", [1, 23, 3], bbox_dict=test_bbox, color=test_color, depth=test_depth, point_cloud=test_point_cloud)
    data_collection.append("Prompt V", [1, 2, 23], bbox_dict=test_bbox, color=test_color, depth=test_depth, point_cloud=test_point_cloud)
    save_path = "test_output"
    data_collection.save(save_path)
    print("Data saved.")
