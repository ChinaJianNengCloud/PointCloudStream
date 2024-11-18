import json, uuid
from typing import Dict, List, AnyStr
import numpy as np


class CollectedData():
    def __init__(self):
        # self.prompts:List[AnyStr] = []
        self.dataids: List[AnyStr] = []
        self.data_list: List[Dict[AnyStr, Dict[AnyStr, List| AnyStr]]] = None
        self.bboxes: Dict[AnyStr, float] = {}

    def __len__(self):
        return len(self.data_list)
    
    @property
    def prompts(self):
        return [data['prompt'] for data in self.data_list]
    
    @property
    def data_json (self):
        data = {key:value for key, value in zip(self.dataids, self.data_list)}
        # data = {key:value for key, value in self.data_list}
        return data
    
    def pop(self, idx):
        self.dataids.pop(idx)
        self.data_list.pop(idx)

    def pop_pose(self, prompt_idx, pose_idx = -1):
        self.data_list[prompt_idx]["pose"].pop(pose_idx)

    def append(self, prompt, pose, bbox_dict: Dict[AnyStr, float] = None):
        if pose is np.ndarray:
            pose = pose.tolist()
        if self.data_list == None:
            self.dataids.append(uuid.uuid4().hex)
            self.data_list = [{"prompt":prompt, "pose":[pose], 'bboxes':self.box_from_dict(bbox_dict)}]
            return True
        if prompt not in self.prompts:
            self.dataids.append(uuid.uuid4().hex)
            self.data_list.append({"prompt":prompt, "pose":[pose], 'bboxes':self.box_from_dict(bbox_dict)})
        else:
            self.data_list[self.prompts.index(prompt)]["pose"].append(pose)
        return True


    def append_pose(self, prompt_idx, pose):
        self.data_list[prompt_idx].append(pose)

    def save(self, path):
        # print(self.data_json)
        js = {
            'data': self.data_json
        }
        json.dump(js, open(path, "w"),indent=2)

    def box_from_dict(self, bbox_dict: Dict[AnyStr, float]):
        key_set = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        bbox = [bbox_dict[key] for key in key_set]
        return bbox

    def bbox_to_dict(self):
        key_set = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
        return {key: self.bboxes[idx] for idx, key in enumerate(key_set)}

if __name__ == "__main__":
    data_collection = CollectedData()
    # print(data_collection.data_str_list)
    data_collection.append("prompta dd sda sd", [1,2,3])
    data_collection.append("prompta dd sda sds", [4,5,6])
    data_collection.append("prompt1 ass dsa s", [2,5,6])

    data_collection.append("prompt1 ass dsa s", [2,5,6])
    data_collection.append("prompt1 ass dsa s", [2,2,6])
    data_collection.append("prompt1 ass dsa s", [2,5,1])

    data_collection.append("prompt dsa s", [2,5,6])
    data_collection.append("prompt1", [2,5,6])
    print(data_collection.data_json)
    data_collection.save("llm_data.json")
    data_collection.pop(0)
    print(data_collection.data_json)
    # data_collection.save("llm_data.json")
    # print(data_collection.robot_poses)
    # print(data_collection.prompts)