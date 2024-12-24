import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from plyfile import PlyElement, PlyData
import open3d as o3d
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager
from functools import partial
from cluster_seg_utils import seg_numpy_pcd

# Initialize YOLO model
model = YOLO("/home/capre/Point-Cloud-Stream/runs/segment/train6/weights/best.pt")

# Paths and configuration
path = Path("/home/capre/disk_4/yutao/leo_data/merged_data")
resources_path = path / 'resources'
all_data = json.load(open(path / 'all_data.json'))

# limit = 10
# all_data = {key: all_data[key] for key in list(all_data.keys())[:limit]}

# Function to read and process a single PCD file
def process_pcd_file(record_key, idx, file_name, resources_path):
    try:
        # Read and process the point cloud
        pcd_path = resources_path / file_name
        pcd_array = read_ply_to_numpy(pcd_path)
        labeled_pcd_array = seg_numpy_pcd(pcd_array)
        return (record_key, idx, labeled_pcd_array)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

# Function to process a single record
# Updated process_record function to unpack arguments
def process_record(args, resources_path):
    record_key, record_data = args  # Unpack the tuple
    current_pcds = []
    current_info = []
    for idx, file_name in enumerate(record_data['point_cloud_files']):
        result = process_pcd_file(record_key, idx, file_name, resources_path)
        if result:
            record_key, idx, labeled_pcd_array = result
            current_pcds.append(labeled_pcd_array)
            current_info.append((record_key, idx))
        else:
            return None  # Skip the entire record if an error occurs
    return (current_pcds, current_info, record_key)

# Main multiprocessing wrapper
def process_all_data(all_data, resources_path):
    with Manager() as manager:
        batch_pcds = manager.list()
        batch_info = manager.list()
        res_data = manager.dict()

        # Use a Pool for parallel processing
        with Pool() as pool:
            process_func = partial(process_record, resources_path=resources_path)
            # No need to unpack tuple here; just pass as-is
            results = list(tqdm(pool.imap_unordered(process_func, [(key, all_data[key]) for key in all_data.keys()]), 
                                desc="Processing Records", total=len(all_data)))

        for result in results:
            if result:
                current_pcds, current_info, record_key = result
                batch_pcds.extend(current_pcds)
                batch_info.extend(current_info)
                res_data[record_key] = all_data[record_key]

        return list(batch_pcds), list(batch_info), dict(res_data)


# Read PLY to NumPy function
def read_ply_to_numpy(file_path):
    pcd = PlyData.read(open(file_path, 'rb'))
    pcd_array = np.array(pcd.elements[0].data)
    points_array = np.vstack([
        pcd_array['x'], pcd_array['y'], pcd_array['z'], 
        pcd_array['red'], pcd_array['green'], pcd_array['blue'], 
        pcd_array['segment_id']
    ]).T
    return points_array

def save_pcd_file(args):
    idx, pcd, batch_info, all_data, resources_path = args
    try:
        record, r_idx = batch_info[idx]
        ply_name = resources_path / all_data[record]['point_cloud_files'][r_idx]
        
        # Prepare vertex array for PLY format
        vertex = np.array(
            [(x_p, y_p, z_p, r_p, g_p, b_p, s_p) for x_p, y_p, z_p, r_p, g_p, b_p, s_p in pcd],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                   ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'), ('segment_id', 'i4')]
        )
        # Write to PLY
        ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
        ply.write(str(ply_name))
        del ply, vertex
        return True
    except Exception as e:
        print(f"Error saving file for index {idx}: {e}")
        return False

# Main saving process with multiprocessing
def save_all_pcd_files(batch_pcds, batch_info, all_data, resources_path:Path):
    resources_path.mkdir(parents=True, exist_ok=True)
    save_args = [(idx, pcd, batch_info, all_data, resources_path) 
                 for idx, pcd in enumerate(batch_pcds)]
    
    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(save_pcd_file, save_args),
                            desc="Saving PCD Files", total=len(batch_pcds)))

    success_count = sum(results)
    print(f"Successfully saved {success_count}/{len(batch_pcds)} point clouds.")


# Main processing
if __name__ == "__main__":
    batch_pcds, batch_info, res_data = process_all_data(all_data, resources_path)
    print(f"Processed {len(batch_pcds)} point clouds successfully.")
    
    # Save PCD files using multiprocessing
    save_all_pcd_files(batch_pcds, batch_info, all_data, resources_path.parent / 'ply_output')
    
    # Save the updated data dictionary
    json.dump(res_data, open(path / 'res_data.json', 'w'), indent=4)