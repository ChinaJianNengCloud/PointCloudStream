import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from plyfile import PlyElement, PlyData
import open3d as o3d
import open3d.core as o3c
import json
import torch
import cv2
import torch.utils.dlpack
from typing import List
from tqdm import tqdm
from app.utils.camera.segmentation_utils import (segment_pcd_from_2d, 
                                                 read_ply_to_numpy, 
                                                 map_point_cloud_colors_from_image,
                                                 batch_segment_and_label)
# Initialize YOLO model
model = YOLO("/home/capre/Point-Cloud-Stream/runs/segment/train6/weights/best.pt")

o3d_device = o3d.core.Device("CUDA:0")
path = Path("/home/capre/disk_4/yutao/leo_data/data_2nd")
resources_path = path / 'resources'
all_data = json.load(open(path / 'all_data.json'))
use_org_pcd = True
# Intrinsic matrix
intrinsic = np.array([
    [610.5961520662408, 0.0,               639.8919938587554],
    [0.0,               617.4130735412369, 358.3889735843055],
    [0.0,               0.0,               1.0]
], dtype=np.float32).T

def visualize_pcd(pcd):
    """
    Visualize a single point cloud using Open3D.

    Parameters:
        pcd_array (numpy.ndarray): Point cloud array of shape (N, 6) with columns representing
                                   x, y, z, r, g, b.
    """

    # Visualize the point cloud
    if isinstance(pcd, np.ndarray):
        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        
        # Set points and colors
        pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pcd_array[:, 3:6])

    o3d.visualization.draw_geometries([pcd],
                                    window_name="Point Cloud Visualization",
                                    width=800,
                                    height=600,
                                    left=50,
                                    top=50,
                                    point_show_normal=False)

batch_pcds = []
batch_rgbs = []

batch_intrinsics = []
batch_extrinsics = []
batch_info = []  # (each_record, idx)
res_data = {}

record_keys = list(all_data.keys())
for each_record in tqdm(record_keys, desc='Read PCD', total=len(record_keys)):
    current_pcds = []
    current_imgs = []
    current_intrs = []
    current_extrs = []
    current_info = []
    saved = True
    
    for idx, each_image in enumerate(all_data[each_record]['color_files']):
        try:
            color_path = resources_path / each_image
            full_color = cv2.imread(str(color_path))
            full_color = cv2.cvtColor(full_color, cv2.COLOR_BGR2RGB)
            if use_org_pcd:
                pcd_array = read_ply_to_numpy(resources_path / all_data[each_record]['point_cloud_files'][idx])
            else:
                depth_path = resources_path / all_data[each_record]['depth_files'][idx]
                depth = np.load(str(depth_path))

                # Convert depth and color images to Open3D format
                depth_o3d = o3d.geometry.Image(depth)
                color_o3d = o3d.geometry.Image(full_color)

                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, 
                    depth_scale=1000.0,  # Scale factor for depth values (adjust as per your data)
                    depth_trunc=2.0,     # Truncate depth beyond 3 meters
                    convert_rgb_to_intensity=False
                )

                # Generate intrinsic matrix in Open3D format
                o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=full_color.shape[1],
                    height=full_color.shape[0],
                    fx=intrinsic.T[0, 0],
                    fy=intrinsic.T[1, 1],
                    cx=intrinsic.T[0, 2],
                    cy=intrinsic.T[1, 2]
                )

                # Create point cloud from RGBD image
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, o3d_intrinsic
                ).voxel_down_sample(0.003)
                
                # Transform point cloud to align with the camera extrinsics (identity matrix here)
                # pcd.transform(current_extrs[-1])

                # Convert point cloud to numpy array
                pcd_array = np.hstack((
                    np.asarray(pcd.points),  # Shape (N, 3)
                    np.asarray(pcd.colors),  # Shape (N, 3)
                    np.zeros((len(pcd.points), 1))
                ))
            visualize_pcd(pcd)
            current_pcds.append(pcd_array)
            current_imgs.append(full_color)
            current_intrs.append(intrinsic.T)
            current_extrs.append(np.eye(4))

            current_info.append([each_record, idx])
            # break
        except Exception as e:
            print(f"Error processing {each_image}: {e}")
            saved = False
            break
    # break
    if saved:
        batch_pcds.extend(current_pcds)
        batch_rgbs.extend(current_imgs)
        batch_extrinsics.extend(current_extrs)
        batch_intrinsics.extend(current_intrs)
        batch_info.extend(current_info)
        res_data[each_record] = all_data[each_record]
    # break

labels_list = batch_segment_and_label(model, batch_pcds, 
                                      batch_rgbs, batch_intrinsics, batch_extrinsics)

for idx, (pcd, rgb) in tqdm(enumerate(zip(batch_pcds, batch_rgbs)), desc="Prossing Map", total=len(batch_pcds)):
    # new_pcd = map_point_cloud_colors_from_image(pcd, rgb, intrinsic)
    new_pcd = pcd
    labels = labels_list[idx]
    # new_pcd[:,3:6] = (new_pcd[:,3:6]*255).astype(np.uint8)
    new_pcd[:,6] = labels
    # print(f"color_range: {np.min(new_pcd[:,3:6])}, {np.max(new_pcd[:,3:6])}")
    # print(f"seg_range: {np.unique(new_pcd[:,6])}")
    record, r_idx = batch_info[idx]
    ply_name = resources_path / all_data[record]['point_cloud_files'][r_idx]
    vertex = np.array(
        [(x_p, y_p, z_p, r_p, g_p, b_p, s_p) for x_p, y_p, z_p, b_p, g_p, r_p, s_p in new_pcd],
        # there bgr -> rgb
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'), ('segment_id', 'i4')]
    )
    ply = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
    ply.write(str(ply_name))