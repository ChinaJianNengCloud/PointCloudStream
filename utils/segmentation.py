import torch
import numpy as np
import open3d as o3d
import open3d.core as o3c
import cv2
import torch.utils
import torch.utils.dlpack
from ultralytics import YOLO

def segment_pcd_from_2d(model: YOLO, pcd, color, intrinsic, extrinsic=np.eye(4)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Run the model to get the segmentation masks
    # color_torch = torch.from_numpy(color).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, 3, H, W)
    if isinstance(color, o3d.geometry.Image):
        color = np.asarray(color)
    elif isinstance(color, o3d.t.geometry.Image):
        color = np.asarray(color.cpu())

    res = model.predict(source=color, verbose=False)
        # 2. Get the 3D points from the point cloud
    if isinstance(pcd, o3d.geometry.PointCloud):
        points_3d = torch.from_numpy(np.asarray(pcd.points))
    elif isinstance(pcd, o3d.t.geometry.PointCloud):
        points_3d = pcd.point["positions"]
        points_3d = o3d_t_to_torch(points_3d).to(device)
    else:
        points_3d = torch.from_numpy(np.asarray(pcd.points)).to(device)  # Shape: (N, 3)
    N = points_3d.shape[0]

    try:
        masks = res[0].masks.data  # Shape: (num_masks, H', W')
    except:
        return np.zeros(N, dtype=np.int64)

    # Resize the masks to the original image size
    if not isinstance(color, np.ndarray):
        color = cv2.imread(color)
    H, W = color.shape[:2]
    masks_resized = torch.nn.functional.interpolate(
        masks.unsqueeze(1).float(),  # Add channel dimension
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # Shape: (num_masks, H, W)



    # 3. Project the 3D points into the 2D image plane
    ones = torch.ones((N, 1), device=device)
    points_3d_hom = torch.cat([points_3d, ones], dim=1)  # Shape: (N, 4)
    if isinstance(intrinsic, o3c.Tensor):
        intrinsic_torch = o3d_t_to_torch(intrinsic).to(device)
    elif isinstance(intrinsic, np.ndarray):
        intrinsic_torch = torch.from_numpy(intrinsic)

    if isinstance(extrinsic, o3c.Tensor):
        extrinsic_torch = o3d_t_to_torch(extrinsic).to(device)
    elif isinstance(extrinsic, np.ndarray):
        extrinsic = extrinsic.astype(np.float32)
        extrinsic_torch = torch.from_numpy(extrinsic).to(device)

    # Convert intrinsic and extrinsic to torch tensors
    # intrinsic_torch = o3d_t_to_torch(intrinsic) # Shape: (3, 3)
    # extrinsic_torch = o3d_t_to_torch(extrinsic) # Shape: (4, 4)

    # Transform points to camera coordinates
    points_cam_hom = (extrinsic_torch @ points_3d_hom.T).T  # Shape: (N, 4)
    points_cam = points_cam_hom[:, :3]  # Shape: (N, 3)

    # Filter out points with negative depth
    valid_depth = points_cam[:, 2] > 0
    points_cam = points_cam[valid_depth]
    indices = torch.nonzero(valid_depth).squeeze(1)

    # Project points to image plane
    fx = intrinsic_torch[0, 0]
    fy = intrinsic_torch[1, 1]
    cx = intrinsic_torch[0, 2]
    cy = intrinsic_torch[1, 2]
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    u = torch.round(u).long()
    v = torch.round(v).long()

    # Ensure u and v are within image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds]
    v = v[in_bounds]
    indices = indices[in_bounds]

    labels_per_mask = res[0].boxes.cls.to(device)  # Shape: (num_masks,)

    # 4. Assign labels based on segmentation masks
    mask_values = masks_resized[:, v, u]  # Shape: (num_masks, num_points)
    mask_scores, mask_ids = torch.max(mask_values, dim=0)  # Shape: (num_points,)
    threshold = 0.5
    valid_mask = mask_scores > threshold
    valid_indices = indices[valid_mask]
    point_labels = torch.full((N,), -1, dtype=torch.int32, device=device)  # Initialize to -1
    point_labels[valid_indices] = labels_per_mask[mask_ids[valid_mask]].int()

    # Convert labels to numpy array
    labels = point_labels.cpu().numpy()
    return labels

def o3d_t_to_torch(o3d_t):
    return torch.utils.dlpack.from_dlpack(o3d_t.to_dlpack())
