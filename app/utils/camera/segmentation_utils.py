import torch
import numpy as np
import open3d as o3d
import open3d.core as o3c
import cv2
import torch.utils
import torch.utils.dlpack
from ultralytics import YOLO
from plyfile import PlyData


def segment_pcd_from_2d(model: YOLO, pcd, color, intrinsic, extrinsic=np.eye(4)):
    """
    Segment a point cloud into classes given a segmentation model, a point cloud,
    color image, intrinsic matrix, and extrinsic matrix.

    Parameters
    ----------
    model : YOLO
        Segmentation model
    pcd : open3d.geometry.PointCloud or open3d.t.geometry.PointCloud or array
        Point cloud with shape (N, 3)
    color : open3d.geometry.Image or open3d.t.geometry.Image or str
        Color image or path to color image
    intrinsic : open3d.core.Tensor or numpy array
        Intrinsic matrix with shape (3, 3)
    extrinsic : open3d.core.Tensor or numpy array, optional
        Extrinsic matrix with shape (4, 4), default is identity

    Returns
    -------
    labels : numpy array
        Labels for each point in the point cloud, shape (N,)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(color, o3d.geometry.Image):
        color = np.asarray(color)
    elif isinstance(color, o3d.t.geometry.Image):
        color = np.asarray(color.cpu())

    res = model.predict(source=color)
        # 2. Get the 3D points from the point cloud
    if isinstance(pcd, o3d.geometry.PointCloud):
        points_3d = torch.from_numpy(np.asarray(pcd.points)).to(device).to(dtype=torch.float32)
    elif isinstance(pcd, o3d.t.geometry.PointCloud):
        points_3d = pcd.point["positions"]
        points_3d = o3d_t_to_torch(points_3d).to(device).to(dtype=torch.float32)
    elif isinstance(pcd, np.ndarray):
        points_3d = torch.from_numpy(pcd).to(device).to(dtype=torch.float32)
    else:
        points_3d = torch.from_numpy(np.asarray(pcd.points)).to(device).to(dtype=torch.float32)  # Shape: (N, 3)
    N = points_3d.shape[0]
    points_3d = points_3d[:, :3]  # Shape: (N, 3)
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
        intrinsic_torch = torch.from_numpy(intrinsic.copy())

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


def read_ply_to_numpy(file_path):
    pcd = PlyData.read(open(file_path, 'rb'))
    pcd_array = np.array(pcd.elements[0].data)
    # Optionally convert structured array to a regular array (if needed)
    # Assuming the fields are 'x', 'y', 'z'
    points_array = np.vstack([pcd_array['x'], pcd_array['y'], pcd_array['z'], pcd_array['red'], pcd_array['green'], pcd_array['blue'], pcd_array['segment_id']]).T

    return points_array


def segment_image_with_yolo(model, color):
    """
    Segment an image using a YOLO model.

    Parameters
    ----------
    model : YOLO
        Segmentation model.
    color : open3d.geometry.Image, open3d.t.geometry.Image, or str
        Color image or path to color image.

    Returns
    -------
    res : YOLO result
        Segmentation result containing masks and classes.
    H : int
        Height of the original image.
    W : int
        Width of the original image.
    """
    if isinstance(color, o3d.geometry.Image):
        color = np.asarray(color)
    elif isinstance(color, o3d.t.geometry.Image):
        color = np.asarray(color.cpu())
    elif isinstance(color, str):
        color = cv2.imread(color)

    H, W = color.shape[:2]
    res = model.predict(source=color)
    return res, H, W

def label_point_cloud_from_segmentation(res, pcd, intrinsic, extrinsic=np.eye(4), H=None, W=None):
    """
    Label a point cloud using YOLO segmentation results.

    Parameters
    ----------
    res : YOLO result
        Segmentation result containing masks and classes.
    pcd : open3d.geometry.PointCloud, open3d.t.geometry.PointCloud, or numpy array
        Point cloud with shape (N, 3).
    intrinsic : open3d.core.Tensor or numpy array
        Intrinsic matrix with shape (3, 3).
    extrinsic : open3d.core.Tensor or numpy array, optional
        Extrinsic matrix with shape (4, 4), default is identity.
    H : int, optional
        Height of the original image (required if resizing masks).
    W : int, optional
        Width of the original image (required if resizing masks).

    Returns
    -------
    labels : numpy array
        Labels for each point in the point cloud, shape (N,).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(pcd, o3d.geometry.PointCloud):
        points_3d = torch.from_numpy(np.asarray(pcd.points)).to(device).to(dtype=torch.float32)
    elif isinstance(pcd, o3d.t.geometry.PointCloud):
        points_3d = o3d_t_to_torch(pcd.point["positions"]).to(device).to(dtype=torch.float32)
    elif isinstance(pcd, np.ndarray):
        points_3d = torch.from_numpy(pcd).to(device).to(dtype=torch.float32)

    N = points_3d.shape[0]
    ones = torch.ones((N, 1), device=device)
    points_3d_hom = torch.cat([points_3d, ones], dim=1)

    if isinstance(intrinsic, np.ndarray):
        intrinsic_torch = torch.from_numpy(intrinsic.copy()).to(device)
    else:
        intrinsic_torch = o3d_t_to_torch(intrinsic).to(device)

    if isinstance(extrinsic, np.ndarray):
        extrinsic_torch = torch.from_numpy(extrinsic.astype(np.float32)).to(device)
    else:
        extrinsic_torch = o3d_t_to_torch(extrinsic).to(device)

    # Transform points to camera coordinates
    points_cam_hom = (extrinsic_torch @ points_3d_hom.T).T
    points_cam = points_cam_hom[:, :3]

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

    # Resize masks if necessary
    
    masks = res[0].masks.data
    masks_resized = torch.nn.functional.interpolate(
        masks.unsqueeze(1).float(), size=(H, W), mode='bilinear', align_corners=False
    ).squeeze(1)

    labels_per_mask = res[0].boxes.cls.to(device)

    # Assign labels
    mask_values = masks_resized[:, v, u]
    mask_scores, mask_ids = torch.max(mask_values, dim=0)
    threshold = 0.5
    valid_mask = mask_scores > threshold
    valid_indices = indices[valid_mask]

    point_labels = torch.full((N,), -1, dtype=torch.int32, device=device)
    point_labels[valid_indices] = labels_per_mask[mask_ids[valid_mask]].int()

    return point_labels.cpu().numpy()

def batch_segment_and_label(model, point_clouds, color_images, intrinsics, extrinsics):
    """
    Batch process point clouds and color images for segmentation and labeling.

    Parameters
    ----------
    model : YOLO
        Segmentation model.
    point_clouds : list
        List of point clouds.
    color_images : list
        List of color images.
    intrinsics : list
        List of intrinsic matrices.
    extrinsics : list
        List of extrinsic matrices.

    Returns
    -------
    results : list
        List of labeled point clouds.
    """
    results = []
    for pcd, color, intrinsic, extrinsic in zip(point_clouds, color_images, intrinsics, extrinsics):
        res, H, W = segment_image_with_yolo(model, color)
        try:
            labels = label_point_cloud_from_segmentation(res, pcd[:, 0:3], intrinsic, extrinsic, H, W)
        except:
            labels = np.zeros(pcd.shape[0], dtype=np.int64)
        results.append(labels)
        # print(np.count_nonzero(labels==80))
    return results


def map_point_cloud_colors_from_image(pcd_array, color_image, intrinsic, extrinsic=np.eye(4)):
    """
    Directly map colors from a 2D image to a 3D point cloud.

    Parameters
    ----------
    pcd_array : numpy.ndarray
        Point cloud as a numpy array with columns [x, y, z, red, green, blue, segment_id].
    color_image : open3d.geometry.Image, open3d.t.geometry.Image, or str
        Color image or path to color image.
    intrinsic : numpy.ndarray
        Intrinsic matrix with shape (3, 3).
    extrinsic : numpy.ndarray, optional
        Extrinsic matrix with shape (4, 4), default is identity.

    Returns
    -------
    mapped_pcd : numpy.ndarray
        Point cloud as a numpy array with updated colors:
        [x, y, z, red, green, blue, segment_id].
    """
    import numpy as np
    import torch
    import cv2
    import open3d as o3d

    # Load the color image
    if isinstance(color_image, str):
        color_image = cv2.imread(color_image)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    elif isinstance(color_image, o3d.geometry.Image):
        color_image = np.asarray(color_image)
    elif isinstance(color_image, o3d.t.geometry.Image):
        color_image = np.asarray(color_image.cpu())

    H, W = color_image.shape[:2]

    # Extract point cloud components
    points_3d = pcd_array[:, :3]
    segment_ids = pcd_array[:, 6]  # Maintain segment_ids

    # Convert matrices and point cloud to torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    intrinsic_torch = torch.from_numpy(intrinsic).float().to(device)
    extrinsic_torch = torch.from_numpy(extrinsic).float().to(device)
    points_3d_torch = torch.from_numpy(points_3d).float().to(device)

    N = points_3d_torch.shape[0]
    ones = torch.ones((N, 1), device=device)
    points_3d_hom = torch.cat([points_3d_torch, ones], dim=1)  # Add homogeneous coordinate

    # Transform points to camera coordinates
    points_cam_hom = (extrinsic_torch @ points_3d_hom.T).T
    points_cam = points_cam_hom[:, :3]

    # Filter out points with negative depth
    valid_depth = points_cam[:, 2] > 0
    points_cam = points_cam[valid_depth]
    indices = torch.nonzero(valid_depth, as_tuple=True)[0]

    # Project points to the 2D image plane
    fx, fy = intrinsic_torch[0, 0], intrinsic_torch[1, 1]
    cx, cy = intrinsic_torch[0, 2], intrinsic_torch[1, 2]

    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    u = torch.round(u).long()
    v = torch.round(v).long()

    # Ensure u and v are within image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[in_bounds], v[in_bounds]
    indices = indices[in_bounds]

    # Retrieve colors from the 2D image
    color_image_tensor = torch.from_numpy(color_image).float().to(device) / 255.0  # Normalize to [0, 1]
    mapped_colors = (color_image_tensor[v, u] * 255).byte()  # Rescale to [0, 255] and convert to byte

    # Create an array for colors
    colors = torch.zeros((N, 3), dtype=torch.uint8, device=device)
    colors[indices] = mapped_colors

    # Combine back into a single tensor
    colors = colors.float()
    segment_ids_torch = torch.from_numpy(segment_ids).to(device).float()  # Ensure segment_ids is on the same device
    segment_ids_filtered = segment_ids_torch[valid_depth][:, None]  # Filtered segment IDs

    # Concatenate all components into a single tensor
    mapped_pcd = torch.cat([points_3d_torch, colors, segment_ids_filtered], dim=1)

    # Convert back to numpy array and ensure it's on the CPU
    mapped_pcd = mapped_pcd.cpu().numpy()

    return mapped_pcd