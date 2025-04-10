import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from pathlib import Path
from plyfile import PlyData
import vtk
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLPolyDataMapper
from vtkmodules.vtkRenderingCore import vtkActor

# Configuration
CONFIG = {
    "ransac_params": {
        "distance_threshold": 0.01,
        "ransac_n": 3,
        "num_iterations": 1000,
    },
    "dbscan_params": {
        "eps": 0.025,  # Neighborhood distance threshold
        "min_samples": 100,  # Minimum points in a cluster
    },
    "data_path": '/home/capre/disk_4/yutao/leo_data/data_2nd/resources',
    "ply_file": '0acd5ab6fd3341ca8a26b6508f311a85_point_cloud_0.ply',
    "z_threshold": 1.5,  # Constraint for filtering points
    "min_cluster_size": 100  # Minimum points required to keep a cluster
}

def read_ply_to_numpy(file_path):
    pcd = PlyData.read(open(file_path, 'rb'))
    pcd_array = np.array(pcd.elements[0].data)
    # Optionally convert structured array to a regular array (if needed)
    # Assuming the fields are 'x', 'y', 'z'
    points_array = np.vstack([pcd_array['x'], pcd_array['y'], pcd_array['z'], pcd_array['red'], pcd_array['green'], pcd_array['blue'], pcd_array['segment_id']]).T

    return points_array
# Function to load point cloud
def load_point_cloud(file_path):

    data = read_ply_to_numpy(file_path)
    points = data[:, :3]  # Extract x, y, z
    colors = data[:, 3:6]  # Extract r, g, b
    return points, colors

# Function to filter points based on constraints
def filter_points(points, colors, z_threshold):
    mask = points[:, 2] < z_threshold  # Keep points where z < z_threshold
    return points[mask], colors[mask], mask

# Function to apply RANSAC segmentation
def apply_ransac(points, params):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Apply RANSAC
    _, inliers = pcd.segment_plane(
        distance_threshold=params["distance_threshold"],
        ransac_n=params["ransac_n"],
        num_iterations=params["num_iterations"],
    )
    inlier_points = np.asarray(pcd.select_by_index(inliers).points)
    outlier_points = np.asarray(pcd.select_by_index(inliers, invert=True).points)

    return inlier_points, outlier_points, inliers

# Function to apply DBSCAN clustering
def apply_dbscan(points, params):
    db = DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit(points)
    labels = db.labels_
    return labels

# Function to remove small clusters
def remove_small_clusters(points, colors, labels, min_cluster_size):
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique_labels[counts >= min_cluster_size]
    valid_mask = np.isin(labels, valid_clusters)
    return points[valid_mask], colors[valid_mask], labels[valid_mask]

# Convert numpy array to VTK PolyData
def numpy_to_vtk(points, color):
    vtk_points = vtkPoints()
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(3)  # RGB
    vtk_colors.SetName("Colors")

    for i in range(points.shape[0]):
        vtk_points.InsertNextPoint(points[i])
        vtk_colors.InsertNextTuple3(*color[i])

    poly_data = vtkPolyData()
    poly_data.SetPoints(vtk_points)

    vertices = vtk.vtkVertexGlyphFilter()
    vertices.SetInputData(poly_data)
    vertices.Update()

    output_poly = vertices.GetOutput()
    output_poly.GetPointData().SetScalars(vtk_colors)
    return output_poly

# Create actor for VTK
def create_actor(poly_data, point_size):
    mapper = vtkOpenGLPolyDataMapper()
    mapper.SetInputData(poly_data)
    mapper.ScalarVisibilityOn()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(point_size)
    return actor

# Function to visualize labels using VTK
def visualize_labeled_points(points, colors, labels, show_seg=True):
    renderer = vtkRenderer()

    # Assign colors to labels
    unique_labels = np.unique(labels)
    rng = np.random.default_rng(42)  # Fixed seed for reproducible colors

    for label in unique_labels:
        if show_seg:
            if label == 0:
                color = [0, 255, 0]  # Green for plane points
            elif label == -1:
                color = [128, 128, 128]  # Gray for noise
            else:
                color = rng.integers(0, 255, size=3).tolist()  # Random color for clusters
        else:
            color = colors[labels == label]

        label_points = points[labels == label]
        if label_points.size > 0:
            if show_seg:
                poly_data = numpy_to_vtk(label_points, np.tile(color, (label_points.shape[0], 1)))
            else:
                poly_data = numpy_to_vtk(label_points, colors[labels == label])
            actor = create_actor(poly_data, point_size=2)
            renderer.AddActor(actor)

    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1280, 720)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    render_window.Render()
    interactor.Start()

# Main pipeline to label points
def label_point_cloud(points, colors, params):
    # Initialize all labels as -1 (unlabeled)
    labels = np.full(points.shape[0], -1, dtype=int)

    # Step 1: Apply RANSAC for the first plane
    inlier_points_1, outlier_points_1, inlier_indices_1 = apply_ransac(points, params["ransac_params"])
    labels[inlier_indices_1] = 0  # Label the first plane as 0

    # Step 2: Apply RANSAC for the second plane on the remaining points
    if len(outlier_points_1) > 0:
        inlier_points_2, outlier_points_2, inlier_indices_2 = apply_ransac(outlier_points_1, params["ransac_params"])
        
        # Map the second plane's inliers to the original indices
        outlier_indices_1 = np.setdiff1d(np.arange(points.shape[0]), inlier_indices_1)
        second_plane_indices = outlier_indices_1[inlier_indices_2]
        labels[second_plane_indices] = 1  # Label the second plane as 1

    # Step 3: Apply DBSCAN to the remaining outliers
    remaining_outliers = np.setdiff1d(np.arange(points.shape[0]), np.where(labels >= 0)[0])
    if len(remaining_outliers) > 0:
        outlier_points = points[remaining_outliers]
        dbscan_labels = apply_dbscan(outlier_points, params["dbscan_params"])

        # Map DBSCAN labels to sequential values starting from 2
        unique_labels = np.unique(dbscan_labels)
        cluster_label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=2)}

        # Assign DBSCAN labels to the outliers
        dbscan_labels_mapped = np.array([cluster_label_map[label] for label in dbscan_labels])

        # Update the original labels array
        labels[remaining_outliers] = dbscan_labels_mapped

    return labels

# Save labeled point cloud
def save_labeled_point_cloud(points, colors, labels, output_file):
    labeled_data = np.hstack((points, colors, labels.reshape(-1, 1)))
    np.savetxt(output_file, labeled_data, fmt="%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%d", header="x,y,z,r,g,b,label", comments="")


def seg_numpy_pcd(pcd: np.ndarray, z_threshold = 1.5):
    assert pcd.shape[1] == 7
    filtered_points, filtered_colors, mask = filter_points(pcd[:, :3], pcd[:, 3:6], z_threshold)
    labels = label_point_cloud(filtered_points, filtered_colors, CONFIG)
    # labels = remove_small_clusters(filtered_points, filtered_colors, labels, CONFIG["min_cluster_size"])
    labeled_pcd = np.hstack((filtered_points, filtered_colors, labels.reshape(-1, 1)))

    return labeled_pcd

# Main function
def main():
    # Load point cloud
    resource_folder = Path(CONFIG["data_path"])
    points, colors = load_point_cloud(resource_folder / CONFIG["ply_file"])

    # Filter points based on z-threshold
    filtered_points, filtered_colors, mask = filter_points(points, colors, CONFIG["z_threshold"])

    # Label points
    labels = label_point_cloud(filtered_points, filtered_colors, CONFIG)

    # Remove small clusters
    filtered_points, filtered_colors, labels = remove_small_clusters(filtered_points, filtered_colors, labels, CONFIG["min_cluster_size"])

    # Visualize labeled points
    visualize_labeled_points(filtered_points, filtered_colors, labels, show_seg=True)

if __name__ == "__main__":
    main()
