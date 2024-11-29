import json
import logging
import numpy as np

import open3d as o3d
import open3d.core as o3c
import cv2

# Import specific modules from vtkmodules
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkActor,
    vtkPolyDataMapper
)
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkLine
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray, vtkDataArray
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(console_handler)

class PCDUpdater:
    def __init__(self, renderer: vtkRenderer, point_size: float = 2.0):
        self.renderer = renderer
        vtk_points = vtkPoints()
        vtk_colors = vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")

        self.polydata = vtkPolyData()
        self.polydata.SetPoints(vtk_points)
        self.polydata.GetPointData().SetScalars(vtk_colors)

        vertices = vtkCellArray()
        self.polydata.SetVerts(vertices)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(self.polydata)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(point_size)
        self.renderer.AddActor(actor)
    
    def update_pcd(self, pcd: o3d.geometry.PointCloud):
        # Convert Open3D point cloud to numpy arrays
        points = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        vtk_points_array = vtkPoints()
        vtk_points_array.SetData(numpy_to_vtk(points))

        vtk_colors_array: vtkDataArray = numpy_to_vtk(colors, deep=True, array_type=VTK_UNSIGNED_CHAR)
        vtk_colors_array.SetName("Colors")

        # Create vertices
        num_points = len(points)
        vtk_cells = vtkCellArray()
        vtk_cells_array = np.hstack((np.ones((num_points, 1), dtype=np.int64),
                                    np.arange(num_points, dtype=np.int64).reshape(-1, 1))).flatten()
        vtk_cells_id_array: vtkDataArray = numpy_to_vtkIdTypeArray(vtk_cells_array, deep=True)
        vtk_cells.SetCells(num_points, vtk_cells_id_array)

        # Update the polydata
        self.polydata.SetPoints(vtk_points_array)
        self.polydata.GetPointData().SetScalars(vtk_colors_array)
        self.polydata.SetVerts(vtk_cells)
        self.polydata.Modified()


class FakeCamera:
    """Fake camera that generates synthetic RGBD frames for debugging."""
    def __init__(self):
        self.width = 1280
        self.height = 720
        self.frame_idx = 0

    def connect(self, index):
        """Fake connect method, always returns True."""
        return True

    def disconnect(self):
        """Fake disconnect method."""
        pass

    def capture_frame(self, enable_align_depth_to_color=True):
        """Generate synthetic depth and color images with missing depth regions."""
        # Create a color image with a moving circle
        color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        center_x = int((self.frame_idx * 5) % self.width)
        center_y = self.height // 2
        cv2.circle(color_image, (center_x, center_y), 50, (0, 255, 0), -1)

        # Generate a depth image as a gradient
        depth_image = np.tile(np.linspace(500, 2000, self.width, dtype=np.uint16), (self.height, 1))

        # Randomly zero out some regions to simulate missing depth
        num_missing_regions = np.random.randint(5, 15)  # Random number of missing regions
        for _ in range(num_missing_regions):
            # Randomly choose the size and position of the missing region
            start_x = np.random.randint(0, self.width - 50)
            start_y = np.random.randint(0, self.height - 50)
            width = np.random.randint(20, 100)
            height = np.random.randint(20, 100)
            
            # Zero out the region
            depth_image[start_y:start_y + height, start_x:start_x + width] = 0

        self.frame_idx += 1

        # Return a fake RGBD frame
        return FakeRGBDFrame(depth_image, color_image)


class FakeRGBDFrame:
    """Fake RGBD frame containing synthetic depth and color images."""
    def __init__(self, depth_image, color_image):
        self.depth = depth_image
        self.color = color_image

class PCDStreamerFromCamera:
    """Fake RGBD streamer that generates synthetic RGBD frames."""
    def __init__(self, align_depth_to_color: bool = True, params: dict=None):
        self.align_depth_to_color = align_depth_to_color
        self.use_fake_camera = params.get('use_fake_camera', False)
        self.o3d_device = o3d.core.Device(params.get('device', 'cuda:0'))
        self.camera_config_file = params.get('camera_config', None)
        self.depth_max = 3.0 
        self.pcd_stride = 2  # downsample point cloud, may increase frame rate
        self.next_frame_func = None
        self.square_size = 0.015
        self.camera = None
        self.flag_normals = False
        self.__intrinsic_matrix:np.ndarray = np.array([[600, 0, 320], 
                                                       [0, 600, 240], 
                                                       [0, 0, 1]])
        self.__extrinsics:np.ndarray = np.eye(4)  # Example extrinsics (camera at origin)
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0])
        self.size = (1280, 720)
        # self.extrinsics = o3d.core.Tensor.eye(4, dtype=o3d.core.Dtype.Float32,
        #                                       device=self.o3d_device)
        self.camera_frustrum:CameraFrustum = CameraFrustum()
    
    @property
    def intrinsic_matrix(self):
        return self.__intrinsic_matrix
    
    @intrinsic_matrix.setter
    def intrinsic_matrix(self, value:np.ndarray=None):
        self.__intrinsic_matrix = value
        self.camera_frustrum.update_frustum(width=self.size[0], height=self.size[1], 
                                            intrinsic_matrix=self.__intrinsic_matrix, 
                                            extrinsics=self.__extrinsics)
        
    @property
    def extrinsics(self):
        return self.__extrinsics
    
    @extrinsics.setter
    def extrinsics(self, value:np.ndarray=None):
        self.__extrinsics = value
        self.camera_frustrum.update_frustum(width=self.size[0], height=self.size[1], 
                                            intrinsic_matrix=self.__intrinsic_matrix, 
                                            extrinsics=self.__extrinsics)

    def get_frame(self, take_pcd: bool = True):
        if self.camera is None:
            logger.warning("No camera connected")
        if take_pcd:
            rgbd_frame = self.camera.capture_frame(True)
            if rgbd_frame is None:
                return {}
            
            depth = o3d.t.geometry.Image(o3c.Tensor(np.asarray(rgbd_frame.depth), 
                                                    device=self.o3d_device))
            color = o3d.t.geometry.Image(o3c.Tensor(np.asarray(rgbd_frame.color), 
                                                    device=self.o3d_device))
            # logger.debug("Stream Debug Point 2.0")
            rgbd_image = o3d.t.geometry.RGBDImage(color, depth)
            pcd_frame = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, 
                o3c.Tensor(self.__intrinsic_matrix, dtype=o3c.Dtype.Float32, device=self.o3d_device), 
                o3c.Tensor(self.__extrinsics, dtype=o3c.Dtype.Float32, device=self.o3d_device),
                self.depth_scale, self.depth_max,
                self.pcd_stride, self.flag_normals)

            depth_in_color = depth.colorize_depth(
                    self.depth_scale, 0, self.depth_max)
            
            return {'pcd': pcd_frame.to_legacy(), 
                    'color': color, 
                    'depth': depth_in_color}

    def get_camera_frustum(self):
        return self.camera_frustrum
    
    def camera_mode_init(self):
        try:
            if self.camera_config_file:
                config = o3d.io.read_azure_kinect_sensor_config(self.camera_config_file)
                if self.camera is None:
                    if self.use_fake_camera:
                        self.camera = FakeCamera()
                    else:
                        self.camera = o3d.io.AzureKinectSensor(config)
                    self.camera_json = json.load(open(self.camera_config_file, 'r'))
                intrinsic = o3d.io.read_pinhole_camera_intrinsic(self.camera_config_file)
            else:
                if self.camera is None:
                    if self.use_fake_camera:
                        self.camera = FakeCamera()
                    else:
                        self.camera = o3d.io.AzureKinectSensor()
                # Use default intrinsics
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
                
            if not self.camera.connect(0):
                raise RuntimeError('Failed to connect to sensor')
            
            self.__intrinsic_matrix = intrinsic.intrinsic_matrix
            self.depth_scale = 1000.0  # Azure Kinect depth scale
            self.camera_frustrum.update_frustum(width=self.size[0], height=self.size[1], 
                                                intrinsic_matrix=self.__intrinsic_matrix, 
                                                extrinsics=self.__extrinsics)
            logger.info("Intrinsic matrix:")
            logger.info(self.__intrinsic_matrix)
            return True
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
        

class CameraFrustum:
    def __init__(self, color=(0.961, 0.475, 0.0), line_width=2.0):
        """
        Initializes the CameraFrustum class for rendering camera visualization.

        Args:
            renderer (vtk.vtkRenderer): The VTK renderer to add the frustum actor.
            color (tuple): RGB color of the frustum lines (default: orange).
            line_width (float): Line width of the frustum (default: 2.0).
        """

        # Initialize the points and lines for the frustum
        self.vtk_points = vtkPoints()
        self.vtk_lines = vtkCellArray()

        # Create a polydata to hold the geometry
        self.polydata = vtkPolyData()
        self.polydata.SetPoints(self.vtk_points)
        self.polydata.SetLines(self.vtk_lines)

        # Create a mapper and actor for the frustum
        self.mapper = vtkPolyDataMapper()
        self.mapper.SetInputData(self.polydata)

        self.actor = vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(color)  # Set the color
        self.actor.GetProperty().SetLineWidth(line_width)  # Set the line width

    def register_renderer(self, renderer: vtkRenderer):
        self.renderer = renderer
        self.renderer.AddActor(self.actor)

    def update_frustum(self, width, height, 
                       intrinsic_matrix, extrinsics, scale=0.1):
        """
        Updates the camera frustum visualization with new parameters.

        Args:
            width (int): Width of the image plane in pixels.
            height (int): Height of the image plane in pixels.
            intrinsic_matrix (np.ndarray): Camera intrinsic matrix.
            extrinsics (np.ndarray): Camera extrinsics matrix (4x4).
            scale (float): Scale for the frustum visualization (default: 1.0).
        """
        # Define the image plane corners in normalized device coordinates (NDC)
        corners = np.array([
            [0, 0, 1],  # Top-left
            [width, 0, 1],  # Top-right
            [width, height, 1],  # Bottom-right
            [0, height, 1],  # Bottom-left
        ])

        # Unproject corners to camera space
        fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
        camera_corners = np.zeros((4, 3))
        for i, (x, y, _) in enumerate(corners):
            camera_corners[i] = [(x - cx) / fx, (y - cy) / fy, 1]

        # Scale the frustum
        camera_corners *= scale

        # Add the camera position (origin in camera space)
        camera_position = np.array([0, 0, 0])
        all_points = np.vstack([camera_position, camera_corners])

        # Transform points to world space using extrinsics
        world_points = np.dot(np.linalg.inv(extrinsics), np.hstack((all_points, np.ones((5, 1)))).T).T[:, :3]

        # Define the lines for the frustum
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from camera center to corners
            [1, 2], [2, 3], [3, 4], [4, 1],  # Edges of the frustum
        ]

        # Update the points
        self.vtk_points.Reset()
        for point in world_points:
            self.vtk_points.InsertNextPoint(point)

        # Update the lines
        self.vtk_lines.Reset()
        for line in lines:
            vtk_line = vtkLine()
            vtk_line.GetPointIds().SetId(0, line[0])
            vtk_line.GetPointIds().SetId(1, line[1])
            self.vtk_lines.InsertNextCell(vtk_line)

        # Update the polydata and notify changes
        self.polydata.Modified()
        self.renderer.GetRenderWindow().Render()