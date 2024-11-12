# pipeline_model.py
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
from scipy.spatial.transform import Rotation as R
import cv2
import json
from utils.segmentation import segment_pcd_from_2d
from utils.robot import RobotInterface
import logging
from utils import ARUCO_BOARD


logger = logging.getLogger(__name__)

# calib = json.load(open('Calibration_results/calibration_results.json'))
# T_cam_to_base = np.array(calib.get('calibration_results').get('Tsai').get('transformation_matrix'))

class PipelineModel:
    """Controls IO (camera, video file, recording, saving frames). Methods run
    in worker threads."""

    def __init__(self, update_view, params: dict):
        """Initialize.

        Args:
            update_view (callback): Callback to update display elements for a
                frame.
            camera_config_file (str): Camera configuration json file.
            rgbd_video (str): RS bag file containing the RGBD video. If this is
                provided, connected cameras are ignored.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
        """
        self.update_view = update_view
        self.params = params
        self.o3d_device = o3d.core.Device(params.get('device', 'cuda:0'))
        self.torch_device = torch.device(params.get('device', 'cuda:0'))
        self.camera_config_file = params.get('camera_config', None)
        self.rgbd_video = params.get('rgbd_video', None)
        self.checkerboard_dims = (10, 7)
        self.video = None
        self.camera = None
        self.rgbd_frame = None
        self.close_stream = None
        self.T_cam_to_base = None
        self.robot:RobotInterface = None
        # self.robot_trans_matrix = calibrated_camera_to_end_effector

        self.cv_capture = threading.Condition()  # condition variable

        # RGBD -> PCD
        self.extrinsics = o3d.core.Tensor.eye(4,
                                              dtype=o3d.core.Dtype.Float32,
                                              device=self.o3d_device)
        
        # self.extrinsics = o3c.Tensor(robot_trans_matrix, dtype=o3d.core.Dtype.Float32,
        #                                       device=self.o3d_device)
        self.depth_max = 3.0  # m
        self.pcd_stride = 2  # downsample point cloud, may increase frame rate
        self.next_frame_func = None
        self.square_size = 0.015
        self.T_cam_to_board = np.eye(4)
        self.__init_gui_signals()
        self.executor = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Capture-Save')
        
        self.calib_exec = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Calibreation')
        
    
    def __init_gui_signals(self):
        self.recording = False  # Are we currently recording
        self.flag_record = False  # Request to start/stop recording
        self.flag_capture = False
        self.flag_normals = False
        self.flag_save_rgbd = False
        self.flag_save_pcd = False
        self.flag_segemtation_mode = False
        self.flag_exit = False
        self.color_mean = None
        self.color_std = None
        self.pcd_frame = None
        self.rgbd_frame = None
        self.flag_stream_init = False
        self.robot_init = False
        self.camera_init = False
        self.hand_eye_calib = False
        self.flag_calibration_mode = False
        self.objp = None
        self.dist_coeffs = None

    @property
    def max_points(self):
        """Max points in one frame for the camera or RGBD video resolution."""
        # Adjusted for Azure Kinect default resolution
        return 1280 * 720  # Adjust according to your camera's resolution

    @property
    def vfov(self):
        """Camera or RGBD video vertical field of view."""
        return np.rad2deg(2 * np.arctan(self.intrinsic_matrix[1, 2].item() /
                                        self.intrinsic_matrix[1, 1].item()))
    
    def camera_mode_init(self):
        if self.camera_config_file:
            config = o3d.io.read_azure_kinect_sensor_config(self.camera_config_file)
            if self.camera is None:
                self.camera = o3d.io.AzureKinectSensor(config)
                self.camera_json = json.load(open(self.camera_config_file, 'r'))
            intrinsic = o3d.io.read_pinhole_camera_intrinsic(self.camera_config_file)
        else:
            if self.camera is None:
                self.camera = o3d.io.AzureKinectSensor()
            # Use default intrinsics
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            
        # print(self.camera.connect(0))
        if not self.camera.connect(0):
            raise RuntimeError('Failed to connect to sensor')
        
        self.intrinsic_matrix = o3d.core.Tensor(
            intrinsic.intrinsic_matrix,
            dtype=o3d.core.Dtype.Float32,
            device=self.o3d_device)
        self.depth_scale = 1000.0  # Azure Kinect depth scale
        logger.info("Intrinsic matrix:")
        logger.info(self.intrinsic_matrix)

        while self.rgbd_frame is None:
            time.sleep(0.01)
            try:
                self.rgbd_frame = self.camera.capture_frame(True)
            except:
                pass
        self.close_stream = self.camera.disconnect
        # self.next_frame_func = self.camera.capture_frame

    def video_mode_init(self):
        self.video = o3d.t.io.RGBDVideoReader.create(self.rgbd_video)
        self.rgbd_metadata = self.video.metadata
        self.status_message = f"Video {self.rgbd_video} opened."

        # Get intrinsics from the video metadata
        self.intrinsic_matrix = o3d.core.Tensor(
            self.rgbd_metadata.intrinsics.intrinsic_matrix,
            dtype=o3d.core.Dtype.Float32,
            device=self.o3d_device)
        self.depth_scale = self.rgbd_metadata.depth_scale
        self.status_message = "RGBD video Loaded."
        logger.info("Intrinsic matrix:")
        logger.info(self.intrinsic_matrix)
        self.rgbd_frame = self.video.next_frame()
        self.close_stream = self.video.close
        # self.next_frame_func = self.video.next_frame

    def objp_update(self, chessboard_dims, square_size=0.02):
        self.params['board_shape'], 
        self.checkerboard_dims = chessboard_dims
        self.square_size = self.params['board_square_size']
        self.objp = np.zeros((self.checkerboard_dims[1] * self.checkerboard_dims[0], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard_dims[0], 0:self.checkerboard_dims[1]].T.reshape(-1, 2) * self.square_size


    def run(self):
        """Run pipeline."""
        n_pts = 0
        frame_id = 0
        t1 = time.perf_counter()
        self.objp_update(self.checkerboard_dims, self.square_size)

        pcd_errors = 0
        while (not self.flag_exit and
               (self.video is None or  # Camera
                (self.video and not self.video.is_eof()))):  # Video
            
            if not self.flag_stream_init:
                continue

            if self.video:
                future_rgbd_frame = self.executor.submit(self.video.next_frame)
            else:
                future_rgbd_frame = self.executor.submit(
                    self.camera.capture_frame, True)
                    
            try:
                # if self.rgbd_frame is None:
                #     continue
                # time.sleep(0.01)
                depth = o3d.t.geometry.Image(o3c.Tensor(np.asarray(self.rgbd_frame.depth), 
                                                        device=self.o3d_device))
                color = o3d.t.geometry.Image(o3c.Tensor(np.asarray(self.rgbd_frame.color), 
                                                        device=self.o3d_device))

                rgbd_image = o3d.t.geometry.RGBDImage(color, depth)

                self.pcd_frame = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, self.intrinsic_matrix, self.extrinsics,
                    self.depth_scale, self.depth_max,
                    self.pcd_stride, self.flag_normals)
                # print(color.columns, color.rows)
                camera_line = o3d.geometry.LineSet.create_camera_visualization(
                    color.columns, color.rows, self.intrinsic_matrix.cpu().numpy(),
                    np.linalg.inv(self.extrinsics.cpu().numpy()), 0.2)
                camera_line.paint_uniform_color([0.961, 0.475, 0.000])
                
                if self.pcd_frame.is_empty():
                    logger.warning(f"No valid depth data in frame {frame_id}")
                    continue
                
                depth_in_color = depth.colorize_depth(
                    self.depth_scale, 0, self.depth_max)

            except RuntimeError as e:
                pcd_errors += 1
                logger.warning(f"Runtime error in frame {frame_id}: {e}")
                continue


            frame_elements = {
                'color': color.cpu(),
                'depth': depth_in_color.cpu(),
                'pcd': self.pcd_frame.cpu().clone(),
                'camera': camera_line,
                # 'status_message': self.status_message
            }

            n_pts += self.pcd_frame.point.positions.shape[0]
            if frame_id % 30 == 0 and frame_id > 0:
                t0, t1 = t1, time.perf_counter()
            #     logger.debug(f"\nframe_id = {frame_id}, \t {(t1-t0)*1000./60:0.2f}"
            #               f"ms/frame \t {(t1-t0)*1e9/n_pts:.2f} ms/Mp\t")
            #     n_pts = 0
                frame_elements['fps'] =  30 * (1.0 / (t1 - t0))

            if frame_id % 120 == 0:
                self.color_mean = np.mean(frame_elements['pcd'].point.colors.cpu().numpy(), axis=0).tolist()  # frame_elements['pcd'].point.colors.mean(dim=0)
                self.color_std = np.std(frame_elements['pcd'].point.colors.cpu().numpy(), axis=0).tolist()  # frame_elements['pcd'].point.colors.std(dim=0)
                logger.debug(f"color_mean = {self.color_mean}, color_std = {self.color_std}")

            if self.robot_init and self.hand_eye_calib:
                # Draw an axis for the robot position pose in the scene
                robot_end_frame, robot_base_frame = self.robot_tracking()
                frame_elements['robot_end_frame'] = robot_end_frame
                frame_elements['robot_base_frame'] = robot_base_frame

            if self.flag_calibration_mode:
                chessboard_pose = self.chessboard_tracking(
                    color_image=self.rgbd_frame.color)
                if chessboard_pose is not None:
                    frame_elements['chessboard'] = chessboard_pose

            if self.flag_segemtation_mode:
                labels = segment_pcd_from_2d(self.pcd_seg_model, 
                                             self.intrinsic_matrix, self.extrinsics, 
                                             self.pcd_frame, np.asarray(self.rgbd_frame.color))
                frame_elements['seg'] = labels

            self.update_view(frame_elements)

            if self.flag_save_rgbd:
                self.save_rgbd()
                self.flag_save_rgbd = False

            if self.flag_save_pcd:
                self.save_pcd()
                self.flag_save_pcd = False

            self.rgbd_frame = future_rgbd_frame.result()
            while self.rgbd_frame is None:
                time.sleep(0.01)
                self.rgbd_frame = self.camera.capture_frame(True)

            with self.cv_capture:  # Wait for capture to be enabled
                self.cv_capture.wait_for(
                    predicate=lambda: self.flag_capture or self.flag_exit)
            self.toggle_record()
            frame_id += 1

        self.close_stream()
        self.executor.shutdown(wait=True)  # Ensure all threads finish cleanly
        self.calib_exec.shutdown(wait=True)
        logger.debug(f"create_from_depth_image() errors = {pcd_errors}")

    def toggle_record(self):
        """Toggle recording RGBD video.
        
        This function is called when the user clicks the "Record" button. It
        sets or unsets the flag to record RGBD video in the PipelineModel.
        """
        pass  # Recording functionality can be implemented if needed
    
    def seg_model_intialization(self):
        self.flag_capture = False
        if not hasattr(self, 'pcd_seg_model'):
            from ultralytics import YOLO, SAM
            # self.executor.submit(get_model, "third_party/scannet200_val.ckpt")
            # model_checkpoint_path = "/home/capre/sapiens/sapiens_host/seg/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"
            self.pcd_seg_model = YOLO("yolo11s-seg.pt")
            # self.pcd_seg_model = SAM("sam2.1_t.pt")
            # self.yoloworld = YOLOWORLD_Frame(image_size=[1280, 720], vocab='')
        # self.flag_model_init = True

    def save_pcd(self):
        """Save current point cloud."""
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"pcd_{now}.ply"
        self.pcd_frame.point.colors = (self.pcd_frame.point.colors * 255).to(
            o3d.core.Dtype.UInt8)
        self.executor.submit(o3d.t.io.write_point_cloud,
                             filename,
                             self.pcd_frame,
                             write_ascii=False,
                             compressed=True,
                             print_progress=False)
        self.status_message = f"Saving point cloud to {filename}."


    def save_rgbd(self):
        """Save current RGBD image pair."""
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename_color = f"color_{now}.jpg"
        filename_depth = f"depth_{now}.png"
        color_image = o3d.t.geometry.Image(
            o3d.core.Tensor(np.asarray(self.rgbd_frame.color),
                            dtype=o3d.core.Dtype.UInt8)
        ).cpu()
        depth_image = o3d.t.geometry.Image(
            o3d.core.Tensor(np.asarray(self.rgbd_frame.depth),
                            dtype=o3d.core.Dtype.UInt16)
        ).cpu()

        self.executor.submit(o3d.t.io.write_image, filename_color, color_image)
        self.executor.submit(o3d.t.io.write_image, filename_depth, depth_image)
        self.status_message = (
            f"Saving RGBD images to {filename_color} and {filename_depth}.")
        
    def handeye_calibration_init(self):
        pass

    def camera_calibration_init(self):
        import cv2
        from utils import CameraInterface
        charuco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_BOARD[self.params['board_type']])
        charuco_board = cv2.aruco.CharucoBoard(
            self.params['board_shape'],
            squareLength=self.params['board_square_size'] / 1000, # to meter
            markerLength=self.params['board_marker_size'] / 1000, # to meter
            dictionary=charuco_dict
        )

        self.camera_interface = CameraInterface(self.camera, charuco_dict, charuco_board)

    def robot_tracking(self):
        pose_robot = self.robot.get_position()
        rvecs, tvects = self.robot.capture_gripper_to_base()
        # Extract position and rotation from pose
        # rotation_robot = R.from_euler('xyz', [rx, ry, rz])
        # # rotation_robot = cv2.Rodrigues(np.array([rx, ry, rz]))
        # robot_R = rotation_robot.as_matrix()
        # rotation_matrix = R.from_euler('xyz', [rx, ry, rz], degrees=False).as_matrix()
        # rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        rotation_matrix, _ = cv2.Rodrigues(rvecs)
        

        T_end_to_base = np.eye(4)
        T_end_to_base[:3, :3] = rotation_matrix
        T_end_to_base[:3, 3] = tvects.ravel()
        # T_base_to_cam = self.T_cam_to_base
        T_base_to_cam =  np.linalg.inv(self.T_cam_to_base)
        # logger.debug(f"Robot pose: {x}, {y}, {z}, {rx}, {ry}, {rz}")
        # T_end_to_base = np.linalg.inv(T_base_to_end)
        # T_ee_to_cam = np.eye(4)
        # T_ee_to_cam[0:3, 0:3] = R_calibrated
        # T_ee_to_cam[0:3, 3] = T_calibrated.ravel()
        T_cam_to_end = T_base_to_cam @ T_end_to_base
        # cur_x, cur_y, cur_z = T_base_to_cam[0, 3], T_base_to_cam[1, 3], T_base_to_cam[2, 3]
        # logger.debug(f"Robot pose: {cur_x}, {cur_y}, {cur_z}")
        # Create a coordinate frame at the robot's position in the scene
        robot_end_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        robot_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        robot_end_frame.transform(T_cam_to_end)
        robot_base_frame.transform(T_base_to_cam)
        # Add the robot frame to the frame elements for visualization
        return robot_end_frame, robot_base_frame

    def chessboard_tracking(self, color_image):
        color = np.asarray(color_image)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_dims, None, flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1,-1), 
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            ret2, board_to_camera_rvecs, board_to_camera_tvecs = cv2.solvePnP(
                    self.objp, corners2, self.intrinsic_matrix.cpu().numpy(), 
                    self.dist_coeffs)
            if ret2:
                
                self.T_cam_to_board [:3, :3] = R.from_euler('xyz', board_to_camera_rvecs.reshape(1, 3), degrees=False).as_matrix().reshape(3, 3)
                self.T_cam_to_board [:3, 3] = board_to_camera_tvecs.ravel()
                chessboard_pose_instance = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                chessboard_pose_instance.transform(self.T_cam_to_board )
                logger.debug("chessboard_detected")
                return chessboard_pose_instance

        return None