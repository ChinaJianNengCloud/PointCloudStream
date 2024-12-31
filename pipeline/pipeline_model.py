# pipeline_model.py
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
import open3d.core as o3c
from scipy.spatial.transform import Rotation as R
import cv2
import json
from utils.segmentation import segment_pcd_from_2d
import logging
from utils import ARUCO_BOARD
import open3d.visualization.gui as gui
from utils import CollectedData

np.set_printoptions(precision=3, suppress=True)

logger = logging.getLogger(__name__)

# calib = json.load(open('Calibration_results/calibration_results.json'))
# T_cam_to_base = np.array(calib.get('calibration_results').get('Tsai').get('transformation_matrix'))

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

class PipelineModel:
    """Controls IO (camera, video file, recording, saving frames). Methods run in worker threads."""

    def __init__(self, update_view, params: dict):
        """Initialize.

        Args:
            update_view (callback): Callback to update display elements for a frame.
            params (dict): Parameters including device, camera config, and other settings.
        """
        self.update_view = update_view
        self.params = params
        self.o3d_device = o3d.core.Device(params.get('device', 'cuda:0'))
        self.torch_device = torch.device(params.get('device', 'cuda:0'))
        self.camera_config_file = params.get('camera_config', None)
        self.render_done = False
        self.pcd_lock = threading.Lock()
        self.rgbd_video = params.get('rgbd_video', None)
        self.checkerboard_dims = (10, 7)
        self.video = None
        self.camera = None
        self.rgbd_frame = None
        self.close_stream = None
        self.T_cam_to_base = None

        # Fake camera flag
        self.use_fake_camera = params.get('use_fake_camera', False)

        # self.robot_trans_matrix = calibrated_camera_to_end_effector

        self.cv_capture = threading.Condition()  # condition variable
        self.cv_render = threading.Condition()

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
        self.calibration_data_init()
        self.collected_data = CollectedData()
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
        self.flag_stream_init = False
        self.flag_robot_init = False
        self.flag_center_to_base = False
        self.flag_tracking_board = False
        self.flag_camera_init = False
        self.flag_handeye_calib_init = False
        self.flag_handeye_calib_success = False
        self.flag_calib_collect = False
        self.flag_calib_axis_to_scene = False

        self.color_mean = None
        self.color_std = None
        self.pcd_frame = None
        self.rgbd_frame = None
        # self.objp = None
        self.dist_coeffs = None


    def calibration_data_init(self):
        from utils import CalibrationData
        charuco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_BOARD[self.params['board_type']])
        charuco_board = cv2.aruco.CharucoBoard(
            self.params['board_shape'],
            squareLength=self.params['board_square_size'] / 1000, # to meter
            markerLength=self.params['board_marker_size'] / 1000, # to meter
            dictionary=charuco_dict
        )
        self.calibration_data = CalibrationData(charuco_board, save_dir=self.params['folder_path'])
        return self.calibration_data

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

        self.rgbd_frame = None
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


    def run(self):
        """Run pipeline."""
        # n_pts = 0
        frame_id = 0
        t1 = time.perf_counter()
        while (not self.flag_exit and
               (self.video is None or  # Camera
                (self.video and not self.video.is_eof()))):  # Video
            self.frame = None
            if self.flag_stream_init:
                # logger.debug("Stream Debug Point 1")
                if self.video:
                    future_rgbd_frame = self.executor.submit(self.video.next_frame)
                else:
                    future_rgbd_frame = self.executor.submit(
                        self.camera.capture_frame, True)
                # logger.debug("Stream Debug Point 2")
                depth = o3d.t.geometry.Image(o3c.Tensor(np.asarray(self.rgbd_frame.depth), 
                                                        device=self.o3d_device))
                color = o3d.t.geometry.Image(o3c.Tensor(np.asarray(self.rgbd_frame.color), 
                                                        device=self.o3d_device))
                # logger.debug("Stream Debug Point 2.0")
                rgbd_image = o3d.t.geometry.RGBDImage(color, depth)

                self.pcd_frame = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, self.intrinsic_matrix, self.extrinsics,
                    self.depth_scale, self.depth_max,
                    self.pcd_stride, self.flag_normals)

                if self.pcd_frame.is_empty():
                    logger.warning(f"No valid depth data in frame {frame_id}")

                depth_in_color = depth.colorize_depth(
                    self.depth_scale, 0, self.depth_max)
                # logger.debug("Stream Debug Point 3")
                
                
                frame_elements = {
                    'color': color,
                    'depth': depth_in_color,
                    'pcd': self.pcd_frame.cpu(),
                    'intrinsic_matrix': self.intrinsic_matrix.cpu().numpy(),
                    'extrinsics': self.extrinsics.cpu().numpy(),
                }
                if frame_id % 30 == 0 and frame_id > 0:
                    t0, t1 = t1, time.perf_counter()
                    frame_elements['fps'] =  30 * (1.0 / (t1 - t0))
                # logger.debug("Stream Debug Point 5")
                if frame_id % 120 == 0:
                    self.color_mean = np.mean(frame_elements['pcd'].point.colors.cpu().numpy(), axis=0).tolist()  # frame_elements['pcd'].point.colors.mean(dim=0)
                    self.color_std = np.std(frame_elements['pcd'].point.colors.cpu().numpy(), axis=0).tolist()  # frame_elements['pcd'].point.colors.std(dim=0)
                    logger.debug(f"color_mean = {self.color_mean}, color_std = {self.color_std}")

                if self.flag_robot_init and self.flag_handeye_calib_success and self.flag_calib_axis_to_scene:
                    robot_end_frame, robot_base_frame = self.robot_tracking()
                    if robot_end_frame is None:
                        pass
                    else:
                        frame_elements['robot_end_frame'] = robot_end_frame
                        frame_elements['robot_base_frame'] = robot_base_frame

                if self.flag_tracking_board:
                    # logger.debug("Tracking board")
                    calib_color= self.camera_board_dectecting(axis_to_scene=self.flag_calib_axis_to_scene)
                    frame_elements['calib_color'] = o3d.t.geometry.Image(o3c.Tensor(calib_color, 
                                                                                    device=o3d.core.Device('cpu:0')))
                    if self.T_cam_to_board is not None:
                        frame_elements['board_pose'] = self.T_cam_to_board

                if self.flag_segemtation_mode:
                    try:
                        labels = segment_pcd_from_2d(self.pcd_seg_model, 
                                                    self.intrinsic_matrix, self.extrinsics, 
                                                    self.pcd_frame, np.asarray(self.rgbd_frame.color))
                    except Exception as e:
                        labels = np.zeros(self.pcd_frame.point.positions.shape[0])
                    frame_elements['seg'] = labels

                self.update_view(frame_elements, self.flag_center_to_base)
                
                with self.cv_render:
                    self.cv_render.wait_for(lambda: self.render_done)
                    self.render_done = False  # Reset for the next loop

                if self.flag_save_rgbd:
                    self.save_rgbd()
                    self.flag_save_rgbd = False

                if self.flag_save_pcd:
                    self.save_pcd()
                    self.flag_save_pcd = False
                
                if self.flag_calib_collect:
                    self.calib_collect(np.asarray(self.rgbd_frame.color), 
                                       self.flag_handeye_calib_init)
                    if len(self.calibration_data) > 0:
                        self.flag_calib_collect = False
                # logger.debug("Stream Debug Point 9")
                self.rgbd_frame = future_rgbd_frame.result()
                while self.rgbd_frame is None:
                    time.sleep(0.01)
                    self.rgbd_frame = self.camera.capture_frame(True)
            # logger.debug("Stream Debug Point 10")
            with self.cv_capture:  # Wait for capture to be enabled
                self.cv_capture.wait_for(
                    predicate=lambda: self.flag_capture or self.flag_exit)
            # logger.debug("Stream Debug Point 11")
            frame_id += 1
        try:
            self.close_stream()
        except Exception as e:
            print(e)

        self.executor.shutdown(wait=True)  # Ensure all threads finish cleanly
        self.calib_exec.shutdown(wait=True)


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
            self.pcd_seg_model = YOLO("yolo11x-seg.pt")
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
        from utils import CalibrationProcess
        try:
            
            self.calibration_process: CalibrationProcess = CalibrationProcess(self.params, 
                                                                          self.camera_interface, 
                                                                          self.robot_interface, 
                                                                          self.calibration_data)
            self.flag_handeye_init = True
            msg = 'Handeye: Initialized'
            msg_color = gui.Color(0, 1, 0)
        except Exception as e:
            msg = f'Handeye: Initialized failed'
            logger.error(msg+f' [{e}]')
            msg_color = gui.Color(1, 0, 0)
            self.flag_handeye_init = False

        return self.flag_handeye_init, msg, msg_color

    def robot_init(self):
        from utils import RobotInterface
        ip = None
        self.robot_interface: RobotInterface = RobotInterface()
        try:
            self.robot_interface.find_device()
            self.robot_interface.connect()
            ip = self.robot_interface.ip_address
            msg = f'Robot: Connected [{ip}]'
            msg_color = gui.Color(0, 1, 0)
            self.flag_robot_init = True
            self.calibration_data.reset()
        except Exception as e:
            msg = f'Robot: Connection failed'
            logger.error(msg+f' [{e}]')
            msg_color = gui.Color(1, 0, 0)
            self.flag_robot_init = False

        return self.flag_robot_init, msg, msg_color

    def camera_calibration_init(self):
        from utils import CameraInterface
        self.calibration_data.reset()
        self.camera_interface: CameraInterface = CameraInterface(self.camera, self.calibration_data)
        msg = "Calibration: Camera calibration initialized"
        return True, msg, gui.Color(0, 1, 0)
    
    def camera_board_dectecting(self, axis_to_scene: True):
        img = cv2.cvtColor(np.asarray(self.rgbd_frame.color), cv2.COLOR_RGB2BGR)
        processed_img, rvec, tvec = self.camera_interface._process_and_display_frame(img, 
                                                                                     self.calibration_data.camera_matrix, 
                                                                                     self.calibration_data.dist_coeffs, 
                                                                                     ret_vecs=True)
        if rvec is None or tvec is None or axis_to_scene is False:
            return processed_img

        self.T_cam_to_board[:3, :3] = cv2.Rodrigues(rvec)[0] #R.from_euler('xyz', rvec.reshape(1, 3), degrees=False).as_matrix().reshape(3, 3)
        self.T_cam_to_board[:3, 3] = tvec.ravel()
        # chessboard_pose_instance = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # chessboard_pose_instance.transform(self.T_cam_to_board)
        return processed_img
    
    def get_cam_space_gripper_pose(self):
        pose = self.robot_interface.capture_gripper_to_base(sep=False)
        t_xyz, r_xyz = pose[0:3], pose[3:6]
        if self.T_cam_to_base is None:
            logger.warning("Camera to base matrix did not detected, use robot pose instead!")
            return pose
        # rotation_matrix, _ = cv2.Rodrigues(rvecs)
        rotation_matrix = R.from_euler('xyz', r_xyz.reshape(1, 3), degrees=False).as_matrix().reshape(3, 3)
        T_end_to_base = np.eye(4)
        T_end_to_base[:3, :3] = rotation_matrix
        T_end_to_base[:3, 3] = t_xyz.ravel()
        T_base_to_cam =  np.linalg.inv(self.T_cam_to_base)
        T_cam_to_end = T_base_to_cam @ T_end_to_base
        # R.from_rotvec
        new_r = R.from_matrix(T_cam_to_end[:3, :3]).as_euler('xyz', degrees=False)
        new_t = T_cam_to_end[:3, 3]
        xyzrxrzry = np.hstack((new_r, new_t.reshape(-1)))
        # Add the robot frame to the frame elements for visualization
        return xyzrxrzry


    def robot_tracking(self):
        if self.T_cam_to_base is None:
            return None, None
        rvecs, tvects = self.robot_interface.capture_gripper_to_base()
        # rotation_matrix, _ = cv2.Rodrigues(rvecs)
        rotation_matrix = R.from_euler('xyz', rvecs.reshape(1, 3), degrees=False).as_matrix().reshape(3, 3)
        T_end_to_base = np.eye(4)
        T_end_to_base[:3, :3] = rotation_matrix
        T_end_to_base[:3, 3] = tvects.ravel()
        T_base_to_cam =  np.linalg.inv(self.T_cam_to_base)
        T_cam_to_end = T_base_to_cam @ T_end_to_base

        return T_base_to_cam, T_cam_to_end
    
    def calib_collect(self, img: np.ndarray, with_robot_pose=False):
        if with_robot_pose:
            logger.info("Capturing robot pose...")
            robot_pose = self.robot_interface.capture_gripper_to_base(sep=False)
        else:
            robot_pose = None
        self.calibration_data.append(img, robot_pose=robot_pose)
    
    def update_camera_matrix(self, intrinsic:np.ndarray, dist_coeffs:np.ndarray):
        # intrinsic = np.array(self.calib.get('camera_matrix'))
        self.dist_coeffs = dist_coeffs
        self.intrinsic_matrix =  o3d.core.Tensor(
                                    intrinsic,
                                    dtype=o3d.core.Dtype.Float32,
                                    device=self.o3d_device)
    
    def auto_calibration(self):
        self.robot_interface.set_teach_mode(False)
        for idx, each_pose in enumerate(self.calibration_data.robot_poses):
            logger.info(f"Moving to pose {idx}")
            self.robot_interface.move_to_pose(each_pose)
            pose = self.robot_interface.capture_gripper_to_base(sep=False)
            img = np.asarray(self.rgbd_frame.color)
            self.calibration_data.modify(idx, img, pose)
        self.robot_interface.set_teach_mode(True)
