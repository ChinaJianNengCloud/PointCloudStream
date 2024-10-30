# pipeline_model.py
import threading
import time
import logging as log
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c

from segmentation import segment_pcd_from_2d


class PipelineModel:
    """Controls IO (camera, video file, recording, saving frames). Methods run
    in worker threads."""

    def __init__(self,
                 update_view,
                 camera_config_file=None,
                 rgbd_video=None,
                 device=None):
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
        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        self.o3d_device = o3d.core.Device(self.device)
        self.torch_device = torch.device('cuda:0')
        
        self.video = None
        self.camera = None

        self.cv_capture = threading.Condition()  # condition variable

        if rgbd_video:  # Video file
            self.video = o3d.t.io.RGBDVideoReader.create(rgbd_video)
            self.rgbd_metadata = self.video.metadata
            self.status_message = f"Video {rgbd_video} opened."

            # Get intrinsics from the video metadata
            self.intrinsic_matrix = o3d.core.Tensor(
                self.rgbd_metadata.intrinsics.intrinsic_matrix,
                dtype=o3d.core.Dtype.Float32,
                device=self.o3d_device)
            self.depth_scale = self.rgbd_metadata.depth_scale

        else:  # Azure Kinect camera
            if camera_config_file:
                config = o3d.io.read_azure_kinect_sensor_config(camera_config_file)
                self.camera = o3d.io.AzureKinectSensor(config)
                intrinsic = o3d.io.read_pinhole_camera_intrinsic(camera_config_file)
            else:
                self.camera = o3d.io.AzureKinectSensor()
                # Use default intrinsics
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            if not self.camera.connect(0):
                raise RuntimeError('Failed to connect to sensor')
            # self.camera.start_capture()
            # Set the intrinsic matrix
            self.intrinsic_matrix = o3d.core.Tensor(
                intrinsic.intrinsic_matrix,
                dtype=o3d.core.Dtype.Float32,
                device=self.o3d_device)
            self.depth_scale = 1000.0  # Azure Kinect depth scale
            self.status_message = "Azure Kinect camera connected."

        log.info("Intrinsic matrix:")
        log.info(self.intrinsic_matrix)

        # RGBD -> PCD
        self.extrinsics = o3d.core.Tensor.eye(4,
                                              dtype=o3d.core.Dtype.Float32,
                                              device=self.o3d_device)
        self.depth_max = 3.0  # m
        self.pcd_stride = 2  # downsample point cloud, may increase frame rate
        self.__init_gui_signals()
        self.executor = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Capture-Save')
    
    def __init_gui_signals(self):
        self.recording = False  # Are we currently recording
        self.flag_record = False  # Request to start/stop recording
        self.flag_capture = False
        self.flag_normals = False
        self.flag_save_rgbd = False
        self.flag_save_pcd = False
        self.flag_model_init = False
        self.flag_exit = False
        self.color_mean = None
        self.color_std = None
        self.pcd_frame = None
        self.rgbd_frame = None

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
    
    def run(self):
        """Run pipeline."""
        n_pts = 0
        frame_id = 0
        t1 = time.perf_counter()

        if self.video:
            self.rgbd_frame = self.video.next_frame()
        else:
            self.rgbd_frame = self.camera.capture_frame(True)
            while self.rgbd_frame is None:
                time.sleep(0.01)
                try:
                    self.rgbd_frame = self.camera.capture_frame(True)
                except:
                    pass

        pcd_errors = 0
        while (not self.flag_exit and
               (self.video is None or  # Camera
                (self.video and not self.video.is_eof()))):  # Video
            if self.video:
                future_rgbd_frame = self.executor.submit(self.video.next_frame)
            else:
                future_rgbd_frame = self.executor.submit(
                    self.camera.capture_frame, True)
                    
            try:
                if self.rgbd_frame is None:
                    continue
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
                    log.warning(f"No valid depth data in frame {frame_id}")
                    continue
                
                depth_in_color = depth.colorize_depth(
                    self.depth_scale, 0, self.depth_max)

            except RuntimeError as e:
                pcd_errors += 1
                log.warning(f"Runtime error in frame {frame_id}: {e}")
                continue

            # if self.pcd_frame.is_empty():
            #     log.warning(f"No valid depth data in frame {frame_id}")
            #     continue


            frame_elements = {
                'color': color.cpu(),
                'depth': depth_in_color.cpu(),
                'pcd': self.pcd_frame.cpu().clone(),
                'camera': camera_line,
                'status_message': self.status_message
            }

            n_pts += self.pcd_frame.point.positions.shape[0]
            if frame_id % 30 == 0 and frame_id > 0:
                t0, t1 = t1, time.perf_counter()
            #     log.debug(f"\nframe_id = {frame_id}, \t {(t1-t0)*1000./60:0.2f}"
            #               f"ms/frame \t {(t1-t0)*1e9/n_pts:.2f} ms/Mp\t")
            #     n_pts = 0
                frame_elements['fps'] =  30 * (1.0 / (t1 - t0))

            if frame_id % 120 == 0:
                self.color_mean = np.mean(frame_elements['pcd'].point.colors.cpu().numpy(), axis=0).tolist()  # frame_elements['pcd'].point.colors.mean(dim=0)
                self.color_std = np.std(frame_elements['pcd'].point.colors.cpu().numpy(), axis=0).tolist()  # frame_elements['pcd'].point.colors.std(dim=0)
                log.debug(f"color_mean = {self.color_mean}, color_std = {self.color_std}")

            if self.flag_model_init:
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

        if self.camera:
            self.camera.disconnect()
        else:
            self.video.close()
        self.executor.shutdown(wait=True)  # Ensure all threads finish cleanly
        log.debug(f"create_from_depth_image() errors = {pcd_errors}")

    def toggle_record(self):
        """Toggle recording RGBD video.
        
        This function is called when the user clicks the "Record" button. It
        sets or unsets the flag to record RGBD video in the PipelineModel.
        """
        pass  # Recording functionality can be implemented if needed
    
    def model_intialization(self):
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
        
    def calibration(self):
        pass