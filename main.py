# main.py
import logging as log
import argparse

from pipeline_controller import PipelineController

if __name__ == "__main__":

    log.basicConfig(level=log.DEBUG)
    parser = argparse.ArgumentParser(
        description="Real-time 3D depth video processing pipeline adjusted for Azure Kinect camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera-config',
                        help='Azure Kinect camera configuration JSON file',
                        default='/home/capre/PCD/reconstruction_system/sensors/default_config.json')
    parser.add_argument('--rgbd-video', help='RGBD video file')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='Device to run computations. e.g. cpu:0 or cuda:0 '
                             'Default is CUDA GPU if available, else CPU.')

    args = parser.parse_args()
    if args.camera_config and args.rgbd_video:
        log.critical(
            "Please provide only one of --camera-config and --rgbd-video arguments"
        )
    else:
        PipelineController(args.camera_config, args.rgbd_video, args.device)
        