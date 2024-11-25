# main.py
import logging
import click
import json
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from pipeline import PipelineController

@click.command()
@click.option('--config', default='config.json', help='Path to configuration file')
def main(config):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    # Default configuration
    
    params = {
        'directory': '.',
        'Image_Amount': 13,
        'board_shape': (7, 10),
        'board_square_size': 23.5,
        'board_marker_size': 19,
        'input_method': 'auto_calibrated_mode',
        'folder_path': '_tmp',
        'pose_file_path': './poses.txt',
        'load_intrinsic': True,
        'calib_path': './Calibration_results/calibration_results.json',
        'device': 'cuda:0',
        'camera_config': './camera_config.json',
        'rgbd_video': None,
        'board_type': 'DICT_4X4_100',
        'data_path': './data',
        'load_in_startup': {
            'camera_init': True,
            'camera_calib_init': True,
            # 'robot_init': True,
            'handeye_calib_init': True,
            'calib_check': True,
            'collect_data_viewer': True
        },
        'use_fake_camera': True
    }
    import open3d as o3d
    import faulthandler
    faulthandler.enable()
    # # Load configuration from file if it exists
    # try:
    #     with open(config, 'r') as f:
    #         file_params = json.load(f)
    #         params.update(file_params)
    # except FileNotFoundError:
    #     logger.warning(f"Configuration file {config} not found. Using default parameters.")
    #     # Save default configuration
    #     with open(config, 'w') as f:
    #         json.dump(params, f, indent=4)
    # except json.JSONDecodeError:
    #     logger.error(f"Error parsing configuration file {config}. Using default parameters.")
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
    controller = PipelineController(params)

if __name__ == "__main__":
    main()