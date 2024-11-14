# main.py
import logging


from pipeline import PipelineController

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)

    params = {
        'directory': '.',  # Change to your directory if needed 
        'Image_Amount': 13,
        'board_shape': (11, 6),
        'board_square_size': 23, # mm
        'board_marker_size': 17.5, # mm
        'input_method': 'auto_calibrated_mode',  # 'capture', 'load_from_folder', or 'auto_calibrated_mode'
        'folder_path': '_tmp',  # Specify the folder path if using 'load_from_folder'
        'pose_file_path': './poses.txt',  # Specify the pose file path for 'auto_calibrated_mode'
        'load_intrinsic': True,  # Set to True or False
        'calib_path': './Calibration_results/calibration_results.json',  # Path to the intrinsic JSON file
        'device': 'cuda:0',
        'camera_config' : './default_config.json',
        'rgbd_video' : None,
        'board_type': 'DICT_4X4_100'
    }

    PipelineController(params)