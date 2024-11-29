import sys
from vtk_pipeline.app import PCDStreamer
from PyQt5 import QtWidgets
import time
import logging
# from Callbacks import WindowCallbacks

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

def main():
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
        'data_path': '/home/capre/disk_4/yutao/data',
        'load_in_startup': {
            'camera_init': True,
            'camera_calib_init': True,
            'robot_init': True,
            'handeye_calib_init': True,
            'calib_check': True,
            'collect_data_viewer': True
        },
        'use_fake_camera': True
    }
    app = QtWidgets.QApplication(sys.argv)

    window = PCDStreamer(params=params)

    # def safe_exit():
    #     logger.info("Performing cleanup before exit...")
    #     if window.streaming:
    #         window.streaming = False
    #         if hasattr(window.streamer, 'camera'):
    #             window.streamer.camera.disconnect()
    #     window.streamer = None
    #     window.current_frame = None

    # app.aboutToQuit.connect(safe_exit)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()