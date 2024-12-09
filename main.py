import sys
from app.main_app import PCDStreamer
from PyQt5 import QtWidgets
import time
import logging
# from Callbacks import WindowCallbacks
from app.utils.logger import setup_logger
logger = setup_logger(__name__)


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
        'yolo_model_path': '/home/capre/Point-Cloud-Stream/runs/segment/train6/weights/best.pt',
        'load_in_startup': {
            'camera_init': True,
            'camera_calib_init': True,
            'robot_init': True,
            'handeye_calib_init': True,
            'calib_check': True,
            'collect_data_viewer': True
        },
        'use_fake_camera': True,
        "service_type": "_agent._tcp.local.",
        "discovery_timeout": 2,
    }
    app = QtWidgets.QApplication(sys.argv)

    window = PCDStreamer(params=params)

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()