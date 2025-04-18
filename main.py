import sys
from app.entry import SceneStreamer
from PySide6 import QtWidgets
from app.utils.logger import setup_logger
logger = setup_logger(__name__)


def main():
    params = {
        'directory': '.',
        'Image_Amount': 13,
        'board_shape': (3, 5),
        'board_square_size': 23.5,
        'board_marker_size': 19,    
        'input_method': 'auto_calibrated_mode',
        'folder_path': '_tmp',
        'camera_list': [
            {
                'id': 6, 
                'name': 'main'
            },
            {
                'id': 0, 
                'name': 'wrist'
            },
            # {
            #     'id': 99, 
            #     'name': 'wrist', 
            #     'http_url': 'http://192.168.1.123:81/stream'
            # }
        ],
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

    window = SceneStreamer(params=params)

    window.show()
    sys.exit(app.exec())

  
if __name__ == "__main__":
    main()