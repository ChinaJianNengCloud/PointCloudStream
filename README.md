# PCDStreamerUI

## Overview

**PCDStreamerUI** is a desktop application built using PyQt5 and VTK that provides an interactive GUI for visualizing and managing 3D point cloud data (PCD). It integrates with robotic systems, camera calibration tools, and data collection workflows, enabling users to perform tasks such as real-time point cloud rendering, robot-hand-eye calibration, and interactive data management.

## Key Features

- **VTK Integration:** Real-time rendering of 3D point clouds.
- **Interactive GUI:** User-friendly PyQt5 interface with customizable tabs and widgets.
- **Robot Integration:** Control and monitor robotic systems.
- **Camera Calibration:** Tools for board calibration with adjustable parameters.
- **Data Collection:** Interface for collecting, managing, and saving point cloud and RGBD data.
- **Agent Communication:** Connects to external servers for chatbot-based commands and logging.
- **Customizable Views:** Switch between camera view, bird's eye view, and more.
- **Data Viewer Tool:** Located in `app.utils.data_management.data_viewer.py`, this standalone viewer allows inspecting point cloud datasets. Use the following key bindings:
   - **A:** View previous scene
   - **D:** View next scene
   - **X:** Label invalid scene
   - **Q:** Save and Exit

---

## Environment Setup

### Create and Configure the Environment

1. Create a Conda environment with Python 3.11:
    ```bash
    yes | conda create -n o3d311 python=3.11
    conda activate o3d311
    ```

2. Install CUDA toolkit and dependencies for PyTorch:
    ```bash
    conda install cuda-toolkit=11.8 cuda-nvcc=11.8 cudnn=8.9 pytorch=2.2 torchvision=0.17.2 torchaudio=2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3. Alternatively, for a different CUDA version:
    ```bash
    cuda_version=12.1
    conda install cuda-toolkit=$cuda_version cuda-nvcc=$cuda_version cuda-compiler=$cuda_version cudnn=8.9 pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 pytorch-cuda=$cuda_version -c pytorch -c nvidia
    ```

### Install Additional Packages

- Install other required Python packages:
    ```bash
    pip install numpy==1.26
    pip install open3d plyfile ipykernel pyqt5 opencv-python-headless zeroconf tqdm matplotlib lebai-sdk
    pip install vtk
    ```

---

## Getting Started

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Activate the environment:
    ```bash
    conda activate o3d311
    ```

3. Run the application:
    ```bash
    python main.py
    ```

---

## UI Layout

### General Tab
- **Stream Init:** Choose between `Camera` or `Video` streaming.
- **Record/Save Controls:** Start and stop data recording.
- **Video Display:** Preview `Color` and `Depth` image streams.
- **Scene Info:** Display the current scene status.

### View Tab
- **Toggle View:** Enable/Disable different views and segmentation modes.
- **Acquisition Mode:** Switch modes for data acquisition.
- **Display Mode:** Choose between `Colors` and `Segmentation`.
- **Camera Controls:** Switch between `Camera view` and `Bird's eye view`.

### Calibration Tab
- **Calibration Settings:** Configure ArUco board type, square size, and marker size.
- **Calibration Image:** Preview detected calibration image.
- **Calibration Actions:** Collect frames, save calibration data, and check calibration results.

### Bbox Tab
- **Bounding Box Controls:** Define and manage bounding boxes for point cloud data.

### Data Tab
- **Data Collection:** Collect data frames and organize them into a tree structure.
- **Prompt Field:** Input custom text prompts for data tagging.
- **Data Folder:** Save collected data to a specific directory.

### Agent Tab
- **Server Connection:** Connect to an external agent server via IP and Port.
- **Chat Interface:** Interact with the server using a chat-like interface.
- **Prompt Editor:** Input commands and receive responses.

---

## Dependencies

The project relies on the following libraries:
- **Open3D:** For point cloud processing and visualization.
- **VTK:** An alternative or complement to Open3D for visualization tasks.
- **PyQt5:** Provides the graphical user interface.
- **PyTorch:** Used for CUDA acceleration and machine learning-related tasks.
- **Other Utilities:** Includes libraries such as NumPy, Matplotlib, and OpenCV for various support functions.

---

## Configuration

Edit the configuration in the `config.json` file:
```json
{
    "directory": ".",
    "Image_Amount": 13,
    "board_shape": [7, 10],
    "board_square_size": 23.5,
    "board_marker_size": 19,
    "input_method": "auto_calibrated_mode",
    "folder_path": "_tmp",
    "pose_file_path": "./poses.txt",
    "load_intrinsic": true,
    "calib_path": "./Calibration_results/calibration_results.json",
    "device": "cuda:0",
    "camera_config": "./camera_config.json",
    "rgbd_video": null,
    "board_type": "DICT_4X4_100",
    "data_path": "./data",
}

```

---

## Troubleshooting

- Ensure you have the correct version of Python (3.11).
- Confirm that your GPU drivers and CUDA version match the installed dependencies.
- Check the logs for errors and consult the respective library documentation.

---

## Contribution

Contributions are welcome! Please follow the standard GitHub workflow:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For support, contact [email@example.com](mailto:email@example.com) or open an issue in the repository.

---

Thank you for using **PCDStreamerUI**! Happy coding! 🚀

