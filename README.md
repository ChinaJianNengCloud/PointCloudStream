# Point Cloud Stream System with Open3D/VTK and PyQt UI

This project is a system for streaming and visualizing point cloud data using Open3D and VTK, with a user interface built using PyQt. It is designed for real-time applications and integrates key tools for 3D data handling and visualization.

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

Install other required Python packages:
```bash
pip install numpy==1.26
pip install open3d plyfile ipykernel pyqt5 opencv-python-headless zeroconf tqdm matplotlib lebai-sdk
pip install vtk
```

---

## Features
- There has a **data viewer** in *app.utils.data_management.data_viewer.py*, 
  Since it's a individual app for data checking, please modified it in the code to set data path for checking (press A for view last scene, D for Next, X for label invalid scene, Q for Save and Exit)

- **Point Cloud Streaming**: Utilizes Open3D and VTK for real-time processing and visualization of 3D point cloud data.
- **Customizable UI**: Built with PyQt5 for a responsive and interactive user interface.
- **CUDA Acceleration**: Leverages CUDA-enabled PyTorch for efficient computation on supported hardware.
- **Modular Design**: Easy to extend and integrate with additional tools or libraries as needed.

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

## Dependencies

The project relies on the following libraries:
- **Open3D**: For point cloud processing and visualization.
- **VTK**: An alternative or complement to Open3D for visualization tasks.
- **PyQt5**: Provides the graphical user interface.
- **PyTorch**: Used for CUDA acceleration and machine learning-related tasks.
- **Other Utilities**: Includes libraries such as NumPy, Matplotlib, and OpenCV for various support functions.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contribution

Contributions are welcome! Please follow the standard GitHub workflow:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

---

## Troubleshooting

- Ensure you have the correct version of Python (3.11).
- Confirm that your GPU drivers and CUDA version match the installed dependencies.
- Check the logs for errors and consult the respective library documentation.

---

This README provides a structured and detailed guide for setting up and working with your Point Cloud Stream System project. Let me know if you'd like to customize it further!