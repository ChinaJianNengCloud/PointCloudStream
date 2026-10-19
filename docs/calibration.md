# 相机标定与标定板定位

标定模块解决两个相互关联的问题：估计相机的成像参数，以及利用已知尺寸的标定板计算每一帧的空间位姿。前者建立像素与空间射线的关系，后者为标定板跟随提供视觉输入。

当前主窗口实现的是相机标定与标定板 PnP 定位。机械臂与相机之间的外参标定没有在现有 `CalibrationData` 中实现。

## 核心实现

| 文件 | 相关对象与职责 |
| --- | --- |
| [calibration_data.py](../app/utils/calibration/calibration_data.py) | 主窗口使用的 `CalibrationData`；样本管理、内参求解、PnP 与保存 |
| [calib_callbacks.py](../app/callbacks/calib_callbacks.py) | 标定板初始化、采集、求解与参数载入 |
| [entry.py](../app/entry.py) | `board_detect()` 将 PnP 输出转为控制链中的 `Pose` |
| [camera_calibration.py](../app/utils/calibration/camera_calibration.py) | 独立相机标定窗口及另一份 `CalibrationData` |
| [calibration_workflow.py](../app/utils/calibration/calibration_workflow.py) | 保留的自动采集与旧手眼标定工作流 |
| [board_generator.py](../app/utils/calibration/board_generator.py) | 按物理尺寸生成 ChArUco 标定板 PDF |

## ChArUco 对应点

相机标定需要每幅图像中已知三维坐标与二维像素之间的对应关系。ChArUco 板将 ArUco 编号与棋盘交点结合，使检测结果可以按编号映射回板上坐标，而不依赖整张板完整出现。

初始化时，程序按字典类型、棋盘列数与行数、方格边长、标记边长构造 `cv2.aruco.CharucoBoard`，再创建 `CharucoDetector`。界面尺寸单位是毫米，构造标定板前除以 1000，因此板上三维点坐标采用米。

`board_dectect()` 执行以下处理：

1. 将输入图像转为灰度，记录 `(width, height)`。
2. 调用 `detector.detectBoard()` 获得 ChArUco 角点及编号。
3. 主窗口版本要求角点数量大于 4。
4. 调用 `board.matchImagePoints()`，建立当前图像的物点与像点数组。

每帧可以具有不同数量的对应点；`objpoints[i]` 与 `imgpoints[i]` 必须逐项对应。`append()` 只在检测通过后同时保存图像副本、物点、像点和可选机器人位姿，失败帧不进入样本集合。

独立 `camera_calibration.py` 中的接受条件是角点数量大于 5，与主窗口并不完全相同。主窗口采集回调只传入 `current_frame['control']`，没有传入机器人位姿，所以该路径的 `robot_poses` 元素为 `None`。

## 相机模型与内参求解

令板坐标中的点为 `P_board`，相机坐标中的点为 `P_cam`。每帧具有自己的外参：

```text
P_cam = R_cam_board · P_board + t_cam_board

K = [ fx   0  cx ]
    [  0  fy  cy ]
    [  0   0   1 ]

x = X_cam / Z_cam
y = Y_cam / Z_cam
```

畸变模型把归一化坐标 `(x, y)` 变成畸变后的 `(xd, yd)`，像素坐标再由 `u = fx·xd + cx`、`v = fy·yd + cy` 得到。仓库保存的五参数畸变数组对应常见的径向与切向参数形式。

`calibrate_camera()` 在至少有三幅有效图像时调用 `cv2.calibrateCamera()`，同时估计所有图像共享的 `K`、畸变参数和各帧外参。其目标是减小观测角点与模型投影之间的像素误差。内参中的焦距和主点均以像素为单位；图像分辨率变化会改变适用的内参。[OpenCV 相机标定说明](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

三幅图像是代码的启动门槛，不是精度保证。当前实现没有检查视角覆盖、姿态退化、模糊程度、图像尺寸一致性或单帧误差。`image_size` 在每次检测时被更新，混用不同分辨率的样本不会被自动拒绝。

`cv2.calibrateCamera()` 的第一个返回值是总体 RMS 重投影误差。代码将它命名为 `ret` 并作布尔判断，没有保存误差值或建立合格阈值；因此日志中的“成功”不能解释为达到某个测量精度。[OpenCV 返回值定义](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

## PnP 的变换方向

在已知 `K` 和畸变系数后，`board_pose_calculation()` 对每个样本调用 `cv2.solvePnP()`。实时检测路径 `draw_board()` 也使用同样的物点与像点求解。

PnP 返回的 `rvec`、`tvec` 将标定板坐标转换到相机坐标，即 `T_cam_board`。相机坐标约定为 X 向右、Y 向下、Z 向前。代码中的 `camera_to_board_rvecs` 名称不能据此解释为相反方向；需要 `T_board_cam` 时必须显式求逆。[OpenCV PnP 定义](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)

```text
T_cam_board = [ Rodrigues(rvec)  tvec ]
              [       0          1   ]

T_board_cam = inverse(T_cam_board)
```

`rvec` 是旋转向量，其方向为转轴，模长为旋转角，不能直接当作三个欧拉角。`tvec` 的长度单位跟随板上物点，当前主窗口为米。

批量 PnP 失败的样本先记录索引，再按逆序删除，以保持图像、物点和像点列表对应。实时路径在求解成功后绘制长度为 0.05 米的坐标轴，但没有配置帧间平滑或重投影误差门控。

### 控制链的数组顺序差异

`Pose.from_1d_array(..., vector_type='rotvec')` 接受 `[tx, ty, tz, rx, ry, rz]`。`SceneStreamer.board_detect()` 当前却拼接 `[rvec, tvec]`，因此平移和旋转向量在转换为 `Pose` 时被对调。

例如，PnP 求得平移 `[0, 0, 0.5]` 米、旋转向量 `[0.1, 0, 0]` 弧度时，控制链接收到的位置变成 `[0.1, 0, 0]`，旋转向量变成 `[0, 0, 0.5]`。图像上绘制的坐标轴直接使用 PnP 原始输出，可能仍显示正常，而下游跟随使用的 `Pose` 已具有不同含义。

因此，图像检测结果与控制链位姿需要分别核对，不能仅通过坐标轴叠加判断整条链路一致。

## 重投影误差的具体定义

`compute_reprojection_error()` 使用求得的每帧位姿将板上点重新投影，再计算：

```text
e_i = sqrt(sum_j ||pixel_ij - projected_ij||²) / n_i
E   = sum_i e_i / m
```

其中 `n_i` 为第 i 幅图像的点数，`m` 为图像数。这个实现先计算整幅图像的 L2 范数，再除以点数，不等于通常的逐点 RMS：

```text
RMS_i = sqrt(sum_j ||pixel_ij - projected_ij||² / n_i)
```

两者相差 `sqrt(n_i)` 的因子，比较不同可见角点数量的样本时必须保持定义一致。该方法还要求物点、像点和 PnP 结果非空且长度一致；主窗口的 `calibrate_all()` 并不会自动调用此误差计算。

## 标定结果与样本保存

`save_calibration_data()` 只写入两个字段：

```json
{
  "camera_matrix": [[1000.0, 0.0, 540.0], [0.0, 1000.0, 960.0], [0.0, 0.0, 1.0]],
  "dist_coeffs": [[0.0, 0.0, 0.0, 0.0, 0.0]]
}
```

上例用于说明结构，数值不是项目相机的标定结果。文件中没有图像尺寸、板参数、相机名称、旋转方式、误差统计或机器人外参，载入者无法仅靠该文件自动验证这些条件。

`save_img_and_pose()` 会删除并重建 `save_dir`，生成 `images/<index>.png` 与 `pose.txt`。当前 `pose.txt` 每行只写图像索引，不能作为机械臂六维位姿文件使用。

独立 `CameraCalibrationProcess.run()` 先把 JSON 写入 `Calibration_results`，随后调用样本保存；其默认 `save_dir` 也是这个目录。后一步的目录重建会删除前一步刚写入的标定 JSON。这是当前独立入口的保存顺序限制。

## 标定重置与历史接口

主窗口的初始化回调在已有 `CalibrationData` 时调用 `reset()`，然后替换 `board` 并赋值 `save_dir`。但 `save_dir` 只有 getter，赋值会抛出异常；替换 `board` 也没有同步重建 `detector`。异常分支将窗口中的 `calibration_data` 设为 `None`。

标定检查回调读取内参与畸变后，还要求 JSON 包含 `calibration_results` 及各方法的 `transformation_matrix`。当前保存器及仓库自带的标定结果没有这些字段，和该载入分支不兼容。选择矩阵时又调用了当前所用 `Pose` 类不存在的 `from_matrix()`。

旧自动工作流还依赖 `calibrate_hand_eye()`，部分界面操作依赖 `modify()`、`load_img_and_pose()` 和旧的 `current_frame['color']`。这些接口不在现有主窗口标定模型中。

## 与手眼标定的关系

相机内参描述成像关系，PnP 描述标定板相对相机的位姿。要将视觉位置直接转换为机器人基座坐标，还需要相机与机器人之间的空间关系。

例如，固定相机、标定板固定在末端时，应满足：

```text
T_base_cam · T_cam_board(i)
    = T_base_tool(i) · T_tool_board
```

其中两个固定变换需要根据多组配对观测求解。当前标定板跟随采用“两个原点之间的相对运动映射”，没有求解上述固定变换。它与相机内参标定、手眼标定是不同的技术环节。

## 物理标定板的尺寸约束

PDF 生成器接收以米为单位的方格和标记尺寸，按 DPI 换算图像像素，再按 PDF 的 72 点每英寸换算页面尺寸。它固定使用 `DICT_4X4_100`，与 GUI 中可选字典的范围不同。

物理打印尺寸、方格数量和字典必须与检测时一致。板尺寸比例错误会直接传递到 PnP 平移尺度；图像显示缩放不能补偿实际板尺寸错误。
