# stereo_project —— 双目立体视觉系统（DECXIN-3412V1 + Fast-FoundationStereo）

> 当前进度：**阶段 1（项目骨架 + 双路采集与预览）**
> 深度/点云使用**未标定**参数时的所有结果，一律只用于流程验证，**不得用于测量**。

## 1. 环境（本机实测）

| 项目 | 值 |
|---|---|
| 操作系统 | Windows 11 25H2（Build 26200） |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB，计算能力 **sm_120** |
| 驱动 | 610.88（CUDA UMD 13.3） |
| Python | **3.12.10**，`D:\31931\Documents\python312\python.exe`（独立安装，非 conda） |
| PyTorch | **2.8.0+cu128**（已实测 CUDA 可用、含 sm_120 内核） |
| OpenCV | opencv-contrib-python 5.0.0.93 |
| Open3D | 0.20.0 |

> ⚠️ 系统 PATH 里的默认 `python` 是 **3.14.7**（`C:\Python314`），**不能**用来跑本项目。
> 所有命令都必须用 `D:\31931\Documents\python312\python.exe`，或先进入"FFS 命令窗口"。

### 运行前必设的环境变量（Windows 无 Triton，必须关掉 torch.compile）

```bat
set TORCHDYNAMO_DISABLE=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set TEMP=D:\31931\Documents\_tmp
set TMP=D:\31931\Documents\_tmp
```

## 2. 目录结构

```
stereo_project/
├── main.py                 程序入口（流程调度、启动、退出）
├── camera.py               双摄像头采集、设备探测、帧读取与释放
├── viewer.py               图像/状态显示（后续接深度图与点云窗口）
├── config.yaml             全部可调参数（设备索引、分辨率、深度范围、FFS 参数）
├── requirements.txt        依赖清单与版本说明
├── calibration.py          标定/去畸变/校正（阶段 5 创建）
├── stereo_depth.py         FFS 模型加载/预处理/推理（阶段 3 创建）
├── depth_utils.py          视差转深度、无效值处理、范围过滤（阶段 4 创建）
├── pointcloud.py           点云生成/过滤/下采样/PLY 导出（阶段 4 创建）
├── tools/
│   └── probe_cameras.py    相机只读探测（设备名/分辨率/帧率/左右拼接判定）
├── third_party/
│   └── Fast-FoundationStereo/   官方源码（**保持官方目录结构，不要改动**）
├── calibration/            标定结果存放
├── weights/                模型权重存放（不入库）
├── output/                 运行输出：快照、视差、深度、点云（不入库）
└── tests/                  测试脚本
```

## 3. 运行命令

```bat
cd /d D:\31931\Documents\stereo_project

REM 1) 看有哪些摄像头（注册表设备名 + 索引快速扫描）
python main.py --list

REM 2) 完整相机探测（推荐先跑，会生成 output\probe\report.json 和快照）
python tools\probe_cameras.py

REM 3) 实时预览左右两路画面（q 或 ESC 退出，p 暂停，s 保存快照）
python main.py

REM 4) 无窗口自动测试（跑 60 帧，保存一次快照）
python main.py --no-display --max-frames 60 --save-snapshot
```

## 4. 阶段进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境/硬件/目录检查 | ✅ 完成 |
| 1 | 项目结构 + 双摄像头采集与显示 | 🔄 进行中 |
| 2 | 接入 FFS（官方示例图像验证） | ⏳ 需要权重文件 |
| 3 | 实时视差图 | ⏳ |
| 4 | 深度图 + Open3D 点云 | ⏳ |
| 5 | 标定 / 去畸变 / 双目校正 | ⏳ |
| 6 | 真实标定参数回归验证 | ⏳ |
| 7 | 性能优化、稳定性、界面 | ⏳ |

## 5. 已知限制（必须知道，避免误用）

1. **未标定**：目前没有内参、没有畸变参数、没有校正映射，基线 60 mm 只是厂家标称值。
   由未校正图像生成的深度与点云**只能看流程是否通，不能测距离/尺寸/三维位置**。
2. **同步性**：若相机是"单设备左右拼接"输出，左右来自同一幅传感器数据；
   若是两个独立设备（dual 模式），两次 `read()` 之间存在毫秒级时间差，**不是严格同步**。
3. **FFS 在 Windows 上需关闭 `torch.compile`**（无 Triton），推理会明显变慢。
4. **OpenCV 5.0.0** 较新，后端按索引打开相机在本机表现不稳定，阶段 1 正在实测确认。
