# stereo_project —— 双目立体视觉系统（DECXIN-3412V1 + Fast-FoundationStereo）

> **当前进度：阶段 1 —— 代码完成，但被相机问题阻塞（详见"5. 当前阻塞"）。**
> 深度/点云在未标定阶段产生的一切结果，都只用于流程验证，**不得用于测量**。

## 1. 环境（本机实测，见 [docs/阶段1_相机实测记录.md](docs/阶段1_相机实测记录.md)）

| 项目 | 值 |
|---|---|
| 操作系统 | Windows 11 25H2（Build 26200.9457） |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB，计算能力 **sm_120** |
| 驱动 | 610.88（CUDA UMD 13.3） |
| Python | **3.12.10** @ `D:\31931\Documents\python312\python.exe`（独立安装，**没有** conda/Miniforge） |
| PyTorch | **2.8.0+cu128**（实测 CUDA 可用、含 sm_120 内核） |
| OpenCV | opencv-contrib-python 5.0.0.93（DirectShow / Media Foundation 均可用） |
| Open3D | 0.20.0 |

> ⚠️ 系统 PATH 里的默认 `python` 是 **3.14.7**（`C:\Python314`），**不能**用。
> ⚠️ RTX 5060 是 sm_120，官方 readme 推荐的 `torch==2.6.0+cu124` **不含 sm_120 内核**，不能用。

### 运行前必设的环境变量

```bat
set TORCHDYNAMO_DISABLE=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set TEMP=D:\31931\Documents\_tmp
set TMP=D:\31931\Documents\_tmp
```

`TORCHDYNAMO_DISABLE=1` 的原因：Windows 版 PyTorch 不带 Triton，
而 FFS 源码里有 `@torch.compile`；不禁用则会编译失败。

## 2. 目录结构

```
stereo_project/
├── main.py                 程序入口（流程调度 / 启动 / 退出）
├── camera.py               双摄像头采集、设备探测、帧读取与释放
├── viewer.py               图像与状态显示（后续扩展深度图、点云窗口）
├── config.yaml             全部可调参数（设备索引、分辨率、深度范围、FFS 参数）
├── requirements.txt        依赖清单与版本说明（含"为什么必须 cu128"）
│
├── calibration.py          标定 / 去畸变 / 双目校正      （阶段 5 创建）
├── stereo_depth.py         FFS 模型加载 / 预处理 / 推理  （阶段 3 创建）
├── depth_utils.py          视差转深度 / 无效值 / 范围过滤（阶段 4 创建）
├── pointcloud.py           点云生成 / 过滤 / 下采样 / PLY（阶段 4 创建）
│
├── tools/
│   ├── probe_cameras.py        相机只读探测（逐配置独立子进程，防驱动挂死）
│   ├── preview_any.py          单路相机预览/自检（不需要配置文件）
│   ├── 相机探测.bat            双击即可跑完整探测
│   └── 相机单路预览.bat        双击 → 输入索引 → 看画面
│
├── docs/
│   └── 阶段1_相机实测记录.md   本阶段全部实测证据与结论
│
├── third_party/
│   └── Fast-FoundationStereo/  官方源码（**保持官方目录结构，不要改动**）
├── calibration/            标定结果存放
├── weights/                模型权重存放（不入库）
├── output/                 运行输出：快照/视差/深度/点云（不入库）
└── tests/                  测试脚本
```

## 3. 运行命令

```bat
cd /d D:\31931\Documents\stereo_project

REM ① 看设备与索引（不设分辨率/格式，最接近驱动默认行为）
python main.py --list

REM ② 完整探测：设备名、每个索引的可用分辨率/格式/帧率/是否黑屏（推荐）
python tools\probe_cameras.py
REM    或只探测某个索引：
python tools\probe_cameras.py --index 2

REM ③ 单路预览窗口（逐个索引试，找出哪一个是双目相机）
python tools\preview_any.py --index 2

REM ④ 双路实时预览（q/ESC 退出，p 暂停，s 保存，i 状态，h 帮助）
python main.py

REM ⑤ 无窗口自检（用于自动化验证，跑 4 帧就退出）
python main.py --no-display --max-frames 4
```

## 4. 阶段进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境 / 硬件 / 目录检查 | ✅ 完成 |
| 1 | 项目结构 + 双摄像头采集与显示 | ⚠️ 代码完成，**被相机阻塞**（见第 5 节） |
| 2 | 接入 FFS（官方示例图像验证） | ⏳ 需要权重 `23-36-37`（尚未下载） |
| 3 | 实时视差图 | ⏳ |
| 4 | 深度图 + Open3D 实时点云 | ⏳ |
| 5 | 标定 / 去畸变 / 双目校正 | ⏳ |
| 6 | 真实标定参数回归验证 | ⏳ |
| 7 | 性能优化、稳定性、界面 | ⏳ |

## 5. 当前阻塞（必须人工确认，代码无法代劳）

实测结论（证据见 `docs/阶段1_相机实测记录.md`）：

1. 本机 **只有 OpenCV 索引 2 能打开**（0/1/3~9 全部立即失败，dshow 与 msmf 一样）。
2. 索引 2 能打开，但**画面无有效信号**：640x480、mean≈2、std≈4、**帧率只有约 1 fps**。
3. 一旦设置 `MJPG`，所有分辨率都变成"读不到帧"。
4. Windows 当前真正在线的相机只有 2 个：`USB Camera`(VID_0BDA&PID_5883) 与
   `HP True Vision FHD Camera`；两个 Intel RealSense D405 是**历史残留**（Disconnected）。
5. 因此"DECXIN 是否双目相机、是不是左右拼接"**尚未确认** —— 640x480 是普通单目网络摄像头的尺寸，
   而左右拼接的典型尺寸是 1280x480 / 2560x720。

需要做的事：见 `docs/阶段1_相机实测记录.md` 的 **C1~C5**（用 Windows 相机 App 交叉验证、
拔插 A/B 测试确认设备身份、换机身 USB3 口直插、确认 USB 插头数量与镜头盖）。

## 6. 已知限制

1. **未标定**：没有内参、没有畸变参数、没有校正映射；基线 60 mm 只是厂家标称值。
   未校正图像产生的深度/点云只能验证流程，**不能测距、不能测尺寸、不能定位**。
2. **同步性**：`side_by_side` 模式左右来自同一幅图；`dual` 模式左右是两次独立 `read()`，
   实测时间差会打印在状态栏，**不是严格同步**。
3. **FFS 在 Windows 上必须关闭 `torch.compile`**（无 Triton），推理会明显慢于官方 3090 数据。
4. **OpenCV 5.0.0** 是较新的大版本。若后续确认相机问题由它引起，
   计划**新建独立环境**装 4.x 验证，**不覆盖**现有可用的 python312 环境（动手前会先征求同意）。
