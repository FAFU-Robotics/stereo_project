# stereo_project · 双目深度相机系统

一套在 Windows 上可直接运行的双目深度相机方案：USB 双目相机采集 → 双目标定校正 → 神经网络立体匹配 → 深度图与彩色点云，附带标定 GUI、Python SDK 与打包好的 EXE。

- **采集**：DECXIN 双目模块（UVC 免驱），1280×480（左右各 640×480），实测 51.6 FPS
- **推理**：Fast-FoundationStereo（NVIDIA 研究版权重），三档预设可切换
- **输出**：深度图（米）、有效掩码、相机坐标系彩色点云（PLY）、可导出 npy / PNG / TIFF
- **交付形态**：`StereoCamera.exe`（免装 Python）、`stereo_sdk`（pip 可装）、`tools/calib_gui.py`（标定工作台）

> **使用限制先说清楚**：内置的立体匹配模型与权重来自 NVIDIA Fast-FoundationStereo，许可为 **NVIDIA Source Code License**，只允许**研究 / 评估**用途，不可商用。本项目自有代码为 MIT。边界与依据见 [NOTICE.md](NOTICE.md)。

---

> English overview: [README_EN.md](README_EN.md)　·　English docs: [docs/en/](docs/en/)

## 快速开始

### 路线 A：只想用起来（不需要装 Python）

1. 打开本仓库的 **Releases** 页面，下载分卷包（`StereoCamera-v0.3.0.7z.001` / `.002` / `.003`），全部下载后放在同一目录。
2. 用 7-Zip 右键第一卷 → **提取到当前位置**，得到 `StereoCamera` 文件夹。**不要只拷 exe。**
3. 确认机器有 NVIDIA 显卡与较新驱动（需支持 CUDA 12.8），接上双目相机。
4. 双击 `StereoCamera.exe` → 打开主界面 Stereo Viewer → 点右上角「开始串流」→ 在 `2D` 标签勾选「深度」或切到 `3D` 看点云。
5. 界面右上角的 `界面语言` 可在 **中文 / English** 之间即时切换；也可用 `--lang en` 启动，或改 `config.yaml` 的 `ui.language`。

完整说明：[docs/软件使用说明.md](docs/软件使用说明.md)

### 路线 B：从源码运行（要改代码、写 SDK 程序）

```bat
git clone <本仓库地址>
cd stereo_project

REM 1) 先装 PyTorch（顺序不能反：默认 PyPI 给的是 CPU 版）
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

REM 2) 再装其余依赖（Python 3.10+；本项目在 3.14.6 上实测）
python -m pip install -r requirements.txt

REM 3) 下载模型权重（67.8 MiB，纯标准库脚本）
python tools\fetch_ffs_weights.py

REM 4) 环境自检（不需要相机）
python app_entry.py --selftest

REM 5) 跑起来
python app_entry.py               REM 主界面（设备 / 深度 / 点云 / 采集 / 标定）
python app_entry.py --gui         REM 标定工作台（四页流程）
python main.py                    REM 命令行实时预览
python main.py --mode depth       REM 命令行实时视差
python main.py --mode pointcloud  REM 命令行实时点云
```

逐步说明与排错：[docs/安装指南.md](docs/安装指南.md)

> **PyTorch 版本是这套系统最容易踩的坑。** RTX 50 系（sm_120）必须用 cu128 及以上轮子；旧版 cu124 轮子不含 sm_120 内核，装上会报 “no kernel image is available”。Python 3.14 目前只有 2.9.x 的 cu128 线提供 cp314 轮子。

---

## 性能摘要

以下为实机端到端数据（RTX 5060 Laptop 8 GB，1280×480 输入，三线程模式，2026-10-27 实测）：

| 预设 | scale / iters | 采集 FPS | 校正+推理 | 点云 | **端到端 FPS** |
|---|---|---|---|---|---|
| 精确 | 1.00 / 8 | 49.7 | 176 ms | 11 ms | **3.92** |
| 平衡 | 0.75 / 8 | 50.5 | 114 ms | 10 ms | **6.38** |
| 快速 | 0.50 / 4 | 56.8 | 62 ms | 4 ms | **14.33** |

- 推理占单帧耗时的 **82%**（1.0 档），是唯一瓶颈；采集余量约 8 倍
- 离线基准（不含窗口开销）串行帧率为 5.01 / 8.90 / 18.22 FPS
- 显存占用 0.73 ~ 2.48 GB（随档位）

完整方法、误差表与复现命令：[docs/性能测试报告.md](docs/性能测试报告.md)

## 精度摘要

| 指标 | 实测值 |
|---|---|
| 双目重投影 RMS | **0.1246 px**（左 0.073 / 右 0.153） |
| 校正后极线行对齐误差 | **0.07 px** |
| 实测基线 | **64.891 mm** |
| 校正后焦距 fx = fy | 547.017 px |
| 1 m 处深度不确定度（视差噪声 0.5 px） | **14.09 mm**（相对误差 1.41%） |
| 要求误差 ≤10 mm 时的最远可用距离 | **0.84 m** |

> 深度不确定度由 `σz = z²·σd / (f·B)` 估算，`tools/bench_presets.py` 与 `examples/04_calib_report.py` 都会打印这张表。这是**噪声模型推算值**，不是实测精度；实测精度的复核方法见 [docs/阶段5_标定实测记录.md](docs/阶段5_标定实测记录.md)。

---

## 五个可用的入口

| 入口 | 命令 | 说明 |
|---|---|---|
| **主界面 Stereo Viewer**（默认） | `python app_entry.py` 或双击 EXE | 设备状态、彩色/深度/点云切换、快照、标定入口 |
| 标定工作台 | `python app_entry.py --gui` | 采集 → 标定 → 检查校正效果 → 保存，四页式 GUI |
| 实时预览 | `python main.py` | 左右图并排，验证相机与曝光 |
| 实时深度 | `python main.py --mode depth` | 左图 \| 右图 \| 视差伪彩，带三路 FPS |
| 实时点云 | `python main.py --mode pointcloud` | OpenCV 深度窗口 + Open3D 彩色点云 |

实时窗口按键：`q/ESC` 退出、`p/空格` 暂停、`s` 保存、`i` 状态、`h` 帮助、`o` 重开 3D 窗口。

## Python SDK

```python
from stereo_sdk import Session

with Session("config.yaml") as cam:      # 启动约 3~10 s（含加载权重）
    while True:
        if not cam.has_new_result():
            continue                     # 没有新结果就跳过，不阻塞
        r = cam.take_result()            # 永不阻塞，取最新一帧
        print(r.summary())               # seq / preset / infer 耗时 / 有效点 / 中位深度
        print(r.depth_at(320, 240))      # 指定像素的深度（米），无效返回 0.0
```

没有相机时可换成 `MockSession`，接口完全一致：

```python
from stereo_sdk import MockSession
with MockSession() as cam:
    ...
```

数据约定（不遵守会得到差 1000 倍或上下颠倒的结果）：

| 项 | 约定 |
|---|---|
| 长度单位 | **米**（`depth`、`points`、`baseline` 一致） |
| 坐标系 | 相机光学系：x 右、**y 下**、z 前 |
| 颜色 | `DepthResult.colors` 为 **BGR**；写 PLY 时内部转成 RGB |
| `depth` 尺寸 | 已上采样到全分辨率（与 `frame.left` 同形） |
| `disparity` 尺寸 | 模型分辨率（`scale=0.75` 时是 480×360），**与 `depth` 不同形** |
| 无效像素 | `depth = 0`、`valid = False`，且不进入 `points` |

六个可直接运行的示例：读深度 / 存点云 / 实时点选测距 / 标定报告 / 质量统计 / 采数据集，见 [examples/](examples/)；字段全集见 [docs/SDK_API参考.md](docs/SDK_API参考.md)。

---

## 目录结构

```
stereo_project/
├── main.py / app_entry.py      两个入口：实时链路、统一 CLI（含 --selftest / --where）
├── camera.py                   采集：设备探测、后端回退、尺寸校验、安全释放
├── calibration.py              标定、去畸变、双目校正
├── stereo_depth.py             FFS 模型加载与推理
├── depth_utils.py / pointcloud.py   视差转深度、过滤、点云生成与 PLY
├── stereo_sdk/                 Python SDK（Session / MockSession / export / calib / recorder）
├── tools/                      命令行工具、标定 GUI、启动器 .bat
├── examples/                   可直接运行的 SDK 示例
├── tests/                      离线自检（假相机 + 假模型，不需要硬件）
├── docs/                       安装、使用、架构、API、性能、标定记录
├── packaging/                  PyInstaller 配置与打包脚本
├── third_party/                Fast-FoundationStereo 官方源码（勿改，许可见 NOTICE.md）
└── weights/ calibration/ output/ logs/   运行时数据（不入库）
```

各目录职责与“哪些不入库、为什么”见 [docs/仓库结构说明.md](docs/仓库结构说明.md)。

## 文档导航

| 想做什么 | 看这份 |
|---|---|
| 装起来跑通 | [docs/安装指南.md](docs/安装指南.md) |
| 用 EXE 或界面 | [docs/软件使用说明.md](docs/软件使用说明.md) |
| 换标定参数 | [docs/标定GUI使用说明.md](docs/标定GUI使用说明.md) |
| 在自己程序里取深度 | [docs/SDK_API参考.md](docs/SDK_API参考.md) · [examples/README.md](examples/README.md) |
| 了解模块划分与数据流 | [docs/架构.md](docs/架构.md) |
| 看性能数据与复现方法 | [docs/性能测试报告.md](docs/性能测试报告.md) |
| 发布 EXE 给别人 | [docs/发布与分发.md](docs/发布与分发.md) |
| 出错排查 | [docs/常见问题.md](docs/常见问题.md) |
| 文档总索引 | [docs/README.md](docs/README.md) |

## 系统要求

| 项 | 最低 | 推荐 |
|---|---|---|
| 操作系统 | Windows 10 64 位 | Windows 11 |
| 显卡 | 支持 CUDA 12.8 轮子的 NVIDIA 显卡 | 本项目在 RTX 5060 Laptop（sm_120）上实测 |
| 显存 | 2 GB（快速档） | 8 GB（精确档峰值 2.48 GB） |
| Python（源码路线） | 3.10（`pyproject.toml` 下限） | 3.14.6（本项目实测版本） |
| 驱动 | 支持 CUDA 12.8 的版本 | 最新 Game Ready / Studio |
| 运行库 | Visual C++ 2015–2022 x64 | 同左 |
| 相机 | UVC 双目模块 | 本项目验证机型 DECXIN-3412V1 |

- 不需要 CUDA Toolkit，不需要管理员权限，不需要 conda。
- 没有 NVIDIA 显卡时：预览、标定、SDK 接口仍可用，**推理与点云不可用**。

## 已知限制

1. **标定与分辨率绑定**。标定按 1280×480 完成，改分辨率必须重新标定，否则深度数值无意义。
2. **模型只允许研究用途**（见 [NOTICE.md](NOTICE.md)）。
3. **推理是瓶颈**。精确档 3.92 FPS，不适合高速运动场景；需要实时性请用快速档（14.33 FPS）。
4. **单机验证**。EXE 只在一台已装开发环境的机器上测过；**没装 Python 的干净机器未验证**，见 [packaging/打包测试报告.md](packaging/打包测试报告.md)。
5. **左右同曝光未向厂家确认**。`side_by_side` 模式下左右来自同一帧，时间同步，但曝光是否独立未验证。

## 许可与引用

- 自有代码与文档：**MIT**，见 [LICENSE](LICENSE)。
- `third_party/Fast-FoundationStereo` 与模型权重：**NVIDIA Source Code License，仅研究/评估用途**，见 [NOTICE.md](NOTICE.md)。
- 引用上游工作时请使用 [NOTICE.md](NOTICE.md) 第 5 节的 BibTeX 条目。

## 贡献

问题与改动建议请开 Issue。提交前请先跑离线自检（不需要相机与显卡，本机实测 7 秒跑完 12 个场景）：

```bat
python tests\test_pipeline_offline.py
```

它会用假相机 + 假模型把 `preview` / `depth` / `pointcloud` 三条链路各跑一遍，专抓“某个分支才暴露的崩溃”。
