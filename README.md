# stereo_project —— 双目立体视觉系统（DECXIN-3412V1 + Fast-FoundationStereo）

> **当前进度：阶段 0–7 已跑通**（采集 / 校正 / 推理 / 点云 / 标定 / 精度验证 / 性能），
> 标定 GUI 与 Python SDK 已就位；**打包成 EXE 尚未开始**，清单见
> [docs/打包成EXE_准备清单.md](docs/打包成EXE_准备清单.md)。

## ⚡ 最快的运行方式（双击即可，**不需要管理员权限**）

在本项目目录里双击：

```
tools\启动_普通权限.bat
```

双击 → 不弹 UAC → 菜单选 `1` 进入**实时双目预览**。
（早期文档里写死过 `D:\31931\...` 这类作者机器路径，已删；一律从项目根目录找 `tools\`。）

> **2026-10-21 复核结论**：普通用户权限（Medium 完整性）下，索引 1 + `CAP_DSHOW`
> 能稳定打开 1280×480 双目画面（左右各 640×480），实测 30 帧 0 失败、0 黑屏、正常释放。
> **不需要管理员权限**。旧文档里"必须提权"的说法已经过时，原因见
> [docs/README.md](docs/README.md)。
>
> ⚠️ 只有在"低完整性 / 沙箱 / 受限会话"里运行时，Windows 才会拒绝物理相机
> （`E_ACCESSDENIED`）。`tools\启动_管理员.bat` 仅保留作受限环境的诊断入口。
>
> **2026-10-22 记录（历史，针对当时的 Python 3.12 解释器）**：根因是解释器文件的
> **文件完整性标签为 Low**（继承自当时的 DSH 工作区）；执行
> `icacls "<解释器路径>" /setintegritylevel Medium` 后普通 PowerShell 即可打开相机。
> 当前机器换成了 Python 3.14 独立安装，普通权限实测可直接打开相机，无需该处理。

## 1. 环境（本机实测，见 [docs/README.md](docs/README.md)）

| 项目 | 值 |
|---|---|
| 操作系统 | Windows 11 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB，计算能力 **sm_120** |
| Python | **3.14.6** @ `C:\Users\never\AppData\Local\Python\pythoncore-3.14-64\python.exe`（独立安装，**没有** conda/Miniforge） |
| PyTorch | **2.9.1+cu128**（`torch.cuda.get_arch_list()` 含 sm_120，fp16 matmul/conv2d 实跑通过） |
| torchvision | 0.24.1+cu128 |
| OpenCV | opencv-contrib-python 5.0.0（DirectShow / Media Foundation 均可用） |
| Open3D | 0.20.0（`--mode pointcloud` 的 3D 窗口；`import open3d` 还要求 plotly + dash） |
| 相机 | **DECXIN（索引 1）**：1280×480 左右拼接（每只眼 640×480），实测采集 57–59 FPS |

> ⚠️ 系统 PATH 里的 `python` 是 WorkBuddy 自带的 **3.13.14**，**没有** torch/open3d，
> 直接用 `python main.py` 会加载失败。请用上表里的 3.14 全路径，或双击 `tools\*.bat`（已做解释器探测）。
> ⚠️ RTX 5060 是 sm_120，`torch==2.6.0+cu124` 这类旧轮子**不含 sm_120 内核**，不能用；
> 且 Python 3.14 只有 cu128 线的 2.9.0/2.9.1 提供 cp314 wheel。

### 运行前必设的环境变量（启动器已自动设置）

```bat
set TORCHDYNAMO_DISABLE=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set TEMP=%CD%\_work\pip_tmp
set TMP=%CD%\_work\pip_tmp
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
├── stereo_depth.py         FFS 模型加载 / 预处理 / 推理  （阶段 2 已创建）
├── depth_utils.py          视差转深度 / 无效值 / 范围过滤（阶段 4 创建）
├── pointcloud.py           点云生成 / 过滤 / 下采样 / PLY（阶段 4 创建）
│
├── tools/
probe_camera_backends.py  后端/分辨率探测（MSMF vs DSHOW、是否免提权可用）
│   ├── preview_any.py          单路相机预览/自检（不需要配置文件）
│   ├── opencv_camera_check.py  独立相机自检（逐索引/后端，打印是否管理员，仅作对照）
│   ├── test_pipeline_offline.py     离线自检（假相机跑三种模式，不需要相机/管理员）
│   ├── capture_calib.py        标定板可行性预检 / 后续的标定图采集（阶段 5）
│   ├── bench_ffs.py            FFS 性能基准（每档重置峰值显存）
│   ├── fetch_ffs_weights.py    下载官方研究版权重（纯标准库，含 Drive 确认令牌处理）
bench_pipeline.py            分阶段耗时基准（校正/推理/点云/显示 + 显存）
│   ├── 启动_普通权限.bat       ★主入口：普通权限 + 菜单（推荐）
│   ├── 启动_管理员.bat         仅用于低完整性/沙箱环境的诊断（普通使用不需要）
标定采集_新板7x9_26mm.bat   用 A4 ChArUco 板采集标定图（板参数在 .bat 顶部）
│   ├── 相机状态检查.bat        帧服务器/HP 组件/DirectShow 抓帧诊断
│   ├── 相机OpenCV自检.bat      普通权限自检（逐索引/后端）
bench_presets.py             三档预设的真实深度误差（已知距离，离线）
│
├── docs/
架构.md / README.md / 接口约定_SDK.md   模块地图、文档索引、GUI 契约
│
├── third_party/
│   └── Fast-FoundationStereo/  官方源码（**保持官方目录结构，不要改动**）
├── calibration/            标定结果存放
├── weights/                模型权重存放（不入库）
├── output/                 运行输出：快照/视差/深度/点云（不入库）
└── tests/                  测试脚本
```

## 3. 运行命令

> ✅ **普通 PowerShell / 普通 cmd 即可**，不需要管理员权限。
> 双击 `tools\启动_普通权限.bat` 最省事；也可以在普通终端里直接执行下面的命令。
> 只有在低完整性/沙箱会话里被拒绝时，才需要用 `tools\启动_管理员.bat` 做对照诊断。
>
> 📌 下文命令里的 `python` 指**第 1 节表格里那个 3.14 解释器**。如果 `where python`
> 不是它，请写全路径（或直接双击 `tools\*.bat`，那些 bat 已内置解释器探测）。

### 阶段 1 / 3：相机预览与实时视差（普通权限）

```bat
cd /d <项目根目录>

REM ★ 最省事：双击 tools\启动_普通权限.bat，然后按菜单选
REM     [1] 实时双目预览        [2] 实时视差(FFS)     [3] 实时点云

REM ① 双路实时预览（q/ESC 退出，p 暂停，s 保存，i 状态，h 帮助）
python main.py

REM ② 实时视差（阶段 3）：左图 | 右图 | 视差伪彩 + 采集/推理/端到端 FPS
python main.py --mode depth

REM ③ 更快的视差预设（分辨率减半，约 2 倍速）
python main.py --mode depth --scale 0.5 --valid-iters 4

REM ④ 实时点云（阶段 4）：OpenCV 窗口显示 左图|右图|深度(米)，Open3D 窗口显示彩色点云
python main.py --mode pointcloud
python main.py --mode pointcloud --scale 0.5 --valid-iters 4     REM 更快预设
python main.py --mode pointcloud --zmin 0.3 --zmax 2.0           REM 调整深度范围
python main.py --mode pointcloud --voxel 0.005                   REM 启用体素下采样

REM ★ 实时演示不会自己停止：只有按下 q / ESC（或 Ctrl+C）才会退出。
REM   关掉 Open3D 的 3D 窗口也只是收起 3D 视图，演示继续；按 o 可以重新打开。
REM   窗口内按键：q/ESC=退出  p/空格=暂停  s=保存  i=状态  h=帮助  o=重开 3D 窗口
REM
REM 关于点云的"上下颠倒"：我们按 **相机光学坐标系**（x 右、y 下、z 前）存储点云与深度，
REM   而 Open3D 的视图把 +y 当作"上"，所以直接显示会上下颠倒。
REM   处理方式：**只在 3D 窗口里把 y 翻过来显示**（窗口标题已注明），
REM   保存的 PLY / depth_meter.npy 等数据**仍是相机坐标系**，不做翻转。

REM ⑤ 无窗口自检（**仅用于自动化验证**：跑 N 帧后自动退出，不是演示模式）
REM    只有显式写 --max-frames / --duration 才会自动退出；演示菜单里都不带这两个参数
python main.py --no-display --max-frames 4
python main.py --mode depth --no-display --max-frames 5
python main.py --mode pointcloud --no-display --no-cloud-window --max-frames 3
```

### 离线自检（改代码后建议先跑这个：不需要相机，也不需要管理员）

```bat
python tests\test_pipeline_offline.py            REM 三种模式 x 清晰/模糊 + 三线程 + 新链路 共 12 个场景
python tests\test_pipeline_offline.py --frames 25
```

它用**假相机 + 假模型**把 `preview` / `depth` / `pointcloud` 三条实时链路完整跑一遍，
专门用来抓"某个模式里变量忘了初始化"这类只在特定场景（例如画面模糊）才暴露的崩溃。
> 为什么要有它：曾经因为 `run_pointcloud` 里漏初始化一个变量，导致画面模糊时崩溃退出
> （清晰时被 Python 的 `and` 短路掩盖），用户实测表现为"实时演示有时会突然停下"。

### 标定 GUI（图形界面：采集 → 标定 → 检查 → 保存）

换标定不用敲命令，双击或直接运行：

```bat
tools\calib_gui.bat
python tools\calib_gui.py
```

| 页 | 做什么 | 产出 |
|---|---|---|
| ① 采集 | 开相机、实时预览、按空格抓一对图；缩略图可翻看/删除 | `output\calib\<时间戳>\left\*.png` + `right\*.png` |
| ② 标定 | 选板型与参数，跑 `stereoCalibrate`，显示重投影 RMS 与每帧误差 | 内存中的标定结果 |
| ③ 检查校正效果 | 去畸变 + 极线行对齐预览，左右/闪烁对比 | 仅预览，不写文件 |
| ④ 保存参数 | 写入 `calibration\stereo_calib.yaml`（旧文件自动备份） | yaml |

快捷键：`空格` 保存当前帧　`F` 强制保存（不满足质量门也存）　`U` 撤销上一张。
焦点在输入框时快捷键自动失效，不会误触发。

保存后**不需要重启程序**：下一节 SDK 的 `reload_calibration()` 会在下一帧生效。
操作细节与判读标准见 [docs/标定GUI使用说明.md](docs/标定GUI使用说明.md)。

### 在别的程序里取深度图与点云（Python SDK）

`stereo_sdk` 是独立的接口层，不依赖 GUI。想当包用就先装一次（可选）：

```bat
python -m pip install -e .
```

最小示例（与 `examples\01_read_depth.py` 同逻辑）：

```python
from stereo_sdk import Session

with Session("config.yaml") as cam:        # __enter__ 里 start()，3~10 秒
    while True:
        if not cam.has_new_result():
            continue                        # 没有新帧就跳过
        r = cam.take_result()               # 永不阻塞，取最新一帧
        print(r.depth[240, 320])            # 画面中心深度（米）
        pts, cols = r.points, r.colors      # (N,3) 米 / (N,3) BGR
```

数据约定（按这个理解，否则数值会错 1000 倍或上下颠倒）：

| 项 | 约定 |
|---|---|
| 单位 | 米（`depth` 与 `points` 都是） |
| 坐标系 | 相机光学系：x 右、y **下**、z 前 |
| 颜色 | **BGR**，与 OpenCV 一致 |
| `take_result()` | 永不阻塞，始终返回最新结果，**可能是重复帧**；要逐帧处理必须先用 `has_new_result()` |
| `depth` 尺寸 | 与 `r.frame.left` 相同（已上采样到全分辨率） |
| 无效像素 | `depth = 0`、`valid = False`，且不进入 `points` |

导出成文件（二进制 PLY 比 ASCII 小约 60%）：

```python
from stereo_sdk import export
paths = export.export_result(r, "out", prefix="scan01")
# left/right png、disparity npy + 16bit png、depth npy + 16bit png(毫米)、valid png、cloud.ply
```

字段与接口全集见 [README_SDK.md](README_SDK.md)、[docs/接口约定_SDK.md](docs/接口约定_SDK.md)；
可直接运行的例子在 `examples\`：01 读深度 / 02 存点云 / 03 实时预览 / 04 标定报告。

### 阶段 5：标定（步骤 0 预检 ✅ 已完成 / 步骤 1 采集 ⏳ 待执行）

标定板：**DFOPTIX Q12-50-3** = 12×9 方格、方格 **3.0 mm**、图案 36×27 mm、内角点 **11×8**。

> **预检实测结论（2026-10-13）**：可用工作距离 **9~17 cm**
> （10 cm 每格 15.5 px / 清晰度 285；15 cm 每格 10.1 px / 清晰度 441；18 cm 以上每格 <8 px 检不到）。
> 好消息：**近距离并不糊**，限制因素是"图案在画面里的像素数"，不是对焦。

```bat
REM ① 验证工具本身（不需要相机）：合成棋盘格跑一遍分析逻辑
python tools\capture_calib.py --selftest

REM ② 可行性预检（普通权限即可；已完成，一般不用再跑）
python tools\capture_calib.py --probe

REM ③ 正式采集标定图（普通权限即可）
REM    工作距离 9~17 cm；空格=记录  u=撤销  f=强制记录  q=结束并写报告
python tools\capture_calib.py --capture
```

**采集清单**（界面实时显示进度，配置在 `config.yaml` 的 `calibration.checklist`）：

| 类别 | 张数 | 为什么必须 |
|---|---|---|
| 中心正对 | 2 | 基准 |
| **画面四角各** | 2 | **畸变系数必须靠边缘数据**才能解出来 |
| **左倾/右倾/上仰/下俯 各**（≥20°） | 3 | **倾斜是解算焦距的关键**（本方案距离跨度窄，靠倾斜补偿） |
| 合计目标 | **26** | |

**采集时的记录**：`output\calib\<时间戳>\left_XX.png / right_XX.png / overlay_XX.png` +
`capture_report.txt`（每张的检出/每格像素/清晰度/对比度/区域/倾角 + 清单完成情况）。

预检的判据（在 `capture_calib.py` 顶部常量里，可调）：

| 指标 | 良好 | 可用下限 | 不达标的含义 |
|---|---|---|---|
| 每格像素 `px_per_square` | ≥ 15 | ≥ 10 | 图案太小 → 角点定位误差大 |
| 清晰度（拉普拉斯方差） | ≥ 100 | ≥ 30 | 太糊 → 角点与立体匹配都不可靠 |
| 图案**最差区域**对比度 | ≥ 180 | ≥ 120 | 反光把黑格洗白（玻璃基板常见） |

> 说明：判据是"**最差区域**"的对比度，不是全图平均 —— 局部反光只影响一块时，
> 全图指标看不出来（这一点是自检用合成光斑测出来的）。

### 相机诊断工具（只读，普通权限即可）

```bat
python main.py --list                                    REM 设备 + 索引快速扫描
python tools\opencv_camera_check.py --quick              REM 快速全扫索引 0~9（一行一个结果）
python tools\opencv_camera_check.py                      REM 详细自检（0~2、两种后端、含是否管理员）
python tools\probe_camera_backends.py 逐配置探测单个索引
python tools\preview_any.py --index 1                    REM 单路预览/自检
python tools\dshow_capture_test.py --index 1             REM 绕开 OpenCV 的 DirectShow 抓帧诊断
REM 或双击：tools\相机状态检查.bat（帧服务器 / HP 组件 / DirectShow 抓帧诊断）
REM          tools\相机OpenCV自检.bat --quick（普通权限快速全扫）
```

### 阶段 2：FFS 官方示例验证（不需要相机）

```bat
cd /d <项目根目录>
set TORCHDYNAMO_DISABLE=1

REM ① 下载官方研究版权重（约 100 MB 级；也可用浏览器手动下载）
python tools\fetch_ffs_weights.py

REM ② 只检查权重与环境（秒回，不加载模型）
bench_pipeline.py            分阶段耗时基准（校正/推理/点云/显示 + 显存）

REM ③ 正式跑官方示例图像（--repeat 3 统计稳定耗时）
bench_pipeline.py            分阶段耗时基准（校正/推理/点云/显示 + 显存）

REM    其他常用：缩小输入提速 / 不看窗口 / 不生成点云
bench_pipeline.py            分阶段耗时基准（校正/推理/点云/显示 + 显存）
bench_pipeline.py            分阶段耗时基准（校正/推理/点云/显示 + 显存）
```

## 4. 阶段进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境 / 硬件 / 目录检查 | ✅ 完成 |
| 1 | 项目结构 + 双摄像头采集与显示 | ✅ **完成**（普通权限实测 1280×480 / ~59 FPS / 0 黑屏） |
| 2 | 接入 FFS（官方示例图像验证） | ✅ **完成**（5/5 通过；640×480 稳态 5.35~6.40 FPS，scale 0.5 时 10.8~14.1 FPS） |
| 3 | 实时视差图（相机 → FFS） | ✅ **完成**（4.81 FPS @640×480；16.38 FPS @scale 0.5；左右眼顺序已核实） |
| 4 | 深度图 + Open3D 实时点云 | ✅ **完成**（180 433 点；深度公式与官方逐像素一致） |
| 5 | 标定 / 去畸变 / 双目校正 | ✅ **完成**（双目 RMS **0.1246 px**、校正后极线失准 **0.07 px**、实测基线 **64.891 mm**） |
| 6 | 真实标定参数回归验证 | ⏳ **下一步**（已知距离 0.5/1.0/1.5 m 的测距误差，目标 <5%） |
| 7 | 性能优化、稳定性、界面 | ⏳ |

### 阶段 4 的验证状态（诚实说明）

| 验证项 | 结果 |
|---|---|
| 深度公式 vs 官方实现 | ✅ **最大绝对差 0.00000000 m**（完全一致） |
| 点云反投影 vs 官方 cloud.ply | ✅ 点数 518400 相同、z 范围 `0.3458..1.4210 m` 完全相同 |
| PLY 导出可被 Open3D 读回 | ✅ 点数/颜色完全一致 |
| 未标定警告写入 PLY 文件头 | ✅ `comment CALIBRATION_STATUS: UNCALIBRATED ...` |
| 深度范围过滤 / 无效值处理 | ✅ 实测 76373 个过近点被 `zmin` 正确剔除 |
| **实机实时点云窗口** | ⚠️ **待用户以普通权限运行确认**（本机采集侧已实测通过） |

## 5. 相机权限问题的最终结论（2026-10-22：`python.exe` 的 Low 文件标签）

> 本节保留当时的原始记录，**其中的 `D:\31931\...python312` 路径属于那台机器**；
> 在自己机器上套用时，把路径换成实际使用的解释器（用 `where python` 或 `python app_entry.py --where` 查）。

> 本节保留当时的原始记录，**其中的 `D://31931//...python312` 路径属于那台机器**；
> 在自己机器上套用时，把路径换成你实际使用的解释器（用 `where python` 或 `app_entry.py --where` 查）。

**最终修复（2026-10-22）**：`icacls` 显示 `D:\31931\Documents\python312\python.exe`
的标签是 `Mandatory Label\Low Mandatory Level:(I)(NW)`（从 DSH 工作区继承）。
Low 进程访问物理相机时，在 DirectShow `BindToObject` 阶段被 `E_ACCESSDENIED` 拒绝。
执行：

```powershell
icacls "D:\31931\Documents\python312\python.exe" /setintegritylevel Medium
```

后，普通 PowerShell 里 Python 令牌变为 `S-1-16-8192`（Medium），索引 1 可正常输出
1280×480；**不需要管理员**。（原取证文档已不在库中，结论保留在本节。）

**旧结论（2026-09-30，已过时）**：普通权限进程打开物理相机 → `HRESULT 0x80070005 (E_ACCESSDENIED)`，
当时用管理员权限绕过。

**2026-10-21 复核**：

| 进程令牌 / 场景 | DirectShow `BindToObject` | 项目 `main.py` 实际采集 |
|---|---|---|
| **普通用户 Medium 完整性**（真实普通 PowerShell） | 索引 0/1 均 ✅ 成功 | ✅ 索引 1 打开 1280×480，30 帧 0 失败、0 黑屏、正常释放 |
| `python.exe` 文件标签为 Low / 沙箱 / 受限会话 | 索引 0/1 ❌ `E_ACCESSDENIED` | ❌ 打不开物理相机 |
| 索引 2（HP 虚拟摄像头） | ✅ 能绑定 | ⚠️ `isOpened()=True`、`read()=True`，但画面全黑（640×480，mean≈0） |

**结论**：

1. **普通用户权限完全够用**，不需要管理员；根因是 `python.exe` 的文件完整性标签
   被 DSH 工作区继承成了 **Low**，用 `icacls ... /setintegritylevel Medium` 修复。
2. 索引 2 的成功只能证明**虚拟摄像头**能打开，不能证明双目相机可用；
   它的 `read()=True` 返回的是黑帧，不能作为验收依据。
3. 索引 1 仍是 DECXIN 双目相机；DSHOW 仍是正确后端（MSMF 在本机按索引打不开，普通/管理员都一样）。
4. `camera.py` 已加入：打开时传参、DSHOW→MSMF→ANY 回退、尺寸校验、重复设备检测、
   失败记录与安全释放；不再把任何失败都硬编码成"需要管理员"。
5. **如果再次打不开**：先查 `icacls "D:\31931\Documents\python312\python.exe" | findstr /I "Mandatory"`；
   若又是 `Low`，重新执行 `icacls ... /setintegritylevel Medium`。
   重装/升级 Python 或 DSH 重新给工作区打标签后，标签可能恢复成 Low。

> 历史取证（2026-09-30）的原始文档已不在库中；其中的"必须管理员"已被本次复核取代。

## 6. 已知限制

1. **不需要提权**：普通权限即可访问相机（见第 5 节）。如果打不开，先查
   `D:\31931\Documents\python312\python.exe` 的完整性标签：
   ```powershell
   icacls "D:\31931\Documents\python312\python.exe" | findstr /I "Mandatory"
   ```
   若是 `Low`：
   ```powershell
   icacls "D:\31931\Documents\python312\python.exe" /setintegritylevel Medium
   ```
   然后重开 PowerShell。不要长期以管理员运行。
2. **未标定**：没有内参、没有畸变参数、没有校正映射；基线 60 mm 只是厂家标称值。
   未校正图像产生的深度/点云只能验证流程，**不能测距、不能测尺寸、不能定位**。
   （好消息：实测左右眼垂直位移 `dy = 0.26 px ≈ 0`，说明该模块**近似已校正**。）
3. **同步性**：`side_by_side` 模式左右来自同一幅图（同一传感器输出），时间上天然同步；
   但"左右是否真的同曝光"仍需向厂家确认。
4. **FFS 在 Windows 上必须关闭 `torch.compile`**（无 Triton）。实测稳态：
   640×480 → 5.35~6.40 FPS；scale 0.5（320×240）→ 10.8~14.1 FPS。
   采集侧（51.6 FPS）完全够用，**瓶颈在 FFS 推理**，实时化需要进一步加速（阶段 7）。

## 7. 网络与下载注意事项（本机实测，很重要）

实测结果：

| 通道 | HTTPS | 说明 |
|---|---|---|
| PowerShell / .NET (`Invoke-WebRequest`) | ❌ 全部失败 | `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS` |
| git (`git clone/ls-remote`) | ❌ 失败 | 同上，git for Windows 默认用 schannel |
| **Python 自带 urllib / pip** | ✅ **正常** | 用 OpenSSL 3.0.16，aliyun / GitHub raw / HuggingFace / Google Drive 全部 200 |

因此本项目的下载动作**统一走 Python**：

```bat
REM pip 走 Python 的 TLS 通道（pip.ini 里已配阿里云镜像）
python -m pip install <包名>

REM git 若必须用，可临时切到 OpenSSL 后端（只对本次命令生效，不改全局配置）
git -c http.sslBackend=openssl clone <仓库地址>
```

Google Drive 大文件会插一层"确认下载"页面，`tools/fetch_ffs_weights.py` 已按表单原样
重建请求处理该情况（实测漏掉 `id/export` 字段会返回 HTTP 400）。
