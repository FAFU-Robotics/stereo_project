# stereo_project —— 双目立体视觉系统（DECXIN-3412V1 + Fast-FoundationStereo）

> **当前进度：阶段 0 / 1 / 2 已完成；下一步是阶段 3（实时视差）。**
> 深度/点云在未标定阶段产生的一切结果，都只用于流程验证，**不得用于测量**。

## ⚡ 最快的运行方式（双击即可）

```
D:\31931\Documents\stereo_project\tools\启动_管理员.bat
```

双击 → UAC 弹窗点 **"是"** → 自动提权并执行 OpenCV 自检 → 菜单选 `1` 进入**实时双目预览**。

> **为什么必须提权**：本机实测，**普通权限进程访问物理相机会被系统拒绝**
> （`HRESULT 0x80070005 / E_ACCESSDENIED`），而**管理员权限进程完全正常**。
> 这是"绕过"而不是"根治"，根因仍未完全解释 —— 完整证据链见
> [docs/阶段1_相机根因分析.md](docs/阶段1_相机根因分析.md) 第 12 节。

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
| 相机 | **DECXIN（索引 1）**：1280×480 左右拼接（每只眼 640×480），实测 **51.6 FPS** |

> ⚠️ 系统 PATH 里的默认 `python` 是 **3.14.7**（`C:\Python314`），**不能**用。
> ⚠️ RTX 5060 是 sm_120，官方 readme 推荐的 `torch==2.6.0+cu124` **不含 sm_120 内核**，不能用。

### 运行前必设的环境变量（启动器已自动设置）

```bat
set TORCHDYNAMO_DISABLE=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set TEMP=D:\31931\Documents\stereo_project\_work\pip_tmp
set TMP=D:\31931\Documents\stereo_project\_work\pip_tmp
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
│   ├── probe_cameras.py        相机只读探测（逐配置独立子进程，防驱动挂死）
│   ├── preview_any.py          单路相机预览/自检（不需要配置文件）
│   ├── fetch_ffs_weights.py    下载官方研究版权重（纯标准库，含 Drive 确认令牌处理）
│   ├── run_ffs_demo.py         官方示例图像验证（阶段 2 验收脚本，Windows 友好）
│   ├── 相机探测.bat            双击即可跑完整探测
│   ├── 相机单路预览.bat        双击 → 输入索引 → 看画面
│   └── 阶段2_FFS官方示例.bat   双击即可跑阶段 2 验证
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

> ⚠️ **所有涉及相机的命令都必须以管理员权限运行**（双击 `tools\启动_管理员.bat`，
> 或在管理员 cmd 里执行下面的命令）。原因见第 5 节。

### 阶段 1 / 3：相机预览与实时视差（需要管理员）

```bat
cd /d D:\31931\Documents\stereo_project

REM ★ 最省事：双击 tools\启动_管理员.bat，然后按菜单选
REM     [1] 实时双目预览        [2] 实时视差(FFS)     [3] 实时视差(更快预设)

REM ① 双路实时预览（q/ESC 退出，p 暂停，s 保存，i 状态，h 帮助）
python main.py

REM ② 实时视差（阶段 3）：左图 | 右图 | 视差伪彩 + 采集/推理/端到端 FPS
python main.py --mode depth

REM ③ 更快的视差预设（分辨率减半，约 2 倍速）
python main.py --mode depth --scale 0.5 --valid-iters 4

REM ④ 无窗口自检（跑 N 帧后自动退出，便于自动化验证）
python main.py --no-display --max-frames 4
python main.py --mode depth --no-display --max-frames 5
```

### 相机诊断工具（只读，不需要管理员；部分功能需要）

```bat
python main.py --list                                    REM 设备 + 索引快速扫描
python tools\probe_cameras.py --index 1                  REM 逐配置探测单个索引
python tools\preview_any.py --index 1                    REM 单路预览/自检
python tools\dshow_capture_test.py --index 1             REM 绕开 OpenCV 的 DirectShow 抓帧诊断
REM 或双击：tools\相机状态检查.bat / tools\相机修复_管理员.bat
```

### 阶段 2：FFS 官方示例验证（不需要相机）

```bat
cd /d D:\31931\Documents\stereo_project
set TORCHDYNAMO_DISABLE=1

REM ① 下载官方研究版权重（约 100 MB 级；也可用浏览器手动下载）
python tools\fetch_ffs_weights.py

REM ② 只检查权重与环境（秒回，不加载模型）
python tools\run_ffs_demo.py --check-only

REM ③ 正式跑官方示例图像（--repeat 3 统计稳定耗时）
python tools\run_ffs_demo.py --repeat 3

REM    其他常用：缩小输入提速 / 不看窗口 / 不生成点云
python tools\run_ffs_demo.py --scale 0.5 --repeat 5 --no-display
python tools\run_ffs_demo.py --valid-iters 4 --max-disp 192
```

## 4. 阶段进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境 / 硬件 / 目录检查 | ✅ 完成 |
| 1 | 项目结构 + 双摄像头采集与显示 | ✅ **完成**（实测 1280×480 / 51.6 FPS / 0 黑屏；需以管理员权限运行） |
| 2 | 接入 FFS（官方示例图像验证） | ✅ **完成**（5/5 通过；640×480 稳态 5.35~6.40 FPS，scale 0.5 时 10.8~14.1 FPS） |
| 3 | 实时视差图（相机 → FFS） | ⏳ **下一步** |
| 4 | 深度图 + Open3D 实时点云 | ⏳ |
| 5 | 标定 / 去畸变 / 双目校正 | ⏳ |
| 6 | 真实标定参数回归验证 | ⏳ |
| 7 | 性能优化、稳定性、界面 | ⏳ |

## 5. 相机问题的最终结论（已解决，但属"绕过"）

**现象**：普通权限进程打开物理相机 → `HRESULT 0x80070005 (E_ACCESSDENIED)`。

**解决**：以**管理员权限**运行即可完全正常（已实测：1280×480 真实画面、51.6 FPS、0 黑屏）。

| 谁在请求 | 结果 |
|---|---|
| 管理员权限进程（OpenCV 与 DirectShow 两条路径） | ✅ 正常 |
| 普通权限进程 | ❌ `E_ACCESSDENIED` |
| Windows 相机 App（UWP） | ✅ 正常 |

**已系统排除的原因**：OpenCV 版本（4.12 与 5.0 一致）、相机硬件/驱动（帧服务器多次 `hr=0x0`）、
USB/格式、隐私开关（Allow）、组策略、DSH 沙箱、进程完整性级别、锁屏、线程模型（STA/MTA）、
`qedit.dll`、厂商过滤驱动、HP 用户态进程 —— 每一项都有实测依据，
完整证据链见 [docs/阶段1_相机根因分析.md](docs/阶段1_相机根因分析.md)。

⚠️ **为什么普通权限被拒仍未完全解释**，因此这是**临时工作方式**而不是根治。
以管理员权限跑日常程序有安全代价；等阶段 3/4 跑通后可以再回头治本。

## 6. 已知限制

1. **必须提权运行**：普通权限进程无法访问物理相机（见第 5 节）。
   所有相机相关的程序都要通过 `tools\启动_管理员.bat` 启动。
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
