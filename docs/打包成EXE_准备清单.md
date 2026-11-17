# 打包成 EXE 的准备清单

> **状态：准备工作完成，打包命令未执行。**
> 本文里的体积数字，标「实测」的是本机量出来的，标「估算」的是按已装包体积推算的，
> 两者不混用。凡未实测的结论都会写明。

---

## 0. 一句话结论

代码层已经具备"能被打包"的条件：路径不再依赖 `__file__` 推断、有一个统一入口、
无控制台时的异常可见、首次运行会把配置播种到可写目录。
**还缺的只有三件决策/实测**：选哪个档位（见 §3）、用干净 venv 构建、torch 的 DLL 收集要实测一次。

### 已经实测通过的（2026-10-27）

| 项 | 证据 |
|---|---|
| torch 2.9.1+cu128 在 Python 3.14 / sm_120 上可用 | `cuda.is_available()=True`，fp16 matmul + conv2d 真跑通 |
| FFS 权重能加载、能推理 | 加载 2.2 s，推理 **86–104 ms/帧**，显存 785 MB |
| 真实图像端到端（校正→推理→深度→点云） | 26 对标定图抽 3 组，深度中位数 28.6/44.4/65.0 cm，点云 307 200 点 |
| 标定 GUI 四页能起 | `--selftest` 退出码 0；环境信息面板 16 行路径全部正确 |
| 既有 SDK 测试 | `tests/test_smoke.py` 8/8 |
| 统一入口 | `--where` / `--selftest` / `--gui-selftest`（主界面）/ `--calib-selftest`（标定工作台）全部退出码 0 |

### 还没验证的

| 项 | 原因 |
|---|---|
| **相机真机采集** | 当前 `dshow` 索引 1 打不开（`main.py --list` 只有索引 0 内置摄像头可用）。可能是相机未插、被占用、或索引变了。**这是硬件状态，不是代码问题** |
| PyInstaller 实际产物 | 按你的要求"打包先不要开始做"，未执行 |

---

## 1. 本次为打包补上的东西

改造前，这个项目**打包出来必然是坏的**——不是"可能要调"，是"一定跑不起来"。
下面每一项都对应一个具体的崩法。

| 文件 | 改动 | 不这样做会怎样 |
|---|---|---|
| `app_paths.py` **（新增）** | 区分「只读资源根」与「可写数据根」，三态识别：源码 / PyInstaller onefile / onedir | `output/`、`calibration/` 会被写进 `_MEIPASS` 临时目录，**程序一关数据就没了**，用户以为标定失败 |
| `app_entry.py` **（新增）** | 唯一入口；`TORCHDYNAMO_DISABLE` 在第一条语句设置；`stdout` 为 `None` 时改写到日志；未捕获异常落盘 + 弹窗 | ① Windows 无 Triton，`torch.compile` 直接报错；② `--noconsole` 下任何 `print()` 抛 `AttributeError`，界面白屏且无提示；③ 崩了之后用户什么都拿不到，无法反馈 |
| `tools/calib_core.py` | `PROJECT_ROOT` 改为 `data_root()`；`config.yaml` 改用 `find()` | 采集图写进临时目录；找不到 config 时静默退回默认标定板参数（12×9/3mm），而用户可能用的是别的板 |
| `tools/calib_manage.py` | `PROJECT_ROOT` 改为 `data_root()` | 标定文件写到只读目录 → `PermissionError`，或写进临时目录后丢失 |
| `tools/calib_gui.py` | 同上；另加 `report_callback_exception` 兜底、顶栏「环境信息」按钮 | 界面回调里的异常静默消失，表现为"点了没反应" |
| `stereo_sdk/session.py` | `_resolve_paths()` 按「只读/可写」分别解析 | `ffs_repo`、`weights` 指向 `_MEIPASS` 里不存在的位置 |
| `stereo_depth.py` | 权重改用 `resource()` 查找，不再靠 `repo_root/../..` 反推 | 目录层级一变就找不到 67.8 MB 权重；用户也无法把权重放在 exe 旁边自行替换 |
| `main.py` | `PROJECT_ROOT` 与 `load_config()` 同上 | CLI 模式（`--mode depth`）路径全错 |
| `runtime.py` | `PROJECT_ROOT` 改为 `data_root()` | 同上 |
| `pyproject.toml` | `py-modules` 补 `app_paths`/`app_entry`；新增 `[project.scripts] stereo-app` | `pip install` 后 `import stereo_sdk` 会因缺 `app_paths` 报 `ModuleNotFoundError` |

改造原则与之前一致：**业务逻辑一行没动**，改的都是路径取值方式；
每个文件都保留了 `try/except ImportError` 兜底，单独拷走 `tools/` 时行为与改造前完全相同。

---

## 2. 环境事实（2026-10-27 本机实测）

| 项 | 值 | 来源 |
|---|---|---|
| Python | 3.14.6（独立安装，无 conda；实际路径用 `where python` 查） | 实测 |
| GPU | RTX 5060 Laptop 8 GB，sm_120，驱动 591.74 | `nvidia-smi` |
| torch | **2.9.1+cu128**（`cp314` wheel 存在） | cu128 索引实测 |
| torchvision | **0.24.1+cu128** | cu128 索引实测 |
| PyInstaller | 6.22.3，`requires_python <3.16,>=3.8` → **支持 3.14** | PyPI 元数据 |
| cv2 | 5.0.0.93，目录 113 MB | 实测 |

**关于 torch 版本的三条硬约束**（这是选 2.9.1 而不是更新的 2.14 的原因）：

1. RTX 5060 是 Blackwell **sm_120**，cu124 及更早的预编译内核里没有它 → 必须 cu128 或更新。
2. `requirements.txt` 原写 `torch==2.8.0+cu128`，但 cu128 索引里 **2.8.0 只有 cp39–cp313，没有 cp314**。
   Python 3.14 上 cu128 线只有 **2.9.0 / 2.9.1** 带 cp314。
3. cu130 索引有 2.14.1（cp314），但 FFS 官方代码是对 torch 2.6 写的，
   跨 5 个大版本升级带来的 API 风险远大于收益。**先按最接近原设计的 2.9.1 走。**

### 2.1 安装后实测（2026-10-27）

| 检查项 | 结果 |
|---|---|
| `torch.__version__` | 2.9.1+cu128 |
| `torch.cuda.is_available()` | **True** |
| 设备 | NVIDIA GeForce RTX 5060 Laptop GPU，capability **sm_120** |
| 编译支持的架构 | `sm_70 / 75 / 80 / 86 / 90 / 100 / 120` — **含 sm_120** |
| fp16 矩阵乘（2048²） | 通过，175.9 ms |
| conv2d（32×64×64） | 通过 |
| 权重加载 `torch.load(weights_only=False)` | 通过，2.2–2.6 s |
| 单帧推理（320×240，iters=4） | **86–104 ms**（≈10–11 FPS），第 1 帧 1.7–2.4 s 预热 |
| 显存峰值 | **785 MB** |

**结论：这台机器上「FFS 推理 + CUDA」是真的能跑的**，不再是"只在 mock 上验证过"。

### 2.2 一个必须记录的平台坑：`torch.compile` 在 Python 3.14 上是硬错误

这不是配置问题，是 torch 主动拒绝：

```python
# torch/__init__.py:2590 (2.9.1)
raise RuntimeError("torch.compile is not supported on Python 3.14+")
```

而 FFS 的 `core/submodule.py:376` 是 `@torch.compile` —— **装饰器在 import 阶段求值**，
所以不处理的话 `import core.foundation_stereo` 直接抛异常，整条推理链路在 3.14 上不可用。
`TORCHDYNAMO_DISABLE=1` 管不住它（那个开关决定"是否真去编译"，而这里是版本检查）。

处理方式（已写入 `stereo_depth.py`，在 `import torch` 之后）：把 `torch.compile`
换成恒等装饰器。语义等价——在 3.12/3.13 + `TORCHDYNAMO_DISABLE=1` 下，
它本来也只是把函数包一层原样执行，FFS 没有任何地方依赖图语义。

---

## 3. 两个档位（需要你选一个）

| | **档 A：瘦版（推荐先做）** | 档 B：全量版 |
|---|---|---|
| 包含 | 标定 GUI 四页 + 环境信息 | 档 A + `--mode preview/depth/pointcloud` |
| 依赖 | cv2 + numpy + tkinter + pyyaml | 再加 torch、torchvision、timm、imageio、open3d |
| 需要 GPU | 不需要 | 需要（CUDA 驱动必须由宿主提供，**打不进包**） |
| 体积 | 约 250–400 MB（估算） | 约 3 GB 起（估算） |
| 打包难度 | 低：`--exclude-module torch` 即可 | 高：torch 的 DLL 收集、`core`/`Utils` 命名冲突都要实测 |
| 适合谁 | 只做标定的现场人员 | 端到端交付 |

**为什么推荐先做档 A**：它不依赖任何未验证的打包行为（torch hook、CUDA 运行时、DLL 搜索路径），
一次成功率最高；而且标定 GUI 本身不需要 GPU，这是实打实的优势。
档 B 可以在 A 验通之后增量加，不用重来。

---

## 4. PyInstaller 参数（草稿，尚未执行）

### 4.1 档 A 命令

```bat
pyinstaller --noconfirm --clean ^
  --name StereoCalib ^
  --windowed --onedir ^
  --paths . --paths tools ^
  --add-data "config.yaml;." ^
  --add-data "calibration;calibration" ^
  --hidden-import calib_gui ^
  --hidden-import calib_core ^
  --hidden-import calib_check ^
  --hidden-import calib_manage ^
  --hidden-import gui_state ^
  --hidden-import gui_theme ^
  --hidden-import app_paths ^
  --exclude-module torch ^
  --exclude-module torchvision ^
  --exclude-module timm ^
  --exclude-module open3d ^
  --exclude-module scipy ^
  --exclude-module skimage ^
  --exclude-module imageio ^
  --exclude-module pandas ^
  --exclude-module matplotlib ^
  app_entry.py
```

几个参数的理由，不是随手加的：

- **`--onedir` 而不是 `--onefile`**：onefile 每次启动都要把 200 MB 解压到临时目录，
  冷启动通常 3–10 秒（估算），而且 `_MEIPASS` 每次都是新路径，不利于用户找日志。
  onedir 冷启动 1–2 秒。
- **`--exclude-module torch` 不能省**：`session.py` 里 `from stereo_depth import ...`
  写在 `start()` 内部（函数级 import），但 PyInstaller 的静态分析照样会跟进去，
  而 `stereo_depth.py` 顶层就是 `import torch`。不排除的话，一个标定 GUI 会被拖进 2–3 GB。
- **`--add-data "config.yaml;."`**：`.py` 会被编译进 PYZ，但**数据文件不会**。
  config 缺失时 `calib_core.spec_from_config()` 会静默退回默认板参数，这种错误不报错、只出错，最难查。
- **`--paths tools`**：`calib_gui.py` 里是 `import calib_core`（平铺），不是 `import tools.calib_core`。

### 4.2 档 B 额外要加的

```bat
  --add-data "third_party/Fast-FoundationStereo;third_party/Fast-FoundationStereo" ^
  --add-data "weights;weights" ^
  --hidden-import Utils ^
  --hidden-import core.foundation_stereo ^
  --hidden-import core.extractor ^
  --hidden-import timm ^
  --hidden-import imageio ^
```

这里有两个**已知风险点，必须实测**，不能假设能过：

1. **`third_party` 必须以原始 `.py` 文件形式打进去（`--add-data`），不能只靠 PYZ。**
   因为 `stereo_depth._ensure_ffs_on_path()` 是把 `repo_root` 插进 `sys.path`，
   再 `import core.foundation_stereo` / `import Utils`。这是**运行时路径查找**，
   只有磁盘上真有这些 `.py` 文件才成立。
2. **`core` 与 `Utils` 是极通用的名字**，容易和其他包撞名。
   如果实测出现 "import 到了别的 core"，解法是给 FFS 目录加 `--paths` 并显式
   `--hidden-import`，或者在 `_ensure_ffs_on_path` 里把 repo_root 提到 `sys.path[0]`（现有代码已是 `insert(0)`）。

### 4.3 构建环境

**建议用干净 venv 构建，不要在现在这套全局环境上直接打。**
现在的全局环境里装着 opencv-python 5.0.0.93 / numpy 2.5.4 / torch（本次安装）等，
全局环境越杂，PyInstaller 误收的模块越多，产物越大、越容易带进无关 DLL。

```bat
py -3.14 -m venv .venv-build
.venv-build\Scripts\activate
pip install pyinstaller pyyaml opencv-python numpy
pyinstaller ...   :: 上面的参数
```

---

## 5. 体积账

| 项 | 数值 | 性质 |
|---|---|---|
| cv2 目录 | 113 MB | 实测 |
| numpy 目录 | 35 MB | 实测 |
| PIL (pillow 12.3.0) | 16 MB | 实测 |
| CPython 运行时 + 标准库 | 30–40 MB | 估算 |
| tkinter + Tcl/Tk | 8–12 MB | 估算 |
| **档 A 合计** | **210–320 MB** | 估算（大部分为实测） |
| 权重 `model_best_bp2_serialize.pth` | 67.8 MB | 实测 |
| `third_party/Fast-FoundationStereo` | 38 MB | 实测（含 demo_data 测试图，可裁） |
| **torch 2.9.1+cu128** | **4.3 GB**（其中 `torch/lib` 单独 4.2 GB） | 实测 |
| torchvision 0.24.1+cu128 | 29 MB | 实测 |
| timm / imageio / omegaconf | 17 / 2.7 / 2.6 MB | 实测 |
| sympy（torch 依赖） | 77 MB | 实测 |
| open3d 0.20.0（含 plotly/dash 连带） | 225 MB | 实测 |
| **档 B 合计** | **5.1 GB 以上** | 实测依赖 5.0 GB + 项目 106 MB |

> 体积不是"打包方式"决定的，而是"选了哪些依赖"决定的。
> 删 `examples/` 一个字节都省不下来（那四个文件合计 15 KB）。
> **torch 的 4.3 GB 几乎全是 CUDA 运行库**（`torch/lib` 一个目录就 4.2 GB），
> 这是档 B 无法压缩的下限——除非改用 CPU 版 torch（`torch-2.9.1+cpu`，约 200 MB），
> 代价是推理从 86 ms 掉到数秒级。

---

## 6. 验收清单（打包后必须逐条过）

1. 在一台**没装 Python**的机器上双击能起，不弹缺 DLL
2. `--where` 输出里，`可写数据根` 是 exe 所在目录，不是 `AppData\Local\Temp\_MEIxxxx`
3. 四页走一遍：采集 → 标定 → 检查 → 保存，`calibration/stereo_calib.yaml` 能写回
4. 关掉程序再开，`output/` 里的采集图还在（验证可写根正确）
5. 点「环境信息」能复制出一份完整路径文本
6. 故意拔掉相机点「打开相机」→ 有中文提示，且**程序不崩**
7. 导出的 PLY 能被 CloudCompare 打开，坐标量级 0.1–3（米），不是 200–3000

---

## 7. 还没解决的（不在本次范围）

| 项 | 说明 |
|---|---|
| **Linux / AppImage** | `camera.py` 的后端字典只有 `dshow` / `msmf` / `any`，**没有 `CAP_V4L2`**。同一个程序搬到 Linux 上相机根本打不开。要出 AppImage 必须先加 v4l2 后端，并因为分辨率绑定而**重新标定** |
| SDK 不能被包进 EXE | EXE 是给不写代码的人用的启动方式，SDK 是给写代码的人用的库，两者互斥。要两者兼得必须同时产出「EXE 安装包 + `pip install` 的 wheel」 |
| 代码签名 | PyInstaller 产物被 Defender / 360 误报是常见现象。正式分发通常需要代码签名证书，本次未涉及 |
| onefile 冷启动 | 未实测。若最终选 onefile，需实测冷启动耗时后再决定 |
