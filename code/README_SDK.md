# stereo_sdk —— 双目深度相机 Python SDK

面向“我要在自己程序里拿到深度图和点云”的人。不需要读仓库里的推理代码，也不需要了解 Fast-FoundationStereo 的细节。

- 版本：**0.3.0**（唯一来源 `stereo_sdk/types.py` 的 `SDK_VERSION`）
- Python：3.10+（本项目实测 3.14.6）
- 字段全集与逐条签名：[docs/SDK_API参考.md](docs/SDK_API参考.md)
- 可直接运行的例子：[examples/](examples/)

---

## 1. 安装

在仓库根目录（含 `pyproject.toml` 那一层）执行。**顺序不能反**：先装 CUDA 版 PyTorch，再装其余依赖，否则会拿到默认 PyPI 的 CPU 版 torch。

```bash
# ① 先装 PyTorch（cu128 示例；按你的显卡与 CUDA 版本改）
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# ② 再装 SDK 与其余依赖
pip install -e .
pip install -r requirements.txt
```

`pip install -e .` 装两样东西：`stereo_sdk` 包，以及 `Session` 运行时依赖的根目录模块（`camera` / `pipeline_threaded` / `calibration` / `pointcloud` / `profiler` …，清单见 `pyproject.toml` 的 `py-modules`）。

**不要手动拷贝 `stereo_sdk/` 目录**，那样必然报 `ModuleNotFoundError: camera`。

只读标定信息、或只用 `MockSession` 跑界面的人，装完第 ② 步就够了（不需要 torch）。

---

## 2. 三行拿到深度

```python
from stereo_sdk import Session

with Session("config.yaml") as cam:      # __enter__ 里 start()，约 3~10 秒
    while True:
        if cam.has_new_result():
            r = cam.take_result()
            print(r.summary())           # seq / preset / infer 耗时 / 有效点 / 中位深度
            d = r.depth_at(320, 240)     # 某像素的深度（米），无效返回 0.0
```

**没有相机时**用 `MockSession`，接口完全一样：

```python
from stereo_sdk import MockSession
with MockSession() as cam:
    ...
```

---

## 3. 三条最容易写错的约定

| 约定 | 写错的后果 |
|---|---|
| `take_result()` **不是消费式的**：返回最新结果、重复调用得到同一对象。要逐帧处理必须用 `has_new_result()` 判断，或用 `r.seq` 去重 | 同一帧被处理 N 次，帧率虚高、下游重复入库 |
| `points` 的 **y 轴向下**（相机光学系），不是 OpenGL 的 y 向上 | 点云上下颠倒；若你自己翻转过一次，就是翻了两次 |
| `depth` 是全分辨率，`disparity` 是模型分辨率，**两者形状不同** | 用同一套像素坐标索引两者，取到错位的数据 |

其余约定：

| 项 | 约定 |
|---|---|
| 长度单位 | **米**（`depth`、`points`、`baseline`） |
| 图像通道 | `frame.left` / `frame.right` / `DepthResult.colors` 均为 **BGR**；写 PLY 时内部转 RGB |
| 无效像素 | `depth = 0`、`valid = False`，且不进入 `points` |
| 时间 | `time.perf_counter()` 秒 |
| 阻塞 | `start()` / `stop()` 阻塞；三个轮询方法全部非阻塞 |

---

## 4. 拿点云并存盘

```python
from stereo_sdk.export import export_result, save_ply

r = cam.take_result()
save_ply("cloud.ply", r.points, r.colors)             # 二进制 PLY，Open3D / MeshLab 直接开
export_result(r, "out/", prefix="seq001")             # 一整套：左右图 / 视差 / 深度 / 掩码 / 点云
```

`export_result` 返回的键名固定：`left / right / disparity_npy / disparity_png / depth_npy / depth_png / valid_png / cloud`（`with_tiff=True` 时多 `depth_tiff`）。

各格式的量化方式：

| 函数 | 格式 | 单位 / 量化 |
|---|---|---|
| `save_depth_png` | 16 位 PNG | 毫米，0 = 无效 |
| `save_depth_npy` | float32 `.npy` | 米，无损 |
| `save_depth_tiff` | 32 位 TIFF | 米（需 `pip install imagecodecs`） |
| `save_disparity_png` | 16 位 PNG | 像素 × 32 定点 |
| `save_ply` | PLY | 米，二进制（默认）/ ASCII |

---

## 5. 标定信息（决定深度能不能当真的东西）

```python
from stereo_sdk.calib import CalibInfo

c = CalibInfo.from_file()          # 自动查找，见下方顺序
print(c.summary())
print(c.baseline_mm, c.fx, c.rect_fx, c.rms_stereo)
print(c.sigma_depth(1.0))          # 1 m 处的深度不确定度（米）
print(c.usable_range(0.01))        # 要求误差 ≤1 cm 时，最远可用距离
```

**标定文件的查找顺序**（`pip install` 之后包被复制进 site-packages，“相对仓库根目录”的默认路径就不再指向你的标定文件了，所以按下面顺序找）：

1. 环境变量 `STEREO_CALIB` 指向的路径（最高优先）
2. 当前工作目录下的 `calibration/stereo_calib.yaml`
3. 包所在目录的上一级（`calibration/stereo_calib.yaml`，源码树直接运行时命中）
4. `~/.stereo/calibration/stereo_calib.yaml`

都不存在时抛 `FileNotFoundError`，**并把查过的每个路径打印出来**。

```bat
:: Windows 显式指定（只对当前命令行窗口有效）
set STEREO_CALIB=D:\my_project\calibration\stereo_calib.yaml
```

`CalibInfo` 是**只读**的，要改标定请用 `tools/calib_gui.py`。只读是刻意的：两处都能写，就会出现“哪份才是生效的”这种事故。

标定改完不必重启会话：

```python
cam.reload_calibration()     # 重读 YAML、替换校正器，几十毫秒，下一帧生效
```

---

## 6. 常用方法速查

| 方法 | 用途 |
|---|---|
| `cam.stats()` | 采集 / 推理 FPS、各阶段耗时、有效点比例、中位深度 |
| `cam.set_params(zmin=0.3, zmax=2.0)` | 热改深度范围；返回 `{"applied":..., "needs_restart":[...]}` |
| `cam.save_snapshot()` | 存一整套到 `output/captures/` |
| `cam.get_params()` | 当前参数快照 |
| `cam.last_error()` | 最近一次错误 |

`set_params` 中**必须重启**才生效的参数（`resolution` / `backend` / `max_disp`）不会被执行，只会出现在 `needs_restart` 里。改 `resolution` 后**必须重新标定**。

---

## 7. 常见错误

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: camera` | 只把 `stereo_sdk/` 拷走了 | 用 `pip install -e .`，别手拷目录 |
| `FileNotFoundError: stereo_calib.yaml` | 没做标定，或工作目录不对 | 跑 `tools/calib_gui.py` 做一次；或设 `STEREO_CALIB` |
| `RuntimeError: image size mismatch`（`reload_calibration`） | 新标定的分辨率与当前相机不同 | 换分辨率后必须重新标定 |
| 深度全是 0 | `valid` 全 False（弱纹理 / 超量程） | 看 `r.summary()` 的 valid 数，检查 `zmax` |
| `torch.compile is not supported on Python 3.14+` | 你的脚本先 `import torch` 再 `import stereo_sdk`，绕过了内部的补丁 | 导入 torch 前先 `set TORCHDYNAMO_DISABLE=1` |
| CUDA 不可用 | 装成 CPU 版 torch，或轮子不含 sm_120 内核 | 见 [docs/常见问题.md](docs/常见问题.md) 第 1 节 |

---

## 8. 版本与兼容

| 版本 | 新增 |
|---|---|
| 0.1.0 | `Session` 轮询接口（`latest_frame` / `has_new_result` / `take_result`）、`MockSession` |
| 0.3.0 | `with` 写法、`stereo_sdk.export`、`stereo_sdk.calib.CalibInfo`、`reload_calibration`、`Recorder` |

0.3.0 全部为**新增**，未改动 0.1.0 的既有签名。历史接口约定见 [docs/接口约定_SDK.md](docs/接口约定_SDK.md)。
