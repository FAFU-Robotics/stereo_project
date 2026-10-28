# 双目深度相机 SDK · 接口约定（v0.1.0）

> 这份文档是 **GUI 同学** 与 **后台同学（我）** 之间唯一的契约。
> 双方只通过 `stereo_sdk` 这一层交互；GUI **不要** import `camera.py` / `stereo_depth.py` /
> `depth_utils.py` / `pointcloud.py`（内部会重构，import 了就会崩）。

---

## 0. 一句话

```python
from stereo_sdk.mock import MockSession      # 开发期：不需要相机/GPU
from stereo_sdk import Session               # 联调期：换成这一行即可

s = Session("config.yaml"); s.start()
# 定时器里（建议 30 Hz）：
frame, seq = s.latest_frame()                # 最新一帧，永不阻塞
if s.has_new_result():
    r = s.take_result()                      # 最新结果，永不阻塞
```

**最关键的一条**：SDK **不提供**同步的 `infer()`。推理要 47~164 ms，
在 UI 线程里同步调用必然卡死。GUI 只能"取已经算好的"，不能"触发计算"。

---

## 1. 架构与职责

```
GUI（你的）                        stereo_sdk（我的）
  QTimer 30Hz  ──latest_frame()──▶  采集线程（51~59 FPS）
  QTimer 30Hz  ──take_result()──▶  推理线程（6~20 FPS，按预设）
  设置面板     ──set_params()───▶  点云/深度/着色
             ◀────stats()────────  统计
```

| 归属 | 内容 |
|---|---|
| **你的** | 窗口/布局/主题/面板/设置栏/快捷键/3D 渲染/图表 |
| **我的** | 采集、校正、推理、深度换算、点云、保存、性能统计、参数生效 |

**相机只能由一个进程打开**：`Session.start()` 内部有单实例锁；如果提示"另一实例正在使用相机"，
先把另一个程序（包括本项目的 CLI）关掉。

---

## 2. 数据契约（必须逐条遵守）

| 字段 | 类型 | 形状/单位 | 注意 |
|---|---|---|---|
| `frame.left` / `frame.right` | `uint8` | `(480, 640, 3)` | **BGR**，不是 RGB |
| `frame.seq` | int | 单调递增 | 判断"有没有新帧" |
| `frame.t_left/t_right` | float | 秒，`time.perf_counter()` 基 | 不是 wall clock |
| `r.disparity` | `float32` | `(h, w)` **模型分辨率**（0.75 档 = 360×480，0.5 档 = 240×320） | **不等于**图像尺寸 ⚠️ |
| `r.depth` | `float32` | `(480, 640)` 已上采样到全分辨率 | **米**，无效处 = 0 |
| `r.valid` | `bool` | `(480, 640)` | 判断有效性**只看它**，别用 `depth==0` 猜 |
| `r.points` | `float32` | `(N, 3)` | **米**，相机系，**y 向下** ⚠️ |
| `r.colors` | `uint8` | `(N, 3)` | **BGR**，与 points 一一对应 |
| `r.timings` | `dict[str,float]` | ms | 键固定：`capture/rectify/infer/cloud/display` |
| `r.preset` / `r.scale` | str / float | — | `"precise"/"balanced"/"fast"`，1.0/0.75/0.5 |
| `s.stats()` | `dict` | — | `capture_fps, infer_fps, valid_ratio, median_m, preset, frames` |

### 三个最容易踩的坑

1. **BGR ≠ RGB** → Qt 用 `QImage.Format_BGR888`（Qt ≥ 5.14）；用 `Format_RGB888` 会红蓝互换。
2. **点云 y 向下** → 3D 视图里直接把 y 取负再画，否则物体是倒着的（数据本身不改）。
3. **视差尺寸 ≠ 图像尺寸** → 显示视差图时要按自己的控件尺寸缩放，别假设和左右图一样大。

---

## 3. API 参考

```python
class Session:
    def __init__(self, config_path: str = "config.yaml") -> None

    # —— 生命周期（start 会阻塞 3~10 秒：开相机 + 加载 67MB 权重）——
    def start(self) -> None          # ⚠️ 必须在后台线程调用，否则主窗口会"未响应"
    def stop(self) -> None           # 幂等；退出时务必调用
    @property
    def started(self) -> bool

    # —— 取数据（永不阻塞，任何线程可调）——
    def latest_frame(self) -> tuple[FramePair | None, int]
    def has_new_result(self) -> bool
    def take_result(self) -> DepthResult | None      # 取过之后 has_new_result() 变 False

    # —— 控制（线程安全）——
    def set_params(self, **kw) -> dict   # {"applied": {...}, "requires_restart": [...]}
    def get_params(self) -> dict
    def pause(self) -> None              # 暂停"显示更新"用；推理仍在跑
    def resume(self) -> None
    def save_snapshot(self, tags: dict | None = None) -> str   # 返回落盘目录
    def stats(self) -> dict
    def last_error(self) -> str | None   # 后台异常只在这里出现，不会抛进 UI 线程
```

### 参数：哪些能热生效，哪些要重启

| 参数 | 热生效 | 说明 |
|---|---|---|
| `zmin` / `zmax` / `depth_range=(a,b)` | ✅ | 深度伪彩范围（米）|
| `point_size` / `voxel` | ✅ | 3D 显示用 |
| `preset="precise"\|"balanced"\|"fast"` | ✅ | 切推理档（内部重建 estimator，约 2 秒空档 ⚠️）|
| `paused` | ✅ | 暂停显示 |
| `resolution="1280x480"` | ❌ 需重启 | **改分辨率必须重新标定**（内参不匹配）⚠️⚠️ |
| `backend="msmf"\|"dshow"\|"any"` | ❌ 需重启 | 采集后端 |
| `max_disp=192` | ❌ 需重启 | 决定最近可测距离（192 → 约 0.19 m）|

> `set_params` 返回 `requires_restart` 非空时，请弹提示"需要重启生效"。

---

## 4. GUI 端参考实现骨架（PySide6）

```python
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow
import numpy as np

from stereo_sdk.mock import MockSession          # 联调时换 Session

def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    h, w, _ = bgr.shape
    return QImage(bgr.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
    # 注意 .copy()：否则 QImage 引用 numpy 缓冲，数组被复用时画面会花

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = QLabel(); self.setCentralWidget(self.label)
        self.sess = MockSession(); self.sess.start()      # 真机：Session("config.yaml").start()
        self.timer = QTimer(self); self.timer.timeout.connect(self.on_tick)
        self.timer.start(33)                              # 30 Hz

    def on_tick(self):
        frame, seq = self.sess.latest_frame()
        if frame is not None:
            self.label.setPixmap(QPixmap.fromImage(bgr_to_qimage(frame.left)))   # 实时画面
        if self.sess.has_new_result():
            r = self.sess.take_result()
            # r.disparity（模型分辨率）/ r.depth（全分辨率）/ r.valid / r.points / r.colors
            self.statusBar().showMessage(r.summary())
```

**3D 点云**：建议用 `QOpenGLWidget` 自绘（点用 `GL_POINTS`），数据直接吃 `r.points/r.colors`。
**不要**嵌入 Open3D 的窗口 —— 它会和 Qt 抢事件循环。

---

## 5. Do / Don't

| ✅ 应该 | ❌ 不要 |
|---|---|
| 用定时器轮询 `latest_frame()` / `take_result()` | 在 UI 线程调用任何耗时操作（`start()`、`save_snapshot()` 也建议后台）|
| 判断有效性用 `r.valid` | 用 `r.depth == 0` 猜 |
| 显示前按控件尺寸缩放 | 假设 `disparity.shape == depth.shape` |
| 退出时调 `stop()` | 直接 `sys.exit()` 不管后台线程 |
| 参数改动看 `requires_restart` | 假设所有参数都能热生效 |
| 自己画 3D | 往 Qt 里塞 Open3D 窗口 |

---

## 6. 联调步骤

| 阶段 | 做法 |
|---|---|
| 1. 纯 UI 开发 | `MockSession`（30 FPS 帧 / 6 FPS 结果，合成墙+方块，5% 无效像素）|
| 2. 真机联调 | `sess = Session("config.yaml")` 替换 `MockSession()`，其余代码不动 |
| 3. 压测 | `set_params(preset="fast")` → 结果 20 FPS；`preset="precise"` → 6 FPS |
| 4. 异常路径 | 拔相机 → `last_error()` 有消息、`started` 变 False；GUI 应能提示并重连 |

---

## 7. 需要 GUI 侧确认的 5 个问题

| # | 问题 | 影响 |
|---|---|---|
| 1 | 用 PySide6 还是 PyQt5？ | 图像转换 API 略不同 |
| 2 | 需要哪些面板？左右图 / 视差 / 深度伪彩 / 3D / FPS 曲线 / 参数栏 | 是否要我预生成伪彩图 |
| 3 | 3D 用 `QOpenGLWidget` 自绘？（推荐） | 性能与事件循环 |
| 4 | 是否要在**无独立显卡**的电脑上跑？ | 是 → 我加 SGBM 回退 |
| 5 | 需要录像/回放吗？ | 是 → 我加 `record()` / 回放源 |

---

## 8. 版本与变更

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.1.0 | 2026-10-22 | 首版：`types.py` + `mock.py` + 本文档 |
| 0.2.0 | 2026-10-24 | `Session.reload_calibration()`：换标定不重启相机、不重载模型 |
| 0.3.0 | 2026-10-24 | 新增 `export.py`（统一导出，二进制 PLY）、`calib.py`（只读 CalibInfo）、`Session/MockSession.__enter__`、`MockSession.reload_calibration`（与 Session 公共方法完全对齐）、`pyproject.toml` 可安装 |

**冻结规则**：字段名、单位、形状一旦冻结，只允许**新增**（向后兼容），不允许改名/改单位。
确需破坏性变更时，会在 `SDK_VERSION` 升 minor 并在本表记录。

## 9. v0.3.0 新增接口（向后兼容，全部为新增）

### 9.1 `with` 写法

```python
from stereo_sdk import Session
with Session("config.yaml") as cam:     # __enter__ 里 start()
    r = cam.take_result()               # __exit__ 里 stop()，异常也不漏
```

`__exit__` 返回 `None`，**不吞异常**——调用方的错误必须能冒出来。

### 9.2 `stereo_sdk.export`

| 函数 | 说明 |
|---|---|
| `save_ply(path, points, colors, binary=True)` | 二进制 PLY 默认；`colors` 为 **BGR**，写入时转 RGB |
| `save_depth_png(path, depth, valid)` | 16 位 PNG，毫米，0=无效 |
| `save_depth_npy` / `save_depth_tiff` | float32 米；TIFF 需 `imagecodecs` |
| `save_disparity_png(path, disp, fixed_scale=32)` | 16 位 PNG，像素 ×32 |
| `save_valid_mask` | 8 位 PNG |
| `export_result(result, out_dir, prefix)` | 一次导出整套，返回固定键名集合 |

二进制 PLY 的实测体积约为 ASCII 的 **0.39 倍**（500 点样例，写成 7.5 KB vs 19.7 KB），
点越多差距越大——ASCII 每点约 39 字节，二进制固定 15 字节。

### 9.3 `stereo_sdk.calib.CalibInfo`（只读）

```python
c = CalibInfo.from_file()      # 或 CalibInfo.latest("calibration/")
c.baseline_mm, c.fx, c.rect_fx, c.rms_stereo
c.sigma_depth(1.0)             # σz = z²/(f·B)·σd
c.usable_range(0.01)           # 误差 ≤1 cm 时的最远距离
c.summary(); c.check(); c.as_dict()
```

**刻意只读**：不提供 save。写标定只有 `tools/calib_gui.py` 一个入口，
避免出现"两处都能写、不知道哪份生效"的事故。

不依赖 OpenCV / torch，只读 YAML + numpy，可以放进 CI 做门禁。
