# SDK API 参考

本文档逐条对应 `stereo_sdk/` 的源码实现（v0.3.0）。凡本文档与源码不一致，以源码为准并请提 Issue。

- 包名：`stereo_sdk`
- 版本：`stereo_sdk.types.SDK_VERSION = "0.3.0"`（唯一版本来源）
- 依赖：`numpy` / `opencv-python` / `pyyaml`。**torch 不在必需依赖里**，只有真正 `Session.start()` 跑推理时才需要。

---

## 1. 顶层导出

```python
from stereo_sdk import (
    Session, MockSession, get_session,     # 会话
    CalibInfo,                              # 标定只读摘要
    export_result,                          # 一次性导出全套文件
    FramePair, DepthResult,                 # 数据类型
    SDK_VERSION, TIMING_KEYS,
    HOT_PARAMS, RESTART_PARAMS,
    check_frame, check_result,              # 断言辅助
)
```

`Session` / `MockSession` / `CalibInfo` / `export_result` 采用**惰性导入**：只有真正访问时才加载 torch 等重依赖，因此 `import stereo_sdk` 本身很快。

---

## 2. 会话：`Session` 与 `MockSession`

两者接口一致，可互换。

### 2.1 构造与生命周期

```python
Session(config_path: str = "config.yaml")
MockSession(config_path: str = "", width: int = 640, height: int = 480,
            frame_fps: float = 30.0, result_fps: float = 6.0)
```

| 方法 | 阻塞 | 说明 |
|---|---|---|
| `start()` | 是，约 3~10 s | 打开相机、加载权重、启动推理线程 |
| `stop()` | 是 | 释放相机与线程 |
| `started() -> bool` | 否 | 是否已启动 |
| `__enter__` / `__exit__` | — | 支持 `with`，退出时自动 `stop()` |

`MockSession` 的 `frame_fps` 是合成画面帧率，`result_fps` 是合成结果帧率；它的 `infer` 耗时是写死的固定值，**不代表真实性能**。

### 2.2 三个轮询方法（核心）

| 方法 | 阻塞 | 返回值 | 语义 |
|---|---|---|---|
| `latest_frame()` | 否 | `(FramePair \| None, seq)` | 取最新采集帧 |
| `has_new_result() -> bool` | 否 | `bool` | 自上次 `take_result()` 之后是否有**新**结果 |
| `take_result() -> DepthResult \| None` | 否 | 最新结果 | **不是消费式的**：重复调用返回同一个对象，只是把“已读”标记推进 |

> SDK **故意不提供同步的 `infer()`**。推理一次 47~164 ms，在 UI 线程里同步调用必然卡死界面。正确用法是一个定时器（建议 30 Hz）轮询 `has_new_result()`，有则 `take_result()`。
>
> 因为 `take_result()` 不会清空结果，**“要不要逐帧处理”必须由 `has_new_result()` 判断**；只写 `take_result()` 会在同一帧上反复处理。

### 2.3 参数

```python
get_params() -> dict            # zmin / zmax / preset / scale / paused / resolution / backend
set_params(**kw) -> dict        # 返回 {"applied": {...}, "needs_restart": [...]}
```

**热生效**（`HOT_PARAMS`，不需要重启）：`zmin`、`zmax`、`depth_range`、`point_size`、`voxel`、`paused`。

**必须重启**（`RESTART_PARAMS`，`set_params` 不会应用，只放进返回值的 `needs_restart`）：`resolution`、`backend`、`max_disp`。

> `resolution` 改动后**必须重新标定**，否则深度数值无意义。

### 2.4 标定、保存、统计

| 方法 | 说明 |
|---|---|
| `reload_calibration() -> dict` | 重读标定文件、替换校正器，**不重启相机、不重载模型**，通常几十毫秒，下一帧生效。返回 `{"rectified", "source", "baseline_m", "rect_fx", "rms_stereo_px", "image_size"}` |
| `save_snapshot(tags=None) -> str` | 保存当前结果一整套，返回目录路径；无结果时返回 `""` |
| `stats() -> dict` | `capture_fps` / `infer_fps` / `frames` / `results` / `infer_ms` / `preset` / `paused` / `started` / `valid_ratio` / `median_m` / `points` / `stages` |
| `last_error() -> str \| None` | 最近一次错误的类型与消息 |
| `pause()` / `resume()` | 暂停 / 恢复结果产出 |
| `version() -> str` | 返回 `SDK_VERSION` |

---

## 3. 数据类型

### 3.1 `FramePair`

| 字段 | 类型 | 说明 |
|---|---|---|
| `left`, `right` | `np.ndarray` uint8 BGR | 已校正图像 |
| `t_left`, `t_right` | `float` | `time.perf_counter()` 时间戳（秒） |
| `seq` | `int` | 帧序号 |
| `mode` | `str` | 默认 `"stereo"` |
| `synced` | `bool` | 默认 `True` |

属性：`.size -> (宽, 高)`、`.is_valid -> bool`。

### 3.2 `DepthResult`

| 字段 | 类型 | 说明 |
|---|---|---|
| `frame` | `FramePair` | 对应的采集帧 |
| `disparity` | `np.ndarray` float32 | 视差（像素），**模型分辨率** |
| `depth` | `np.ndarray` float32 | 深度（米），**已上采样到全分辨率**，0 = 无效 |
| `valid` | `np.ndarray` bool | 有效掩码，与 `depth` 同形 |
| `points` | `np.ndarray` (N,3) float32 | 点云（米），相机系 **y 向下** |
| `colors` | `np.ndarray` (N,3) uint8 | **BGR**，取自左图 |
| `timings` | `dict[str, float]` | 各阶段耗时（毫秒），键见 `TIMING_KEYS` |
| `seq` | `int` | 序号 |
| `preset` | `str` | 当前预设名 |
| `scale` | `float` | 当前缩放 |

方法：

```python
depth_at(u, v) -> float     # 某像素深度（米）；无效或越界返回 0.0
summary() -> str            # "seq=1 preset=balanced scale=0.75 infer=95ms valid=291839 median=0.653m points=72904"
```

`TIMING_KEYS = ("capture", "rectify", "infer", "cloud", "display")`。

> **形状不一致是头号陷阱**：`disparity` 是模型分辨率（`scale=0.75` 时 480×360），`depth` 是全分辨率（640×480）。不要用同一套像素坐标索引两者。

---

## 4. 导出：`stereo_sdk.export`

| 函数 | 格式 | 单位 / 量化 |
|---|---|---|
| `save_ply(path, points, colors=None, binary=True, comments=None)` | PLY | 米；`colors` 传入 BGR，写入时自动转 RGB |
| `save_depth_png(path, depth, valid=None, scale=1000.0, max_mm=65535)` | 16 位 PNG | 毫米，0 = 无效 |
| `save_depth_npy(path, depth)` | float32 `.npy` | 米，无损 |
| `save_depth_tiff(path, depth)` | 32 位 TIFF | 米；需 `pip install imagecodecs` |
| `save_disparity_png(path, disparity, fixed_scale=32.0)` | 16 位 PNG | 像素 × 32 定点 |
| `save_disparity_npy(path, disparity)` | float32 `.npy` | 像素 |
| `save_valid_mask(path, valid)` | 8 位 PNG | 255 = 有效 |
| `export_result(result, out_dir, prefix="frame", ply_binary=True, with_tiff=False, comments=None)` | 一套 | 见下 |

`export_result` 返回 `dict`，键名固定：`left` / `right` / `disparity_npy` / `disparity_png` / `depth_npy` / `depth_png` / `valid_png` / `cloud`（`with_tiff=True` 时多一个 `depth_tiff`）。

单个文件写失败不会中断其余文件，但会抛真实异常（如磁盘满）。

```python
from stereo_sdk.export import export_result
paths = export_result(r, "out", prefix="scan01")
```

---

## 5. 标定信息：`stereo_sdk.calib.CalibInfo`

```python
CalibInfo.from_file(path: str = "") -> CalibInfo     # 按候选顺序查找；找不到抛 FileNotFoundError 并列出查过的路径
CalibInfo.latest(calib_dir: str = "") -> CalibInfo   # 取最新一份
```

查找顺序（`resolve_calib_path`）：

1. 环境变量 `STEREO_CALIB`
2. 当前工作目录的 `calibration/stereo_calib.yaml`
3. 包所在目录上一级的 `calibration/stereo_calib.yaml`（源码树直接运行时命中）
4. `~/.stereo/calibration/stereo_calib.yaml`

**原始矩阵字段**：`K1`、`D1`、`K2`、`D2`、`R`、`T`（米）、`image_size`、`date`、`rms_left`、`rms_right`、`rms_stereo`、`axis_angle_deg`、`meta`、`path`。

**派生属性**：

| 属性 | 含义 |
|---|---|
| `baseline_mm` / `baseline_m` | 基线 |
| `fx` / `fy` | 左目焦距（像素） |
| `rect_fx` | 校正后焦距 |
| `rms` | 双目重投影 RMS |
| `principal_point` | 主点 `(cx, cy)` |
| `fov_deg` | 视场角 `(水平, 垂直)` |

**方法**：

```python
sigma_depth(z_m, disp_sigma_px=0.5) -> float          # 该距离下的深度不确定度（米）
usable_range(max_sigma_m=0.01, disp_sigma_px=0.5) -> float   # 满足误差要求的最远距离（米）
summary() -> str
as_dict() -> dict
check() -> list[tuple[item, detail, bool]]            # 轻量体检，不依赖 OpenCV
```

`CalibInfo` **只读**，不提供保存。要改标定请用 `tools/calib_gui.py`——两处都能写就会出现“哪份生效”的事故。

---

## 6. 录制：`stereo_sdk.recorder.Recorder`

```python
Recorder(out_root, save_images=True, save_disparity=True,
         save_depth=True, save_pointcloud=True, tag="cloud")
Recorder.save(result, meta_extra=None) -> str    # 返回新建的会话目录
```

目录命名 `时间戳_毫秒_标签`，内含 `left_raw.png` / `right_raw.png` / `disparity.npy` / `depth_meter.npy` / `valid.npy` / `depth_meter.txt` / `cloud.ply` / `meta.txt`。

与 `export_result` 的区别：`Recorder` 的 PLY 是 **ASCII** 格式且文件名固定（便于按目录批量读），`export_result` 的 PLY 默认 **binary** 且文件名带 `prefix`。

---

## 7. 断言辅助

```python
check_frame(frame)   # 非 None、左右同形、非空
check_result(res)    # 非 None、depth/valid/points 齐全
```

在你自己写的数据管道里调用它们，能在错误传播很远之前就定位。

---

## 8. 线程与阻塞约定

| 事项 | 约定 |
|---|---|
| `start()` / `stop()` | 阻塞，不要在 UI 主线程的定时器里调用 |
| 三个轮询方法 | 全部非阻塞，可在 UI 线程 30 Hz 调用 |
| 回调 | SDK 不提供回调，只有轮询 |
| 并发 | 内部有锁；`take_result()` 返回的是共享对象，**跨线程修改它会影响其他消费者** |

---

## 9. 一段完整的最小可用代码

```python
import time
from stereo_sdk import Session, export_result

with Session("config.yaml") as cam:
    last_seq = -1
    saved = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 10:          # 跑 10 秒
        if not cam.has_new_result():
            time.sleep(0.01)
            continue
        r = cam.take_result()
        if r is None or r.seq == last_seq:        # 去重，保证逐帧
            continue
        last_seq = r.seq
        print(r.summary())
        if saved < 1 and r.valid.mean() > 0.5:    # 有效点过半才存
            export_result(r, "out", prefix=f"seq{r.seq:04d}")
            saved += 1
    print(cam.stats())
```

更多可直接运行的例子见 [../examples/](../examples/)。
