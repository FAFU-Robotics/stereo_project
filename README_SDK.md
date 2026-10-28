# stereo_sdk —— 双目深度相机 Python SDK

面向"我要在自己程序里拿到深度图和点云"的人。不需要读仓库里的推理代码，也不需要知道
FoundationStereo 的细节。

- 版本：0.3.0
- Python：3.10+
- 单位约定：**长度一律米**，焦距一律像素，图像一律 BGR uint8

---

## 1. 安装

```bash
# 在仓库根目录（含 pyproject.toml 的那一层）
pip install -e .
```

这会装三样东西：`stereo_sdk` 包、以及 `Session` 运行时依赖的根目录模块
（`camera` / `pipeline_threaded` / `calibration` / `pointcloud` / `profiler` …）。

**torch 不在默认依赖里。** 只有真正 `Session.start()` 跑推理时才需要它，
按你的 CUDA 版本单独装：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 示例，按实际版本改
```

只读标定信息、或只想用 `MockSession` 跑界面的人，装完 `pip install -e .` 就够了。

---

## 2. 三行拿到深度

```python
from stereo_sdk import Session

with Session("config.yaml") as cam:      # __enter__ 里 start()，__exit__ 里 stop()
    while True:
        if cam.has_new_result():
            r = cam.take_result()
            print(r.summary())           # seq / preset / infer 耗时 / 有效点数 / 中位深度
            d = r.depth_at(320, 240)     # 某像素的深度（米），无效返回 0.0
```

**没有相机时**用 `MockSession`，接口完全一样：

```python
from stereo_sdk import MockSession
with MockSession() as cam:
    ...
```

---

## 3. 拿点云并存盘

```python
from stereo_sdk.export import export_result, save_ply

r = cam.take_result()
save_ply("cloud.ply", r.points, r.colors)             # 二进制 PLY，Open3D/MeshLab 直接开
export_result(r, "out/", prefix="seq001")             # 一整套：左右图/视差/深度/掩码/点云
```

`export_result` 返回的键名固定：`left / right / disparity_npy / disparity_png /
depth_npy / depth_png / valid_png / cloud`。

各格式的量化方式：

| 函数 | 格式 | 单位 / 量化 |
|---|---|---|
| `save_depth_png` | 16 位 PNG | 毫米，0 = 无效 |
| `save_depth_npy` | float32 .npy | 米，无损 |
| `save_depth_tiff` | 32 位 TIFF | 米（需 `pip install imagecodecs`） |
| `save_disparity_png` | 16 位 PNG | 像素 × 32 定点 |
| `save_ply` | PLY | 米，二进制（默认）/ ASCII |

---

## 4. 标定信息（决定深度能不能当真的东西）

```python
from stereo_sdk.calib import CalibInfo

c = CalibInfo.from_file()          # 自动查找，见下方查找顺序
print(c.summary())
print(c.baseline_mm, c.fx, c.rect_fx, c.rms_stereo)
print(c.sigma_depth(1.0))          # 1 m 处的深度不确定度（米）
print(c.usable_range(0.01))        # 要求误差 ≤1 cm 时，最远可用距离
```

**标定文件的查找顺序**（`pip install` 之后包被复制进 site-packages，
"相对仓库根目录"那个默认路径就不再指向你的标定文件了，所以按下面顺序找）：

1. 环境变量 `STEREO_CALIB` 指向的路径（显式指定，最高优先）
2. 当前工作目录下的 `calibration/stereo_calib.yaml`
3. 包所在目录的上一级（`calibration/stereo_calib.yaml`，源码树直接运行时命中）
4. `~/.stereo/calibration/stereo_calib.yaml`（想把这个文件拷出来单独用）

都不存在时会抛 `FileNotFoundError`，**并把查过的每个路径打印出来**，不会含糊地说"找不到文件"。

```bat
:: Windows 显式指定（只对当前命令行窗口有效）
set STEREO_CALIB=D:\my_project\calibration\stereo_calib.yaml
```

`CalibInfo` 是**只读**的：它不提供保存。要改标定，请用 `tools/calib_gui.py`。
只读是刻意的——两处都能写，就会出现"哪份才是生效的"这种事故。

---

## 5. 接口约定（踩过坑的三条）

1. **`take_result()` 是消费式的**：取走一次就没了。要多次使用请自己存变量。
2. **`points` 的 y 轴向下**（相机坐标系），不是 OpenGL 的 y 向上。写 PLY 时未做翻转，
   在 MeshLab 里看到上下颠倒属于查看器的预期行为。
3. **`depth` 已上采样到全分辨率**，而 `disparity` 是模型分辨率——两者形状通常不同，
   不要按同一套像素坐标去索引。

---

## 6. 常见错误

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: camera` | 只把 `stereo_sdk/` 拷走了 | 用 `pip install -e .`，别手拷目录 |
| `FileNotFoundError: stereo_calib.yaml` | 没做标定 | 跑 `tools/calib_gui.py` 做一次 |
| `RuntimeError: image size mismatch`（`reload_calibration`） | 新标定的分辨率与当前相机不同 | 换分辨率后必须重新标定 |
| 深度全是 0 | `valid` 全 False（弱纹理 / 超量程） | 看 `r.summary()` 的 valid 数，检查 `zmax` |
