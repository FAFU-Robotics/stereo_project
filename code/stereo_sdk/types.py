"""stereo_sdk.types —— 数据契约（GUI 与后台之间唯一的数据定义）

=============================================================================
【为什么单独一个文件】
    这个文件**只依赖 numpy 和标准库**，不 import torch / cv2 / 相机模块。
    这样 GUI 同学可以放心 `from stereo_sdk.types import ...`，
    既不会拖入 3 GB 的推理依赖，也不会因为后台正在重构而崩。

【三个必须记住的约定】（踩过的坑，写在这里避免重复）
    1) 所有图像都是 **BGR**（OpenCV 习惯），不是 RGB！
       Qt 里要用 QImage.Format_BGR888（Qt>=5.14），别用 Format_RGB888。
    2) 点云坐标是**相机坐标系**：x 向右、y 向下、z 向前，单位**米**。
       3D 显示时要翻 y（否则画面上下颠倒）。
    3) 深度图里 **0 表示无效**；判断有效性请用 valid 掩码，不要靠 depth==0 猜。
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# 接口版本：GUI 端请把它显示在"关于"里，联调时一眼看出两边版本是否一致。
# 这是**唯一版本来源**——pyproject.toml 用 dynamic 从这里读，不要再在别处写死版本号。
SDK_VERSION = "0.3.0"

# timing 字典里保证出现的键（GUI 画曲线时按这些键取，不要假设别的键）
TIMING_KEYS = ("capture", "rectify", "infer", "cloud", "display")


@dataclass(frozen=True)
class FramePair:
    """一对左右图（双目并排相机的一帧）。

    left / right : uint8, (H, W, 3), **BGR**
    t_left/t_right: float, 秒。**time.perf_counter() 时基**（不是 wall clock）
    seq          : int, 单调递增的帧号（判断"有没有新帧"）
    synced       : bool, 采集端是否认为是同一时刻的左右图
    """

    left: np.ndarray
    right: np.ndarray
    t_left: float
    t_right: float
    seq: int
    mode: str = "stereo"
    synced: bool = True

    @property
    def size(self) -> Tuple[int, int]:
        """(宽, 高)，单眼尺寸。"""
        return int(self.left.shape[1]), int(self.left.shape[0])

    def is_valid(self) -> bool:
        return (self.left is not None and self.right is not None
                and self.left.shape == self.right.shape and self.left.size > 0)


@dataclass(frozen=True)
class DepthResult:
    """一次推理的完整结果（GUI 只需要这一个对象就能画所有面板）。

    frame      : 与该结果**配对**的帧（已校正后的左右图；GUI 若同时显示"实时画面"
                 和"深度"，两者会差 1 帧左右，这是线程解耦的正常现象）
    disparity  : float32 (h, w)，**模型分辨率**，单位像素
    depth      : float32 (H, W)，**已上采样到全分辨率**（= frame.left 的尺寸），单位**米**，
                 无效处为 0
    valid      : bool (H, W)，与 depth 同尺寸
    points     : float32 (N, 3)，相机坐标系，**米**，**y 向下**
    colors     : uint8 (N, 3)，**BGR**，与 points 一一对应
    timings    : dict[str, float]，毫秒；键见 TIMING_KEYS
    preset     : 当前预设名（"precise" / "balanced" / "fast" / "custom"）
    scale      : 模型输入缩放（1.0 / 0.75 / 0.5）
    """

    frame: FramePair
    disparity: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    timings: Dict[str, float] = field(default_factory=dict)
    seq: int = 0
    preset: str = "custom"
    scale: float = 1.0

    def depth_at(self, u: int, v: int) -> float:
        """取某像素深度（米）；无效或越界返回 0.0。GUI 点选测距用。"""
        if (self.depth is None or v < 0 or u < 0
                or v >= self.depth.shape[0] or u >= self.depth.shape[1]):
            return 0.0
        if self.valid is not None and not bool(self.valid[v, u]):
            return 0.0
        return float(self.depth[v, u])

    def summary(self) -> str:
        """一行摘要，状态栏直接显示。"""
        n = 0
        med = 0.0
        if self.depth is not None and self.valid is not None and self.valid.any():
            vals = self.depth[self.valid]
            n = int(self.valid.sum())
            med = float(np.median(vals))
        inf = self.timings.get("infer", 0.0)
        return (f"seq={self.seq} preset={self.preset} scale={self.scale} "
                f"infer={inf:.0f}ms valid={n} median={med:.3f}m points={len(self.points)}")


# ---------------------------------------------------------------- 参数
#: set_params() 接受的热生效参数（GUI 的"参数栏"照着这个做就行）
HOT_PARAMS: Dict[str, str] = {
    "zmin": "深度显示下限（米），float",
    "zmax": "深度显示上限（米），float",
    "point_size": "3D 点大小，float",
    "voxel": "体素下采样尺寸（米），0=不采样",
    "depth_range": "(zmin, zmax) 元组形式，等价于同时设两个",
    "paused": "是否暂停显示更新（推理继续），bool",
}

#: 需要**重启 Session** 才能生效的参数（GUI 要提示用户）
RESTART_PARAMS: Dict[str, str] = {
    "resolution": "采集分辨率，如 '1280x480'；改动后**必须重新标定** ⚠️",
    "backend": "采集后端：'msmf' / 'dshow' / 'any'",
    "max_disp": "FFS 最大视差（决定最近可测距离）",
}


def check_frame(frame: Optional[FramePair]) -> None:
    """GUI 自检用：断言数据符合契约（开发期调用，生产可去掉）。"""
    if frame is None:
        return
    assert frame.left.dtype == np.uint8, "left 必须是 uint8"
    assert frame.left.ndim == 3 and frame.left.shape[2] == 3, "left 必须是 (H,W,3) BGR"
    assert frame.left.shape == frame.right.shape, "左右图尺寸必须一致"


def check_result(res: Optional[DepthResult]) -> None:
    """GUI 自检用：断言结果符合契约。"""
    if res is None:
        return
    assert res.depth.dtype == np.float32, "depth 必须是 float32"
    assert res.depth.shape == res.valid.shape, "depth/valid 形状必须一致"
    assert res.depth.ndim == 2, "depth 必须是 (H,W)"
    assert res.disparity.ndim == 2, "disparity 必须是 (h,w)"
    assert res.points.ndim == 2 and res.points.shape[1] == 3, "points 必须是 (N,3)"
    assert res.colors.shape[0] == res.points.shape[0], "colors 必须与 points 一一对应"
    assert res.depth.shape[0] == res.frame.left.shape[0], "depth 必须已上采样到全分辨率"
