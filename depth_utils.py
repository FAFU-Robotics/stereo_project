"""depth_utils.py —— 视差 → 深度、无效值处理、深度范围过滤、参数与单位说明
=============================================================================
职责（单一）：把"视差图（像素）"变成"有物理单位的深度图（米）"，并管好三件事：
    1) 相机参数（内参/基线）从配置读取，并明确区分【已标定】与【占位临时值】；
    2) 无效值/异常值处理：视差 <= 0、视差过小（深度无穷大）、非有限值一律判为无效；
    3) 深度范围过滤（只保留 zmin..zmax 内的像素），并给出统计信息。

【数学关系（与官方 run_demo.py 完全一致）】
    depth = fx * baseline / disparity
    其中 fx 与 disparity 必须处于**同一个分辨率**：
        disparity 是模型输入分辨率下的像素值（通常是原图的 scale 倍），
        因此 fx 也要乘同一个 scale（官方做法就是 K[:2] *= scale）。

【三维反投影（左相机光学坐标系）】
    z = depth
    x = (u - cx) * z / fx      u 是列号（向右为正）
    y = (v - cy) * z / fy      v 是行号（向下为正）
    单位：米。x 右、y 下、z 前 —— 这是 OpenCV/Open3D 的常规相机坐标系。

【必须遵守的诚实原则】
    depth.calibrated == false 时，一切深度/点云都只能用于"流程验证"，
    不得声称其距离、尺寸或三维位置准确。本模块把这层状态写进参数对象，
    并强制在保存数据时把警告写进文件（见 CameraParams.describe / save_depth_npy）。
=============================================================================
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraParams:
    """相机内参与基线（含"是否已真正标定"的状态）。

    字段：
        fx, fy, cx, cy : 完整分辨率（单眼）下的内参，单位像素
        baseline_m     : 左右相机基线，单位米
        calibrated     : True=来自真实标定；False=占位临时值（禁止测量）
        source         : 参数来源的文字说明（会写进保存的文件里）
        width, height  : 该内参对应的图像尺寸（单眼，像素）
    """

    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    calibrated: bool = False
    source: str = ""
    width: int = 0
    height: int = 0

    # ---------------------------------------------------------------- 构造
    @classmethod
    def from_config(cls, cfg: Dict[str, Any], eye_size: Optional[Tuple[int, int]] = None) -> "CameraParams":
        """从 config.yaml 读取参数。

        已标定（depth.calibrated=true）时读 calibration_file 指向的标定结果；
        未标定（false）时读 depth.temp 下的占位值，并标记 calibrated=False。

        输入：
            cfg      : 完整配置字典（含 project / depth 段）
            eye_size : 可选 (W,H)，单眼分辨率，用于校验内参对应的分辨率
        输出：CameraParams
        """
        depth_cfg = cfg.get("depth", {}) or {}
        calibrated = bool(depth_cfg.get("calibrated", False))

        if calibrated:
            path = str(depth_cfg.get("calibration_file", ""))
            if not os.path.isabs(path):
                path = os.path.join(str(cfg.get("project", {}).get("calibration_dir", "")), os.path.basename(path))
            from calibration import load_stereo_calibration  # 阶段 5 提供

            data = load_stereo_calibration(path)
            return cls(
                fx=float(data["fx"]), fy=float(data["fy"]),
                cx=float(data["cx"]), cy=float(data["cy"]),
                baseline_m=float(data["baseline_m"]),
                calibrated=True,
                source=f"标定文件 {path}（标定日期 {data.get('date', '未知')}）",
                width=int(data.get("width", 0)), height=int(data.get("height", 0)),
            )

        t = depth_cfg.get("temp", {}) or {}
        w, h = (eye_size if eye_size else (int(t.get("width", 640)), int(t.get("height", 480))))
        return cls(
            fx=float(t.get("fx", 500.0)), fy=float(t.get("fy", 500.0)),
            cx=float(t.get("cx", w / 2.0)), cy=float(t.get("cy", h / 2.0)),
            baseline_m=float(t.get("baseline_m", 0.06)),
            calibrated=False,
            source=str(t.get("source", "占位临时值")),
            width=int(w), height=int(h),
        )

    # ---------------------------------------------------------------- 派生
    def scaled(self, scale: float) -> "CameraParams":
        """返回把内参按比例缩放后的副本（用于视差处于模型分辨率时的换算）。

        为什么需要：视差是在"模型输入分辨率"（= 原图 × scale）下算出来的，
        换算深度时 fx 必须同比例缩放，否则深度会整体错一个倍数。
        注意：cx/cy 也必须一起缩放，否则反投影出来的点云会整体平移。
        """
        s = float(scale)
        return CameraParams(
            fx=self.fx * s, fy=self.fy * s, cx=self.cx * s, cy=self.cy * s,
            baseline_m=self.baseline_m, calibrated=self.calibrated, source=self.source,
            width=int(round(self.width * s)), height=int(round(self.height * s)),
        )

    def scaled_to_match(self, width: int) -> "CameraParams":
        """按"目标宽度 / 本参数对应的宽度"**自动求缩放系数**并返回新参数。

        为什么要有这个函数（实测踩过坑）：
            FFS 的 scale 定义是 "模型输入宽度 = 原图宽度 × scale"（例如 0.5）。
            手算时极易把分子分母搞反（写成 原图/模型 = 2.0），结果焦距错 4 倍、
            深度整体错 4 倍，而且画面看起来"有深度"，很难发现。
            直接用本函数传目标宽度，就不可能反。

        输入：width —— 目标图像宽度（通常是视差图的宽度）
        输出：按 width/self.width 缩放后的 CameraParams
        """
        if self.width <= 0:
            raise ValueError("CameraParams.width 未知，无法自动求缩放系数；请显式调用 scaled()")
        return self.scaled(float(width) / float(self.width))

    def K(self) -> np.ndarray:
        """返回 3x3 内参矩阵（float32），供 Open3D / 官方工具使用。"""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )

    def describe(self) -> str:
        """一行式说明（会显示在状态栏、写进保存的文件）。"""
        state = "已标定" if self.calibrated else "未标定（占位临时值，禁止测量）"
        return (
            f"{state} | fx={self.fx:.2f} fy={self.fy:.2f} cx={self.cx:.2f} cy={self.cy:.2f} "
            f"baseline={self.baseline_m:.4f} m | {self.width}x{self.height} | 来源: {self.source}"
        )


# -----------------------------------------------------------------------------
# 视差 -> 深度
# -----------------------------------------------------------------------------
def disparity_to_depth(
    disp: np.ndarray,
    params: CameraParams,
    min_valid_disp: float = 0.5,
    max_valid_disp: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """把视差图换算成深度图（米）。

    输入：
        disp           : (H,W) 视差，单位像素（**必须与 params 处于同一分辨率**）
        params         : CameraParams（注意：若视差是模型分辨率的，要先用 params.scaled(scale)）
        min_valid_disp : 小于该视差的像素视为无效（避免除零与"深度无穷大"）
        max_valid_disp : 可选。大于该视差的像素视为无效。
                         用途：FFS 的搜索范围是 max_disp，**超出该范围的视差是网络外推出来的，
                         不可信**。实测本机一帧数据里这类像素占 2.6%（深度约 0.15 m）。
                         建议传 0.98 * max_disp，作为一道便宜的安全网。
    输出：
        (depth (H,W) float32 单位米, valid (H,W) bool)
        无效处 depth 置 0，valid=False。

    处理规则（明确写出来，避免含糊）：
        - disp 非有限值(NaN/Inf) -> 无效
        - disp < min_valid_disp  -> 无效（对应深度过大/无匹配）
        - disp > max_valid_disp  -> 无效（超出搜索范围的外推值，若给了该参数）
        - 结果非有限值或 <=0     -> 无效
    """
    d = np.asarray(disp, dtype=np.float32)
    valid = np.isfinite(d) & (d >= min_valid_disp)
    if max_valid_disp is not None and max_valid_disp > 0:
        valid &= (d <= float(max_valid_disp))

    depth = np.zeros(d.shape, dtype=np.float32)
    if params.fx <= 0 or params.baseline_m <= 0:
        return depth, np.zeros(d.shape, dtype=bool)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = params.fx * params.baseline_m / np.maximum(d, min_valid_disp)
    z = z.astype(np.float32)
    valid &= np.isfinite(z) & (z > 0)
    depth[valid] = z[valid]
    return depth, valid


def filter_depth_range(depth: np.ndarray, zmin_m: float, zmax_m: float) -> np.ndarray:
    """返回深度范围掩码（只保留 zmin <= z <= zmax 的像素）。"""
    d = np.asarray(depth, dtype=np.float32)
    return np.isfinite(d) & (d >= float(zmin_m)) & (d <= float(zmax_m))


def filter_speckles(valid: np.ndarray, min_size: int = 200) -> np.ndarray:
    """去掉孤立的"散斑"小连通区域。

    为什么需要它（实测依据）：
        在**无纹理区域**（墙面、光滑表面、深色衣物），立体匹配缺少可靠依据，
        容易产生**孤立的虚假近距离**小斑块。实测本机一帧点云里，
        P90 视差换算深度只有 13 cm，而画面里的椅子明显在 40 cm 外 —— 就是这类散斑。
        这些斑点会让点云出现"靠近相机的碎片"，严重干扰观感与后续处理。

    做法：对有效像素掩码做 8 邻域连通域分析，丢弃面积小于 min_size 的连通域。
    复杂度 O(像素数)，640x480 约几毫秒，可实时使用。

    输入：
        valid    : (H,W) bool 有效掩码
        min_size : 最小保留面积（像素）；<=0 表示不过滤
    输出：(H,W) bool 过滤后的掩码
    """
    m = np.asarray(valid, dtype=bool)
    if min_size <= 0 or not m.any():
        return m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_size)   # 0 号是背景
    return keep[labels]


def depth_stats(depth: np.ndarray, valid: Optional[np.ndarray] = None) -> Dict[str, float]:
    """统计深度信息（供状态栏显示）。

    输出：{"valid_ratio": 百分比, "min_m","median_m","max_m", "count"}
    """
    d = np.asarray(depth, dtype=np.float32)
    m = (np.isfinite(d) & (d > 0)) if valid is None else (np.asarray(valid, bool) & np.isfinite(d) & (d > 0))
    n = int(m.sum())
    if n == 0:
        return {"valid_ratio": 0.0, "min_m": 0.0, "median_m": 0.0, "max_m": 0.0, "count": 0}
    v = d[m]
    return {
        "valid_ratio": 100.0 * n / d.size,
        "min_m": float(v.min()),
        "median_m": float(np.median(v)),
        "max_m": float(v.max()),
        "count": n,
    }


def colorize_depth(
    depth: np.ndarray,
    valid: np.ndarray,
    zmin_m: float,
    zmax_m: float,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """把深度图变成伪彩图（近=暖色，远=冷色）；无效区域涂黑。

    与视差伪彩不同，这里用**固定的 zmin/zmax** 映射：
    深度图的物理量程是用户设定的，固定量程才能让颜色在实时画面中稳定可比。
    """
    d = np.asarray(depth, dtype=np.float32)
    out = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    m = np.asarray(valid, bool) & np.isfinite(d) & (d > 0)
    if not m.any():
        return out
    span = max(float(zmax_m) - float(zmin_m), 1e-6)
    norm = np.clip((d - float(zmin_m)) / span, 0.0, 1.0)
    # 反转：近处(小 z)用暖色 -> 用 1-norm 让近处落在 colormap 的高端
    colored = cv2.applyColorMap(((1.0 - norm) * 255.0).astype(np.uint8), colormap)
    out[m] = colored[m]
    return out


def save_depth_npy(depth: np.ndarray, out_dir: str, params: CameraParams,
                   extra: Optional[Dict[str, Any]] = None) -> str:
    """保存深度图为 .npy，并写一份同名 .txt 说明单位与参数来源。

    为什么必须配套写说明：单独的 .npy 里没有单位、没有坐标系、没有标定状态，
    以后拿到文件的人无法判断能不能用。
    输出：保存的 .npy 路径
    """
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "depth_meter.npy")
    np.save(p, np.asarray(depth, dtype=np.float32))

    lines = [
        "# 深度图数据说明",
        f"保存时间     : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "文件         : depth_meter.npy（float32，单位【米】）",
        f"数组形状     : {depth.shape[1]}x{depth.shape[0]}（高x宽 -> H={depth.shape[0]}, W={depth.shape[1]}）",
        "无效值       : 0.0 表示无效（不是『距离为0』，请用 >0 判断有效）",
        "坐标系       : 左相机光学坐标系（x 向右, y 向下, z 向前/深度），单位米",
        f"标定状态     : {params.describe()}",
        "警告         : " + ("已标定参数，仍需按标定精度使用" if params.calibrated
                             else "【未标定】占位内参 + 厂家标称基线；本数据不能用于测距/测尺寸/定位"),
    ]
    if extra:
        for k, v in extra.items():
            lines.append(f"{k:<13}: {v}")
    with open(os.path.join(out_dir, "depth_meter.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return p
