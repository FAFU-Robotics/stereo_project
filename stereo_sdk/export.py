"""export.py —— 把 DepthResult 导成外部工具能读的文件

为什么单独一个模块而不是塞进 recorder.py：
  recorder 是「整帧落盘」，讲究目录结构与既有分析脚本兼容；
  export 是「用户在自己程序里随手存一份」，讲究一个函数一个格式、不产生目录。
  两者需求不同，混在一起会让导出的默认行为互相牵制。

支持：
  · PLY 点云：二进制（默认，体积约为 ASCII 的 1/8、写入快一个数量级）与 ASCII
  · 深度图：16 位 PNG（毫米，×1000 量化）、float32 .npy、32 位 TIFF（米）
  · 视差图：16 位 PNG（×32 定点）与 .npy
  · 有效性掩码：8 位 PNG

坐标与单位约定（与 DepthResult 完全一致，不做任何翻转）：
  points 为米、相机坐标系、x 向右、**y 向下**、z 向前；colors 为 BGR。
  PLY 里颜色按 RGB 写出（Open3D / MeshLab 都按 RGB 读）。

注意：写出的 PLY **不带** y 翻转。若你在 MeshLab 里看到上下颠倒，
      那是查看器默认 +y 向上的预期行为，不是文件错了。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

from .types import DepthResult

__all__ = [
    "save_ply", "save_depth_png", "save_depth_npy", "save_depth_tiff",
    "save_disparity_png", "save_disparity_npy", "save_valid_mask",
    "export_result",
]


# --------------------------------------------------------------------------- PLY
def save_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None,
             binary: bool = True, comments: Optional[Sequence[str]] = None) -> str:
    """写 PLY 点云。

    points : (N,3) float32/float64，单位米
    colors : (N,3) uint8，**BGR**（与 DepthResult.colors 一致）；None 则全灰
    binary : True 写 binary_little_endian（默认，推荐）；False 写 ASCII
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = int(len(pts))
    if colors is not None and len(colors):
        rgb = np.asarray(colors, np.uint8).reshape(-1, 3)[:, ::-1]
    else:
        rgb = np.full((n, 3), 128, np.uint8)
    if len(rgb) < n:
        pad = np.full((n - len(rgb), 3), 128, np.uint8)
        rgb = np.vstack([rgb, pad])
    rgb = rgb[:n]

    head = ["ply", f"format {'binary_little_endian' if binary else 'ascii'} 1.0"]
    for c in (comments or []):
        ascii_c = str(c).encode("ascii", errors="ignore").decode("ascii")
        head.append(f"comment {ascii_c}")
    head += [f"element vertex {n}",
             "property float x", "property float y", "property float z",
             "property uchar red", "property uchar green", "property uchar blue",
             "end_header"]
    header = ("\n".join(head) + "\n").encode("ascii")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(header)
        if binary:
            # 逐点 12 字节 float + 3 字节 uchar = 15 字节，用结构化 dtype 一次写完
            rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            fh.write(rec.tobytes())
        else:
            lines = []
            for i in range(n):
                x, y, z = pts[i]
                r, g, b = rgb[i]
                lines.append(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}")
            fh.write(("\n".join(lines) + "\n").encode("ascii"))
    return path


# ----------------------------------------------------------------------- 深度图
def save_depth_png(path: str, depth: np.ndarray, valid: Optional[np.ndarray] = None,
                   scale: float = 1000.0, max_mm: int = 65535) -> str:
    """深度存 16 位 PNG（单位**毫米**，0 表示无效）。

    scale=1000 时，1 mm 分辨率；超过 max_mm 的深度会被截断（默认 65.5 m）。
    valid 为 False 的位置强制写 0，避免把无效点当成"很近"。
    """
    d = np.asarray(depth, np.float32)
    mm = np.clip(d * float(scale), 0, max_mm)
    if valid is not None:
        mm = np.where(np.asarray(valid, bool), mm, 0.0)
    out = np.round(mm).astype(np.uint16)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not cv2.imwrite(path, out):
        raise IOError(f"写 PNG 失败：{path}")
    return path


def save_depth_npy(path: str, depth: np.ndarray) -> str:
    """float32 原始深度（米，0=无效），无损，供自己的脚本直接 np.load。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.save(path, np.asarray(depth, np.float32))
    return path


def save_depth_tiff(path: str, depth: np.ndarray) -> str:
    """32 位 float TIFF（米）。OpenCV 只在装了 imagecodecs 时支持写 TIFF。"""
    try:
        import imagecodecs  # noqa: F401
    except ImportError as exc:            # pragma: no cover
        raise RuntimeError(
            "写 TIFF 需要 imagecodecs：pip install imagecodecs\n"
            "（不想装的话用 save_depth_npy 或 save_depth_png）") from exc
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not cv2.imwrite(path, np.asarray(depth, np.float32)):
        raise IOError(f"写 TIFF 失败：{path}")
    return path


# ----------------------------------------------------------------------- 视差图
def save_disparity_png(path: str, disparity: np.ndarray, fixed_scale: float = 32.0) -> str:
    """视差存 16 位 PNG（像素 × 32 定点，0 表示无效）。

    ×32 是因为多数双目模型的视差精度是 1/32 像素；这样存能保留小数位。
    """
    d = np.clip(np.asarray(disparity, np.float32), 0, 65535.0 / fixed_scale)
    out = np.round(d * float(fixed_scale)).astype(np.uint16)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not cv2.imwrite(path, out):
        raise IOError(f"写 PNG 失败：{path}")
    return path


def save_disparity_npy(path: str, disparity: np.ndarray) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.save(path, np.asarray(disparity, np.float32))
    return path


def save_valid_mask(path: str, valid: np.ndarray) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out = (np.asarray(valid, bool).astype(np.uint8) * 255)
    if not cv2.imwrite(path, out):
        raise IOError(f"写 PNG 失败：{path}")
    return path


# ------------------------------------------------------------------------- 批量
def export_result(result: DepthResult, out_dir: str, prefix: str = "frame",
                  ply_binary: bool = True, with_tiff: bool = False,
                  comments: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """一次导出一整套：左右图、视差、深度（npy+png）、掩码、点云。

    返回 {种类: 路径}，键名固定：left / right / disparity_npy / disparity_png /
    depth_npy / depth_png / valid_png / cloud。
    单独某个文件写失败不会中断其余（会在返回里缺这个键），但会抛 FileNotFound 之外的
    真实异常（比如磁盘满）。
    """
    os.makedirs(out_dir, exist_ok=True)
    out: Dict[str, str] = {}

    def p(ext: str) -> str:
        return os.path.join(out_dir, f"{prefix}.{ext}")

    if result.frame is not None:
        cv2.imwrite(p("left.png"), result.frame.left)
        out["left"] = p("left.png")
        cv2.imwrite(p("right.png"), result.frame.right)
        out["right"] = p("right.png")
    if result.disparity is not None and np.size(result.disparity):
        out["disparity_npy"] = save_disparity_npy(p("disparity.npy"), result.disparity)
        out["disparity_png"] = save_disparity_png(p("disparity16.png"), result.disparity)
    if result.depth is not None and np.size(result.depth):
        out["depth_npy"] = save_depth_npy(p("depth.npy"), result.depth)
        out["depth_png"] = save_depth_png(p("depth16.png"), result.depth, result.valid)
        if with_tiff:
            out["depth_tiff"] = save_depth_tiff(p("depth.tiff"), result.depth)
    if result.valid is not None and np.size(result.valid):
        out["valid_png"] = save_valid_mask(p("valid.png"), result.valid)
    if len(result.points):
        cm = list(comments or []) + [
            "units: meters",
            "coords: camera frame, x right, y DOWN, z forward",
            "colors: RGB, from left image",
            f"generated by stereo_sdk.export (seq={result.seq})",
        ]
        out["cloud"] = save_ply(p("cloud.ply"), result.points, result.colors,
                                binary=ply_binary, comments=cm)
    return out
