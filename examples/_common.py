"""examples/_common.py —— 四个示例共用的两件小事：解析参数、等一帧。

为什么单独一个文件：示例的价值在于"复制这段就能用"，
若把 20 行 argparse 抄进每个例子，看的人会先被无关代码劝退。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Optional

import numpy as np


def build_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="config.yaml", help="配置文件路径")
    ap.add_argument("--mock", action="store_true",
                    help="用 MockSession（无需相机与 torch，先看效果用这个）")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="等待第一帧的最长秒数（真机首次加载模型可能要 10 s 以上）")
    return ap


def open_camera(args: Any):
    """按 --mock 返回 MockSession 或真 Session，并已 start()。"""
    from stereo_sdk import MockSession, Session
    if args.mock:
        s = MockSession()
    else:
        s = Session(args.config)
    s.start()
    return s


def wait_result(s: Any, timeout: float = 60.0):
    """阻塞等到第一个 DepthResult。

    SDK 故意不提供同步 infer()——那会卡死 UI 线程。这里是**示例程序**，
    没有界面，所以用轮询把一次推理等到，这是正确用法；GUI 里不要这么写。
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if s.has_new_result():
            r = s.take_result()
            if r is not None:
                return r
        time.sleep(0.01)
    raise TimeoutError(
        f"{timeout:.0f} s 内没有拿到推理结果。\n"
        "真机模式下通常是模型还没加载完，或用 --mock 先看效果。")


def colorize_depth(depth: np.ndarray, valid: Optional[np.ndarray],
                   zmin: float = 0.2, zmax: float = 3.0) -> np.ndarray:
    """深度图 -> 伪彩色（近=暖色）。无效处涂黑。"""
    import cv2
    d = np.asarray(depth, np.float32)
    norm = np.clip((d - zmin) / max(zmax - zmin, 1e-6), 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8)
    rgb = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    if valid is not None:
        rgb[~np.asarray(valid, bool)] = (0, 0, 0)
    return rgb
