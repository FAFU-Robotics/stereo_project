"""点云着色的通道顺序：蓝色必须还是蓝色，不能变成橙色。

用法：
    python tests/test_color_channels.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for _p in (ROOT, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from depth_utils import CameraParams  # noqa: E402
from pointcloud import PointCloudBuilder, bgr_u8_from_open3d, cloud_color_bgr  # noqa: E402


def test_model_rgb_blue_roundtrips_to_bgr_blue() -> None:
    """模型左图是 RGB。蓝色 (R 低、B 高) 进点云再写回契约，仍应是 BGR 的蓝。"""
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[:] = (20, 60, 220)
    bgr = cloud_color_bgr(None, rgb, rgb.shape)
    assert bgr is not None
    assert bgr[0, 0].tolist() == [220, 60, 20]

    depth = np.full((8, 8), 1.0, np.float32)
    params = CameraParams(fx=100, fy=100, cx=3.5, cy=3.5, baseline_m=0.06,
                          width=8, height=8)
    builder = PointCloudBuilder({"zmin_m": 0.1, "zmax_m": 3.0, "use_color": True,
                                 "max_points": 0, "voxel_size_m": 0})
    _pts, cols = builder.build_arrays(depth, params, bgr)
    assert cols is not None and len(cols) > 0
    # Open3D 要 RGB 0~1：蓝通道接近 220/255，红通道接近 20/255
    assert cols[0, 2] > 0.8
    assert cols[0, 0] < 0.12
    stored = bgr_u8_from_open3d(cols)
    assert int(stored[0, 0]) > 200
    assert int(stored[0, 2]) < 40


def test_bgr_camera_buffer_is_not_swapped() -> None:
    """OpenCV 的蓝像素是 BGR (255, 40, 20)。默认不要再交换，否则封面会变成橙色。"""
    import cv2
    from camera import StereoCamera

    cam = StereoCamera({"mode": "side_by_side", "color_order": "bgr",
                        "backend": "dshow", "width": 4, "height": 4})
    bgr_bytes = np.zeros((4, 4, 3), np.uint8)
    bgr_bytes[:] = (255, 40, 20)
    kept = cam._as_bgr(bgr_bytes)
    assert kept[0, 0].tolist() == [255, 40, 20]
    shown = cv2.cvtColor(kept[:1, :1], cv2.COLOR_BGR2RGB)[0, 0].tolist()
    assert shown == [20, 40, 255]
    text = open(os.path.join(ROOT, "config.yaml"), encoding="utf-8").read()
    assert "color_order: bgr" in text


def test_truecolor_keeps_camera_blue() -> None:
    """左图是 BGR 的蓝。就算同时带了模型 RGB，真实色彩也必须还是这块蓝，不能变成红。"""
    import cv2

    bgr = np.zeros((8, 8, 3), np.uint8)
    bgr[:] = (220, 40, 15)
    model_rgb = bgr[:, :, ::-1].copy()
    picked = cloud_color_bgr(bgr, model_rgb, bgr.shape)
    assert picked is not None
    assert picked[0, 0].tolist() == [220, 40, 15]

    depth = np.full((8, 8), 1.0, np.float32)
    params = CameraParams(fx=100, fy=100, cx=3.5, cy=3.5, baseline_m=0.06,
                          width=8, height=8)
    builder = PointCloudBuilder({"zmin_m": 0.1, "zmax_m": 3.0, "use_color": True,
                                 "max_points": 0, "voxel_size_m": 0})
    _pts, cols = builder.build_arrays(depth, params, picked)
    assert builder._bgr_u8 is not None
    assert builder._bgr_u8[0].tolist() == [220, 40, 15]
    # Open3D 那份仍然是 RGB：蓝通道高、红通道低
    assert cols is not None and cols[0, 2] > 0.8 and cols[0, 0] < 0.1

    img = np.zeros((2, 2, 3), np.uint8)
    img[0, 0] = builder._bgr_u8[0]
    shown = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[0, 0]
    assert int(shown[2]) > 200 and int(shown[0]) < 40


def test_viewer_does_not_swap_bgr_cloud_colors() -> None:
    from viewer_shell import _as_bgr_u8
    blue_bgr = np.array([[220, 60, 20]], np.uint8)
    out = _as_bgr_u8(blue_bgr, 1)
    assert out[0].tolist() == [220, 60, 20]


if __name__ == "__main__":
    test_model_rgb_blue_roundtrips_to_bgr_blue()
    test_bgr_camera_buffer_is_not_swapped()
    test_truecolor_keeps_camera_blue()
    test_viewer_does_not_swap_bgr_cloud_colors()
    print("[OK] color channels")
