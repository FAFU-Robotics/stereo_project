"""stereo_sdk.mock —— MockSession：没有相机、没有 GPU 也能开发 GUI

=============================================================================
【用途】
    GUI 同学拿到这个就能立刻开工：帧、视差、深度、点云全部是**合成**的，
    但**类型、形状、单位、更新节奏**与真 Session 完全一致。
    联调时把 `MockSession()` 换成 `Session()` 即可，GUI 代码一行不用改。

【合成场景】
    · 一面"墙"在 z 轴上缓慢前后移动（z = 0.55 ~ 0.75 m）
    · 墙上挂一个"方块"，比墙近 8 cm，并在画面里左右移动
    · 5% 的像素随机标为无效（模拟无纹理区域），用来检验 GUI 有没有正确用 valid 掩码
    · 左右图有一点视差（方块在右图里整体左移），看起来像真的双目图像对

【节奏】
    · 画面：30 FPS（每 1/30 s 一帧新图）
    · 结果：6 FPS（每 1/6 s 一个新 DepthResult）
    这样 GUI 的"取帧"和"取结果"两个节奏都能被真实地测到（而不是永远同一帧）。

【依赖】
    只有 numpy + 标准库。**不 import torch / cv2 / 相机模块**，
    所以任何电脑（无独显、未装推理依赖）都能跑起来做 UI 开发。
=============================================================================
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .types import DepthResult, FramePair, RESTART_PARAMS, SDK_VERSION

# 合成数据用的"相机内参"（只为了几何自洽，不必与真实标定一致）
_FX = 547.0
_FY = 547.0
_CX = 320.0
_CY = 240.0
_BASELINE_M = 0.0649


class MockSession:
    """与 Session 同接口的假实现（合成画面）。"""

    def __init__(self, config_path: str = "", width: int = 640, height: int = 480,
                 frame_fps: float = 30.0, result_fps: float = 6.0) -> None:
        self.width, self.height = int(width), int(height)
        self.frame_period = 1.0 / max(frame_fps, 1e-6)
        self.result_period = 1.0 / max(result_fps, 1e-6)

        self._lock = threading.Lock()
        self._started = False
        self._paused = False
        self._t0 = time.perf_counter()

        self._frame: Optional[FramePair] = None
        self._frame_seq = 0
        self._frame_next_t = 0.0

        self._result: Optional[DepthResult] = None
        self._result_seq = 0
        self._result_next_t = 0.0
        self._consumed = 0

        # 可写参数（与真 Session 的键一致）
        self._params: Dict[str, Any] = {
            "zmin": 0.15, "zmax": 3.0, "point_size": 2.0, "voxel": 0.0,
            "preset": "balanced", "scale": 0.75,
        }
        self._error: Optional[str] = None

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        with self._lock:
            self._started = True
            self._t0 = time.perf_counter()
            self._frame_next_t = 0.0
            self._result_next_t = 0.0
        # 第一帧/第一个结果先做好，调用方 start() 之后立刻就有数据
        self._maybe_produce()
        self._maybe_produce_result()

    def stop(self) -> None:
        with self._lock:
            self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def version(self) -> str:
        return SDK_VERSION

    # ------------------------------------------------------------------ 取数据
    def latest_frame(self) -> Tuple[Optional[FramePair], int]:
        self._maybe_produce()
        self._maybe_produce_result()
        with self._lock:
            return self._frame, self._frame_seq

    def has_new_result(self) -> bool:
        self._maybe_produce_result()
        with self._lock:
            return self._result is not None and self._result_seq > self._consumed

    def take_result(self) -> Optional[DepthResult]:
        self._maybe_produce_result()
        with self._lock:
            self._consumed = self._result_seq
            return self._result

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "capture_fps": 1.0 / self.frame_period,
                "infer_fps": 1.0 / self.result_period,
                "frames": float(self._frame_seq),
                "results": float(self._result_seq),
                "valid_ratio": 95.0,
                "median_m": 0.62,
                "preset": self._params.get("preset", "balanced"),
                "mock": 1.0,
            }

    def last_error(self) -> Optional[str]:
        return self._error

    # ------------------------------------------------------------------ 控制
    def get_params(self) -> Dict[str, Any]:
        return dict(self._params)

    def set_params(self, **kw: Any) -> Dict[str, Any]:
        applied, restart = {}, []
        for k, v in kw.items():
            if k in RESTART_PARAMS:
                restart.append(k)          # Mock 不真的重启，只按契约返回
                continue
            if k == "depth_range" and isinstance(v, (tuple, list)) and len(v) == 2:
                self._params["zmin"], self._params["zmax"] = float(v[0]), float(v[1])
                applied["depth_range"] = list(v)
            elif k in self._params or k in ("zmin", "zmax", "point_size", "voxel", "preset"):
                self._params[k] = v
                applied[k] = v
        return {"applied": applied, "requires_restart": restart}

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def save_snapshot(self, tags: Optional[Dict[str, Any]] = None) -> str:
        """Mock 只返回一个假路径（不写盘，避免污染仓库）。"""
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"<mock>/output/captures/{ts}_cloud (MockSession 不落盘)"

    # ------------------------------------------------------------------ 合成
    def _maybe_produce(self) -> None:
        now = time.perf_counter() - self._t0
        if now < self._frame_next_t:
            return
        with self._lock:
            if now < self._frame_next_t:
                return
            self._frame_next_t = now + self.frame_period
            t = now
            self._frame_seq += 1
            frame = self._make_frame(t, self._frame_seq)
            self._frame = frame

    def _maybe_produce_result(self) -> None:
        now = time.perf_counter() - self._t0
        if now < self._result_next_t:
            return
        with self._lock:
            if now < self._result_next_t:
                return
            self._result_next_t = now + self.result_period
            self._result_seq += 1
            self._result = self._make_result(now, self._result_seq)

    def _scene(self, t: float) -> Tuple[float, float, float]:
        """(墙面深度, 方块深度, 方块中心 x 像素)"""
        z_wall = 0.65 + 0.10 * float(np.sin(t * 0.8))
        z_box = z_wall - 0.08
        u_box = 320 + 180 * float(np.sin(t * 1.3))
        return z_wall, z_box, u_box

    def _make_frame(self, t: float, seq: int) -> FramePair:
        h, w = self.height, self.width
        z_wall, _z_box, u_box = self._scene(t)
        # 背景：竖直渐变 + 轻微噪声（模拟纹理）
        gx = np.linspace(0, 60, w, dtype=np.float32)[None, :]
        base = 90.0 + gx + 8.0 * np.sin(np.linspace(0, 6.28, h, dtype=np.float32))[:, None]
        base = base + np.random.default_rng(seq).normal(0, 1.5, (h, w)).astype(np.float32)
        img = np.clip(base, 0, 255).astype(np.uint8)
        bgr = np.dstack([img, img, img]).copy()
        # 方块（比背景亮），左图在 u_box，右图整体左移一点（模拟视差）
        bw, bh = int(120 * (0.65 / max(z_wall, 1e-6))), 110
        disp_box = int(round(_FX * _BASELINE_M / 0.5))
        for img_i, du in ((0, 0), (1, disp_box)):
            x0 = int(u_box) + du - bw // 2
            y0 = (h - bh) // 2
            x0 = int(np.clip(x0, 0, w - 1))
            x1 = int(np.clip(x0 + bw, 0, w))
            y1 = int(np.clip(y0 + bh, 0, h))
            bgr[y0:y1, x0:x1] = (60, 200, 200)
        return FramePair(left=bgr, right=bgr.copy(), t_left=t, t_right=t,
                         seq=seq, mode="stereo", synced=True)

    def _make_result(self, t: float, seq: int) -> DepthResult:
        h, w = self.height, self.width
        z_wall, z_box, u_box = self._scene(t)
        # 深度图：墙 + 方块
        depth = np.full((h, w), z_wall, dtype=np.float32)
        bw, bh = int(120 * (0.65 / max(z_wall, 1e-6))), 110
        x0 = int(np.clip(int(u_box) - bw // 2, 0, w - 1))
        x1 = int(np.clip(x0 + bw, 0, w))
        y0, y1 = (h - bh) // 2, (h - bh) // 2 + bh
        depth[int(y0):int(y1), int(x0):int(x1)] = z_box
        # 5% 随机无效（模拟无纹理区域）
        rng = np.random.default_rng(1234 + seq)
        valid = rng.random((h, w)) > 0.05
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        # 视差（与深度自洽）：disp = f*B/z
        disp = np.where(valid, _FX * _BASELINE_M / np.maximum(depth, 1e-6), 0.0).astype(np.float32)
        # ⚠️ 契约里 disparity 是**模型分辨率**（scale=0.75 时是 360x480），
        #    这里故意按 scale 降采样，逼 GUI 端处理"视差尺寸 != 图像尺寸"的情况，
        #    否则联调时切到 0.5 档就会崩。
        sc = float(self._params.get("scale", 0.75) or 1.0)
        if 0 < sc < 1.0:
            hy, wx = int(round(h * sc)), int(round(w * sc))
            iy = np.clip((np.arange(hy) / sc).astype(int), 0, h - 1)
            ix = np.clip((np.arange(wx) / sc).astype(int), 0, w - 1)
            disp = disp[np.ix_(iy, ix)].astype(np.float32)
        # 点云：每 2 像素取一个，颜色取左图
        step = 2
        vs, us = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step), indexing="ij")
        z = depth[vs, us]
        m = z > 0
        z = z[m]
        u = us[m].astype(np.float32)
        v = vs[m].astype(np.float32)
        x = (u - _CX) * z / _FX
        y = (v - _CY) * z / _FY
        pts = np.column_stack([x, y, z]).astype(np.float32)
        frame = self._frame if self._frame is not None else self._make_frame(t, seq)
        cols = frame.left[vs[m], us[m]].astype(np.uint8)
        timings = {"capture": 1.0, "rectify": 0.6, "infer": 95.0, "cloud": 10.0, "display": 6.0}
        return DepthResult(frame=frame, disparity=disp, depth=depth, valid=valid,
                           points=pts, colors=cols, timings=timings, seq=seq,
                           preset=str(self._params.get("preset", "balanced")),
                           scale=float(self._params.get("scale", 0.75) or 0.75))
