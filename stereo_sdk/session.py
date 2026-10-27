"""stereo_sdk.session —— 真 Session（接口见 docs/接口约定_SDK.md）

=============================================================================
【它做什么】
    把项目里现成的模块组装起来，并**只暴露契约里的那几个方法**：
      · 采集 + 推理 + 深度 + 点云 全部跑在后台线程（pipeline_threaded）
      · 后处理（视差→深度、上采样、点云构建）也在**推理线程**里完成
        → 所以 take_result() 拿到的是成品，**UI 线程永远不会被阻塞**
      · 相机/模型的错误只写进 last_error()，不抛进 UI 线程

【必须记住】
    start() 会阻塞 3~10 秒（开相机 + 加载 67MB 权重）→ **务必在后台线程调用**。
    切换 preset 同样会重建 estimator（约 2 秒），也是在调用线程里阻塞。
=============================================================================
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

from .types import (HOT_PARAMS, RESTART_PARAMS, SDK_VERSION, DepthResult, FramePair)
from profiler import Profiler  # 统一阶段耗时统计（任务1）
from .recorder import Recorder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 预设 → (scale, valid_iters)。数值来自 tools/bench_presets.py 的实测（速度/精度）
PRESETS: Dict[str, tuple] = {"precise": (1.0, 8), "balanced": (0.75, 8), "fast": (0.5, 4)}


def _is_oom(exc: BaseException) -> bool:
    """粗判显存不足（不 import torch，避免在没 GPU 的环境里额外拉 CUDA）。
    同时认 torch 的 OutOfMemoryError 与各种 'out of memory' 文本。"""
    txt = f"{type(exc).__name__}: {exc}".lower()
    return ("outofmemory" in txt) or ("out of memory" in txt)


def _next_lower(scale: float):
    """降一档: 1.0 -> (0.75, 8) -> (0.5, 4) -> None"""
    if scale > 0.9:
        return (0.75, 8)
    if scale > 0.6:
        return (0.5, 4)
    return None


def _resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把 config 里的相对路径改成绝对路径（与 tools/bench_* 的约定一致）。"""
    project_cfg = dict(cfg.get("project", {}) or {})
    for key in ("ffs_repo", "output_dir", "weights_dir", "calibration_dir"):
        if key in project_cfg:
            project_cfg[key] = os.path.normpath(os.path.join(PROJECT_ROOT, project_cfg[key]))
    return project_cfg


class Session:
    """真实相机 + FFS 的深度相机会话。"""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path = config_path if os.path.isabs(config_path) else \
            os.path.join(PROJECT_ROOT, config_path)
        self._cfg: Dict[str, Any] = {}
        self._cam = None
        self._est = None
        self._rect = None
        self._params = None
        self._builder = None
        self._pipe = None
        self._recorder: Optional[Recorder] = None
        self._disp_scale, self._disp_offset = 1.0, 0.0
        self._max_valid: Optional[float] = None

        self._lock = threading.Lock()
        self._result: Optional[DepthResult] = None
        self._result_seq = 0
        self._consumed = 0
        self._error: Optional[str] = None
        self._started = False
        self._paused = False
        self._preset = "balanced"
        self._zmin, self._zmax = 0.15, 3.0
        self._infer_ms = deque(maxlen=30)
        self.prof = Profiler(window=60)   # capture/rectify/infer/cloud/display 滚动统计
        self._t_infer = 0.0

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        """打开相机、加载模型、启动后台流水线。**阻塞 3~10 秒，请在后台线程调用。**"""
        if self._started:
            return
        with open(self.config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self._cfg = cfg
        project_cfg = _resolve_paths(cfg)

        # 延迟导入：只有真正 start() 才需要 torch / opencv 相机模块
        from camera import StereoCamera
        from depth_utils import (CameraParams, disp_correction_from_config,
                                 disparity_to_depth, load_rectifier)
        from pipeline_threaded import ThreadedPipeline
        from pointcloud import PointCloudBuilder
        from stereo_depth import FFSDepthEstimator

        ffs_cfg = dict(cfg.get("ffs", {}) or {})
        self._preset = str(ffs_cfg.pop("preset", "") or "")
        if self._preset in PRESETS:
            ffs_cfg["scale"], ffs_cfg["valid_iters"] = PRESETS[self._preset]
        else:
            self._preset = "custom"

        self._params = None
        # 稳定性加固②: 先抢单实例锁（防两个进程抢同一台相机导致"时好时坏"），
        # 再带重试打开相机（占用类失败会 5 次 x 300ms 退避并给出可读排查步骤）。
        from camera_guard import acquire_camera_lock, open_with_retry, release_camera_lock
        acquire_camera_lock()
        self._cam = StereoCamera(cfg.get("camera", {}) or {})
        try:
            open_with_retry(self._cam)
        except BaseException:
            release_camera_lock()          # 打不开就别占着锁
            raise
        try:
            self._rect = load_rectifier(cfg)
        except Exception as exc:  # noqa: BLE001
            self._error = f"校正器加载失败，将直接用原始图推理：{exc}"
            self._rect = None

        self._est = FFSDepthEstimator(project_cfg, ffs_cfg)
        ok, msg = self._est.weights_ready()
        if not ok:
            self._cam.close()
            raise RuntimeError(f"FFS 权重不可用：{msg}")
        self._est.load()

        # 预热（用真实画面，尺寸必须与正式推理一致）
        # 稳定性⑤: 显存不足时自动降一档预设后重试（小显存机器不至于直接崩）
        first = self._cam.read()
        retried = 0
        while True:
            try:
                self._est.warmup(self._est.warmup_iters, left=first.left, right=first.right)
                break
            except BaseException as exc:  # noqa: BLE001
                lower = _next_lower(float(ffs_cfg.get("scale", 1.0) or 1.0))
                if not _is_oom(exc) or lower is None or retried >= 2:
                    raise
                retried += 1
                old_desc = f"scale={ffs_cfg.get('scale')} iters={ffs_cfg.get('valid_iters')}"
                ffs_cfg["scale"], ffs_cfg["valid_iters"] = lower
                self._preset = "fast" if lower[0] <= 0.5 else "balanced"
                self._error = (f"显存不足（{type(exc).__name__}），已自动降档到 "
                               f"scale={lower[0]} iters={lower[1]}；原设置 {old_desc}。"
                               f"如需恢复请调小 ffs.scale 或关闭占用显存的程序。")
                try:
                    self._est.close()
                except Exception:  # noqa: BLE001
                    pass
                self._est = FFSDepthEstimator(project_cfg, ffs_cfg)
                ok, msg = self._est.weights_ready()
                if not ok:
                    raise RuntimeError(f"FFS 权重不可用：{msg}")
                self._est.load()

        self._params = CameraParams.from_config(cfg, eye_size=self._cam.eye_size)
        pc_cfg = dict(cfg.get("pointcloud", {}) or {})
        self._builder = PointCloudBuilder(pc_cfg, calibrated=self._params.calibrated)
        self._disp_scale, self._disp_offset = disp_correction_from_config(cfg)
        depth_cfg = cfg.get("depth", {}) or {}
        ratio = float(depth_cfg.get("disp_extrapolation_ratio", 0.98) or 0.0)
        self._max_valid = (self._est.max_disp * ratio) if ratio > 0 else None
        self._zmin = float(depth_cfg.get("zmin_m", 0.15) or 0.15)
        self._zmax = float(depth_cfg.get("zmax_m", 3.0) or 3.0)

        out_root = os.path.join(project_cfg.get("output_dir",
                                                os.path.join(PROJECT_ROOT, "output")), "captures")
        self._recorder = Recorder(out_root)

        self._pipe = ThreadedPipeline(self._cam, self._est, rectifier=self._rect,
                                      post_fn=self._postprocess).start()
        self._started = True
        # 注意: 不能无脑清空 _error —— OOM 自动降档等警告是在 start() 里产生的，
        # 清掉的话用户就看不到"已降档"这条重要提示了。只在没有消息时置空。
        self._error = self._error or None

    def stop(self) -> None:
        """幂等；GUI 退出时务必调用。"""
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:  # noqa: BLE001
                pass
        if self._est is not None:
            try:
                self._est.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            from camera_guard import release_camera_lock
            release_camera_lock()
        except Exception:  # noqa: BLE001
            pass
        self._pipe = self._cam = self._est = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def version(self) -> str:
        return SDK_VERSION

    # ------------------------------------------------------------------ 取数据（永不阻塞）
    def latest_frame(self):
        if self._pipe is None:
            return None, 0
        return self._pipe.latest_frame()

    def has_new_result(self) -> bool:
        with self._lock:
            return self._result is not None and self._result_seq > self._consumed

    def take_result(self) -> Optional[DepthResult]:
        with self._lock:
            self._consumed = self._result_seq
            return self._result

    # ------------------------------------------------------------------ 后处理（在推理线程里跑）
    def _postprocess(self, frame: FramePair, out: Dict[str, Any]) -> DepthResult:
        from depth_utils import disparity_to_depth  # 延迟导入，避免构造期依赖

        disp = out["disp"]
        h, w = frame.left.shape[:2]
        p_use = self._params if disp.shape[1] == w else self._params.scaled_to_match(disp.shape[1])
        depth, valid = disparity_to_depth(disp, p_use, max_valid_disp=self._max_valid,
                                          disp_scale=self._disp_scale,
                                          disp_offset=self._disp_offset)
        # 点云用**模型分辨率**的深度构建（实测 21.6ms/307200 点 -> 4ms/76800 点，
        # 快约 5 倍；点云精度不受影响，因为上采样只是复制像素）。
        t0 = time.perf_counter()
        rgb_model = out.get("left_model_rgb")
        if rgb_model is None or tuple(rgb_model.shape[:2]) != tuple(depth.shape[:2]):
            # 兜底：模型没回传模型分辨率的左图时，自己缩一张 —— 必须与深度同尺寸，
            # 否则 build() 里逐像素取色会因形状不一致抛异常（自检抓到过）。
            rgb_model = cv2.resize(frame.left, (depth.shape[1], depth.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        pcd, _stats = self._builder.build(depth, p_use, rgb_model)
        cloud_ms = (time.perf_counter() - t0) * 1000.0
        pts = np.asarray(pcd.points, dtype=np.float32) if len(pcd.points) else np.zeros((0, 3), np.float32)
        cols = (np.asarray(pcd.colors, dtype=np.float32) * 255.0).astype(np.uint8) \
            if len(pcd.colors) else np.zeros((0, 3), np.uint8)
        # 上采样到全分辨率（契约要求 depth 与 left 同尺寸）—— 放在点云之后，
        # 这样点云不必处理 4 倍的像素。
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            valid = cv2.resize(valid.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        timings = {k: float(v) for k, v in (out.get("timings_ms") or {}).items()}
        timings["cloud"] = cloud_ms
        # 统一统计（缺失的阶段 tick 会自动丢弃，不会记成 0）
        self.prof.tick("capture", timings.get("capture"))
        self.prof.tick("rectify", timings.get("rectify"))
        self.prof.tick("infer", timings.get("infer"))
        self.prof.tick("cloud", cloud_ms)
        self.prof.mark("frame")
        timings.setdefault("capture", 0.0)
        timings.setdefault("display", 0.0)
        res = DepthResult(frame=frame, disparity=np.asarray(disp, np.float32),
                          depth=depth, valid=valid, points=pts, colors=cols,
                          timings=timings, seq=int(frame.seq), preset=self._preset,
                          scale=float(out.get("scale", 1.0) or 1.0))
        with self._lock:
            self._result = res
            self._result_seq += 1
            self._infer_ms.append(timings.get("infer", 0.0))
        return res

    # ------------------------------------------------------------------ 控制
    def get_params(self) -> Dict[str, Any]:
        return {"zmin": self._zmin, "zmax": self._zmax, "preset": self._preset,
                "scale": self._result.scale if self._result is not None else None,
                "paused": self._paused, "resolution": self._resolution_str(),
                "backend": (self._cfg.get("camera", {}) or {}).get("backend", "")}

    def _resolution_str(self) -> str:
        if self._cam is None:
            return ""
        try:
            w, h = self._cam.eye_size
            return f"{w * 2}x{h}"
        except Exception:  # noqa: BLE001
            return ""

    def set_params(self, **kw: Any) -> Dict[str, Any]:
        applied: Dict[str, Any] = {}
        restart = []
        for k, v in kw.items():
            if k in RESTART_PARAMS:
                restart.append(k)
                continue
            if k == "depth_range" and isinstance(v, (tuple, list)) and len(v) == 2:
                self._zmin, self._zmax = float(v[0]), float(v[1])
                applied["depth_range"] = [self._zmin, self._zmax]
            elif k == "zmin":
                self._zmin = float(v); applied["zmin"] = self._zmin
            elif k == "zmax":
                self._zmax = float(v); applied["zmax"] = self._zmax
            elif k == "paused":
                self._paused = bool(v); applied["paused"] = self._paused
            elif k == "preset":
                name = str(v)
                if name not in PRESETS:
                    continue
                if self._switch_preset(name):
                    applied["preset"] = name
            # point_size / voxel 属于纯显示参数，由 GUI 自己保存即可
            elif k in ("point_size", "voxel"):
                applied[k] = v
        return {"applied": applied, "requires_restart": restart}

    def _switch_preset(self, name: str) -> bool:
        """切换推理档（会重建 estimator，阻塞约 2 秒 —— 请在后台线程调用）。"""
        if name == self._preset:
            return True
        if not self._started:
            self._preset = name
            return True
        stop_pipe = self._pipe
        try:
            if stop_pipe is not None:
                stop_pipe.stop()
            scale, iters = PRESETS[name]
            ffs_cfg = dict(self._cfg.get("ffs", {}) or {})
            ffs_cfg["scale"], ffs_cfg["valid_iters"] = scale, iters
            from pipeline_threaded import ThreadedPipeline
            from stereo_depth import FFSDepthEstimator
            self._est.close()
            self._est = FFSDepthEstimator(_resolve_paths(self._cfg), ffs_cfg)
            ok, msg = self._est.weights_ready()
            if not ok:
                raise RuntimeError(msg)
            self._est.load()
            self._pipe = ThreadedPipeline(self._cam, self._est, rectifier=self._rect,
                                          post_fn=self._postprocess).start()
            self._preset = name
            return True
        except Exception as exc:  # noqa: BLE001
            self._error = f"切换预设失败：{type(exc).__name__}: {exc}"
            return False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def save_snapshot(self, tags: Optional[Dict[str, Any]] = None) -> str:
        with self._lock:
            res = self._result
        if res is None or self._recorder is None:
            return ""
        extra = {"preset": self._preset, "sdk_version": SDK_VERSION,
                 "zmin": self._zmin, "zmax": self._zmax}
        if tags:
            extra.update(tags)
        return self._recorder.save(res, meta_extra=extra)

    # ------------------------------------------------------------------ 统计
    def stats(self) -> Dict[str, Any]:
        pipe = self._pipe.stats() if self._pipe is not None else {}
        with self._lock:
            res = self._result
            infer_ms = (sum(self._infer_ms) / len(self._infer_ms)) if self._infer_ms else 0.0
        out: Dict[str, Any] = {"capture_fps": pipe.get("capture_fps", 0.0),
                               "infer_fps": pipe.get("infer_fps", 0.0),
                               "frames": pipe.get("frames", 0.0),
                               "results": pipe.get("results", 0.0),
                               "infer_ms": infer_ms, "preset": self._preset,
                               "paused": self._paused, "started": self._started}
        if res is not None:
            out["valid_ratio"] = float(res.valid.mean()) * 100.0 if res.valid.size else 0.0
            out["median_m"] = float(np.median(res.depth[res.valid])) if res.valid.any() else 0.0
            out["points"] = int(len(res.points))
        out["stages"] = self.prof.summary()   # {stage: {n,mean,p50,p95,max}}
        return out

    def last_error(self) -> Optional[str]:
        if self._error:
            return self._error
        if self._pipe is not None and self._pipe.error is not None:
            return f"{type(self._pipe.error).__name__}: {self._pipe.error}"
        return None
