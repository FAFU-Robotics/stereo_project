"""stereo_sdk.session —— 真 Session（接口见 docs/接口约定_SDK.md）

=============================================================================
【它做什么】
    把项目里现成的模块组装起来，并**只暴露契约里的那几个方法**：
      · 采集 + 推理 + 深度 + 点云 全部跑在后台线程（pipeline_threaded）
      · 后处理（视差→深度、上采样、点云构建）也在**推理线程**里完成
        → 所以 take_result() 拿到的是成品，**UI 线程永远不会被阻塞**
      · 相机/模型的错误只写进 last_error()，不抛进 UI 线程

【必须记住】
    第一次 start() 会阻塞数秒（开相机 + 加载权重）→ **务必在后台线程调用**。
    界面会先 prime()：左右相机和模型在启动后后台准备好，再点开始串流就只拉起线程。
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

# ---- 路径解析（源码运行 = 项目根；打包成 EXE 后 = exe 所在目录）----
# 原来用 dirname(dirname(__file__))：pip 安装后指向 site-packages、打包后指向 _MEIPASS，
# 两种情况下 config / 权重 / 输出目录都会找错地方。统一交给 app_paths（见其文件头）。
try:
    from app_paths import data_root as _data_root, data as _data_path, resource as _resource
except ImportError:                    # app_paths 缺失时的兜底（行为与改造前一致）
    _ROOT_FALLBACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _data_root() -> str:
        return _ROOT_FALLBACK

    def _data_path(rel: str) -> str:
        return os.path.normpath(os.path.join(_ROOT_FALLBACK, rel))

    def _resource(rel: str) -> str:
        return os.path.normpath(os.path.join(_ROOT_FALLBACK, rel))

PROJECT_ROOT = _data_root()

#: config.yaml 的 project 段里，哪些是"程序要写"的（必须落在可写根），哪些是"只读资源"
_WRITABLE_PROJECT_KEYS = ("output_dir", "calibration_dir")

#: 预设 → (scale, valid_iters)。
# precise（1.0/8）已去掉：五组已知距离上平均误差 0.18%，与平衡 0.19%、快速 0.19%
# 几乎相同，最大误差还不比平衡小；全分辨率却把误匹配变成飞点。
PRESETS: Dict[str, tuple] = {"balanced": (0.75, 8), "fast": (0.5, 4)}


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
    """把 config 里的相对路径改成绝对路径（与 tools/bench_* 的约定一致）。

    只读资源（ffs_repo / weights_dir）走 resource()：打包后在 _MEIPASS 里也能找到；
    可写数据（output_dir / calibration_dir）走 data()：永远落在 exe 旁边，
    绝不会写进"退出即删"的临时目录。
    """
    project_cfg = dict(cfg.get("project", {}) or {})
    for key in ("ffs_repo", "output_dir", "weights_dir", "calibration_dir"):
        if key not in project_cfg:
            continue
        rel = str(project_cfg[key])
        if os.path.isabs(rel):
            project_cfg[key] = os.path.normpath(rel)
        elif key in _WRITABLE_PROJECT_KEYS:
            project_cfg[key] = _data_path(rel)
        else:
            project_cfg[key] = _resource(rel)
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
        # 打开相机、加载模型、热插拔断开共用一把锁，避免启动和拔线叠在一起。
        self._life = threading.RLock()
        self._cameras_ready = False
        self._model_ready = False
        self._warmed = False
        self._lock_held = False
        self._retain_cameras = False      # prime 成功后为 True：start 失败也别把相机关掉
        self._loss_handled = False
        self._usb_bound: Dict[str, tuple] = {}
        self._sole_device = False
        self._project_cfg: Dict[str, Any] = {}
        self._ffs_cfg: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ 生命周期
    def prime(self, preload_model: bool = True) -> Dict[str, Any]:
        """启动时调用：打开左右相机，可选地把模型加载并预热。不启动采集线程。

        相机已经开着就直接返回。模型预热失败只记 last_error，相机保持打开，
        之后点「开始串流」会再试一次模型。相机本身打不开则释放锁并抛异常。
        """
        with self._life:
            self._ensure_config()
            try:
                if not self._camera_usable():
                    self._open_cameras()
                self._retain_cameras = True
            except BaseException:
                self._retain_cameras = False
                self._stop_unlocked(release_cameras=True, release_model=False)
                raise
            if preload_model and not (self._model_ready and self._warmed):
                try:
                    self._ensure_rectifier()
                    self._load_model_and_warmup()
                except BaseException as exc:  # noqa: BLE001
                    self._error = f"{type(exc).__name__}: {exc}"
            return self.link_status()

    def start(self) -> None:
        """打开相机、加载模型、启动后台流水线。请在后台线程调用。

        谁还没准备好就补谁。相机和模型都要做时并行；prime() 已完成时只拉起线程。
        """
        with self._life:
            if self._started:
                return
            self._start_unlocked()

    def _start_unlocked(self) -> None:
        """调用方必须已持有 self._life。"""
        self._ensure_config()
        from camera_link import plan_stream_start

        self._error = None
        model_err: list = []

        def load_job() -> None:
            try:
                self._load_estimator()
            except BaseException as exc:  # noqa: BLE001
                model_err.append(exc)

        worker = None
        cam_exc: Optional[BaseException] = None
        # prime() 成功过就保留相机；命令行直接 start() 失败时跟以前一样全部释放。
        keep_on_fail = self._retain_cameras
        try:
            plan = plan_stream_start(
                cameras_ready=self._camera_usable(),
                model_ready=bool(self._model_ready and self._est is not None
                                 and getattr(self._est, "loaded", False)),
                warmed=bool(self._warmed and self._est is not None),
            )
            if plan.load_model:
                worker = threading.Thread(target=load_job, name="dsh-model-load", daemon=True)
                worker.start()
            if plan.open_cameras:
                try:
                    self._open_cameras()
                except BaseException as exc:  # noqa: BLE001
                    cam_exc = exc
            self._ensure_rectifier()
            if worker is not None:
                worker.join()
            if cam_exc is not None:
                raise cam_exc
            if model_err:
                raise model_err[0]
            if not self._warmed:
                self._warmup_estimator()
            self._finish_pipeline()
        except BaseException:
            self._abort_start(keep_cameras=keep_on_fail)
            raise

    def _ensure_config(self) -> None:
        if self._cfg and self._ffs_cfg is not None:
            return
        with open(self.config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        # 把 project 里的相对路径写回 cfg。校正器 / 内参读取用的是 cfg 本身，
        # 不是 _resolve_paths 的返回值；不写回的话，工作目录不是程序目录时
        # calibration/stereo_calib.yaml 会找不到。
        project_cfg = _resolve_paths(cfg)
        cfg["project"] = project_cfg
        self._cfg = cfg
        self._project_cfg = project_cfg
        ffs_cfg = dict(cfg.get("ffs", {}) or {})
        self._preset = str(ffs_cfg.pop("preset", "") or "")
        if self._preset in PRESETS:
            ffs_cfg["scale"], ffs_cfg["valid_iters"] = PRESETS[self._preset]
        else:
            self._preset = "custom"
        self._ffs_cfg = ffs_cfg

    def _camera_usable(self) -> bool:
        cam = self._cam
        if cam is None:
            return False
        try:
            return {"left", "right"} <= set(cam.open_sides())
        except Exception:  # noqa: BLE001
            return False

    def _ensure_lock(self) -> None:
        from camera_guard import acquire_camera_lock, lock_held_by_self
        if self._lock_held or lock_held_by_self():
            self._lock_held = True
            return
        acquire_camera_lock()
        self._lock_held = True

    def _release_lock(self) -> None:
        from camera_guard import release_camera_lock
        try:
            release_camera_lock()
        except Exception:  # noqa: BLE001
            pass
        self._lock_held = False

    def _open_cameras(self) -> None:
        """抢锁并打开左右相机。已经开好就直接返回，避免开始串流时再关开一次。"""
        from camera import StereoCamera
        from camera_guard import open_with_retry
        self._ensure_lock()
        if self._cam is None:
            self._cam = StereoCamera((self._cfg.get("camera") or {}))
        if not self._camera_usable():
            open_with_retry(self._cam)
        if not self._camera_usable():
            raise RuntimeError("左右相机没有都打开")
        self._cameras_ready = True
        self._remember_usb()

    def _remember_usb(self) -> None:
        from camera_link import bind_snapshot, list_watched_ids
        cam_cfg = self._cfg.get("camera", {}) or {}
        present = list_watched_ids(str(cam_cfg.get("usb_match") or ""))
        if present is None or self._cam is None:
            self._usb_bound = {"left": (), "right": ()}
            self._sole_device = False
            return
        self._usb_bound = bind_snapshot(
            self._cam.mode, present,
            str(cam_cfg.get("usb_match") or ""),
            str(cam_cfg.get("left_usb") or ""),
            str(cam_cfg.get("right_usb") or ""),
        )
        union = set(self._usb_bound.get("left") or ()) | set(self._usb_bound.get("right") or ())
        self._sole_device = len(union) == 1

    def _ensure_rectifier(self) -> None:
        if self._rect is not None:
            return
        from depth_utils import load_rectifier
        try:
            self._rect = load_rectifier(self._cfg)
        except Exception as exc:  # noqa: BLE001
            self._error = f"校正器加载失败，将直接用原始图推理：{exc}"
            self._rect = None

    def _load_estimator(self):
        from stereo_depth import FFSDepthEstimator
        if self._est is not None and getattr(self._est, "loaded", False):
            self._model_ready = True
            return self._est
        project_cfg = self._project_cfg or _resolve_paths(self._cfg)
        ffs_cfg = dict(self._ffs_cfg or {})
        est = FFSDepthEstimator(project_cfg, ffs_cfg)
        ok, msg = est.weights_ready()
        if not ok:
            raise RuntimeError(f"FFS 权重不可用：{msg}")
        est.load()
        self._est = est
        self._model_ready = True
        self._warmed = False
        return est

    def _load_model_and_warmup(self) -> None:
        self._load_estimator()
        self._warmup_estimator()

    def _warmup_estimator(self) -> None:
        """用真实画面预热。显存不够时降一档再试，逻辑与原来的 start() 相同。"""
        if self._warmed and self._est is not None:
            return
        if self._cam is None:
            raise RuntimeError("相机还没打开，不能预热模型")
        self._load_estimator()
        project_cfg = self._project_cfg or _resolve_paths(self._cfg)
        ffs_cfg = self._ffs_cfg if self._ffs_cfg is not None else {}
        from stereo_depth import FFSDepthEstimator
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
                self._ffs_cfg = ffs_cfg
                self._preset = "fast" if lower[0] <= 0.5 else "balanced"
                self._error = (f"显存不足（{type(exc).__name__}），已自动降档到 "
                               f"scale={lower[0]} iters={lower[1]}；原设置 {old_desc}。"
                               f"如需恢复请调小 ffs.scale 或关闭占用显存的程序。")
                try:
                    self._est.close()
                except Exception:  # noqa: BLE001
                    pass
                self._est = FFSDepthEstimator(project_cfg, dict(ffs_cfg))
                ok, msg = self._est.weights_ready()
                if not ok:
                    raise RuntimeError(f"FFS 权重不可用：{msg}")
                self._est.load()
        self._model_ready = True
        self._warmed = True

    def _finish_pipeline(self) -> None:
        from depth_utils import CameraParams, disp_correction_from_config
        from pipeline_threaded import ThreadedPipeline
        from pointcloud import PointCloudBuilder

        cfg = self._cfg
        project_cfg = self._project_cfg or _resolve_paths(cfg)
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
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:  # noqa: BLE001
                pass
        self._pipe = ThreadedPipeline(self._cam, self._est, rectifier=self._rect,
                                      post_fn=self._postprocess).start()
        self._loss_handled = False
        self._started = True
        # OOM 自动降档的提示是在预热里写进 _error 的，成功后不能清掉。
        self._error = self._error or None

    def _abort_start(self, keep_cameras: bool) -> None:
        """start 中途失败。命令行路径把相机、模型和锁都还回去；界面预打开过的相机留下。"""
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:  # noqa: BLE001
                pass
            self._pipe = None
        self._started = False
        if keep_cameras:
            self._cameras_ready = self._camera_usable()
            return
        self._stop_unlocked(release_cameras=True, release_model=True)

    def _remaining_sides(self) -> list:
        cam = self._cam
        if cam is None:
            return []
        try:
            return list(cam.open_sides())
        except Exception:  # noqa: BLE001
            return []

    def _apply_plan_unlocked(self, plan) -> Dict[str, Any]:
        """关掉 plan 里的那几路，必要时停掉串流，并让锁和 started 跟剩余句柄一致。"""
        closed: list = []
        cam = self._cam
        streaming = self._started
        if cam is not None:
            for side in plan.close_sides:
                try:
                    closed.extend(cam.disconnect_side(side))
                except Exception:  # noqa: BLE001
                    if side not in closed:
                        closed.append(side)
        if streaming and plan.software_disconnect:
            if self._pipe is not None:
                try:
                    self._pipe.stop()
                except Exception:  # noqa: BLE001
                    pass
                self._pipe = None
            self._started = False
        remaining = self._remaining_sides()
        if not remaining:
            if cam is not None:
                try:
                    cam.close()
                except Exception:  # noqa: BLE001
                    pass
            self._cam = None
            self._cameras_ready = False
            self._retain_cameras = False
            self._usb_bound = {}
            self._release_lock()
        else:
            self._cameras_ready = {"left", "right"} <= set(remaining)
            self._retain_cameras = True
        if plan.reason:
            self._error = plan.reason
        self._loss_handled = True
        uniq: list = []
        for side in closed:
            if side not in uniq:
                uniq.append(side)
        return {
            "changed": bool(uniq) or bool(streaming and plan.software_disconnect),
            "closed": uniq,
            "remaining": remaining,
            "streaming_stopped": bool(streaming and plan.software_disconnect),
            "reason": plan.reason,
            "needs_probe": False,
        }

    def apply_usb_removal(self, removed_ids) -> Dict[str, Any]:
        """USB 实例消失时调用。确认是本机相机才关对应连接，并在串流中走软件断开。"""
        with self._life:
            return self._apply_usb_unlocked(removed_ids)

    def _apply_usb_unlocked(self, removed_ids) -> Dict[str, Any]:
        from camera_link import DisconnectPlan, plan_usb_disconnect
        empty = {"changed": False, "closed": [], "remaining": self._remaining_sides(),
                 "streaming_stopped": False, "reason": "", "needs_probe": False}
        cam = self._cam
        if cam is None:
            return empty
        cam_cfg = (self._cfg.get("camera") or {}) if self._cfg else {}
        mode = getattr(cam, "mode", "side_by_side")
        open_sides = list(cam.open_sides())
        plan = plan_usb_disconnect(
            mode=mode,
            open_sides=open_sides,
            bound=self._usb_bound,
            removed_ids=list(removed_ids or ()),
            usb_match=str(cam_cfg.get("usb_match") or ""),
            streaming=self._started,
            sole_device=self._sole_device,
        )
        if plan.needs_probe:
            if self._started:
                return {"changed": False, "needs_probe": True, "reason": plan.reason,
                        "closed": [], "remaining": open_sides, "streaming_stopped": False}
            dead: list = []
            try:
                dead = list(cam.dead_sides())
                if not dead:
                    dead = list(cam.probe_sides())
            except Exception as exc:  # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"
                dead = list(open_sides)
            if not dead:
                return empty
            plan = DisconnectPlan(tuple(dead), True, False, "USB 设备断开，读帧确认相机已掉线")
        if not plan.close_sides:
            return empty
        return self._apply_plan_unlocked(plan)

    def force_disconnect(self, reason: str) -> Dict[str, Any]:
        """采集卡住或画面已经不对时，停掉串流并释放相机句柄。

        拔掉 USB 后 DirectShow 经常还在返回旧画面，读帧不会抛错。
        这种时候必须把句柄关掉，否则再插上仍会用着已经失效的设备。
        相机已经关掉、串流也停了时再叫一次是空操作。
        """
        with self._life:
            empty = {"changed": False, "closed": [], "remaining": self._remaining_sides(),
                     "streaming_stopped": False, "reason": "", "needs_probe": False}
            if self._cam is None and not self._started:
                return empty
            from camera_link import DisconnectPlan
            sides = tuple(self._remaining_sides())
            if not sides and (self._cam is not None or self._started):
                sides = ("left", "right")
            if not sides:
                return empty
            info = self._apply_plan_unlocked(DisconnectPlan(tuple(sides), True, False, reason))
            if self._cam is not None and not self._remaining_sides():
                try:
                    self._cam.close()
                except Exception:  # noqa: BLE001
                    pass
                self._cam = None
                self._cameras_ready = False
                self._retain_cameras = False
                self._usb_bound = {}
                self._release_lock()
                info = dict(info)
                info["changed"] = True
                info["remaining"] = []
            return info

    def poll_runtime_loss(self) -> bool:
        """采集线程已经因读失败停了、但软件侧还没做断开。"""
        with self._life:
            pipe = self._pipe
            return bool(pipe is not None and pipe.error is not None
                        and not self._loss_handled and self._started)

    def consume_runtime_loss(self) -> Dict[str, Any]:
        """把采集线程的读失败收成和 USB 拔出同一套断开流程。只生效一次。"""
        with self._life:
            empty = {"changed": False, "closed": [], "remaining": self._remaining_sides(),
                     "streaming_stopped": False, "reason": "", "needs_probe": False}
            pipe = self._pipe
            if pipe is None or pipe.error is None or self._loss_handled:
                return empty
            err = pipe.error
            self._loss_handled = True
            msg = f"{type(err).__name__}: {err}"
            from camera_link import DisconnectPlan, sides_from_read_error
            mode = self._cam.mode if self._cam is not None else "side_by_side"
            open_sides = set(self._cam.open_sides()) if self._cam is not None else {"left", "right"}
            if mode == "side_by_side":
                sides = tuple(s for s in ("left", "right") if s in open_sides) or ("left", "right")
            else:
                sides = tuple(s for s in sides_from_read_error(str(err), mode) if s in open_sides)
                if not sides:
                    sides = tuple(open_sides) or ("left", "right")
            return self._apply_plan_unlocked(DisconnectPlan(sides, True, False, msg))

    @property
    def cameras_ready(self) -> bool:
        return self._camera_usable()

    @property
    def model_ready(self) -> bool:
        return bool(self._model_ready and self._est is not None)

    def link_status(self) -> Dict[str, Any]:
        sides = set(self._remaining_sides())
        detail = ""
        cam = self._cam
        if cam is not None:
            try:
                detail = cam.describe()
            except Exception:  # noqa: BLE001
                detail = ""
        return {
            "left": "left" in sides,
            "right": "right" in sides,
            "model": self.model_ready,
            "warmed": bool(self._warmed),
            "started": self._started,
            "detail": detail,
            "error": self._error,
        }

    # ------------------------------------------------------------ 上下文管理
    def __enter__(self) -> "Session":
        """`with Session(...) as cam:` —— 退出时一定释放相机与推理线程。"""
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
        return None                      # 不吞异常

    def reload_calibration(self) -> Dict[str, Any]:
        """重新读取标定文件，刷新校正器 / 内参 / 点云构建器（**不重启相机、不重载模型**）。

        存在意义：start() 要 3~10 秒并重新加载 FFS 权重；标定 GUI 刚保存完参数时，
        用户希望"立刻生效"。本方法只重读 YAML 并替换引用，通常在几十毫秒内完成。

        生效机制：推理线程每帧读取 self.rectifier 与 self._params，
        因此替换引用后**下一帧**就使用新参数，无需停止线程。

        返回：
            {"rectified", "source", "baseline_m", "rect_fx", "rms_stereo_px", "image_size"}
        异常：
            RuntimeError —— 未 start()、标定文件读不到、或图像尺寸与当前分辨率不一致
                            （抛异常时**原有参数保持不变**，不会出现半更新状态）。
        """
        if not self._started:
            raise RuntimeError("Session 尚未 start()，无需 reload_calibration()")

        with open(self.config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        from depth_utils import (CameraParams, disp_correction_from_config,
                                 load_rectifier)
        from pointcloud import PointCloudBuilder

        try:
            rect = load_rectifier(cfg)
        except Exception as exc:                    # noqa: BLE001
            self._error = f"重载校正器失败：{exc}"
            raise RuntimeError(f"重载校正器失败：{exc}") from exc

        eye = self._cam.eye_size
        if rect is not None:
            size = tuple(getattr(rect, "image_size", (0, 0)) or (0, 0))
            if len(size) == 2 and size[0] and size[1] and \
                    (int(size[0]), int(size[1])) != (int(eye[0]), int(eye[1])):
                raise RuntimeError(
                    f"标定文件图像尺寸 {int(size[0])}x{int(size[1])} 与相机实际 "
                    f"{int(eye[0])}x{int(eye[1])} 不一致，这份标定不能用于当前分辨率")

        params = CameraParams.from_config(cfg, eye_size=eye)
        builder = PointCloudBuilder(dict(cfg.get("pointcloud", {}) or {}),
                                    calibrated=params.calibrated)
        depth_cfg = cfg.get("depth", {}) or {}
        scale, offset = disp_correction_from_config(cfg)
        ratio = float(depth_cfg.get("disp_extrapolation_ratio", 0.98) or 0.0)

        # ---- 以下只做引用替换（原子操作），全部校验通过后才执行
        self._cfg = cfg
        self._rect = rect
        self._params = params
        self._builder = builder
        self._disp_scale, self._disp_offset = scale, offset
        self._max_valid = (self._est.max_disp * ratio) if (ratio > 0 and self._est) else None
        self._zmin = float(depth_cfg.get("zmin_m", self._zmin) or self._zmin)
        self._zmax = float(depth_cfg.get("zmax_m", self._zmax) or self._zmax)
        if self._pipe is not None:
            self._pipe.rectifier = rect
        self._error = None

        return {
            "rectified": rect is not None,
            "source": str(depth_cfg.get("calibration_file", "")),
            "baseline_m": float(rect.baseline_m) if rect is not None else
                          float(getattr(params, "baseline_m", 0.0)),
            "rect_fx": float(rect.rect_fx) if rect is not None else float(params.fx),
            "rms_stereo_px": float(getattr(rect, "rms_stereo", 0.0)) if rect is not None else 0.0,
            "image_size": [int(eye[0]), int(eye[1])],
        }

    def stop(self, release_cameras: bool = True, release_model: bool = True) -> None:
        """幂等。默认释放相机、模型和锁（命令行 / with 语句的原行为）。

        界面点「停止串流」传 release_cameras=False, release_model=False：
        线程停掉，相机和模型留着，下一次开始串流只拉起线程。
        打开标定前要 release_cameras=True，把设备交出去，模型可以留在显存里。
        """
        with self._life:
            self._stop_unlocked(release_cameras, release_model)

    def _stop_unlocked(self, release_cameras: bool = True, release_model: bool = True) -> None:
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:  # noqa: BLE001
                pass
            self._pipe = None
        if release_cameras:
            if self._cam is not None:
                try:
                    self._cam.close()
                except Exception:  # noqa: BLE001
                    pass
            self._cam = None
            self._cameras_ready = False
            self._usb_bound = {}
            self._sole_device = False
            self._retain_cameras = False
            self._release_lock()
        if release_model:
            if self._est is not None:
                try:
                    self._est.close()
                except Exception:  # noqa: BLE001
                    pass
            self._est = None
            self._model_ready = False
            self._warmed = False
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
        # 飞点和散斑在三档里都会出现，全分辨率（旧 precise）最明显。
        # 先按邻域中值去掉跳变，再按面积去掉孤立小块。阈值随视差图面积缩放。
        from depth_utils import filter_disp_spikes, filter_speckles, scaled_speckle_size
        valid = filter_disp_spikes(disp, valid)
        depth_cfg = (self._cfg.get("depth") or {}) if self._cfg else {}
        speckle_base = int(depth_cfg.get("min_speckle_size", 200) or 0)
        valid = filter_speckles(valid, scaled_speckle_size(valid.shape, speckle_base))
        depth = np.where(valid, depth, np.float32(0.0))
        # 点云用**模型分辨率**的深度构建（实测 21.6ms/307200 点 -> 4ms/76800 点，
        # 快约 5 倍；点云精度不受影响，因为上采样只是复制像素）。
        t0 = time.perf_counter()
        from pointcloud import bgr_u8_from_open3d, cloud_color_bgr
        # 真实色彩：点的颜色取校正后的左图原像素（BGR），不走模型 RGB。
        # 模型图再交换一次红蓝时，蓝色物体会画成红色。
        pc_cfg = (self._cfg.get("pointcloud") or {}) if self._cfg else {}
        truecolor = str(pc_cfg.get("color_mode", "truecolor") or "truecolor").lower() != "model"
        if truecolor:
            color_bgr = cloud_color_bgr(frame.left, None, depth.shape)
        else:
            color_bgr = cloud_color_bgr(None, out.get("left_model_rgb"), depth.shape)
        pcd, _stats = self._builder.build(depth, p_use, color_bgr)
        cloud_ms = (time.perf_counter() - t0) * 1000.0
        pts = np.asarray(pcd.points, dtype=np.float32) if len(pcd.points) else np.zeros((0, 3), np.float32)
        cols = getattr(self._builder, "last_colors_bgr", None)
        if cols is None or len(cols) != len(pts):
            cols = bgr_u8_from_open3d(pcd.colors)
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
            self._model_ready = True
            # 新模型还没预热。停掉再开始时 start() 会补这一步，避免用冷启动的第一帧当结果。
            self._warmed = False
            self._loss_handled = False
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
