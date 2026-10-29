"""runtime.py —— CLI 显示运行时（任务 1 的「编排层」第二步，只新增文件）

=============================================================================
【为什么需要它】
    main.py 里三个 run_* 各自复制了一遍"取帧 → 推理 → 拼面板 → 按键处理 → 统计"。
    但采集/校正/推理/深度/点云 已经在 stereo_sdk.Session 里了（已测试），
    所以这里**只做显示循环**，并且**只用 Session 的公开 API**：
        latest_frame / has_new_result / take_result / set_params / stats / save_snapshot
    → CLI 和 GUI 走同一条数据路径，不会出现"GUI 好使、CLI 行为不同"。

【三种模式只差"显示什么"】
    preview    : LEFT | RIGHT
    depth      : LEFT | RIGHT | DISPARITY
    pointcloud : LEFT | RIGHT | DEPTH(伪彩)
    3D 点云窗口仍由调用方（main.py）负责，这里不碰 Open3D。

【键盘】
    q/ESC 退出   i 打印状态   p 保存快照   space 暂停显示
    [ / ] 深度范围 zmin/zmax 缩小/放大      m 循环切换预设（precise/balanced/fast）
    （用 viewer.is_quit() 判定退出，其余按键按字符处理，避免和 viewer 的键冲突）

【自检（不需要相机/GPU）】
    python runtime.py --selftest       # 假相机 + 假模型 + 无窗口跑 30 帧
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from app_paths import data_root as _data_root
except ImportError:                        # 兜底：与改造前行为一致
    def _data_root() -> str:
        return os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = _data_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from depth_utils import attach_range_bar, colorize_depth  # noqa: E402
from profiler import Profiler           # noqa: E402
from logger import get_logger, setup  # noqa: E402  日志落盘（稳定性③）

_PRESET_CYCLE = ("precise", "balanced", "fast")


def disparity_to_bgr(disp: np.ndarray) -> np.ndarray:
    """视差图 → BGR 伪彩（0 视差当无效，涂黑）。自包含实现，不依赖 viewer。"""
    d = np.asarray(disp, np.float32)
    ok = np.isfinite(d) & (d > 0.5)
    if not ok.any():
        return np.zeros((*d.shape, 3), np.uint8)
    lo = float(np.percentile(d[ok], 2))
    hi = float(np.percentile(d[ok], 98))
    rng = max(hi - lo, 1e-6)
    norm = np.clip((d - lo) / rng, 0, 1)
    # Jet：大视差（近）偏红，小视差（远）偏蓝，和深度色标同一套
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[~ok] = (0, 0, 0)
    return attach_range_bar(vis, hi, lo, unit="px")


class DisplayRuntime:
    """在 Session 之上跑显示循环。"""

    def __init__(self, session: Any, viewer: Any = None, mode: str = "depth",
                 no_display: bool = False, max_frames: int = 0,
                 duration: float = 0.0, perf_csv: str = "",
                 perf_every_s: float = 5.0) -> None:
        self.s = session
        self.viewer = viewer
        self.mode = mode if mode in ("preview", "depth", "pointcloud") else "depth"
        self.no_display = bool(no_display) or viewer is None
        self.max_frames = int(max_frames)
        self.duration = float(duration)
        self.perf_csv = perf_csv
        self.perf_every_s = float(perf_every_s)
        self.prof = getattr(session, "prof", None) or Profiler(window=60)
        self.log = get_logger()
        self.paused = False
        self.max_loop_fps = 30.0     # 显示循环上限，避免和推理线程抢 CPU
        # 3D 点云窗口（点云模式 + 有窗口时启用）—— 功能对齐旧链路
        self.vis3d = None
        self.show3d = (self.mode == "pointcloud") and (not self.no_display)
        self._frames = 0
        self._t_start = time.perf_counter()
        self._last_perf = 0.0
        self._params = session.get_params() if hasattr(session, "get_params") else {}

    # ------------------------------------------------------------------ 面板
    def _panels(self, frame: Any, res: Any) -> List[Tuple[str, np.ndarray, List[str]]]:
        panels: List[Tuple[str, np.ndarray, List[str]]] = []
        if frame is not None:
            panels.append(("Color", frame.left, [f"{frame.left.shape[1]}x{frame.left.shape[0]}"]))
            panels.append(("Color R", frame.right, []))
        if self.mode in ("depth", "pointcloud") and res is not None:
            if self.mode == "depth":
                panels.append(("Disparity", disparity_to_bgr(res.disparity),
                               [f"{res.disparity.shape[1]}x{res.disparity.shape[0]}"]))
            else:
                zmin = float(self._params.get("zmin", 0.15) or 0.15)
                zmax = float(self._params.get("zmax", 3.0) or 3.0)
                depth_vis = attach_range_bar(
                    colorize_depth(res.depth, res.valid, zmin, zmax), zmin, zmax, unit="m")
                panels.append(("Depth", depth_vis, [f"{zmin:.2f}-{zmax:.2f} m"]))
        return panels

    def _status(self, res: Any) -> List[str]:
        st = self.s.stats() if hasattr(self.s, "stats") else {}
        lines = [f"frames={self._frames} | capture {st.get('capture_fps', 0):.1f} fps | "
                 f"infer {st.get('infer_fps', 0):.2f} fps | preset {st.get('preset', '?')} | "
                 f"paused={self.paused}"]
        stg = st.get("stages") or {}
        if stg:
            lines.append("stage p50: " + "  ".join(
                f"{k}={stg[k]['p50']:.1f}ms" for k in ("capture", "rectify", "infer", "cloud", "display")
                if k in stg))
        if res is not None:
            lines.append(res.summary())
        elif st:
            lines.append(f"valid {st.get('valid_ratio', 0):.1f}% | median {st.get('median_m', 0):.3f} m "
                         f"| points {st.get('points', 0)}")
        return lines

    # ------------------------------------------------------------------ 按键
    def _handle_key(self, key: int) -> bool:
        """返回 True 表示要继续。"""
        if key < 0:
            return True
        if self.viewer is not None and hasattr(self.viewer, "is_quit") and self.viewer.is_quit(key):
            return False
        ch = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 128 else ""
        if ch == "q" or key == 27:
            return False
        if ch == " ":
            self.paused = not self.paused
        elif ch == "i":
            print(f"  [状态] {self.s.stats()}")
            print(f"  [参数] {self._params}")
        elif ch in ("p", "s"):          # s 是旧链路的保存键，保留习惯
            d = self.s.save_snapshot() if hasattr(self.s, "save_snapshot") else ""
            print(f"  [保存] {d or '（还没有结果，未保存）'}")
        elif ch in "[]":
            zmin = float(self._params.get("zmin", 0.15) or 0.15)
            zmax = float(self._params.get("zmax", 3.0) or 3.0)
            step = 0.05 * max(zmax - zmin, 0.1)
            if ch == "[":
                zmin, zmax = max(0.02, zmin - step), max(zmin + step, zmax - step)
            else:
                zmin, zmax = max(0.02, zmin + step), zmax + step
            self._params.update(self.s.set_params(zmin=zmin, zmax=zmax).get("applied", {}))
        elif ch == "o" and self.mode == "pointcloud":
            self._toggle_3d()
        elif ch == "m":
            cur = str(self._params.get("preset", "balanced"))
            nxt = _PRESET_CYCLE[(_PRESET_CYCLE.index(cur) + 1) % len(_PRESET_CYCLE)] \
                if cur in _PRESET_CYCLE else _PRESET_CYCLE[1]
            t0 = time.perf_counter()
            got = self.s.set_params(preset=nxt)
            print(f"  [预设] {nxt}（切换耗时 {(time.perf_counter()-t0)*1000:.0f} ms）"
                  f" -> {got}")
            self._params.update(got.get("applied", {}))
        return True

    # ------------------------------------------------------------------ 主循环
    def run(self) -> int:
        last_frame, last_res = None, None
        while True:
            t0 = time.perf_counter()
            frame, _seq = self.s.latest_frame()
            if frame is not None:
                last_frame = frame
            if self.s.has_new_result():
                r = self.s.take_result()
                if r is not None:
                    last_res = r
                    self._update_3d(r)
            if self.paused:
                time.sleep(0.03)
            if last_frame is not None or last_res is not None:
                panels = self._panels(last_frame, last_res)
                status = self._status(last_res)
                if not self.no_display and self.viewer is not None and panels:
                    key = self.viewer.show_panels(panels, status_lines=status, paused=self.paused)
                    self.prof.tick("display", (time.perf_counter() - t0) * 1000.0)
                    if not self._handle_key(key):
                        break
                elif self.no_display:
                    # 无窗口模式：按帧上限/时长自动退出（便于自检与压测）
                    pass
            self._frames += 1
            self.prof.mark("frame")
            if self.max_frames and self._frames >= self.max_frames:
                break
            if self.duration and (time.perf_counter() - self._t_start) >= self.duration:
                break
            if self.no_display:
                time.sleep(0.01)
            else:
                # 限帧：显示循环原本会跑满 ~40 FPS，白白和推理线程抢 CPU/GIL，
                # 实测会把推理从 56ms 拖到 64ms。按目标帧率补足剩余时间。
                budget = 1.0 / max(self.max_loop_fps, 1.0)
                used = time.perf_counter() - t0
                if used < budget:
                    time.sleep(budget - used)
            self._maybe_perf()
        if self.vis3d is not None:
            self.vis3d.close()
            self.vis3d = None
        if self.perf_csv:
            self.prof.to_csv(self.perf_csv, self._perf_extra())
        return 0

    # ------------------------------------------------------------------ 3D
    def _update_3d(self, res: Any) -> None:
        """刷新 3D 点云窗口；窗口被用户关掉后自动停用（不报错）。"""
        if not self.show3d or res is None or not len(getattr(res, "points", ())):
            return
        if self.vis3d is None:
            try:
                from view3d import CloudWindow3D
                self.vis3d = CloudWindow3D(point_size=float(self._params.get("point_size", 2.0) or 2.0))
                self.log.info("3D 点云窗口已打开（o 键开关）")
            except Exception as exc:  # noqa: BLE001
                self.show3d = False
                self.log.warning("3D 窗口打不开，已停用：%s", exc)
                return
        if not self.vis3d.update(res.points, res.colors):
            self.vis3d = None
            self.show3d = False
            self.log.info("3D 点云窗口已关闭，后续不再刷新")

    def _toggle_3d(self) -> None:
        if self.vis3d is not None:
            self.vis3d.close()
            self.vis3d = None
            self.show3d = False
        else:
            self.show3d = True

    def _maybe_perf(self) -> None:
        now = time.perf_counter()
        if self.perf_csv and (now - self._last_perf) >= self.perf_every_s:
            self._last_perf = now
            self.prof.to_csv(self.perf_csv, self._perf_extra())
            try:
                extra = self._perf_extra()
                st = self.s.stats() if hasattr(self.s, "stats") else {}
                self.log.info("frames=%d fps=%.2f infer_fps=%.2f rss=%.0fMB vram=%.0fMB | %s",
                              self._frames, self.prof.fps("frame") or 0.0,
                              st.get("infer_fps", 0.0), extra.get("rss_mb", 0) or 0,
                              extra.get("vram_mb", 0) or 0, self.prof.summary_text())
            except Exception:  # noqa: BLE001 —— 日志不允许拖垮主循环
                pass

    def _perf_extra(self) -> Dict[str, Any]:
        """perf.csv 的附加列：帧数 + 内存/显存采样（监控失败不能拖垮主循环）。"""
        extra: Dict[str, Any] = {"frames": self._frames}
        try:
            from monitor import sample
            extra.update(sample())
        except Exception:  # noqa: BLE001
            pass
        return extra


# -----------------------------------------------------------------------------
def _selftest() -> int:
    """假相机 + 假模型 + 无窗口，跑 60 帧，验证面板/状态/按键/统计都正常。"""
    sys.path.insert(0, PROJECT_ROOT)
    import camera as camera_mod
    import stereo_depth as sd_mod
    from camera import FramePair

    class FakeCam:
        eye_size = (640, 480)

        def __init__(self, cfg=None):
            self.n = 0

        def open(self):
            pass

        def describe(self):
            return "FakeCam"

        def read(self):
            self.n += 1
            img = (np.random.default_rng(self.n).random((480, 640, 3)) * 30 + 110).astype(np.uint8)
            t = time.perf_counter()
            return FramePair(img, img.copy(), t, t, self.n, "stereo", True)

        def close(self):
            pass

    class FakeEst:
        warmup_iters = 1
        max_disp = 192

        def __init__(self, project_cfg, ffs_cfg):
            self.scale = float(ffs_cfg.get("scale", 1.0) or 1.0)
            self.loaded = False

        def weights_ready(self):
            return True, ""

        def load(self):
            self.loaded = True

        def warmup(self, n=1, left=None, right=None):
            return 0.0

        def infer(self, left, right):
            h, w = int(left.shape[0] * self.scale), int(left.shape[1] * self.scale)
            disp = np.full((h, w), 40.0, np.float32)
            return {"disp": disp, "timings_ms": {"infer": 35.0, "capture": 1.0}, "scale": self.scale}

        def close(self):
            pass

    real_cam, real_est = camera_mod.StereoCamera, sd_mod.FFSDepthEstimator
    camera_mod.StereoCamera = FakeCam
    sd_mod.FFSDepthEstimator = FakeEst
    try:
        from stereo_sdk.session import Session
        ok = True
        for mode in ("preview", "depth", "pointcloud"):
            s = Session(os.path.join(PROJECT_ROOT, "config.yaml"))
            s.start()
            s._recorder.out_root = os.path.join(PROJECT_ROOT, "_work", "runtime_selftest")
            rt = DisplayRuntime(s, viewer=None, no_display=True, mode=mode, max_frames=60)
            # 等首帧与首份结果（后台线程刚启动，不能假设立刻有数据）
            f, r = None, None
            t_wait = time.perf_counter()
            while time.perf_counter() - t_wait < 8.0:
                f, _ = s.latest_frame()
                if s.has_new_result():
                    r = s.take_result()
                if f is not None and r is not None:
                    break
                time.sleep(0.02)
            if f is None or r is None:
                print(f"  ❌ {mode}: 8s 内没等到首帧/首结果（frame={f is not None}, result={r is not None}）")
                s.stop()
                ok = False
                continue
            panels = rt._panels(f, r)
            want = {"preview": 2, "depth": 3, "pointcloud": 3}[mode]
            if len(panels) != want:
                print(f"  ❌ {mode}: 面板数 {len(panels)} != {want}")
                ok = False
            for name, img, _ in panels:
                if img is None or img.dtype != np.uint8 or img.ndim != 3:
                    print(f"  ❌ {mode}/{name}: 面板图像不合法 {None if img is None else img.dtype}")
                    ok = False
            st = rt._status(r)
            if not st or not isinstance(st[0], str):
                print(f"  ❌ {mode}: 状态栏为空")
                ok = False
            # 按键：i/p/space/[ /]/m 都不应抛异常，q 应退出
            for k in (ord("i"), ord("p"), ord(" "), ord("["), ord("]"), ord("m")):
                if not rt._handle_key(k):
                    print(f"  ❌ {mode}: 按键 {chr(k)} 意外触发退出")
                    ok = False
            if rt._handle_key(ord("q")):
                print(f"  ❌ {mode}: q 未触发退出")
                ok = False
            rc = rt.run()
            fps = rt.prof.fps("frame")
            print(f"  {mode:10s} 面板 {len(panels)} | frames {rt._frames} | run()={rc} | "
                  f"fps={fps:.1f} | " + rt.prof.summary_text())
            if rt._frames < 60:
                print(f"  ❌ {mode}: 跑了 {rt._frames} 帧 < 60")
                ok = False
            s.stop()
        print("  " + ("✅ runtime 自检全部通过" if ok else "❌ runtime 自检有失败项"))
        return 0 if ok else 1
    finally:
        camera_mod.StereoCamera = real_cam
        sd_mod.FFSDepthEstimator = real_est


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="CLI 显示运行时（在 stereo_sdk.Session 之上）")
    ap.add_argument("--selftest", action="store_true", help="假相机 + 假模型 + 无窗口自检")
    ap.add_argument("--mode", default="depth", choices=["preview", "depth", "pointcloud"])
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--perf-csv", default="")
    ap.add_argument("--no-display", action="store_true")
    a = ap.parse_args(argv)
    log = setup(os.path.join(PROJECT_ROOT, "output", "logs"))
    log.info("启动: mode=%s display=%s max_frames=%d duration=%.1fs config=%s",
             a.mode, not a.no_display, a.max_frames, a.duration, a.config)
    if a.selftest:
        return _selftest()
    from stereo_sdk.session import Session
    s = Session(a.config)
    print("  start() …")
    s.start()
    viewer = None
    if not a.no_display:
        import yaml
        cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
        from viewer import StereoViewer
        rt_cfg = cfg.get("runtime") or {}
        viewer = StereoViewer(
            window_name=str(rt_cfg.get("window_name") or "Stereo Viewer"),
            display_scale=float(rt_cfg.get("display_scale", 1.0) or 1.0),
        )
    rt = DisplayRuntime(s, viewer=viewer, mode=a.mode, no_display=a.no_display,
                        max_frames=a.max_frames, duration=a.duration, perf_csv=a.perf_csv)
    rc = 2
    try:
        rc = rt.run()
        log.info("正常结束: frames=%d", rt._frames)
        return rc
    except BaseException:
        # 稳定性④: 顶层兜底 —— 任何未捕获异常都记进日志（含 traceback），
        # 仍然释放相机与窗口，避免"进程还在但相机被占着"
        log.exception("运行时异常退出（frames=%d）", rt._frames)
        return 2
    finally:
        s.stop()
        if viewer is not None:
            viewer.close()
        log.info("资源已释放（rc=%s）", rc)


if __name__ == "__main__":
    sys.exit(main())
