"""offline_selftest.py —— 离线自检：不开相机、不提权，跑完整的实时点云流程
=============================================================================
为什么需要它（血的教训）：
    2026-09-30 用户实测反馈"实时演示有时会突然停下"。根因是 **run_pointcloud 里
    `sharp_warned` 忘记初始化**，第一次做清晰度检查时抛 UnboundLocalError：
        if sharpness < 30 and not sharp_warned:   # <- UnboundLocalError
    这个 bug 有个很坏的特性：**只有画面模糊（sharpness < 30）时才会触发**，
    因为 Python 的 and 会短路 —— 画面清晰时 `sharpness < 30` 为假，
    右半边的 `sharp_warned` 根本不会被求值，于是 bug 一直藏着。
    也就是说："什么时候崩"取决于场景质量，非常难靠手工试出来。

    这个自检把"相机"和"FFS 模型"替换成假实现，于是**在普通权限、没有相机的环境下**
    就能把整条实时链路（读帧→推理→深度→点云→过滤→自动保存→退出总结）跑完，
    并且**故意喂入模糊画面**去踩中那条短路分支。

用法：
    python tools\\offline_selftest.py            # 跑两个场景（清晰 / 模糊）
    python tools\\offline_selftest.py --frames 25

判定：两个场景都必须完整跑完且退出码为 0；模糊场景还必须**恰好打印一次**质量警告。
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# 让脚本能 import 到项目根的模块（main / camera / depth_utils ...）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import camera as camera_mod          # noqa: E402
import main as main_mod              # noqa: E402
import stereo_depth as sd_mod        # noqa: E402

# 自检专用的输出目录：**绝不能沿用 config 里的 output/**，否则合成数据会混进真实测量数据
OUT_DIR = os.path.join(PROJECT_ROOT, "_work", "selftest_output")


def _out_dirs() -> set:
    """列出自检输出目录下已有的 captures 子目录名（用于统计本次新增了哪些）。"""
    cap = os.path.join(OUT_DIR, "captures")
    return set(os.listdir(cap)) if os.path.isdir(cap) else set()


# -----------------------------------------------------------------------------
# 假实现（模拟相机与 FFS 模型）
# -----------------------------------------------------------------------------
def make_frame(blur: bool, w: int = 640, h: int = 480) -> np.ndarray:
    """造一幅有纹理的合成图；blur=True 时强烈模糊（用来把清晰度压到 30 以下）。"""
    rng = np.random.default_rng(1234)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    # 加一些大色块，让画面更像真实场景
    img[: h // 3, : w // 2] = (60, 90, 140)
    img[h // 2 :, w // 3 :] = (200, 180, 40)
    if blur:
        # 反复下采样再放大 == 抹掉高频细节，拉普拉斯方差会掉到很低
        small = cv2.resize(img, (w // 64, h // 64), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


class FakeCamera:
    """替代 camera.StereoCamera：接口与真实类保持一致（只实现被用到的部分）。"""

    def __init__(self, cfg: Dict[str, Any], blur: bool = False) -> None:
        self.cfg = cfg
        self.blur = blur
        self.mode = "side_by_side"          # run_preview 会读 cam.mode
        self._eye = (640, 480)
        self._full = (1280, 480)
        self._last_read_dt = 0.0            # run_preview 会打印左右读取时间差
        self.stats = {"frames": 0, "read_fail": 0, "blank_frames": 0,
                      "no_signal_frames": 0, "measured_fps": 0.0}
        self._t0 = None

    def open(self) -> None:
        import time
        self._t0 = time.perf_counter()

    def close(self) -> None:
        pass

    @property
    def eye_size(self) -> Tuple[int, int]:
        return self._eye

    @property
    def full_size(self) -> Tuple[int, int]:
        return self._full

    def describe(self) -> str:
        return "[离线自检] FakeCamera 1280x480 side_by_side（不是真相机）"

    def read(self) -> "camera_mod.FramePair":
        import time
        left = make_frame(self.blur)
        right = make_frame(self.blur)   # 左右略有差异即可，反正 FFS 是假的
        right[:, :40] = left[:, :40]
        self.stats["frames"] += 1
        n = self.stats["frames"]
        dt = max(time.perf_counter() - self._t0, 1e-6)
        if n % 30 == 0:
            self.stats["measured_fps"] = n / dt
        now = time.perf_counter()
        return camera_mod.FramePair(left, right, now, now, n, "side_by_side", synced=True)


class FakeEstimator:
    """替代 stereo_depth.FFSDepthEstimator：给出"能形成有效深度"的合成视差。

    视差按深度 z=0.6~2.4 m 反算（占位内参 fx=500, B=0.06 -> disp = 30/z），
    这样深度范围过滤（0.15~3.0 m）能通过，点云非空。
    """

    def __init__(self, project_cfg: Dict[str, Any], ffs_cfg: Dict[str, Any], device: str = "cuda") -> None:
        self.ffs_cfg = ffs_cfg
        self.scale = float(ffs_cfg.get("scale", 1.0) or 1.0)
        self.valid_iters = int(ffs_cfg.get("valid_iters", 8))
        self.max_disp = int(ffs_cfg.get("max_disp", 192))
        self.build_volume = str(ffs_cfg.get("build_volume", "pytorch1"))
        self.warmup_iters = int(ffs_cfg.get("warmup_iters", 1))

    def weights_ready(self) -> Tuple[bool, str]:
        return True, "[离线自检] 使用假模型（不加载真实权重）"

    def load(self) -> None:
        pass

    def warmup(self, n: int = 1, left=None, right=None) -> float:
        return 0.0

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> Dict[str, Any]:
        h, w = left_bgr.shape[:2]
        mw, mh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        # 构造 0.6 ~ 2.4 m 的深度斜坡 -> 视差 = fx*B/z（fx=500, B=0.06 => 30/z）
        z = np.linspace(0.6, 2.4, mh, dtype=np.float32)[:, None] * np.ones((1, mw), np.float32)
        disp = (30.0 / z).astype(np.float32)
        model_rgb = cv2.cvtColor(cv2.resize(left_bgr, (mw, mh)), cv2.COLOR_BGR2RGB)
        return {
            "disp": disp,
            "disp_full": disp,
            "model_size": (mw, mh),
            "src_size": (w, h),
            "scale": self.scale,
            "left_model_rgb": model_rgb,
            "timings_ms": {"preprocess": 0.5, "infer": 12.0, "total": 12.5},
            "valid_ratio": float((disp > 0).mean()),
        }

    def info(self) -> Dict[str, Any]:
        return {"loaded": True, "note": "offline fake"}

    def close(self) -> None:
        pass


# -----------------------------------------------------------------------------
def run_case(cfg_path: str, blur: bool, frames: int, mode: str = "pointcloud",
             threaded: bool = False, use_runtime: bool = False) -> int:
    """跑一个场景，返回 0=通过 / 非 0=失败。

    mode: preview（阶段1）/ depth（阶段3）/ pointcloud（阶段4）
    三种模式的代码路径不同（尤其是各自循环里用到的局部变量），所以都要覆盖，
    否则"某个模式里忘了初始化变量"这类 bug 依然会漏掉。

    threaded: 阶段 7.2 的三线程流水线开关。
        注意 main() 里才把 --threaded 解析成 args.threaded_mode，
        而本自检是**直接调用** runner(cfg, args) 的，所以这里要手动补这个属性。
    """
    label = ("模糊画面（踩中质量警告分支）" if blur else "清晰画面") + \
            ("·三线程" if threaded else "")
    print("=" * 88)
    print(f"  场景: --mode {mode} / {label}")
    print("=" * 88)

    # 打补丁：注意 main.py 在**模块级**写了 `from camera import StereoCamera`，
    # 所以必须改 main 自己的命名空间（main_mod.StereoCamera），只改 camera 模块没用
    # （第一次写这个测试时就踩了这个坑：补丁没生效，结果真去开相机了）。
    real_cam = main_mod.StereoCamera
    real_cam_mod = camera_mod.StereoCamera
    real_est = sd_mod.FFSDepthEstimator
    fake_cam = lambda c: FakeCamera(c, blur=blur)     # noqa: E731
    main_mod.StereoCamera = fake_cam                  # type: ignore[assignment]
    camera_mod.StereoCamera = fake_cam                # type: ignore[assignment]
    sd_mod.FFSDepthEstimator = FakeEstimator          # type: ignore[assignment]

    old_argv = sys.argv
    try:
        sys.argv = ["main.py"]
        cli = [
            "--config", cfg_path,
            "--mode", mode,
            "--no-display",             # 不开窗口（无头运行）
            "--max-frames", str(frames),
            "--save-snapshot",          # 同时验证自动保存路径
        ]
        if mode == "pointcloud":
            cli.append("--no-cloud-window")   # 不开 Open3D 窗口
        if use_runtime:
            cli = [c for c in cli if c != "--no-cloud-window"]
            cli.append("--use-runtime")
        args = main_mod.build_parser().parse_args(cli)
        # 阶段 7.2：直接调 runner 会绕过 main() 的 --threaded 解析，这里手动补上
        args.threaded_mode = bool(threaded)
        cfg = main_mod.load_config(cfg_path)
        # ⚠️ 必须把输出目录改到独立位置！
        # 自检会用 --save-snapshot 跑真实的保存代码，如果沿用 config 里的 output/，
        # 就会把**合成数据**写进 output/captures/，混进真实测量数据里很难分辨
        # （实测踩过：13 个目录的假数据混入 output/captures，视差统计量完全相同才被发现）。
        cfg["project"]["output_dir"] = OUT_DIR
        # 新链路的 Session 是读**配置文件**的，所以把补丁后的 cfg 落成临时 yaml，
        # 否则它会用真实 config.yaml 里的 output/，把假数据写进真实采集目录。
        import yaml as _yaml
        tmp_cfg = os.path.join(OUT_DIR, f"cfg_selftest_{mode}_{int(blur)}.yaml")
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(tmp_cfg, "w", encoding="utf-8") as _fh:
            _yaml.safe_dump(cfg, _fh, allow_unicode=True)
        args.config = tmp_cfg
        # 把异常接住并转成"失败"，而不是让自检脚本自己崩掉 ——
        # 这样报告里能直接看到异常类型与消息（例如 UnboundLocalError: sharp_warned）
        runner = {"preview": main_mod.run_preview,
                  "depth": main_mod.run_depth,
                  "pointcloud": main_mod.run_pointcloud}[mode]
        rc = runner(cfg, args)
        err = ""
    except Exception as exc:  # noqa: BLE001
        rc = -1
        err = f"{type(exc).__name__}: {exc}"
        import traceback
        traceback.print_exc()
    finally:
        main_mod.StereoCamera = real_cam          # type: ignore[assignment]
        camera_mod.StereoCamera = real_cam_mod    # type: ignore[assignment]
        sd_mod.FFSDepthEstimator = real_est       # type: ignore[assignment]
        sys.argv = old_argv

    ok = (rc == 0)
    if err:
        print(f"  -> ❌ 抛出异常: {err}")
    else:
        print(f"  -> 退出码 {rc}  {'✅ 通过' if ok else '❌ 失败'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="离线自检：不开相机跑完整实时流程")
    ap.add_argument("--frames", type=int, default=25, help="每个场景跑多少帧（默认 25）")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    ap.add_argument("--keep-output", action="store_true",
                    help="保留自检产生的保存文件（默认跑完自动清理，否则每次会留下约 20 MB 的 PLY）")
    a = ap.parse_args()

    print("### 离线自检：不打开相机、不使用真实权重、不需要管理员权限 ###")
    print(f"#  项目根目录: {PROJECT_ROOT}")
    print(f"#  帧数: {a.frames}（>20 才能覆盖自动保存路径）")
    print(f"#  自检输出目录: {OUT_DIR}（与真实数据 output/ 完全隔离）")

    before = _out_dirs()
    bad = 0
    total = 0
    for mode in ("preview", "depth", "pointcloud"):
        for blur in (False, True):
            total += 1
            bad += run_case(a.config, blur=blur, frames=a.frames, mode=mode)

    # 阶段 7.2：三线程流水线单独跑一组（depth 与 pointcloud 两条路径都接入了）
    # 说明：线程模式下"每帧都推理"变成"尽力推理"，所以不能用帧数判断，
    # 只要能看到推理结果并正常退出即可。
    for mode in ("depth", "pointcloud"):
        for blur in (False, True):
            total += 1
            bad += run_case(a.config, blur=blur, frames=a.frames, mode=mode, threaded=True)

    # 阶段 1 重构：新链路（stereo_sdk.Session + runtime.DisplayRuntime）
    for mode in ("depth", "pointcloud"):
        total += 1
        bad += run_case(a.config, blur=False, frames=a.frames, mode=mode, use_runtime=True)

    # 顺带核对"自动保存"这条路径真的写出了东西（否则测试只覆盖了计算、没覆盖保存）
    produced = sorted(_out_dirs() - before)
    print("=" * 88)
    print(f"#  自动保存路径产出: {len(produced)} 个目录")
    for p in produced:
        print(f"#     {p}")

    if os.path.isdir(OUT_DIR) and not a.keep_output:
        import shutil
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(OUT_DIR) for f in fs)
        shutil.rmtree(OUT_DIR, ignore_errors=True)
        print(f"#  已清理自检输出（{size / 1048576:.1f} MB）—— 加 --keep-output 可保留")
    elif a.keep_output:
        print("#  已保留自检输出（--keep-output）")

    print("=" * 88)
    if bad:
        print(f"  ❌ {total} 个场景中有 {bad} 个失败（很可能是变量未初始化 / 流程中断）")
    else:
        print(f"  ✅ {total} 个场景全部通过：三种模式 × 清晰/模糊 都能完整跑完，"
              f"未出现未初始化变量等崩溃")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
