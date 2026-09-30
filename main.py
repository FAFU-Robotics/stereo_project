"""main.py —— 双目视觉系统入口（阶段 1：双路采集与实时显示）
=============================================================================
用法：
    python main.py                 # 实时预览左右两路画面（按 q 或 ESC 退出）
    python main.py --list          # 只列出摄像头相关信息（设备名 + 快速扫描）
    python main.py --no-display    # 无窗口模式：跑固定帧数并保存快照（用于自动化测试）
    python main.py --max-frames 60 --no-display --save-snapshot
    python main.py --mode depth    # 阶段 3 才会实现，现在会明确报错提示

按键（实时预览窗口内）：
    q / ESC  退出
    p / 空格 暂停 / 恢复
    s        保存当前左右原始图像 + 参数说明
    i        在控制台打印详细状态
    h        显示 / 隐藏按键帮助

阶段说明：
    本文件只做"流程调度"：读配置 -> 开相机 -> 取帧 -> 交给 viewer 显示 -> 处理按键。
    深度/点云等逻辑在后续阶段接入 depth_utils.py / stereo_depth.py / pointcloud.py。
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from camera import CameraError, CameraReadError, StereoCamera, list_video_devices_from_registry, quick_probe_index
from viewer import StereoViewer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# 配置加载
# -----------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """读取 config.yaml（UTF-8）并做最基本的合法性检查。

    输入：path —— 配置文件路径
    输出：dict —— 配置内容
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式不对（顶层应为字典）：{path}")
    # 把相对路径统一解析成"相对项目根目录"的绝对路径，避免换个工作目录就跑不起来
    for key in ("ffs_repo", "output_dir", "weights_dir", "calibration_dir"):
        if key in cfg.get("project", {}):
            cfg["project"][key] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["project"][key]))
    return cfg


# -----------------------------------------------------------------------------
# 保存数据
# -----------------------------------------------------------------------------
def make_session_dir(base_dir: str, tag: str = "") -> str:
    """创建一次运行/一次快照的独立目录，返回该目录路径。

    命名：output/snapshots/20260930_110642[_tag]/
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{tag}" if tag else stamp
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_frame_pair(
    left: np.ndarray,
    right: np.ndarray,
    out_dir: str,
    cfg: Dict[str, Any],
    note: str = "",
) -> List[str]:
    """保存一对左右原始图像，并写一份参数说明 meta.txt。

    为什么要写 meta.txt：
        图像和参数必须成套保存，否则以后无法还原"这张图是用什么参数、有没有标定"。

    输入：
        left/right : BGR uint8 数组（单眼大小）
        out_dir    : 输出目录（会创建）
        cfg        : 完整配置字典
        note       : 额外说明（例如"未标定，仅流程验证"）
    输出：写入的文件路径列表
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    p_left = os.path.join(out_dir, "left_raw.png")
    p_right = os.path.join(out_dir, "right_raw.png")
    cv2.imwrite(p_left, left)
    cv2.imwrite(p_right, right)
    paths += [p_left, p_right]

    depth_status = "已标定" if cfg.get("depth", {}).get("calibrated") else "未标定（仅用于流程验证，禁止用于测量）"
    meta = [
        "# stereo_project 数据说明",
        f"保存时间      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"图像尺寸(单眼): {left.shape[1]}x{left.shape[0]}  通道={left.shape[2] if left.ndim == 3 else 1}",
        f"相机模式      : {cfg.get('camera', {}).get('mode')}",
        f"相机索引      : single={cfg.get('camera', {}).get('single_index')} "
        f"left={cfg.get('camera', {}).get('left_index')} right={cfg.get('camera', {}).get('right_index')}",
        f"请求分辨率    : {cfg.get('camera', {}).get('width')}x{cfg.get('camera', {}).get('height')} "
        f"FOURCC={cfg.get('camera', {}).get('fourcc')} 后端={cfg.get('camera', {}).get('backend')}",
        f"标定状态      : {depth_status}",
        f"内参来源      : {cfg.get('depth', {}).get('temp', {}).get('source', '')}",
        f"基线(米)      : {cfg.get('depth', {}).get('temp', {}).get('baseline_m') if not cfg.get('depth', {}).get('calibrated') else '(见标定文件)'}",
        f"备注          : {note}",
        "",
        "警告：在 depth.calibrated=false 期间，任何由这些图像生成的深度/点云都",
        "      不具备测量意义，不能用于测距、尺寸或三维定位。",
    ]
    p_meta = os.path.join(out_dir, "meta.txt")
    with open(p_meta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(meta) + "\n")
    paths.append(p_meta)
    return paths


# -----------------------------------------------------------------------------
# --list：设备信息
# -----------------------------------------------------------------------------
def run_list(cfg: Dict[str, Any]) -> int:
    """列出注册表设备名 + 对候选索引做快速扫描（每个索引独立、失败即跳过）。"""
    print("=" * 88)
    print("  Windows 上当前真正连接的 UVC 视频设备（历史残留会单独标注）")
    print("=" * 88)
    devs = list_video_devices_from_registry()
    if not devs:
        print("  （读不到设备名）")
    for d in devs:
        flag = "在线" if d.get("present") == "是" else "历史残留(未插)"
        print(f"  [{flag}] {d['vid_pid']:<12} {d['friendly_name']:<32} {d['instance_id']}")

    print("\n" + "=" * 88)
    print("  OpenCV 索引快速扫描（不设置分辨率/格式，最接近驱动默认行为）")
    print("=" * 88)
    cam = cfg.get("camera", {})
    for idx in range(6):
        r = quick_probe_index(
            idx,
            backend=cam.get("backend", "dshow"),
            fourcc="",           # 不设格式：实测设 MJPG 反而读不到帧
            width=0,
            height=0,
        )
        if r["opened"]:
            # 用"无有效信号"的更宽判定，避免把驱动空缓冲误当成"环境暗"
            no_sig = r["mean"] < 5.0 and r["std"] < 12.0
            print(
                f"  索引 {idx}: 可打开  {r['width']}x{r['height']}  fps~{r['fps']}  "
                f"mean={r['mean']} std={r['std']}  {'无有效信号' if no_sig else '有画面'}"
            )
        else:
            print(f"  索引 {idx}: 不可用（{r['error']}）")
    print("\n提示：完整探测请运行  python tools/probe_cameras.py")
    return 0


# -----------------------------------------------------------------------------
# 实时预览主循环
# -----------------------------------------------------------------------------
def run_preview(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """打开相机，实时显示左右两路画面，处理按键，安全退出。

    输入：cfg —— 配置字典；args —— 命令行参数
    输出：进程退出码（0 正常）
    """
    runtime = cfg.get("runtime", {})
    viewer = StereoViewer(
        window_name=str(runtime.get("window_name", "stereo_project")),
        display_scale=float(args.display_scale if args.display_scale is not None else runtime.get("display_scale", 1.0)),
    )
    cam = StereoCamera(cfg.get("camera", {}))
    out_root = os.path.join(cfg["project"]["output_dir"], "snapshots")

    paused = False
    last_left: Optional[np.ndarray] = None
    last_right: Optional[np.ndarray] = None
    saved_count = 0
    status_every = int(runtime.get("status_every_n_frames", 30))
    t_loop_start = time.perf_counter()

    print("=" * 88)
    print("  阶段 1：双目实时预览")
    print("=" * 88)
    print(f"  配置文件 : {args.config}")
    print(f"  相机模式 : {cam.mode}")
    print("  提示     : 窗口内按 q 或 ESC 退出；p 暂停；s 保存快照；i 打印状态；h 帮助")
    if args.no_display:
        print("  --no-display 已开启：不打开窗口，跑完指定帧数后自动退出")

    try:
        cam.open()
        print(f"  已打开   : {cam.describe()}")
        print(f"  单眼尺寸 : {cam.eye_size[0]}x{cam.eye_size[1]}   原始输出: {cam.full_size[0]}x{cam.full_size[1]}")
        if cam.mode == "dual":
            print("  [注意] dual 模式下左右是两次独立 read()，**不是严格同步**，")
            print("         实测左右读取时间差会打印在状态里；运动场景可能有错配。")

        frame_i = 0
        while True:
            if not paused:
                try:
                    pair = cam.read()
                except CameraReadError as exc:
                    print(f"[错误] {exc}")
                    print("       建议：检查 USB 连接/供电，或运行 python tools/probe_cameras.py 复查。")
                    break
                last_left, last_right = pair.left, pair.right
                frame_i += 1

            if last_left is None or last_right is None:
                print("[错误] 尚未取到任何帧，退出。")
                break

            fps = cam.stats["measured_fps"]
            status = [
                f"camera: {cam.describe()}  eye={last_left.shape[1]}x{last_left.shape[0]}",
                f"capture fps: {fps:.1f}   frames read: {cam.stats['frames']}   "
                f"read_fail: {cam.stats['read_fail']}   blank: {cam.stats['blank_frames']}",
                f"left-right read dt: {cam._last_read_dt * 1000:.2f} ms   "
                f"(dual 模式下这是两次 read 的时间差；!=0 即非严格同步)",
                f"depth status: {'CALIBRATED' if cfg.get('depth', {}).get('calibrated') else 'UNCALIBRATED - process validation only'}",
            ]

            if args.no_display:
                # 无窗口模式：只打印，不 imshow
                if frame_i % max(1, status_every) == 0 or frame_i == 1:
                    print(f"  [帧 {frame_i}] " + " | ".join(status[:2]))
                if args.save_snapshot and frame_i == max(1, args.max_frames // 2):
                    out_dir = make_session_dir(out_root, "autotest")
                    files = save_frame_pair(
                        last_left, last_right, out_dir, cfg, note="--no-display 自动测试保存"
                    )
                    saved_count += 1
                    print("  已保存：")
                    for f in files:
                        print(f"    {f}")
                key = -1
            else:
                panels = [
                    ("LEFT (raw, unrectified)", last_left, [f"{last_left.shape[1]}x{last_left.shape[0]}"]),
                    ("RIGHT (raw, unrectified)", last_right, [f"{last_right.shape[1]}x{last_right.shape[0]}"]),
                ]
                key = viewer.show_panels(panels, status_lines=status, paused=paused)

            # ---- 按键处理 ----
            if not args.no_display:
                if StereoViewer.is_quit(key):
                    print("  收到退出按键，正在释放相机…")
                    break
                if StereoViewer.is_pause(key):
                    paused = not paused
                    print(f"  {'已暂停' if paused else '已恢复'}")
                if StereoViewer.is_help(key):
                    print(f"  帮助显示：{'开' if viewer.toggle_help() else '关'}")
                if StereoViewer.is_info(key):
                    print(f"  [状态] {cam.describe()} eye={cam.eye_size} stats={cam.stats} "
                          f"左右读取时间差={cam._last_read_dt * 1000:.2f} ms")
                if StereoViewer.is_snapshot(key):
                    out_dir = make_session_dir(out_root, "manual")
                    files = save_frame_pair(last_left, last_right, out_dir, cfg, note="手动按键 s 保存")
                    saved_count += 1
                    print(f"  已保存 {len(files)} 个文件到：{out_dir}")

            # ---- 自动退出条件（便于无人值守测试）----
            if args.max_frames and cam.stats["frames"] >= args.max_frames:
                print(f"  已达 --max-frames={args.max_frames}，自动退出")
                break
            if args.duration and (time.perf_counter() - t_loop_start) >= args.duration:
                print(f"  已达 --duration={args.duration}s，自动退出")
                break

    except CameraError as exc:
        print(f"[相机错误] {exc}")
        return 2
    except KeyboardInterrupt:
        print("\n  收到 Ctrl+C，正在释放相机…")
    finally:
        cam.close()
        viewer.close()

    dt = max(time.perf_counter() - t_loop_start, 1e-9)
    print("-" * 88)
    print("  运行结束")
    print(f"  总帧数        : {cam.stats['frames']}")
    print(f"  平均采集帧率  : {cam.stats['frames'] / dt:.2f} FPS（含显示开销）")
    print(f"  读取失败次数  : {cam.stats['read_fail']}")
    print(f"  全黑帧次数    : {cam.stats['blank_frames']}")
    print(f"  无有效信号帧  : {cam.stats['no_signal_frames']}  <-- 大于 0 说明相机没有给出真实图像")
    print(f"  保存快照次数  : {saved_count}")
    print("  相机已释放。若再次运行提示被占用，请确认没有别的程序（相机 App / 浏览器）在用它。")
    return 0


# -----------------------------------------------------------------------------
# 阶段 3：实时视差（相机 -> FFS -> 视差伪彩显示）
# -----------------------------------------------------------------------------
def run_depth(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """实时双目视差：读相机 -> FFS 推理 -> 显示 左图|右图|视差伪彩 + 状态。

    设计要点（都来自前几个阶段实测得到的教训）：
        1) 模型只加载一次，**用真实相机画面预热**（预热尺寸必须与正式输入一致，
           否则第一次正式推理仍会冷启动，实测 2088ms vs 270ms）；
        2) 不排队、不堆积：每轮都读"最新一帧"，推理慢了自然丢帧，不会无限积压；
        3) 视差伪彩用百分位 + EMA 平滑（见 viewer.DisparityColorizer），实时不闪；
        4) 画面上必须明确标注"未标定，仅流程验证"——因为此时既没有内参也没有校正。

    输入：cfg（完整配置）、args（命令行参数，可覆盖 ffs.scale/valid_iters/max_disp）
    输出：进程退出码
    """
    from stereo_depth import FFSError, FFSDepthEstimator
    from viewer import DisparityColorizer

    ffs_cfg = dict(cfg.get("ffs", {}))
    if getattr(args, "scale", None) is not None:
        ffs_cfg["scale"] = float(args.scale)
    if getattr(args, "valid_iters", None) is not None:
        ffs_cfg["valid_iters"] = int(args.valid_iters)
    if getattr(args, "max_disp", None) is not None:
        ffs_cfg["max_disp"] = int(args.max_disp)

    runtime = cfg.get("runtime", {})
    viewer = StereoViewer(
        window_name=str(runtime.get("window_name", "stereo_project")),
        display_scale=float(args.display_scale if args.display_scale is not None else runtime.get("display_scale", 1.0)),
    )
    colorizer = DisparityColorizer()

    print("=" * 88)
    print("  阶段 3：实时视差（FFS）")
    print("=" * 88)
    print(f"  ffs.scale={ffs_cfg.get('scale')}  valid_iters={ffs_cfg.get('valid_iters')}  "
          f"max_disp={ffs_cfg.get('max_disp')}  build_volume={ffs_cfg.get('build_volume')}")
    print("  [警告] 当前未标定：视差仅用于流程验证，**不具备测量意义**（不能测距/测尺寸/定位）")

    est = FFSDepthEstimator(cfg["project"], ffs_cfg)
    ok, msg = est.weights_ready()
    print(f"  [权重] {msg}")
    if not ok:
        viewer.close()
        return 3
    try:
        est.load()
    except FFSError as exc:
        print(f"[FFS 错误] {exc}")
        viewer.close()
        return 4

    cam = StereoCamera(cfg.get("camera", {}))
    out_root = os.path.join(cfg["project"]["output_dir"], "captures")
    paused = False
    saved = 0
    auto_saved = False
    last_left = last_right = None
    last_disp = None
    t_loop_start = time.perf_counter()
    n_processed = 0
    rec_infer_ms: List[float] = []

    try:
        cam.open()
        print(f"  相机已打开: {cam.describe()}  单眼 {cam.eye_size[0]}x{cam.eye_size[1]}")

        # 预热：必须用真实相机画面（尺寸要与正式推理一致）
        first = cam.read()
        last_left, last_right = first.left, first.right
        print("  用真实相机画面预热模型（第一次会比较慢）…")
        est.warmup(est.warmup_iters, left=last_left, right=last_right)

        while True:
            if not paused:
                try:
                    pair = cam.read()          # 永远只取最新帧，不做队列堆积
                except CameraReadError as exc:
                    print(f"[读取错误] {exc}")
                    break
                last_left, last_right = pair.left, pair.right
                try:
                    out = est.infer(last_left, last_right)
                except FFSError as exc:
                    print(f"[FFS 错误] {exc}")
                    break
                last_disp = out["disp"]
                n_processed += 1
                rec_infer_ms.append(out["timings_ms"]["infer"])
                if len(rec_infer_ms) > 30:
                    rec_infer_ms.pop(0)

            if last_disp is None or last_left is None:
                print("[错误] 还没有任何视差结果，退出")
                break

            disp_vis, (lo, hi) = colorizer.colorize(last_disp)
            # 未标定警告直接画在视差面板下方（ASCII，cv2 画不了中文）
            warn = np.zeros((28, disp_vis.shape[1], 3), dtype=np.uint8)
            StereoViewer._put_text(warn, "UNCALIBRATED - process validation only", (6, 20), (0, 220, 255), 0.5, 1, shadow=False)
            disp_panel = np.vstack([disp_vis, warn])

            elapsed = max(time.perf_counter() - t_loop_start, 1e-9)
            e2e_fps = n_processed / elapsed
            avg_infer = float(np.mean(rec_infer_ms)) if rec_infer_ms else 0.0
            valid_ratio = float((last_disp > 0).mean()) * 100.0

            status = [
                f"capture {cam.stats['measured_fps']:.1f} fps | ffs {avg_infer:.0f} ms "
                f"({1000.0 / avg_infer:.1f} fps) | end-to-end {e2e_fps:.2f} fps | frames {n_processed}",
                f"model input {last_disp.shape[1]}x{last_disp.shape[0]} (scale={out['scale']}) | "
                f"valid disparity {valid_ratio:.1f}% | display range {lo:.1f}..{hi:.1f} px",
                f"max_disp={est.max_disp} valid_iters={est.valid_iters} | "
                f"UNCALIBRATED: no intrinsics, no rectification - NOT for measurement",
            ]

            if args.no_display:
                if n_processed % 5 == 0 or n_processed == 1:
                    print(f"  [帧 {n_processed}] " + " | ".join(status[:2]))
                key = -1
            else:
                panels = [
                    ("LEFT (raw, unrectified)", last_left, [f"{last_left.shape[1]}x{last_left.shape[0]}"]),
                    ("RIGHT (raw, unrectified)", last_right, [f"{last_right.shape[1]}x{last_right.shape[0]}"]),
                    ("DISPARITY (px, uncalibrated)", disp_panel, []),
                ]
                key = viewer.show_panels(panels, status_lines=status, paused=paused)

            if not args.no_display:
                if StereoViewer.is_quit(key):
                    print("  收到退出按键，正在释放相机与模型…")
                    break
                if StereoViewer.is_pause(key):
                    paused = not paused
                    print(f"  {'已暂停' if paused else '已恢复'}")
                    if not paused:
                        colorizer.reset()
                if StereoViewer.is_help(key):
                    viewer.toggle_help()
                if StereoViewer.is_info(key):
                    print(f"  [状态] {est.info()}")
                    print(f"         相机 {cam.describe()} stats={cam.stats}")
                if StereoViewer.is_snapshot(key):
                    d = make_session_dir(out_root, "capture")
                    files = save_frame_pair(last_left, last_right, d, cfg, note="阶段 3 实时视差保存")
                    p = os.path.join(d, "disparity.npy")
                    np.save(p, last_disp)
                    files.append(p)
                    p2 = os.path.join(d, "disparity_vis.png")
                    cv2.imwrite(p2, disp_vis)
                    files.append(p2)
                    # 视差参数必须和视差数据成套保存：
                    #   阶段 4 用 depth = fx*scale * baseline / disp 换算深度时，
                    #   必须知道 scale 与模型输入尺寸，否则深度会整体错一个倍数。
                    p3 = os.path.join(d, "disp_meta.txt")
                    with open(p3, "w", encoding="utf-8") as fh:
                        fh.write("\n".join([
                            "# 视差数据参数说明（与 disparity.npy 配套，缺一不可）",
                            f"保存时间        : {time.strftime('%Y-%m-%d %H:%M:%S')}",
                            f"视差数组形状    : {last_disp.shape[1]}x{last_disp.shape[0]}（模型输入分辨率，不是原图）",
                            f"视差单位        : 像素（**模型输入分辨率下**的像素）",
                            f"ffs.scale       : {out['scale']}   <-- 换算到原图必须乘这个系数",
                            f"原图单眼尺寸    : {last_left.shape[1]}x{last_left.shape[0]}",
                            f"换算关系        : 原图视差 = 本视差 × {1.0 / out['scale'] if out['scale'] else 0:.4g}",
                            f"valid_iters     : {est.valid_iters}",
                            f"max_disp        : {est.max_disp}（模型像素；超过此值时视差不可信）",
                            f"build_volume    : {est.build_volume}",
                            f"有效视差比例    : {valid_ratio:.2f}%",
                            f"推理耗时        : {out['timings_ms']['infer']:.1f} ms",
                            f"端到端帧率      : {e2e_fps:.2f} FPS",
                            "",
                            "⚠️ 未标定：无内参、无畸变参数、无校正映射。",
                            "   用这些数据算出的深度/点云**不具备测量意义**。",
                        ]) + "\n")
                    files.append(p3)
                    saved += 1
                    print(f"  已保存 {len(files)} 个文件到：{d}")

            if args.max_frames and n_processed >= args.max_frames:
                print(f"  已达 --max-frames={args.max_frames}，自动退出")
                break
            if args.duration and (time.perf_counter() - t_loop_start) >= args.duration:
                print(f"  已达 --duration={args.duration}s，自动退出")
                break

            # --save-snapshot：自动保存一次（默认第 20 帧；便于无人值守留证，不用掐时机按键）
            if args.save_snapshot and not auto_saved:
                target = 20 if not args.max_frames else max(1, min(20, args.max_frames // 2))
                if n_processed >= target:
                    d = make_session_dir(out_root, "autosave")
                    files = save_frame_pair(last_left, last_right, d, cfg,
                                            note=f"阶段3 自动保存（第 {n_processed} 帧，未标定，仅流程验证）")
                    p = os.path.join(d, "disparity.npy")
                    np.save(p, last_disp)
                    files.append(p)
                    p2 = os.path.join(d, "disparity_vis.png")
                    cv2.imwrite(p2, disp_vis)
                    files.append(p2)
                    p3 = os.path.join(d, "meta_stage3.txt")
                    with open(p3, "w", encoding="utf-8") as fh:
                        fh.write("\n".join([
                            "# 阶段 3 实时视差 自动保存",
                            f"时间            : {time.strftime('%Y-%m-%d %H:%M:%S')}",
                            f"处理帧序号      : {n_processed}",
                            f"相机            : {cam.describe()}  单眼 {cam.eye_size[0]}x{cam.eye_size[1]}",
                            f"模型输入尺寸    : {last_disp.shape[1]}x{last_disp.shape[0]}  (ffs.scale={out['scale']})",
                            f"valid_iters={est.valid_iters} max_disp={est.max_disp} build_volume={est.build_volume}",
                            f"推理耗时        : {out['timings_ms']['infer']:.1f} ms",
                            f"有效视差比例    : {valid_ratio:.2f}%",
                            f"端到端帧率      : {e2e_fps:.2f} FPS",
                            "",
                            "⚠️ 未标定：无内参、无畸变参数、无校正映射。",
                            "   本目录数据仅用于流程验证，不能测距、不能测尺寸、不能定位。",
                        ]) + "\n")
                    files.append(p3)
                    auto_saved = True
                    saved += 1
                    print(f"  已自动保存 {len(files)} 个文件到：{d}")
                    for f in files:
                        print(f"    {f}")

    except CameraError as exc:
        print(f"[相机错误] {exc}")
        return 2
    except KeyboardInterrupt:
        print("\n  收到 Ctrl+C，正在释放资源…")
    finally:
        cam.close()
        est.close()
        viewer.close()

    dt = max(time.perf_counter() - t_loop_start, 1e-9)
    print("-" * 88)
    print("  运行结束")
    print(f"  处理帧数        : {n_processed}")
    print(f"  端到端帧率      : {n_processed / dt:.2f} FPS（采集+推理+显示）")
    print(f"  采集帧率        : {cam.stats['measured_fps']:.2f} FPS")
    if rec_infer_ms:
        print(f"  最近推理耗时    : 平均 {np.mean(rec_infer_ms):.1f} ms（约 {1000.0 / np.mean(rec_infer_ms):.1f} FPS）")
    print(f"  丢弃帧数(估算)  : {max(cam.stats['frames'] - n_processed, 0)}（推理比采集慢，只保留最新帧）")
    print(f"  读取失败 / 全黑 / 无信号 : {cam.stats['read_fail']} / {cam.stats['blank_frames']} / {cam.stats['no_signal_frames']}")
    print(f"  保存次数        : {saved}")
    print("  相机与模型已释放。")
    return 0


# -----------------------------------------------------------------------------
# 阶段 4：深度图 + 实时点云（相机 -> FFS -> 深度 -> 点云）
# -----------------------------------------------------------------------------
def run_pointcloud(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """实时点云：读相机 -> FFS 视差 -> 深度(米) -> Open3D 点云(左图着色) + 实时显示。

    与阶段 3 的区别：
        - 多出"视差 -> 深度 -> 点云"两步，需要相机内参与基线；
        - 多开一个 Open3D 窗口显示点云（可用 --no-cloud-window 关闭，便于无人值守测试）；
        - 深度/点云的**数值是否可信完全取决于标定状态**：
            未标定时用占位内参 + 厂家标称基线，界面上与保存的文件里都会强制标注警告。

    性能注意（需求里明确要求）：
        - 点云对象**复用同一个**（PointCloudBuilder 内部持有），逐帧只更新点/颜色；
        - 若启用体素下采样，open3d 会返回**新对象**，这里检测到对象变化时重新挂载几何体。

    输入：cfg / args（可覆盖 ffs.scale、valid_iters、max_disp、depth 范围、点大小）
    输出：进程退出码
    """
    from depth_utils import (CameraParams, colorize_depth, depth_stats,
                             disparity_to_depth, save_depth_npy)
    from pointcloud import PointCloudBuilder
    from stereo_depth import FFSError, FFSDepthEstimator

    # Open3D 体积大、导入慢，只在点云模式里按需导入（预览/视差模式不必付这个开销）
    try:
        import open3d as o3d
    except ImportError:
        print("[错误] 未安装 open3d，无法使用 --mode pointcloud")
        return 5

    ffs_cfg = dict(cfg.get("ffs", {}))
    for key, attr in (("scale", "scale"), ("valid_iters", "valid_iters"), ("max_disp", "max_disp")):
        v = getattr(args, attr, None)
        if v is not None:
            ffs_cfg[key] = v

    pc_cfg = dict(cfg.get("pointcloud", {}))
    if getattr(args, "zmin", None) is not None:
        pc_cfg["zmin_m"] = float(args.zmin)
    if getattr(args, "zmax", None) is not None:
        pc_cfg["zmax_m"] = float(args.zmax)
    if getattr(args, "voxel", None) is not None:
        pc_cfg["voxel_size_m"] = float(args.voxel)
    zmin_m = float(pc_cfg.get("zmin_m", 0.15))
    zmax_m = float(pc_cfg.get("zmax_m", 3.0))

    runtime = cfg.get("runtime", {})
    viewer = StereoViewer(
        window_name=str(runtime.get("window_name", "stereo_project")),
        display_scale=float(args.display_scale if args.display_scale is not None else runtime.get("display_scale", 1.0)),
    )

    print("=" * 88)
    print("  阶段 4：深度图 + 实时点云")
    print("=" * 88)
    print(f"  ffs: scale={ffs_cfg.get('scale')} valid_iters={ffs_cfg.get('valid_iters')} max_disp={ffs_cfg.get('max_disp')}")
    print(f"  点云: z {zmin_m}..{zmax_m} m, 体素下采样 {pc_cfg.get('voxel_size_m')} m, "
          f"点数上限 {pc_cfg.get('max_points')}, 着色={pc_cfg.get('use_color')}")

    est = FFSDepthEstimator(cfg["project"], ffs_cfg)
    ok, msg = est.weights_ready()
    print(f"  [权重] {msg}")
    if not ok:
        viewer.close()
        return 3
    try:
        est.load()
    except FFSError as exc:
        print(f"[FFS 错误] {exc}")
        viewer.close()
        return 4

    cam = StereoCamera(cfg.get("camera", {}))
    params_full: Optional[CameraParams] = None
    builder: Optional[PointCloudBuilder] = None
    vis = None
    pcd = None
    out_root = os.path.join(cfg["project"]["output_dir"], "captures")
    paused = False
    saved = 0
    n_processed = 0
    rec_infer_ms: List[float] = []
    t_loop_start = time.perf_counter()
    last = {}

    try:
        cam.open()
        print(f"  相机已打开: {cam.describe()}  单眼 {cam.eye_size[0]}x{cam.eye_size[1]}")
        params_full = CameraParams.from_config(cfg, eye_size=cam.eye_size)
        print(f"  相机参数: {params_full.describe()}")
        if not params_full.calibrated:
            print("  " + "!" * 76)
            print("  [严重警告] 当前【未标定】：使用占位内参 + 厂家标称基线。")
            print("             由此产生的深度与点云【不具备测量意义】，不能测距/测尺寸/定位。")
            print("  " + "!" * 76)
        builder = PointCloudBuilder(pc_cfg, calibrated=params_full.calibrated)

        first = cam.read()
        print("  用真实相机画面预热模型…")
        est.warmup(est.warmup_iters, left=first.left, right=first.right)

        if not args.no_cloud_window:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="stereo_project point cloud" +
                              ("" if params_full.calibrated else "  [UNCALIBRATED]"),
                              width=int(args.cloud_window), height=int(args.cloud_window))
            opt = vis.get_render_option()
            opt.point_size = float(args.point_size)
            opt.background_color = np.array([0.08, 0.08, 0.10])
            print("  Open3D 点云窗口已打开（关闭窗口或按 q 退出）")

        while True:
            if not paused:
                try:
                    pair = cam.read()
                except CameraReadError as exc:
                    print(f"[读取错误] {exc}")
                    break
                try:
                    out = est.infer(pair.left, pair.right)
                except FFSError as exc:
                    print(f"[FFS 错误] {exc}")
                    break
                rec_infer_ms.append(out["timings_ms"]["infer"])
                if len(rec_infer_ms) > 30:
                    rec_infer_ms.pop(0)

                # 视差是"模型分辨率"下的像素：内参必须同比例缩放，否则深度整体错倍数
                scale = float(out["scale"])
                params_model = params_full.scaled(scale)
                # 一致性自检：模型分辨率必须与视差图尺寸吻合。
                # 这一步能挡住"scale 算反/忘记缩放"这类隐蔽错误（实测踩过：焦距错 4 倍）。
                consist = (params_model.width == out["disp"].shape[1]
                           and params_model.height == out["disp"].shape[0])
                if not consist and n_processed == 0:
                    print(f"  [警告] 内参分辨率 {params_model.width}x{params_model.height} 与视差尺寸 "
                          f"{out['disp'].shape[1]}x{out['disp'].shape[0]} 不一致，深度可能整体错倍数！")
                depth, valid = disparity_to_depth(out["disp"], params_model)
                st = depth_stats(depth, valid)
                t_pc0 = time.perf_counter()
                pcd, pc_stats = builder.build(depth, params_model, out["left_model_rgb"])
                pc_ms = (time.perf_counter() - t_pc0) * 1000.0
                n_processed += 1
                last = {"left": pair.left, "right": pair.right, "disp": out["disp"],
                        "depth": depth, "valid": valid, "params": params_model,
                        "left_model_rgb": out["left_model_rgb"], "scale": scale,
                        "stats": st, "pc_stats": pc_stats, "infer_ms": out["timings_ms"]["infer"],
                        "pc_ms": pc_ms}

                # Open3D：复用几何体；若因体素下采样换了新对象，则重新挂载
                if vis is not None:
                    vis.add_geometry(pcd)
                    if not last.get("vis_added"):
                        last["vis_added"] = True
                    vis.update_geometry(pcd)
                    if not vis.poll_events():
                        print("  点云窗口已关闭，退出")
                        break
                    vis.update_renderer()

            if not last:
                print("[错误] 还没有任何结果，退出")
                break

            # ---- 显示：左图 | 右图 | 深度伪彩 ----
            depth_vis = colorize_depth(last["depth"], last["valid"], zmin_m, zmax_m)
            warn = np.zeros((28, depth_vis.shape[1], 3), dtype=np.uint8)
            StereoViewer._put_text(
                warn,
                ("UNCALIBRATED - no metric depth" if not params_full.calibrated else "CALIBRATED"),
                (6, 20), (0, 220, 255) if not params_full.calibrated else (0, 255, 0), 0.5, 1, shadow=False)
            depth_panel = np.vstack([depth_vis, warn])

            elapsed = max(time.perf_counter() - t_loop_start, 1e-9)
            e2e = n_processed / elapsed
            avg_infer = float(np.mean(rec_infer_ms)) if rec_infer_ms else 0.0
            st = last["stats"]
            status = [
                f"capture {cam.stats['measured_fps']:.1f} fps | ffs {avg_infer:.0f} ms "
                f"| cloud {last['pc_ms']:.0f} ms | end-to-end {e2e:.2f} fps | frames {n_processed}",
                f"depth valid {st['valid_ratio']:.1f}% | median {st['median_m']:.3f} m | "
                f"range {zmin_m}..{zmax_m} m | points {last['pc_stats']['points']}",
                f"intrinsics: {'CALIBRATED' if params_full.calibrated else 'UNCALIBRATED (placeholder fx='
                              f'{params_full.fx:.0f}, baseline={params_full.baseline_m:.3f}m)'}"
                f" | scale={last['scale']} valid_iters={est.valid_iters}",
            ]

            if args.no_display:
                if n_processed % 5 == 0 or n_processed == 1:
                    print(f"  [帧 {n_processed}] " + " | ".join(status[:2]))
                key = -1
            else:
                panels = [
                    ("LEFT (raw, unrectified)", last["left"], [f"{last['left'].shape[1]}x{last['left'].shape[0]}"]),
                    ("RIGHT (raw, unrectified)", last["right"], [f"{last['right'].shape[1]}x{last['right'].shape[0]}"]),
                    (f"DEPTH (m, {zmin_m}..{zmax_m})", depth_panel, []),
                ]
                key = viewer.show_panels(panels, status_lines=status, paused=paused)

            if not args.no_display:
                if StereoViewer.is_quit(key):
                    print("  收到退出按键，正在释放资源…")
                    break
                if StereoViewer.is_pause(key):
                    paused = not paused
                    print(f"  {'已暂停' if paused else '已恢复'}")
                if StereoViewer.is_help(key):
                    viewer.toggle_help()
                if StereoViewer.is_info(key):
                    print(f"  [状态] {est.info()}")
                    print(f"         相机 {cam.describe()} stats={cam.stats}")
                    print(f"         点云 {last['pc_stats']}  深度 {last['stats']}")
                    print(f"         参数 {params_full.describe()}")
                if StereoViewer.is_snapshot(key):
                    d = make_session_dir(out_root, "cloud")
                    files = save_frame_pair(last["left"], last["right"], d, cfg,
                                            note="阶段 4 深度+点云保存")
                    p = os.path.join(d, "disparity.npy"); np.save(p, last["disp"]); files.append(p)
                    p = save_depth_npy(last["depth"], d, last["params"],
                                       extra={"scale": last["scale"], "valid_iters": est.valid_iters,
                                              "max_disp": est.max_disp, "有效深度比例": f"{last['stats']['valid_ratio']:.2f}%"})
                    files.append(p)
                    cv2.imwrite(os.path.join(d, "depth_vis.png"), depth_vis)
                    files.append(os.path.join(d, "depth_vis.png"))
                    # 点云：为保存单独 clone 一份（避免与实时对象互相影响）
                    ply = builder.save_ply(o3d.geometry.PointCloud(pcd), d, last["params"],
                                           extra={"scale": last["scale"], "valid_iters": est.valid_iters,
                                                  "max_disp": est.max_disp,
                                                  "points": last["pc_stats"]["points"]})
                    files.append(ply)
                    saved += 1
                    print(f"  已保存 {len(files)} 个文件到：{d}")

            if args.max_frames and n_processed >= args.max_frames:
                print(f"  已达 --max-frames={args.max_frames}，自动退出")
                break
            if args.duration and (time.perf_counter() - t_loop_start) >= args.duration:
                print(f"  已达 --duration={args.duration}s，自动退出")
                break

    except CameraError as exc:
        print(f"[相机错误] {exc}")
        return 2
    except KeyboardInterrupt:
        print("\n  收到 Ctrl+C，正在释放资源…")
    finally:
        cam.close()
        est.close()
        viewer.close()
        if vis is not None:
            try:
                vis.destroy_window()
            except Exception:
                pass

    dt = max(time.perf_counter() - t_loop_start, 1e-9)
    print("-" * 88)
    print("  运行结束")
    print(f"  处理帧数        : {n_processed}")
    print(f"  端到端帧率      : {n_processed / dt:.2f} FPS（采集+推理+深度+点云+显示）")
    print(f"  采集帧率        : {cam.stats['measured_fps']:.2f} FPS")
    if rec_infer_ms:
        print(f"  最近推理耗时    : 平均 {np.mean(rec_infer_ms):.1f} ms")
    if last:
        print(f"  最后一次深度    : 有效 {last['stats']['valid_ratio']:.1f}% 中位 {last['stats']['median_m']:.3f} m")
        print(f"  最后一次点云    : {last['pc_stats']['points']} 点（构建 {last['pc_ms']:.0f} ms）")
    print(f"  标定状态        : {'已标定' if (params_full and params_full.calibrated) else '未标定（占位参数，禁止测量）'}")
    print(f"  读取失败/全黑/无信号 : {cam.stats['read_fail']} / {cam.stats['blank_frames']} / {cam.stats['no_signal_frames']}")
    print(f"  保存次数        : {saved}")
    return 0


# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="stereo_project —— 双目视觉系统")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"), help="配置文件路径")
    p.add_argument("--mode", default="preview", choices=["preview", "depth", "pointcloud"],
                   help="preview=实时双目预览(阶段1)  depth=实时视差(阶段3)  pointcloud=点云(阶段4)")
    p.add_argument("--list", action="store_true", help="只列出相机设备信息后退出")
    p.add_argument("--no-display", action="store_true", help="不开窗口（用于自动化测试）")
    p.add_argument("--max-frames", type=int, default=0, help="跑满 N 帧后自动退出（0=不限制）")
    p.add_argument("--duration", type=float, default=0.0, help="运行 N 秒后自动退出（0=不限制）")
    p.add_argument("--save-snapshot", action="store_true", help="运行中保存一次左右原始图像")
    p.add_argument("--display-scale", type=float, default=None, help="显示缩放系数（覆盖配置）")
    # 阶段 3：FFS 参数临时覆盖（不改配置文件）
    p.add_argument("--scale", type=float, default=None, help="覆盖 ffs.scale（0.5 可明显提速）")
    p.add_argument("--valid-iters", type=int, default=None, help="覆盖 ffs.valid_iters")
    p.add_argument("--max-disp", type=int, default=None, help="覆盖 ffs.max_disp")
    # 阶段 4：点云相关
    p.add_argument("--zmin", type=float, default=None, help="覆盖 pointcloud.zmin_m（米）")
    p.add_argument("--zmax", type=float, default=None, help="覆盖 pointcloud.zmax_m（米）")
    p.add_argument("--voxel", type=float, default=None, help="覆盖 pointcloud.voxel_size_m（0=不下采样）")
    p.add_argument("--point-size", type=float, default=2.0, help="Open3D 点云显示点大小")
    p.add_argument("--cloud-window", type=int, default=900, help="Open3D 点云窗口边长（像素）")
    p.add_argument("--no-cloud-window", action="store_true", help="不打开 Open3D 窗口（便于无人值守测试）")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"[配置错误] {exc}")
        return 1

    if args.list:
        return run_list(cfg)

    if args.mode == "preview":
        return run_preview(cfg, args)
    if args.mode == "depth":
        return run_depth(cfg, args)
    if args.mode == "pointcloud":
        return run_pointcloud(cfg, args)
    print(f"[未实现] --mode {args.mode}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
