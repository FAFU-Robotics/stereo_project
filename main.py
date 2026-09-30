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
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="stereo_project —— 双目视觉系统（阶段 1：采集与预览）")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"), help="配置文件路径")
    p.add_argument("--mode", default="preview", choices=["preview", "depth", "pointcloud"],
                   help="运行模式；depth/pointcloud 将在后续阶段实现")
    p.add_argument("--list", action="store_true", help="只列出相机设备信息后退出")
    p.add_argument("--no-display", action="store_true", help="不开窗口（用于自动化测试）")
    p.add_argument("--max-frames", type=int, default=0, help="跑满 N 帧后自动退出（0=不限制）")
    p.add_argument("--duration", type=float, default=0.0, help="运行 N 秒后自动退出（0=不限制）")
    p.add_argument("--save-snapshot", action="store_true", help="运行中保存一次左右原始图像")
    p.add_argument("--display-scale", type=float, default=None, help="显示缩放系数（覆盖配置）")
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

    if args.mode != "preview":
        print(f"[未实现] --mode {args.mode} 将在后续阶段实现：")
        print("         depth      -> 阶段 3（接入 FFS 视差/深度）")
        print("         pointcloud -> 阶段 4（Open3D 实时点云）")
        return 1

    return run_preview(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
