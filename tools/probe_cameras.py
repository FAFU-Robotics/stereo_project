r"""tools/probe_cameras.py —— 双目相机只读探测工具（逐配置独立子进程版）
=============================================================================
目的：**用实测数据**回答以下问题，绝不靠猜：
    1) 本机有哪些摄像头？OpenCV 索引分别是多少？设备真名叫什么？
    2) 每个索引支持哪些分辨率 / 像素格式？实测帧率多少？
    3) 画面是否全黑？左右半幅是否不同（=> 是否"单设备左右拼接"）？
    4) 输出快照图片，供人眼确认设备身份与左右顺序。

【为什么用"父进程 + 每个配置一个子进程 + 超时"的结构】
    实测发现：某些驱动在设置不支持的分辨率/格式时会**永久阻塞**在 read() 上。
    如果按"一个索引跑完所有配置"来分进程，一个坏配置就会让整个索引失去结果。
    因此本工具把粒度降到**单个配置**：
        父进程 -> 子进程(索引 i, 后端 b, 配置 c)，超时(默认 20s)就杀掉并记为 TIMEOUT。
    这样坏配置只会"牺牲自己"，好配置的结果照样拿到。

探测分两个阶段：
    阶段 A：**不设置任何参数**直接打开读帧（__DEFAULT__），最接近驱动原生行为，
            用来判断"这个索引到底能不能出画面"，并得到原生分辨率。
    阶段 B：只对阶段 A 成功的索引，逐个尝试候选的 分辨率 x 像素格式 组合。

用法（在项目根目录执行）：
    python tools/probe_cameras.py                    # 完整两阶段探测（索引 0..5）
    python tools/probe_cameras.py --index 2          # 只探测索引 2（推荐先跑这个）
    python tools/probe_cameras.py --list             # 只列设备名，不探测
    python tools/probe_cameras.py --index 2 --no-stage-b   # 只做阶段 A

输出：
    output/probe/report.json    完整探测结果（可复现证据）
    output/probe/snapshots/*.png 非黑画面快照

说明：本工具**只读** —— 只打开相机读帧并保存图片，不修改任何系统设置。
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# 让本脚本既能当工具跑，也能被 main.py 复用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera import BACKENDS, list_video_devices_from_registry  # noqa: E402

# 候选配置：(fourcc, 宽, 高)
#   ("__DEFAULT__", 0, 0) 表示"什么都不设置，直接读"
#   1280x480 与 2560x720 是"左右拼接"型双目相机的典型分辨率（2 x 640x480 / 2 x 1280x720）
CANDIDATE_CONFIGS: List[Tuple[str, int, int]] = [
    ("__DEFAULT__", 0, 0),
    ("MJPG", 1280, 480),
    ("MJPG", 2560, 720),
    ("MJPG", 640, 480),
    ("MJPG", 1280, 720),
    ("MJPG", 1920, 1080),
    ("YUY2", 1280, 480),
    ("YUY2", 640, 480),
    ("", 1280, 480),
    ("", 640, 480),
]


def config_tag(fourcc: str, w: int, h: int) -> str:
    """把配置转成命令行可传递的字符串，例如 'MJPG:1280x480' 或 '__DEFAULT__'。"""
    if fourcc == "__DEFAULT__":
        return "__DEFAULT__"
    return f"{fourcc}:{w}x{h}"


def parse_tag(tag: str) -> Tuple[str, int, int]:
    """config_tag 的逆操作。"""
    if tag == "__DEFAULT__":
        return "__DEFAULT__", 0, 0
    fourcc, res = tag.split(":")
    w, h = res.lower().split("x")
    return fourcc, int(w), int(h)


# -----------------------------------------------------------------------------
# 子进程模式：探测"一个索引 + 一个后端 + 一个配置"
# -----------------------------------------------------------------------------
def probe_single_config(
    index: int,
    backend: str,
    tag: str,
    out_json: str,
    snapshot_dir: str,
    warmup: int,
    measure: int,
) -> Dict[str, Any]:
    """在当前进程中探测单个配置，结果写入 out_json（父进程靠文件取结果，比管道稳）。

    输入：
        index        : OpenCV 相机索引
        backend      : "dshow" / "msmf" / "any"
        tag          : 配置字符串（见 config_tag）
        out_json     : 结果 JSON 路径
        snapshot_dir : 快照目录
        warmup       : 丢弃的前若干帧（等自动曝光收敛）
        measure      : 用于统计帧率的帧数
    输出：dict（同时写盘）
    """
    fourcc, w, h = parse_tag(tag)
    api = BACKENDS.get(backend, cv2.CAP_ANY)
    item: Dict[str, Any] = {
        "index": index,
        "backend": backend,
        "config": tag,
        "fourcc_request": fourcc,
        "request_w": w,
        "request_h": h,
        "opened": False,
    }
    cap: Optional[cv2.VideoCapture] = None
    try:
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(index, api)
        item["open_seconds"] = round(time.perf_counter() - t0, 2)
        if not cap.isOpened():
            item["note"] = "打开失败"
            return item
        item["opened"] = True

        if fourcc == "__DEFAULT__":
            item["note"] = "未设置任何参数（驱动原生默认）"
        else:
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        for _ in range(max(0, warmup)):
            cap.read()

        frames: List[np.ndarray] = []
        t1 = time.perf_counter()
        for _ in range(max(1, measure)):
            ok, f = cap.read()
            if ok and f is not None and f.size > 0:
                frames.append(f)
        dt = max(time.perf_counter() - t1, 1e-9)

        item["read_ok"] = len(frames)
        item["read_total"] = measure
        item["measured_fps"] = round(len(frames) / dt, 1) if frames else 0.0
        item["reported_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        item["reported_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        item["reported_fps"] = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
        cc = int(cap.get(cv2.CAP_PROP_FOURCC))
        item["actual_fourcc"] = (
            "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00") if cc else ""
        )

        if not frames:
            item["note"] = "能打开但读不到帧"
            return item

        last = frames[-1]
        hh, ww = last.shape[:2]
        item["real_width"], item["real_height"] = int(ww), int(hh)
        gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY) if last.ndim == 3 else last
        item["mean"] = round(float(gray.mean()), 2)
        item["std"] = round(float(gray.std()), 2)
        item["blank"] = bool(gray.mean() < 1.0 and gray.std() < 1.0)
        # 左右半幅差异：差异明显 => 左右半幅是不同画面 => 很可能是"单设备左右拼接"
        half = ww // 2
        left = last[:, :half].astype(np.int16)
        right = last[:, half : half * 2].astype(np.int16)
        item["half_diff_mean"] = round(float(np.abs(left - right).mean()), 2)
        item["half_left_mean"] = round(float(left.mean()), 2)
        item["half_right_mean"] = round(float(right.mean()), 2)

        if not item["blank"]:
            os.makedirs(snapshot_dir, exist_ok=True)
            snap = os.path.join(snapshot_dir, f"snap_idx{index}_{backend}_{tag.replace(':', '_')}.png")
            cv2.imwrite(snap, last)
            item["snapshot"] = snap
        return item
    except Exception as exc:  # noqa: BLE001 —— 探测函数不把异常抛给调用方
        item["error"] = f"{type(exc).__name__}: {exc}"
        return item
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        try:
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(item, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# 父进程：调度子进程
# -----------------------------------------------------------------------------
def run_child(
    index: int,
    backend: str,
    tag: str,
    tmp_dir: str,
    snapshot_dir: str,
    timeout: float,
    warmup: int,
    measure: int,
) -> Dict[str, Any]:
    """跑一个"单配置"子进程；超时则杀掉并返回 TIMEOUT 结果。"""
    safe = f"{index}_{backend}_{tag.replace(':', '_').replace('__', '')}"
    out_json = os.path.join(tmp_dir, f"probe_{safe}.json")
    if os.path.exists(out_json):
        os.remove(out_json)
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--child",
        "--index",
        str(index),
        "--backend",
        backend,
        "--config",
        tag,
        "--out-json",
        out_json,
        "--snapshot-dir",
        snapshot_dir,
        "--warmup",
        str(warmup),
        "--measure",
        str(measure),
    ]
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {
            "index": index,
            "backend": backend,
            "config": tag,
            "opened": True,
            "timeout": True,
            "note": f"子进程超时(>{timeout}s)被强制结束：该配置会阻塞驱动",
            "elapsed": round(time.perf_counter() - t0, 2),
        }
    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as fh:
            item = json.load(fh)
        item["timeout"] = False
        item["elapsed"] = round(time.perf_counter() - t0, 2)
        return item
    return {
        "index": index,
        "backend": backend,
        "config": tag,
        "opened": False,
        "timeout": False,
        "note": "子进程未产出结果（崩溃或提前退出）",
        "elapsed": round(time.perf_counter() - t0, 2),
    }


def phase_a(indices: List[int], backends: List[str], tmp_dir: str, snap_dir: str,
            timeout: float, warmup: int, measure: int) -> List[Dict[str, Any]]:
    """阶段 A：不设参数直接打开读帧，判断哪些 (索引, 后端) 能出画面。"""
    print("\n===== 阶段 A：默认参数打开测试（不设置分辨率/格式）=====")
    results: List[Dict[str, Any]] = []
    for idx in indices:
        for backend in backends:
            tag = "__DEFAULT__"
            print(f"[A] 索引 {idx} / {backend} ...", flush=True)
            item = run_child(idx, backend, tag, tmp_dir, snap_dir, timeout, warmup, measure)
            results.append(item)
            if item.get("timeout"):
                print(f"    超时（挂死）")
            elif item.get("read_ok"):
                print(f"    成功 {item['real_width']}x{item['real_height']} fps~{item['measured_fps']} "
                      f"mean={item['mean']} {'全黑' if item.get('blank') else '有画面'}")
            else:
                print(f"    不可用（{item.get('note') or item.get('error') or '打开失败'}）")
    return results


def phase_b(ok_pairs: List[Tuple[int, str]], tmp_dir: str, snap_dir: str,
            timeout: float, warmup: int, measure: int) -> List[Dict[str, Any]]:
    """阶段 B：对能出画面的 (索引, 后端) 逐个尝试候选分辨率/格式。"""
    print("\n===== 阶段 B：分辨率 / 像素格式逐配置测试 =====")
    results: List[Dict[str, Any]] = []
    for idx, backend in ok_pairs:
        for fourcc, w, h in CANDIDATE_CONFIGS:
            tag = config_tag(fourcc, w, h)
            if tag == "__DEFAULT__":
                continue  # 阶段 A 已测
            print(f"[B] 索引 {idx} / {backend} / {tag} ...", flush=True)
            item = run_child(idx, backend, tag, tmp_dir, snap_dir, timeout, warmup, measure)
            results.append(item)
            if item.get("timeout"):
                print("    超时（该配置会阻塞驱动）")
            elif item.get("read_ok"):
                print(f"    成功 实际{item['real_width']}x{item['real_height']} "
                      f"fps~{item['measured_fps']} mean={item['mean']} std={item['std']} "
                      f"左右半幅差={item.get('half_diff_mean')} {'全黑' if item.get('blank') else '有画面'}")
            else:
                print(f"    不可用（{item.get('note') or item.get('error') or '打开失败'}）")
    return results


def print_summary(all_items: List[Dict[str, Any]]) -> None:
    """把结果打印成人类可读的表格。"""
    print("\n" + "=" * 112)
    print("  探测汇总（只列出能读到帧的配置）")
    print("=" * 112)
    good = [i for i in all_items if i.get("read_ok")]
    if not good:
        print("  没有任何配置能读到帧。请检查：相机是否插好 / 是否被其他程序占用 / USB 供电。")
    for i in good:
        blank = "全黑" if i.get("blank") else "有画面"
        print(
            f"  索引 {i['index']:<2} {i['backend']:<6} {i['config']:<18} "
            f"实际 {i['real_width']}x{i['real_height']:<5} fps~{i['measured_fps']:<6} "
            f"mean={i['mean']:<7} std={i['std']:<7} 左右半幅差={i.get('half_diff_mean', 0):<7} {blank}"
        )
    timeouts = [i for i in all_items if i.get("timeout")]
    if timeouts:
        print("\n  以下配置会阻塞驱动（已强制结束，请勿在正式程序里使用）：")
        for i in timeouts:
            print(f"    索引 {i['index']} {i['backend']} {i['config']}")
    print("=" * 112)


def main() -> int:
    parser = argparse.ArgumentParser(description="双目相机只读探测工具（逐配置独立子进程）")
    parser.add_argument("--max-index", type=int, default=6, help="索引上限（不含），默认 6")
    parser.add_argument("--backends", default="dshow,msmf", help="要探测的后端，逗号分隔")
    parser.add_argument("--index", type=int, default=None, help="只探测这个索引")
    parser.add_argument("--timeout", type=float, default=20.0, help="单个配置的子进程超时（秒）")
    parser.add_argument("--warmup", type=int, default=8, help="丢弃的前若干帧")
    parser.add_argument("--measure", type=int, default=15, help="统计帧率用的帧数")
    parser.add_argument("--no-stage-b", action="store_true", help="只做阶段 A")
    parser.add_argument("--list", action="store_true", help="只列设备名，不探测")
    parser.add_argument("--out-dir", default=os.path.join("output", "probe"), help="结果输出目录")
    # 子进程内部参数
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", default="dshow", help=argparse.SUPPRESS)
    parser.add_argument("--config", default="__DEFAULT__", help=argparse.SUPPRESS)
    parser.add_argument("--out-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-dir", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ---------------- 子进程模式 ----------------
    if args.child:
        assert args.index is not None, "--child 必须带 --index"
        probe_single_config(
            index=args.index,
            backend=args.backend,
            tag=args.config,
            out_json=args.out_json,
            snapshot_dir=args.snapshot_dir,
            warmup=args.warmup,
            measure=args.measure,
        )
        return 0

    # ---------------- 打印注册表设备名 ----------------
    print("=" * 112)
    print("  注册表里的 UVC 视频设备（OpenCV 拿不到设备名，这里靠注册表交叉印证）")
    print("  注意：注册表同时包含『历史上插过但当前未连接』的设备，需与 pnputil /enum-devices /connected 对照。")
    print("=" * 112)
    devices = list_video_devices_from_registry()
    if not devices:
        print("  （读不到；可能不是 Windows 或注册表权限受限）")
    present = [d for d in devices if d.get("present") == "是"]
    absent = [d for d in devices if d.get("present") != "是"]
    print("  ── 当前真正连接的相机 ──")
    for d in present or devices:
        print(f"  {d['vid_pid']:<14} {d['friendly_name']:<38} {d['instance_id']}")
    if absent:
        print("  ── 历史残留（现在没插，忽略）──")
        for d in absent:
            print(f"  {d['vid_pid']:<14} {d['friendly_name']:<38}")

    if args.list:
        print("\n（--list 模式：跳过探测）")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    snap_dir = os.path.join(args.out_dir, "snapshots")
    os.makedirs(tmp_dir, exist_ok=True)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    indices = [args.index] if args.index is not None else list(range(args.max_index))

    results_a = phase_a(indices, backends, tmp_dir, snap_dir, args.timeout, args.warmup, args.measure)
    ok_pairs = [(i["index"], i["backend"]) for i in results_a if i.get("read_ok")]
    results_b: List[Dict[str, Any]] = []
    if ok_pairs and not args.no_stage_b:
        results_b = phase_b(ok_pairs, tmp_dir, snap_dir, args.timeout, args.warmup, args.measure)
    elif not ok_pairs:
        print("\n  阶段 A 没有任何 (索引, 后端) 能读到帧，跳过阶段 B。")

    all_items = results_a + results_b
    print_summary(all_items)

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"devices": devices, "results": all_items}, fh, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存：{report_path}")

    snaps = [i["snapshot"] for i in all_items if i.get("snapshot")]
    if snaps:
        print("快照文件：")
        for s in snaps:
            print(f"  {s}")
    else:
        print("没有生成快照（没有任何配置读到非黑画面）。")
    print("\n下一步：把最优的分辨率/格式填进 config.yaml 的 camera 段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
