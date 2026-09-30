"""tools/probe_cameras.py —— 双目相机只读探测工具
=============================================================================
目的：**用实测数据**回答以下问题，绝不靠猜：
    1) 本机有哪些摄像头？OpenCV 索引分别是多少？设备真名叫什么？
    2) 每个索引支持哪些分辨率 / 像素格式？实测帧率多少？
    3) 画面是否全黑？左右半幅是否不同（=> 是否"单设备左右拼接"）？
    4) 输出快照图片，供人眼确认设备身份与左右顺序。

为什么用"父进程 + 子进程 + 超时"的结构：
    某些摄像头驱动在打开失败时会**卡死**（尤其是 MSMF 后端按索引打开不存在的设备）。
    如果直接在同一个进程里循环探测，整个程序会被一个坏索引挂住。
    因此：父进程为每个 (索引, 后端) 组合启动一个子进程，超时就直接杀掉并记录 "TIMEOUT"。

用法（在项目根目录执行）：
    python tools/probe_cameras.py                 # 完整探测（默认索引 0..5）
    python tools/probe_cameras.py --max-index 3   # 只探测 0..3
    python tools/probe_cameras.py --list          # 只列出注册表里的设备名 + 快速索引扫描
    python tools/probe_cameras.py --index 2       # 只详细探测索引 2（会打印到控制台）

输出：
    output/probe/report.json        完整探测结果（可复现证据）
    output/probe/snap_*.png         每个可用配置的快照

说明：本工具**只读**——只打开相机读帧，不写入相机任何设置文件、不改动系统配置。
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

# 让本脚本既能当工具跑，也能被 main.py 的 --list 复用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera import BACKENDS, list_video_devices_from_registry  # noqa: E402

# 候选配置：(fourcc, 宽, 高)
# 1280x480 与 2560x720 是"左右拼接"型双目相机的典型分辨率（2 x 640x480 / 2 x 1280x720）
CANDIDATE_CONFIGS: List[Tuple[str, int, int]] = [
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


# -----------------------------------------------------------------------------
# 子进程模式：探测单个 (索引, 后端)
# -----------------------------------------------------------------------------
def probe_one_backend(
    index: int,
    backend: str,
    out_json: str,
    snapshot_dir: str,
    warmup: int = 10,
    measure: int = 20,
) -> Dict[str, Any]:
    """在**当前进程**里探测一个索引 + 一个后端，遍历候选配置，结果写入 out_json。

    输入：
        index        : OpenCV 相机索引
        backend      : "dshow" / "msmf" / "any"
        out_json     : 结果 JSON 路径（父进程靠它取结果，比管道更稳）
        snapshot_dir : 快照保存目录
        warmup       : 每个配置丢弃的前若干帧
        measure      : 每个配置用于测帧率的帧数
    输出：dict（同时写盘）
    """
    api = BACKENDS.get(backend, cv2.CAP_ANY)
    report: Dict[str, Any] = {
        "index": index,
        "backend": backend,
        "configs": [],
        "snapshots": [],
    }

    for fourcc, w, h in CANDIDATE_CONFIGS:
        item: Dict[str, Any] = {"fourcc": fourcc, "request_w": w, "request_h": h}
        cap: Optional[cv2.VideoCapture] = None
        t0 = time.perf_counter()
        try:
            cap = cv2.VideoCapture(index, api)
            item["open_seconds"] = round(time.perf_counter() - t0, 2)
            if not cap.isOpened():
                item["opened"] = False
                item["note"] = "打开失败"
                report["configs"].append(item)
                continue
            item["opened"] = True
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
            item["actual_fourcc"] = "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)) if cc else ""

            if not frames:
                item["note"] = "能打开但读不到帧"
                report["configs"].append(item)
                continue

            last = frames[-1]
            hh, ww = last.shape[:2]
            item["real_width"], item["real_height"] = int(ww), int(hh)
            gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY) if last.ndim == 3 else last
            item["mean"] = round(float(gray.mean()), 2)
            item["std"] = round(float(gray.std()), 2)
            item["blank"] = bool(gray.mean() < 1.0 and gray.std() < 1.0)
            # 左右半幅差异：差异明显 => 左右半幅是不同画面 => 很可能是"单设备左右拼接"
            left = last[:, : ww // 2].astype(np.int16)
            right = last[:, ww // 2 : 2 * (ww // 2)].astype(np.int16)
            item["half_diff_mean"] = round(float(np.abs(left - right).mean()), 2)
            item["half_left_mean"] = round(float(left.mean()), 2)
            item["half_right_mean"] = round(float(right.mean()), 2)

            # 保存快照（只保存非全黑的，避免一堆黑图）
            if not item["blank"]:
                os.makedirs(snapshot_dir, exist_ok=True)
                tag = f"{fourcc or 'AUTO'}_{ww}x{hh}"
                snap = os.path.join(snapshot_dir, f"snap_idx{index}_{backend}_{tag}.png")
                cv2.imwrite(snap, last)
                item["snapshot"] = snap
                report["snapshots"].append(snap)

            report["configs"].append(item)
        except Exception as exc:  # noqa: BLE001
            item["opened"] = False
            item["error"] = f"{type(exc).__name__}: {exc}"
            report["configs"].append(item)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report


# -----------------------------------------------------------------------------
# 父进程模式
# -----------------------------------------------------------------------------
def run_child(
    index: int,
    backend: str,
    tmp_dir: str,
    snapshot_dir: str,
    timeout: float,
) -> Dict[str, Any]:
    """启动子进程探测 (index, backend)，超时则杀掉。返回结果 dict。"""
    out_json = os.path.join(tmp_dir, f"probe_{index}_{backend}.json")
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
        "--out-json",
        out_json,
        "--snapshot-dir",
        snapshot_dir,
    ]
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {
            "index": index,
            "backend": backend,
            "configs": [],
            "timeout": True,
            "note": f"子进程超时（>{timeout}s）被强制结束：该后端在此索引上会挂死",
            "elapsed": round(time.perf_counter() - t0, 2),
        }
    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as fh:
            rep = json.load(fh)
        rep["timeout"] = False
        rep["elapsed"] = round(time.perf_counter() - t0, 2)
        return rep
    return {
        "index": index,
        "backend": backend,
        "configs": [],
        "timeout": False,
        "note": "子进程没有产出结果文件（崩溃或提前退出）",
        "elapsed": round(time.perf_counter() - t0, 2),
    }


def print_summary(reports: List[Dict[str, Any]]) -> None:
    """把探测结果打印成人类可读的表格。"""
    print("\n" + "=" * 108)
    print("  相机探测汇总")
    print("=" * 108)
    for rep in reports:
        head = f"索引 {rep['index']} / 后端 {rep['backend']}"
        if rep.get("timeout"):
            print(f"{head:<28} -> 超时（挂死）：{rep.get('note', '')}")
            continue
        good = [c for c in rep.get("configs", []) if c.get("opened") and c.get("read_ok")]
        if not good:
            notes = {(c.get("note") or c.get("error") or "打开失败") for c in rep.get("configs", [])}
            print(f"{head:<28} -> 无可用配置：{' / '.join(sorted(notes))[:80]}")
            continue
        print(f"{head:<28} -> 可用配置 {len(good)} 个：")
        for c in good:
            flag = "黑屏" if c.get("blank") else "有画面"
            print(
                f"     {str(c['fourcc']) or 'AUTO':<5} 请求 {c['request_w']}x{c['request_h']:<5}"
                f" 实际 {c['real_width']}x{c['real_height']:<5}"
                f" fps~{c['measured_fps']:<5} mean={c['mean']:<7} std={c['std']:<7}"
                f" 左右半幅差={c['half_diff_mean']:<7} {flag}"
            )
    print("=" * 108)


def main() -> int:
    parser = argparse.ArgumentParser(description="双目相机只读探测工具")
    parser.add_argument("--max-index", type=int, default=6, help="探测的索引上限（不含），默认 6")
    parser.add_argument("--backends", default="dshow,msmf", help="要探测的后端，逗号分隔")
    parser.add_argument("--timeout", type=float, default=45.0, help="每个组合的子进程超时（秒）")
    parser.add_argument("--list", action="store_true", help="只列出注册表设备名 + 快速索引扫描")
    parser.add_argument("--index", type=int, default=None, help="只详细探测这个索引（打印到控制台）")
    parser.add_argument("--out-dir", default=os.path.join("output", "probe"), help="结果输出目录")
    # 子进程内部参数
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", default="dshow", help=argparse.SUPPRESS)
    parser.add_argument("--out-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-dir", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # ---------------- 子进程模式 ----------------
    if args.child:
        assert args.index is not None, "--child 必须带 --index"
        probe_one_backend(
            index=args.index,
            backend=args.backend,
            out_json=args.out_json,
            snapshot_dir=args.snapshot_dir,
        )
        return 0

    # ---------------- 打印注册表设备名 ----------------
    print("=" * 108)
    print("  注册表里当前在线的 UVC 视频设备（OpenCV 拿不到设备名，这里靠注册表交叉印证）")
    print("=" * 108)
    devices = list_video_devices_from_registry()
    if not devices:
        print("  （读不到；可能不是 Windows 或注册表权限受限）")
    for d in devices:
        print(f"  {d['vid_pid']:<14} {d['friendly_name']:<34} {d['instance_id']}")

    # ---------------- 只列设备 ----------------
    if args.list:
        print("\n（--list 模式：跳过详细探测。去掉 --list 可做完整探测）")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_tmp")
    snap_dir = os.path.join(args.out_dir, "snapshots")
    os.makedirs(tmp_dir, exist_ok=True)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    indices = [args.index] if args.index is not None else list(range(args.max_index))

    reports: List[Dict[str, Any]] = []
    for idx in indices:
        for backend in backends:
            print(f"[探测] 索引 {idx} / 后端 {backend} ...", flush=True)
            rep = run_child(idx, backend, tmp_dir, snap_dir, args.timeout)
            reports.append(rep)

    print_summary(reports)

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"devices": devices, "reports": reports}, fh, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存：{report_path}")

    snaps = [s for rep in reports for s in rep.get("snapshots", [])]
    if snaps:
        print("快照文件：")
        for s in snaps:
            print(f"  {s}")
    else:
        print("没有生成快照（所有配置都读不到非黑画面）。")
    print("\n下一步：把上面『可用配置』里最优的分辨率/格式填进 config.yaml 的 camera 段。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
