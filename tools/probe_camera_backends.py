"""probe_camera_backends.py —— 摄像头后端 / 分辨率探测（阶段 3 稳定性前置）

=============================================================================
【要回答的问题】
    本机用 DirectShow 打开双目相机需要**管理员权限**（实测 HRESULT 0x80070005）。
    这会让"打包成 EXE 双击运行"完全失败。所以必须先验证：
        · OpenCV 的 MSMF 后端能不能**免提权**打开？
        · 能不能拿到项目需要的 **1280x480**（两个 640x480 并排）？
        · 帧率够不够（推理只要 6~20 FPS，30 FPS 就够）？
        · 采集出来的左右两半是不是真的不同（确认并排布局没错位）？

【怎么用】
    1) **必须在普通（非管理员）窗口里运行**，否则结论无意义：
         whoami /groups | findstr /i "Mandatory"
         -> Medium Mandatory Level  表示普通权限，可以测
         -> High   Mandatory Level  表示管理员，换窗口
    2) 关掉所有可能占相机的程序（RealSense Viewer / 其他 python / 会议软件）
    3) python tools\\probe_camera_backends.py
       只看环境不碰相机:  python tools\\probe_camera_backends.py --dry-run

【输出】
    · 每个 (后端, 设备号) 一张小表：实际分辨率 / FOURCC / 帧率 / 首帧形状
    · 左右两半平均亮度差（几十以上说明左右确实不同，布局正常）
    · 结论摘要（可直接贴给别人）
    · 成功的后端各存一张图到 _work/probe_backend/ 供肉眼确认
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# 必须在 import cv2 **之前**设置，否则 OpenCV 的 C++ 层 WARN 会刷满屏幕
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 项目需要的分辨率排在最前；后面是这台相机的其它常见模式（作为回退候选）
CANDIDATE_SIZES: List[Tuple[int, int]] = [(1280, 480), (2560, 720), (1280, 720), (640, 480)]
BACKENDS: List[Tuple[str, int]] = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW),
                                   ("ANY", cv2.CAP_ANY)]
MAX_INDEX = 5


# ---------------------------------------------------------------- 环境检查
def list_pnp_cameras() -> List[str]:
    """用 Windows PnP 列出系统里真实存在的相机设备（判断"没接"还是"打不开"）。

    这一步很关键：如果这里能看到相机、但 OpenCV 打不开，才是"后端/权限"问题；
    如果这里也看不到，说明相机根本没接上/没上电/驱动异常。
    """
    names: List[str] = []
    for cls in ("Camera", "Image"):
        try:
            ps = ("Get-PnpDevice -Class " + cls + " -Status OK -ErrorAction SilentlyContinue | "
                  "Select-Object -ExpandProperty FriendlyName")
            out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                                 capture_output=True, timeout=20)
            text = (out.stdout or b"").decode("mbcs", errors="ignore")
            for line in text.splitlines():
                line = line.strip()
                if line and line not in names:
                    names.append(line)
        except Exception:  # noqa: BLE001
            continue
    return names


def integrity_level() -> str:
    """判断当前进程有没有管理员权限（UAC 过滤器之后的有效状态）。

    优先用 IsUserAnAdmin()：它检查的是**当前令牌**里 Administrators 是否启用，
    所以"管理员账户但被 UAC 过滤"的普通进程会返回 False —— 正是我们要的信息。
    """
    try:
        import ctypes
        if ctypes.windll.shell32.IsUserAnAdmin():
            return "High（管理员，测出来的结果不能代表普通用户）"
        return "Medium（普通权限，测试有效）"
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(["whoami", "/groups"], capture_output=True, timeout=10)
        text = (out.stdout or b"").decode("mbcs", errors="ignore")
        for sid, label in (("S-1-16-12288", "High（管理员）"),
                           ("S-1-16-16384", "System"),
                           ("S-1-16-8192", "Medium（普通权限）")):
            if sid in text:
                return label
    except Exception as exc:  # noqa: BLE001
        return f"读取失败({type(exc).__name__})"
    return "未知"


def available_backends() -> str:
    try:
        names = []
        for be in cv2.videoio_registry.getCameraBackends():
            names.append(getattr(be, "name", None) or str(be))
        return f"{len(names)} 个: {', '.join(names)}"
    except Exception as exc:  # noqa: BLE001
        return f"无法枚举({type(exc).__name__})"


def backend_name(cap: cv2.VideoCapture) -> str:
    try:
        be = int(cap.get(cv2.CAP_PROP_BACKEND))
    except Exception:  # noqa: BLE001
        return "?"
    table = {int(getattr(cv2, n)): n for n in dir(cv2) if n.startswith("CAP_") and isinstance(
        getattr(cv2, n), int) and n.endswith(("MSMF", "DSHOW", "V4L2", "ANY", "FFMPEG", "GSTREAMER"))}
    return table.get(be, str(be))


def fourcc_str(v: float) -> str:
    try:
        iv = int(v)
    except Exception:  # noqa: BLE001
        return "?"
    s = "".join(chr((iv >> (8 * i)) & 0xFF) for i in range(4))
    return s if s.isprintable() else f"0x{iv:08x}"


# ---------------------------------------------------------------- 单个组合
def safe_grab(cap: cv2.VideoCapture, n: int = 1) -> None:
    """grab 可能抛 cv2.error（MSMF 在设备状态异常时会这样），不能让它中断探测。"""
    for _ in range(max(n, 0)):
        try:
            cap.grab()
        except cv2.error:
            return


def safe_read(cap: cv2.VideoCapture, tries: int = 3) -> Tuple[bool, Optional[Any]]:
    """read 可能抛 cv2.error（实测 MSMF 切到不支持的尺寸后会抛 Assertion failed），
    所以必须包起来；返回 (成功?, 帧)。"""
    for _ in range(max(tries, 1)):
        try:
            ok, frame = cap.read()
        except cv2.error:
            ok, frame = False, None
        if ok and frame is not None and getattr(frame, "size", 0):
            return True, frame
        time.sleep(0.03)
    return False, None


def probe_one(api: int, idx: int, warmup: int, frames: int) -> Optional[Dict[str, Any]]:
    """对每个候选分辨率**重新打开一次设备**再试。

    为什么要重开：实测 MSMF 在 cap.set() 到它不支持的尺寸后，设备会进入异常状态，
    后续 read() 直接抛 Assertion failed（整个程序崩掉）。重开比"原地切换"稳得多。
    """
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    results: List[Dict[str, Any]] = []
    for (want_w, want_h) in CANDIDATE_SIZES:
        cap = cv2.VideoCapture(idx, api)
        if not cap.isOpened():
            cap.release()
            break                      # 这个 (后端, 设备号) 根本打不开，后面的尺寸也不用试
        try:
            # MJPG 必须先设：USB2 带宽下不压缩跑不到高帧率
            cap.set(cv2.CAP_PROP_FOURCC, mjpg)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
            safe_grab(cap, warmup)      # 丢掉前几帧（自动曝光/首帧未就绪）
            ok, frame = safe_read(cap, 3)
        except cv2.error:
            ok, frame = False, None
        if not ok or frame is None:
            results.append({"want": (want_w, want_h), "ok": False})
            cap.release()
            time.sleep(0.25)
            continue
        got_h, got_w = frame.shape[:2]
        fps = 0.0
        if frames > 0:
            t0 = time.perf_counter()
            n_ok = 0
            for _ in range(frames):
                r, _f = safe_read(cap, 1)
                n_ok += 1 if r else 0
            fps = n_ok / max(time.perf_counter() - t0, 1e-9)
        # 左右两半亮度差（确认并排布局，不是单张图被劈开）
        half_diff = float("nan")
        if got_w >= 2 and got_h > 0:
            mid = got_w // 2
            gl = float(cv2.cvtColor(np.ascontiguousarray(frame[:, :mid]), cv2.COLOR_BGR2GRAY).mean())
            gr = float(cv2.cvtColor(np.ascontiguousarray(frame[:, mid:]), cv2.COLOR_BGR2GRAY).mean())
            half_diff = abs(gl - gr)
        try:
            fourcc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            be = backend_name(cap)
        except cv2.error:
            fourcc, be = "?", "?"
        rec = {
            "want": (want_w, want_h), "ok": True, "got": (got_w, got_h),
            "requested_ok": (got_w, got_h) == (want_w, want_h),
            "fourcc": fourcc, "fps": fps, "half_diff": half_diff,
            "backend": be, "frame": frame,
        }
        results.append(rec)
        cap.release()
        time.sleep(0.25)
        if rec["requested_ok"]:
            break                      # 要的分辨率拿到了，不必再试更小的
    ok_recs = [r for r in results if r.get("ok")]
    if not ok_recs:
        return None
    exact = [r for r in ok_recs if r.get("requested_ok")]
    best = exact[0] if exact else ok_recs[0]
    best["all"] = results
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="摄像头后端/分辨率探测（免提权可用性）")
    ap.add_argument("--dry-run", action="store_true", help="只看环境，不打开相机")
    ap.add_argument("--warmup", type=int, default=8, help="每次打开丢弃的帧数（默认 8）")
    ap.add_argument("--frames", type=int, default=30, help="测帧率时连读的帧数（默认 30）")
    ap.add_argument("--max-index", type=int, default=MAX_INDEX, help="设备号上限（默认 5）")
    ap.add_argument("--force", action="store_true",
                    help="即使 Windows 没识别到相机也继续探测（默认会直接提示并退出）")
    ap.add_argument("--save", default=os.path.join(PROJECT_ROOT, "_work", "probe_backend"),
                    help="成功时的样图保存目录")
    args = ap.parse_args()

    print("=" * 96)
    print("  摄像头后端探测：MSMF 能不能免提权打开、能不能拿到 1280x480")
    print("=" * 96)
    # 静音 OpenCV 自己的 WARN 刷屏（否则粘贴出来全是噪音）
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
    except Exception:  # noqa: BLE001
        pass
    lvl = integrity_level()
    print(f"  OpenCV      : {cv2.__version__}")
    print(f"  当前权限级别 : {lvl}")
    print(f"  可用后端     : {available_backends()}")
    print(f"  候选分辨率   : {', '.join(f'{w}x{h}' for w, h in CANDIDATE_SIZES)}")
    cams = list_pnp_cameras()
    if cams:
        print(f"  Windows 识别的相机设备（{len(cams)} 个）:")
        for i, c in enumerate(cams):
            print(f"       [{i}] {c}")
    else:
        print("  Windows 识别的相机设备 : 无（可能没接上/没上电/驱动异常）")
    print("-" * 96)
    if args.dry_run:
        print("  --dry-run：不打开相机，仅环境检查完成")
        return 0
    if not cams and not args.force:
        print("\n  ❌ Windows 没有识别到任何相机设备 —— 先插好相机再跑这个工具。")
        print("     排查顺序：① USB 是否插紧/换个 USB 口；② 是否需要上电/开关打开；")
        print("               ③ 设备管理器里有没有黄色感叹号；④ 是否被别的程序独占。")
        print("     确认插好后重新运行本工具；只想跳过这一步可加 --force。")
        return 2

    successes: List[Dict[str, Any]] = []
    for be_name, api in BACKENDS:
        print(f"\n  >>> 后端 {be_name}")
        found_any = False
        for idx in range(args.max_index):
            rec = probe_one(api, idx, args.warmup, args.frames)
            if rec is None:
                continue
            found_any = True
            gw, gh = rec["got"]
            mark = "✅" if rec["requested_ok"] else "⚠️"
            print(f"      index={idx}  实际 {gw}x{gh} {mark}  FOURCC={rec['fourcc']:4s}  "
                  f"{rec['fps']:5.1f} FPS  左右亮度差 {rec['half_diff']:5.1f}  "
                  f"(后端={rec['backend']})")
            # 把没命中的候选尺寸也列出来，方便判断该后端支持哪些模式
            for r in rec.get("all", []):
                if not r.get("ok"):
                    print(f"          请求 {r['want'][0]}x{r['want'][1]} -> 打不开/读不到")
            successes.append({"backend": be_name, "index": idx, **{k: v for k, v in rec.items()
                                                                   if k != "frame"},
                              "_frame": rec.get("frame")})
        if not found_any:
            print("      没有任何设备能被这个后端打开")

    print("\n" + "=" * 96)
    print("  结论")
    print("=" * 96)
    if not successes:
        print("  ❌ 三个后端都打不开相机。可能原因：① 相机被别的程序占用；")
        print("     ② 需要管理员权限（再试一次管理员窗口对比）；③ USB/驱动异常（重插）")
        return 1
    exact = [s for s in successes if s.get("requested_ok")]
    if exact:
        s = exact[0]
        print(f"  ✅ 免提权可用：后端={s['backend']}  index={s['index']}  "
              f"{s['got'][0]}x{s['got'][1]}  {s['fps']:.1f} FPS")
        print(f"     -> EXE 打包可以不用管理员权限（用 {s['backend']} 后端）")
    else:
        best = max(successes, key=lambda x: x["fps"])
        print(f"  ⚠️ 能打开但拿不到 1280x480。最接近：后端={best['backend']} "
              f"index={best['index']} {best['got'][0]}x{best['got'][1]} {best['fps']:.1f} FPS")
        print("     注意：本项目内参是按 640x480/眼 标定的，换分辨率必须重新标定！")
    print(f"\n  共 {len(successes)} 个组合可用；明细见上表")

    # 存样图（供肉眼确认左右并排没错位）
    if successes and args.save:
        os.makedirs(args.save, exist_ok=True)
        for s in successes:
            fr = s.get("_frame")
            if fr is None:
                continue
            p = os.path.join(args.save, f"{s['backend']}_idx{s['index']}_{s['got'][0]}x{s['got'][1]}.png")
            cv2.imwrite(p, fr)
        print(f"  样图已保存: {os.path.relpath(args.save, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
