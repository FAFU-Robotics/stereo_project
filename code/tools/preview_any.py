r"""tools/preview_any.py —— 单摄像头"傻瓜预览"工具
=============================================================================
用途：**不依赖 config.yaml**，直接指定索引看这一路相机到底有没有画面。
      用于排查"到底是哪个索引对应哪个摄像头""画面是不是全黑""分辨率/格式能不能用"。

用法：
    python tools\preview_any.py --index 2
    python tools\preview_any.py --index 2 --res 1280x480 --fourcc MJPG
    python tools\preview_any.py --index 0 --backend msmf
    python tools\preview_any.py --index 2 --no-display --max-frames 30   （无人值守自检）
    python tools\preview_any.py --index 2 --save out\probe\idx2.png

窗口内按键：
    q / ESC  退出
    s        保存当前帧到 --save 指定路径（或 output\single_idx{N}.png）
    i        打印当前状态

屏幕上显示的信息：
    实际分辨率、实测 FPS、画面平均灰度 mean、标准差 std
    判断标准：mean≈0 且 std≈0 => 驱动没给真实数据（全黑），不是"环境太暗"。
              正常室内画面 mean 一般 20~200，std 一般 >10。
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera import BACKENDS  # noqa: E402

# "无有效信号"判定阈值（与 config.yaml 同名参数的默认值保持一致）：
#   灰度均值和标准差同时很小 => 驱动给的是空缓冲/噪声，而不是"环境太暗"
NO_SIGNAL_MEAN = 5.0
NO_SIGNAL_STD = 12.0


def main() -> int:
    p = argparse.ArgumentParser(description="单摄像头预览/自检工具")
    p.add_argument("--index", type=int, required=True, help="相机索引（OpenCV 用）")
    p.add_argument("--backend", default="dshow", choices=sorted(BACKENDS.keys()), help="视频后端")
    p.add_argument("--fourcc", default="", help="像素格式，如 MJPG / YUY2；留空表示不设置")
    p.add_argument("--res", default="", help="分辨率，如 1280x480；留空表示不设置")
    p.add_argument("--fps", type=int, default=30, help="请求帧率")
    p.add_argument("--warmup", type=int, default=15, help="丢弃的前若干帧")
    p.add_argument("--max-frames", type=int, default=0, help="跑满 N 帧自动退出（0=不限制）")
    p.add_argument("--no-display", action="store_true", help="不开窗口（无人值守自检）")
    p.add_argument("--save", default="", help="按 s 时的保存路径")
    args = p.parse_args()

    api = BACKENDS[args.backend]
    print(f"打开索引 {args.index}（后端 {args.backend}）…")
    cap = cv2.VideoCapture(args.index, api)
    if not cap.isOpened():
        print("  [失败] 打不开。可能：索引不对 / 设备被占用 / 该后端不支持按索引打开。")
        print("  建议：python tools\\probe_camera_backends.py --index", args.index)
        return 2

    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    if args.res:
        w, h = (int(x) for x in args.res.lower().split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    for _ in range(max(0, args.warmup)):
        cap.read()

    print("  已打开。开始读帧…（窗口内 q 退出，s 保存）")
    n = 0
    n_ok = 0
    n_blank = 0
    t0 = time.perf_counter()
    last: Optional[np.ndarray] = None
    try:
        while True:
            ok, frame = cap.read()
            t_read = time.perf_counter()
            if ok and frame is not None and frame.size > 0:
                n_ok += 1
                last = frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean, std = float(gray.mean()), float(gray.std())
                if mean < NO_SIGNAL_MEAN and std < NO_SIGNAL_STD:
                    n_blank += 1
            else:
                mean = std = -1.0
            n += 1
            dt = max(t_read - t0, 1e-9)
            fps = n_ok / dt

            if n % 15 == 0 or n == 1:
                shape = "无" if last is None else f"{last.shape[1]}x{last.shape[0]}"
                print(
                    f"  第 {n:4d} 帧  成功 {n_ok:4d}  分辨率 {shape}  "
                    f"实测 {fps:5.2f} fps  mean={mean:7.2f} std={std:7.2f}  "
                    f"{'【无有效信号：不是环境暗，是驱动没给真实图像】' if 0 <= mean < NO_SIGNAL_MEAN and std < NO_SIGNAL_STD else ''}"
                )
                if mean == -1.0:
                    print("        ↑ 这一帧读取失败（read() 返回 False）")

            if not args.no_display and last is not None:
                disp = last.copy()
                cv2.putText(disp, f"idx={args.index} {last.shape[1]}x{last.shape[0]} fps={fps:.1f}",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(disp, f"mean={mean:.1f} std={std:.1f}  q=quit s=save",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(f"preview idx {args.index}", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    path = args.save or os.path.join("output", f"single_idx{args.index}.png")
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    cv2.imwrite(path, last)
                    print(f"  已保存：{path}")
                if key == ord("i"):
                    print(f"  [状态] idx={args.index} 请求 W={cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} "
                          f"H={cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
                          f"FPS={cap.get(cv2.CAP_PROP_FPS):.1f} "
                          f"格式={int(cap.get(cv2.CAP_PROP_FOURCC))}")
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\n  中断")
    finally:
        cap.release()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass

    print("-" * 70)
    print(f"  总帧 {n}，成功 {n_ok}，其中判定『无有效信号』 {n_blank}")
    if n_ok == 0:
        print("  【结论】完全读不到帧：驱动/格式/占用问题。")
    elif n_blank >= n_ok:
        print("  【结论】所有帧都无有效信号（灰度均值与标准差同时很小）：")
        print("          这是驱动层没有给出真实图像数据，**不是**『环境太暗』。")
        print("          请用 Windows 相机 App 交叉验证，并检查 USB 端口/线材/镜头盖。")
    elif n_blank > 0:
        print(f"  【结论】{n_blank}/{n_ok} 帧无有效信号，画面质量不稳定，需要排查。")
    else:
        print("  【结论】能拿到真实画面。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
