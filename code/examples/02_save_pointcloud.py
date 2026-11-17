"""02_save_pointcloud.py —— 连续存 N 帧点云，并给出每帧耗时

运行：
    python examples/02_save_pointcloud.py --mock --frames 5

用途：验证"能不能连续跑"这件事。单帧好看不代表连续稳——
真实项目里 90% 的问题（显存涨、队列堆积、掉帧）都要连跑几分钟才暴露。
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import build_parser, open_camera      # noqa: E402


def main() -> int:
    ap = build_parser("连续保存 N 帧点云")
    ap.add_argument("--frames", type=int, default=5, help="存多少帧")
    ap.add_argument("--out", default=os.path.join("output", "example02"))
    args = ap.parse_args()

    from stereo_sdk.export import save_ply

    os.makedirs(args.out, exist_ok=True)
    s = open_camera(args)
    fps_t0, n_ok, n_wait = time.time(), 0, 0
    timings = []
    try:
        print(f"目标 {args.frames} 帧，输出目录 {args.out}")
        while n_ok < args.frames:
            if not s.has_new_result():
                n_wait += 1
                if n_wait > 6000:                  # 60 s 没结果就放弃，避免死等
                    raise TimeoutError("60 s 内没有新结果")
                time.sleep(0.01)
                continue
            n_wait = 0
            r = s.take_result()
            if r is None or not len(r.points):
                continue
            p = os.path.join(args.out, f"cloud_{r.seq:04d}.ply")
            save_ply(p, r.points, r.colors, binary=True,
                     comments=[f"seq={r.seq}", "units: meters", "y is DOWN"])
            n_ok += 1
            timings.append(float(r.timings.get("infer", float("nan"))))
            print(f"  [{n_ok}/{args.frames}] seq={r.seq:4d} points={len(r.points):6d} "
                  f"infer={r.timings.get('infer', 0):6.1f} ms  "
                  f"{os.path.getsize(p) / 1024 / 1024:.2f} MB")
    finally:
        s.stop()

    dt = time.time() - fps_t0
    print()
    print(f"共 {n_ok} 帧，用时 {dt:.2f} s，端到端 {n_ok / dt:.2f} fps（含写盘）")
    if timings and all(t == t for t in timings):
        import statistics as st
        print(f"推理耗时 p50={st.median(timings):.1f} ms  max={max(timings):.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
