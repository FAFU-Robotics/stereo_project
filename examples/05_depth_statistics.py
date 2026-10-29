"""05_depth_statistics.py —— 跑一段时间，统计深度质量并写成 CSV

运行：
    python examples/05_depth_statistics.py --mock --seconds 3
    python examples/05_depth_statistics.py --seconds 30 --csv output/depth_stats.csv

用途：回答"这套配置在我的场景下稳不稳"。
单帧的 median 好看没有意义——真实项目关心的是有效像素比例是否掉、
中位深度是否漂、推理耗时是否有长尾。本脚本把这三件事逐帧记下来，
再用 p50 / p95 / min 三列给结论。

为什么要有这个脚本：02 只告诉你"能连续存 N 帧"，但不告诉你
"这 N 帧的质量有没有退化"。两者是不同的验收项。
"""

from __future__ import annotations

import csv
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import build_parser, open_camera      # noqa: E402


def _pct(values, q: float) -> float:
    """简单分位数（线性插值）。numpy.percentile 也行，这里避免多一个依赖假设。"""
    if not values:
        return float("nan")
    v = sorted(values)
    if len(v) == 1:
        return float(v[0])
    pos = (len(v) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (pos - lo))


def main() -> int:
    ap = build_parser("统计一段时间内的深度质量")
    ap.add_argument("--seconds", type=float, default=5.0, help="采样时长（秒）")
    ap.add_argument("--csv", default="", help="把逐帧数据写到这个 CSV（留空则不写）")
    ap.add_argument("--min-valid", type=float, default=0.0,
                    help="只统计有效像素比例 >= 该值的帧（0~1）")
    args = ap.parse_args()

    import numpy as np

    rows = []
    s = open_camera(args)
    t0 = time.time()
    last_seq = -1
    try:
        print(f"采样 {args.seconds:.1f} s …（--mock 时数据为合成场景，不代表真实精度）")
        while time.time() - t0 < args.seconds:
            if not s.has_new_result():
                time.sleep(0.01)
                continue
            r = s.take_result()
            if r is None or r.seq == last_seq:
                continue          # take_result() 不是消费式的，必须自己按 seq 去重
            last_seq = r.seq

            valid = np.asarray(r.valid, bool)
            ratio = float(valid.mean()) if valid.size else 0.0
            if ratio < args.min_valid:
                continue
            d = np.asarray(r.depth, np.float32)[valid]
            rows.append({
                "seq": int(r.seq),
                "t_s": round(time.time() - t0, 4),
                "preset": r.preset,
                "scale": float(r.scale),
                "valid_ratio": round(ratio, 4),
                "median_m": round(float(np.median(d)), 4) if d.size else 0.0,
                "p05_m": round(float(np.percentile(d, 5)), 4) if d.size else 0.0,
                "p95_m": round(float(np.percentile(d, 95)), 4) if d.size else 0.0,
                "points": int(len(r.points)),
                "infer_ms": round(float(r.timings.get("infer", 0.0)), 2),
            })
    finally:
        s.stop()

    dt = time.time() - t0
    if not rows:
        print("没有采到任何结果。真机模式下通常是模型还没加载完，可先用 --mock 看效果。")
        return 1

    ratio = [x["valid_ratio"] for x in rows]
    med = [x["median_m"] for x in rows]
    inf = [x["infer_ms"] for x in rows]

    print()
    print(f"帧数 {len(rows)}（{len(rows) / dt:.2f} 结果 FPS，统计窗口 {dt:.1f} s）")
    print(f"{'指标':<14}{'min':>10}{'p50':>10}{'p95':>10}{'max':>10}")
    for name, vals in (("有效像素比例", ratio), ("中位深度 (m)", med), ("推理耗时 (ms)", inf)):
        print(f"{name:<14}{min(vals):>10.4f}{_pct(vals, 0.5):>10.4f}"
              f"{_pct(vals, 0.95):>10.4f}{max(vals):>10.4f}")

    drift = abs(med[-1] - med[0])
    print()
    print(f"中位深度漂移（首帧→末帧）：{drift * 1000:.1f} mm")
    if drift > 0.05:
        print("  ⚠ 漂移超过 50 mm：留意曝光/自动增益变化，或场景里确有移动物体。")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"逐帧数据已写入 {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
