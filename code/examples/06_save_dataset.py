"""06_save_dataset.py —— 采一小段"数据集"：每帧一整套文件 + 一份索引清单

运行：
    python examples/06_save_dataset.py --mock --frames 5
    python examples/06_save_dataset.py --frames 60 --out output/dataset

与 02 的区别（两者都需要，不要混用）：
    02 只存点云，用 Recorder，文件名固定、PLY 为 ASCII —— 适合"快速看几帧"
    06 存整套（左右图/视差/深度/掩码/点云），用 export_result，PLY 为二进制
       并生成 manifest.csv —— 适合"交给别人复现"或"批量后处理"

为什么要 manifest.csv：深度数据的价值一半在索引。没有清单时，
别人拿到一堆 seq0007_*.ply 无法知道当时用的哪份标定、哪个预设、有效点多少。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import build_parser, open_camera      # noqa: E402


def main() -> int:
    ap = build_parser("采集一小段数据集（整套文件 + 索引清单）")
    ap.add_argument("--frames", type=int, default=5, help="采多少帧")
    ap.add_argument("--out", default=os.path.join("output", "dataset"), help="输出根目录")
    ap.add_argument("--min-valid", type=float, default=0.3,
                    help="有效像素比例低于该值的帧跳过（0~1）")
    ap.add_argument("--tiff", action="store_true", help="额外存 32 位 TIFF 深度（需 imagecodecs）")
    args = ap.parse_args()

    import numpy as np

    from stereo_sdk.calib import CalibInfo
    from stereo_sdk.export import export_result

    os.makedirs(args.out, exist_ok=True)

    # 先取标定摘要：写进清单，别人才能复现
    calib_brief = {}
    try:
        c = CalibInfo.from_file()
        calib_brief = {"path": os.path.abspath(c.path),
                       "baseline_mm": round(float(c.baseline_mm), 4),
                       "rect_fx": round(float(c.rect_fx), 4),
                       "rms_stereo_px": round(float(c.rms_stereo), 4),
                       "image_size": list(c.image_size)}
    except Exception as exc:                       # noqa: BLE001
        print(f"[提示] 未读到标定文件（{type(exc).__name__}: {exc}），清单里该项留空。")

    s = open_camera(args)
    rows, skipped = [], 0
    t0 = time.time()
    last_seq = -1
    try:
        print(f"目标 {args.frames} 帧，输出目录 {args.out}")
        while len(rows) < args.frames:
            if not s.has_new_result():
                time.sleep(0.01)
                if time.time() - t0 > 120:
                    raise TimeoutError("120 s 内没有采满目标帧数")
                continue
            r = s.take_result()
            if r is None or r.seq == last_seq:
                continue          # take_result() 不是幂等消费，必须按 seq 去重
            last_seq = r.seq

            valid = np.asarray(r.valid, bool)
            ratio = float(valid.mean()) if valid.size else 0.0
            if ratio < args.min_valid:
                skipped += 1
                continue

            prefix = f"seq{r.seq:04d}"
            paths = export_result(r, args.out, prefix=prefix, with_tiff=args.tiff,
                                  comments=[f"stereo_sdk {r.preset}",
                                            "camera frame, x right, y DOWN, z forward"])
            rows.append({
                "seq": int(r.seq),
                "prefix": prefix,
                "preset": r.preset,
                "scale": float(r.scale),
                "valid_ratio": round(ratio, 4),
                "median_m": round(float(np.median(np.asarray(r.depth)[valid])), 4),
                "points": int(len(r.points)),
                "infer_ms": round(float(r.timings.get("infer", 0.0)), 2),
                "cloud": os.path.basename(paths.get("cloud", "")),
            })
            print(f"  [{len(rows)}/{args.frames}] {prefix} 有效 {ratio * 100:5.1f}% "
                  f"中位 {rows[-1]['median_m']:.3f} m  点 {len(r.points):6d}")
    finally:
        s.stop()

    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": len(rows), "skipped_low_valid": skipped,
            "units": "meters", "coords": "camera frame, y DOWN",
            "colors": "PointResult.colors is BGR; PLY stores RGB",
            "calib": calib_brief,
            "elapsed_s": round(time.time() - t0, 2)}
    with open(os.path.join(args.out, "dataset.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print()
    print(f"完成：{len(rows)} 帧，跳过 {skipped} 帧（有效点不足 {args.min_valid:.0%}）")
    print(f"索引清单 {manifest}")
    print()
    print("后续可以直接这样读：")
    print("    import os, csv, numpy as np")
    print(f"    out = r'{os.path.abspath(args.out)}'")
    print("    rows = list(csv.DictReader(open(os.path.join(out, 'manifest.csv'), encoding='utf-8-sig')))")
    print("    depth = np.load(os.path.join(out, rows[0]['prefix'] + '.depth.npy'))   # 米，0 = 无效")
    return 0


if __name__ == "__main__":
    sys.exit(main())
