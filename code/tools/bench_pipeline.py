"""bench_pipeline.py —— 分阶段基准测试（阶段 7.1）

=============================================================================
【与 bench_ffs.py 的分工】
    bench_ffs.py   : 只测 **推理** 单项（scale x valid_iters 矩阵），模型只加载一次
    bench_pipeline : 测 **整条链路的分阶段耗时** + 端到端 FPS + 显存
                     （校正 / 推理 / 点云 / 显示合成），并给出"三线程后能到多少"

【为什么需要它】
    实测当前端到端 4.99 FPS（≈200 ms/帧），其中推理 155 ms 占 78%。
    要判断"优化能从哪挤出来"，必须先把每一段的耗时量出来，而不是靠感觉。
    输出还能作为阶段 7.2/7.3/7.4 的**前后对比基线**。

【输入用什么图】
    默认取 output/captures 下最新一份采集里的 left_raw.png + right_raw.png。
    注意：启用标定校正后，这两个文件**本身就是校正后的图**，尺寸 640x480，
    正好等于运行时送进 FFS 的输入 —— 所以用它测出来的推理耗时是真实的。

【用法】
    python tools\\bench_pipeline.py                     REM 默认 1.0:8, 0.75:8, 0.75:6, 0.5:4
    python tools\\bench_pipeline.py --combos 1.0:8,0.5:4
    python tools\\bench_pipeline.py --repeat 10
    python tools\\bench_pipeline.py --dir output\\captures\\20261016_201251_cloud
    python tools\\bench_pipeline.py --no-cloud          REM 不测点云构建

【输出】
    控制台分阶段表格 + output\\bench\\pipeline_<时间戳>.json
=============================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from depth_utils import (CameraParams, colorize_depth, disp_correction_from_config,  # noqa: E402
                         disparity_to_depth, load_rectifier)
from pointcloud import PointCloudBuilder, cloud_color_bgr  # noqa: E402
from stereo_depth import FFSDepthEstimator  # noqa: E402


def newest_capture_with_pairs(root: str) -> str:
    """找最新一份同时含 left_raw.png / right_raw.png 的采集目录。"""
    cands = []
    for d in glob.glob(os.path.join(root, "output", "captures", "*")):
        if os.path.exists(os.path.join(d, "left_raw.png")) and \
           os.path.exists(os.path.join(d, "right_raw.png")):
            cands.append(d)
    if not cands:
        return ""
    return max(cands, key=os.path.getmtime)


def parse_combos(s: str) -> List[Tuple[float, int]]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        sc, it = part.split(":")
        out.append((float(sc), int(it)))
    return out


def med(xs: List[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="分阶段基准测试（阶段 7.1）：校正/推理/点云/显示 + 端到端")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    ap.add_argument("--dir", default="", help="含 left_raw.png/right_raw.png 的采集目录（默认取最新）")
    ap.add_argument("--combos", default="1.0:8,0.75:8,0.75:6,0.5:4",
                    help="性能组合 scale:iters，逗号分隔（默认 1.0:8,0.75:8,0.75:6,0.5:4）")
    ap.add_argument("--repeat", type=int, default=8, help="每个组合正式测量帧数（默认 8）")
    ap.add_argument("--warmup", type=int, default=2, help="预热帧数（默认 2）")
    ap.add_argument("--no-cloud", action="store_true", help="不测点云构建")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    project_cfg = dict(cfg.get("project", {}))
    for key in ("ffs_repo", "output_dir", "weights_dir", "calibration_dir"):
        if key in project_cfg:
            project_cfg[key] = os.path.normpath(os.path.join(PROJECT_ROOT, project_cfg[key]))

    frames_dir = args.dir or newest_capture_with_pairs(PROJECT_ROOT)
    if not frames_dir or not os.path.isdir(frames_dir):
        print("[错误] 找不到含 left_raw.png / right_raw.png 的采集目录；用 --dir 指定")
        return 1
    left = cv2.imread(os.path.join(frames_dir, "left_raw.png"))
    right = cv2.imread(os.path.join(frames_dir, "right_raw.png"))
    if left is None or right is None:
        print(f"[错误] 读不出图像：{frames_dir}")
        return 1

    eye_size = (left.shape[1], left.shape[0])
    params = CameraParams.from_config(cfg, eye_size)
    disp_scale, disp_offset = disp_correction_from_config(cfg)
    rectifier = load_rectifier(cfg)
    depth_cfg = cfg.get("depth", {}) or {}
    pc_cfg = dict(cfg.get("pointcloud", {}) or {})
    builder = None if args.no_cloud else PointCloudBuilder(pc_cfg, calibrated=params.calibrated)

    print("=" * 100)
    print("  阶段 7.1  分阶段基准测试")
    print("=" * 100)
    print(f"  图像来源 : {os.path.relpath(frames_dir, PROJECT_ROOT)}")
    print(f"  图像尺寸 : {eye_size[0]}x{eye_size[1]}  （启用校正时这两个文件本身已是校正后图像）")
    print(f"  相机参数 : {params.describe()}")
    print(f"  视差校正 : scale={disp_scale}  offset={disp_offset}")
    print(f"  标定校正 : {'启用（每帧会先做 rectify_pair）' if rectifier is not None else '未启用'}")
    print(f"  测量方案 : 预热 {args.warmup} 帧 + 正式 {args.repeat} 帧，取中位数")
    print("-" * 100)

    import torch
    rows: List[Dict[str, Any]] = []
    for scale, iters in parse_combos(args.combos):
        ffs_cfg = dict(cfg.get("ffs", {}))
        ffs_cfg["scale"] = float(scale)
        ffs_cfg["valid_iters"] = int(iters)
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        print(f"\n  >>> 组合 scale={scale}  valid_iters={iters}")
        t_load = time.perf_counter()
        try:
            est = FFSDepthEstimator(project_cfg, ffs_cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"      [跳过] 模型初始化失败：{type(exc).__name__}: {exc}")
            continue
        ok, msg = est.weights_ready()
        if not ok:
            print(f"      [跳过] {msg}")
            continue
        try:
            est.load()          # 必须先 load() 才能 infer()（stereo_depth 的接口约定）
        except Exception as exc:  # noqa: BLE001
            print(f"      [跳过] 模型加载失败：{type(exc).__name__}: {exc}")
            continue
        print(f"      模型就绪（加载 {time.perf_counter()-t_load:.1f}s）  输入 "
              f"{int(eye_size[0]*scale)}x{int(eye_size[1]*scale)}  max_disp={est.max_disp}")

        # 预热（用真实帧）
        for _ in range(max(args.warmup, 1)):
            src_l, src_r = (rectifier.rectify_pair(left, right) if rectifier is not None
                            else (left, right))
            est.infer(src_l, src_r)

        rec = {"scale": scale, "iters": iters, "rect": [], "infer": [], "cloud": [], "disp": []}
        for _ in range(max(args.repeat, 1)):
            t0 = time.perf_counter()
            if rectifier is not None:
                l2, r2 = rectifier.rectify_pair(left, right)
            else:
                l2, r2 = left, right
            t1 = time.perf_counter()
            out = est.infer(l2, r2)
            t2 = time.perf_counter()
            # 注意：scale<1.0 时模型输出的视差是**模型分辨率**（如 480x360），
            # 深度换算必须用与之匹配的内参（scaled_to_match），否则深度会整体错倍数。
            disp = out["disp"]
            p_use = params if disp.shape[1] == eye_size[0] else params.scaled_to_match(disp.shape[1])
            depth, valid = disparity_to_depth(
                disp, p_use,
                max_valid_disp=(est.max_disp * float(depth_cfg.get("disp_extrapolation_ratio", 0.98) or 0)),
                disp_scale=disp_scale, disp_offset=disp_offset)
            if builder is not None:
                builder.build(depth, p_use, cloud_color_bgr(l2, out.get("left_model_rgb"), depth.shape))
            t3 = time.perf_counter()
            # 显示合成（不含 OpenCV 窗口的 imshow/waitKey，那部分约 20ms 实测另计）
            dv = colorize_depth(depth, valid, float(pc_cfg.get("zmin_m", 0.15)),
                                float(pc_cfg.get("zmax_m", 3.0)))
            if dv.shape[0] != l2.shape[0] or dv.shape[1] != l2.shape[1]:
                dv = cv2.resize(dv, (l2.shape[1], l2.shape[0]), interpolation=cv2.INTER_NEAREST)
            panel = np.hstack([l2, r2, dv])
            cv2.putText(panel, f"scale {scale} iters {iters}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            t4 = time.perf_counter()
            rec["rect"].append((t1 - t0) * 1000)
            rec["infer"].append((t2 - t1) * 1000)
            rec["cloud"].append((t3 - t2) * 1000)
            rec["disp"].append((t4 - t3) * 1000)

        try:
            vram = float(torch.cuda.max_memory_allocated()) / (1024 ** 3)
        except Exception:
            vram = float("nan")
        row = {
            "scale": scale, "iters": iters,
            "rect_ms": med(rec["rect"]), "infer_ms": med(rec["infer"]),
            "cloud_ms": med(rec["cloud"]), "disp_ms": med(rec["disp"]),
            "vram_gb": vram,
            "input": f"{int(eye_size[0]*scale)}x{int(eye_size[1]*scale)}",
        }
        row["sum_ms"] = row["rect_ms"] + row["infer_ms"] + row["cloud_ms"] + row["disp_ms"]
        row["sum_fps"] = 1000.0 / row["sum_ms"] if row["sum_ms"] > 0 else float("nan")
        # 三线程后的估计：推理与"其余环节"并行 -> 帧周期 ≈ max(推理, 其余) + 少量同步开销
        row["threaded_ms"] = max(row["infer_ms"], row["rect_ms"] + row["cloud_ms"] + row["disp_ms"]) + 2.0
        row["threaded_fps"] = 1000.0 / row["threaded_ms"]
        rows.append(row)
        print(f"      校正 {row['rect_ms']:6.2f} | 推理 {row['infer_ms']:7.2f} | 点云 {row['cloud_ms']:6.2f} | "
              f"显示合成 {row['disp_ms']:5.2f} ms   -> 串行 {row['sum_fps']:5.2f} FPS / "
              f"三线程(估) {row['threaded_fps']:5.2f} FPS   显存 {vram:.2f} GB")
        del est
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n" + "=" * 100)
    print("  汇总（中位数；不含 OpenCV 窗口的 imshow/waitKey，实测那部分约 +20~25 ms）")
    print("=" * 100)
    print(f"  {'scale':>6}{'iters':>7}{'输入':>12}{'校正':>8}{'推理':>9}{'点云':>8}{'显示':>7}"
          f"{'串行 FPS':>10}{'三线程 FPS(估)':>16}{'显存GB':>8}")
    for r in rows:
        print(f"  {r['scale']:>6}{r['iters']:>7}{r['input']:>12}{r['rect_ms']:>8.2f}{r['infer_ms']:>9.2f}"
              f"{r['cloud_ms']:>8.2f}{r['disp_ms']:>7.2f}{r['sum_fps']:>10.2f}"
              f"{r['threaded_fps']:>16.2f}{r['vram_gb']:>8.2f}")
    print("-" * 100)
    print("  说明：")
    print("   * 串行 FPS = 1000 / 各阶段之和，对应现在单线程实现的端到端帧率（未计窗口开销）")
    print("   * 三线程 FPS(估) = 1000 / (max(推理, 其余) + 2ms)，是阶段 7.2 的预期上限")
    print("   * 精度影响不在本表：各档预设的真实深度误差需用已知距离数据离线测量（阶段 7.3）")

    out_dir = os.path.join(PROJECT_ROOT, "output", "bench")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"frames_dir": frames_dir, "eye_size": list(eye_size),
                   "camera": params.describe(), "disp_scale": disp_scale,
                   "disp_offset": disp_offset,
                   "rectify": rectifier is not None, "rows": rows}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\n  已保存: {os.path.relpath(out_path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
