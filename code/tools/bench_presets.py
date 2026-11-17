"""bench_presets.py —— 各"速度/精度预设"的真实深度误差（阶段 7.3，纯离线）

=============================================================================
【要回答的问题】
    阶段 7.1 测出了各 scale 的速度（1.0 -> 164ms，0.75 -> 95ms，0.5 -> 47ms），
    但**精度代价没测**。本工具用现有"已知真实距离"的采集数据，
    把每档预设都跑一遍，量出它实际的深度误差 —— 这样选档才有依据。

【真值从哪来】
    板子② 是 5x7 / 36mm 的 ChArUco 板，打印后实测方格 35.983 mm（误差 0.05%），
    所以用它的位姿（solvePnP，基于标定的 P1 内参）就能得到**每个角点的真实距离 Z**。

【怎么算误差】
    对每帧：在角点像素处采样该档预设算出的深度 -> 与位姿给出的真实 Z 比较
    误差 = 中位深度 / 中位真值 - 1

【为什么同一套视差校正(1.0221)适用于所有档】
    校正是"视差乘以 1.0221"（纯比例），与分辨率无关；
    各档之间偏差是否随分辨率变化，正是本工具要测的。

【用法】
    python tools\\bench_presets.py
    python tools\\bench_presets.py --combos 1.0:8,0.75:8,0.5:4
    python tools\\bench_presets.py --repeat 30          REM 每帧重复推理次数（默认 1，取一次即可）
=============================================================================
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from depth_utils import (CameraParams, disp_correction_from_config,  # noqa: E402
                         disparity_to_depth)
from stereo_depth import FFSDepthEstimator  # noqa: E402

# 已知真实距离的采集（板子② 5x7/36mm 的位姿给真值）
DEFAULT_CAPS = ["20261016_195821_cloud", "20261016_201251_cloud",
                "20261018_193125_cloud", "20261018_193219_cloud", "20261018_193225_cloud"]


def load_aruco_module():
    spec = importlib.util.spec_from_file_location(
        "aa", os.path.join(PROJECT_ROOT, "tools", "aruco_align.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="各预设的真实深度误差（离线）")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    ap.add_argument("--combos", default="1.0:8,0.75:8,0.5:4", help="scale:iters 列表")
    ap.add_argument("--caps", default=",".join(DEFAULT_CAPS), help="采集目录名列表（逗号分隔）")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    project_cfg = dict(cfg.get("project", {}))
    for key in ("ffs_repo", "output_dir", "weights_dir", "calibration_dir"):
        if key in project_cfg:
            project_cfg[key] = os.path.normpath(os.path.join(PROJECT_ROOT, project_cfg[key]))

    cal = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "calibration", "stereo_calib.yaml"),
                              encoding="utf-8"))
    P1 = np.asarray(cal["P1"], np.float64)
    FX, FY = float(P1[0, 0]), float(P1[1, 1])
    CX, CY = float(P1[0, 2]), float(P1[1, 2])
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float64)
    B = float(cal["baseline_mm"]) / 1000.0
    disp_scale, disp_offset = disp_correction_from_config(cfg)

    aa = load_aruco_module()
    board = aa.build_board(5, 7, 36.0, 27.0, "DICT_4X4_1000")
    det = aa.make_detector(board)

    caps = [c.strip() for c in args.caps.split(",") if c.strip()]
    eye_size = (640, 480)
    params = CameraParams(fx=FX, fy=FY, cx=CX, cy=CY, baseline_m=B,
                          calibrated=True, width=eye_size[0], height=eye_size[1])

    print("=" * 104)
    print("  阶段 7.3  各预设的真实深度误差（离线，真值来自 ChArUco 板位姿）")
    print("=" * 104)
    print(f"  视差校正: scale={disp_scale} offset={disp_offset}   基线 {B*1000:.3f} mm   校正后 fx {FX:.2f}")

    # 预先算好每帧的真值（与预设无关，只算一次）
    truth: Dict[str, Dict[str, Any]] = {}
    for cap in caps:
        d = os.path.join(PROJECT_ROOT, "output", "captures", cap)
        img = cv2.imread(os.path.join(d, "left_raw.png"))
        if img is None:
            print(f"  [跳过] 读不到 {cap}")
            continue
        res = aa.detect_one(det, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if res["n_corners"] < 6:
            print(f"  [跳过] {cap} 板子检出不足（{res['n_corners']} 角点）")
            continue
        rvec, tvec, rms, n = aa.pose_from_corners(board, res, K, np.zeros(5), dist_model="none")
        R = cv2.Rodrigues(rvec)[0]
        objp, _ = board.matchImagePoints(res["corners"], res["ids"])
        objp = np.asarray(objp, np.float64).reshape(-1, 3)
        z_true = ((R @ objp.T).T + tvec.reshape(3))[:, 2]
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, None)
        truth[cap] = {"uv": proj.reshape(-1, 2), "z": z_true, "rms": rms,
                      "img": img, "n_markers": res["n_markers"]}
    if not truth:
        print("[错误] 没有可用样本")
        return 1

    rows: List[Dict[str, Any]] = []
    import torch
    for combo in [c for c in args.combos.split(",") if c.strip()]:
        sc_s, it_s = combo.split(":")
        scale, iters = float(sc_s), int(it_s)
        ffs_cfg = dict(cfg.get("ffs", {}))
        ffs_cfg["scale"], ffs_cfg["valid_iters"] = scale, iters
        print(f"\n>>> 预设 scale={scale} iters={iters}（输入 {int(eye_size[0]*scale)}x{int(eye_size[1]*scale)}）")
        try:
            est = FFSDepthEstimator(project_cfg, ffs_cfg)
            ok, msg = est.weights_ready()
            if not ok:
                print(f"    [跳过] {msg}")
                continue
            est.load()
        except Exception as exc:  # noqa: BLE001
            print(f"    [跳过] 模型初始化失败: {type(exc).__name__}: {exc}")
            continue

        recs = []
        for cap, t in truth.items():
            d = os.path.join(PROJECT_ROOT, "output", "captures", cap)
            left = cv2.imread(os.path.join(d, "left_raw.png"))
            right = cv2.imread(os.path.join(d, "right_raw.png"))
            t_inf = time.perf_counter()
            out = est.infer(left, right)
            infer_ms = (time.perf_counter() - t_inf) * 1000.0
            disp = out["disp"]
            k = disp.shape[1] / float(eye_size[0])          # 模型分辨率 / 原图分辨率
            p_use = params if abs(k - 1.0) < 1e-6 else params.scaled_to_match(disp.shape[1])
            d_max = est.max_disp * float((cfg.get("depth", {}) or {}).get("disp_extrapolation_ratio", 0.98) or 0)
            depth, valid = disparity_to_depth(disp, p_use, max_valid_disp=d_max,
                                              disp_scale=disp_scale, disp_offset=disp_offset)
            # 在角点处采样（角点坐标是原图坐标，要按 k 缩放到深度图坐标）
            vals = []
            for (u, v) in t["uv"]:
                ui, vi = int(round(u * k)), int(round(v * k))
                if not (0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1]):
                    continue
                q = depth[max(vi - 1, 0):vi + 2, max(ui - 1, 0):ui + 2]
                q = q[q > 0]
                if q.size:
                    vals.append(float(np.median(q)))
            if len(vals) < 6:
                print(f"    {cap}: 有效深度采样不足（{len(vals)}），跳过")
                continue
            z_meas = float(np.median(vals))
            z_true = float(np.median(t["z"]))
            recs.append({"cap": cap, "z_true_mm": z_true * 1000, "z_meas_mm": z_meas * 1000,
                         "err_mm": (z_meas - z_true) * 1000,
                         "err_pct": (z_meas / z_true - 1.0) * 100.0,
                         "infer_ms": infer_ms, "n": len(vals),
                         "valid_ratio": float(valid.mean()) * 100.0})
            print(f"    {cap}: 真值 {z_true*1000:7.1f} mm  测得 {z_meas*1000:7.1f} mm  "
                  f"误差 {recs[-1]['err_mm']:+6.2f} mm ({recs[-1]['err_pct']:+.2f}%)  "
                  f"有效 {recs[-1]['valid_ratio']:.1f}%  推理 {infer_ms:.0f} ms")
        del est
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if not recs:
            continue
        errs = [abs(r["err_pct"]) for r in recs]
        rows.append({"scale": scale, "iters": iters,
                     "infer_ms": statistics.median([r["infer_ms"] for r in recs]),
                     "mean_abs_pct": statistics.mean(errs),
                     "max_abs_pct": max(errs),
                     "max_abs_mm": max(abs(r["err_mm"]) for r in recs),
                     "per_frame": recs})

    print("\n" + "=" * 104)
    print("  汇总：速度 ↔ 精度")
    print("=" * 104)
    print(f"  {'scale':>6}{'iters':>7}{'推理ms':>9}{'串行FPS':>9}{'三线程FPS':>11}"
          f"{'平均误差%':>11}{'最大误差%':>11}{'最大误差mm':>12}")
    for r in rows:
        fps_serial = 1000.0 / (r["infer_ms"] + 35.0)     # 其余环节约 35ms（bench_pipeline 实测）
        fps_thread = 1000.0 / (max(r["infer_ms"], 35.0) + 2.0)
        r["serial_fps"], r["threaded_fps"] = fps_serial, fps_thread
        print(f"  {r['scale']:>6}{r['iters']:>7}{r['infer_ms']:>9.1f}{fps_serial:>9.2f}"
              f"{fps_thread:>11.2f}{r['mean_abs_pct']:>11.2f}{r['max_abs_pct']:>11.2f}"
              f"{r['max_abs_mm']:>12.2f}")
    print("-" * 104)
    print("  说明：样本是 0.344~0.392 m 的 5 组已知距离；误差 = 测得深度/真实距离 - 1")
    print("        '其余环节约 35ms' 取自 tools/bench_pipeline.py 实测（点云+显示+校正）")

    out_dir = os.path.join(PROJECT_ROOT, "output", "bench")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"presets_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"disp_scale": disp_scale, "disp_offset": disp_offset, "rows": rows},
                  fh, ensure_ascii=False, indent=2)
    print(f"\n  已保存: {os.path.relpath(out_path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
