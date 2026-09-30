r"""tools/bench_ffs.py —— FFS 性能基准（单进程，模型只加载一次）
=============================================================================
为什么单独写：
    用 tools/run_ffs_demo.py 跑参数矩阵要**每个配置重新加载一次模型**（本机约 3 秒），
    而且第一次推理还有冷启动开销，容易把数字看错。本脚本在一个进程里：
        加载模型一次 -> 对每个 (scale, valid_iters) 组合：
            用**真实尺寸的图**预热 -> 连续跑 N 次 -> 记录稳态均值
    并把结果写成 JSON，便于阶段 7 对比优化效果。

用法：
    python tools\bench_ffs.py
    python tools\bench_ffs.py --scales 1.0,0.5 --iters 8,4 --repeat 8
    python tools\bench_ffs.py --left a.png --right b.png

输出：
    控制台表格 + output\bench\bench_<时间戳>.json
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo_depth import FFSDepthEstimator  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description="FFS 性能基准（模型只加载一次）")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    ap.add_argument("--left", default="", help="左图（默认官方 demo_data/left.png）")
    ap.add_argument("--right", default="", help="右图（默认官方 demo_data/right.png）")
    ap.add_argument("--scales", default="1.0,0.5", help="要测的缩放系数，逗号分隔")
    ap.add_argument("--iters", default="8,4", help="要测的 valid_iters，逗号分隔")
    ap.add_argument("--repeat", type=int, default=6, help="每个组合的正式推理次数（第 1 次算冷启动）")
    ap.add_argument("--warmup", type=int, default=2, help="每个组合的预热次数（用真实尺寸图）")
    ap.add_argument("--size", default="", help="把输入图缩放到指定尺寸再测，如 640x480（用于模拟真实相机输入）")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("ffs_repo", "output_dir", "weights_dir"):
        cfg["project"][key] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["project"][key]))

    demo_dir = os.path.join(cfg["project"]["ffs_repo"], "demo_data")
    left_file = args.left or os.path.join(demo_dir, "left.png")
    right_file = args.right or os.path.join(demo_dir, "right.png")
    img_l = cv2.imread(left_file, cv2.IMREAD_COLOR)
    img_r = cv2.imread(right_file, cv2.IMREAD_COLOR)
    if img_l is None or img_r is None:
        print(f"[失败] 读不到图像：{left_file} / {right_file}")
        return 2
    if args.size:
        w, h = (int(x) for x in args.size.lower().split("x"))
        img_l = cv2.resize(img_l, (w, h))
        img_r = cv2.resize(img_r, (w, h))
        print(f"  [已按 --size 缩放输入到 {w}x{h}，用于模拟真实相机输入]")

    est = FFSDepthEstimator(cfg["project"], cfg["ffs"])
    ok, msg = est.weights_ready()
    if not ok:
        print("[失败] " + msg)
        return 3
    print("=" * 96)
    print("  FFS 性能基准（单进程；每组都用真实尺寸图预热，冷启动单独列出）")
    print("=" * 96)
    print(f"  图像      : {left_file} ({img_l.shape[1]}x{img_l.shape[0]})")
    print(f"  GPU       : {_gpu_name()}")
    est.load()

    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    iters_list = [int(x) for x in args.iters.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for scale in scales:
        for iters in iters_list:
            est.scale = scale
            est.valid_iters = iters
            if est.model is not None:
                est.model.args.valid_iters = iters
            # 显存峰值必须按组合重置，否则后面所有组合都会显示"历史最大"，看不出差异
            _reset_peak_mem()
            # 预热（真实尺寸）
            for _ in range(max(0, args.warmup)):
                est.infer(img_l, img_r)
            times: List[float] = []
            infer_ms: List[float] = []
            for _ in range(max(1, args.repeat)):
                out = est.infer(img_l, img_r)
                times.append(out["timings_ms"]["total"])
                infer_ms.append(out["timings_ms"]["infer"])
            steady = np.array(times[1:]) if len(times) > 1 else np.array(times)
            row = {
                "scale": scale,
                "valid_iters": iters,
                "model_input": list(out["model_size"]),
                "cold_total_ms": round(times[0], 1),
                "steady_total_ms_mean": round(float(steady.mean()), 1),
                "steady_total_ms_min": round(float(steady.min()), 1),
                "steady_infer_ms_mean": round(float(np.mean(infer_ms[1:] if len(infer_ms) > 1 else infer_ms)), 1),
                "steady_fps": round(1000.0 / float(steady.mean()), 2),
                "peak_mem_mb": round(est.stats["peak_mem_mb"], 0),
                "valid_disp_ratio": round(float((out["disp"] > 0).mean()), 4),
            }
            rows.append(row)
            print(f"  scale={scale:<4} iters={iters:<2} 输入={row['model_input'][0]}x{row['model_input'][1]:<5} "
                  f"冷启动={row['cold_total_ms']:>7.1f}ms  稳态={row['steady_total_ms_mean']:>7.1f}ms  "
                  f"{row['steady_fps']:>5.2f} FPS  显存={row['peak_mem_mb']:>6.0f}MB  有效视差={row['valid_disp_ratio']*100:.1f}%")

    out_dir = os.path.join(cfg["project"]["output_dir"], "bench")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"bench_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "gpu": _gpu_name(),
                "torch": _torch_ver(),
                "torchdynamo_disable": os.environ.get("TORCHDYNAMO_DISABLE"),
                "build_volume": est.build_volume,
                "image": left_file,
                "repeat": args.repeat,
                "rows": rows,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print("\n  结果已保存：" + path)
    print("  说明：稳态=第 2 次起的平均；冷启动含一次性显存池扩张/算子 autotune。")
    return 0


def _gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def _torch_ver() -> str:
    import torch
    return torch.__version__


def _reset_peak_mem() -> None:
    """重置 CUDA 显存峰值统计（每个测试组合开始前调用）。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
