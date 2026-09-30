r"""tools/run_ffs_demo.py —— 用**官方示例图像**验证 FFS（阶段 2 的验收脚本，Windows 友好版）
=============================================================================
为什么要单独写这个脚本，而不直接跑官方的 scripts/run_demo.py：
    官方 run_demo.py 里有几处 Windows 上会出问题的写法：
      1) os.system('rm -rf {out_dir} && mkdir -p {out_dir}')  —— Windows 的 cmd 没有 rm / mkdir -p
      2) 结尾 cv2.waitKey(0) + Open3D 窗口，无人值守时无法自动结束
      3) 直接 imageio.imread 读图（RGB），与本项目 camera.py 给的 BGR 顺序不一致
    本脚本**复用官方的模型加载、预处理、后处理代码**（core.*、Utils.*），
    只把上述"外壳"换成 Windows/本项目可用的写法，并把结果与耗时记录下来。

用法：
    REM 先检查权重是否就位（不加载模型，秒回）
    python tools\run_ffs_demo.py --check-only

    REM 完整跑一遍官方示例（需要权重）
    python tools\run_ffs_demo.py

    REM 不看窗口 + 重复 10 次统计耗时 + 缩小输入提速
    python tools\run_ffs_demo.py --no-display --repeat 10 --scale 0.5

输出（默认 output\ffs_demo\<时间戳>\）：
    left.png / right.png    送进模型的左右图（缩放后）
    disp.npy                视差（float32，模型输入分辨率，单位：像素）
    disp_vis.png            左图 | 右图 | 视差伪彩 拼图
    depth_meter.npy         深度（float32，米），由 disp 与内参/基线算出
    cloud.ply               点云（用左图着色，已按 zfar 过滤）
    meta.txt                本次运行的全部参数与来源说明（含是否标定）
    timings.txt             耗时统计
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo_depth import FFSError, FFSDepthEstimator  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> Dict[str, Any]:
    """读取 config.yaml 并把相对路径解析为绝对路径（与 main.py 行为一致）。"""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("ffs_repo", "output_dir", "weights_dir", "calibration_dir"):
        if key in cfg.get("project", {}):
            cfg["project"][key] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["project"][key]))
    return cfg


def read_intrinsics(path: str) -> Tuple[np.ndarray, float]:
    """读取官方格式的内参文件。

    格式（官方 readme 明确说明）：
        第 1 行：9 个数字 = 展开的 3x3 内参矩阵
        第 2 行：基线 baseline，单位 **米**
    输出：(K(3x3 float32), baseline(float))
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"内参文件格式不对（至少要有 2 行）：{path}")
    K = np.array(list(map(float, lines[0].split()))).astype(np.float32).reshape(3, 3)
    baseline = float(lines[1].split()[0])
    return K, baseline


def make_out_dir(root: str) -> str:
    """创建本次运行的输出目录（时间戳命名，不覆盖历史结果）。"""
    path = os.path.join(root, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="FFS 官方示例图像验证（Windows 友好版）")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    parser.add_argument("--left", default="", help="左图路径（默认用官方 demo_data/left.png）")
    parser.add_argument("--right", default="", help="右图路径（默认用官方 demo_data/right.png）")
    parser.add_argument("--intrinsic", default="", help="内参文件（默认用官方 demo_data/K.txt）")
    parser.add_argument("--out-dir", default="", help="输出目录（默认 output/ffs_demo/<时间戳>）")
    parser.add_argument("--scale", type=float, default=None, help="覆盖 ffs.scale（0.5 可明显提速）")
    parser.add_argument("--valid-iters", type=int, default=None, help="覆盖 ffs.valid_iters")
    parser.add_argument("--max-disp", type=int, default=None, help="覆盖 ffs.max_disp")
    parser.add_argument("--repeat", type=int, default=1, help="推理重复次数（>1 用于统计稳定耗时）")
    parser.add_argument("--zfar", type=float, default=100.0, help="点云最大深度（米）")
    parser.add_argument("--no-display", action="store_true", help="不打开 Open3D 窗口")
    parser.add_argument("--no-pointcloud", action="store_true", help="不生成点云")
    parser.add_argument("--check-only", action="store_true", help="只检查权重与环境，不推理")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ffs_cfg = dict(cfg.get("ffs", {}))
    if args.scale is not None:
        ffs_cfg["scale"] = args.scale
    if args.valid_iters is not None:
        ffs_cfg["valid_iters"] = args.valid_iters
    if args.max_disp is not None:
        ffs_cfg["max_disp"] = args.max_disp

    # 官方的默认输入
    demo_dir = os.path.join(str(cfg["project"]["ffs_repo"]), "demo_data")
    left_file = args.left or os.path.join(demo_dir, "left.png")
    right_file = args.right or os.path.join(demo_dir, "right.png")
    intr_file = args.intrinsic or os.path.join(demo_dir, "K.txt")

    print("=" * 92)
    print("  阶段 2：Fast-FoundationStereo 官方示例图像验证")
    print("=" * 92)
    print(f"  官方仓库 : {cfg['project']['ffs_repo']}")
    print(f"  PyTorch  : {_torch_version()}")
    if sys.platform == "win32":
        print(f"  TORCHDYNAMO_DISABLE = {os.environ.get('TORCHDYNAMO_DISABLE')}（Windows 无 Triton，必须为 1）")

    est = FFSDepthEstimator(cfg["project"], ffs_cfg)

    # ---------- 权重与环境检查 ----------
    ok, msg = est.weights_ready()
    print(f"\n  [检查] {msg}")
    if not ok:
        print("\n  结论：**权重尚未就位，无法进入阶段 2 的推理验证。**")
        print("  下载方式（研究版，需能访问 Google Drive）：")
        print("    官方 readme 给的 Google Drive 文件夹 -> 下载 23-36-37 整个文件夹")
        print(f"    放到：{est.weights_dir}")
        print(f"    该目录下必须同时有：{est.model_file} 和 cfg.yaml")
        return 3
    if args.check_only:
        print("\n  [检查] 权重文件完整。--check-only 模式到此结束（未加载模型、未推理）。")
        return 0

    # ---------- 加载模型 ----------
    try:
        est.load()
    except FFSError as exc:
        print(f"\n  [失败] {exc}")
        return 4

    # ---------- 读图 ----------
    img_l = cv2.imread(left_file, cv2.IMREAD_COLOR)
    img_r = cv2.imread(right_file, cv2.IMREAD_COLOR)
    if img_l is None or img_r is None:
        print(f"  [失败] 读不到图像：{left_file} / {right_file}")
        return 5
    K, baseline = read_intrinsics(intr_file)
    print(f"\n  输入图像 : left={img_l.shape[1]}x{img_l.shape[0]}  right={img_r.shape[1]}x{img_r.shape[0]}")
    print(f"  内参与基线: fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}  baseline={baseline} m")
    print("             来源：官方 demo_data/K.txt（这是官方示例数据自带的标定，可用于该示例图）")

    # ---------- 预热 + 推理 ----------
    print("\n  预热中（官方提示：第一次运行会因编译/缓存很慢，不计入统计）…")
    print("           注意：这里用**真实尺寸的图**预热，因为 CUDA 按张量形状缓存算子；")
    print("           用小图预热会让第一次正式推理仍然冷启动（实测 2088 ms vs 270 ms）。")
    est.warmup(est.warmup_iters, left=img_l, right=img_r)

    print(f"  正式推理 {args.repeat} 次 …")
    times: List[float] = []
    out: Dict[str, Any] = {}
    for i in range(max(1, args.repeat)):
        t0 = time.perf_counter()
        out = est.infer(img_l, img_r)
        times.append((time.perf_counter() - t0) * 1000.0)
        if args.repeat > 1:
            print(f"    第 {i + 1}/{args.repeat} 次：total={times[-1]:.1f} ms  "
                  f"infer={out['timings_ms']['infer']:.1f} ms")

    disp = out["disp"]                 # 模型输入分辨率下的视差（K 需要乘 scale）
    scale = out["scale"]
    left_model = out["left_model_rgb"]
    timings = out["timings_ms"]

    times_arr = np.array(times)
    print("\n" + "-" * 92)
    print("  耗时统计（毫秒）")
    print(f"    本次 total      : {timings['total']:.1f}")
    print(f"    本次 预处理     : {timings['preprocess']:.1f}")
    print(f"    本次 模型推理   : {timings['infer']:.1f}")
    if args.repeat > 1:
        # 第一次仍可能包含一次性开销（显存池扩张/算子 autotune），单独列出来更诚实
        steady = times_arr[1:] if len(times_arr) > 1 else times_arr
        print(f"    第 1 次(冷启动)  : {times_arr[0]:.1f}")
        print(f"    稳态平均(第2次起): {steady.mean():.1f}  最小 {steady.min():.1f}  最大 {steady.max():.1f}")
        print(f"    等效端到端帧率   : {1000.0 / steady.mean():.2f} FPS（稳态、仅推理，不含采集与显示）")
    print(f"    模型输入尺寸     : {out['model_size'][0]}x{out['model_size'][1]}  (scale={scale})")
    if est.stats["peak_mem_mb"] > 0:
        print(f"    GPU 峰值显存     : {est.stats['peak_mem_mb']:.0f} MB")

    valid_ratio = float((disp > 0).mean())
    print("\n  视差质量")
    print(f"    有效视差比例     : {valid_ratio * 100:.1f}%   （>0 的像素占比）")
    print(f"    视差范围         : min={disp.min():.2f}  max={disp.max():.2f}  中位={np.median(disp):.2f}")

    # ---------- 视差转深度（官方公式：depth = fx * baseline / disp）----------
    K_scaled = K.copy()
    K_scaled[:2] *= scale          # 官方 run_demo.py 的做法：内参随缩放同步
    fx = float(K_scaled[0, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp > 0, fx * baseline / np.maximum(disp, 1e-6), 0.0)
    depth[disp <= 0] = 0.0
    d_valid = depth[(depth > 0) & np.isfinite(depth)]
    print("\n  深度（米，按官方公式 depth = fx*B/disp 换算）")
    if d_valid.size:
        print(f"    有效深度比例     : {d_valid.size / depth.size * 100:.1f}%")
        print(f"    深度范围         : min={d_valid.min():.3f}  中位={np.median(d_valid):.3f}  max={d_valid.max():.3f}")
    else:
        print("    [警告] 没有任何有效深度像素 —— 视差图为空或全 0，请检查输入图像与左右顺序！")

    # ---------- 保存 ----------
    out_dir = args.out_dir or make_out_dir(os.path.join(str(cfg["project"]["output_dir"]), "ffs_demo"))
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    p = os.path.join(out_dir, "left.png");   cv2.imwrite(p, cv2.cvtColor(left_model, cv2.COLOR_RGB2BGR)); saved.append(p)
    p = os.path.join(out_dir, "disp.npy");   np.save(p, disp); saved.append(p)
    p = os.path.join(out_dir, "depth_meter.npy"); np.save(p, depth.astype(np.float32)); saved.append(p)

    # 伪彩视差图：复用官方 Utils.vis_disparity（它返回 RGB 顺序，cv2.imwrite 需要 BGR）
    from Utils import vis_disparity  # 官方实现
    vis = vis_disparity(disp, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
    right_model = cv2.resize(
        cv2.cvtColor(cv2.imread(right_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
        (left_model.shape[1], left_model.shape[0]),
    )
    panel = np.concatenate([left_model, right_model, vis.astype(np.uint8)], axis=1)
    p = os.path.join(out_dir, "disp_vis.png")
    cv2.imwrite(p, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    saved.append(p)

    n_points = 0
    if not args.no_pointcloud:
        from Utils import depth2xyzmap, toOpen3dCloud, o3d  # 官方实现
        if o3d is None:
            print("  [警告] open3d 不可用，跳过点云导出")
        else:
            xyz_map = depth2xyzmap(depth, K_scaled)
            pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), left_model.reshape(-1, 3))
            pts = np.asarray(pcd.points)
            keep = (pts[:, 2] > 0) & (pts[:, 2] <= args.zfar) & np.isfinite(pts[:, 2])
            pcd = pcd.select_by_index(np.arange(len(pts))[keep])
            n_points = len(np.asarray(pcd.points))
            p = os.path.join(out_dir, "cloud.ply")
            o3d.io.write_point_cloud(p, pcd)
            saved.append(p)
            print(f"\n  点云             : {n_points} 个点（z 范围 0~{args.zfar} m）-> {os.path.basename(p)}")

    # 参数与来源说明（图像和参数必须成套保存）
    meta = [
        "# FFS 官方示例验证 —— 本次运行参数与数据说明",
        f"时间            : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"左图 / 右图     : {left_file} / {right_file}",
        f"内参文件        : {intr_file}",
        f"内参(缩放后)    : fx={fx:.4f} fy={K_scaled[1,1]:.4f} cx={K_scaled[0,2]:.4f} cy={K_scaled[1,2]:.4f}",
        f"基线(米)        : {baseline}",
        f"内参来源说明    : 官方 demo_data/K.txt —— 对官方示例图像而言这是**有效标定**",
        f"模型权重        : {est.model_path}",
        f"valid_iters     : {est.valid_iters}",
        f"max_disp        : {est.max_disp}",
        f"scale           : {scale}",
        f"代价体后端      : {est.build_volume}（Windows 无 Triton，只能用 pytorch1）",
        f"torch.compile   : 已禁用（TORCHDYNAMO_DISABLE=1）",
        f"模型输入尺寸    : {out['model_size'][0]}x{out['model_size'][1]}",
        f"有效视差比例    : {valid_ratio * 100:.1f}%",
        f"耗时(ms)        : total={timings['total']:.1f} preprocess={timings['preprocess']:.1f} infer={timings['infer']:.1f}",
        f"点云点数        : {n_points}",
        "",
        "单位说明：disp.npy 单位是像素（模型输入分辨率下）；depth_meter.npy 单位是米；",
        "          cloud.ply 坐标系为左相机光学坐标系（x 右, y 下, z 前），单位米。",
    ]
    p = os.path.join(out_dir, "meta.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(meta) + "\n")
    saved.append(p)

    p = os.path.join(out_dir, "timings.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("total_ms per run: " + ", ".join(f"{t:.1f}" for t in times) + "\n")
        if args.repeat > 1:
            steady = times_arr[1:] if len(times_arr) > 1 else times_arr
            fh.write(f"cold_start_ms={times_arr[0]:.1f}\n")
            fh.write(f"steady_mean_ms={steady.mean():.1f} steady_min_ms={steady.min():.1f} steady_max_ms={steady.max():.1f}\n")
            fh.write(f"steady_fps={1000.0 / steady.mean():.2f}\n")
        fh.write(f"valid_iters={est.valid_iters} max_disp={est.max_disp} scale={scale}\n")
        fh.write(f"model_input={out['model_size'][0]}x{out['model_size'][1]} peak_mem_mb={est.stats['peak_mem_mb']:.0f}\n")
    saved.append(p)

    # ---------- 验收结论 ----------
    print("\n" + "=" * 92)
    print("  阶段 2 验收结论")
    print("=" * 92)
    checks = [
        ("权重文件完整（.pth + cfg.yaml）", True),
        ("模型在 GPU 上加载成功", est.loaded),
        ("推理成功并输出视差图", disp.size > 0),
        ("视差图非空（有效像素 > 5%）", valid_ratio > 0.05),
        ("保存了视差/深度/伪彩图", os.path.exists(os.path.join(out_dir, "disp.npy"))),
    ]
    all_ok = True
    for name, good in checks:
        print(f"    [{'通过' if good else '未通过'}] {name}")
        all_ok = all_ok and bool(good)
    if not all_ok:
        print("\n  注意：若有未通过项，请把上面的完整输出发我，再定位原因。")
    print(f"\n  输出目录：{out_dir}")
    for s in saved:
        print(f"    {s}")

    # ---------- 可选可视化 ----------
    if not args.no_display and not args.no_pointcloud and n_points > 0:
        from Utils import o3d
        print("\n  打开点云窗口（关闭窗口即可退出程序）…")
        vis_pcd = o3d.io.read_point_cloud(os.path.join(out_dir, "cloud.ply"))
        o3d.visualization.draw_geometries([vis_pcd], window_name="FFS demo point cloud (ESC to quit)")
    return 0


def _torch_version() -> str:
    """延迟 import torch，避免在 --help 时也把 CUDA 初始化起来。"""
    import torch
    return f"{torch.__version__} (CUDA {torch.version.cuda}, available={torch.cuda.is_available()})"


if __name__ == "__main__":
    raise SystemExit(main())
