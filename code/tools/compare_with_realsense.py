"""compare_with_realsense.py —— 把 DIY 双目与 RealSense 的点云导出到同一板坐标系，并做一致性量化

=============================================================================
【用途】
    验证"自制双目相机的深度"和"一台出厂标定的 RealSense"在**同一世界坐标系**下差多少。
    输出两份可直接拖进 CloudCompare 的 PLY（板坐标系，单位米），以及量化指标。

【原理】
    两台相机各自解出同一块 ChArUco 板(5x7 / 36mm)的位姿，
    用该位姿把各自点云变换到"板坐标系"——板子就是两台的**公共世界坐标系**。
    于是：
      - 视觉上：两份点云应重合（CloudCompare 里能直接看出来）
      - 量化上：板面 z@原点 的差 = 两台相机的深度尺度差

【真实数据踩过的两个坑（务必保留）】
    1) **D405 的深度单位是 0.1 mm**（depth_scale=0.0001 m），不是 1 mm。
       写死 `* 0.001` 会让所有深度差 10 倍 → 必须从传感器读 `get_depth_scale()`。
    2) pyrealsense2 的 intrinsics 主点字段叫 **ppx / ppy**，不是 cx / cy。
    3) db3 回放时用 rs.align(stream.color) 把深度对齐到彩色帧，
       这样深度与彩色共用同一套内参，反投影才是同一个坐标系。

【用法】
    python tools\\compare_with_realsense.py                       REM 自动取最新的 DIY 采集 + 最新 db3
    python tools\\compare_with_realsense.py --diy output\\captures\\20261021_164225_cloud
    python tools\\compare_with_realsense.py --db3 "C:\\Users\\31931\\Documents\\20261021_164319.db3"
    python tools\\compare_with_realsense.py --out output\\realign\\v4 --no-export   REM 只算不导出

【输出】
    <out>\\diy_board.ply        自制相机点云（板坐标系，米，带颜色）
    <out>\\realsense_board.ply  RealSense 点云（板坐标系，米，带颜色）
    控制台：板面 z@原点、平面残差、板区最近邻距离统计
=============================================================================
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 板子纸面矩形（米）：ChArUco 5x7 / 36mm，图案 180x252mm，边缘留一点余量
X0, X1, Y0, Y1 = 0.005, 0.175, 0.005, 0.247


def newest(pattern: str, need_file: str = "") -> str:
    if need_file:
        cands = [d for d in glob.glob(pattern) if os.path.exists(os.path.join(d, need_file))]
    else:
        cands = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    return max(cands, key=os.path.getmtime) if cands else ""


def load_aruco():
    spec = importlib.util.spec_from_file_location(
        "aa", os.path.join(PROJECT_ROOT, "tools", "aruco_align.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_plane_z(points: np.ndarray):
    """板面矩形内拟合 z = ax+by+c，返回 (z@原点 mm, 点数, 残差RMS mm)"""
    p = np.asarray(points, np.float64)
    m = (p[:, 0] > X0) & (p[:, 0] < X1) & (p[:, 1] > Y0) & (p[:, 1] < Y1) & (np.abs(p[:, 2]) < 0.05)
    p = p[m]
    if len(p) < 200:
        return float("nan"), len(p), float("nan")
    A = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    coef, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
    resid = p[:, 2] - A @ coef
    return coef[2] * 1000.0, len(p), float(np.sqrt((resid ** 2).mean())) * 1000.0


def board_subset(p: np.ndarray) -> np.ndarray:
    m = (p[:, 0] > X0) & (p[:, 0] < X1) & (p[:, 1] > Y0) & (p[:, 1] < Y1) & (np.abs(p[:, 2]) < 0.05)
    return p[m]


def c2c(src: np.ndarray, dst: np.ndarray, max_samples: int = 20000) -> dict:
    if len(src) < 100 or len(dst) < 100:
        return {}
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst)))
    step = max(len(src) // max_samples, 1)
    d = []
    for q in src[::step]:
        k, _, d2 = tree.search_knn_vector_3d(q, 1)
        if k > 0:
            d.append(float(np.sqrt(d2[0])) * 1000.0)
    a = np.array(d)
    return {"median": float(np.median(a)), "mean": float(a.mean()),
            "p90": float(np.percentile(a, 90)), "lt10": float((a < 10).mean() * 100.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description="DIY vs RealSense 点云一致性（板坐标系）")
    ap.add_argument("--diy", default="", help="DIY 采集目录（默认取最新的含 cloud.ply 的）")
    ap.add_argument("--db3", default="", help="RealSense 录制文件（默认取 Documents 下最新 .db3）")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "output", "realign", "v3"))
    ap.add_argument("--no-export", action="store_true", help="只算指标，不写 PLY")
    ap.add_argument("--crop", action="store_true",
                    help="额外汇出仅板面区域的 PLY（在 CloudCompare 里量精度要用这两份）")
    args = ap.parse_args()

    diy_dir = args.diy or newest(os.path.join(PROJECT_ROOT, "output", "captures", "*_cloud"), "cloud.ply")
    db3 = args.db3 or newest(os.path.join(os.path.expanduser("~"), "Documents", "*.db3"))
    if not diy_dir or not db3:
        print("[错误] 找不到 DIY 采集或 .db3")
        return 1

    aa = load_aruco()
    board = aa.build_board(5, 7, 36.0, 27.0, "DICT_4X4_1000")
    det = aa.make_detector(board)

    print("=" * 92)
    print("  DIY vs RealSense 点云一致性（统一到 ChArUco 板坐标系）")
    print("=" * 92)
    print(f"  DIY 采集 : {os.path.relpath(diy_dir, PROJECT_ROOT)}")
    print(f"  db3 录制 : {db3}")

    # ---------------------------------------------------------------- DIY
    cal = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "calibration", "stereo_calib.yaml"),
                              encoding="utf-8"))
    P1 = np.asarray(cal["P1"], np.float64)
    K1 = np.array([[P1[0, 0], 0, P1[0, 2]], [0, P1[1, 1], P1[1, 2]], [0, 0, 1]], np.float64)
    img = cv2.imread(os.path.join(diy_dir, "left_raw.png"))
    if img is None:
        print("[错误] DIY 采集缺少 left_raw.png")
        return 1
    res = aa.detect_one(det, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if res["n_corners"] < 6:
        print(f"[错误] DIY 图上板子检出不足（{res['n_corners']} 角点），无法定位板坐标系")
        return 1
    rvec, tvec, rms, _ = aa.pose_from_corners(board, res, K1, np.zeros(5), dist_model="none")
    Td = aa.invert_T(aa.mat4_from_Rt(cv2.Rodrigues(rvec)[0], tvec))
    diy_pcd = o3d.io.read_point_cloud(os.path.join(diy_dir, "cloud.ply"))
    diy_pts = np.asarray(diy_pcd.points)
    diy_col = np.asarray(diy_pcd.colors) if diy_pcd.has_colors() else None
    diy = diy_pts @ Td[:3, :3].T + Td[:3, 3]
    print(f"\n  [DIY] 板子 {res['n_markers']} 标记/{res['n_corners']} 角点, 位姿RMS {rms:.3f} px, "
          f"距离 {np.linalg.norm(tvec)*1000:.1f} mm, 点云 {len(diy)} 点"
          f"{', 带颜色' if diy_col is not None else ', 无颜色'}")

    # ---------------------------------------------------------------- RealSense
    import pyrealsense2 as rs

    cfg = rs.config()
    cfg.enable_device_from_file(db3, repeat_playback=False)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    try:
        # 深度单位必须从传感器读（D405 默认 0.1mm，不是 1mm）
        try:
            depth_scale = float(prof.get_device().first_depth_sensor().get_depth_scale())
        except Exception:  # noqa: BLE001
            depth_scale = 0.001
            print("  [警告] 取不到 depth_scale，按 1mm 处理（结果可能差 10 倍）")
        align = rs.align(rs.stream.color)
        frames = None
        for _ in range(30):
            frames = pipe.wait_for_frames(timeout_ms=10000)
        aligned = align.process(frames)
        d = aligned.get_depth_frame()
        c = aligned.get_color_frame()
        if (not d) or (not c):
            print("[错误] 录制里没有对齐后的深度/彩色帧")
            return 1
        intr = d.profile.as_video_stream_profile().get_intrinsics()
        raw = np.asarray(d.get_data())
        depth = raw.astype(np.float32) * depth_scale
        color = np.asarray(c.get_data())[:, :, :3]
    finally:
        pipe.stop()

    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    m = np.isfinite(depth) & (depth > 0)
    rs_pts = np.column_stack([(uu[m] - intr.ppx) / intr.fx * depth[m],
                              (vv[m] - intr.ppy) / intr.fy * depth[m],
                              depth[m]])
    rs_col = color[m].astype(np.float64) / 255.0
    Kc = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float64)
    res_rs = aa.detect_one(det, cv2.cvtColor(cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
                                             cv2.COLOR_BGR2GRAY))
    if res_rs["n_corners"] < 6:
        print(f"[错误] RealSense 图上板子检出不足（{res_rs['n_corners']} 角点）")
        return 1
    rvec2, tvec2, rms2, _ = aa.pose_from_corners(board, res_rs, Kc, np.asarray(intr.coeffs, np.float64),
                                                 dist_model="inv_brown")
    Tr = aa.invert_T(aa.mat4_from_Rt(cv2.Rodrigues(rvec2)[0], tvec2))
    rs_board = rs_pts @ Tr[:3, :3].T + Tr[:3, 3]
    print(f"  [RS ] 板子 {res_rs['n_markers']} 标记/{res_rs['n_corners']} 角点, 位姿RMS {rms2:.3f} px, "
          f"距离 {np.linalg.norm(tvec2)*1000:.1f} mm, 点云 {len(rs_pts)} 点, "
          f"depth_scale={depth_scale} ({depth_scale*1000:.3f} mm/unit)")

    # ---------------------------------------------------------------- 指标
    print("\n" + "=" * 92)
    print("  量化对比（板坐标系）")
    print("=" * 92)
    zd, nd, rd = fit_plane_z(diy)
    zr, nr, rr = fit_plane_z(rs_board)
    print(f"  板面 z@原点 :  DIY {zd:+7.2f} mm (板内 {nd} 点, 平面残差 {rd:.2f} mm)")
    print(f"                 RS  {zr:+7.2f} mm (板内 {nr} 点, 平面残差 {rr:.2f} mm)")
    if np.isfinite(zd) and np.isfinite(zr):
        print(f"                **相对差 {zd - zr:+.2f} mm "
              f"({(zd - zr) / 350 * 100:+.2f}% @350mm)**")
    bd, br = board_subset(diy), board_subset(rs_board)
    a1 = c2c(bd, br)
    a2 = c2c(br, bd)
    if a1:
        print(f"  板区 C2C DIY->RS: 中位 {a1['median']:.2f} | 均值 {a1['mean']:.2f} | "
              f"P90 {a1['p90']:.2f} mm | <10mm {a1['lt10']:.1f}%")
    if a2:
        print(f"  板区 C2C RS->DIY: 中位 {a2['median']:.2f} | 均值 {a2['mean']:.2f} | "
              f"P90 {a2['p90']:.2f} mm | <10mm {a2['lt10']:.1f}%")

    # ---------------------------------------------------------------- 导出
    if not args.no_export:
        os.makedirs(args.out, exist_ok=True)
        p1 = os.path.join(args.out, "diy_board.ply")
        p2 = os.path.join(args.out, "realsense_board.ply")
        pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(diy))
        if diy_col is not None:
            pc1.colors = o3d.utility.Vector3dVector(diy_col)
        pc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rs_board))
        pc2.colors = o3d.utility.Vector3dVector(rs_col)
        o3d.io.write_point_cloud(p1, pc1, write_ascii=False)
        o3d.io.write_point_cloud(p2, pc2, write_ascii=False)
        print(f"\n  已导出（板坐标系，单位米，带颜色）:")
        print(f"    {os.path.relpath(p1, PROJECT_ROOT)}   {len(diy)} 点")
        print(f"    {os.path.relpath(p2, PROJECT_ROOT)}   {len(rs_board)} 点")

        # 再导一份"只保留板面矩形"的版本。
        # 为什么需要：在 CloudCompare 里对**整片**点云算 C2C 时，
        # 非重叠区域（桌面/剪贴板/背景）根本找不到对应点，会被裁到 max-distance
        # 那一根柱子上，看起来像"差 50mm"，其实是"没有对应点"。
        # 裁到板面矩形后，直方图就直接等于精度本身。
        if args.crop:
            keep_d = board_subset(diy)
            keep_r = board_subset(rs_board)
            pc3 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(keep_d))
            if diy_col is not None:
                pc3.colors = o3d.utility.Vector3dVector(board_subset(diy_col))
            pc4 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(keep_r))
            pc4.colors = o3d.utility.Vector3dVector(board_subset(rs_col))
            p3 = os.path.join(args.out, "diy_board_only.ply")
            p4 = os.path.join(args.out, "realsense_board_only.ply")
            o3d.io.write_point_cloud(p3, pc3, write_ascii=False)
            o3d.io.write_point_cloud(p4, pc4, write_ascii=False)
            print(f"    {os.path.relpath(p3, PROJECT_ROOT)}   {len(keep_d)} 点  <- 仅板面区域")
            print(f"    {os.path.relpath(p4, PROJECT_ROOT)}   {len(keep_r)} 点  <- 仅板面区域")
            print("\n  量精度请用 *_board_only.ply 这两份（直方图 = 精度本身）")
            print("  整片版本适合看覆盖范围差异，不适合量精度")

        print("\n  CloudCompare 里怎么比:")
        print("    1) 两个 PLY 都拖进去（它们已经在同一坐标系，**不要再做对齐/ICP**）")
        print("    2) 各选一个 **... - Cloud** 子节点（不是 .ply 容器节点）")
        print("    3) Tools > Distances > Cloud/Cloud Dist（C2C）：Local modeling = 2D1/2，Max dist 0.05 m")
        print("       注意 C2C 的标量场只加在**先选中**的那片上 -> 想两边都看就交换顺序再跑一次")
        print("    4) 看直方图时把 max distance 收到 0.02 m，避免最后一根无匹配柱子误导")
    return 0


if __name__ == "__main__":
    sys.exit(main())
