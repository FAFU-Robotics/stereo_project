"""run_calibration.py —— 步骤 2：从采集目录计算标定参数（阶段 5）
=============================================================================
做四件事：
    ① 读取采集目录里的左右图，**严格按规格尺寸**取亚像素角点
    ② 标定：单眼内参 → 双目外参 → 校正矩阵（全部 OpenCV 官方函数）
    ③ **校正验证**：把图像校正后重新检测角点，测量极线失准 |dy|
       （校正正确的标志是同一物点落在左右图的同一行）
    ④ 保存 calibration/stereo_calib.yaml 并打印质量门结论

用法：
    python tools\\run_calibration.py                       REM 用 output/calib 里最新的采集
    python tools\\run_calibration.py --dir output\\calib\\20261013_151253
    python tools\\run_calibration.py --out calibration\\stereo_calib.yaml
    python tools\\run_calibration.py --refix-intrinsic     REM 不固定内参（联合优化）

质量门（双目 RMS）：<0.3 px 优秀 / <0.5 px 良好 / >1.0 px 建议重拍
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from calibration import (BoardSpec, StereoCalibration, calibrate_stereo,   # noqa: E402
                         epipolar_error, load_frame_points)


def newest_capture_dir(root: str) -> str:
    if not os.path.isdir(root):
        return ""
    subs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]
    if not subs:
        return ""
    return max(subs, key=os.path.getmtime)


def image_size_of(frames_dir: str) -> tuple:
    """读采集目录里第一张左图的真实尺寸 (W, H)。

    原先这里写死 (640, 480)：换分辨率重标时会用错 image_size，
    标定能"成功"但结果与实际图像不匹配，且不报错。
    """
    import glob
    import cv2
    for p in sorted(glob.glob(os.path.join(frames_dir, "left_*.png"))):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return (int(img.shape[1]), int(img.shape[0]))
    return (640, 480)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="双目标定计算（阶段 5 步骤 2）")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    ap.add_argument("--dir", default="", help="采集目录（默认取 output/calib 下最新）")
    ap.add_argument("--out", default="", help="输出标定文件（默认 calibration/stereo_calib.yaml）")
    ap.add_argument("--refix-intrinsic", action="store_true",
                    help="不在双目标定中固定内参（联合优化；默认固定）")
    ap.add_argument("--no-verify", action="store_true", help="跳过校正验证（不推荐）")
    args = ap.parse_args(argv)

    # ---- 配置
    import yaml
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cal_cfg = dict(cfg.get("calibration", {}) or {})
    spec = BoardSpec(cols=int(cal_cfg.get("board_cols", 12)),
                     rows=int(cal_cfg.get("board_rows", 9)),
                     square_mm=float(cal_cfg.get("square_mm", 3.0)))

    cap_root = os.path.join(PROJECT_ROOT, "output", "calib")
    frames_dir = args.dir or newest_capture_dir(cap_root)
    if not frames_dir or not os.path.isdir(frames_dir):
        print(f"[错误] 找不到采集目录。请先执行采集（菜单 [12]）或指定 --dir")
        return 1
    out_path = args.out or os.path.join(PROJECT_ROOT, "calibration", "stereo_calib.yaml")

    print("=" * 88)
    print("  阶段 5 · 步骤 2：双目标定计算")
    print("=" * 88)
    print(f"  采集目录 : {frames_dir}")
    print(f"  标定板   : {spec.describe()}")
    print(f"  输出     : {out_path}")
    print(f"  内参策略 : {'固定（各眼先单独标定）' if not args.refix_intrinsic else '联合优化'}")
    print("-" * 88)

    # ---- ① 角点
    t0 = time.perf_counter()
    left_pts, right_pts, rows = load_frame_points(frames_dir, spec, refine=True)
    both = sorted(set(left_pts) & set(right_pts))
    print(f"  ① 角点（严格 {spec.pattern_size[0]}x{spec.pattern_size[1]}）："
          f"左 {len(left_pts)} 帧 / 右 {len(right_pts)} 帧 / 双眼 {len(both)} 帧"
          f"（耗时 {time.perf_counter()-t0:.1f}s）")
    single = [r["idx"] for r in rows if (r["left"] != r["right"])]
    if single:
        print(f"     仅单眼有效（只用于该眼内参）：{single}")
    if len(both) < 10:
        print(f"     ⚠️ 双眼有效帧只有 {len(both)} 帧，建议 ≥15（结果可能不稳）")

    # ---- ②③ 标定
    t0 = time.perf_counter()
    img_size = image_size_of(frames_dir)
    print(f"  图像尺寸（从实际图片读出）：{img_size[0]}x{img_size[1]}")
    try:
        calib = calibrate_stereo(left_pts, right_pts, spec, img_size,
                                 fix_intrinsic=not args.refix_intrinsic,
                                 extra_meta={"capture_dir": os.path.basename(frames_dir)})
    except Exception as exc:  # noqa: BLE001
        print(f"[标定失败] {exc}")
        return 2
    print(f"  ②③ 标定完成（耗时 {time.perf_counter()-t0:.1f}s）")
    print(calib.summary())

    # ---- 质量门
    rms = calib.rms_stereo
    verdict = ("✅ 优秀（<0.3 px）" if rms < 0.3 else
               "✅ 良好（<0.5 px）" if rms < 0.5 else
               "⚠️ 可用但偏大（<1.0 px）" if rms < 1.0 else "❌ 建议重拍（≥1.0 px）")
    print(f"  质量门   : 双目 RMS = {rms:.4f} px  -> {verdict}")

    # 最差帧
    per_frame = calib.meta.get("per_frame_errors", [])
    if per_frame:
        worst = ", ".join(f"#{p['idx']}({p['err_mean_px']:.2f}px)" for p in per_frame[:5])
        print(f"  误差最大的帧: {worst}")
        print(f"     （如果某帧明显偏大，可以删掉它的图后重跑本次计算）")

    # ---- ③ 校正验证
    if not args.no_verify:
        t0 = time.perf_counter()
        print("-" * 88)
        print("  ③ 校正验证：校正后重新检测角点，测量极线失准 |dy|")
        # 校正前：直接用同一批帧测（未校正）
        from calibration import detect_corners_strict  # 局部导入，避免顶部太杂
        import cv2

        def pre_dy(i: int):
            L = cv2.imread(os.path.join(frames_dir, f"left_{i:02d}.png"))
            R = cv2.imread(os.path.join(frames_dir, f"right_{i:02d}.png"))
            if L is None or R is None:
                return None
            cl = detect_corners_strict(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), spec.pattern_size)
            cr = detect_corners_strict(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), spec.pattern_size)
            if cl is None or cr is None:
                return None
            return float(np.abs(cl.reshape(-1, 2)[:, 1] - cr.reshape(-1, 2)[:, 1]).mean())

        pre = [d for d in (pre_dy(i) for i in both) if d is not None]
        post_rows = epipolar_error(calib, frames_dir, spec, frames=both)
        post = [r for r in post_rows if r.get("found")]
        if pre:
            print(f"     校正前 |dy|: 平均 {np.mean(pre):.2f} px  最大 {np.max(pre):.2f} px")
        if post:
            dy = np.array([r["dy_mean_px"] for r in post])
            dymax = np.array([r["dy_max_px"] for r in post])
            dx = np.array([r["dx_mean_px"] for r in post])
            print(f"     校正后 |dy|: 平均 {dy.mean():.2f} px  最大 {dymax.max():.2f} px   "
                  f"（{len(post)}/{len(both)} 帧成功复检）")
            print(f"     校正后视差: 平均 {dx.mean():.1f} px  范围 {dx.min():.1f} ~ {dx.max():.1f}")
            ok = dy.mean() < 0.5
            print(f"     判定: {'✅ 极线已对齐（<0.5 px）' if ok else '⚠️ 仍有失配，需检查'}"
                  f"   耗时 {time.perf_counter()-t0:.1f}s")
            if not ok:
                print("       可能原因：个别帧角点误差大 / 采集时板子移动（运动模糊）/ 内参不稳")
        else:
            print("     ⚠️ 校正后没有一帧能重新检出角点（检查图像与映射）")

    # ---- ④ 保存
    calib.save(out_path)
    print("-" * 88)
    print(f"  ④ 已保存标定文件：{out_path}")
    print()
    print("  下一步（阶段 5 步骤 4：接入运行时）:")
    print("    1) config.yaml 里把 depth.calibrated 改成 true（calibration_file 已指好路径）")
    print("    2) 运行时启用 depth.rectify_before_infer: true（先用映射表校正再送 FFS）")
    print("    3) 用已知距离验证：把板/尺子放在 0.5 / 1.0 / 1.5 m，看深度读数误差（目标 <5%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
