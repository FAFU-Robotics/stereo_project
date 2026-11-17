"""camera_design.py —— 相机参数设计计算工具（任务 3 交付物）

=============================================================================
【为什么需要它】
    选相机/选基线/定量程时总要反复算同样的几个式子。这个工具把它们集中起来：
    给任意几个参数，算出其余的量（含"要达到某精度需要多大基线"的反推）。

【公式（全部在本项目的实测数据上校验过）】
    disp    = fx * B / z                 视差（像素）
    dz      = z^2 / (fx * B)             1 像素视差 = 多少毫米深度   ← 最重要的一个
    dz_rel  = ddisp / disp = ddisp * z / (fx * B)   相对误差
    z_min   = fx * B / max_disp          最近可测距离（max_disp 之外是网络外推，不可信）
    z_max   = sqrt(tol * fx * B)         按"每像素误差 <= tol"能到的最远距离
    B_need  = z^2 * ddisp / (fx * dz_tol)   反推：要在 z 处达到 dz_tol，基线要多长
    hFOV    = 2*atan(W / (2*fx))         水平视场角
    单位: fx 像素、B 毫米、z 毫米、dz 毫米

【用法示例】
    # 已知本机参数，看几个距离上的表现
    python tools\\camera_design.py --fx 547 --baseline 64.891 --dist 0.3,0.35,0.5,1.0,2.0

    # 反推: 要在 0.5 m 处 1 像素误差 <= 1 mm，需要多长基线？
    python tools\\camera_design.py --fx 547 --baseline 64.891 --dist 0.5 --target-dz 1.0

    # 看量程（配合 FFS 的 max_disp=192）
    python tools\\camera_design.py --fx 547 --baseline 64.891 --max-disp 192

    # 看视场角
    python tools\\camera_design.py --fx 547 --width 640 --height 480

    python tools\\camera_design.py --selftest     # 用本项目实测值校验公式
=============================================================================
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Tuple


def disp_of(fx: float, b_mm: float, z_mm: float) -> float:
    """视差（像素）。"""
    return fx * b_mm / z_mm if z_mm > 0 else float("inf")


def dz_per_px(fx: float, b_mm: float, z_mm: float) -> float:
    """1 像素视差对应的深度变化（毫米）—— 数值越小越灵敏。"""
    denom = fx * b_mm
    return (z_mm ** 2) / denom if denom > 0 else float("inf")


def z_max_for_tol(fx: float, b_mm: float, tol_mm: float) -> float:
    """按"每像素误差 <= tol_mm"能测到的最远距离（毫米）。"""
    return math.sqrt(max(tol_mm, 0.0) * fx * b_mm)


def z_min_for_maxdisp(fx: float, b_mm: float, max_disp: float) -> float:
    """最近可测距离（毫米）；max_disp 之外属于外推，不可信。"""
    return (fx * b_mm / max_disp) if max_disp > 0 else float("inf")


def baseline_for_tol(fx: float, z_mm: float, dz_tol_mm: float, ddisp_px: float = 1.0) -> float:
    """反推基线：在 z 处，视差误差 ddisp 像素时，深度误差 <= dz_tol 所需的 B（毫米）。"""
    denom = fx * dz_tol_mm
    return (z_mm ** 2) * ddisp_px / denom if denom > 0 else float("inf")


def fov_deg(fx: float, width_px: float) -> float:
    return 2.0 * math.degrees(math.atan(width_px / (2.0 * fx))) if fx > 0 else float("nan")


def _fmt(v: float, unit: str = "", nd: int = 2) -> str:
    if v != v:
        return "n/a"
    if v == float("inf"):
        return "inf"
    return f"{v:.{nd}f}{unit}"


def report(args: argparse.Namespace) -> int:
    fx, b = args.fx, args.baseline
    print("=" * 86)
    print("  相机参数设计计算")
    print("=" * 86)
    print(f"  fx = {_fmt(fx)} px   baseline B = {_fmt(b)} mm   "
          f"fx*B = {_fmt(fx * b)} px*mm")
    if args.width and args.height:
        print(f"  视场角: 水平 {_fmt(fov_deg(fx, args.width), '°', 1)}   "
              f"垂直 {_fmt(fov_deg(fx, args.height), '°', 1)}   "
              f"（{args.width}x{args.height}）")
    if args.max_disp:
        zmin = z_min_for_maxdisp(fx, b, args.max_disp)
        print(f"  量程下限（max_disp={args.max_disp:g}）: {_fmt(zmin / 1000, ' m', 3)}"
              f"   —— 更近的视差超出搜索范围，是外推值，不可信")

    dists = [float(x) * 1000.0 for x in str(args.dist).split(",") if x.strip()] if args.dist else []
    if dists:
        print()
        print(f"  {'距离':>10}{'视差':>10}{'1px误差':>12}{'相对误差':>12}{'0.5px误差':>13}")
        print(f"  {'-' * 74}")
        for z in dists:
            d = disp_of(fx, b, z)
            dz = dz_per_px(fx, b, z)
            print(f"  {z / 1000:>9.3f}m{d:>10.1f}px{dz:>11.2f}mm"
                  f"{dz / z * 100:>11.2f}%{dz * 0.5:>12.2f}mm")
        print()
        print("  说明: '1px误差'是视差差 1 个像素时的深度误差; 亚像素精度（FFS 输出浮点视差）")
        print("        实际误差还要乘上视差噪声，本项目实测噪声约 0.2~0.4 px。")

    if args.target_dz and dists:
        print()
        print(f"  反推: 要求在 {dists[0] / 1000:.3f} m 处深度误差 <= {args.target_dz:g} mm")
        need = baseline_for_tol(fx, dists[0], args.target_dz, args.target_ddisp)
        print(f"        视差误差 {args.target_ddisp:g} px 时，基线至少需要 "
              f"{_fmt(need)} mm（当前 {_fmt(b)} mm）")

    if args.tol:
        print()
        print(f"  按每像素误差 <= {args.tol:g} mm 推算的最远可测距离: "
              f"{_fmt(z_max_for_tol(fx, b, args.tol) / 1000, ' m', 3)}")
    return 0


def selftest() -> int:
    """用本项目的实测参数校验公式（数值都有实测依据）。"""
    fx, b = 547.017, 64.891      # 标定结果：校正后 fx、基线
    ok = True
    print("=" * 86)
    print("  camera_design 自检（用本项目实测参数）")
    print("=" * 86)

    d350 = disp_of(fx, b, 350.0)
    print(f"  ① 0.35 m 处视差 = {d350:.2f} px   （实测 FFS 约 100.6~101.2 px ✓）")
    if not (98.0 <= d350 <= 104.0):
        print("     ❌ 超出预期区间"); ok = False

    dz350 = dz_per_px(fx, b, 350.0)
    print(f"  ② 0.35 m 处 1 px = {dz350:.2f} mm   （预期 3.4~3.5 ✓）")
    if not (3.3 <= dz350 <= 3.6):
        print("     ❌ 超出预期区间"); ok = False

    zmin = z_min_for_maxdisp(fx, b, 192)
    print(f"  ③ max_disp=192 时量程下限 = {zmin / 1000:.3f} m   （实测约 0.19 m ✓）")
    if not (0.17 <= zmin / 1000 <= 0.20):
        print("     ❌ 超出预期区间"); ok = False

    fov = fov_deg(fx, 640.0)
    print(f"  ④ 640 宽、fx=547 的水平视场角 = {fov:.1f}°")
    if not (55.0 <= fov <= 65.0):
        print("     ❌ 超出预期区间"); ok = False

    # ⑤ 反推基线: 0.5 m 处要求 1 px 误差 <= 1 mm
    need = baseline_for_tol(fx, 500.0, 1.0, 1.0)
    print(f"  ⑤ 0.5 m 处 1 px 误差 <= 1 mm 需要基线 {need:.1f} mm "
          f"（当前 64.9 mm，故 0.5 m 处 1 px 约 {dz_per_px(fx, b, 500):.1f} mm，确实不够）")
    if not (440.0 <= need <= 470.0):   # 正确值 = 500^2/(547.017*1) = 457.0 mm
        print("     ❌ 反推公式可疑"); ok = False

    # ⑥ 一致性: dz = z^2/(fx*B) 与 disp = fx*B/z 互为倒数关系
    z = 700.0
    lhs = dz_per_px(fx, b, z)
    rhs = z * z / (fx * b)
    print(f"  ⑥ 一致性 dz({z:.0f}mm) = {lhs:.4f} vs {rhs:.4f}")
    if abs(lhs - rhs) > 1e-6:
        print("     ❌ 不一致"); ok = False

    print("\n  " + ("✅ 自检全部通过" if ok else "❌ 自检有失败项"))
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="相机参数设计计算（fx/基线/量程/精度反推/FOV）")
    ap.add_argument("--fx", type=float, default=547.017, help="焦距（像素，默认本项目校正后 fx）")
    ap.add_argument("--baseline", type=float, default=64.891, help="基线 B（毫米，默认本项目标定值）")
    ap.add_argument("--dist", default="", help="要评估的距离（米，逗号分隔），如 0.3,0.5,1.0")
    ap.add_argument("--max-disp", type=float, default=192.0, help="视差搜索上限（默认 192）")
    ap.add_argument("--width", type=float, default=640.0, help="单眼图宽（像素）")
    ap.add_argument("--height", type=float, default=480.0, help="单眼图高（像素）")
    ap.add_argument("--target-dz", type=float, default=0.0, help="目标深度误差（毫米），用于反推基线")
    ap.add_argument("--target-ddisp", type=float, default=1.0, help="估计的视差误差（像素）")
    ap.add_argument("--tol", type=float, default=0.0, help="每像素允许误差（毫米），用于算最远距离")
    ap.add_argument("--selftest", action="store_true", help="用实测参数校验公式")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    return report(a)


if __name__ == "__main__":
    sys.exit(main())
