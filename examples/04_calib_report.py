"""04_calib_report.py —— 这份标定能不能用？给数字，不给感觉

运行：
    python examples/04_calib_report.py
    python examples/04_calib_report.py --path calibration/stereo_calib.yaml

它做的事：
    1. 读标定文件，打印基线、焦距、RMS、光轴夹角
    2. 按 σz = z²/(f·B)·σd 给出 0.5/1/2/3 m 处的深度不确定度
    3. 反算"要求误差 ≤ 1 cm 时最远能测多远"
    4. 逐条体检，指出哪一项可疑

不依赖相机、不依赖 torch，只读 YAML。可以放进 CI 当门禁。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stereo_sdk.calib import CalibInfo      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="标定文件体检与精度预算")
    ap.add_argument("--path", default="", help="标定 yaml 路径，留空用默认位置")
    ap.add_argument("--disp-sigma", type=float, default=0.5,
                    help="视差噪声（像素），默认 0.5；用你实测的值更准")
    args = ap.parse_args()

    try:
        c = CalibInfo.from_file(args.path)
    except FileNotFoundError as exc:
        print(f"[错误] {exc}")
        return 2

    print(c.summary())
    print()
    print(f"=== 深度不确定度（按视差噪声 {args.disp_sigma} px 估算）===")
    print(f"{'距离':>8} {'σz':>12} {'相对误差':>10}")
    for z in (0.5, 1.0, 2.0, 3.0, 5.0):
        sz = c.sigma_depth(z, args.disp_sigma)
        print(f"{z:6.1f} m {sz * 1000:10.2f} mm {sz / z * 100:9.2f} %")

    print()
    print("=== 可用量程 ===")
    for lim in (0.005, 0.01, 0.02, 0.05):
        print(f"  要求误差 ≤ {lim * 1000:4.0f} mm  ->  最远 {c.usable_range(lim, args.disp_sigma):.2f} m")

    print()
    print("=== 体检 ===")
    bad = 0
    for name, desc, ok in c.check():
        print(f"  [{'OK ' if ok else 'BAD'}] {name:12s} {desc}")
        bad += 0 if ok else 1

    print()
    if bad:
        print(f"有 {bad} 项不合格：这份标定不建议直接用于测量，先回标定 GUI 重标或检查。")
        return 1
    print("全部通过。注意：这只说明文件本身合理，不代表实测精度——")
    print("实测精度要用已知距离反算：把相机对准已知距离的平面，")
    print("用 examples/03_live_view.py 鼠标点选读数作对照；")
    print("行对齐（真正能证明校正成功）看标定 GUI 第③页，判据同 tools/calib_check.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
