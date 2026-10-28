"""01_read_depth.py —— 最小可用：拿一帧、读一个像素的深度、存一套文件

运行：
    python examples/01_read_depth.py --mock
    python examples/01_read_depth.py                 # 真机，需要相机 + torch

它回答三个最常被问的问题：
    1. 深度值的单位是什么           -> 米（float32），0 表示无效
    2. 怎么取某个像素的距离         -> DepthResult.depth_at(u, v)
    3. 怎么把结果存下来给别人看     -> export_result()
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import build_parser, open_camera, wait_result     # noqa: E402


def main() -> int:
    args = build_parser("读一帧深度并导出").parse_args()
    s = open_camera(args)
    try:
        print(f"已启动（mock={args.mock}），等第一帧…")
        r = wait_result(s, args.timeout)

        print()
        print("=== 一行摘要 ===")
        print(r.summary())

        print()
        print("=== 关键形状与单位 ===")
        print(f"视差 disparity : {r.disparity.shape}  float32 像素（模型分辨率）")
        print(f"深度 depth     : {r.depth.shape}  float32 **米**（已上采样到全分辨率）")
        print(f"有效掩码 valid : {r.valid.shape}  bool，{r.valid.mean() * 100:.1f}% 有效")
        print(f"点云 points    : {r.points.shape}  float32 米，y 向下")

        h, w = r.depth.shape[:2]
        u, v = w // 2, h // 2
        z = r.depth_at(u, v)
        print()
        print(f"=== 单点测距 ===")
        print(f"中心像素 ({u},{v}) 深度 = {z:.4f} m" if z > 0
              else f"中心像素 ({u},{v}) 无效（弱纹理或超量程）")

        out = os.path.join("output", "example01")
        from stereo_sdk.export import export_result
        files = export_result(r, out, prefix=f"seq{r.seq:03d}")
        print()
        print(f"=== 已导出到 {out} ===")
        for k, p in sorted(files.items()):
            print(f"  {k:14s} {os.path.getsize(p):>9,d} B  {os.path.basename(p)}")
        print()
        print("提示：cloud.ply 是二进制 PLY，可直接用 MeshLab / Open3D 打开。")
    finally:
        s.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
