"""03_live_view.py —— 实时窗口：左图 + 深度伪彩色 + 鼠标点选测距

运行：
    python examples/03_live_view.py --mock
    python examples/03_live_view.py                      # 真机

按键：
    q / ESC  退出
    鼠标左键  在窗口任意位置点一下，读出该像素深度（打印到控制台并在图上标注）
    s        存一份当前结果（export_result）
    +/-      调整深度显示上限（zmax）
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2                                                     # noqa: E402

from _common import build_parser, colorize_depth, open_camera    # noqa: E402

WIN = "stereo depth (q=quit, click=distance, s=save)"
state = {"frame": None, "result": None, "zmax": 3.0, "marks": []}


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN or state["result"] is None:
        return
    r = state["result"]
    h, w = r.depth.shape[:2]
    # 窗口里左右两幅并排，点右半边要换算成深度图的 x
    vw = param
    u = x if x < w else x - (vw // 2)
    z = r.depth_at(min(max(u, 0), w - 1), min(max(y, 0), h - 1))
    state["marks"].append((u, min(max(y, 0), h - 1), z))
    print(f"像素 ({u},{y}) 深度 = {z:.4f} m" if z > 0 else f"像素 ({u},{y}) 无效")


def main() -> int:
    args = build_parser("实时显示深度与点选测距").parse_args()
    s = open_camera(args)
    print("窗口已打开；q 退出，鼠标左键测距，s 存盘，+/- 调深度上限。")
    try:
        while True:
            if s.has_new_result():
                r = s.take_result()
                if r is not None:
                    state["result"] = r
                    state["marks"] = []

            r = state["result"]
            if r is not None:
                left = r.frame.left if r.frame is not None else None
                dep = colorize_depth(r.depth, r.valid, 0.2, state["zmax"])
                if left is not None and left.shape[:2] != dep.shape[:2]:
                    dep = cv2.resize(dep, (left.shape[1], left.shape[0]))
                for (u, v, z) in state["marks"]:
                    cv2.circle(dep, (u, v), 6, (255, 255, 255), 2)
                    if z > 0:
                        cv2.putText(dep, f"{z:.2f}m", (u + 8, v - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if left is not None:
                    canvas = cv2.hconcat([left, dep])
                else:
                    canvas = dep
                cv2.putText(canvas, f"seq={r.seq} preset={r.preset} "
                                    f"infer={r.timings.get('infer', 0):.0f}ms "
                                    f"zmax={state['zmax']:.1f}m",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(WIN, canvas)
                cv2.setMouseCallback(WIN, on_mouse, canvas.shape[1])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and state["result"] is not None:
                from stereo_sdk.export import export_result
                files = export_result(state["result"], os.path.join("output", "example03"),
                                      prefix=f"seq{state['result'].seq:03d}")
                print(f"已存 {len(files)} 个文件到 output/example03")
            if key in (ord("+"), ord("=")):
                state["zmax"] = min(state["zmax"] + 0.5, 20.0)
            if key == ord("-"):
                state["zmax"] = max(state["zmax"] - 0.5, 0.3)
    finally:
        s.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
