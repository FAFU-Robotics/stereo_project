r"""tools/dshow_capture_test.py -- 绕开 OpenCV 的 DirectShow 采集测试（诊断用）
=============================================================================
为什么需要它：
    OpenCV 打不开物理相机时只给一句笼统的话（"backend ... can't be used to
    capture by index"），拿不到真实原因。本工具用 pygrabber（comtypes 封装
    DirectShow）自己建采集图，并把每一步的 COM HRESULT 打出来，
    例如 0x80070005 = E_ACCESSDENIED（Windows 帧服务器拒绝访问）。

它做什么：
    1) 列出 DirectShow 视频输入设备（索引顺序 = OpenCV 的索引顺序）
    2) 对指定索引：绑定过滤器 -> 建图 -> 抓若干帧
    3) 成功则保存一张 PNG（供人眼确认左右拼接布局），失败则打印 HRESULT 与排查建议

依赖（隔离安装，不污染现有环境）：
    python -m pip install --no-deps --target _work\dsgraph pygrabber comtypes
运行前把它加到 PYTHONPATH：
    set PYTHONPATH=<项目>\ _work\dsgraph

用法：
    python tools\dshow_capture_test.py --index 1
    python tools\dshow_capture_test.py --index 1 --seconds 5 --out output\probe\decxin.png
=============================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HRESULT 解释表（只列与本问题相关的常见值）
HRESULT_HINTS = {
    0x80070005: "E_ACCESSDENIED —— 访问被拒绝。物理 UVC 相机由 Windows 帧服务器(FrameServer)代理，"
                "该错误通常表示帧服务器拒绝或未运行、设备被独占、或被隐私/策略拦截。",
    0x8007001F: "ERROR_GEN_FAILURE —— 设备故障（驱动/USB 供电/线材）",
    0x800700AA: "ERROR_BUSY —— 设备忙（被其他程序占用）",
    0x80070002: "ERROR_FILE_NOT_FOUND —— 设备不存在（已拔出？）",
    0x800705AA: "ERROR_NO_SYSTEM_RESOURCES —— 资源不足（带宽/内存）",
    0x80004005: "E_FAIL —— 通用失败（格式协商或引脚连接失败）",
}


def explain(hresult: int) -> str:
    """把 HRESULT 变成可读解释。"""
    key = hresult & 0xFFFFFFFF
    return HRESULT_HINTS.get(key, "（未收录的 HRESULT，可在 MSDN 错误码表查询）")


def main() -> int:
    ap = argparse.ArgumentParser(description="DirectShow 采集测试（诊断用，绕开 OpenCV）")
    ap.add_argument("--index", type=int, default=1, help="设备索引（本机 1 = USB Camera/DECXIN）")
    ap.add_argument("--seconds", type=float, default=5.0, help="抓帧持续秒数")
    ap.add_argument("--out", default="", help="成功时保存的图像路径")
    ap.add_argument("--no-save", action="store_true", help="成功也不保存图片")
    args = ap.parse_args()

    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print("[错误] 找不到 pygrabber。请先隔离安装，并把目录加到 PYTHONPATH：")
        print(r"  python -m pip install --no-deps --target _work\dsgraph pygrabber comtypes")
        print(r"  set PYTHONPATH=%CD%\_work\dsgraph")
        return 10

    graph = FilterGraph()
    names = graph.get_input_devices()
    print("DirectShow 视频输入设备（索引顺序 = OpenCV 的索引顺序）:")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    if not (0 <= args.index < len(names)):
        print(f"[错误] 索引 {args.index} 超出范围")
        return 2
    print(f"\n目标设备: [{args.index}] {names[args.index]}")

    frames: List[np.ndarray] = []

    def on_frame(mat) -> None:
        arr = None
        for attr in ("frame", "data", "array", "img"):
            v = getattr(mat, attr, None)
            if v is not None and hasattr(v, "shape"):
                arr = v
                break
        if arr is None and isinstance(mat, np.ndarray):
            arr = mat
        if arr is not None:
            frames.append(np.asarray(arr).copy())

    print("\n--- 建立采集图 ---")
    try:
        graph.add_video_input_device(args.index)
    except Exception as e:  # noqa: BLE001
        hr = getattr(e, "hresult", None) or (e.args[0] if e.args else None)
        if isinstance(hr, int):
            print(f"  [失败] add_video_input_device  HRESULT=0x{hr & 0xFFFFFFFF:08X}")
            print(f"         {explain(hr)}")
        else:
            print(f"  [失败] add_video_input_device: {e!r}")
        print("\n  排查顺序：")
        print("    1) 帧服务器是否在运行：  sc query FrameServer   （应为 RUNNING）")
        print("       若不是：管理员身份运行  net start FrameServer   或重启电脑")
        print("    2) 是否有 HP 的相机相关进程/驱动：任务管理器看 HP.OMEN.VirtualCamHelper、OmenCap")
        print("    3) 把相机换到机身 USB3 口直插，或拔掉重插（清除设备卡死状态）")
        print("    4) 相机隐私设置：设置 -> 隐私和安全性 -> 相机，两项都要打开")
        return 3

    print("  add_video_input_device OK")
    try:
        graph.add_sample_grabber(on_frame)
        graph.add_null_render()
        graph.prepare_preview_graph()
        print("  prepare_preview_graph OK")
        graph.run()
        print("  run OK")
    except Exception as e:  # noqa: BLE001
        hr = getattr(e, "hresult", None) or (e.args[0] if e.args else None)
        if isinstance(hr, int):
            print(f"  [失败] 建图后半段失败 HRESULT=0x{hr & 0xFFFFFFFF:08X}  {explain(hr)}")
        else:
            print(f"  [失败] {e!r}")
        return 4

    print(f"\n--- 抓帧 {args.seconds} 秒 ---")
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        try:
            graph.grab_frame()
        except Exception as e:  # noqa: BLE001
            print(f"  grab_frame 异常: {e!r}")
            break
        time.sleep(0.12)
    try:
        graph.stop()
    except Exception:
        pass

    print(f"\n共收到 {len(frames)} 帧")
    if not frames:
        print("  [结论] 图建起来了但没有帧：设备在 DirectShow 路径下不出流（格式/带宽/供电）")
        return 5

    f = frames[-1]
    try:
        import cv2
    except ImportError:
        print(f"  最后一帧形状 {f.shape}（未安装 cv2，跳过统计与保存）")
        return 0

    print(f"  最后一帧形状: {f.shape}   dtype={f.dtype}")
    arr = f
    if arr.ndim == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = arr.shape[:2]
        half = w // 2
        L = gray[:, :half]
        R = gray[:, half : half * 2]
        print(f"  灰度 mean={gray.mean():.2f} std={gray.std():.2f}")
        print(f"  左半幅 mean={L.mean():.2f}  右半幅 mean={R.mean():.2f}")
        if gray.mean() < 5 and gray.std() < 12:
            print("  [结论] 有帧但无有效信号（全黑/空缓冲）")
        else:
            print("  [结论] ✅ 拿到真实画面！")
            if not args.no_save:
                out = args.out or os.path.join(PROJECT_ROOT, "output", "probe", f"dshow_idx{args.index}.png")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                cv2.imwrite(out, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                print(f"  已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
