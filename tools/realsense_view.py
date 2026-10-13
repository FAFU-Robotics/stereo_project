"""realsense_view.py —— Intel RealSense 深度相机实时预览（深度图 + 彩色点云）
=============================================================================
【这是什么】
    一个**独立的新工具**，用 Intel RealSense SDK（pyrealsense2）驱动 D405 等 D400 系列相机：
        - 实时显示：彩色图 | 深度伪彩（带色标）
        - 实时显示：**彩色点云**（Open3D 窗口）
        - 按键保存：PLY 点云 + 深度 npy/png + 彩色图 + 参数说明
        - `--check`：只枚举设备并打印**出厂内参/基线/可用流**，不启动视频流

【与项目其它部分的关系】
    - **不修改任何现有文件**（FFS 目录 third_party/Fast-FoundationStereo 完全不碰）
    - 复用现有模块做"显示"与"点云导出"：viewer.StereoViewer、pointcloud.PointCloudBuilder
    - RealSense 是**出厂标定**的深度相机：深度本身带真实尺度（毫米），
      所以这里的点云标记为 CALIBRATED（与 DECXIN 那条"需要自己标定"的路线不同）

【重要：必须在【管理员终端】里运行】
    本机实测：普通权限进程访问物理相机会被 Windows 拒绝。
    另外 RealSense SDK 需要创建命名互斥体做跨进程设备锁，
    在受限环境（含 DSH 沙箱）里会报 "CreateNamedMutex returned Mutex_TotalFailure"。
    => 双击 tools\\RealSense预览_管理员.bat（会自动提权），或在管理员 cmd 里运行本脚本。

【坐标系与显示约定（与项目一致）】
    点云按 RealSense/OpenCV 相机坐标系：x 右、y 下、z 前，单位米。
    保存的 PLY 就是这个约定；**只有 3D 窗口**里把 y 翻成"向上"以便观看（窗口标题已注明）。

用法：
    python tools\\realsense_view.py --check                    REM 只查设备与参数（秒回）
    python tools\\realsense_view.py                            REM 实时预览（彩色|深度 + 点云）
    python tools\\realsense_view.py --depth-res 640x480 --color-res 640x480 --fps 30
    python tools\\realsense_view.py --max-points 120000 --voxel 0.003
    python tools\\realsense_view.py --depth-only               REM 只用深度（不取彩色）
    python tools\\realsense_view.py --no-cloud-window          REM 不开 3D 窗口（只看 2D）

窗口内按键：
    q / ESC  退出      s  保存一帧（PLY + 深度 + 彩色 + 说明）      p  暂停/继续
    h  帮助            o  重新打开 3D 点云窗口（若被关掉）
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 复用项目已有模块（**只读使用，不修改**）
from depth_utils import CameraParams                                   # noqa: E402
from pointcloud import PointCloudBuilder                               # noqa: E402
from viewer import StereoViewer                                        # noqa: E402

try:
    import pyrealsense2 as rs
except ImportError:
    _msg = [
        "[错误] 没有安装 pyrealsense2（或用了别的 Python 解释器）。",
        f"       当前解释器: {sys.executable}",
        '       安装（用项目自带的 Python）:',
        '       "D:\\31931\\Documents\\python312\\python.exe" -m pip install pyrealsense2',
        "       若 pip 因权限/临时目录失败，可改成：下载 wheel 后用 zip 解压到 site-packages（见对话记录）。",
    ]
    for _l in _msg:
        print(_l)
    # 这种失败也会让窗口"一闪就没"，所以同样落一份日志
    try:
        _lp = os.path.join(PROJECT_ROOT, "output", "realsense", "last_run.log")
        os.makedirs(os.path.dirname(_lp), exist_ok=True)
        with open(_lp, "w", encoding="utf-8") as _fh:
            _fh.write("\n".join(_msg) + f"\nPython: {sys.version}\n")
        print(f"[日志] 已写入: {_lp}")
    except Exception:
        pass
    sys.exit(5)


# -----------------------------------------------------------------------------
def parse_res(s: str) -> Tuple[int, int]:
    w, h = (int(x) for x in str(s).lower().split("x"))
    return w, h


def list_devices() -> List[Any]:
    """枚举 RealSense 设备。注意：在受限环境里读取设备信息可能抛异常（命名互斥体被拒）。"""
    ctx = rs.context()
    devs = list(ctx.query_devices())
    return devs


def describe_device(dev) -> Dict[str, str]:
    """读取设备基本信息；读不到就留空（不抛异常，便于在受限环境下也能跑）。"""
    out: Dict[str, str] = {}
    for key in ("name", "serial_number", "firmware_version", "usb_type_descriptor",
                "product_line", "physical_port"):
        try:
            out[key] = str(dev.get_info(getattr(rs.camera_info, key)))
        except Exception:
            out[key] = "?"
    return out


def factory_params(profile) -> Tuple[CameraParams, Dict[str, Any]]:
    """从流配置里取**出厂标定**的内参与（可得的）基线，构造 CameraParams。

    RealSense 的深度是出厂标定过的（真实尺度），所以这里标 calibrated=True。
    注意：D405 的深度由左右两颗红外成像器算出，而彩色也来自左目，
    因此"深度→彩色"的外参平移几乎为 0 —— **这不是立体基线**，
    也不参与点云计算（深度本身已是米制）。这里只把它记录下来，读不到就置 0。
    """
    intr = profile.get_intrinsics()
    baseline_m = 0.0
    try:
        dev = profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        color_sensor = None
        for s in dev.sensors:
            try:
                if "color" in str(s.get_info(rs.camera_info.name)).lower() or \
                   "rgb" in str(s.get_info(rs.camera_info.name)).lower():
                    color_sensor = s
                    break
            except Exception:
                continue
        if color_sensor is not None:
            ext = depth_sensor.get_extrinsics_to(color_sensor)
            baseline_m = float(np.linalg.norm(np.asarray(ext.translation, dtype=np.float64)))
    except Exception:
        baseline_m = 0.0
    params = CameraParams(
        fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.ppx), cy=float(intr.ppy),
        # 仅供 CameraParams 完整；点云不用它（深度已是真实尺度）
        baseline_m=baseline_m if baseline_m > 1e-6 else 0.001,
        calibrated=True,
        source=f"RealSense 出厂标定（{intr.width}x{intr.height}，畸变模型 {intr.model}）",
        width=int(intr.width), height=int(intr.height),
    )
    meta = {"model": str(intr.model), "coeffs": list(intr.coeffs),
            "depth_to_color_m": baseline_m}
    return params, meta


def is_admin() -> bool:
    """当前进程是否以管理员权限运行。

    为什么要打印它：本机实测普通权限进程打不开物理相机；而且 RealSense SDK 还需要
    创建命名互斥体。用户如果直接双击 .py（或用错 Python）就会"窗口一闪就没了"，
    启动时把这一行打出来能立刻暴露这类问题。
    """
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def print_check() -> int:
    """--check：枚举设备 + 打印可用流 + 出厂内参。不启动视频流。"""
    print("=" * 88)
    print("  RealSense 设备检查（只查询，不启动视频流）")
    print("=" * 88)
    try:
        devs = list_devices()
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 枚举失败：{type(exc).__name__}: {exc}")
        print("     常见原因：① 没有以管理员运行（本机物理相机需要管理员权限）")
        print("               ② 在受限环境（沙箱）里 SDK 无法创建命名互斥体")
        return 2
    print(f"  检测到设备数: {len(devs)}")
    if not devs:
        print("  ❌ 没有检测到 RealSense 设备。请确认：")
        print("     - USB 线插好（D405 需要 USB 3.x 口）")
        print("     - 没有被 RealSense Viewer 等其它程序占用")
        return 2

    for i, d in enumerate(devs):
        info = describe_device(d)
        print(f"\n  [设备 {i}] {info.get('name')}  SN={info.get('serial_number')}")
        print(f"            固件 {info.get('firmware_version')}   USB {info.get('usb_type_descriptor')}"
              f"   产品线 {info.get('product_line')}")
        for s in d.sensors:
            try:
                name = s.get_info(rs.camera_info.name)
            except Exception:
                name = "?"
            print(f"      传感器: {name}")
            seen = {}
            for p in s.get_stream_profiles():
                vsp = p.as_video_stream_profile()
                key = (str(p.stream_type()), str(p.format()))
                seen.setdefault(key, []).append((vsp.width(), vsp.height(), p.fps()))
            for (stype, fmt), lst in sorted(seen.items()):
                uniq = sorted({(w, h, f) for w, h, f in lst})
                brief = ", ".join(f"{w}x{h}@{f}" for w, h, f in uniq[:6])
                more = "" if len(uniq) <= 6 else f" …共 {len(uniq)} 种"
                print(f"          {stype:<12} {fmt:<6} {brief}{more}")
            if str(s.get_info(rs.camera_info.name)).lower().find("depth") >= 0:
                try:
                    print(f"          深度单位: {s.get_depth_scale()} 米/单位")
                except Exception:
                    pass
    print("\n  提示：以上内参/基线在实时预览启动时会自动读取并打印。")
    return 0


# -----------------------------------------------------------------------------
def run_view(a: argparse.Namespace) -> int:
    dw, dh = parse_res(a.depth_res)
    cw, ch = parse_res(a.color_res)

    # ---- 配置数据流
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, a.fps)
    if not a.depth_only:
        cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, a.fps)

    print("=" * 88)
    print("  RealSense 实时预览（深度 + 彩色点云）")
    print("=" * 88)
    print(f"  深度流: {dw}x{dh}@{a.fps}   彩色流: {'关闭' if a.depth_only else f'{cw}x{ch}@{a.fps}'}")

    try:
        profile = pipe.start(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 启动失败：{type(exc).__name__}: {exc}")
        print("     排查顺序：")
        print("       1) 是否以【管理员】运行？（本机物理相机需要管理员权限）")
        print("       2) 相机是否被别的程序占用（RealSense Viewer / 相机 App / 浏览器）？")
        print("       3) USB 是否插在 USB 3.x 口？换线/换口再试")
        print("       4) 分辨率/帧率组合是否被支持？跑 --check 看可用流清单")
        return 2

    depth_scale = 1.0
    params_color: Optional[CameraParams] = None
    try:
        dev = profile.get_device()
        ds = dev.first_depth_sensor()
        depth_scale = float(ds.get_depth_scale())
        print(f"  设备: {describe_device(dev).get('name')}  "
              f"固件 {describe_device(dev).get('firmware_version')}  "
              f"USB {describe_device(dev).get('usb_type_descriptor')}")
        print(f"  深度单位: {depth_scale} 米/单位（即 {depth_scale*1000:.1f} mm）")
    except Exception as exc:  # noqa: BLE001
        print(f"  [提示] 读取设备信息失败（{type(exc).__name__}），继续尝试取流")

    align = None if a.depth_only else rs.align(rs.stream.color)
    out_root = os.path.join(PROJECT_ROOT, "output", "realsense")

    viewer = StereoViewer(window_name="RealSense  D405", display_scale=float(a.display_scale or 1.0))
    builder = PointCloudBuilder(
        {"zmin_m": a.zmin, "zmax_m": a.zmax, "voxel_size_m": a.voxel,
         "use_color": not a.depth_only, "max_points": a.max_points},
        calibrated=True)

    vis = disp_pcd = None
    vis_added = False          # 【必须放在循环外】记录几何体是否已加入窗口
    _flip_y = np.array([1.0, -1.0, 1.0])
    last: Dict[str, Any] = {}
    paused = False
    saved = 0
    fps_t0 = time.perf_counter()
    fps_n = 0
    fps = 0.0

    def open_cloud_window():
        v = o3d.visualization.Visualizer()
        v.create_window(window_name="RealSense point cloud  (display: Y up | file: Y down)",
                        width=int(a.cloud_window), height=int(a.cloud_window))
        ro = v.get_render_option()
        ro.point_size = float(a.point_size)
        ro.background_color = np.array([0.08, 0.08, 0.10])
        return v

    try:
        import open3d as o3d
    except ImportError:
        print("[错误] 没有安装 open3d，无法显示点云（可加 --no-cloud-window 只看 2D）")
        try:
            pipe.stop()
        except Exception:
            pass
        return 5
    disp_pcd = o3d.geometry.PointCloud()
    if not a.no_cloud_window:
        vis = open_cloud_window()
        print("  3D 点云窗口已打开（关掉它不会结束预览；按 o 可重开，按 q 退出）")
    print("  按键: q 退出  s 保存  p 暂停  h 帮助  o 重开 3D 窗口")
    print("-" * 88)

    try:
        while True:
            if not paused:
                try:
                    frames = pipe.wait_for_frames(timeout_ms=5000)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [等待帧失败] {type(exc).__name__}: {exc}")
                    break
                if align is not None:
                    frames = align.process(frames)
                dframe = frames.get_depth_frame()
                cframe = None if a.depth_only else frames.get_color_frame()
                if not dframe:
                    continue

                depth_mm = np.asanyarray(dframe.get_data())          # uint16，单位 mm
                depth_m = depth_mm.astype(np.float32) * depth_scale   # -> 米
                color_bgr = None
                if cframe is not None:
                    color_bgr = np.asanyarray(cframe.get_data())

                # 内参：用**彩色**内参（因为 depth 已 align 到彩色坐标系）
                if params_color is None:
                    prof = (cframe.get_profile() if cframe is not None else dframe.get_profile())
                    params_color, imeta = factory_params(prof)
                    print(f"  内参({params_color.width}x{params_color.height}): "
                          f"fx={params_color.fx:.2f} fy={params_color.fy:.2f} "
                          f"cx={params_color.cx:.2f} cy={params_color.cy:.2f}")
                    print(f"  畸变模型 {imeta['model']}  系数 "
                          f"{['%.4g' % c for c in imeta['coeffs']]}")
                    print(f"  深度→彩色 平移量 {imeta['depth_to_color_m']*1000:.2f} mm"
                          f"（D405 彩色与深度共目，接近 0 属正常；点云不用它）")

                # 深度范围过滤 + 点云
                valid = np.isfinite(depth_m) & (depth_m >= a.zmin) & (depth_m <= a.zmax)
                depth_f = np.where(valid, depth_m, 0.0).astype(np.float32)
                t0 = time.perf_counter()
                pcd, pc_stats = builder.build(depth_f, params_color, color_bgr)
                pc_ms = (time.perf_counter() - t0) * 1000.0

                d = depth_m[valid]
                valid_ratio = 100.0 * valid.sum() / valid.size
                med = float(np.median(d)) if d.size else 0.0
                fps_n += 1
                if fps_n % 30 == 0:
                    fps = 30.0 / max(time.perf_counter() - fps_t0, 1e-9)
                    fps_t0 = time.perf_counter()
                last = {"depth": depth_m, "valid": valid, "color": color_bgr,
                        "params": params_color, "pc_stats": pc_stats, "pc_ms": pc_ms,
                        "valid_ratio": valid_ratio, "median_m": med}

                if vis is not None:
                    try:
                        pts = np.asarray(pcd.points)
                        if len(pts):
                            pts = pts * _flip_y          # 只翻显示
                        disp_pcd.points = o3d.utility.Vector3dVector(pts)
                        disp_pcd.colors = pcd.colors
                        # 只在**第一次**加入几何体；重复 add_geometry 会让窗口崩掉/卡死
                        # （这是本工具第一版的真 bug：vis_added 存在每帧重建的 last 里，
                        #   导致每帧都调用一次 add_geometry）
                        if not vis_added:
                            vis.add_geometry(disp_pcd)
                            vis_added = True
                        if pc_stats["points"] > 0:
                            vis.update_geometry(disp_pcd)
                        if not vis.poll_events():
                            print("  3D 窗口已关闭（预览继续；按 o 可重开）")
                            vis = None
                        else:
                            vis.update_renderer()
                    except Exception as exc:  # noqa: BLE001
                        # 3D 渲染出问题不应该把整个预览带崩：记录并关掉 3D 窗口继续跑
                        print(f"  [3D 窗口异常] {type(exc).__name__}: {exc}")
                        print("  已关闭 3D 窗口，继续 2D 预览（可用 --no-cloud-window 完全跳过）")
                        try:
                            vis.destroy_window()
                        except Exception:
                            pass
                        vis = None

            if not last:
                continue

            # ---- 2D 显示：彩色 | 深度伪彩（带色标）
            depth_vis = colorize_depth(last["depth"], last["valid"], a.zmin, a.zmax)
            left_img = last["color"] if last["color"] is not None else depth_vis
            ltitle = "COLOR" if last["color"] is not None else "DEPTH (no color stream)"
            status = [
                f"RealSense   depth {dw}x{dh} @ {a.fps}   loop fps {fps:.1f}   "
                f"cloud build {last['pc_ms']:.0f} ms",
                f"depth valid {last['valid_ratio']:.1f}%   median {last['median_m']:.3f} m   "
                f"range {a.zmin}..{a.zmax} m   points {last['pc_stats']['points']}",
                f"intrinsics fx={last['params'].fx:.1f} fy={last['params'].fy:.1f} "
                f"cx={last['params'].cx:.1f} cy={last['params'].cy:.1f}   FACTORY CALIBRATED",
                "keys: q=quit  s=save(PLY+depth+color)  p=pause  h=help  o=reopen 3D window",
            ]
            key = viewer.show_panels(
                [(ltitle, left_img, []), (f"DEPTH ({a.zmin}~{a.zmax} m)", depth_vis, [])],
                status_lines=status, paused=paused)

            if StereoViewer.is_quit(key):
                print("  收到退出按键")
                break
            if StereoViewer.is_pause(key):
                paused = not paused
                print(f"  {'已暂停' if paused else '已恢复'}")
            if StereoViewer.is_help(key):
                viewer.toggle_help()
            if key == ord("o") and vis is None and not a.no_cloud_window:
                vis = open_cloud_window()
                vis_added = False          # 新窗口里需要重新 add_geometry 一次
                print("  3D 窗口已重新打开")
            if StereoViewer.is_snapshot(key):
                d = time.strftime("%Y%m%d_%H%M%S")
                out_dir = os.path.join(out_root, d)
                os.makedirs(out_dir, exist_ok=True)
                files = []
                if last["color"] is not None:
                    p = os.path.join(out_dir, "color.png"); cv2.imwrite(p, last["color"]); files.append(p)
                p = os.path.join(out_dir, "depth_meter.npy"); np.save(p, last["depth"]); files.append(p)
                p = os.path.join(out_dir, "depth_vis.png"); cv2.imwrite(p, depth_vis); files.append(p)
                ply = builder.save_ply(o3d.geometry.PointCloud(pcd), out_dir, last["params"],
                                       extra={"source": "Intel RealSense（出厂标定）",
                                              "depth_scale_m": f"{depth_scale}"})
                files.append(ply)
                with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as fh:
                    fh.write("\n".join([
                        "# RealSense 预览保存",
                        f"时间        : {time.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"深度流      : {dw}x{dh}@{a.fps}   深度单位 {depth_scale} m/unit",
                        f"彩色流      : {'关闭' if a.depth_only else f'{cw}x{ch}@{a.fps}'}",
                        f"内参        : fx={last['params'].fx:.3f} fy={last['params'].fy:.3f} "
                        f"cx={last['params'].cx:.3f} cy={last['params'].cy:.3f}",
                        f"标定来源    : {last['params'].source}",
                        "坐标系      : 相机光学坐标系（x 右, y 下, z 前），单位米",
                        f"深度范围    : {a.zmin} ~ {a.zmax} m",
                        f"有效深度    : {last['valid_ratio']:.2f}%   中位 {last['median_m']:.4f} m",
                        f"点数        : {last['pc_stats']['points']}",
                        "",
                        "说明: RealSense 深度为**出厂标定**的真实尺度，可直接用于测距。",
                    ]) + "\n")
                files.append(os.path.join(out_dir, "meta.txt"))
                saved += 1
                print(f"  已保存 {len(files)} 个文件到: {out_dir}")

    except KeyboardInterrupt:
        print("\n  收到 Ctrl+C")
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        viewer.close()
        if vis is not None:
            try:
                vis.destroy_window()
            except Exception:
                pass

    print("-" * 88)
    print(f"  结束：保存 {saved} 次")
    print("  提示：点云/深度为 RealSense 出厂标定结果（真实尺度），可直接用于测距；")
    print("        本项目 DECXIN 那条路线（FFS + 自标定）与这条路线互不影响。")
    return 0


def colorize_depth(depth_m: np.ndarray, valid: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    """深度伪彩 + 右侧色标（纯 ASCII 文字，cv2.putText 画不了中文/emoji）。"""
    d = np.asarray(depth_m, dtype=np.float32)
    out = np.zeros((d.shape[0], d.shape[1], 3), np.uint8)
    m = np.asarray(valid, bool) & np.isfinite(d) & (d > 0)
    if m.any():
        span = max(float(zmax) - float(zmin), 1e-6)
        norm = np.clip((d - float(zmin)) / span, 0.0, 1.0)
        colored = cv2.applyColorMap(((1.0 - norm) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        out[m] = colored[m]
    # 色标（右侧竖条 + 刻度）
    h, w = out.shape[:2]
    bar_w = 18
    bar = np.linspace(255, 0, h).astype(np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(bar, bar_w, axis=1), cv2.COLORMAP_JET)
    out[:, w - bar_w:] = bar
    for i in range(5):
        z = zmin + (zmax - zmin) * i / 4.0
        y = int(h - 1 - (h - 1) * i / 4.0)
        cv2.line(out, (w - bar_w - 6, y), (w - bar_w, y), (255, 255, 255), 1)
        cv2.putText(out, f"{z:.2f}", (w - bar_w - 52, min(max(y + 4, 12), h - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# -----------------------------------------------------------------------------
# 日志：把控制台输出同时写进文件
# 为什么要它：用户实测"启动后窗口自动退出了"，控制台里的报错来不及看就没了。
# 有了日志文件，即使窗口闪退，也能把 output\realsense\last_run.log 拿来定位。
# -----------------------------------------------------------------------------
class _Tee:
    """把写入同时送到原控制台和日志文件。"""

    def __init__(self, stream, path: str):
        self._stream = stream
        self._path = path
        self._fh = open(path, "a", encoding="utf-8", errors="replace")

    def write(self, data: str) -> int:
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:
            pass
        return len(data)

    def flush(self) -> None:
        for obj in (self._stream, self._fh):
            try:
                obj.flush()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def install_log(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:      # 每次运行重新开始，避免越积越大
        fh.write("")
    sys.stdout = _Tee(sys.stdout, path)
    sys.stderr = _Tee(sys.stderr, path)
    return path


# -----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Intel RealSense 深度相机实时预览（深度 + 彩色点云）")
    ap.add_argument("--check", action="store_true", help="只枚举设备并打印出厂参数，不启动视频流")
    ap.add_argument("--depth-res", default="848x480", help="深度分辨率（默认 848x480）")
    ap.add_argument("--color-res", default="848x480", help="彩色分辨率（默认 848x480）")
    ap.add_argument("--fps", type=int, default=30, help="帧率（默认 30）")
    ap.add_argument("--depth-only", action="store_true", help="只取深度，不取彩色")
    ap.add_argument("--zmin", type=float, default=0.07, help="深度下限(米)；D405 最近约 0.07")
    ap.add_argument("--zmax", type=float, default=3.0, help="深度上限(米)")
    ap.add_argument("--max-points", type=int, default=200000, help="点云点数上限（默认 20 万）")
    ap.add_argument("--voxel", type=float, default=0.0, help="体素下采样边长(米)，0=不下采样")
    ap.add_argument("--point-size", type=float, default=2.0, help="3D 窗口点大小")
    ap.add_argument("--cloud-window", type=int, default=900, help="3D 窗口边长")
    ap.add_argument("--no-cloud-window", action="store_true", help="不开 3D 窗口")
    ap.add_argument("--display-scale", type=float, default=None, help="2D 显示缩放")
    a = ap.parse_args(argv)

    log_path = install_log(os.path.join(PROJECT_ROOT, "output", "realsense", "last_run.log"))
    admin = is_admin()
    print("=" * 88)
    print(f"  RealSense 预览   启动于 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python     : {sys.executable}")
    print(f"  管理员权限 : {'是 (OK)' if admin else '否 (!!) 请改用 tools\\RealSense预览_管理员.bat 启动'}")
    print(f"  本次输出同时写入日志: {log_path}")
    print("=" * 88)
    if not admin:
        print("  [警告] 当前不是管理员权限：本机普通权限进程打不开物理相机，")
        print("         RealSense SDK 也会报 CreateNamedMutex / 访问被拒。")
        print("         请双击 tools\\RealSense预览_管理员.bat（会自动请求提权）。")
    try:
        rc = print_check() if a.check else run_view(a)
    except KeyboardInterrupt:
        print("\n  收到 Ctrl+C，退出")
        rc = 0
    except BaseException:                     # noqa: BLE001
        import traceback
        print("\n" + "!" * 88)
        print("  [崩溃] 下面是完整堆栈（这张日志已存盘，可直接发给我定位）:")
        traceback.print_exc()
        print("!" * 88)
        rc = 9
    print(f"\n[日志] 本次输出已保存在: {log_path}")
    print("[提示] 如果窗口一闪就没了，把上面这个 last_run.log 发我即可定位。")
    for s in (sys.stdout, sys.stderr):
        if isinstance(s, _Tee):
            s.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
