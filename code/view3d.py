"""view3d.py —— Open3D 3D 点云窗口（供新链路 runtime.py 使用）

=============================================================================
【为什么单独一个文件】
    main.py 里的 open_cloud_window() 是**定义在 run_pointcloud 内部**的局部函数，
    外部无法 import；新链路又要功能对齐（旧链路的 3D 窗口不能丢），
    所以这里做一个自包含的等价实现，不去动 main.py。

【踩过的坑（都处理了）】
    1) Open3D 的 add_geometry() 只能在**初始化时调用一次**；
       逐帧再 add 会让窗口立刻死掉（本项目早期就是这样把 3D 窗口搞崩的）。
       => 这里只 new 一个 PointCloud 对象，之后每帧只替换它的 points/colors。
    2) 点云坐标是"相机系 y 向下"，直接显示会上下颠倒。
       => 显示前把 y 取负（只影响显示，不影响保存的数据）。
    3) 窗口被用户关掉后，继续 poll/update 会报错。
       => poll() 返回 False 时把窗口置空，调用方停止更新即可。
    4) Open3D 的 SetViewPoint 刷屏告警在 pointcloud.py 里已全局静音。
=============================================================================
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class CloudWindow3D:
    """一个可复用的 Open3D 点云窗口。"""

    def __init__(self, point_size: float = 2.0, width: int = 900, height: int = 700,
                 window_name: str = "3D") -> None:
        import open3d as o3d          # 延迟导入：预览/视差模式不必付这个开销

        self._o3d = o3d
        self._vis = o3d.visualization.Visualizer()
        self._visible = bool(self._vis.create_window(window_name=window_name,
                                                     width=int(width), height=int(height)))
        if not self._visible:
            raise RuntimeError("Open3D 窗口创建失败")
        self._pcd = o3d.geometry.PointCloud()
        self._pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))   # 占位，避免空几何
        self._pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        self._vis.add_geometry(self._pcd)          # ← 只加一次
        self._opt = self._vis.get_render_option()
        self._opt.point_size = float(point_size)
        self._opt.background_color = np.array([0.0, 0.0, 0.0])
        try:
            self._opt.light_on = False          # 顶点色，不打光，和 RealSense 3D 一样
        except Exception:  # noqa: BLE001
            pass
        self._n = 0

    # ------------------------------------------------------------------ 更新
    def update(self, points: np.ndarray, colors: np.ndarray) -> bool:
        """用新点云刷新窗口。points (N,3) 米（相机系 y 向下）；colors (N,3) BGR uint8。"""
        if not self._visible:
            return False
        if points is None or len(points) == 0:
            return True
        pts = np.asarray(points, dtype=np.float64).copy()
        pts[:, 1] = -pts[:, 1]                     # y 向下 -> 向上，只为显示
        self._pcd.points = self._o3d.utility.Vector3dVector(pts)
        if colors is not None and len(colors) == len(pts):
            rgb = np.asarray(colors, dtype=np.float64)[:, ::-1] / 255.0   # BGR -> RGB(0~1)
            self._pcd.colors = self._o3d.utility.Vector3dVector(rgb)
        self._vis.update_geometry(self._pcd)
        if self._n == 0:
            # 第一帧把视角对准数据（否则默认视角可能什么都看不到）
            try:
                self._vis.reset_view_point(True)
            except Exception:  # noqa: BLE001
                pass
        self._n += 1
        return self.poll()

    def poll(self) -> bool:
        """处理事件并重绘；返回 False 表示窗口已被关闭。"""
        if not self._visible:
            return False
        try:
            if not self._vis.poll_events():
                self._visible = False
                return False
            self._vis.update_renderer()
        except Exception:  # noqa: BLE001 —— 窗口被销毁时各版本异常类型不一致
            self._visible = False
            return False
        return True

    def set_point_size(self, size: float) -> None:
        try:
            self._opt.point_size = float(size)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ 关闭
    def close(self) -> None:
        if self._visible:
            self._visible = False
            try:
                self._vis.destroy_window()
            except Exception:  # noqa: BLE001
                pass

    @property
    def alive(self) -> bool:
        return self._visible


def _selftest() -> int:
    """无头自检：建窗口(可见) -> 更新 3 帧 -> 关闭。不需要相机。"""
    import sys
    import time

    print("=" * 64)
    print("  view3d 自检")
    print("=" * 64)
    ok = True
    n = 500
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 0.05, (n, 3)).astype(np.float32)
    pts[:, 2] += 0.6                      # 放到 z=0.6 m 附近
    cols = (rng.random((n, 3)) * 255).astype(np.uint8)
    w: Optional[CloudWindow3D] = None
    try:
        w = CloudWindow3D(point_size=3.0, width=320, height=240)
        print(f"  ✅ 窗口已创建（{n} 点）")
        for i in range(3):
            alive = w.update(pts, cols)
            time.sleep(0.05)
        print("  ✅ 连续更新 3 帧未抛异常" + ("（窗口仍存活）" if alive else "（窗口已关闭）"))
        w.set_point_size(5.0)
        print("  ✅ set_point_size 可用")
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ 失败: {type(exc).__name__}: {exc}")
        ok = False
    finally:
        if w is not None:
            w.close()
            print("  ✅ close() 后 alive =", w.alive)
            ok = ok and (not w.alive)
    print("\n  " + ("✅ 自检通过" if ok else "❌ 自检失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
