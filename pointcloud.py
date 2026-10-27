"""pointcloud.py —— 点云生成、过滤、下采样、着色与 PLY 导出
=============================================================================
职责（单一）：把"深度图 + 内参 + 左图"变成"带颜色的三维点云"，并负责：
    1) 反投影（向量化，避免 Python 逐像素循环，保证实时性）；
    2) 无效深度剔除 + 深度范围过滤（zmin/zmax）；
    3) 体素下采样（可选）与点数上限保护（避免点数爆炸拖垮显示）；
    4) 保存为 PLY，并且**把坐标系/单位/参数来源/未标定警告写进文件本身**。

【坐标系与单位（必须牢记）】
    左相机光学坐标系：x 向右、y 向下、z 向前（深度），单位**米**。
    这与 OpenCV 针孔模型、Open3D 的默认显示习惯一致。

【性能要点（需求里明确要求"避免每帧重复创建大量对象"）】
    - 反投影全部用 numpy 向量化；
    - 复用同一个 o3d.geometry.PointCloud 对象，逐帧只更新 points/colors
      （见 PointCloudBuilder.update_inplace 与 main.py 里的用法）；
    - 提供点数上限（pointcloud.max_points），超限时做均匀抽样而不是让显示卡死。

【诚实原则】
    depth.calibrated=false 时，点云里的坐标**不具备测量意义**。
    本模块会把警告写进 PLY 的注释行（comment）和配套的 .txt，避免文件离开项目后失去上下文。
=============================================================================
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
    # Open3D 在窗口尺寸未确定/窗口不可见时，会**每帧**打印
    #   [Open3D WARNING] [ViewControl] SetViewPoint() failed because window height and width are not set.
    # 这是 Open3D 自己的已知告警，不是本项目的错误；把级别抬到 Error 可彻底静音
    # （真正的 Error 仍会打印）。需要排错时把它改回 Warning 即可。
    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    except Exception:  # noqa: BLE001
        pass
except Exception:  # noqa: BLE001 —— 没有 open3d 时仍允许导入本模块（仅导出/统计用）
    o3d = None

from depth_utils import CameraParams, filter_depth_range


def _insert_ply_comments(path: str, comments) -> bool:
    """把若干 comment 行插入 PLY 头部（**必须插在 format 行之后**）。

    为什么有这条限制（实测踩过）：
        PLY 规范里 comment 可以出现在头部任意位置，但常用的 RPly 解析器
        （Open3D 底层就用它）要求 `format` 必须紧跟在首行 `ply` 之后。
        如果把 comment 插在 ply 与 format 之间，文件会变成"非法格式"，
        Open3D 读回来直接报 "unable to parse header"。
        实测：Open3D 自己写文件时也是把 comment 放在 format 之后。

    输入：path —— PLY 路径；comments —— 字符串列表（会转成 ASCII，中文替换为 ?）
    输出：True=已插入；False=文件结构不符合预期（调用方不应因此失败）
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return False
    if not data.startswith(b"ply"):
        return False
    idx = data.find(b"end_header")
    if idx < 0:
        return False

    header_lines = data[:idx].split(b"\n")
    out = []
    inserted = False
    for ln in header_lines:
        out.append(ln)
        if (not inserted) and ln.strip().startswith(b"format"):
            for c in comments:
                out.append(b"comment " + str(c).encode("ascii", "replace"))
            inserted = True
    if not inserted:
        # 兜底：没有 format 行时插在第一行之后（这种情况极少）
        out = [header_lines[0]] + [b"comment " + str(c).encode("ascii", "replace") for c in comments] + header_lines[1:]

    new = b"\n".join(out) + data[idx:]
    try:
        with open(path, "wb") as fh:
            fh.write(new)
    except OSError:
        return False
    return True


class PointCloudBuilder:
    """深度图 -> 彩色点云（可复用同一几何对象做实时更新）。

    输入 cfg：config.yaml 的 pointcloud 段
        zmin_m / zmax_m : 深度过滤范围（米）
        voxel_size_m    : 体素下采样边长；0 或负数表示不下采样
        use_color       : 是否用左图 RGB 着色
        max_points      : 点数上限保护
    另可用参数：
        calibrated      : 是否已标定（只用于生成警告文字）
    """

    def __init__(self, cfg: Dict[str, Any], calibrated: bool = False) -> None:
        self.zmin_m = float(cfg.get("zmin_m", 0.15))
        self.zmax_m = float(cfg.get("zmax_m", 3.0))
        self.voxel_size_m = float(cfg.get("voxel_size_m", 0.0) or 0.0)
        self.use_color = bool(cfg.get("use_color", True))
        self.max_points = int(cfg.get("max_points", 400000) or 0)
        self.calibrated = bool(calibrated)
        # 复用对象（第一次创建后一直沿用，避免每帧新建）
        self._pcd = None
        self.stats: Dict[str, Any] = {"points": 0, "filtered_points": 0, "build_ms": 0.0, "downsampled": 0}

    # ------------------------------------------------------------ 反投影
    def build_arrays(
        self,
        depth: np.ndarray,
        params: CameraParams,
        color_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """把深度图反投影成点云数组（不依赖 Open3D，方便单元测试）。

        输入：
            depth     : (H,W) float32，单位米；<=0 视为无效
            params    : CameraParams（必须与 depth 同分辨率）
            color_bgr : 可选 (H,W,3) uint8，OpenCV 的 BGR 顺序（会用左图着色）
        输出：
            (points (N,3) float32 单位米, colors (N,3) float64 0..1 或 None)
        """
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim != 2:
            raise ValueError(f"深度图必须是 2D，当前形状 {d.shape}")
        h, w = d.shape[:2]

        mask = np.isfinite(d) & (d > 0)
        if color_bgr is not None:
            c = np.asarray(color_bgr)
            if c.shape[:2] != (h, w):
                raise ValueError(f"着色图尺寸 {c.shape[:2]} 与深度图 {(h, w)} 不一致"
                                 "（应使用与视差同分辨率的左图）")

        vs, us = np.nonzero(mask)
        if vs.size == 0:
            return np.zeros((0, 3), dtype=np.float32), (np.zeros((0, 3), dtype=np.float64) if self.use_color else None)

        z = d[vs, us].astype(np.float32)
        x = (us.astype(np.float32) - float(params.cx)) * z / float(params.fx)
        y = (vs.astype(np.float32) - float(params.cy)) * z / float(params.fy)
        pts = np.stack([x, y, z], axis=1).astype(np.float32)

        colors = None
        if self.use_color and color_bgr is not None:
            # OpenCV 是 BGR，Open3D 需要 RGB，并归一化到 0..1
            rgb = c[vs, us][:, ::-1].astype(np.float64) / 255.0
            colors = rgb
        return pts, colors

    # ------------------------------------------------------------ 组装点云
    def build(
        self,
        depth: np.ndarray,
        params: CameraParams,
        color_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """生成（或原地更新）Open3D 点云对象，并应用深度过滤 / 下采样 / 点数上限。

        输出：(pcd, stats)
            stats: {"points","filtered_points","build_ms","downsampled"}
        """
        if o3d is None:
            raise RuntimeError("未安装 open3d，无法生成点云对象（可只用 build_arrays 做数值验证）")

        t0 = time.perf_counter()

        # 1) 深度范围过滤（在反投影之前做，省掉无效点的计算）
        rng = filter_depth_range(depth, self.zmin_m, self.zmax_m)
        depth_f = np.where(rng, depth, 0.0).astype(np.float32)
        n_before = int((np.asarray(depth, np.float32) > 0).sum())

        pts, colors = self.build_arrays(depth_f, params, color_bgr)

        # 2) 点数上限保护（均匀抽样，保持空间分布）
        n = pts.shape[0]
        if self.max_points and n > self.max_points:
            idx = np.linspace(0, n - 1, self.max_points).astype(np.int64)
            pts = pts[idx]
            if colors is not None:
                colors = colors[idx]

        # 3) 复用同一个 PointCloud 对象
        if self._pcd is None:
            self._pcd = o3d.geometry.PointCloud()
        pcd = self._pcd
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if colors is not None and colors.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        # 4) 体素下采样（可选；下采样会重建点云对象，所以放在最后）
        down = 0
        if self.voxel_size_m and self.voxel_size_m > 0 and len(pcd.points) > 0:
            n0 = len(pcd.points)
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size_m)
            down = n0 - len(pcd.points)
            self._pcd = pcd

        self.stats = {
            "points": len(pcd.points),
            "filtered_points": max(n_before - pts.shape[0], 0),
            "build_ms": (time.perf_counter() - t0) * 1000.0,
            "downsampled": down,
        }
        return pcd, dict(self.stats)

    # ------------------------------------------------------------ 保存
    def save_ply(
        self,
        pcd: Any,
        out_dir: str,
        params: CameraParams,
        extra: Optional[Dict[str, Any]] = None,
        name: str = "cloud.ply",
    ) -> str:
        """保存 PLY，并把坐标系/单位/参数来源/未标定警告写进 PLY 注释与配套 txt。

        为什么要改 PLY 头部：PLY 格式支持 comment 行，把警告写进去以后，
        这个文件即使被单独拷走，别人也能看到"这是未标定数据"。
        输出：PLY 文件路径
        """
        if o3d is None:
            raise RuntimeError("未安装 open3d，无法导出 PLY")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=False)

        warn = ("已标定" if params.calibrated else "UNCALIBRATED / 未标定：占位内参+厂家标称基线，禁止测量")
        comments = [
            f"stereo_project point cloud",
            f"coordinate_system: left camera optical frame (x right, y down, z forward)",
            f"length_unit: meter",
            # PLY 的 comment 必须写成 ASCII：中文字符会被替换成 '?'，所以这里用英文，
            # 中文说明放在配套的 cloud_info.txt 里。
            f"CALIBRATION_STATUS: " + ("CALIBRATED" if params.calibrated else
                                        "UNCALIBRATED - placeholder intrinsics, NOT for measurement"),
            f"intrinsics_source: {params.source}",
            f"fx={params.fx:.4f} fy={params.fy:.4f} cx={params.cx:.4f} cy={params.cy:.4f} baseline_m={params.baseline_m:.6f}",
            f"depth_range_m: {self.zmin_m} .. {self.zmax_m}",
            f"created: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if extra:
            for k, v in extra.items():
                comments.append(f"{k}: {v}")

        if not _insert_ply_comments(path, comments):
            # 注释写不进去也不能让保存失败：退化为只写配套 txt
            pass

        with open(os.path.join(out_dir, "cloud_info.txt"), "w", encoding="utf-8") as fh:
            fh.write("# 点云数据说明\n")
            fh.write(f"文件           : {name}\n")
            fh.write("坐标系         : 左相机光学坐标系（x 向右, y 向下, z 向前）\n")
            fh.write("长度单位       : 米（meter）\n")
            fh.write(f"点数           : {len(pcd.points)}\n")
            fh.write(f"标定状态       : {params.describe()}\n")
            fh.write(f"深度过滤范围   : {self.zmin_m} .. {self.zmax_m} m\n")
            fh.write(f"体素下采样     : {self.voxel_size_m} m（0=未下采样）\n")
            fh.write("警告           : " + ("使用已标定参数" if params.calibrated else
                                             "【未标定】本点云不具备测量意义，不能用于测距/尺寸/定位") + "\n")
            if extra:
                for k, v in extra.items():
                    fh.write(f"{k:<15}: {v}\n")
        return path


def quick_cloud_from_depth(
    depth: np.ndarray,
    params: CameraParams,
    color_bgr: Optional[np.ndarray] = None,
    zmin_m: float = 0.15,
    zmax_m: float = 3.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """一次性反投影（不依赖 Open3D、不做下采样）—— 便于脚本/测试直接使用。

    输出：(points (N,3) float32 米, colors (N,3) float64 0..1 或 None)
    """
    b = PointCloudBuilder({"zmin_m": zmin_m, "zmax_m": zmax_m, "voxel_size_m": 0.0,
                           "use_color": color_bgr is not None, "max_points": 0},
                          calibrated=params.calibrated)
    rng = filter_depth_range(depth, zmin_m, zmax_m)
    return b.build_arrays(np.where(rng, depth, 0.0).astype(np.float32), params, color_bgr)
