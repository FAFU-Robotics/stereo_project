"""calibration.py —— 双目标定（阶段 5）：角点 → 单眼内参 → 双目外参 → 校正映射

=============================================================================
职责（单一）：把"标定板图像"变成"可用的双目标定参数"，并负责保存/读取/应用。

完整流程（全部用 OpenCV 官方函数，不自己造轮子）：
    ① 严格尺寸的亚像素角点      findChessboardCorners + cornerSubPix
    ② 单眼内参                  cv2.calibrateCamera（左右各自，用该眼所有有效帧）
    ③ 双目外参                  cv2.stereoCalibrate（只用双眼都有效的帧，
                                并用 CALIB_FIX_INTRINSIC 固定②的结果）
    ④ 校正映射                  cv2.stereoRectify + cv2.initUndistortRectifyMap
    => 输出 calibration/stereo_calib.yaml（只存矩阵与元信息，**不存映射表**）

【重要设计决定】
1) 角点必须**严格匹配规格尺寸**（本例 11x8）。
   实测踩过：候选尺寸搜索会"退而求其次"匹配到 (10,8) 子棋盘（80 个角点），
   而物点仍按 11x8 生成 —— 尺寸不一致会**静默污染参数**，
   而且采集报告里还会被误标成"双眼可用"。

2) 映射表**不入库**：它由 K/D/R/P 唯一确定，且 640x480 四张 float32 表写进 YAML
   会有十几 MB。这里在需要时用 initUndistortRectifyMap 现算（几毫秒）。

3) 深度用的内参在"校正后"与"校正前"是两套：
   - 运行时若先校正再推理（推荐），用**校正后**的左目投影矩阵 P1：
         depth = P1[0,0] * |T| / disparity
   - 若不校正，则用原始 K1（此时边缘误差明显更大）
   load_stereo_calibration() 用 rectified 参数区分这两种取法，
   与 depth_utils.CameraParams.from_config 的约定一致。

坐标系与单位：
    物点单位与 square_mm 相同（毫米），因此 T 是毫米；对外统一暴露 baseline_m（米）。
    三维点使用左相机（校正后）坐标系：x 右、y 下、z 前。
=============================================================================
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

# -----------------------------------------------------------------------------
# 标定板规格
# -----------------------------------------------------------------------------
@dataclass
class BoardSpec:
    """标定板规格（方格数 + 方格实际边长）。

    注意：capture_calib.py 里有一个同名同类，那份用于采集时的姿态/尺寸标注；
    这份是"标定计算"用的权威定义（生成物点）。两处都接受同样的三个参数，
    参数来源统一是 config.yaml 的 calibration 段。
    """
    cols: int = 12
    rows: int = 9
    square_mm: float = 3.0

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """OpenCV 要的**内角点**数 = (cols-1, rows-1)。"""
        return (self.cols - 1, self.rows - 1)

    def object_points(self) -> np.ndarray:
        """标定用物点 (N,3) float32，单位毫米，z=0 平面。"""
        nx, ny = self.pattern_size
        pts = np.zeros((nx * ny, 3), np.float32)
        pts[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        return pts * float(self.square_mm)

    def as_dict(self) -> Dict[str, Any]:
        return {"cols": self.cols, "rows": self.rows, "square_mm": self.square_mm,
                "inner_corners": list(self.pattern_size)}

    def describe(self) -> str:
        return (f"{self.cols}x{self.rows} 方格, 方格 {self.square_mm:g} mm, "
                f"图案 {self.cols*self.square_mm:g}x{self.rows*self.square_mm:g} mm, "
                f"内角点 {self.pattern_size[0]}x{self.pattern_size[1]}")


# -----------------------------------------------------------------------------
# 角点检测（严格尺寸）
# -----------------------------------------------------------------------------
def detect_corners_strict(gray: np.ndarray, pattern_size: Tuple[int, int],
                          refine: bool = True) -> Optional[np.ndarray]:
    """只按给定尺寸检测角点；检不到返回 None。

    输出 (N,1,2) float32 —— 统一形状，避免 OpenCV 5.0 返回 (N,2) 造成的索引错误。
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, tuple(pattern_size), flags)
    if not ok:
        return None
    corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    if refine:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return corners


def load_frame_points(frames_dir: str, spec: BoardSpec, refine: bool = True,
                      left_pat: str = "left_*.png", right_pat: str = "right_*.png"
                      ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[Dict[str, Any]]]:
    """扫描采集目录，返回 (左眼角点字典, 右眼角点字典, 每帧报告)。

    键是帧号（从文件名里的数字解析）。某一眼检不到就不放进对应字典 ——
    该帧仍可能对另一眼的单眼内参有用。
    """
    import glob
    import re

    def idx_of(p: str) -> int:
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    left_pts: Dict[int, np.ndarray] = {}
    right_pts: Dict[int, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    lefts = {idx_of(p): p for p in glob.glob(os.path.join(frames_dir, left_pat))}
    rights = {idx_of(p): p for p in glob.glob(os.path.join(frames_dir, right_pat))}
    for i in sorted(set(lefts) | set(rights)):
        row: Dict[str, Any] = {"idx": i, "left": False, "right": False}
        if i in lefts:
            img = cv2.imread(lefts[i])
            if img is not None:
                c = detect_corners_strict(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                          spec.pattern_size, refine)
                if c is not None:
                    left_pts[i] = c
                    row["left"] = True
        if i in rights:
            img = cv2.imread(rights[i])
            if img is not None:
                c = detect_corners_strict(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                          spec.pattern_size, refine)
                if c is not None:
                    right_pts[i] = c
                    row["right"] = True
        row["stereo"] = bool(row["left"] and row["right"])
        rows.append(row)
    return left_pts, right_pts, rows


# -----------------------------------------------------------------------------
# 标定结果
# -----------------------------------------------------------------------------
@dataclass
class StereoCalibration:
    """双目标定结果（含校正映射的按需生成）。"""
    image_size: Tuple[int, int]
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    rms_left: float = 0.0
    rms_right: float = 0.0
    rms_stereo: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    _maps: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    # ---------------------------------------------------------------- 派生量
    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def baseline_m(self) -> float:
        return self.baseline_mm / 1000.0

    @property
    def rect_fx(self) -> float:
        """校正后的等效焦距（深度公式里用的那个 fx）。"""
        return float(self.P1[0, 0])

    @property
    def axis_angle_deg(self) -> float:
        """两眼光轴夹角（度）。接近 0 说明两相机近乎平行。"""
        return float(np.degrees(np.arccos(np.clip((np.trace(self.R) - 1.0) / 2.0, -1.0, 1.0))))

    # ------------------------------------------------------------ 校正/应用
    def maps(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回左右校正映射表 (map1x, map1y, map2x, map2y)，首次调用时现算并缓存。"""
        if self._maps is None:
            m1x, m1y = cv2.initUndistortRectifyMap(
                self.K1, self.D1, self.R1, self.P1, self.image_size, cv2.CV_32FC1)
            m2x, m2y = cv2.initUndistortRectifyMap(
                self.K2, self.D2, self.R2, self.P2, self.image_size, cv2.CV_32FC1)
            self._maps = (m1x, m1y, m2x, m2y)
        return self._maps

    def rectify_pair(self, left: np.ndarray, right: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """把一对原图做去畸变+校正（行对齐）。灰度/彩色都支持。"""
        m1x, m1y, m2x, m2y = self.maps()
        return (cv2.remap(left, m1x, m1y, cv2.INTER_LINEAR),
                cv2.remap(right, m2x, m2y, cv2.INTER_LINEAR))

    # ---------------------------------------------------------------- 读写
    def to_dict(self) -> Dict[str, Any]:
        def arr(a) -> list:
            return np.asarray(a).tolist()
        return {
            "format": "stereo_project.stereo_calib",
            "version": 1,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_size": list(self.image_size),
            "K1": arr(self.K1), "D1": arr(self.D1),
            "K2": arr(self.K2), "D2": arr(self.D2),
            "R": arr(self.R), "T": arr(self.T), "E": arr(self.E), "F": arr(self.F),
            "R1": arr(self.R1), "R2": arr(self.R2),
            "P1": arr(self.P1), "P2": arr(self.P2), "Q": arr(self.Q),
            "baseline_mm": self.baseline_mm,
            "baseline_m": self.baseline_m,
            "rectified_fx": self.rect_fx,
            "axis_angle_deg": self.axis_angle_deg,
            "rms_left_px": self.rms_left,
            "rms_right_px": self.rms_right,
            "rms_stereo_px": self.rms_stereo,
            "meta": self.meta,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True, sort_keys=False)
        return path

    @classmethod
    def load(cls, path: str) -> "StereoCalibration":
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        if not isinstance(d, dict) or "K1" not in d:
            raise ValueError(f"不是有效的标定文件：{path}")
        a = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        return cls(
            image_size=tuple(int(x) for x in d.get("image_size", (640, 480))),
            K1=a("K1"), D1=a("D1"), K2=a("K2"), D2=a("D2"),
            R=a("R"), T=a("T").reshape(3, 1), E=a("E"), F=a("F"),
            R1=a("R1"), R2=a("R2"), P1=a("P1"), P2=a("P2"), Q=a("Q"),
            rms_left=float(d.get("rms_left_px", 0.0)),
            rms_right=float(d.get("rms_right_px", 0.0)),
            rms_stereo=float(d.get("rms_stereo_px", 0.0)),
            meta=dict(d.get("meta", {})),
        )

    def summary(self) -> str:
        lines = [
            f"图像尺寸      : {self.image_size[0]}x{self.image_size[1]}",
            f"单眼 RMS      : 左 {self.rms_left:.4f} px / 右 {self.rms_right:.4f} px",
            f"双目 RMS      : {self.rms_stereo:.4f} px",
            f"基线          : {self.baseline_mm:.3f} mm（{self.baseline_m:.6f} m）",
            f"两眼光轴夹角  : {self.axis_angle_deg:.3f} 度",
            f"左目内参      : fx={self.K1[0,0]:.2f} fy={self.K1[1,1]:.2f} "
            f"cx={self.K1[0,2]:.2f} cy={self.K1[1,2]:.2f}",
            f"右目内参      : fx={self.K2[0,0]:.2f} fy={self.K2[1,1]:.2f} "
            f"cx={self.K2[0,2]:.2f} cy={self.K2[1,2]:.2f}",
            f"校正后等效焦距: fx={self.rect_fx:.2f}（深度公式用的就是它）",
        ]
        if self.meta:
            if "frames_stereo" in self.meta:
                lines.append(f"参与标定帧数  : 单眼 左 {self.meta.get('frames_left')} / "
                             f"右 {self.meta.get('frames_right')}，双目 {self.meta.get('frames_stereo')}")
            if "board" in self.meta:
                lines.append(f"标定板        : {self.meta['board']}")
        return "\n".join("  " + x for x in lines)


# -----------------------------------------------------------------------------
# 标定计算
# -----------------------------------------------------------------------------
def calibrate_stereo(left_pts: Dict[int, np.ndarray], right_pts: Dict[int, np.ndarray],
                     spec: BoardSpec, image_size: Tuple[int, int],
                     fix_intrinsic: bool = True,
                     extra_meta: Optional[Dict[str, Any]] = None) -> StereoCalibration:
    """执行完整标定。

    输入：
        left_pts/right_pts : {帧号: 角点 (N,1,2)}，允许两边的帧集合不同
        spec               : 标定板规格（决定物点）
        image_size         : (W,H)
        fix_intrinsic      : True = 先用各眼全部帧标定内参，再在双目标定中固定它们
                             （推荐：能把"只单眼有效"的帧也利用起来）
    输出：StereoCalibration（含逐帧误差，放在 meta['per_frame_errors']）
    """
    objp = spec.object_points()
    if len(left_pts) < 4 or len(right_pts) < 4:
        raise ValueError(f"有效帧太少：左 {len(left_pts)}、右 {len(right_pts)}（至少各 4 帧，建议 ≥15）")
    both = sorted(set(left_pts) & set(right_pts))
    if len(both) < 4:
        raise ValueError(f"双眼都有效的帧太少：{len(both)}（至少 4，建议 ≥15）")

    # ② 单眼内参（各眼用自己全部有效帧）
    rms_l, K1, D1, _, _ = cv2.calibrateCamera(
        [objp] * len(left_pts), [left_pts[i] for i in sorted(left_pts)], image_size, None, None)
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(
        [objp] * len(right_pts), [right_pts[i] for i in sorted(right_pts)], image_size, None, None)

    # ③ 双目外参
    o = [objp] * len(both)
    il = [left_pts[i] for i in both]
    ir = [right_pts[i] for i in both]
    flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsic else 0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    rms_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        o, il, ir, K1, D1, K2, D2, image_size, flags=flags, criteria=crit)

    # 逐帧误差（用于找出"害群之马"）
    # 注意：cv2.stereoCalibrate **不返回**每帧位姿，所以左相机位姿要单独求一次
    per_frame: List[Dict[str, Any]] = []
    rv_l, tv_l = None, None
    try:
        _, _, _, rv_l, tv_l = cv2.calibrateCamera(
            o, il, image_size, K1.copy(), D1.copy(), flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    except cv2.error:
        rv_l = tv_l = None
    if rv_l is not None:
        for k, i in enumerate(both):
            pl, _ = cv2.projectPoints(objp, rv_l[k], tv_l[k], K1, D1)
            Rr = R @ cv2.Rodrigues(rv_l[k])[0]
            tr = R @ tv_l[k] + T
            pr, _ = cv2.projectPoints(objp, cv2.Rodrigues(Rr)[0], tr, K2, D2)
            el = float(np.sqrt(((pl.reshape(-1, 2) - il[k].reshape(-1, 2)) ** 2).sum(1)).mean())
            er = float(np.sqrt(((pr.reshape(-1, 2) - ir[k].reshape(-1, 2)) ** 2).sum(1)).mean())
            per_frame.append({"idx": i, "err_left_px": el, "err_right_px": er,
                              "err_mean_px": (el + er) / 2})
        per_frame.sort(key=lambda x: -x["err_mean_px"])

    # ④ 校正
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    meta: Dict[str, Any] = {
        "board": spec.describe(),
        "board_params": spec.as_dict(),
        "frames_left": len(left_pts), "frames_right": len(right_pts),
        "frames_stereo": len(both),
        "stereo_frame_indices": both,
        "fix_intrinsic": bool(fix_intrinsic),
        "per_frame_errors": per_frame[:10],     # 只存最差的 10 帧，避免文件过大
        "square_mm": spec.square_mm,
    }
    if extra_meta:
        meta.update(extra_meta)

    return StereoCalibration(
        image_size=tuple(image_size), K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        rms_left=float(rms_l), rms_right=float(rms_r), rms_stereo=float(rms_s),
        meta=meta)


# -----------------------------------------------------------------------------
# 供运行时读取（与 depth_utils.CameraParams.from_config 的约定一致）
# -----------------------------------------------------------------------------
def load_stereo_calibration(path: str, rectified: bool = True) -> Dict[str, Any]:
    """读取标定文件，返回**深度计算所需的左目参数**。

    输入：
        path      : stereo_calib.yaml
        rectified : True  = 运行时"先校正再推理"，返回校正后的等效内参（P1）；
                    False = 运行时用原始图像，返回原始左目内参（K1）
    输出：{"fx","fy","cx","cy","baseline_m","date","width","height","rectified"}
    """
    with open(path, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    if not isinstance(d, dict) or "K1" not in d:
        raise ValueError(f"不是有效的标定文件：{path}")
    K1 = np.asarray(d["K1"], dtype=np.float64)
    P1 = np.asarray(d["P1"], dtype=np.float64)
    T = np.asarray(d["T"], dtype=np.float64).reshape(3)
    M = P1 if rectified else K1
    w, h = (int(x) for x in d.get("image_size", (640, 480)))
    return {
        "fx": float(M[0, 0]), "fy": float(M[1, 1]),
        "cx": float(M[0, 2]), "cy": float(M[1, 2]),
        "baseline_m": float(np.linalg.norm(T)) / 1000.0,
        "date": d.get("date", "未知"),
        "width": w, "height": h,
        "rectified": bool(rectified),
        "rms_stereo_px": float(d.get("rms_stereo_px", 0.0)),
    }


# -----------------------------------------------------------------------------
# 校正验证
# -----------------------------------------------------------------------------
def epipolar_error(calib: StereoCalibration, frames_dir: str, spec: BoardSpec,
                   frames: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """**校正验证**：把图像校正后重新检测角点，测量同一角点的垂直失配 |dy|。

    原理：校正正确的标志是"同一物点在左右图中位于同一行"（|dy| ≈ 0）。
    这是最直接的独立验证 —— 它不依赖标定时的拟合残差。

    输出：每帧 {"idx","dy_mean_px","dy_max_px","dx_mean_px","found"}
    """
    import glob
    import re

    def idx_of(p: str) -> int:
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    lefts = {idx_of(p): p for p in glob.glob(os.path.join(frames_dir, "left_*.png"))}
    rights = {idx_of(p): p for p in glob.glob(os.path.join(frames_dir, "right_*.png"))}
    todo = sorted(set(lefts) & set(rights)) if frames is None else sorted(frames)

    out: List[Dict[str, Any]] = []
    for i in todo:
        if i not in lefts or i not in rights:
            continue
        L = cv2.imread(lefts[i])
        R = cv2.imread(rights[i])
        if L is None or R is None:
            continue
        Lr, Rr = calib.rectify_pair(L, R)
        cl = detect_corners_strict(cv2.cvtColor(Lr, cv2.COLOR_BGR2GRAY), spec.pattern_size)
        cr = detect_corners_strict(cv2.cvtColor(Rr, cv2.COLOR_BGR2GRAY), spec.pattern_size)
        if cl is None or cr is None:
            out.append({"idx": i, "found": False})
            continue
        a = cl.reshape(-1, 2)
        b = cr.reshape(-1, 2)
        dy = a[:, 1] - b[:, 1]
        dx = a[:, 0] - b[:, 0]
        out.append({"idx": i, "found": True,
                    "dy_mean_px": float(np.abs(dy).mean()), "dy_max_px": float(np.abs(dy).max()),
                    "dx_mean_px": float(dx.mean()), "dx_min_px": float(dx.min()), "dx_max_px": float(dx.max())})
    return out
