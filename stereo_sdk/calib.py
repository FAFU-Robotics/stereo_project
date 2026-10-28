"""calib.py —— 只读的标定信息视图（给"外部程序"用的那层）

为什么需要它：
  Session 内部当然知道自己的标定参数，但调用方拿不到——他只知道"深度单位是米"，
  不知道 **fx 是多少、基线是多少、RMS 多可信、这份标定是哪天做的**。
  这三件事决定了深度值能不能被当真：基线错 1 mm，1 m 处就错 15 mm。

设计约束：
  · **只读**。这里不提供 save / 修改，改标定请走标定 GUI（避免两处各写一份）。
  · **不依赖 OpenCV / torch**。只读 YAML + numpy，所以可以在纯客户端环境里 import。
  · 单位统一：长度一律**米**，焦距一律**像素**，RMS 一律**像素**。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError as exc:                # pragma: no cover
    raise ImportError("读标定文件需要 pyyaml：pip install pyyaml") from exc

__all__ = ["CalibInfo", "DEFAULT_CALIB_PATH"]


def _vec3(d: Dict[str, Any], key: str, path: str) -> np.ndarray:
    """把 [[x],[y],[z]] 或 [x,y,z] 统一成 (3,) float64。"""
    if key not in d:
        raise KeyError(f"标定文件缺少 {key}：{path}")
    a = np.asarray(d[key], np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"{key} 应有 3 个分量，实际 {a.size} 个")
    return a


#: 项目默认的标定文件位置（相对仓库根目录）
DEFAULT_CALIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "calibration", "stereo_calib.yaml")


def candidate_paths() -> List[str]:
    """按优先级列出可能的标定文件位置（去重、保持顺序）。

    为什么需要：`pip install` 之后，包被复制到 site-packages，
    原来"相对仓库根目录"的那个默认路径就指向 site-packages/calibration/ 了，
    那儿没有标定文件。所以查找顺序改为：
      1) 环境变量 STEREO_CALIB（显式指定，最高优先）
      2) 当前工作目录下的 calibration/stereo_calib.yaml（用户在自己的项目里跑）
      3) 包所在目录的上一级（源码树直接运行时命中）
      4) 用户主目录的 .stereo/calibration/（把这个文件拷出来单独用）
    """
    out: List[str] = []
    env = os.environ.get("STEREO_CALIB", "").strip()
    if env:
        out.append(env)
    out.append(os.path.join(os.getcwd(), "calibration", "stereo_calib.yaml"))
    out.append(DEFAULT_CALIB_PATH)
    out.append(os.path.expanduser(os.path.join("~", ".stereo", "calibration",
                                               "stereo_calib.yaml")))
    seen, uniq = set(), []
    for p in out:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def resolve_calib_path(path: str = "") -> str:
    """返回第一个存在的标定文件路径；都不存在时抛 FileNotFoundError 并列出查过哪里。"""
    if path:
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"指定的标定文件不存在：{path}")
    cands = candidate_paths()
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "找不到标定文件，已依次查找：\n  "
        + "\n  ".join(cands)
        + "\n请显式传入路径，或设置环境变量 STEREO_CALIB=<你的 stereo_calib.yaml>，"
          "\n或先用标定 GUI（tools/calib_gui.py）做一次标定。")


__all__ = ["CalibInfo", "DEFAULT_CALIB_PATH", "candidate_paths", "resolve_calib_path"]


@dataclass
class CalibInfo:
    """一份标定文件的只读摘要 + 原始矩阵。"""

    path: str
    image_size: Tuple[int, int]                    # (宽, 高)
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray                                  # 米
    date: str = ""
    rms_left: float = 0.0
    rms_right: float = 0.0
    rms_stereo: float = 0.0
    axis_angle_deg: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ 读
    @classmethod
    def from_file(cls, path: str = "") -> "CalibInfo":
        p = resolve_calib_path(path)
        with open(p, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        fmt = str(d.get("format", ""))
        if fmt and fmt != "stereo_project.stereo_calib":
            raise ValueError(f"不是本项目的标定文件（format={fmt}）：{p}")

        def mat(key: str, shape: Tuple[int, ...]) -> np.ndarray:
            if key not in d:
                raise KeyError(f"标定文件缺少 {key}：{p}")
            a = np.asarray(d[key], np.float64)
            if a.shape != shape:
                raise ValueError(f"{key} 形状应为 {shape}，实际 {a.shape}")
            return a

        size = tuple(int(v) for v in d.get("image_size", (640, 480)))
        info = cls(
            path=os.path.abspath(p),
            image_size=(int(size[0]), int(size[1])),
            K1=mat("K1", (3, 3)), D1=np.asarray(d.get("D1", []), np.float64).reshape(-1),
            K2=mat("K2", (3, 3)), D2=np.asarray(d.get("D2", []), np.float64).reshape(-1),
            R=mat("R", (3, 3)),
            T=_vec3(d, "T", p),
            date=str(d.get("date", "")),
            rms_left=float(d.get("rms_left_px", 0.0) or 0.0),
            rms_right=float(d.get("rms_right_px", 0.0) or 0.0),
            rms_stereo=float(d.get("rms_stereo_px", 0.0) or 0.0),
            axis_angle_deg=float(d.get("axis_angle_deg", 0.0) or 0.0),
            meta=dict(d.get("meta") or {}),
            _raw=d,
        )
        return info

    @classmethod
    def latest(cls, calib_dir: str = "") -> "CalibInfo":
        """取目录下最新的一份 *.yaml（按修改时间）。"""
        if calib_dir:
            d = calib_dir
        else:
            try:
                d = os.path.dirname(resolve_calib_path())
            except FileNotFoundError:
                d = os.path.dirname(DEFAULT_CALIB_PATH)
        cands: List[str] = []
        for root, _dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".yaml", ".yml")):
                    cands.append(os.path.join(root, f))
        if not cands:
            raise FileNotFoundError(f"{d} 下没有 yaml 标定文件")
        return cls.from_file(max(cands, key=os.path.getmtime))

    # -------------------------------------------------------------- 派生量
    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    @property
    def baseline_m(self) -> float:
        return self.baseline_mm / 1000.0

    @property
    def rms(self) -> float:
        """双目重投影 RMS（像素）。等价于 rms_stereo，只是名字短一点。"""
        return float(self.rms_stereo)

    @property
    def fx(self) -> float:
        """左目 x 方向焦距（像素）。"""
        return float(self.K1[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K1[1, 1])

    @property
    def rect_fx(self) -> float:
        """校正后的等效焦距（算深度用的是这个）。"""
        return float(self._raw.get("rectified_fx") or self.fx)

    @property
    def principal_point(self) -> Tuple[float, float]:
        return float(self.K1[0, 2]), float(self.K1[1, 2])

    @property
    def fov_deg(self) -> Tuple[float, float]:
        """水平 / 垂直视场角（按针孔近似；鱼眼镜头实际更大）。"""
        import math
        w, h = self.image_size
        return (math.degrees(2 * math.atan(w / (2 * self.fx))),
                math.degrees(2 * math.atan(h / (2 * self.fy))))

    def sigma_depth(self, z_m: float, disp_sigma_px: float = 0.5) -> float:
        """给定距离与视差噪声，估算深度不确定度（米）。

        σz = z² / (f·B) · σd。disp_sigma_px 默认 0.5 px 是这类模型的常见量级，
        真实值请用你自己测到的重投影/视差噪声替换。
        """
        if self.rect_fx <= 0 or self.baseline_m <= 0:
            return float("nan")
        return (z_m ** 2) / (self.rect_fx * self.baseline_m) * float(disp_sigma_px)

    def usable_range(self, max_sigma_m: float = 0.01,
                     disp_sigma_px: float = 0.5) -> float:
        """在给定深度误差上限下，最远可用的距离（米）。"""
        denom = max_sigma_m * self.rect_fx * self.baseline_m
        if denom <= 0:
            return float("nan")
        return (denom / max(disp_sigma_px, 1e-9)) ** 0.5

    # ------------------------------------------------------------------ 输出
    def summary(self) -> str:
        fov_h, fov_v = self.fov_deg
        lines = [
            f"标定文件    : {self.path}",
            f"标定日期    : {self.date or '(未记录)'}",
            f"图像尺寸    : {self.image_size[0]}x{self.image_size[1]}",
            f"左目 fx/fy  : {self.fx:.2f} / {self.fy:.2f} px",
            f"视场(近似)  : {fov_h:.1f}° x {fov_v:.1f}°",
            f"基线        : {self.baseline_mm:.3f} mm（{self.baseline_m:.5f} m）",
            f"两眼光轴夹角: {self.axis_angle_deg:.3f}°",
            f"校正后 fx   : {self.rect_fx:.2f} px",
            f"重投影 RMS  : 左 {self.rms_left:.3f} / 右 {self.rms_right:.3f} / "
            f"双目 {self.rms_stereo:.3f} px",
        ]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        """给 JSON / 日志用的扁平字典（不含大矩阵）。"""
        return {
            "path": self.path,
            "date": self.date,
            "image_size": list(self.image_size),
            "fx": self.fx,
            "fy": self.fy,
            "rect_fx": self.rect_fx,
            "baseline_mm": self.baseline_mm,
            "axis_angle_deg": self.axis_angle_deg,
            "rms_stereo_px": self.rms_stereo,
            "meta": dict(self.meta),
        }

    def check(self) -> List[Tuple[str, str, bool]]:
        """轻量体检：[(项, 说明, 是否通过)]，不依赖 OpenCV。

        阈值与 tools/calib_core.validate 保持一致的大方向，但这里更宽松——
        它只是"这份文件能不能用"的第一道关，真正的体检在 GUI 里。
        """
        items: List[Tuple[str, str, bool]] = []
        items.append(("文件存在", self.path, True))
        items.append(("图像尺寸 > 0", f"{self.image_size[0]}x{self.image_size[1]}",
                      min(self.image_size) > 0))
        items.append(("基线合理", f"{self.baseline_mm:.2f} mm",
                      20.0 < self.baseline_mm < 500.0))
        items.append(("焦距为正", f"fx={self.fx:.1f}", self.fx > 0))
        rms = self.rms_stereo
        items.append(("双目 RMS", f"{rms:.3f} px",
                      True if rms <= 0 else rms < 1.0))
        items.append(("光轴夹角", f"{self.axis_angle_deg:.2f}°",
                      self.axis_angle_deg < 5.0))
        return items
