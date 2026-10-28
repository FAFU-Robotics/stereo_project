"""calib_core.py —— 标定采集/计算的"无界面"核心（GUI 与 CLI 共用）
=============================================================================
【为什么单独抽这一层】
    GUI 代码一旦混进"质检判据、文件命名、角点统计"，就没法自动化测试了。
    这里把所有不含界面的逻辑集中起来：
        CaptureSession   采集会话（质检门 + 去重 + 覆盖统计 + 写报告）
        calibrate_from_dir  从采集目录算出 StereoCalibration（后台线程里跑）
        validate         标定结果的"合理性体检"（rms 之外还查什么）
    这样 calib_gui.py 只剩"按钮 → 调这里 → 显示"，出错时能一眼分清是算法错还是界面错。

【复用而不是重写】
    质检判据、姿态分类、清单规则全部来自 tools/capture_calib.py（纯函数，可直接 import）：
        analyze_pair / classify_pose / gate_messages / _pose_signature
    标定计算来自 calibration.py：
        load_frame_points / calibrate_stereo / epipolar_error
    本文件**不修改**上面两个模块的任何行为，只负责把它们串成会话流程。

【一条重要设计】
    图像尺寸一律**从实际图片读**，不写死 640x480。
    旧版 tools/run_calibration.py 写死 (640, 480)，换分辨率后标定会算错且不报错。

【坐标系与单位（与 calibration.py 一致）】
    物点与 T 的单位是毫米；对外暴露 baseline_m 是米；深度用 P1 的 fx。
=============================================================================
"""

from __future__ import annotations

import glob
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---- 路径解析（源码运行 = 项目根；打包成 EXE 后 = exe 所在目录）----
# 原来这里写的是 dirname(dirname(__file__))，打包后 __file__ 指向 _MEIPASS 临时目录，
# 采集图与标定文件会被写进"退出即删"的文件夹。统一交给 app_paths 判断。
try:
    from app_paths import data_root as _data_root, find as _find
except ImportError:                    # tools/ 被单独拷走使用时的兜底，行为与改造前一致
    def _data_root() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _find(rel: str) -> str:
        return os.path.join(_data_root(), rel)

PROJECT_ROOT = _data_root()
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CAPTURE_ROOT = os.path.join(PROJECT_ROOT, "output", "calib")
CALIB_DIR = os.path.join(PROJECT_ROOT, "calibration")
CURRENT_CALIB = os.path.join(CALIB_DIR, "stereo_calib.yaml")

#: 采集质检门默认值（与 config.yaml 的 calibration 段保持一致；GUI 里的输入框会覆盖它们）
DEFAULT_THR: Dict[str, float] = {
    "min_px_per_square": 8.0,
    "min_sharpness": 100.0,
    "min_contrast": 120.0,
}

#: 采集清单默认值（键名必须出现在 capture_calib.CHECK_RULES 里）
DEFAULT_CHECKLIST: Dict[str, int] = {
    "中心正对": 2,
    "左上角": 1, "右上角": 1, "左下角": 1, "右下角": 1,
    "左倾大于20度": 3, "右倾大于20度": 3, "上仰大于20度": 3, "下俯大于20度": 3,
}

ProgressFn = Optional[Callable[[str, str], None]]


# -----------------------------------------------------------------------------
# 延迟导入：capture_calib / calibration 会连带 import camera、viewer、yaml，
# 让"只想画一张热力图"的调用方不必付这份代价。
# -----------------------------------------------------------------------------
_CC = None
_CAL = None


def _cc():
    global _CC
    if _CC is None:
        import capture_calib as m          # tools/ 已在 sys.path
        _CC = m
    return _CC


def _cal():
    global _CAL
    if _CAL is None:
        import calibration as m            # 项目根已在 sys.path
        _CAL = m
    return _CAL


def default_spec(cols: int = 12, rows: int = 9, square_mm: float = 3.0):
    """返回 calibration.BoardSpec（标定计算的权威定义）。"""
    return _cal().BoardSpec(cols=int(cols), rows=int(rows), square_mm=float(square_mm))


def spec_from_config(config_path: str = "") -> Tuple[Any, Dict[str, Any]]:
    """从 config.yaml 读标定段，返回 (spec, calibration 段字典)。读不到就用默认 12x9/3mm。"""
    # config.yaml 是只读资源，打包后可能只在 _MEIPASS 里，所以用 find() 而不是拼 PROJECT_ROOT
    path = config_path or _find("config.yaml")
    cfg: Dict[str, Any] = {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:                       # noqa: BLE001 —— 读不到配置不该让 GUI 起不来
        cfg = {}
    cal = dict(cfg.get("calibration", {}) or {})
    spec = default_spec(cal.get("board_cols", 12), cal.get("board_rows", 9),
                        cal.get("square_mm", 3.0))
    return spec, cal


def notify(progress: ProgressFn, stage: str, detail: str = "") -> None:
    """调用进度回调；回调自身出错不允许影响标定。"""
    if progress is None:
        return
    try:
        progress(stage, detail)
    except Exception:                       # noqa: BLE001
        pass


# -----------------------------------------------------------------------------
# "还差几张"的计算（纯函数，GUI 与报告共用）
# -----------------------------------------------------------------------------
#: 清单项 -> 补拍建议。键名来自 DEFAULT_CHECKLIST / capture_calib.CHECK_RULES
_ADVICE = {
    "中心正对": "板子正对镜头、放在画面中央",
    "左上角": "把板子移到画面左上",
    "右上角": "把板子移到画面右上",
    "左下角": "把板子移到画面左下",
    "右下角": "把板子移到画面右下",
    "左倾大于20度": "板子向左倾 20° 以上",
    "右倾大于20度": "板子向右倾 20° 以上",
    "上仰大于20度": "板子上仰 20° 以上",
    "下俯大于20度": "板子下俯 20° 以上",
}


def shortage_plan(count: int, target: int,
                  got: Dict[str, int], need: Dict[str, int]) -> Dict[str, Any]:
    """算"还差几张、缺哪类姿态、下一步怎么拍"。纯函数，无 IO，可直接单测。

    返回：
        short   还差多少对（负数取 0）
        missing 未达标的清单项名（按 need 的顺序）
        done    数量与清单都达标
        text    一行可直接显示的摘要
        advice  最多 4 条补拍动作
    """
    count, target = int(count), int(target)
    short = max(0, target - count)
    missing = [n for n, k in need.items() if int(got.get(n, 0)) < int(k)]
    done = (short == 0 and not missing)

    parts: List[str] = []
    if short > 0:
        parts.append(f"还差 {short} 对（已 {count}/{target}）")
    elif missing:
        parts.append(f"数量够了（{count}/{target}），但姿态清单没齐")
    else:
        parts.append(f"数量与姿态清单都已达标（{count}/{target}）")
    if missing:
        detail = "、".join(f"{n} {int(got.get(n, 0))}/{int(need[n])}" for n in missing[:4])
        if len(missing) > 4:
            detail += f" 等 {len(missing)} 项"
        parts.append(f"缺：{detail}")

    advice = [_ADVICE[n] for n in missing if n in _ADVICE][:4]
    if short > 0 and not advice:
        advice = ["换几个不同位置和角度再拍，不必追求同一姿势"]

    return {"short": short, "missing": missing, "done": done,
            "text": "；".join(parts), "advice": advice}


# -----------------------------------------------------------------------------
# 异常
# -----------------------------------------------------------------------------
class CaptureRejected(RuntimeError):
    """这一帧不满足质检门 / 姿态重复，未保存。"""


class CalibrationFailed(RuntimeError):
    """标定无法继续（帧数不足、角点全丢等），消息是给人看的中文原因。"""


class Cancelled(RuntimeError):
    """用户点了「取消」。属于正常流程，界面上不该当错误弹窗。"""


def check_cancel(cancel: Optional["threading.Event"]) -> None:
    """取消标志被置位时抛 Cancelled。

    为什么只在阶段边界检查：cv2.calibrateCamera / stereoCalibrate 是**一次不可分割的
    原生调用**，中途无法打断。所以在"读完角点 / 主计算前后 / 每帧复检之间"设检查点，
    真实效果是：读取与复检阶段可立即停，主计算阶段最多等它算完（通常 1~3 s）。
    """
    if cancel is not None and cancel.is_set():
        raise Cancelled("用户取消了本次计算")


# -----------------------------------------------------------------------------
# 单帧记录
# -----------------------------------------------------------------------------
@dataclass
class CaptureRecord:
    """一条已保存的采集记录。corners 只留在内存里，用于撤销与覆盖统计，不写盘。"""
    idx: int
    left_path: str
    right_path: str
    res: Dict[str, Any]
    pose: Optional[Dict[str, Any]] = None
    stamp: float = field(default_factory=time.time)
    forced: bool = False
    corners: Optional[np.ndarray] = None
    shape: Tuple[int, int] = (480, 640)      # (高, 宽)，覆盖统计要用真实尺寸

    @property
    def stereo(self) -> bool:
        return bool(self.res.get("found_l") and self.res.get("found_r"))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "left": os.path.basename(self.left_path),
            "right": os.path.basename(self.right_path),
            "stereo": self.stereo,
            "forced": self.forced,
            "px_per_square": round(float(self.res.get("px_per_square", 0.0)), 2),
            "sharpness": round(float(self.res.get("sharpness", 0.0)), 1),
            "contrast_min": round(float(self.res.get("pattern_contrast_min", 0.0)), 1),
            "pose": self.pose or {},
            "time": time.strftime("%H:%M:%S", time.localtime(self.stamp)),
        }


# -----------------------------------------------------------------------------
# 采集会话
# -----------------------------------------------------------------------------
class CaptureSession:
    """一次标定采集 = 一个目录 + 一组记录 + 一张覆盖热力图。

    用法（GUI 里就是这样调的）：
        s = CaptureSession(spec, target=26)
        res, pose, reasons, dup = s.analyze(left, right)
        if not reasons and not dup:
            s.record(left, right, res, pose)
        s.finalize()          # 写 capture_report.txt / .json
    """

    GRID = (12, 16)           # 覆盖统计网格（行, 列）

    def __init__(self, spec: Any, out_root: str = "", target: int = 26,
                 thr: Optional[Dict[str, float]] = None,
                 checklist: Optional[Dict[str, int]] = None,
                 session_dir: str = "") -> None:
        self.spec = spec
        self.target = int(target)
        self.thr = dict(DEFAULT_THR)
        self.thr.update(thr or {})
        self.checklist = dict(DEFAULT_CHECKLIST)
        self.checklist.update(checklist or {})
        self.out_root = out_root or CAPTURE_ROOT
        self.dir = session_dir or os.path.join(self.out_root, time.strftime("%Y%m%d_%H%M%S"))
        self.records: List[CaptureRecord] = []
        self._sigs: set = set()
        self._grid = np.zeros(self.GRID, np.int32)
        self._next = 1
        self._finalized = False

    # -------------------------------------------------------------- 分析
    def analyze(self, left: np.ndarray, right: np.ndarray
                ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[str], bool]:
        """返回 (指标, 姿态, 不通过原因列表, 是否姿态重复)。纯计算，不改状态。"""
        cc = _cc()
        res = cc.analyze_pair(left, right, self.spec, strict=True)
        pose = cc.classify_pose(res, left.shape[:2], self.spec)
        reasons = [m[0] for m in cc.gate_messages(res, self.thr)]
        dup = bool(pose) and (cc._pose_signature(pose) in self._sigs)
        return res, pose, reasons, dup

    def gate_reasons(self, res: Dict[str, Any]) -> List[str]:
        return [m[0] for m in _cc().gate_messages(res, self.thr)]

    # -------------------------------------------------------------- 记录
    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def stereo_count(self) -> int:
        return sum(1 for r in self.records if r.stereo)

    def record(self, left: np.ndarray, right: np.ndarray, res: Dict[str, Any],
               pose: Optional[Dict[str, Any]] = None, force: bool = False) -> CaptureRecord:
        """保存一帧。不达标时抛 CaptureRejected（force=True 可跳过质检门与去重）。"""
        if not force:
            reasons = self.gate_reasons(res)
            if reasons:
                raise CaptureRejected("；".join(reasons))
            if pose is not None and _cc()._pose_signature(pose) in self._sigs:
                raise CaptureRejected("这张与之前某张姿态过于相似，换位置或换倾斜方向")

        os.makedirs(self.dir, exist_ok=True)
        idx = self._next
        lp = os.path.join(self.dir, f"left_{idx:02d}.png")
        rp = os.path.join(self.dir, f"right_{idx:02d}.png")
        ok_l = cv2.imwrite(lp, left)
        ok_r = cv2.imwrite(rp, right)
        if not (ok_l and ok_r):
            raise RuntimeError(f"写入失败：{lp} / {rp}")

        corners = res.get("corners_l") if res.get("found_l") else res.get("corners_r")
        rec = CaptureRecord(idx=idx, left_path=lp, right_path=rp, res=dict(res),
                            pose=pose, forced=bool(force),
                            corners=None if corners is None else np.asarray(corners, np.float32),
                            shape=(int(left.shape[0]), int(left.shape[1])))
        self.records.append(rec)
        if pose is not None:
            self._sigs.add(_cc()._pose_signature(pose))
        self._accumulate(rec)
        self._next += 1
        return rec

    def undo(self) -> Optional[CaptureRecord]:
        """撤销最后一帧（同时删除它的两张图）。"""
        if not self.records:
            return None
        rec = self.records.pop()
        for p in (rec.left_path, rec.right_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        self._rebuild()
        return rec

    def clear(self) -> int:
        """清空本次会话的全部记录与文件（只动本会话目录）。"""
        n = len(self.records)
        for rec in list(self.records):
            for p in (rec.left_path, rec.right_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
        self.records.clear()
        self._rebuild()
        return n

    def _rebuild(self) -> None:
        """撤销后重算：去重签名、覆盖网格、下一个文件编号。"""
        cc = _cc()
        self._sigs = set()
        self._grid = np.zeros(self.GRID, np.int32)
        for rec in self.records:
            if rec.pose is not None:
                self._sigs.add(cc._pose_signature(rec.pose))
            self._accumulate(rec)
        self._next = (max((r.idx for r in self.records), default=0) + 1)

    def _accumulate(self, rec: CaptureRecord) -> None:
        if rec.corners is None or len(rec.corners) == 0:
            return
        h, w = int(rec.shape[0]), int(rec.shape[1])
        if h <= 0 or w <= 0:
            return
        pts = np.asarray(rec.corners, np.float32).reshape(-1, 2)
        gx = np.clip((pts[:, 0] / float(w) * self.GRID[1]).astype(int), 0, self.GRID[1] - 1)
        gy = np.clip((pts[:, 1] / float(h) * self.GRID[0]).astype(int), 0, self.GRID[0] - 1)
        np.add.at(self._grid, (gy, gx), 1)

    # -------------------------------------------------------------- 统计
    def checklist_progress(self) -> Dict[str, int]:
        """每个清单项当前达成数（键与 self.checklist 一致）。"""
        cc = _cc()
        poses = [r.pose for r in self.records if r.pose]
        out: Dict[str, int] = {}
        for name in self.checklist:
            rule = cc.CHECK_RULES.get(name)
            out[name] = sum(1 for p in poses if rule and rule(p)) if rule else 0
        return out

    def checklist_lines(self) -> List[str]:
        """采集清单进度（"上仰 3/3" 这样的行）。"""
        got = self.checklist_progress()
        lines: List[str] = []
        for name, need in self.checklist.items():
            mark = "[OK]" if got.get(name, 0) >= need else "[  ]"
            lines.append(f"{mark} {name:<12} {got.get(name, 0)}/{need}")
        return lines

    def shortage(self) -> Dict[str, Any]:
        """还差多少张、缺哪类姿态。返回 shortage_plan() 的结果，供界面直接显示。"""
        return shortage_plan(self.count, self.target,
                             self.checklist_progress(), self.checklist)

    def grid_counts(self) -> np.ndarray:
        return self._grid.copy()

    # -------------------------------------------------------------- 删除
    def remove(self, rec: "CaptureRecord") -> bool:
        """删除指定的某一帧（同时删除它的两张图），并重算去重签名/覆盖网格。

        与 undo() 的区别：undo 只能删最后一帧，remove 可以删列表中任意一帧。
        返回是否真的删掉了（记录不存在时返回 False）。
        """
        for i, r in enumerate(self.records):
            if r is rec or r.idx == rec.idx:
                del self.records[i]
                for p in (r.left_path, r.right_path):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass
                self._rebuild()
                return True
        return False

    def frame_rows(self) -> List[Tuple[int, str, int]]:
        """给界面用的帧清单：(序号, 姿态摘要, 角点数)。"""
        region_cn = {"center": "中心", "tl": "左上", "tr": "右上", "bl": "左下", "br": "右下"}
        tilt_cn = {"none": "正对", "left": "左倾", "right": "右倾", "up": "上仰", "down": "下俯"}
        out: List[Tuple[int, str, int]] = []
        for r in self.records:
            pose = r.pose or {}
            desc = f"{region_cn.get(str(pose.get('region')), '?')}/" \
                   f"{tilt_cn.get(str(pose.get('tilt_dir')), '?')}" \
                   f"{float(pose.get('tilt_deg', 0.0)):.0f}°"
            res = r.res if isinstance(r.res, dict) else {}
            c = res.get("corners_l") if res.get("found_l") else res.get("corners_r")
            n = 0 if c is None else int(len(c))
            out.append((int(r.idx), desc, n))
        return out

    def coverage_ratio(self, min_hits: int = 1) -> float:
        """被至少 min_hits 个角点覆盖过的格子占比（0~1）。"""
        return float((self._grid >= int(min_hits)).mean())

    def metrics_table(self) -> str:
        """采集指标表（报告与界面共用）。"""
        lines = [
            f"会话目录      : {self.dir}",
            f"标定板        : {self.spec.describe()}",
            f"已保存        : {self.count} 对（目标 {self.target}），其中双眼可用 {self.stereo_count} 对",
            f"覆盖格子      : {self.coverage_ratio() * 100:.1f}%（{self.GRID[0]}x{self.GRID[1]} 网格）",
            "",
            "采集清单：",
        ]
        lines += ["  " + x for x in self.checklist_lines()]
        plan = self.shortage()
        lines.append("")
        lines.append(f"进度：{plan['text']}")
        if plan["advice"]:
            lines.append("下一步：" + "；".join(plan["advice"]))
        if self.records:
            pxs = [r.res.get("px_per_square", 0.0) for r in self.records]
            shp = [r.res.get("sharpness", 0.0) for r in self.records]
            con = [r.res.get("pattern_contrast_min", 0.0) for r in self.records]
            lines += [
                "",
                f"每格像素      : {min(pxs):.1f} ~ {max(pxs):.1f}",
                f"清晰度        : {min(shp):.0f} ~ {max(shp):.0f}",
                f"最差区域对比度: {min(con):.0f} ~ {max(con):.0f}",
            ]
            single = [r.idx for r in self.records if not r.stereo]
            if single:
                lines.append(f"仅单眼有效    : {single}（只用于该眼内参）")
            forced = [r.idx for r in self.records if r.forced]
            if forced:
                lines.append(f"强制保存      : {forced}（未过质检门，建议复查）")
        return "\n".join(lines)

    def finalize(self) -> Dict[str, str]:
        """写报告（txt + json）。可重复调用，返回两种格式的路径。"""
        if not self.records:
            raise CalibrationFailed("还没有采集任何一帧，无法生成报告")
        os.makedirs(self.dir, exist_ok=True)
        txt_path = os.path.join(self.dir, "capture_report.txt")
        json_path = os.path.join(self.dir, "capture_report.json")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(f"标定采集报告  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write("=" * 70 + "\n")
            fh.write(self.metrics_table() + "\n\n")
            fh.write("-" * 70 + "\n逐帧明细：\n")
            for rec in self.records:
                d = rec.as_dict()
                fh.write(f"  #{d['idx']:02d} {'双眼' if d['stereo'] else '单眼'} "
                         f"px/sq={d['px_per_square']:5.1f} sharp={d['sharpness']:5.0f} "
                         f"contrast={d['contrast_min']:4.0f} "
                         f"pose={d['pose'].get('region', '-')}/{d['pose'].get('tilt_dir', '-')}"
                         f"/{d['pose'].get('tilt_deg', 0):.0f}deg"
                         f"{'  [强制]' if d['forced'] else ''}\n")
        payload = {
            "version": 1,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dir": self.dir,
            "board": self.spec.as_dict() if hasattr(self.spec, "as_dict") else {},
            "board_text": self.spec.describe(),
            "target": self.target,
            "thresholds": self.thr,
            "checklist": self.checklist,
            "count": self.count,
            "stereo_count": self.stereo_count,
            "coverage_ratio": self.coverage_ratio(),
            "grid": self.GRID,
            "records": [r.as_dict() for r in self.records],
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        self._finalized = True
        return {"txt": txt_path, "json": json_path}


# -----------------------------------------------------------------------------
# 从采集目录计算标定
# -----------------------------------------------------------------------------
def image_size_of(frames_dir: str, pattern: str = "left_*.png") -> Tuple[int, int]:
    """读采集目录里第一张图的真实尺寸，返回 (宽, 高)。

    为什么要这一步：旧版 run_calibration.py 写死 (640, 480)，
    换分辨率重标时会用错 image_size，标定结果与实际图像不匹配且不报错。
    """
    paths = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return int(img.shape[1]), int(img.shape[0])
    raise CalibrationFailed(f"目录里没有可读的图像：{frames_dir}")


def list_capture_dirs(root: str = "") -> List[Dict[str, Any]]:
    """列出采集目录（按时间倒序），供 GUI 下拉框使用。"""
    root = root or CAPTURE_ROOT
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        n_l = len(glob.glob(os.path.join(p, "left_*.png")))
        n_r = len(glob.glob(os.path.join(p, "right_*.png")))
        if n_l == 0 and n_r == 0:
            continue
        out.append({"dir": p, "name": name, "pairs": min(n_l, n_r),
                    "mtime": os.path.getmtime(p)})
    out.sort(key=lambda d: -d["mtime"])
    return out


def calibrate_from_dir(frames_dir: str, spec: Any, fix_intrinsic: bool = True,
                       progress: ProgressFn = None, verify: bool = True,
                       cancel: Optional[threading.Event] = None) -> Dict[str, Any]:
    """从采集目录算出 StereoCalibration，并做校正验证。

    **这个函数会阻塞数秒到数十秒，GUI 必须放到后台线程调用。**

    输入：
        frames_dir     : 含 left_XX.png / right_XX.png 的目录
        spec           : calibration.BoardSpec（决定物点）
        fix_intrinsic  : True = 各眼先单独标内参，再在双目中固定（推荐）
        progress       : 进度回调 (阶段中文, 细节)，可为 None
        verify         : 是否做校正后极线复检
        cancel         : threading.Event，置位后在下一个阶段边界抛 Cancelled
    输出：
        {"calib", "frames_left", "frames_right", "frames_stereo", "rows",
         "image_size", "elapsed_s", "post_rows", "pre_dy"}
    异常：
        CalibrationFailed —— 帧数不足、目录无效等（消息可直接显示给用户）
        Cancelled         —— 用户取消（不是错误，界面上不要当失败弹窗）
    """
    cal = _cal()
    if not frames_dir or not os.path.isdir(frames_dir):
        raise CalibrationFailed(f"采集目录不存在：{frames_dir}")

    t0 = time.perf_counter()
    check_cancel(cancel)
    notify(progress, "读取角点", f"目录 {os.path.basename(frames_dir)}")
    left_pts, right_pts, rows = cal.load_frame_points(frames_dir, spec, refine=True)
    check_cancel(cancel)
    both = sorted(set(left_pts) & set(right_pts))
    notify(progress, "读取角点",
           f"左 {len(left_pts)} 帧 / 右 {len(right_pts)} 帧 / 双眼 {len(both)} 帧")

    if len(both) < 4:
        raise CalibrationFailed(
            f"双眼都有效的帧只有 {len(both)} 帧（至少 4，建议 ≥15）。"
            "请确认采集时左右眼都能看到标定板，且板子没有超出画面。")

    image_size = image_size_of(frames_dir)
    notify(progress, "标定计算", f"图像尺寸 {image_size[0]}x{image_size[1]}，内参策略 "
                                 f"{'固定' if fix_intrinsic else '联合优化'}"
                                 "（这一步内部不可中断，通常 1~3 s）")
    try:
        calib = cal.calibrate_stereo(
            left_pts, right_pts, spec, image_size,
            fix_intrinsic=bool(fix_intrinsic),
            extra_meta={"capture_dir": os.path.basename(frames_dir),
                        "source": "calib_gui"})
    except Exception as exc:                # noqa: BLE001
        raise CalibrationFailed(f"标定计算失败：{type(exc).__name__}: {exc}") from exc
    check_cancel(cancel)

    post_rows: List[Dict[str, Any]] = []
    pre_dy: List[float] = []
    if verify:
        notify(progress, "校正验证", "把图像校正后重新检测角点，测极线失准 |dy|")
        try:
            post_rows = cal.epipolar_error(calib, frames_dir, spec, frames=both)
        except Exception as exc:            # noqa: BLE001
            post_rows = [{"idx": -1, "found": False, "error": f"{type(exc).__name__}: {exc}"}]
        check_cancel(cancel)
        pre_dy = _pre_rectify_dy(frames_dir, spec, both, cancel=cancel)

    return {
        "calib": calib,
        "frames_left": len(left_pts),
        "frames_right": len(right_pts),
        "frames_stereo": len(both),
        "rows": rows,
        "image_size": image_size,
        "elapsed_s": time.perf_counter() - t0,
        "post_rows": post_rows,
        "pre_dy": pre_dy,
        "left_pts": left_pts,
        "right_pts": right_pts,
    }


def _pre_rectify_dy(frames_dir: str, spec: Any,
                    frames: Sequence[int],
                    cancel: Optional[threading.Event] = None) -> List[float]:
    """校正**前**的 |dy|（对照用）：同一批帧直接测左右同一角点的垂直差。"""
    cal = _cal()
    out: List[float] = []
    for i in frames:
        check_cancel(cancel)                # 逐帧可中断：这里是最慢的一段
        lp = os.path.join(frames_dir, f"left_{i:02d}.png")
        rp = os.path.join(frames_dir, f"right_{i:02d}.png")
        L = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if L is None or R is None:
            continue
        cl = cal.detect_corners_strict(L, spec.pattern_size)
        cr = cal.detect_corners_strict(R, spec.pattern_size)
        if cl is None or cr is None:
            continue
        a = np.asarray(cl, np.float32).reshape(-1, 2)
        b = np.asarray(cr, np.float32).reshape(-1, 2)
        if a.shape != b.shape or len(a) == 0:
            continue
        out.append(float(np.abs(a[:, 1] - b[:, 1]).mean()))
    return out


def summarize_post(post_rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """把 epipolar_error 的输出压成几个数字，界面与报告都用它。"""
    found = [r for r in post_rows if r.get("found")]
    if not found:
        return {"n": 0, "dy_mean": float("nan"), "dy_max": float("nan"),
                "dx_mean": float("nan")}
    dy = np.array([r["dy_mean_px"] for r in found], np.float64)
    dymax = np.array([r["dy_max_px"] for r in found], np.float64)
    dx = np.array([r["dx_mean_px"] for r in found], np.float64)
    return {"n": int(len(found)), "dy_mean": float(dy.mean()),
            "dy_max": float(dymax.max()), "dx_mean": float(dx.mean())}


# -----------------------------------------------------------------------------
# 标定结果的"体检"（rms 之外还要看什么）
# -----------------------------------------------------------------------------
@dataclass
class CheckItem:
    """一条体检结论。level: ok / warn / bad；advice 是给人看的下一步。"""
    name: str
    value: str
    level: str
    advice: str = ""

    @property
    def ok(self) -> bool:
        return self.level == "ok"


def validate(calib: Any, image_size: Tuple[int, int],
             post_rows: Sequence[Dict[str, Any]] = (),
             expect_baseline_mm: Optional[float] = None,
             expect_fx: Optional[float] = None) -> List[CheckItem]:
    """对标定结果做一组独立判据检查，返回 CheckItem 列表。

    这些判据不替代 RMS，而是抓 RMS 抓不到的问题：
        基线单位错（米/毫米混用）、T 方向反、图像尺寸不匹配、
        畸变系数发散、光轴夹角过大、极线复检没通过。
    """
    items: List[CheckItem] = []
    rms = float(getattr(calib, "rms_stereo", 0.0))
    items.append(CheckItem(
        "双目 RMS", f"{rms:.4f} px",
        "ok" if rms < 0.3 else "warn" if rms < 1.0 else "bad",
        "正常，无需处理" if rms < 0.3 else
        "偏大：检查是否有运动模糊帧，或用逐帧误差列表剔除最差的几帧"))

    rl, rr = float(calib.rms_left), float(calib.rms_right)
    worst = max(rl, rr)
    items.append(CheckItem(
        "单眼 RMS", f"左 {rl:.4f} / 右 {rr:.4f} px",
        "ok" if worst < 0.3 else "warn" if worst < 1.0 else "bad",
        "两眼均衡" if worst < 0.3 else
        ("右眼偏差较大：检查右眼是否有反光或对焦差异" if rr > rl else
         "左眼偏差较大：检查左眼是否有反光或对焦差异")))

    b_mm = float(calib.baseline_mm)
    lvl = "ok"
    adv = "与标称 65 mm 同量级"
    if b_mm < 20 or b_mm > 300:
        lvl, adv = "bad", "量级异常：很可能是单位或物点尺寸填错（应为毫米量级）"
    elif expect_baseline_mm:
        dev = abs(b_mm - expect_baseline_mm) / expect_baseline_mm * 100.0
        lvl = "ok" if dev < 3.0 else "warn" if dev < 8.0 else "bad"
        adv = (f"与配置标称 {expect_baseline_mm:.3f} mm 相差 {dev:.2f}%" +
               ("，在打印/装配误差范围内" if dev < 3.0 else
                "，建议核对方块实际边长与标称是否一致" if dev < 8.0 else
                "，差异过大：优先怀疑标定板实际尺寸与填写的 square_mm 不符"))
    items.append(CheckItem("基线", f"{b_mm:.3f} mm（{calib.baseline_m:.6f} m）", lvl, adv))

    ang = float(calib.axis_angle_deg)
    items.append(CheckItem(
        "两眼光轴夹角", f"{ang:.3f} 度",
        "ok" if ang < 3.0 else "warn" if ang < 8.0 else "bad",
        "接近平行，符合刚性模组预期" if ang < 3.0 else
        "偏大：确认采集时标定板是否覆盖了画面四周"))

    t = np.asarray(calib.T, np.float64).reshape(3)
    sign_ok = t[0] < 0
    items.append(CheckItem(
        "T 方向", f"Tx={t[0]:.3f} mm",
        "ok" if sign_ok else "warn",
        "左相机到右相机的平移为 -x，方向正确" if sign_ok else
        "Tx 为正：若 confirm 过左右眼顺序（config.camera.split）会得到反向的物理基线"))

    same_size = (int(calib.image_size[0]) == int(image_size[0])
                 and int(calib.image_size[1]) == int(image_size[1]))
    items.append(CheckItem(
        "图像尺寸", f"标定 {calib.image_size[0]}x{calib.image_size[1]} / "
                    f"实际 {image_size[0]}x{image_size[1]}",
        "ok" if same_size else "bad",
        "一致" if same_size else "不一致：这份标定不能用于当前分辨率，必须重标"))

    d = np.asarray(calib.D1, np.float64).reshape(-1)
    dmax = float(np.max(np.abs(d))) if d.size else 0.0
    items.append(CheckItem(
        "畸变系数量级", f"max|D1|={dmax:.3f}",
        "ok" if dmax < 5.0 else "warn",
        "在正常范围" if dmax < 5.0 else
        "偏大：鱼眼视场用针孔模型时边缘畸变系数会发散，属于已知限制"))

    # 校正缩放比：P1.fx / K1.fx 明显 >1 说明 stereoRectify 为了"无黑边"大幅放大画面，
    # 这是畸变系数在边缘失稳的直接后果（标定板只覆盖画面中央时最常出现）。
    try:
        zoom = float(calib.rect_fx) / max(float(calib.K1[0, 0]), 1e-9)
    except Exception:                       # noqa: BLE001
        zoom = float("nan")
    if np.isfinite(zoom):
        items.append(CheckItem(
            "校正缩放比", f"P1.fx/K1.fx = {zoom:.3f}",
            "ok" if zoom < 1.5 else "warn" if zoom < 2.5 else "bad",
            "正常（裁边幅度可接受）" if zoom < 1.5 else
            "画面被放大后裁切：说明畸变在画面四周失稳。常见原因是标定板只覆盖了中央区域，"
            "边缘没有约束。建议扩大采集覆盖（四角都要拍到），或改用鱼眼模型重标；"
            "缩放过大会把有效视场裁掉一大块，深度图的可用范围也随之缩小"))

    s = summarize_post(post_rows)
    if s["n"]:
        lvl = "ok" if s["dy_mean"] < 0.5 else "warn" if s["dy_mean"] < 2.0 else "bad"
        items.append(CheckItem(
            "校正后极线失准", f"均值 {s['dy_mean']:.3f} px / 最大 {s['dy_max']:.3f} px"
                              f"（{s['n']} 帧）",
            lvl,
            "行已对齐" if lvl == "ok" else
            "未对齐：优先检查最差帧（逐帧误差），再考虑增加四周覆盖"))
    else:
        items.append(CheckItem("校正后极线失准", "未检出（复检失败）", "warn",
                               "校正后连角点都检不到，检查映射表与图像是否同尺寸"))
    return items


def validate_text(items: Sequence[CheckItem]) -> str:
    """把检查结果排成可直接显示的多行文本。"""
    icon = {"ok": "[OK]  ", "warn": "[WARN]", "bad": "[BAD] "}
    lines = []
    for it in items:
        head = f"{icon.get(it.level, '[?]   ')} {it.name:<12} {it.value}"
        lines.append(head)
        if it.advice and it.level != "ok":
            lines.append(f"        └─ {it.advice}")
    bad = sum(1 for i in items if i.level == "bad")
    warn = sum(1 for i in items if i.level == "warn")
    lines.append("")
    lines.append(f"结论：{bad} 项不可用，{warn} 项需注意，其余正常")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 命令行（不开 GUI 也能用：检查一个目录能否标定、标完打印体检表）
# -----------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="标定核心：无界面自检/计算")
    ap.add_argument("--dir", default="", help="采集目录（默认取最新）")
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--square-mm", type=float, default=3.0)
    ap.add_argument("--out", default="", help="标定文件输出路径（默认只算不存）")
    ap.add_argument("--list", action="store_true", help="列出所有采集目录")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for d in list_capture_dirs():
            print(f"{d['name']}  {d['pairs']} 对  {d['dir']}")
        return 0

    spec = default_spec(args.cols, args.rows, args.square_mm)
    frames_dir = args.dir
    if not frames_dir:
        ds = list_capture_dirs()
        if not ds:
            print(f"[错误] {CAPTURE_ROOT} 下没有采集目录")
            return 1
        frames_dir = ds[0]["dir"]

    def prog(stage: str, detail: str) -> None:
        print(f"  [{stage}] {detail}")

    try:
        out = calibrate_from_dir(frames_dir, spec, progress=prog,
                                 verify=not args.no_verify)
    except CalibrationFailed as exc:
        print(f"[失败] {exc}")
        return 2

    calib = out["calib"]
    print(calib.summary())
    print()
    print(validate_text(validate(calib, out["image_size"], out["post_rows"],
                                 expect_baseline_mm=65.0)))
    if args.out:
        calib.save(args.out)
        print(f"\n已保存：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
