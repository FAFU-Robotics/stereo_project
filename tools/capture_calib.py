"""capture_calib.py —— 标定图采集 与 标定板可行性预检（阶段 5）
=============================================================================
本文件目前实现【步骤 0：可行性预检】。之后会继续加入【步骤 1：正式采集】。

为什么先做预检（不直接写标定算法）：
    用户的标定板是 DFOPTIX Q12-50-3：图案区只有 **36 x 27 mm**（方格 3.0 mm）。
    要让角点定位准，图案在画面里必须占足够多的像素 —— 粗算需要拍到 **10~20 cm**。
    而这台相机是**定焦**的，之前实测 7 cm 处清晰度只有 14（很糊）。
    所以"近到什么距离还能拍清楚"是本次标定的**最大风险**，
    必须先用实测数据确认可用距离区间，再决定后面怎么写采集与标定。

两种运行模式：
    --probe     交互式预检：把板依次放到指定距离，按空格各拍一帧，
                程序对每帧给出「能否检出角点 / 图案像素尺寸 / 清晰度 / 反光比例」
                并给出该距离是否可用的判定与总结表。
    --selftest  自检（不需要相机）：用合成棋盘格验证分析逻辑本身是否正确，
                并顺便测出"每格至少多少像素才能稳定检出"，用于校准判据阈值。

用法：
    python tools\\capture_calib.py --selftest                 REM 先验证工具本身（无需相机）
    python tools\\capture_calib.py --probe                    REM 提权运行：交互式预检
    python tools\\capture_calib.py --probe --distances 10,12,15,18,20,25,30
    python tools\\capture_calib.py --probe --res 2560x960     REM 换更高分辨率试试

输出的证据（便于事后核对，也会写一份文本报告）：
    output\\calib_probe\\<时间戳>\\d<距离>cm_left.png / _right.png / _overlay.png
    output\\calib_probe\\<时间戳>\\probe_report.txt
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camera import CameraError, CameraReadError, StereoCamera, frame_sharpness  # noqa: E402
from viewer import StereoViewer                                                 # noqa: E402

# -----------------------------------------------------------------------------
# 判据阈值（都可用命令行覆盖；来源见 selftest 的实测）
# -----------------------------------------------------------------------------
SHARP_GOOD = 100.0      # 清晰度 >= 100 视为良好（与项目其它模块一致）
SHARP_MIN = 30.0        # < 30 视为过糊、角点与立体匹配都不可靠
CONTRAST_GOOD = 180.0   # 图案区内 p95-p5 >= 180 视为对比度良好
CONTRAST_MIN = 120.0    # < 120 视为对比度不足（多半是反光把黑格洗白了）
PX_PER_SQUARE_MIN = 10.0   # 每格最少像素（低于此角点定位误差会明显变大）
PX_PER_SQUARE_GOOD = 15.0  # 每格 >= 15 px 视为优秀


@dataclass
class BoardSpec:
    """标定板规格（DFOPTIX Q12-50-3）。

    输入：
        cols/rows : **方格**数（12 x 9）
        square_mm : 单个方格边长（毫米）—— 这个数字直接决定深度的绝对尺度
    派生：
        pattern_size : OpenCV 要的**内角点**数 = (cols-1, rows-1) = (11, 8)
    """
    cols: int = 12
    rows: int = 9
    square_mm: float = 3.0

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return (self.cols - 1, self.rows - 1)

    def candidates(self) -> List[Tuple[int, int]]:
        """候选内角点尺寸列表（自动尝试，避免我数错）。

        第一个是规格推出的 (11,8)，其余是常见误读/旋转后的尺寸。
        """
        base = self.pattern_size
        cands = [base, (base[1], base[0]), (self.cols, self.rows), (self.rows, self.cols),
                 (base[0] - 1, base[1]), (base[0] + 1, base[1]),
                 (base[0], base[1] - 1), (base[0], base[1] + 1)]
        seen, out = set(), []
        for c in cands:
            if c[0] > 2 and c[1] > 2 and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def object_points(self) -> np.ndarray:
        """标定用的三维物点（单位：与 square_mm 相同，这里是毫米）。"""
        pts = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        pts[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        pts *= float(self.square_mm)
        return pts

    def describe(self) -> str:
        return (f"{self.cols}x{self.rows} 方格, 方格边长 {self.square_mm} mm, "
                f"图案 {self.cols * self.square_mm:g}x{self.rows * self.square_mm:g} mm, "
                f"内角点 {self.pattern_size[0]}x{self.pattern_size[1]}")

    def describe_ascii(self) -> str:
        """英文/ASCII 版说明 —— 画到图像上的文字必须是 ASCII（cv2.putText 画不出中文）。"""
        return (f"{self.cols}x{self.rows} squares, {self.square_mm:g} mm each, "
                f"pattern {self.cols * self.square_mm:g}x{self.rows * self.square_mm:g} mm, "
                f"inner corners {self.pattern_size[0]}x{self.pattern_size[1]}")


# -----------------------------------------------------------------------------
# 角点检测与分析
# -----------------------------------------------------------------------------
def _worst_tile_contrast(roi: np.ndarray, tiles: int = 2) -> float:
    """把图案区切成 tiles×tiles 块，返回**最差**一块的对比度(p95-p5)。

    为什么要分块取最差：整块的 p95-p5 对"**局部**反光"不敏感 ——
    光斑只覆盖一部分时，最暗的 5% 仍来自没被影响的黑格，整块对比度看着还很好
    （早期版本就是这样，被自检抓出来了）。
    分块后，只要有一块被反光洗白，这里就会掉下来。
    tiles=2 时每块约覆盖 6x4.5 个方格，保证块内黑白两种格子都存在，
    不会出现"某块全是黑色导致对比度为 0"的误判。
    """
    h, w = roi.shape[:2]
    vals: List[float] = []
    for i in range(tiles):
        for j in range(tiles):
            y0, y1 = i * h // tiles, (i + 1) * h // tiles
            x0, x1 = j * w // tiles, (j + 1) * w // tiles
            t = roi[y0:y1, x0:x1]
            if t.size < 16:
                continue
            vals.append(float(np.percentile(t, 95) - np.percentile(t, 5)))
    return min(vals) if vals else 0.0


def detect_corners(gray: np.ndarray, spec: BoardSpec, refine: bool = True, strict: bool = False
                   ) -> Tuple[bool, Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """检测棋盘角点，返回 (是否检出, 角点(N,1,2) float32, 命中的尺寸)。

    strict=True  —— **只认规格推出的那一个尺寸**（标定/采集必须用这个！）
    strict=False —— 依次尝试多个候选尺寸（只适合"还不知道板子规格"的探测场景）

    ⚠️ 为什么标定必须 strict（实测踩过）：
        候选搜索会"退而求其次"匹配到更小的子棋盘。用户采集的 26 帧里，
        第 20/21 帧的**右眼被匹配成 (10,8)=80 个角点**（正确是 (11,8)=88 个），
        而物点仍按 (11,8) 生成 —— 这种尺寸不一致会**静默污染标定参数**，
        而且在采集报告里还会被误标成"双眼可用"。严格模式下它只会"检不到"，
        由质量门正常拦住并标为单眼帧。

    refine=True 时做亚像素精化（cornerSubPix）—— 正式标定必须开。

    注意（实测踩过）：OpenCV 5.0 的 findChessboardCorners 返回的角点形状是 (N,2)，
    而老版本是 (N,1,2)。这里统一 reshape 成 (N,1,2)，
    否则后续 `corners[:,0,0]` 之类的索引会直接抛 IndexError。
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    sizes = [spec.pattern_size] if strict else spec.candidates()
    for size in sizes:
        ok, corners = cv2.findChessboardCorners(gray, size, flags)
        if not ok:
            continue
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        if refine:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        return True, corners, size
    return False, None, None


def analyze_pair(left_bgr: np.ndarray, right_bgr: np.ndarray, spec: BoardSpec,
                 strict: bool = False) -> Dict[str, Any]:
    """对一对左右图做分析，返回指标字典（纯计算，不显示、不保存）。

    strict=True 时角点检测**只认规格推出的那一个尺寸**（采集/标定必须用），
    否则会退而求其次匹配到更小的子棋盘、静默污染标定数据（详见 detect_corners 的说明）。

    指标：
        found_l/found_r : 左右是否检出角点
        size_l/size_r   : 命中的内角点尺寸
        bbox_w/h        : 角点网格外接框（像素）
        px_per_square   : 每格多少像素（角点网格宽 / (内角点列数-1)）
        sharpness       : 左图清晰度（拉普拉斯方差）
        pattern_contrast: **图案区域内**的对比度 p95-p5
                          —— 这是"反光有没有把黑格洗白"的真正指标
        overexposed_pct : 全图 >=250 像素占比（**仅供参考**：棋盘格本来就有大量白色，
                          所以它不能单独用来判断反光，早期版本误用它当判据，已被自检抓出）
        mean/std        : 左图灰度统计
        verdict/level   : 人话判定与等级(ok/warn/bad)
    """
    gl = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY) if left_bgr.ndim == 3 else left_bgr
    gr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY) if right_bgr.ndim == 3 else right_bgr

    found_l, c_l, size_l = detect_corners(gl, spec, strict=strict)
    found_r, c_r, size_r = detect_corners(gr, spec, strict=strict)

    bbox_w = bbox_h = px_per_square = 0.0
    pattern_contrast = 0.0
    pattern_contrast_min = 0.0
    # 优先用左眼算尺寸；若只有右眼检出，就用右眼（否则"只有右眼"的帧会显示每格 0 px）
    src_corners, src_size = (c_l, size_l) if found_l else (c_r, size_r)
    if src_corners is not None:
        pts = np.asarray(src_corners, dtype=np.float32).reshape(-1, 2)   # 兼容 (N,1,2) 与 (N,2)
        xs, ys = pts[:, 0], pts[:, 1]
        bbox_w = float(xs.max() - xs.min())
        bbox_h = float(ys.max() - ys.min())
        cols_minus_1 = max((src_size[0] - 1) if src_size else (spec.pattern_size[0] - 1), 1)
        px_per_square = bbox_w / cols_minus_1
        # 角点外接框内再各留半格余量，取该区域作为"图案区"
        pad = max(int(px_per_square * 0.5), 2)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, gl.shape[1])
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, gl.shape[0])
        if x1 - x0 > 8 and y1 - y0 > 8:
            roi = gl[y0:y1, x0:x1]
            pattern_contrast = float(np.percentile(roi, 95) - np.percentile(roi, 5))
            pattern_contrast_min = _worst_tile_contrast(roi, tiles=2)

    sharp = frame_sharpness(left_bgr)
    overexp = float((gl >= 250).mean()) * 100.0

    # ---- 人话判定（按优先级；对比度用"最差一块"来判定，对局部反光更敏感） ----
    if not found_l or not found_r:
        which = ("左" if not found_l else "") + ("右" if not found_r else "")
        level, verdict = "bad", f"❌ 角点检不到（{which}眼）——太糊/太小/反光/太斜"
    elif sharp < SHARP_MIN:
        level, verdict = "bad", f"❌ 过糊（清晰度 {sharp:.0f} < {SHARP_MIN:.0f}）"
    elif pattern_contrast_min < CONTRAST_MIN:
        level, verdict = "bad", (f"❌ 图案对比度不足（最差区域 {pattern_contrast_min:.0f} "
                                 f"< {CONTRAST_MIN:.0f}）——多半是反光把黑格洗白了")
    elif px_per_square < PX_PER_SQUARE_MIN:
        level, verdict = "bad", f"❌ 图案太小（每格 {px_per_square:.1f} px < {PX_PER_SQUARE_MIN:.0f}）"
    elif (px_per_square < PX_PER_SQUARE_GOOD or sharp < SHARP_GOOD
          or pattern_contrast_min < CONTRAST_GOOD):
        level, verdict = "warn", (f"⚠️ 勉强可用（每格 {px_per_square:.1f} px，清晰度 {sharp:.0f}，"
                                  f"对比度 最差{pattern_contrast_min:.0f}/整体{pattern_contrast:.0f}）")
    else:
        level, verdict = "ok", (f"✅ 可用（每格 {px_per_square:.1f} px，清晰度 {sharp:.0f}，"
                                f"对比度 最差{pattern_contrast_min:.0f}/整体{pattern_contrast:.0f}）")

    return {
        "found_l": found_l, "found_r": found_r,
        "size_l": size_l, "size_r": size_r,
        "corners_l": c_l, "corners_r": c_r,
        "bbox_w": bbox_w, "bbox_h": bbox_h, "px_per_square": px_per_square,
        "sharpness": sharp,
        "pattern_contrast": pattern_contrast,
        "pattern_contrast_min": pattern_contrast_min,
        "overexposed_pct": overexp,
        "mean": float(gl.mean()), "std": float(gl.std()),
        "verdict": verdict, "level": level,
    }


# 等级 -> 纯 ASCII 标签（画到图上的文字不能有中文/emoji：cv2.putText 只支持 ASCII）
LEVEL_TAG = {"ok": "[OK]", "warn": "[WARN]", "bad": "[BAD]"}


def ascii_summary(res: Dict[str, Any]) -> str:
    """把指标压成一行**纯 ASCII** 文字，用于画在图像上。

    为什么需要它：原来直接把中文判定（还带 ⚠️/✅ emoji）画到图上，
    cv2.putText 画不出来，屏幕上只显示 "??"。控制台仍然打印中文版本。
    """
    tag = LEVEL_TAG.get(str(res.get("level")), "[?]")
    parts = [f"{tag} px/sq {res['px_per_square']:.1f}", f"sharp {res['sharpness']:.0f}",
             f"contrast {res['pattern_contrast_min']:.0f}"]
    if not res.get("found_l") or not res.get("found_r"):
        which = ("L" if not res.get("found_l") else "") + ("R" if not res.get("found_r") else "")
        parts.insert(0, f"no corners ({which})")
    return "  ".join(parts)


def draw_overlay(img: np.ndarray, res: Dict[str, Any], spec: BoardSpec,
                 side: str = "left") -> np.ndarray:
    """把角点与判定画到图上（返回副本），用于人工确认。

    side 必须与传入的图像一致！
    实测踩过：一开始这个函数写死用左眼坐标（corners_l），
    却被同时用在右图上，导致右图的叠加整体偏移了"一个视差"
    （10/12/15 cm 处实测视差 276/215/161 px），点全跑到板子外面的手上。
    """
    out = img.copy()
    corners = res["corners_l"] if side == "left" else res["corners_r"]
    size = res["size_l"] if side == "left" else res["size_r"]
    if corners is not None and size is not None:
        cv2.drawChessboardCorners(out, size, np.asarray(corners, np.float32).reshape(-1, 1, 2), True)
    color = {"ok": (0, 255, 0), "warn": (0, 200, 255), "bad": (0, 0, 255)}.get(
        str(res.get("level")), (200, 200, 200))
    StereoViewer._put_text(out, f"{side.upper()}  {ascii_summary(res)}",
                           (10, out.shape[0] - 14), color, 0.6, 2)
    return out


def make_synthetic_board(spec: BoardSpec, px_per_square: float,
                         blur_sigma: float = 0.0, glare: float = 0.0) -> np.ndarray:
    """合成一张标定板图像（**只用于自检**，不参与任何真实标定）。

    棋盘从左上角开始交替；外面留一圈白边模拟基板。
    glare —— 反光覆盖比例（0=无反光，0.35=局部光斑，0.8=大面积泛白）：
        叠加一块**柔和的白色光斑**（模拟玻璃反光），它会把这区域内的黑格抬亮到接近白色，
        从而真的降低图案对比度 —— 这正是真实反光的破坏方式。
    """
    sq = int(round(px_per_square))
    pat_w, pat_h = sq * spec.cols, sq * spec.rows
    margin = int(sq * 2)
    img = np.full((pat_h + 2 * margin, pat_w + 2 * margin), 255, np.uint8)
    for r in range(spec.rows):
        for c in range(spec.cols):
            if (r + c) % 2 == 0:
                y0, x0 = margin + r * sq, margin + c * sq
                img[y0:y0 + sq, x0:x0 + sq] = 0
    if blur_sigma > 0:
        k = int(blur_sigma * 6) | 1
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if glare > 0:
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), np.float32)
        radius = max(6, int(pat_w * float(glare) * 0.75))
        cv2.circle(mask, (w // 2, h // 2), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, sq * 0.8)) * 0.92   # 柔和的 92% 白
        m3 = np.dstack([mask] * 3)
        bgr = (bgr.astype(np.float32) * (1 - m3) + 255.0 * m3).astype(np.uint8)
    return bgr


# -----------------------------------------------------------------------------
# 模式一：自检（不需要相机）—— 验证分析逻辑并校准判据阈值
# -----------------------------------------------------------------------------
def selftest_mode(spec: BoardSpec) -> int:
    print("=" * 88)
    print("  标定板分析逻辑自检（合成图像，不需要相机）")
    print("=" * 88)
    print(f"  板子规格: {spec.describe()}")
    print(f"  候选内角点尺寸: {spec.candidates()}")
    print(f"  判据: 每格 >= {PX_PER_SQUARE_MIN:g} px 且 清晰度 >= {SHARP_MIN:g} 且 图案对比度 >= {CONTRAST_MIN:g}")
    print("-" * 88)
    print(f"  {'每格px':>7} {'检出':>5} {'命中尺寸':>10} {'外接框宽':>9} {'清晰度':>8} {'对比度':>7}  判定")
    ok_count = 0
    fail_bad = 0
    for pps in (4, 6, 8, 10, 12, 15, 20, 30):
        img = make_synthetic_board(spec, pps)
        res = analyze_pair(img, img, spec)
        sz = res["size_l"] if res["found_l"] else "-"
        print(f"  {pps:>7} {'✓' if res['found_l'] else '✗':>5} {str(sz):>10} "
              f"{res['bbox_w']:>9.1f} {res['sharpness']:>8.1f} {res['pattern_contrast']:>7.0f}  {res['verdict']}")
        if res["level"] == "ok":
            ok_count += 1
        if pps >= 15 and res["level"] == "bad":
            fail_bad += 1

    print("-" * 88)
    print("  模糊鲁棒性（固定每格 30 px，逐步加大模糊）:")
    for sigma in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        img = make_synthetic_board(spec, 30, blur_sigma=sigma)
        res = analyze_pair(img, img, spec)
        print(f"    模糊 sigma={sigma:<4} 检出={'✓' if res['found_l'] else '✗'}  "
              f"清晰度={res['sharpness']:>8.1f}  对比度={res['pattern_contrast']:>5.0f}  {res['verdict']}")

    print("-" * 88)
    print("  反光鲁棒性（固定每格 30 px；模拟玻璃反光把黑格洗白）:")
    print("    说明: 局部小光斑通常仍能检出角点（说明它不致命）；"
          "真正要抓的是大面积泛白")
    for glare in (0.0, 0.35, 0.6, 0.8):
        img = make_synthetic_board(spec, 30, glare=glare)
        res = analyze_pair(img, img, spec)
        print(f"    反光覆盖={glare:<5} 检出={'✓' if res['found_l'] else '✗'}  "
              f"对比度 最差{res['pattern_contrast_min']:>4.0f} / 整体{res['pattern_contrast']:>4.0f}  "
              f"{res['verdict']}")

    print("-" * 88)
    print("  真实相机数据不可用时的对照（用项目里已保存的真实图像，应当检不到棋盘）:")
    real = os.path.join(PROJECT_ROOT, "output", "probe", "opencv_check_idx1_dshow.png")
    if os.path.exists(real):
        img = cv2.imread(real)
        res = analyze_pair(img, img, spec)
        print(f"    {os.path.basename(real)}: 检出={'✓' if res['found_l'] else '✗（符合预期）'}  "
              f"清晰度={res['sharpness']:.1f}")
    else:
        print("    （找不到对照图，跳过）")

    print("=" * 88)
    ok = (ok_count >= 3) and (fail_bad == 0)
    print(f"  结论: {'✅ 分析逻辑正常（能检出合格图、能拒绝不合格图）' if ok else '❌ 判据异常，需要检查'}")
    return 0 if ok else 1


# -----------------------------------------------------------------------------
# 模式二：交互式预检（需要相机；普通权限即可）
# -----------------------------------------------------------------------------
def probe_mode(cfg: Dict[str, Any], args: argparse.Namespace, spec: BoardSpec) -> int:
    distances = [float(x) for x in str(args.distances).split(",") if x.strip()]
    out_root = os.path.join(PROJECT_ROOT, "output", "calib_probe")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, stamp)

    runtime = cfg.get("runtime", {})
    viewer = StereoViewer(window_name="capture_calib (probe)",
                          display_scale=float(args.display_scale or runtime.get("display_scale", 1.0)))

    print("=" * 88)
    print("  标定板可行性预检（步骤 0）")
    print("=" * 88)
    print(f"  板子规格  : {spec.describe()}")
    print(f"  距离序列  : {', '.join(f'{d:g}cm' for d in distances)}")
    print(f"  判据      : 每格 >= {PX_PER_SQUARE_MIN:g} px、清晰度 >= {SHARP_MIN:g}、"
          f"图案最差区域对比度 >= {CONTRAST_MIN:g}")
    print("  操作      : 把板正对相机放在指定距离 → 按【空格】拍一帧 → 程序给出判定")
    print("             按 q 或 ESC 提前结束；拍完会打印总结表并保存证据")
    print("-" * 88)

    cam = StereoCamera(cfg.get("camera", {}))
    rows: List[Dict[str, Any]] = []
    quit_all = False
    try:
        cam.open()
        print(f"  相机已打开: {cam.describe()}  单眼 {cam.eye_size[0]}x{cam.eye_size[1]}")
        os.makedirs(out_dir, exist_ok=True)

        for dist in distances:
            if quit_all:
                break
            print(f"\n  >>> 请把标定板放在【{dist:g} cm】处、正对相机、亮度正常，然后按空格…")
            captured = False
            while not captured:
                try:
                    pair = cam.read()
                except CameraReadError as exc:
                    print(f"  [读取错误] {exc}")
                    quit_all = True
                    break
                # 预检阶段**不**用 strict：目的正是要"发现板子是多少格的"，
                # 所以候选择尺寸搜索保留在这里（采集/标定才必须 strict）
                res = analyze_pair(pair.left, pair.right, spec)
                status = [
                    f"target distance: {dist:g} cm   |  {spec.describe_ascii()}",
                    f"corners L/R: {'OK' if res['found_l'] else 'no'} / {'OK' if res['found_r'] else 'no'}   "
                    f"contrast worst/overall: {res['pattern_contrast_min']:.0f}/{res['pattern_contrast']:.0f}",
                    f"px per square: {res['px_per_square']:.1f}   sharpness: {res['sharpness']:.0f}   "
                    f"mean/std: {res['mean']:.0f}/{res['std']:.0f}",
                    ascii_summary(res),      # 纯 ASCII：原来直接画中文判定，图上会显示成 ??
                ]
                overlay = draw_overlay(pair.left, res, spec, side="left")
                panels = [("LEFT + corners", overlay, []), ("RIGHT", pair.right, [])]
                key = viewer.show_panels(panels, status_lines=status)

                if StereoViewer.is_quit(key):
                    print("  收到退出按键，结束预检（仍会写出总结与报告）")
                    captured = True      # 跳出当前距离的等待
                    quit_all = True
                    break
                if key == ord(" "):
                    rows.append({"dist_cm": dist, **{k: v for k, v in res.items()
                                                     if k not in ("corners_l", "corners_r")}})
                    tag = f"d{dist:g}cm"
                    cv2.imwrite(os.path.join(out_dir, f"{tag}_left.png"), pair.left)
                    cv2.imwrite(os.path.join(out_dir, f"{tag}_right.png"), pair.right)
                    cv2.imwrite(os.path.join(out_dir, f"{tag}_overlay.png"), overlay)
                    print(f"      已记录 {dist:g} cm: 每格 {res['px_per_square']:.1f} px "
                          f"清晰度 {res['sharpness']:.0f} "
                          f"对比度 {res['pattern_contrast_min']:.0f}/{res['pattern_contrast']:.0f}  "
                          f"{res['verdict']}")
                    captured = True

    except CameraError as exc:
        print(f"[相机错误] {exc}")
        return 2
    except KeyboardInterrupt:
        print("\n  收到 Ctrl+C，结束预检")
    finally:
        cam.close()
        viewer.close()

    # ---------------------------------------------------------------- 总结
    print("\n" + "=" * 88)
    print("  预检总结")
    print("=" * 88)
    print(f"  {'距离':>7} {'检出L':>6} {'检出R':>6} {'每格px':>7} {'图案宽':>7} {'清晰度':>7} {'对比度':>7}  判定")
    for r in rows:
        print(f"  {r['dist_cm']:>6g}c {'OK' if r['found_l'] else '--':>6} "
              f"{'OK' if r['found_r'] else '--':>6} {r['px_per_square']:>7.1f} "
              f"{r['bbox_w']:>7.1f} {r['sharpness']:>7.0f} {r['pattern_contrast_min']:>7.0f}  {r['verdict']}")

    usable = [r["dist_cm"] for r in rows if r["level"] in ("ok", "warn")]
    best = [r["dist_cm"] for r in rows if r["level"] == "ok"]
    print("-" * 88)
    if best:
        print(f"  ✅ 最佳距离: {', '.join(f'{d:g}cm' for d in best)}")
    if usable:
        print(f"  ✅ 可用距离区间: {min(usable):g} ~ {max(usable):g} cm（含勉强可用）")
    else:
        print("  ❌ 没有任何距离达标 —— 需要按下面的建议调整")
        print("     可能原因与对策:")
        print("       1) 近距离太糊 → 相机定焦在较远处：请改用【大尺寸打印版棋盘格】")
        print("          （我可以生成 12x9、方格 10mm、图案 120x90mm 的 A4 打印图，按 100% 打印）")
        print("       2) 图案太小 → 换更高分辨率采集: --res 2560x960（每眼 1280x480）")
        print("       3) 反光 → 改漫射光/避开点光源，或把板略微倾斜几度")
    if rows:
        rep = os.path.join(out_dir, "probe_report.txt")
        with open(rep, "w", encoding="utf-8") as fh:
            fh.write(f"# 标定板可行性预检\n板子: {spec.describe()}\n")
            fh.write(f"判据: 每格>={PX_PER_SQUARE_MIN:g}px, 清晰度>={SHARP_MIN:g}, "
                     f"图案最差区域对比度>={CONTRAST_MIN:g}\n\n")
            for r in rows:
                fh.write(f"{r['dist_cm']:g}cm  每格{r['px_per_square']:.1f}px  "
                         f"图案宽{r['bbox_w']:.1f}px  清晰度{r['sharpness']:.0f}  "
                         f"对比度(最差/整体){r['pattern_contrast_min']:.0f}/{r['pattern_contrast']:.0f}  "
                         f"{r['verdict']}\n")
            fh.write(f"\n可用距离区间: "
                     f"{f'{min(usable):g}~{max(usable):g} cm' if usable else '无'}\n")
        print(f"\n  报告与证据已保存到: {out_dir}")
    return 0


# -----------------------------------------------------------------------------
# 姿态分类（只用于采集时的清单进度与去重，**不参与标定计算**）
# -----------------------------------------------------------------------------
def classify_pose(res: Dict[str, Any], shape: Tuple[int, int], spec: BoardSpec
                  ) -> Optional[Dict[str, Any]]:
    """把当前姿态分类成"画面区域 + 倾斜方向/角度"，供采集清单进度与去重使用。

    ⚠️ 这里用的是**占位内参**（fx=fy=500），所以角度只是**粗略分类**用途，
       绝不参与标定计算 —— 标定用的内参由 calibration.py 从角点严格解算。
    """
    corners = res.get("corners_l") if res.get("found_l") else res.get("corners_r")
    if corners is None:
        return None
    h, w = shape[:2]
    K = np.array([[500.0, 0.0, w / 2.0], [0.0, 500.0, h / 2.0], [0.0, 0.0, 1.0]], np.float64)
    objp = spec.object_points().reshape(-1, 1, 3).astype(np.float64)
    imgp = np.asarray(corners, np.float64).reshape(-1, 1, 2)
    try:
        ok, rvec, _ = cv2.solvePnP(objp, imgp, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]                                   # 板面法向（相机坐标系）
    tilt = float(np.degrees(np.arccos(np.clip(abs(float(n[2])), -1.0, 1.0))))
    tilt_dir = "none"
    if tilt >= 15.0:
        if abs(float(n[0])) >= abs(float(n[1])):
            tilt_dir = "left" if float(n[0]) < 0 else "right"
        else:
            tilt_dir = "up" if float(n[1]) < 0 else "down"
    pts = np.asarray(corners, np.float32).reshape(-1, 2)
    cx = float(pts[:, 0].mean()) / w
    cy = float(pts[:, 1].mean()) / h
    if 0.32 <= cx <= 0.68 and 0.32 <= cy <= 0.68:
        region = "center"
    elif cy < 0.5:
        region = "tl" if cx < 0.5 else "tr"
    else:
        region = "bl" if cx < 0.5 else "br"
    return {"region": region, "tilt_deg": tilt, "tilt_dir": tilt_dir, "cx": cx, "cy": cy}


# 采集清单：配置里的中文键 -> 判定规则（同一个姿态可同时计入"区域"和"倾斜"两类）
CHECK_RULES = {
    "中心正对": lambda p: p["region"] == "center" and p["tilt_deg"] < 20,
    "左上角": lambda p: p["region"] == "tl",
    "右上角": lambda p: p["region"] == "tr",
    "左下角": lambda p: p["region"] == "bl",
    "右下角": lambda p: p["region"] == "br",
    "左倾大于20度": lambda p: p["tilt_deg"] >= 20 and p["tilt_dir"] == "left",
    "右倾大于20度": lambda p: p["tilt_deg"] >= 20 and p["tilt_dir"] == "right",
    "上仰大于20度": lambda p: p["tilt_deg"] >= 20 and p["tilt_dir"] == "up",
    "下俯大于20度": lambda p: p["tilt_deg"] >= 20 and p["tilt_dir"] == "down",
}


def gate_messages(res: Dict[str, Any], thr: Dict[str, float]) -> List[Tuple[str, str]]:
    """质量门的判断结果，返回 [(中文原因, ASCII原因), ...]；空列表 = 通过。

    为什么要两份：**画在图像上的文字必须纯 ASCII**（cv2.putText 画不出中文，
    会变成 ?），而控制台里要给人看中文。
    """
    if not res["found_l"] and not res["found_r"]:
        return [("左右眼都没检出角点（太远/太斜/反光/模糊）", "corners not found")]
    msgs: List[Tuple[str, str]] = []
    if res["px_per_square"] < thr["min_px_per_square"]:
        msgs.append((f"图案太小：每格 {res['px_per_square']:.1f} < {thr['min_px_per_square']:g} px（靠近一点）",
                     f"too small: {res['px_per_square']:.1f} < {thr['min_px_per_square']:g} px/sq"))
    if res["sharpness"] < thr["min_sharpness"]:
        msgs.append((f"太糊：清晰度 {res['sharpness']:.0f} < {thr['min_sharpness']:g}",
                     f"too blurry: sharp {res['sharpness']:.0f}"))
    if res["pattern_contrast_min"] < thr["min_contrast"]:
        msgs.append((f"对比度不足：最差区域 {res['pattern_contrast_min']:.0f} < {thr['min_contrast']:g}"
                     f"（多半是反光，试试把板略微倾斜）",
                     f"low contrast: {res['pattern_contrast_min']:.0f} (glare?)"))
    return msgs


def quality_gate(res: Dict[str, Any], thr: Dict[str, float]) -> Tuple[bool, List[str]]:
    """采集质量门：返回 (是否通过, 不通过的中文原因列表)。

    注意：**只单眼检出也允许通过**（那一帧仍能用于该眼的单眼内参标定），
    但会在记录里标记为 single-eye，双目外参只用双眼都检出的帧。
    """
    msgs = gate_messages(res, thr)
    return (len(msgs) == 0), [m[0] for m in msgs]


def _pose_signature(pose: Dict[str, Any]) -> Tuple:
    """把姿态压成一个粗粒度签名，用于"这张和之前那张太像"的去重判断。"""
    return (pose["region"], int(pose["tilt_deg"] // 12), int(pose["tilt_dir"] != "none"),
            int(pose["cx"] * 8), int(pose["cy"] * 8))


def _checklist_lines(records: Sequence[Dict[str, Any]], checklist: Dict[str, int]) -> List[str]:
    """按清单统计已采集数量，返回显示用文本行。"""
    lines = []
    for name, need in checklist.items():
        rule = CHECK_RULES.get(name)
        got = sum(1 for r in records if r.get("pose") and rule and rule(r["pose"])) if rule else 0
        mark = "OK " if got >= need else "   "
        lines.append(f"  {mark}{name:<12} {got}/{need}")
    return lines


# -----------------------------------------------------------------------------
# 模式三：正式采集（需要相机；普通权限即可）
# -----------------------------------------------------------------------------
def capture_mode(cfg: Dict[str, Any], args: argparse.Namespace, spec: BoardSpec) -> int:
    cal = dict(cfg.get("calibration", {}) or {})
    thr = {
        "min_px_per_square": float(cal.get("min_px_per_square", PX_PER_SQUARE_MIN)),
        "min_sharpness": float(cal.get("min_sharpness", SHARP_MIN)),
        "min_contrast": float(cal.get("min_contrast", CONTRAST_MIN)),
    }
    checklist = dict(cal.get("checklist", {}) or {})
    target = int(cal.get("target_frames", 26))
    dmin, dmax = cal.get("distance_min_cm", 9), cal.get("distance_max_cm", 17)

    out_dir = os.path.join(PROJECT_ROOT, "output", "calib", time.strftime("%Y%m%d_%H%M%S"))
    runtime = cfg.get("runtime", {})
    viewer = StereoViewer(window_name="capture_calib (capture)",
                          display_scale=float(args.display_scale or runtime.get("display_scale", 1.0)))

    print("=" * 88)
    print("  标定图采集（步骤 1）")
    print("=" * 88)
    print(f"  板子规格  : {spec.describe()}")
    print(f"  工作距离  : {dmin:g} ~ {dmax:g} cm（预检实测的可用区间）")
    print(f"  目标张数  : {target}")
    print(f"  质量门    : 每格 >= {thr['min_px_per_square']:g} px、清晰度 >= {thr['min_sharpness']:g}、"
          f"对比度 >= {thr['min_contrast']:g}")
    print("  操作      : 【空格】记录一帧（不达标会被拒绝并说明原因）")
    print("              【u】撤销上一张   【f】强制记录（跳过质量门）   【q】结束并写报告")
    print("  提示      : 近距离视差很大，板要放在画面偏中右的位置，才能同时被左右眼看到")
    print("-" * 88)
    print("  采集清单（实时更新）:")
    for ln in _checklist_lines([], checklist):
        print(ln)
    print("-" * 88)

    cam = StereoCamera(cfg.get("camera", {}))
    records: List[Dict[str, Any]] = []
    sigs: set = set()
    try:
        cam.open()
        print(f"  相机已打开: {cam.describe()}  单眼 {cam.eye_size[0]}x{cam.eye_size[1]}")
        os.makedirs(out_dir, exist_ok=True)

        while True:
            try:
                pair = cam.read()
            except CameraReadError as exc:
                print(f"  [读取错误] {exc}")
                break
            # 采集必须 strict：只认规格推出的 (11,8)，避免匹配到更小的子棋盘而静默污染数据
            res = analyze_pair(pair.left, pair.right, spec, strict=True)
            pose = classify_pose(res, pair.left.shape[:2], spec)
            gate = gate_messages(res, thr)          # [(中文, ASCII), ...]
            ok = (len(gate) == 0)
            reasons = [m[0] for m in gate]           # 控制台用中文
            reasons_ascii = [m[1] for m in gate]     # 画到图上的状态栏必须纯 ASCII
            dup = bool(pose) and (_pose_signature(pose) in sigs)
            both = bool(res["found_l"] and res["found_r"])

            pose_txt = (f"region {pose['region']}  tilt {pose['tilt_deg']:.0f}deg {pose['tilt_dir']}"
                        if pose else "pose: -")
            status = [
                f"target distance {dmin:g}-{dmax:g} cm   |   saved {len(records)}/{target}",
                f"corners L/R: {'OK' if res['found_l'] else '--'} / {'OK' if res['found_r'] else '--'}"
                f"   {'(stereo ok)' if both else '(single eye only)'}   {pose_txt}",
                f"px per square: {res['px_per_square']:.1f}   sharpness: {res['sharpness']:.0f}   "
                f"contrast worst: {res['pattern_contrast_min']:.0f}",
                ("READY: press SPACE to record" if (ok and not dup)
                 else ("BLOCKED: " + "; ".join(reasons_ascii) if reasons_ascii
                       else "BLOCKED: pose too similar")),
            ]
            # 左图用左眼角点、右图用右眼角点（曾经两处都用左眼，导致右图叠加偏移一个视差）
            panels = [("LEFT + corners", draw_overlay(pair.left, res, spec, side="left"), []),
                      ("RIGHT + corners", draw_overlay(pair.right, res, spec, side="right"), [])]
            key = viewer.show_panels(panels, status_lines=status)

            save_now = force = False
            if StereoViewer.is_quit(key):
                print("  收到退出按键，结束采集")
                break
            if key == ord(" "):
                save_now = True
            if key == ord("f"):
                save_now, force = True, True
            if key == ord("u") and records:
                last = records.pop()
                sigs.discard(last["sig"])
                for p in last["files"]:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                print(f"  已撤销第 {last['idx']} 张（删除 {len(last['files'])} 个文件）")
                continue
            if not save_now:
                continue

            if not force and reasons:
                print("  ❌ 拒绝记录：" + "; ".join(reasons))
                continue
            if not force and dup:
                print("  ❌ 这张与已有帧姿态太接近，请换个角度/位置（确实要留请按 f 强制记录）")
                continue

            idx = len(records) + 1
            files = []
            for tag, img in (("left", pair.left), ("right", pair.right)):
                p = os.path.join(out_dir, f"{tag}_{idx:02d}.png")
                cv2.imwrite(p, img)
                files.append(p)
            p = os.path.join(out_dir, f"overlay_{idx:02d}.png")
            cv2.imwrite(p, draw_overlay(pair.left, res, spec, side="left"))
            files.append(p)
            rec = {
                "idx": idx, "files": files, "sig": _pose_signature(pose) if pose else None,
                "pose": pose, "stereo": both,
                "px_per_square": res["px_per_square"], "sharpness": res["sharpness"],
                "contrast": res["pattern_contrast_min"], "bbox_w": res["bbox_w"],
                "found_l": res["found_l"], "found_r": res["found_r"],
                "levels": (res["level"], len(reasons), force, dup),
            }
            records.append(rec)
            if rec["sig"]:
                sigs.add(rec["sig"])
            flag = "强制" if force else ("双眼" if both else "仅单眼")
            print(f"  ✅ 已记录第 {idx:02d} 张 [{flag}]：每格 {res['px_per_square']:.1f} px  "
                  f"清晰度 {res['sharpness']:.0f}  对比度 {res['pattern_contrast_min']:.0f}  "
                  f"{pose_txt}")
            if idx % 5 == 0 or idx == target:
                print("  采集清单:")
                for ln in _checklist_lines(records, checklist):
                    print(ln)

    except CameraError as exc:
        print(f"[相机错误] {exc}")
        cam.close()
        viewer.close()
        return 2
    except KeyboardInterrupt:
        print("\n  收到 Ctrl+C，结束采集")
    finally:
        cam.close()
        viewer.close()

    # ------------------------------------------------- 总结与报告（一定写出）
    stereo_n = sum(1 for r in records if r["stereo"])
    print("\n" + "=" * 88)
    print("  采集总结")
    print("=" * 88)
    print(f"  共采集        : {len(records)} 对（目标 {target}）")
    print(f"  双眼都检出    : {stereo_n} 对（用于双目外参标定）")
    print(f"  仅单眼检出    : {len(records) - stereo_n} 对（只用于该眼的单眼内参）")
    print(f"  每格像素      : {min([r['px_per_square'] for r in records], default=0):.1f} ~ "
          f"{max([r['px_per_square'] for r in records], default=0):.1f}")
    print("  清单完成情况:")
    missing = []
    for name, need in checklist.items():
        rule = CHECK_RULES.get(name)
        got = sum(1 for r in records if r["pose"] and rule and rule(r["pose"])) if rule else 0
        flag = "OK " if got >= need else "缺 "
        print(f"    {flag}{name:<12} {got}/{need}")
        if got < need:
            missing.append(f"{name}({got}/{need})")
    if missing:
        print(f"  ⚠️ 还缺: {', '.join(missing)}")
        print("     缺的类别会削弱标定质量（尤其是四角→畸变、倾斜→焦距），建议补齐后再算参数")
    else:
        print("  ✅ 清单已全部满足")
    if len(records) < 10:
        print("  ⚠️ 张数偏少（<10），标定结果可能不稳，建议至少 16~20 对")

    if records:
        rep = os.path.join(out_dir, "capture_report.txt")
        with open(rep, "w", encoding="utf-8") as fh:
            fh.write(f"# 标定图采集记录\n板子: {spec.describe()}\n")
            fh.write(f"工作距离建议: {dmin:g}~{dmax:g} cm\n")
            fh.write(f"质量门: 每格>={thr['min_px_per_square']:g}px 清晰度>={thr['min_sharpness']:g} "
                     f"对比度>={thr['min_contrast']:g}\n\n")
            fh.write("idx  双眼  每格px  清晰度  对比度  区域    倾角  方向\n")
            for r in records:
                p = r["pose"] or {}
                fh.write(f"{r['idx']:>3}  {'Y' if r['stereo'] else 'N':>4}  "
                         f"{r['px_per_square']:>6.1f}  {r['sharpness']:>6.0f}  {r['contrast']:>6.0f}  "
                         f"{p.get('region', '-'):<7} {p.get('tilt_deg', 0):>5.0f}  {p.get('tilt_dir', '-')}\n")
            fh.write(f"\n共 {len(records)} 对，其中双眼可用 {stereo_n} 对\n")
            if missing:
                fh.write("缺的类别: " + ", ".join(missing) + "\n")
        print(f"\n  证据与报告已保存到: {out_dir}")
        print("  下一步: python tools\\run_calibration.py  (步骤 2：计算标定参数)")
    else:
        print("\n  ⚠️ 没有采集到任何一帧（未写出报告）")
    return 0


# -----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="标定图采集与标定板可行性预检（阶段 5）")
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    ap.add_argument("--probe", action="store_true", help="交互式可行性预检（需要相机，普通权限即可）")
    ap.add_argument("--capture", action="store_true", help="正式采集标定图（需要相机，普通权限即可）")
    ap.add_argument("--selftest", action="store_true", help="用合成图自检分析逻辑（不需要相机）")
    ap.add_argument("--distances", default="10,12,15,18,20,25,30",
                    help="预检的距离序列（cm，逗号分隔）")
    ap.add_argument("--res", default="", help="覆盖相机分辨率，例如 2560x960（默认用 config）")
    ap.add_argument("--display-scale", type=float, default=None, help="显示缩放（默认用 config）")
    # 标定板规格（默认就是 DFOPTIX Q12-50-3）
    ap.add_argument("--board-cols", type=int, default=12, help="方格列数（默认 12）")
    ap.add_argument("--board-rows", type=int, default=9, help="方格行数（默认 9）")
    ap.add_argument("--square-mm", type=float, default=3.0, help="方格边长 mm（默认 3.0）")
    args = ap.parse_args(argv)

    spec = BoardSpec(cols=args.board_cols, rows=args.board_rows, square_mm=args.square_mm)

    if args.selftest:
        return selftest_mode(spec)
    if not args.probe and not args.capture:
        ap.print_help()
        print("\n提示：先用 --selftest 验证工具本身（不需要相机），"
              "再用 --probe 做可行性预检，最后用 --capture 正式采集")
        return 1

    # 读配置（--res 可覆盖相机分辨率）
    import yaml
    if not os.path.exists(args.config):
        print(f"[错误] 找不到配置文件: {args.config}")
        return 1
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("ffs_repo", "output_dir", "weights_dir", "calibration_dir"):
        if key in cfg.get("project", {}):
            cfg["project"][key] = os.path.normpath(os.path.join(PROJECT_ROOT, cfg["project"][key]))
    if args.res:
        try:
            w, h = (int(x) for x in str(args.res).lower().split("x"))
            cfg.setdefault("camera", {})["width"] = w
            cfg["camera"]["height"] = h
            print(f"[覆盖] 相机分辨率 -> {w}x{h}")
        except Exception:
            print(f"[错误] --res 应形如 2560x960，当前是 {args.res!r}")
            return 1

    return capture_mode(cfg, args, spec) if args.capture else probe_mode(cfg, args, spec)


if __name__ == "__main__":
    sys.exit(main())
