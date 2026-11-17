"""calib_check.py —— 校正效果的可视化与分区统计（无界面依赖）
=============================================================================
【为什么需要单独一层】
    标定完只看到一个 "RMS 0.12 px" 是不够的：RMS 是标定时拟合出来的残差，
    它不能证明"校正后左右图的同一物点真的在同一行"。能证明这件事的只有两种证据：
        ① 数字：校正后重新检测角点，测同一角点的垂直差 |dy|
        ② 画面：把校正后左右图上下叠起来，画贯穿的水平参考线，用眼睛看
    本文件把这两件事做成可复用的函数，GUI 与命令行都能调。

【提供的画面（全部返回 BGR ndarray，不落盘、不显示）】
    stack_with_lines   上下叠放 + 贯穿水平线（最直观的行对齐检查）
    side_by_side       并排 + 网格线
    anaglyph           红青立体（戴红青眼镜可看深度）
    diff_map           左右差值图（对齐好的话棋盘边缘呈双线）
    blend_checker      棋盘式混合（左右交替拼块，错位会立刻显形）
    coverage_overlay   角点覆盖热力图（看清"哪些区域根本没被采样"）
    region_stats_image 分区 |dy| 热力图（中心与边缘分别统计）

【一句提醒（与项目实测一致）】
    alpha=0 的 stereoRectify 会为"全图无黑边"做裁切，视场会损失一部分。
    alpha_panel() 能量出每种 alpha 下的有效像素比例，别在不知情的情况下丢掉视场。
=============================================================================
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calib_core import _cal, check_cancel  # noqa: E402  —— 复用 detect_corners_strict 与取消判据


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------
def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _put(img: np.ndarray, text: str, org: Tuple[int, int],
         color: Tuple[int, int, int] = (0, 255, 255), scale: float = 0.55) -> None:
    """写英文/数字标签（cv2.putText 不支持中文，界面里的中文由 tkinter 负责）。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# 画面：行对齐检查
# -----------------------------------------------------------------------------
def stack_with_lines(left_rect: np.ndarray, right_rect: np.ndarray,
                     step: int = 40, gap: int = 6) -> np.ndarray:
    """把校正后的左右图**上下叠放**，画贯穿两幅图的水平线。

    判据：同一物点在左图与右图应落在同一条水平线上。
    若某个角点在两幅图里明显错开一条线的高度，校正就没做好。
    """
    L = _to_bgr(left_rect)
    R = _to_bgr(right_rect)
    if L.shape != R.shape:
        R = cv2.resize(R, (L.shape[1], L.shape[0]))
    h, w = L.shape[:2]
    canvas = np.zeros((h * 2 + gap, w, 3), np.uint8)
    canvas[:h] = L
    canvas[h + gap:] = R

    for y in range(0, h, max(int(step), 10)):
        cv2.line(canvas, (0, y), (w - 1, y), (0, 200, 255), 1, cv2.LINE_AA)
        y2 = y + h + gap
        if y2 < canvas.shape[0]:
            cv2.line(canvas, (0, y2), (w - 1, y2), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, h + gap // 2), (w - 1, h + gap // 2), (255, 255, 255), 1)
    _put(canvas, "LEFT (rectified)", (8, 20), (0, 255, 0))
    _put(canvas, "RIGHT (rectified) - same horizontal lines must hit the same feature",
         (8, h + gap + 20), (0, 255, 0))
    return canvas


def side_by_side(left_rect: np.ndarray, right_rect: np.ndarray,
                 gap: int = 8, step: int = 40) -> np.ndarray:
    """左右并排 + 每 step 像素一条水平参考线（跨两幅图连续）。"""
    L = _to_bgr(left_rect)
    R = _to_bgr(right_rect)
    if L.shape != R.shape:
        R = cv2.resize(R, (L.shape[1], L.shape[0]))
    h, w = L.shape[:2]
    canvas = np.zeros((h, w * 2 + gap, 3), np.uint8)
    canvas[:, :w] = L
    canvas[:, w + gap:] = R
    for y in range(0, h, max(int(step), 10)):
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), (0, 200, 255), 1, cv2.LINE_AA)
    _put(canvas, "LEFT", (8, 20), (0, 255, 0))
    _put(canvas, "RIGHT", (w + gap + 8, 20), (0, 255, 0))
    return canvas


# -----------------------------------------------------------------------------
# 画面：其它对比方式
# -----------------------------------------------------------------------------
def anaglyph(left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
    """红青立体图：左图取红通道、右图取绿蓝通道。"""
    L = _to_bgr(left_rect)
    R = _to_bgr(right_rect)
    if L.shape != R.shape:
        R = cv2.resize(R, (L.shape[1], L.shape[0]))
    out = np.zeros_like(L)
    out[:, :, 2] = L[:, :, 2]                 # BGR 里的 R 通道
    out[:, :, 1] = R[:, :, 1]
    out[:, :, 0] = R[:, :, 0]
    return out


def diff_map(left_rect: np.ndarray, right_rect: np.ndarray,
             amplify: float = 1.0) -> np.ndarray:
    """左右差值图（乘 amplify 便于观察）。对齐良好时棋盘边缘呈细双线。"""
    L = _to_bgr(left_rect)
    R = _to_bgr(right_rect)
    if L.shape != R.shape:
        R = cv2.resize(R, (L.shape[1], L.shape[0]))
    d = cv2.absdiff(_gray(L), _gray(R)).astype(np.float32) * float(amplify)
    return cv2.applyColorMap(np.clip(d, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def blend_checker(left_rect: np.ndarray, right_rect: np.ndarray,
                  tiles: int = 10) -> np.ndarray:
    """棋盘式混合：相邻小块分别取左图/右图。未对齐的地方会看到断裂。"""
    L = _to_bgr(left_rect)
    R = _to_bgr(right_rect)
    if L.shape != R.shape:
        R = cv2.resize(R, (L.shape[1], L.shape[0]))
    h, w = L.shape[:2]
    th, tw = max(h // max(tiles, 1), 1), max(w // max(tiles, 1), 1)
    out = L.copy()
    for i in range(tiles):
        for j in range(tiles):
            if (i + j) % 2 == 0:
                continue
            y0, y1 = i * th, min((i + 1) * th, h)
            x0, x1 = j * tw, min((j + 1) * tw, w)
            if y1 > y0 and x1 > x0:
                out[y0:y1, x0:x1] = R[y0:y1, x0:x1]
    return out


# -----------------------------------------------------------------------------
# 覆盖率与分区统计
# -----------------------------------------------------------------------------
def coverage_overlay(reference: np.ndarray, grid: np.ndarray,
                     alpha: float = 0.45, min_show: int = 1) -> np.ndarray:
    """把覆盖计数网格叠在参考图上（红=没采到，绿=采得多）。"""
    base = _to_bgr(reference).copy()
    h, w = base.shape[:2]
    g = np.asarray(grid, np.float64)
    if g.ndim != 2 or g.size == 0:
        return base
    gh, gw = g.shape
    vmax = max(float(g.max()), 1.0)
    norm = np.clip(g / vmax, 0.0, 1.0)
    heat = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
    out = cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0.0)

    for i in range(gh + 1):
        y = int(round(i * h / gh))
        cv2.line(out, (0, min(y, h - 1)), (w - 1, min(y, h - 1)), (60, 60, 60), 1)
    for j in range(gw + 1):
        x = int(round(j * w / gw))
        cv2.line(out, (min(x, w - 1), 0), (min(x, w - 1), h - 1), (60, 60, 60), 1)

    covered = int((g >= int(min_show)).sum())
    _put(out, f"coverage cells: {covered}/{g.size} ({(g >= min_show).mean() * 100:.0f}%)",
         (8, 20), (255, 255, 255))
    return out


def region_stats(calib: Any, frames_dir: str, spec: Any,
                 grid: Tuple[int, int] = (3, 4),
                 cancel: Optional[Any] = None) -> Dict[str, Any]:
    """校正后按**画面分区**统计极线失准 |dy|。

    为什么要分区：全图平均会把边缘的大误差摊薄。鱼眼视场下，边缘往往比中心差一个量级，
    只看平均值会得出过于乐观的结论。

    输出：{"grid": (行,列), "mean": 二维数组, "count": 二维数组,
           "overall": 均值, "center": 中心格均值, "edge": 边缘格均值}
    """
    cal = _cal()
    lefts = {_idx_of(p): p for p in glob.glob(os.path.join(frames_dir, "left_*.png"))}
    rights = {_idx_of(p): p for p in glob.glob(os.path.join(frames_dir, "right_*.png"))}
    keys = sorted(set(lefts) & set(rights))
    gh, gw = int(grid[0]), int(grid[1])
    acc = np.zeros((gh, gw), np.float64)
    cnt = np.zeros((gh, gw), np.int32)

    for i in keys:
        check_cancel(cancel)
        L = cv2.imread(lefts[i])
        R = cv2.imread(rights[i])
        if L is None or R is None:
            continue
        Lr, Rr = calib.rectify_pair(L, R)
        cl = cal.detect_corners_strict(_gray(Lr), spec.pattern_size)
        cr = cal.detect_corners_strict(_gray(Rr), spec.pattern_size)
        if cl is None or cr is None:
            continue
        a = np.asarray(cl, np.float32).reshape(-1, 2)
        b = np.asarray(cr, np.float32).reshape(-1, 2)
        if a.shape != b.shape:
            continue
        dy = np.abs(a[:, 1] - b[:, 1])
        h, w = Lr.shape[:2]
        gx = np.clip((a[:, 0] / w * gw).astype(int), 0, gw - 1)
        gy = np.clip((a[:, 1] / h * gh).astype(int), 0, gh - 1)
        np.add.at(acc, (gy, gx), dy)
        np.add.at(cnt, (gy, gx), 1)

    mean = np.divide(acc, np.maximum(cnt, 1))
    valid = cnt > 0
    overall = float(acc.sum() / max(cnt.sum(), 1))
    center = float(mean[gh // 2, gw // 2]) if cnt[gh // 2, gw // 2] else float("nan")
    edge_cells = [(i, j) for i in range(gh) for j in range(gw)
                  if (i in (0, gh - 1)) or (j in (0, gw - 1))]
    edge_vals = [mean[i, j] for i, j in edge_cells if cnt[i, j] > 0]
    return {"grid": (gh, gw), "mean": mean, "count": cnt, "valid": valid,
            "overall": overall, "center": center,
            "edge": float(np.mean(edge_vals)) if edge_vals else float("nan"),
            "frames_used": int(len(keys))}


def region_stats_image(stats: Dict[str, Any], cell: int = 64) -> np.ndarray:
    """把分区统计画成热力图（每格标 |dy| 与样本数）。"""
    mean = np.asarray(stats["mean"], np.float64)
    cnt = np.asarray(stats["count"], np.int32)
    vmax = max(float(np.nanmax(mean)) if np.isfinite(mean).any() else 1.0, 0.2)
    gh, gw = mean.shape
    img = np.zeros((gh * cell, gw * cell, 3), np.uint8)
    for i in range(gh):
        for j in range(gw):
            v = mean[i, j]
            if cnt[i, j] == 0:
                color = (40, 40, 40)
                txt = "-"
            else:
                t = float(np.clip(v / vmax, 0.0, 1.0))
                color = cv2.applyColorMap(np.uint8([[int(t * 255)]]), cv2.COLORMAP_JET)[0, 0]
                color = tuple(int(c) for c in color)
                txt = f"{v:.2f}"
            y0, x0 = i * cell, j * cell
            cv2.rectangle(img, (x0, y0), (x0 + cell - 1, y0 + cell - 1), color, -1)
            cv2.rectangle(img, (x0, y0), (x0 + cell - 1, y0 + cell - 1), (90, 90, 90), 1)
            _put(img, txt, (x0 + 6, y0 + 24), (255, 255, 255), 0.45)
            _put(img, f"n={cnt[i, j]}", (x0 + 6, y0 + 42), (220, 220, 220), 0.35)
    hdr = (f"|dy| by region (px)  overall={stats['overall']:.3f}  "
           f"center={stats['center']:.3f}  edge={stats['edge']:.3f}")
    canvas = np.zeros((img.shape[0] + 24, img.shape[1], 3), np.uint8)
    canvas[24:] = img
    _put(canvas, hdr, (6, 17), (0, 255, 255), 0.45)
    return canvas


def region_stats_text(stats: Dict[str, Any]) -> str:
    mean = np.asarray(stats["mean"], np.float64)
    cnt = np.asarray(stats["count"], np.int32)
    lines = [f"分区 |dy|（{stats['grid'][0]}x{stats['grid'][1]}，单位 px，括号内为样本数）"]
    for i in range(mean.shape[0]):
        row = []
        for j in range(mean.shape[1]):
            row.append(f"{mean[i, j]:6.2f}({cnt[i, j]:4d})" if cnt[i, j] else "     - (   0)")
        lines.append("  " + "  ".join(row))
    lines.append(f"  整体 {stats['overall']:.3f} / 中心 {stats['center']:.3f} / "
                 f"边缘 {stats['edge']:.3f}")
    if int(np.asarray(stats["count"]).sum()) == 0:
        lines.append("  ⚠ 校正后一帧都没检出角点。常见原因：畸变系数在边缘失稳，"
                     "stereoRectify 把画面放得过大，板子被裁出视野。")
        lines.append("    先看 alpha_panel 里的 P1.fx 与 K1.fx 之比，若远大于 1 就属于这种情形。")
    elif np.isfinite(stats["edge"]) and np.isfinite(stats["center"]) and stats["center"] > 0:
        ratio = stats["edge"] / stats["center"]
        lines.append(f"  边缘/中心 = {ratio:.2f} 倍" +
                     ("（边缘明显更差，说明畸变模型在四周吃紧）" if ratio > 2.0 else ""))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# alpha 与有效视场
# -----------------------------------------------------------------------------
def alpha_panel(calib: Any, alphas: Sequence[float] = (0.0, 0.5, 1.0),
                image_size: Optional[Tuple[int, int]] = None
                ) -> List[Dict[str, Any]]:
    """对每个 alpha 重新 stereoRectify，量出有效像素比例与等效焦距。

    alpha=0  → 只保留两眼共视区域，无黑边，但视场最小（项目当前用的就是它）
    alpha=1  → 保留全部像素，出现黑边，视场最大
    有效比例用"把全白图按新映射重采样后非零像素占比"实测，不是估算。
    """
    size = tuple(image_size or calib.image_size)
    out: List[Dict[str, Any]] = []
    for a in alphas:
        try:
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                calib.K1, calib.D1, calib.K2, calib.D2, size,
                calib.R, calib.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=float(a))
        except cv2.error as exc:
            out.append({"alpha": float(a), "error": str(exc)})
            continue
        m1x, m1y = cv2.initUndistortRectifyMap(calib.K1, calib.D1, R1, P1, size, cv2.CV_32FC1)
        white = np.full((size[1], size[0]), 255, np.uint8)
        valid = cv2.remap(white, m1x, m1y, cv2.INTER_NEAREST) > 0
        out.append({
            "alpha": float(a),
            "valid_ratio": float(valid.mean()),
            "rect_fx": float(P1[0, 0]),
            "cx": float(P1[0, 2]), "cy": float(P1[1, 2]),
            "roi1": [int(x) for x in roi1], "roi2": [int(x) for x in roi2],
        })
    return out


def alpha_text(panel: Sequence[Dict[str, Any]]) -> str:
    lines = ["alpha 对有效视场的影响（有效像素比例用实际重采样实测）"]
    for p in panel:
        if "error" in p:
            lines.append(f"  alpha={p['alpha']:.1f} 失败：{p['error']}")
            continue
        lines.append(f"  alpha={p['alpha']:.1f}  有效 {p['valid_ratio'] * 100:5.1f}%  "
                     f"fx={p['rect_fx']:.2f}  roi={p['roi1']}")
    lines.append("  说明：alpha 越小视场越窄但无黑边；alpha 越大保留越多但出现黑边。")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 一次生成全部检查图（命令行 / GUI 共用）
# -----------------------------------------------------------------------------
def render_all(calib: Any, frames_dir: str, spec: Any,
               out_dir: str = "", grid: Tuple[int, int] = (3, 4),
               alpha: float = 0.0,
               cancel: Optional[Any] = None) -> Dict[str, Any]:
    """对采集目录里的一帧生成全套检查图，返回 {"images": {...}, "stats": {...}}。

    选定"最典型的一帧"的规则：取**覆盖角点数最多**的那一帧（板子最完整）。
    cancel：threading.Event，置位后抛 Cancelled（每帧检测一次，可逐帧中断）。
    """
    cal = _cal()
    lefts = {_idx_of(p): p for p in glob.glob(os.path.join(frames_dir, "left_*.png"))}
    rights = {_idx_of(p): p for p in glob.glob(os.path.join(frames_dir, "right_*.png"))}
    keys = sorted(set(lefts) & set(rights))
    if not keys:
        raise FileNotFoundError(f"目录里没有成对的 left/right 图：{frames_dir}")

    check_cancel(cancel)
    best_i, best_score = keys[0], -1
    for i in keys:
        check_cancel(cancel)
        g = cv2.imread(lefts[i], cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        c = cal.detect_corners_strict(g, spec.pattern_size)
        score = 0 if c is None else len(c)
        if score > best_score:
            best_i, best_score = i, score

    L = cv2.imread(lefts[best_i])
    R = cv2.imread(rights[best_i])
    if L is None or R is None:
        raise FileNotFoundError(f"读不到第 {best_i} 帧")

    Lr, Rr = calib.rectify_pair(L, R)
    images: Dict[str, np.ndarray] = {
        "raw_L": L, "raw_R": R, "rect_L": Lr, "rect_R": Rr,
        "stack": stack_with_lines(Lr, Rr),
        "sbs": side_by_side(Lr, Rr),
        "anaglyph": anaglyph(Lr, Rr),
        "diff": diff_map(Lr, Rr),
        "checker": blend_checker(Lr, Rr),
    }

    stats = region_stats(calib, frames_dir, spec, grid=grid, cancel=cancel)
    images["region"] = region_stats_image(stats)
    grid_counts = np.zeros((12, 16), np.int32)
    for i in keys:
        check_cancel(cancel)
        g = cv2.imread(lefts[i], cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        c = cal.detect_corners_strict(g, spec.pattern_size)
        if c is None:
            continue
        pts = np.asarray(c, np.float32).reshape(-1, 2)
        h, w = g.shape[:2]
        gx = np.clip((pts[:, 0] / w * 16).astype(int), 0, 15)
        gy = np.clip((pts[:, 1] / h * 12).astype(int), 0, 11)
        np.add.at(grid_counts, (gy, gx), 1)
    images["coverage"] = coverage_overlay(L, grid_counts)

    panel = alpha_panel(calib, image_size=calib.image_size)
    saved: List[str] = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, img in images.items():
            p = os.path.join(out_dir, f"check_{name}_{best_i:02d}.png")
            from depth_utils import imwrite_image
            if imwrite_image(p, img):
                saved.append(p)
        with open(os.path.join(out_dir, "check_region_stats.txt"), "w", encoding="utf-8") as fh:
            fh.write(region_stats_text(stats) + "\n\n" + alpha_text(panel) + "\n")

    return {"images": images, "stats": stats, "alpha": panel,
            "frame_index": best_i, "saved": saved}


def _idx_of(path: str) -> int:
    import re
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


# -----------------------------------------------------------------------------
# 命令行
# -----------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="校正效果检查（生成检查图 + 分区统计）")
    ap.add_argument("--calib", default=os.path.join(PROJECT_ROOT, "calibration",
                                                    "stereo_calib.yaml"))
    ap.add_argument("--dir", default="", help="采集目录（默认取标定文件里记录的 capture_dir）")
    ap.add_argument("--out", default="", help="输出目录（默认 output/calib_check/<时间戳>）")
    ap.add_argument("--grid", default="3x4", help="分区网格，例如 3x4")
    args = ap.parse_args(argv)

    cal = _cal()
    if not os.path.exists(args.calib):
        print(f"[错误] 标定文件不存在：{args.calib}")
        return 1
    calib = cal.StereoCalibration.load(args.calib)

    frames_dir = args.dir
    if not frames_dir:
        cd = str((calib.meta or {}).get("capture_dir", ""))
        frames_dir = os.path.join(PROJECT_ROOT, "output", "calib", cd) if cd else ""
    if not frames_dir or not os.path.isdir(frames_dir):
        print(f"[错误] 采集目录无效：{frames_dir!r}，请用 --dir 指定")
        return 1

    bp = (calib.meta or {}).get("board_params", {}) or {}
    spec = cal.BoardSpec(cols=int(bp.get("cols", 12)), rows=int(bp.get("rows", 9)),
                         square_mm=float((calib.meta or {}).get(
                             "square_mm", bp.get("square_mm", 3.0))))
    out_dir = args.out or os.path.join(PROJECT_ROOT, "output", "calib_check",
                                       os.path.basename(frames_dir))
    try:
        gh, gw = (int(x) for x in str(args.grid).lower().split("x"))
    except Exception:                       # noqa: BLE001
        gh, gw = 3, 4

    res = render_all(calib, frames_dir, spec, out_dir=out_dir, grid=(gh, gw))
    print(f"用第 {res['frame_index']} 帧生成检查图，输出到：{out_dir}")
    for p in res["saved"]:
        print("  " + os.path.basename(p))
    print()
    print(region_stats_text(res["stats"]))
    print()
    print(alpha_text(res["alpha"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
