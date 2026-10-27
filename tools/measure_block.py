"""measure_block.py —— 用已知尺寸的物块测量双目精度

=============================================================================
【测什么】
    卡尺量过的正面宽、高是真值。重建出的三维边长如果偏 2%，
    说明深度尺度也偏 2%（点云是按深度整体放大的）。另外两项只看噪声：
        平面度   正面内部点到拟合平面的残差，和角点点得准不准基本无关
        重复性   物块不动，连测若干帧，看距离和边长跳多少

【不要用卷尺当主真值】
    镜头外壳到物块的距离，和左相机光心到物块的深度不是同一个点，
    差几个毫米到一两厘米都正常。--distance-mm 只作对照。

【用法】
    python tools\\measure_block.py --scale 1.0 --iters 8
    python tools\\measure_block.py --width-mm 100 --height-mm 60
    python tools\\measure_block.py --selftest

画面左上角直接写出物块正面的长、短和距离。物块放到画面中间即可，
也可以点一下要量的那个面。
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# -----------------------------------------------------------------------------
# 几何
# -----------------------------------------------------------------------------
def polygon_area(pts: np.ndarray) -> float:
    x = np.asarray(pts, np.float64)[:, 0]
    y = np.asarray(pts, np.float64)[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def cross(p, q, r) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    return cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0


def is_simple_quad(pts: np.ndarray, min_area: float = 400.0) -> bool:
    p = np.asarray(pts, np.float64).reshape(4, 2)
    if segments_intersect(p[0], p[1], p[2], p[3]):
        return False
    if segments_intersect(p[1], p[2], p[3], p[0]):
        return False
    return polygon_area(p) >= float(min_area)


def inset_polygon(pts: np.ndarray, ratio: float) -> np.ndarray:
    """朝质心收缩。ratio=0.12 表示每条边往里收 12%。"""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    c = p.mean(axis=0)
    return c + (1.0 - float(ratio)) * (p - c)


def fill_mask(shape_hw: Tuple[int, int], pts: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)
    p = np.round(np.asarray(pts, np.float64).reshape(-1, 2)).astype(np.int32)
    p[:, 0] = np.clip(p[:, 0], 0, w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    cv2.fillPoly(mask, [p.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def unproject(u: np.ndarray, v: np.ndarray, z: np.ndarray,
              fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    z = np.asarray(z, np.float64)
    x = (np.asarray(u, np.float64) - cx) * z / fx
    y = (np.asarray(v, np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def ray_dirs(u: np.ndarray, v: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.stack([
        (np.asarray(u, np.float64) - cx) / fx,
        (np.asarray(v, np.float64) - cy) / fy,
        np.ones(np.shape(u), np.float64),
    ], axis=-1)


def fit_plane(pts: np.ndarray, min_points: int = 40) -> Optional[Tuple[np.ndarray, float, float, float, int]]:
    """稳健平面拟合。返回 (单位法向, d, rms 米, mad 米, 内点数)，满足 n·X = d，法向朝相机。"""
    pts = np.asarray(pts, np.float64)
    if len(pts) < min_points:
        return None
    keep = np.ones(len(pts), dtype=bool)
    result = None
    for _ in range(4):
        used = pts[keep]
        if len(used) < min_points:
            break
        centroid = used.mean(axis=0)
        _, _, vt = np.linalg.svd(used - centroid, full_matrices=False)
        normal = vt[-1].copy()
        if normal[2] > 0:
            normal = -normal
        nrm = float(np.linalg.norm(normal))
        if nrm < 1e-12:
            return None
        normal /= nrm
        d = float(normal @ centroid)
        resid = pts @ normal - d
        sel = resid[keep]
        med = float(np.median(sel))
        mad = float(np.median(np.abs(sel - med))) * 1.4826
        rms = float(np.sqrt(np.mean((sel - med) ** 2)))
        result = (normal, d, rms, mad, int(keep.sum()))
        thresh = max(2.5 * mad, 5e-4)
        new_keep = np.abs(resid - med) <= thresh
        if int(new_keep.sum()) < min_points or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return result


def intersect_rays(dirs: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """dirs 的 z 分量为 1。返回 (N,3)，无效行为 NaN。交点的 z 就是 t。"""
    denom = dirs @ normal
    t = np.full(denom.shape, np.nan, np.float64)
    ok = np.abs(denom) > 1e-8
    t[ok] = d / denom[ok]
    t[~(t > 0)] = np.nan
    return dirs * t[:, None]


def _edge_pair_mm(corners3: np.ndarray) -> Tuple[float, float, float]:
    """返回 (边A毫米, 边B毫米, 对角线差毫米)。边A 是 0-1 与 3-2，边B 是 1-2 与 0-3。"""
    c = corners3

    def mean_len(i, j, k, l) -> float:
        return 0.5 * (float(np.linalg.norm(c[j] - c[i])) + float(np.linalg.norm(c[l] - c[k])))

    a = mean_len(0, 1, 3, 2) * 1000.0
    b = mean_len(1, 2, 0, 3) * 1000.0
    diag = abs(float(np.linalg.norm(c[2] - c[0])) - float(np.linalg.norm(c[3] - c[1]))) * 1000.0
    return a, b, diag


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normal / np.linalg.norm(normal)
    helper = np.array([0.0, 1.0, 0.0]) if abs(float(n[1])) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, n)
    x = x / np.linalg.norm(x)
    y = np.cross(n, x)
    return x, y


def _auto_rectangle(
    depth: np.ndarray,
    valid: np.ndarray,
    corners: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    normal: np.ndarray, d: float, rms: float,
) -> Dict[str, Any]:
    """在用户四边形周围找同一平面的外轮廓，量其三维最小外接矩形。"""
    h, w = depth.shape[:2]
    p = np.asarray(corners, np.float64).reshape(4, 2)
    x0 = int(np.floor(p[:, 0].min()))
    y0 = int(np.floor(p[:, 1].min()))
    x1 = int(np.ceil(p[:, 0].max()))
    y1 = int(np.ceil(p[:, 1].max()))
    pad = int(np.clip(0.8 * max(x1 - x0, y1 - y0), 12, 120))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad + 1), min(h, y1 + pad + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return {"auto_ok": False, "auto_warn": "搜索范围太小"}

    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    dirs = ray_dirs(xs, ys, fx, fy, cx, cy)
    plane_z = intersect_rays(dirs.reshape(-1, 3), normal, d).reshape(ys.shape + (3,))[:, :, 2]
    measured = depth[y0:y1, x0:x1]
    inside = valid[y0:y1, x0:x1] & np.isfinite(plane_z) & (plane_z > 0)
    thresh = float(min(max(2.5 * rms, 0.0015), 0.004))
    inliers = inside & (np.abs(measured - plane_z) <= thresh)

    full = np.zeros((h, w), np.uint8)
    full[y0:y1, x0:x1] = inliers.astype(np.uint8)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(full, connectivity=8)
    seed = fill_mask((h, w), inset_polygon(p, 0.25))
    labs = labels[seed]
    labs = labs[labs > 0]
    if nlab <= 1 or len(labs) == 0:
        return {"auto_ok": False, "auto_warn": "正面和周围深度分不开，自动找边失败"}
    label = int(np.bincount(labs).argmax())
    comp = (labels == label).astype(np.uint8)
    ys_c, xs_c = np.nonzero(comp)
    touch = bool(xs_c.min() <= x0 + 1 or ys_c.min() <= y0 + 1
                 or xs_c.max() >= x1 - 2 or ys_c.max() >= y1 - 2)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return {"auto_ok": False, "auto_warn": "没有找到外轮廓"}
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 80:
        return {"auto_ok": False, "auto_warn": "平面区域太小"}
    pix = cnt.reshape(-1, 2).astype(np.float64)
    xyz = intersect_rays(ray_dirs(pix[:, 0], pix[:, 1], fx, fy, cx, cy), normal, d)
    good = np.isfinite(xyz[:, 2])
    if int(good.sum()) < 20:
        return {"auto_ok": False, "auto_warn": "轮廓无法交到平面上"}
    xyz, pix = xyz[good], pix[good]
    ux, uy = _plane_basis(normal)
    plane_xy = np.stack([xyz @ ux, xyz @ uy], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(plane_xy)
    box2 = cv2.boxPoints(rect).astype(np.float64)
    box3 = box2[:, 0:1] * ux + box2[:, 1:2] * uy + d * normal
    long_m, short_m, _, _ = _rect_sides_corrected(box3, fx, fy, cx, cy)
    area = abs(float(cv2.contourArea(plane_xy.reshape(-1, 1, 2))))
    rw, rh = float(rect[1][0]), float(rect[1][1])
    fill = area / (rw * rh) if rw * rh > 1e-12 else 0.0
    box_uv = _project(box3, fx, fy, cx, cy)
    warn = ""
    ok = (not touch) and fill >= 0.82 and short_m > 0.005
    if touch:
        warn = "平面区域顶到了搜索边界，背景可能和物块贴在同一深度上，自动尺寸不要用"
    elif fill < 0.82:
        warn = "自动轮廓不像一块实心矩形，自动尺寸不要用"
    elif not ok:
        warn = "自动尺寸不可靠"
    return {
        "auto_ok": bool(ok),
        "auto_warn": warn,
        "auto_long_mm": long_m * 1000.0,
        "auto_short_mm": short_m * 1000.0,
        "auto_fill": float(fill),
        "auto_box_xy": box_uv,
    }


def _project(xyz: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    z = np.maximum(xyz[:, 2], 1e-9)
    return np.stack([fx * xyz[:, 0] / z + cx, fy * xyz[:, 1] / z + cy], axis=1)


def _rect_sides_corrected(box3: np.ndarray, fx: float, fy: float, cx: float, cy: float
                          ) -> Tuple[float, float, float, float]:
    """返回 (长边, 短边, 0-1 边, 1-2 边)，单位米。轮廓在像素中心上，每边补回 1 个像素。"""
    pix = _project(box3, fx, fy, cx, cy)
    lengths = []
    for i in (0, 1):
        i0, i1 = i, (i + 1) % 4
        j0, j1 = (i + 2) % 4, (i + 3) % 4
        m = 0.5 * (float(np.linalg.norm(box3[i1] - box3[i0])) + float(np.linalg.norm(box3[j1] - box3[j0])))
        px = 0.5 * (float(np.linalg.norm(pix[i1] - pix[i0])) + float(np.linalg.norm(pix[j1] - pix[j0])))
        if px > 1.0:
            m += m / px
        lengths.append(m)
    long_m, short_m = max(lengths), min(lengths)
    return long_m, short_m, lengths[0], lengths[1]


def measure_quad(
    depth_m: np.ndarray,
    valid: np.ndarray,
    corners_xy: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    baseline_mm: float,
    fx_full: float,
    inset: float = 0.12,
) -> Dict[str, Any]:
    """量正面四边形。corners_xy 与 depth 同一像素坐标系，顺序任意但必须是简单四边形。

    边长用四条角点射线和拟合平面的交点，不用角点上那个往往很脏的深度。
    """
    corners = np.asarray(corners_xy, np.float64).reshape(4, 2)
    out: Dict[str, Any] = {"ok": False, "error": "", "warnings": []}
    if not is_simple_quad(corners):
        out["error"] = "四个点交叉了，或框得太小。请按左上、右上、右下、左下重标，点可以稍进面里，不要点到背景。"
        return out

    inset_pts = inset_polygon(corners, inset)
    mask = fill_mask(depth_m.shape[:2], inset_pts)
    use = mask & np.asarray(valid, bool) & np.isfinite(depth_m) & (depth_m > 0)
    n_mask = int(mask.sum())
    n_valid = int(use.sum())
    ratio = (n_valid / n_mask) if n_mask else 0.0
    if n_valid < 40:
        out["error"] = f"框内有效深度只有 {n_valid} 个点。正面需要纹理，并且距离要落在相机量程里。"
        out["valid_ratio"] = ratio
        return out

    vs, us = np.nonzero(use)
    pts = unproject(us, vs, depth_m[vs, us], fx, fy, cx, cy)
    fitted = fit_plane(pts)
    if fitted is None:
        out["error"] = "平面拟合失败。"
        return out
    normal, d, rms, mad, n_in = fitted
    hits = intersect_rays(ray_dirs(corners[:, 0], corners[:, 1], fx, fy, cx, cy), normal, d)
    if not np.isfinite(hits).all():
        out["error"] = "有个角点的射线和平面不相交，请把四个点都点在正面上。"
        return out
    edge_a, edge_b, diag = _edge_pair_mm(hits)
    center = hits.mean(axis=0)
    z_mm = float(center[2] * 1000.0)
    dz = (z_mm ** 2) / (fx_full * baseline_mm) if fx_full > 0 and baseline_mm > 0 else float("nan")
    warnings: List[str] = []
    if abs(float(normal[2])) < 0.5:
        warnings.append("正面相对相机倾斜超过约 60°，边长仍可用，但平面度会被拉大。")
    if diag > 0.03 * max(edge_a, edge_b, 1.0):
        warnings.append(f"两条对角线差 {diag:.2f} mm，四个点不像同一个矩形的四个角。")
    if ratio < 0.6:
        warnings.append(f"框内只有 {ratio * 100:.0f}% 的像素有深度。纹理不够、太近，或点到了背景。")

    auto = _auto_rectangle(depth_m, valid, corners, fx, fy, cx, cy, normal, d, rms)
    if auto.get("auto_warn"):
        warnings.append(str(auto["auto_warn"]))
    out.update({
        "ok": True,
        "error": "",
        "warnings": warnings,
        "edge_a_mm": edge_a,
        "edge_b_mm": edge_b,
        "z_mm": z_mm,
        "plane_mm": abs(d) * 1000.0,
        "flat_rms_mm": rms * 1000.0,
        "flat_mad_mm": mad * 1000.0,
        "diag_diff_mm": diag,
        "inliers": n_in,
        "valid_ratio": ratio,
        "dz_px_mm": dz,
        "tilt_cos": abs(float(normal[2])),
        "inset_xy": inset_pts,
        "auto_ok": bool(auto.get("auto_ok", False)),
        "auto_long_mm": auto.get("auto_long_mm"),
        "auto_short_mm": auto.get("auto_short_mm"),
        "auto_fill": auto.get("auto_fill"),
        "auto_box_xy": auto.get("auto_box_xy"),
    })
    return out


def _depth_thresh(rms: float) -> float:
    """平面内点允许的深度偏差（米）。要盖过匹配噪声，又要和后面的背景分开。"""
    return float(min(max(3.0 * max(rms, 0.0), 0.003), 0.008))


def _snap_valid(depth: np.ndarray, valid: np.ndarray, u: float, v: float, radius: int = 28):
    h, w = depth.shape[:2]
    ui = int(np.clip(round(u), 0, w - 1))
    vi = int(np.clip(round(v), 0, h - 1))
    if valid[vi, ui] and depth[vi, ui] > 0:
        return ui, vi
    y0, y1 = max(0, vi - radius), min(h, vi + radius + 1)
    x0, x1 = max(0, ui - radius), min(w, ui + radius + 1)
    ys, xs = np.nonzero(valid[y0:y1, x0:x1] & (depth[y0:y1, x0:x1] > 0))
    if len(xs) == 0:
        return None
    d2 = (xs + x0 - ui) ** 2 + (ys + y0 - vi) ** 2
    k = int(np.argmin(d2))
    return int(xs[k] + x0), int(ys[k] + y0)


def _component_at(depth, valid, normal, d, thresh, fx, fy, cx, cy, seed_uv):
    h, w = depth.shape[:2]
    vs, us = np.nonzero(valid & np.isfinite(depth) & (depth > 0))
    if len(us) == 0:
        return None
    pz = intersect_rays(
        ray_dirs(us.astype(np.float64), vs.astype(np.float64), fx, fy, cx, cy), normal, d,
    )[:, 2]
    good = np.isfinite(pz) & (np.abs(depth[vs, us].astype(np.float64) - pz) <= thresh)
    if not np.any(good):
        return None
    mask = np.zeros((h, w), np.uint8)
    mask[vs[good], us[good]] = 1
    nlab, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nlab <= 1:
        return None
    ui = int(np.clip(round(seed_uv[0]), 0, w - 1))
    vi = int(np.clip(round(seed_uv[1]), 0, h - 1))
    lab = int(labels[vi, ui])
    if lab == 0:
        window = labels[max(0, vi - 15):min(h, vi + 16), max(0, ui - 15):min(w, ui + 16)]
        window = window[window > 0]
        if len(window) == 0:
            return None
        lab = int(np.bincount(window).argmax())
    comp = labels == lab
    if int(comp.sum()) < 80:
        return None
    return comp


def _rectangle_of(comp, normal, d, fx, fy, cx, cy):
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 80:
        return None
    pix = cnt.reshape(-1, 2).astype(np.float64)
    xyz = intersect_rays(ray_dirs(pix[:, 0], pix[:, 1], fx, fy, cx, cy), normal, d)
    good = np.isfinite(xyz[:, 2])
    if int(good.sum()) < 20:
        return None
    xyz = xyz[good]
    ux, uy = _plane_basis(normal)
    plane_xy = np.stack([xyz @ ux, xyz @ uy], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(plane_xy)
    box2 = cv2.boxPoints(rect).astype(np.float64)
    box3 = box2[:, 0:1] * ux + box2[:, 1:2] * uy + d * normal
    long_m, short_m, side0_m, side1_m = _rect_sides_corrected(box3, fx, fy, cx, cy)
    rw, rh = float(rect[1][0]), float(rect[1][1])
    area = abs(float(cv2.contourArea(plane_xy.reshape(-1, 1, 2))))
    fill = area / (rw * rh) if rw * rh > 1e-12 else 0.0
    return {
        "long_mm": long_m * 1000.0,
        "short_mm": short_m * 1000.0,
        "side0_mm": side0_m * 1000.0,
        "side1_mm": side1_m * 1000.0,
        "z_mm": float(np.mean(box3[:, 2]) * 1000.0),
        "fill": float(fill),
        "box_xy": _project(box3, fx, fy, cx, cy),
    }


def measure_from_seed(
    depth_m: np.ndarray,
    valid: np.ndarray,
    seed_xy: Sequence[float],
    fx: float, fy: float, cx: float, cy: float,
    baseline_mm: float,
    fx_full: float,
) -> Dict[str, Any]:
    """从画面上的一个点长出它所在的平面，直接给出这个面的长、短和距离。"""
    out: Dict[str, Any] = {"ok": False, "error": "", "warnings": []}
    snapped = _snap_valid(depth_m, np.asarray(valid, bool), float(seed_xy[0]), float(seed_xy[1]))
    if snapped is None:
        out["error"] = "这里没有深度。把物块放到画面中间，或点在物块正面上。"
        return out
    ui, vi = snapped
    h, w = depth_m.shape[:2]
    r = 20
    x0, x1 = max(0, ui - r), min(w, ui + r + 1)
    y0, y1 = max(0, vi - r), min(h, vi + r + 1)
    z0 = float(depth_m[vi, ui])
    patch = (valid[y0:y1, x0:x1] & np.isfinite(depth_m[y0:y1, x0:x1])
             & (np.abs(depth_m[y0:y1, x0:x1] - z0) < 0.03))
    ys, xs = np.nonzero(patch)
    if len(xs) < 40:
        out["error"] = "点的周围有效深度太少。正面贴一张有字的纸再测。"
        return out
    pts = unproject(xs + x0, ys + y0, depth_m[y0:y1, x0:x1][ys, xs], fx, fy, cx, cy)
    fitted = fit_plane(pts)
    if fitted is None:
        out["error"] = "这个位置拟合不出平面。"
        return out
    normal, d, rms, mad, _n = fitted
    comp = _component_at(depth_m, valid, normal, d, _depth_thresh(rms), fx, fy, cx, cy, (ui, vi))
    if comp is None:
        out["error"] = "没有找到一块完整的平面。点在物块正面中央。"
        return out
    cvs, cus = np.nonzero(comp)
    sample = (cvs, cus)
    if len(cus) > 5000:
        sel = np.linspace(0, len(cus) - 1, 5000).astype(int)
        sample = (cvs[sel], cus[sel])
    pts2 = unproject(sample[1], sample[0], depth_m[sample[0], sample[1]], fx, fy, cx, cy)
    fitted2 = fit_plane(pts2)
    if fitted2 is not None:
        normal, d, rms, mad, _n = fitted2
        comp2 = _component_at(depth_m, valid, normal, d, _depth_thresh(rms), fx, fy, cx, cy, (ui, vi))
        if comp2 is not None:
            comp = comp2
    rect = _rectangle_of(comp, normal, d, fx, fy, cx, cy)
    if rect is None or rect["short_mm"] < 5.0:
        out["error"] = "这块平面的轮廓太碎，量不出边长。"
        return out
    ys_c, xs_c = np.nonzero(comp)
    touch = bool(xs_c.min() <= 1 or ys_c.min() <= 1 or xs_c.max() >= w - 2 or ys_c.max() >= h - 2)
    warnings: List[str] = []
    if touch:
        warnings.append("这块平面延伸到画面外面，显示的是画面里能看到的那一段。")
    if rect["fill"] < 0.75:
        warnings.append("正面有一大块没有深度，尺寸可能偏小。给正面加一些纹理。")
    dz = (rect["z_mm"] ** 2) / (fx_full * baseline_mm) if fx_full > 0 and baseline_mm > 0 else float("nan")
    out.update({
        "ok": True,
        "warnings": warnings,
        "long_mm": rect["long_mm"],
        "short_mm": rect["short_mm"],
        "edge_a_mm": rect["long_mm"],
        "edge_b_mm": rect["short_mm"],
        "side0_mm": rect["side0_mm"],
        "side1_mm": rect["side1_mm"],
        "z_mm": rect["z_mm"],
        "plane_mm": abs(float(d)) * 1000.0,
        "flat_rms_mm": float(rms) * 1000.0,
        "flat_mad_mm": float(mad) * 1000.0,
        "fill": rect["fill"],
        "auto_fill": rect["fill"],
        "auto_ok": (not touch) and rect["fill"] >= 0.75,
        "auto_long_mm": rect["long_mm"],
        "auto_short_mm": rect["short_mm"],
        "box_xy": rect["box_xy"],
        "dz_px_mm": dz,
        "inliers": int(comp.sum()),
    })
    return out


def match_lengths(
    len_a: float, len_b: float,
    true_w: Optional[float], true_h: Optional[float],
) -> Optional[Dict[str, float]]:
    """把两条测得边配到真值上，使两边相对误差之和最小。"""
    truths = [t for t in (true_w, true_h) if t is not None and t > 0]
    if len(truths) < 2:
        return None
    best = None
    for ta, tb in ((truths[0], truths[1]), (truths[1], truths[0])):
        e1 = (len_a / ta - 1.0) * 100.0
        e2 = (len_b / tb - 1.0) * 100.0
        score = abs(e1) + abs(e2)
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "m1": len_a, "t1": ta, "e1_mm": len_a - ta, "e1_pct": e1,
                "m2": len_b, "t2": tb, "e2_mm": len_b - tb, "e2_pct": e2,
                "scale_pct": 0.5 * (e1 + e2),
            }
    return best


# -----------------------------------------------------------------------------
# 自检（不打开相机、不加载模型）
# -----------------------------------------------------------------------------
def _paint_plane(h: int, w: int, corners3: np.ndarray,
                 fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv = _project(corners3, fx, fy, cx, cy)
    normal, d, _, _, _ = fit_plane(corners3, min_points=3)
    mask = fill_mask((h, w), uv)
    ys, xs = np.nonzero(mask)
    xyz = intersect_rays(ray_dirs(xs, ys, fx, fy, cx, cy), normal, d)
    depth = np.zeros((h, w), np.float32)
    valid = np.zeros((h, w), bool)
    ok = np.isfinite(xyz[:, 2])
    depth[ys[ok], xs[ok]] = xyz[ok, 2].astype(np.float32)
    valid[ys[ok], xs[ok]] = True
    return depth, valid, uv


def _check(name: str, got: float, exp: float, tol: float, fails: List[str]) -> None:
    err = abs(got - exp)
    ok = err <= tol
    flag = "通过" if ok else "失败"
    print(f"  [{flag}] {name}: 测得 {got:.3f}  期望 {exp:.3f}  差 {err:.3f}  容差 {tol:.3f}")
    if not ok:
        fails.append(name)


def run_selftest() -> int:
    """合成一块已知长方体正面，确认边长和距离能被算回来。"""
    print("=" * 72)
    print("  物块测量几何自检（合成数据，不开相机）")
    print("=" * 72)
    fx = fy = 547.017
    cx, cy = 320.0, 240.0
    h, w = 480, 640
    baseline_mm = 64.891
    fails: List[str] = []

    z = 0.400
    face = np.array([
        [-0.050, -0.030, z],
        [0.050, -0.030, z],
        [0.050, 0.030, z],
        [-0.050, 0.030, z],
    ], np.float64)
    depth, valid, uv = _paint_plane(h, w, face, fx, fy, cx, cy)
    m = measure_quad(depth, valid, uv, fx, fy, cx, cy, baseline_mm, fx, inset=0.12)
    if not m["ok"]:
        print("  [失败] 正对平面没有测出结果：" + m["error"])
        return 1
    matched = match_lengths(m["edge_a_mm"], m["edge_b_mm"], 100.0, 60.0)
    _check("正对 手动长边", max(m["edge_a_mm"], m["edge_b_mm"]), 100.0, 0.20, fails)
    _check("正对 手动短边", min(m["edge_a_mm"], m["edge_b_mm"]), 60.0, 0.20, fails)
    _check("正对 面心深度", m["z_mm"], 400.0, 0.20, fails)
    _check("正对 尺度误差%", matched["scale_pct"], 0.0, 0.15, fails)
    if m.get("auto_ok"):
        _check("正对 自动长边", m["auto_long_mm"], 100.0, 1.50, fails)
        _check("正对 自动短边", m["auto_short_mm"], 60.0, 1.50, fails)
    else:
        print("  [失败] 正对平面的自动找边没有通过: " + "; ".join(m["warnings"]))
        fails.append("auto")

    center_seed = measure_from_seed(depth, valid, (cx, cy), fx, fy, cx, cy, baseline_mm, fx)
    if not center_seed.get("ok"):
        print("  [失败] 画面中央没有量出物块：" + center_seed.get("error", ""))
        fails.append("seed")
    else:
        _check("中央点选 长", center_seed["long_mm"], 100.0, 1.50, fails)
        _check("中央点选 短", center_seed["short_mm"], 60.0, 1.50, fails)
        _check("中央点选 距离", center_seed["z_mm"], 400.0, 1.00, fails)

    depth_bg = np.full((h, w), 0.80, np.float32)
    valid_bg = np.ones((h, w), bool)
    depth_bg[valid] = depth[valid]
    behind = measure_from_seed(depth_bg, valid_bg, (cx, cy), fx, fy, cx, cy, baseline_mm, fx)
    if not behind.get("ok"):
        print("  [失败] 有背景时没有量出物块：" + behind.get("error", ""))
        fails.append("seed-bg")
    else:
        _check("有背景 长", behind["long_mm"], 100.0, 1.50, fails)
        _check("有背景 短", behind["short_mm"], 60.0, 1.50, fails)
        if behind.get("long_mm", 0) > 150:
            print("  [失败] 尺寸把背景墙也吃进去了")
            fails.append("seed-bg-leak")

    ang = np.deg2rad(25.0)
    rot = np.array([
        [np.cos(ang), 0.0, np.sin(ang)],
        [0.0, 1.0, 0.0],
        [-np.sin(ang), 0.0, np.cos(ang)],
    ])
    local = face.copy()
    local[:, 2] = 0.0
    tilted = (rot @ local.T).T + np.array([0.02, -0.01, 0.450])
    depth_t, valid_t, uv_t = _paint_plane(h, w, tilted, fx, fy, cx, cy)
    mt = measure_quad(depth_t, valid_t, uv_t, fx, fy, cx, cy, baseline_mm, fx, inset=0.12)
    if not mt["ok"]:
        print("  [失败] 倾斜平面没有测出结果：" + mt["error"])
        fails.append("tilt")
    else:
        _check("倾斜 手动长边", max(mt["edge_a_mm"], mt["edge_b_mm"]), 100.0, 0.40, fails)
        _check("倾斜 手动短边", min(mt["edge_a_mm"], mt["edge_b_mm"]), 60.0, 0.40, fails)

    depth_s = np.where(valid, depth * np.float32(1.02), np.float32(0))
    ms = measure_quad(depth_s, valid, uv, fx, fy, cx, cy, baseline_mm, fx, inset=0.12)
    if ms["ok"]:
        _check("深度放大 2% 后的长边", max(ms["edge_a_mm"], ms["edge_b_mm"]), 102.0, 0.30, fails)
        _check("深度放大 2% 后的面心", ms["z_mm"], 408.0, 0.30, fails)
    else:
        print("  [失败] 缩放测试没有测出结果")
        fails.append("scale")

    rng = np.random.default_rng(0)
    noisy = depth.copy()
    noisy[valid] = depth[valid] + rng.normal(0.0, 0.0003, int(valid.sum())).astype(np.float32)
    mn = measure_quad(noisy, valid, uv, fx, fy, cx, cy, baseline_mm, fx, inset=0.12)
    if mn["ok"]:
        ok_noise = 0.15 <= mn["flat_rms_mm"] <= 0.55
        print(f"  [{'通过' if ok_noise else '失败'}] 0.30 mm 噪声的平面度 RMS = {mn['flat_rms_mm']:.3f} mm")
        if not ok_noise:
            fails.append("noise")
    else:
        fails.append("noise")

    bow = uv.copy()
    bow[2], bow[3] = uv[3].copy(), uv[2].copy()
    mb = measure_quad(depth, valid, bow, fx, fy, cx, cy, baseline_mm, fx)
    ok_bow = (not mb["ok"])
    print(f"  [{'通过' if ok_bow else '失败'}] 交叉四点被拒绝")
    if not ok_bow:
        fails.append("bowtie")

    overlay = np.zeros((220, 480, 3), np.uint8)
    _paint_readout(overlay, ["长    100.0 mm", "短    60.0 mm", "距离  400 mm"])
    ok_text = int(overlay.sum()) > 0
    print(f"  [{'通过' if ok_text else '失败'}] 画面上能写出尺寸")
    if not ok_text:
        fails.append("overlay")

    print("-" * 72)
    if fails:
        print(f"  自检失败 {len(fails)} 项: {', '.join(fails)}")
        return 1
    print("  自检通过。手动四点能找回边长和距离；深度整体放大时，边长同比放大。")
    return 0


# -----------------------------------------------------------------------------
# 报告
# -----------------------------------------------------------------------------
def _fmt_pct(v: float) -> str:
    return f"{v:+.2f}%"


def describe(m: Dict[str, Any], true_w: Optional[float], true_h: Optional[float],
             true_z: Optional[float], click_px: Optional[Tuple[float, float]] = None) -> List[str]:
    """中文结果，打印到控制台。"""
    if not m.get("ok"):
        return ["没有量出结果：" + str(m.get("error", ""))]
    lines = [
        f"手动四点  边A {m['edge_a_mm']:.2f} mm    边B {m['edge_b_mm']:.2f} mm",
        f"面心深度 {m['z_mm']:.2f} mm    到平面的垂直距离 {m['plane_mm']:.2f} mm",
        f"平面度 RMS {m['flat_rms_mm']:.3f} mm（MAD {m['flat_mad_mm']:.3f} mm）"
        f"    这个距离上 1 个像素视差约 {m['dz_px_mm']:.2f} mm",
    ]
    matched = match_lengths(m["edge_a_mm"], m["edge_b_mm"], true_w, true_h)
    if matched:
        lines.append(
            f"手动尺度  {matched['m1']:.2f}/{matched['t1']:.2f} mm = {_fmt_pct(matched['e1_pct'])}"
            f"    {matched['m2']:.2f}/{matched['t2']:.2f} mm = {_fmt_pct(matched['e2_pct'])}"
            f"    平均 {_fmt_pct(matched['scale_pct'])}"
        )
        if abs(matched["e1_pct"] - matched["e2_pct"]) > 2.0:
            lines.append("两边的相对误差差超过 2 个百分点：角点可能不在同一块矩形的边上，建议重标。")
    if m.get("auto_long_mm") and m.get("auto_short_mm"):
        tag = "可用" if m.get("auto_ok") else "不要用"
        lines.append(
            f"自动轮廓[{tag}]  长边 {m['auto_long_mm']:.2f} mm    短边 {m['auto_short_mm']:.2f} mm"
            f"    实心度 {float(m.get('auto_fill') or 0):.2f}"
        )
        am = match_lengths(m["auto_long_mm"], m["auto_short_mm"], true_w, true_h)
        if am and m.get("auto_ok"):
            lines.append(f"自动尺度  平均 {_fmt_pct(am['scale_pct'])}    "
                         f"（长边 {_fmt_pct(am['e1_pct']) if am['m1'] >= am['m2'] else _fmt_pct(am['e2_pct'])}，"
                         f"短边 {_fmt_pct(am['e2_pct']) if am['m1'] >= am['m2'] else _fmt_pct(am['e1_pct'])}）")
    if true_z and true_z > 0:
        lines.append(
            f"卷尺 {true_z:.1f} mm 对比面心深度 {m['z_mm'] - true_z:+.2f} mm。"
            "卷尺从外壳量起，和光心不是同一个点，这项只作对照。"
        )
    if click_px is not None:
        pa, pb = click_px
        if pa > 1 and pb > 1:
            sa = m["edge_a_mm"] / pa
            sb = m["edge_b_mm"] / pb
            lines.append(f"点击敏感度  边A 1 像素约 {sa:.2f} mm，边B 1 像素约 {sb:.2f} mm。"
                         "这是你点偏带来的误差，不是相机噪声。")
    for w in m.get("warnings") or []:
        lines.append("注意  " + str(w))
    return lines


def ascii_lines(m: Dict[str, Any], true_w: Optional[float], true_h: Optional[float],
                true_z: Optional[float], frozen: bool, n_pts: int, infer_ms: float,
                repeat: Optional[Dict[str, Any]]) -> List[Tuple[str, Tuple[int, int, int]]]:
    """画在画面上的 ASCII。OpenCV 的 putText 写不出中文。"""
    g, y, r, c = (80, 220, 120), (0, 220, 255), (60, 60, 255), (230, 230, 230)
    rows: List[Tuple[str, Tuple[int, int, int]]] = [(
        "SPACE freeze   mouse: TL TR BR BL    U undo   C clear   R repeat   S save   Q quit",
        c,
    )]
    state = "FROZEN" if frozen else "LIVE"
    rows.append((f"{state}   points {n_pts}/4   infer {infer_ms:.0f} ms", y if frozen else g))
    if not m or not m.get("ok"):
        rows.append((m.get("error", "Freeze, then click the 4 corners of the front face.") if m else
                     "Freeze, then click the 4 corners of the front face.", c))
        return rows
    rows.append((f"manual  A {m['edge_a_mm']:.2f} mm    B {m['edge_b_mm']:.2f} mm", c))
    matched = match_lengths(m["edge_a_mm"], m["edge_b_mm"], true_w, true_h)
    if matched:
        color = g if abs(matched["scale_pct"]) < 0.5 else (y if abs(matched["scale_pct"]) < 1.5 else r)
        rows.append((f"manual scale {matched['scale_pct']:+.2f}%   "
                     f"({matched['e1_pct']:+.2f}% , {matched['e2_pct']:+.2f}%)", color))
    if m.get("auto_long_mm"):
        tag = "OK" if m.get("auto_ok") else "REJECT"
        rows.append((f"auto[{tag}]  long {m['auto_long_mm']:.2f} mm   short {m['auto_short_mm']:.2f} mm", c))
        am = match_lengths(m["auto_long_mm"], m["auto_short_mm"], true_w, true_h)
        if am and m.get("auto_ok"):
            color = g if abs(am["scale_pct"]) < 0.5 else (y if abs(am["scale_pct"]) < 1.5 else r)
            rows.append((f"auto scale {am['scale_pct']:+.2f}%", color))
    zline = f"Z {m['z_mm']:.1f} mm   flat RMS {m['flat_rms_mm']:.3f} mm   1px -> {m['dz_px_mm']:.2f} mm"
    if true_z and true_z > 0:
        zline += f"   tape {m['z_mm'] - true_z:+.1f} mm"
    rows.append((zline, c))
    if repeat and repeat.get("n", 0) >= 2:
        rows.append((
            f"repeat x{repeat['n']}  Z {repeat['z_mm_mean']:.2f} +/- {repeat['z_mm_std']:.3f} mm"
            f"   A +/- {repeat.get('edge_a_mm_std', 0):.3f}   B +/- {repeat.get('edge_b_mm_std', 0):.3f} mm",
            g,
        ))
    if m.get("warnings"):
        rows.append(("see console for warnings", y))
    return rows


def _put(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


def aggregate(measures: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("edge_a_mm", "edge_b_mm", "z_mm", "flat_rms_mm", "auto_long_mm", "auto_short_mm")
    out: Dict[str, Any] = {"n": len(measures)}
    for key in keys:
        vals = [float(m[key]) for m in measures
                if m.get("ok") and m.get(key) is not None and (not str(key).startswith("auto") or m.get("auto_ok"))]
        if not vals:
            continue
        out[key + "_mean"] = float(np.mean(vals))
        out[key + "_std"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
        out[key + "_n"] = len(vals)
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# -----------------------------------------------------------------------------
# 实机
# -----------------------------------------------------------------------------
def _instructions(args: argparse.Namespace, scale: float, iters: int) -> None:
    print("=" * 72)
    print("  物块尺寸")
    print("=" * 72)
    print("  把物块正面朝向相机，放到画面中间。左上角直接显示长、短和距离。")
    print("  点一下正面，就改量那一块。空格冻结。S 保存这一帧。Q 退出。")
    print("  正面贴一张有字的纸。物块和后面的东西隔开几厘米，否则会把背景一起量进去。")
    if args.width_mm and args.height_mm:
        print(f"  卡尺真值  宽 {args.width_mm:.3f} mm    高 {args.height_mm:.3f} mm"
              "。画面上会多一行尺度误差。")
    print(f"  推理档 scale={scale:g}  iters={iters}。要更贴边可用 --scale 1.0 --iters 8。")
    print("  正在打开相机并加载模型……")


def _load_runtime(args: argparse.Namespace):
    import yaml

    from camera import CameraError, StereoCamera
    from depth_utils import CameraParams, disp_correction_from_config, disparity_to_depth, load_rectifier
    from stereo_depth import FFSDepthEstimator, FFSError

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    project = dict(cfg.get("project", {}) or {})
    for key in ("ffs_repo", "output_dir", "weights_dir", "calibration_dir"):
        if key in project:
            project[key] = os.path.normpath(os.path.join(PROJECT_ROOT, project[key]))
    ffs = dict(cfg.get("ffs", {}) or {})
    if args.scale is not None:
        ffs["scale"] = float(args.scale)
    if args.iters is not None:
        ffs["valid_iters"] = int(args.iters)
    depth_cfg = cfg.get("depth", {}) or {}
    return cfg, project, ffs, depth_cfg, CameraParams, disp_correction_from_config, disparity_to_depth, load_rectifier, StereoCamera, CameraError, FFSDepthEstimator, FFSError


def _depth_of(left, right, est, rectifier, params, disparity_to_depth, disp_scale, disp_offset, max_valid, zmin, zmax):
    if rectifier is not None:
        left, right = rectifier.rectify_pair(left, right)
    out = est.infer(left, right)
    disp = out["disp"]
    full_h, full_w = left.shape[:2]
    sx = disp.shape[1] / float(full_w)
    sy = disp.shape[0] / float(full_h)
    from depth_utils import CameraParams
    p_use = CameraParams(
        fx=params.fx * sx, fy=params.fy * sy, cx=params.cx * sx, cy=params.cy * sy,
        baseline_m=params.baseline_m, calibrated=params.calibrated,
        source=params.source, width=disp.shape[1], height=disp.shape[0],
    )
    depth, valid = disparity_to_depth(
        disp, p_use, max_valid_disp=max_valid, disp_scale=disp_scale, disp_offset=disp_offset,
    )
    valid = valid & np.isfinite(depth) & (depth >= zmin) & (depth <= zmax)
    return left, depth, valid, p_use, float(out["timings_ms"]["infer"])


def _scale_corners(corners: Sequence[Tuple[float, float]], sx: float, sy: float) -> np.ndarray:
    p = np.asarray(corners, np.float64).reshape(-1, 2)
    return np.stack([p[:, 0] * sx, p[:, 1] * sy], axis=1)


def _wrap_zh(text: str, width: int = 16) -> List[str]:
    text = str(text or "")
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


def _readout_lines(measure: Optional[Dict[str, Any]], true_w: Optional[float], true_h: Optional[float],
                   repeat: Optional[Dict[str, Any]]) -> List[str]:
    if not measure or not measure.get("ok"):
        err = (measure or {}).get("error") or "把物块正面放到画面中间"
        return _wrap_zh(err) + ["或点一下要量的那个面"]
    lines = [
        f"长    {measure['long_mm']:.1f} mm",
        f"短    {measure['short_mm']:.1f} mm",
        f"距离  {measure['z_mm']:.0f} mm",
    ]
    matched = match_lengths(measure["long_mm"], measure["short_mm"], true_w, true_h)
    if matched:
        lines.append(f"误差  {matched['scale_pct']:+.2f}%")
    if repeat and repeat.get("n", 0) >= 2 and repeat.get("z_mm_std") is not None:
        lines.append(f"跳动  距离 ±{repeat['z_mm_std']:.2f} mm")
    for warning in (measure.get("warnings") or [])[:1]:
        lines.extend(_wrap_zh(warning, 18))
    lines.append("点正面更换   空格冻结   Q退出")
    return lines


def _paint_readout(bgr: np.ndarray, lines: Sequence[str], origin: Tuple[int, int] = (12, 12)) -> None:
    """把尺寸直接写到画面上。OpenCV 画不出中文，这里用系统字体。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        y = origin[1] + 22
        for line in lines:
            _put(bgr, line, (origin[0], y), (0, 255, 255))
            y += 26
        return
    font_path = ""
    for candidate in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
                      r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.isfile(candidate):
            font_path = candidate
            break
    size = 28 if font_path else 18
    font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    tw = max((box[2] - box[0]) for box in boxes) if boxes else 40
    line_h = [box[3] - box[1] for box in boxes]
    gap = 6
    th = sum(line_h) + gap * max(len(lines) - 1, 0) + 20
    x, y = origin
    box = [x, y, x + tw + 28, y + th]
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(box, radius=8, fill=(16, 16, 16))
    else:
        draw.rectangle(box, fill=(16, 16, 16))
    cy = y + 10
    for line, lh in zip(lines, line_h):
        fill = (255, 214, 80) if line.startswith(("长", "短", "距")) else (230, 230, 230)
        draw.text((x + 14, cy), line, font=font, fill=fill)
        cy += lh + gap
    bgr[:] = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _draw(left: np.ndarray, depth: np.ndarray, valid: np.ndarray, p_use,
          seed_xy: Tuple[float, float], measure: Optional[Dict[str, Any]],
          lines: Sequence[str]) -> np.ndarray:
    from depth_utils import colorize_depth

    h, w = left.shape[:2]
    vals = depth[valid] if np.any(valid) else None
    if vals is not None and len(vals) > 20:
        lo, hi = np.percentile(vals, [5, 95])
        if hi - lo < 0.02:
            lo, hi = float(lo) - 0.01, float(hi) + 0.01
    else:
        lo, hi = 0.2, 1.0
    depth_full = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    valid_full = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    vis = colorize_depth(depth_full, valid_full, float(lo), float(hi))
    view = left.copy()
    sx = p_use.width / float(w) if p_use is not None and p_use.width else depth.shape[1] / float(w)
    sy = p_use.height / float(h) if p_use is not None and p_use.height else depth.shape[0] / float(h)

    def to_full(pts) -> np.ndarray:
        q = np.asarray(pts, np.float64).reshape(-1, 2).copy()
        q[:, 0] /= max(sx, 1e-6)
        q[:, 1] /= max(sy, 1e-6)
        return np.round(q).astype(np.int32)

    if measure and measure.get("ok") and measure.get("box_xy") is not None:
        poly = to_full(measure["box_xy"])
        cv2.polylines(view, [poly], True, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.polylines(vis, [poly], True, (255, 200, 0), 2, cv2.LINE_AA)
        for index, key in ((0, "side0_mm"), (1, "side1_mm")):
            if measure.get(key) is None:
                continue
            a = poly[index]
            b = poly[(index + 1) % 4]
            mid = (int((a[0] + b[0]) / 2) + 4, int((a[1] + b[1]) / 2) - 4)
            _put(view, f"{measure[key]:.1f}", mid, (255, 255, 255))
            _put(vis, f"{measure[key]:.1f}", mid, (255, 255, 255))
    sx_i, sy_i = int(round(seed_xy[0])), int(round(seed_xy[1]))
    cv2.drawMarker(view, (sx_i, sy_i), (0, 255, 255), cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)
    cv2.drawMarker(vis, (sx_i, sy_i), (0, 255, 255), cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)
    _paint_readout(view, lines)
    return np.hstack([view, vis])


def run_measure(args: argparse.Namespace) -> int:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    (cfg, project, ffs, depth_cfg, CameraParams, disp_correction_from_config,
     disparity_to_depth, load_rectifier, StereoCamera, CameraError,
     FFSDepthEstimator, FFSError) = _load_runtime(args)
    _instructions(args, float(ffs.get("scale", 1.0)), int(ffs.get("valid_iters", 8)))

    try:
        rectifier = load_rectifier(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 校正器加载失败：{exc}")
        return 1
    params = CameraParams.from_config(cfg)
    if not params.calibrated:
        print("[错误] config.yaml 里 depth.calibrated 不是 true。未标定的深度不能用来测精度。")
        return 1
    disp_scale, disp_offset = disp_correction_from_config(cfg)
    ratio = float(depth_cfg.get("disp_extrapolation_ratio", 0.98) or 0.0)
    zmin = float(depth_cfg.get("zmin_m", 0.15) or 0.15)
    zmax = float(depth_cfg.get("zmax_m", 3.0) or 3.0)

    cam = None
    est = None
    still = None
    try:
        est = FFSDepthEstimator(project, ffs)
        ok, msg = est.weights_ready()
        if not ok:
            print("[错误] " + msg)
            return 1
        est.load()
        if args.left and args.right:
            left0 = cv2.imread(args.left)
            right0 = cv2.imread(args.right)
            if left0 is None or right0 is None:
                print("[错误] 读不到 --left / --right 图片")
                return 1
            still = (left0, right0)
        else:
            cam = StereoCamera(cfg.get("camera", {}) or {})
            cam.open()
        max_valid = (est.max_disp * ratio) if ratio > 0 else None

        def grab():
            if still is not None:
                return still
            frame = cam.read()
            return frame.left, frame.right

        left0, right0 = grab()
        est.warmup(1, left=left0, right=right0)
        print(params.describe())
        print(f"视差校正  scale={disp_scale}  offset={disp_offset}")
        near = params.fx * params.baseline_m * 1000.0 / max(est.max_disp * max(ratio, 0.5), 1.0)
        print(f"深度范围 {zmin:.2f}~{zmax:.2f} m。视差搜索对应的最近大约 {near:.0f} mm，再近的结果不可信。")
        print("-" * 72)
    except CameraError as exc:
        print(f"[错误] 相机打开失败：{exc}")
        return 1
    except FFSError as exc:
        print(f"[错误] 模型不可用：{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 初始化失败：{type(exc).__name__}: {exc}")
        if cam is not None:
            cam.close()
        return 1

    win = "block size"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = {
        "seed": None,
        "frozen": still is not None,
        "left": None,
        "depth": None,
        "valid": None,
        "p_use": None,
        "measure": None,
        "infer_ms": 0.0,
        "repeat": None,
        "h": 0,
        "w": 0,
    }
    samples: List[Dict[str, Any]] = []
    session = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PROJECT_ROOT, "output", "measure_block")
    os.makedirs(out_dir, exist_ok=True)

    def seed_full() -> Tuple[float, float]:
        if state["seed"] is None:
            return (state["w"] / 2.0, state["h"] / 2.0)
        return state["seed"]

    def on_mouse(event, x, y, _flags, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or state["h"] <= 0:
            return
        if not (0 <= y < state["h"]):
            return
        if x >= state["w"]:
            x -= state["w"]
        if 0 <= x < state["w"]:
            state["seed"] = (float(x), float(y))

    cv2.setMouseCallback(win, on_mouse)

    def measure_seed() -> None:
        if state["depth"] is None or state["p_use"] is None:
            return
        p = state["p_use"]
        su, sv = seed_full()
        seed_d = (su * p.fx / params.fx, sv * p.fy / params.fy)
        state["measure"] = measure_from_seed(
            state["depth"], state["valid"], seed_d,
            p.fx, p.fy, p.cx, p.cy, params.baseline_m * 1000.0, params.fx,
        )

    def capture_into_state(left, right) -> None:
        left_r, depth, valid, p_use, infer_ms = _depth_of(
            left, right, est, rectifier, params, disparity_to_depth,
            disp_scale, disp_offset, max_valid, zmin, zmax,
        )
        state["left"], state["depth"], state["valid"] = left_r, depth, valid
        state["p_use"], state["infer_ms"] = p_use, infer_ms
        state["h"], state["w"] = left_r.shape[:2]

    try:
        while True:
            if not state["frozen"] or state["left"] is None:
                capture_into_state(*grab())
            measure_seed()
            lines = _readout_lines(state["measure"], args.width_mm, args.height_mm, state["repeat"])
            canvas = _draw(state["left"], state["depth"], state["valid"], state["p_use"],
                           seed_full(), state["measure"], lines)
            cv2.imshow(win, canvas)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                if still is not None:
                    continue
                state["frozen"] = not state["frozen"]
                if state["frozen"] and state["measure"] and state["measure"].get("ok"):
                    m = state["measure"]
                    print(f"  冻结  长 {m['long_mm']:.1f} mm    短 {m['short_mm']:.1f} mm    "
                          f"距离 {m['z_mm']:.0f} mm")
            elif key in (ord("c"), ord("C")):
                state["seed"] = None
                state["repeat"] = None
                measure_seed()
            elif key in (ord("r"), ord("R")):
                if still is not None:
                    print("  离线图片只有一帧，不能看跳动。")
                    continue
                got = []
                print(f"  连测 {args.repeat} 帧，物块不要动……")
                for i in range(int(args.repeat)):
                    capture_into_state(*grab())
                    measure_seed()
                    m = state["measure"] or {}
                    if m.get("ok"):
                        got.append(m)
                    print(f"    {i + 1}/{args.repeat}  " + ("ok" if m.get("ok") else m.get("error", "")))
                if not got:
                    print("  这几帧都没有量出来。")
                    continue
                state["measure"] = got[-1]
                state["repeat"] = aggregate(got)
                rep = state["repeat"]
                print(f"  {rep['n']} 帧  长 {rep['edge_a_mm_mean']:.1f} ± {rep['edge_a_mm_std']:.2f} mm"
                      f"    短 {rep['edge_b_mm_mean']:.1f} ± {rep['edge_b_mm_std']:.2f} mm"
                      f"    距离 {rep['z_mm_mean']:.1f} ± {rep['z_mm_std']:.2f} mm")
            elif key in (ord("s"), ord("S")):
                if not state["measure"] or not state["measure"].get("ok"):
                    print("  还没有量到物块。")
                    continue
                su, sv = seed_full()
                rec = {
                    "time": time.strftime("%H:%M:%S"),
                    "seed_xy": [float(su), float(sv)],
                    "single": _jsonable(state["measure"]),
                    "repeat": state["repeat"],
                }
                samples.append(rec)
                _write_session(out_dir, session, args, ffs, params, disp_scale, samples)
                shot = os.path.join(out_dir, f"{session}_{len(samples):02d}.png")
                cv2.imwrite(shot, canvas)
                m = state["measure"]
                print(f"  已保存  长 {m['long_mm']:.1f} mm  短 {m['short_mm']:.1f} mm  "
                      f"距离 {m['z_mm']:.0f} mm    {shot}")
    finally:
        if cam is not None:
            cam.close()
        if est is not None:
            est.close()
        cv2.destroyAllWindows()

    _write_session(out_dir, session, args, ffs, params, disp_scale, samples)
    _print_summary(samples, args)
    report = os.path.join(out_dir, session + ".json")
    if samples:
        print(f"报告  {report}")
    return 0


def _write_session(out_dir, session, args, ffs, params, disp_scale, samples) -> None:
    if not samples:
        return
    path = os.path.join(out_dir, session + ".json")
    payload = {
        "session": session,
        "width_mm": args.width_mm,
        "height_mm": args.height_mm,
        "distance_mm": args.distance_mm,
        "scale": float(ffs.get("scale", 1)),
        "valid_iters": int(ffs.get("valid_iters", 0)),
        "fx": params.fx,
        "baseline_mm": params.baseline_m * 1000.0,
        "disp_scale": disp_scale,
        "samples": samples,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, ensure_ascii=False, indent=2)


def _print_summary(samples: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    print("=" * 72)
    if not samples:
        print("  没有按 S 记录。刚才屏幕上的结果没有写入文件。")
        return
    print(f"  已记录 {len(samples)} 组")
    scales = []
    for i, rec in enumerate(samples, 1):
        src = rec["single"]
        use_auto = bool(src.get("auto_ok"))
        if use_auto:
            matched = match_lengths(src["auto_long_mm"], src["auto_short_mm"], args.width_mm, args.height_mm)
            kind = "自动"
        else:
            matched = match_lengths(src["edge_a_mm"], src["edge_b_mm"], args.width_mm, args.height_mm)
            kind = "手动"
        rep = rec.get("repeat") or {}
        extra = f"    重复性 Z ±{rep['z_mm_std']:.3f} mm" if rep.get("z_mm_std") is not None else ""
        if matched:
            scales.append(matched["scale_pct"])
            print(f"  #{i} {kind}尺度 {_fmt_pct(matched['scale_pct'])}   "
                  f"Z {src['z_mm']:.1f} mm   平面度 {src['flat_rms_mm']:.3f} mm{extra}")
        else:
            print(f"  #{i} {kind} 边 {src['edge_a_mm']:.2f} / {src['edge_b_mm']:.2f} mm   "
                  f"Z {src['z_mm']:.1f} mm{extra}")
    if scales:
        print(f"  各组尺度误差平均 {_fmt_pct(float(np.mean(scales)))}    "
              f"组间标准差 {float(np.std(scales, ddof=1)) if len(scales) > 1 else 0:.2f} 个百分点")
        print("  组间标准差主要来自每次点击的差别。平面度才是相机本身的深度噪声。")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="用已知尺寸的物块测量双目尺度误差、平面度和重复性")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config.yaml"))
    p.add_argument("--width-mm", type=float, default=None, help="卡尺量的正面宽，毫米")
    p.add_argument("--height-mm", type=float, default=None, help="卡尺量的正面高，毫米")
    p.add_argument("--distance-mm", type=float, default=None, help="可选。卷尺量的外壳到正面距离，只作对照")
    p.add_argument("--scale", type=float, default=None, help="覆盖 config 的 ffs.scale")
    p.add_argument("--iters", type=int, default=None, help="覆盖 config 的 ffs.valid_iters")
    p.add_argument("--inset", type=float, default=0.12, help="拟合平面时四边形向内收缩的比例")
    p.add_argument("--repeat", type=int, default=8, help="按 R 时连测的帧数")
    p.add_argument("--left", default=None, help="离线左图，需同时给 --right")
    p.add_argument("--right", default=None)
    p.add_argument("--selftest", action="store_true", help="用合成物块检查几何，不开相机")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.selftest:
        return run_selftest()
    if (args.left is None) ^ (args.right is None):
        print("[错误] --left 和 --right 要同时给")
        return 1
    return run_measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
