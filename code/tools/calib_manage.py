"""calib_manage.py —— 标定文件的归档 / 激活 / 对比 / 校验
=============================================================================
【为什么需要它】
    现状是 calib.save() 直接覆盖 calibration/stereo_calib.yaml：
        · 没有备份，重标一次就把上一份"好的"冲掉了
        · 上次能用的 v2 文件是靠人工另存才留下的
        · 想比较两份标定差多少，只能手工翻 YAML
    本文件只做**文件级**事务，不碰标定算法：
        archive   把当前生效的标定复制进 calibration/archive/ 并记一条备注
        list      列出所有归档（日期、基线、RMS、备注）
        activate  把某个归档复制回生效路径（会先自动归档当前那份，可回滚）
        diff      两份标定逐项对比（基线、fx、RMS、光轴夹角）
        validate  单文件结构与单位自检（抓 K1 缺失、矩阵形状错、mm/m 混用）

【安全约定】
    1) activate 前一定先 archive 当前文件 —— 保证任何一次激活都能退回。
    2) 只读、只复制，不删除任何标定文件（除用户显式 --force 覆盖同名归档）。
    3) 归档文件名带时间戳，不会互相覆盖。
=============================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

# ---- 路径解析（源码运行 = 项目根；打包成 EXE 后 = exe 所在目录）----
# 标定文件必须是**可写**的，所以固定落在 data_root 下，不去 _MEIPASS 找。
try:
    from app_paths import data_root as _data_root
except ImportError:                    # 单独拷走 tools/ 使用时的兜底
    def _data_root() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = _data_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CALIB_DIR = os.path.join(PROJECT_ROOT, "calibration")
CURRENT = os.path.join(CALIB_DIR, "stereo_calib.yaml")
ARCHIVE_DIR = os.path.join(CALIB_DIR, "archive")
MANIFEST = os.path.join(ARCHIVE_DIR, "manifest.json")

REQUIRED_KEYS = ("K1", "D1", "K2", "D2", "R", "T", "P1")


# -----------------------------------------------------------------------------
# 读与校验
# -----------------------------------------------------------------------------
def read_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    if not isinstance(d, dict):
        raise ValueError(f"不是 YAML 字典：{path}")
    return d


def validate_file(path: str) -> List[Tuple[str, str, bool]]:
    """结构自检，返回 [(项目, 说明, 是否通过)]。"""
    items: List[Tuple[str, str, bool]] = []
    if not os.path.exists(path):
        return [("文件", f"不存在：{path}", False)]
    try:
        d = read_file(path)
    except Exception as exc:                # noqa: BLE001
        return [("解析", f"{type(exc).__name__}: {exc}", False)]

    missing = [k for k in REQUIRED_KEYS if k not in d]
    items.append(("必需键", "缺失：" + (", ".join(missing) if missing else "无"), not missing))
    if missing:
        return items

    def arr(k: str, shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
        try:
            a = np.asarray(d[k], np.float64)
        except Exception:                   # noqa: BLE001
            return None
        if shape is not None and a.shape != shape:
            return None
        return a

    k1 = arr("K1", (3, 3))
    k2 = arr("K2", (3, 3))
    p1 = arr("P1", (3, 4))
    r = arr("R", (3, 3))
    t = arr("T")
    items.append(("K1/K2 形状", "3x3" if k1 is not None and k2 is not None else "异常", 
                  k1 is not None and k2 is not None))
    items.append(("P1 形状", "3x4" if p1 is not None else "异常", p1 is not None))
    items.append(("R 形状", "3x3" if r is not None else "异常", r is not None))
    t_ok = t is not None and t.size == 3
    items.append(("T 形状", str(None if t is None else t.shape), t_ok))

    if t is not None and t.size == 3:
        norm_mm = float(np.linalg.norm(t.reshape(3)))
        # T 的约定是毫米；若数量级 < 1，几乎一定是把米当毫米写进来了
        unit_ok = 5.0 < norm_mm < 1000.0
        items.append(("T 单位", f"|T|={norm_mm:.4f}（应为毫米量级 20~300）", unit_ok))

    if k1 is not None:
        fx = float(k1[0, 0])
        items.append(("K1 焦距", f"fx={fx:.2f}（应 >100 且 <5000）", 100.0 < fx < 5000.0))
    if p1 is not None:
        fx_r = float(p1[0, 0])
        items.append(("P1 焦距", f"fx={fx_r:.2f}（校正后等效焦距，应与 K1 同量级）",
                      100.0 < fx_r < 5000.0))

    size = d.get("image_size")
    size_ok = (isinstance(size, (list, tuple)) and len(size) == 2
               and all(isinstance(x, (int, float)) and x > 0 for x in size))
    items.append(("image_size", str(size), size_ok))

    if r is not None:
        tr = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(tr)))
        items.append(("光轴夹角", f"{ang:.3f} 度（<8 度符合刚性模组）", ang < 8.0))
    return items


def validate_text(items: Sequence[Tuple[str, str, bool]]) -> str:
    lines = []
    for name, info, ok in items:
        lines.append(f"{'[OK]  ' if ok else '[FAIL]'} {name:<12} {info}")
    bad = sum(1 for _, _, ok in items if not ok)
    lines.append(f"结论：{'通过' if bad == 0 else f'{bad} 项不通过'}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 摘要
# -----------------------------------------------------------------------------
def summarize(path: str) -> Dict[str, Any]:
    """提取对比用的关键数字。字段缺失时用 NaN，不抛异常。"""
    d = read_file(path)
    k1 = np.asarray(d.get("K1", np.full((3, 3), np.nan)), np.float64).reshape(3, 3)
    p1 = np.asarray(d.get("P1", np.full((3, 4), np.nan)), np.float64).reshape(3, 4)
    t = np.asarray(d.get("T", np.full(3, np.nan)), np.float64).reshape(-1)
    r = np.asarray(d.get("R", np.eye(3)), np.float64).reshape(3, 3)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))
    size = d.get("image_size") or [0, 0]
    return {
        "path": path,
        "date": d.get("date", "-"),
        "size": f"{size[0]}x{size[1]}",
        "fx": float(k1[0, 0]), "fy": float(k1[1, 1]),
        "cx": float(k1[0, 2]), "cy": float(k1[1, 2]),
        "rect_fx": float(p1[0, 0]),
        "baseline_mm": float(np.linalg.norm(t)),
        "axis_deg": ang,
        "rms_stereo": float(d.get("rms_stereo_px", float("nan"))),
        "rms_left": float(d.get("rms_left_px", float("nan"))),
        "rms_right": float(d.get("rms_right_px", float("nan"))),
        "board": (d.get("meta") or {}).get("board", "-"),
        "capture_dir": (d.get("meta") or {}).get("capture_dir", "-"),
    }


def format_summary(s: Dict[str, Any]) -> str:
    return (f"日期 {s['date']}  尺寸 {s['size']}\n"
            f"  基线 {s['baseline_mm']:.3f} mm   fx {s['fx']:.2f}  rect_fx {s['rect_fx']:.2f}  "
            f"cx/cy {s['cx']:.2f}/{s['cy']:.2f}\n"
            f"  RMS 双目 {s['rms_stereo']:.4f} / 左 {s['rms_left']:.4f} / 右 {s['rms_right']:.4f}  "
            f"光轴夹角 {s['axis_deg']:.3f} 度")


# -----------------------------------------------------------------------------
# 归档 / 激活
# -----------------------------------------------------------------------------
def _load_manifest() -> List[Dict[str, Any]]:
    if not os.path.exists(MANIFEST):
        return []
    try:
        with open(MANIFEST, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:                       # noqa: BLE001
        return []


def _save_manifest(entries: List[Dict[str, Any]]) -> None:
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=2)


def archive(src: str = "", note: str = "") -> str:
    """把 src（默认当前生效文件）复制进归档目录，返回归档路径。"""
    src = src or CURRENT
    if not os.path.exists(src):
        raise FileNotFoundError(f"要归档的文件不存在：{src}")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(src))[0]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(ARCHIVE_DIR, f"{stamp}_{base}.yaml")
    n = 1
    while os.path.exists(dst):              # 同一秒内多次归档时加后缀
        dst = os.path.join(ARCHIVE_DIR, f"{stamp}_{base}_{n}.yaml")
        n += 1
    shutil.copy2(src, dst)
    try:
        s = summarize(dst)
        entry = {"file": os.path.basename(dst), "note": note, "date": s["date"],
                 "baseline_mm": s["baseline_mm"], "rms_stereo": s["rms_stereo"],
                 "archived_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "source": os.path.abspath(src)}
    except Exception:                       # noqa: BLE001
        entry = {"file": os.path.basename(dst), "note": note,
                 "archived_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "source": os.path.abspath(src)}
    entries = _load_manifest()
    entries.append(entry)
    _save_manifest(entries)
    return dst


def list_archives() -> List[Dict[str, Any]]:
    """列出归档（优先用 manifest；manifest 缺失时扫描目录）。"""
    entries = _load_manifest()
    if entries:
        for e in entries:
            e["path"] = os.path.join(ARCHIVE_DIR, e.get("file", ""))
        return entries
    out = []
    for p in sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.yaml"))):
        try:
            s = summarize(p)
            out.append({"file": os.path.basename(p), "path": p, "note": "",
                        "date": s["date"], "baseline_mm": s["baseline_mm"],
                        "rms_stereo": s["rms_stereo"],
                        "archived_at": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))})
        except Exception:                   # noqa: BLE001
            out.append({"file": os.path.basename(p), "path": p, "note": "(无法解析)"})
    return out


def activate(archive_path: str, note: str = "") -> Dict[str, str]:
    """把归档文件激活为当前生效标定。

    步骤：先归档当前生效文件（可回滚）→ 复制归档内容到 CURRENT → 写激活记录。
    """
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"归档文件不存在：{archive_path}")
    backup = ""
    if os.path.exists(CURRENT):
        backup = archive(CURRENT, note=f"activate 前自动备份（{note or '无备注'}）")
    os.makedirs(CALIB_DIR, exist_ok=True)
    shutil.copy2(archive_path, CURRENT)
    return {"activated": archive_path, "current": CURRENT, "backup": backup}


# -----------------------------------------------------------------------------
# 对比
# -----------------------------------------------------------------------------
def diff(a: str, b: str) -> List[Tuple[str, str, str, str]]:
    """逐项对比两份标定，返回 [(项目, A 值, B 值, 变化)]。"""
    sa, sb = summarize(a), summarize(b)
    rows: List[Tuple[str, str, str, str]] = []

    def add(name: str, x: float, y: float, unit: str = "", digits: int = 3) -> None:
        delta = ""
        if np.isfinite(x) and np.isfinite(y) and abs(x) > 1e-12:
            pct = (y - x) / x * 100.0
            delta = f"{(y - x):+.{digits}f}{unit} ({pct:+.2f}%)"
        elif np.isfinite(y):
            delta = f"{y:.{digits}f}{unit}"
        rows.append((name, f"{x:.{digits}f}{unit}", f"{y:.{digits}f}{unit}", delta))

    add("基线", sa["baseline_mm"], sb["baseline_mm"], " mm")
    add("左目 fx", sa["fx"], sb["fx"], " px", 2)
    add("左目 cx", sa["cx"], sb["cx"], " px", 2)
    add("左目 cy", sa["cy"], sb["cy"], " px", 2)
    add("校正后 fx", sa["rect_fx"], sb["rect_fx"], " px", 2)
    add("双目 RMS", sa["rms_stereo"], sb["rms_stereo"], " px", 4)
    add("光轴夹角", sa["axis_deg"], sb["axis_deg"], " 度", 3)
    rows.append(("图像尺寸", sa["size"], sb["size"], "不同则不可互换" if sa["size"] != sb["size"] else "一致"))
    rows.append(("标定板", str(sa["board"]), str(sb["board"]), ""))
    rows.append(("采集目录", str(sa["capture_dir"]), str(sb["capture_dir"]), ""))
    rows.append(("日期", str(sa["date"]), str(sb["date"]), ""))
    return rows


def diff_text(a: str, b: str) -> str:
    rows = diff(a, b)
    w1 = max(len(r[0]) for r in rows) + 2
    w2 = max(len(r[1]) for r in rows) + 2
    w3 = max(len(r[2]) for r in rows) + 2
    head = f"{'项目'.ljust(w1)}{'A'.ljust(w2)}{'B'.ljust(w3)}变化"
    lines = [head, "-" * (w1 + w2 + w3 + 24)]
    for name, va, vb, d in rows:
        lines.append(f"{name.ljust(w1)}{va.ljust(w2)}{vb.ljust(w3)}{d}")
    try:
        sa, sb = summarize(a), summarize(b)
        if np.isfinite(sa["baseline_mm"]) and np.isfinite(sb["baseline_mm"]):
            pct = abs(sb["baseline_mm"] - sa["baseline_mm"]) / sa["baseline_mm"] * 100.0
            if pct > 2.0:
                lines.append(f"\n注意：基线变化 {pct:.2f}%，超过 2%。若两次用的是同一块板，"
                             "优先怀疑打印/装配变化或标定板实际尺寸与填写值不符。")
    except Exception:                       # noqa: BLE001
        pass
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 命令行
# -----------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="标定文件管理（归档/激活/对比/校验）")
    sub = ap.add_subparsers(dest="cmd")

    p_arch = sub.add_parser("archive", help="归档当前生效标定")
    p_arch.add_argument("--src", default="")
    p_arch.add_argument("--note", default="")

    sub.add_parser("list", help="列出所有归档")

    p_act = sub.add_parser("activate", help="把某个归档激活为生效标定")
    p_act.add_argument("name", help="归档文件名或序号（先 list 查看）")
    p_act.add_argument("--note", default="")

    p_diff = sub.add_parser("diff", help="对比两份标定")
    p_diff.add_argument("a", nargs="?", default="")
    p_diff.add_argument("b", nargs="?", default="")

    p_val = sub.add_parser("validate", help="结构自检")
    p_val.add_argument("path", nargs="?", default="")

    args = ap.parse_args(argv)

    if args.cmd == "archive":
        dst = archive(args.src, args.note)
        print(f"已归档：{dst}")
        return 0

    if args.cmd == "list":
        rows = list_archives()
        if not rows:
            print(f"还没有归档：{ARCHIVE_DIR}")
            return 0
        for i, r in enumerate(rows, 1):
            rms = r.get("rms_stereo")
            rms_s = f"{rms:.4f}" if isinstance(rms, (int, float)) else "-"
            base = r.get("baseline_mm")
            base_s = f"{base:.3f}" if isinstance(base, (int, float)) else "-"
            print(f"[{i}] {r.get('file', '-')}  基线 {base_s} mm  RMS {rms_s}  "
                  f"{r.get('archived_at', '')}  {r.get('note', '')}")
        return 0

    if args.cmd == "activate":
        rows = list_archives()
        target = args.name
        if target.isdigit():
            idx = int(target) - 1
            if not (0 <= idx < len(rows)):
                print(f"[错误] 序号超出范围（共 {len(rows)} 条）")
                return 1
            target = rows[idx]["path"]
        elif not os.path.isabs(target):
            target = os.path.join(ARCHIVE_DIR, target)
        res = activate(target, args.note)
        print(f"已激活：{res['activated']}")
        print(f"当前文件：{res['current']}")
        if res["backup"]:
            print(f"原文件已备份：{res['backup']}")
        return 0

    if args.cmd == "diff":
        a = args.a or CURRENT
        rows = list_archives()
        b = args.b
        if not b:
            if not rows:
                print("[错误] 没有归档可比，请用 diff <a> <b>")
                return 1
            b = rows[-1]["path"]
        print(f"A = {a}\nB = {b}\n")
        print(diff_text(a, b))
        return 0

    if args.cmd == "validate":
        path = args.path or CURRENT
        print(validate_text(validate_file(path)))
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
