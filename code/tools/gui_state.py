"""gui_state.py —— 界面记忆（窗口尺寸/位置、上次目录、质检阈值、标定板参数）

为什么要单独一个文件：
    这些都是"下次打开还想保持原样"的值，与标定算法无关。
    单独放一层，就能对它做纯函数测试：文件缺失、JSON 损坏、字段类型不对、
    数值越界，任何一种都不该让 GUI 启动不了。

设计原则：
    · `sanitize()` 是纯函数：输入任意 dict，输出**一定合法**的 dict（缺的补默认、越界的夹紧、
      类型不对的丢弃）。读取路径只调它一次，界面拿到的永远是安全值。
    · `load()` 永不抛异常：读失败就当没有记忆，返回默认值。
    · `save()` 用"写临时文件再替换"，中途崩溃不会留下半个 JSON 文件。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 记忆文件位置：放在 output/ 下（该目录已在 .gitignore 里，不会污染仓库）
DEFAULT_PATH = os.path.join(PROJECT_ROOT, "output", "gui_state.json")

SCHEMA = 1

#: 检查页七种显示方式（与 calib_gui 里的下拉框保持一致）
CHECK_MODES = ("上下叠放 + 水平线", "并排 + 参考线", "红青立体",
               "左右差值", "棋盘混合", "覆盖热力图", "分区 |dy| 热力图")

#: 每项的含义、默认值、允许区间（None 表示不做数值夹紧）
_NUMERIC = {
    "px":        (8.0, 1.0, 200.0),        # 每格最少像素
    "sharp":     (100.0, 0.0, 5000.0),     # 清晰度下限
    "contrast":  (120.0, 0.0, 1000.0),     # 对比度下限
    "target":    (26, 4, 300),             # 目标对数
}

_GEOM_RE = re.compile(r"^\d{3,5}x\d{3,5}([+-]\d{1,5}[+-]\d{1,5})?$")


def state_path() -> str:
    """记忆文件路径；可用环境变量 STEREO_GUI_STATE 改到别处（便于多套配置）。"""
    env = os.environ.get("STEREO_GUI_STATE", "").strip()
    return env or DEFAULT_PATH


def defaults() -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "geometry": "",                    # 留空 = 用程序里的默认尺寸
        "tab": 0,
        "dir": "",                         # 上次用的采集目录
        "thresholds": {k: v[0] for k, v in _NUMERIC.items()},
        "board": {"cols": 12, "rows": 9, "square": 3.0},
        "check_mode": CHECK_MODES[0],
        "alpha": 0.0,
        "fix_intrinsic": True,
    }


def _clamp(val: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(val)
    except (TypeError, ValueError):
        return default
    if x != x:                             # NaN
        return default
    return max(lo, min(hi, x))


def sanitize(raw: Any) -> Dict[str, Any]:
    """把任意读到的内容收敛成合法配置。纯函数，无副作用，永不抛异常。"""
    out = defaults()
    if not isinstance(raw, dict):
        return out

    geom = raw.get("geometry")
    if isinstance(geom, str) and _GEOM_RE.match(geom.strip()):
        out["geometry"] = geom.strip()

    tab = raw.get("tab")
    if isinstance(tab, int) and 0 <= tab <= 3:
        out["tab"] = tab

    d = raw.get("dir")
    if isinstance(d, str) and len(d) < 512:
        out["dir"] = d

    thr = raw.get("thresholds")
    if isinstance(thr, dict):
        for k, (default, lo, hi) in _NUMERIC.items():
            if k in thr:
                out["thresholds"][k] = _clamp(thr[k], lo, hi, default)
        # target 是整数，夹紧后再取整
    out["thresholds"]["target"] = int(round(out["thresholds"]["target"]))

    b = raw.get("board")
    if isinstance(b, dict):
        out["board"]["cols"] = int(round(_clamp(b.get("cols", 12), 3, 30, 12)))
        out["board"]["rows"] = int(round(_clamp(b.get("rows", 9), 3, 30, 9)))
        out["board"]["square"] = round(_clamp(b.get("square", 3.0), 0.5, 200.0, 3.0), 3)

    mode = raw.get("check_mode")
    if isinstance(mode, str) and mode in CHECK_MODES:
        out["check_mode"] = mode

    out["alpha"] = round(_clamp(raw.get("alpha", 0.0), 0.0, 1.0, 0.0), 3)

    if isinstance(raw.get("fix_intrinsic"), bool):
        out["fix_intrinsic"] = raw["fix_intrinsic"]
    return out


def load(path: str = "") -> Dict[str, Any]:
    """读记忆；文件不存在/损坏/不是 JSON 一律返回默认值，不抛异常。"""
    p = path or state_path()
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return defaults()
    return sanitize(raw)


def save(state: Dict[str, Any], path: str = "") -> bool:
    """写记忆（原子替换）。返回是否写成功；失败只是不记忆，不影响使用。"""
    p = path or state_path()
    data = sanitize(state)
    try:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
        return True
    except OSError:
        try:
            os.remove(p + ".tmp")
        except OSError:
            pass
        return False


def main(argv: Optional[list] = None) -> int:
    """命令行自检：读当前记忆并打印，便于排查"为什么尺寸没记住"。"""
    p = state_path()
    print(f"记忆文件：{p}")
    print(f"存在     ：{os.path.exists(p)}")
    for k, v in load().items():
        print(f"  {k:<14} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
