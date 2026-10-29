"""viewer_shell.py —— RealSense Viewer 式主界面
=============================================================================
双击 EXE（以及不带参数运行 app_entry.py）时打开的就是这个窗口。布局对照
Intel RealSense Viewer：

    顶栏        设备名 + 开始/停止串流
    左侧        连接状态、画面开关、深度范围 / 点大小 / 推理档位
    中间        2D（彩色 / 深度 / 视差）与 3D 点云，同一块视图里切换
    右侧        当前标定、历史归档、采集目录和保存下来的深度/点云
    底栏        帧率、分辨率、光标处深度

采集和标定仍走原来的标定工作台（它要独占相机）。打开之前会先停掉串流。
工作台关掉后，右侧结果会自动刷新；点一项即可在中间回看彩色、深度或点云。
=============================================================================
"""

from __future__ import annotations

import glob
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from app_paths import bundle_root as _bundle_root, data as _data, find as _find
except ImportError:
    def _bundle_root() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _data(rel: str, create: bool = False) -> str:
        return os.path.join(_bundle_root(), rel)

    def _find(rel: str) -> str:
        return os.path.join(_bundle_root(), rel)

_ROOT = _bundle_root()
for _p in (_ROOT, os.path.join(_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from depth_utils import attach_range_bar, colorize_depth  # noqa: E402

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print(t("[错误] 当前 Python 没有 tkinter。请安装官方 CPython，并勾选 tcl/tk。"))
    raise SystemExit(1)

try:
    from PIL import Image, ImageTk
    _HAVE_PIL = True
except ImportError:
    Image = ImageTk = None  # type: ignore
    _HAVE_PIL = False

import gui_theme as theme
from i18n import (t, tf, retranslate, set_language, get_language,
                  language_label, save_language)

# 与标定工作台同一套科技黑，颜色只在 gui_theme.py 里改。
BG = theme.BG
SIDE = theme.PANEL
RAISED = theme.PANEL_HI
LINE = theme.BORDER
TEXT = theme.TEXT
MUTED = theme.MUTED
ACCENT = theme.ACCENT
ACCENT_DK = theme.ACCENT_DK
ON_ACCENT = theme.ON_ACCENT
DANGER = theme.BAD
OK = theme.OK
WARN = theme.WARN
BLACK = theme.IMG_BG
TOP = theme.TOP
SELECT = theme.SELECT
HOVER = theme.HOVER
SEAM = theme.SEAM

PRESET_ITEMS = ((t("精确"), "precise"), (t("平衡"), "balanced"), (t("快速"), "fast"))


def _enable_dpi() -> float:
    if sys.platform != "win32":
        return 1.0
    import ctypes
    scale = 1.0
    try:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:  # noqa: BLE001
            ctypes.windll.user32.SetProcessDPIAware()
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        if dpi:
            scale = float(dpi) / 96.0
    except Exception:  # noqa: BLE001
        scale = 1.0
    return scale


def _font_family(root: tk.Misc) -> str:
    try:
        import tkinter.font as tkfont
        have = set(tkfont.families(root))
    except Exception:  # noqa: BLE001
        return "TkDefaultFont"
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei"):
        if name in have:
            return name
    return "TkDefaultFont"


def _imread(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.isfile(path):
        return None
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _suggest_dist(points: np.ndarray) -> float:
    if points is None or len(points) == 0:
        return 1.2
    center = np.median(np.asarray(points, np.float32), axis=0)
    radius = float(np.percentile(np.linalg.norm(points - center, axis=1), 90))
    return float(max(radius * 2.6, 0.25))


def _rgb_uint8_to_bgr(colors: Optional[np.ndarray], n: int) -> np.ndarray:
    """Session / PLY 里的颜色是 RGB。OpenCV 画面要 BGR。"""
    if colors is None or len(colors) != n:
        return np.full((n, 3), 180, np.uint8)
    c = np.asarray(colors)
    if c.dtype != np.uint8:
        scale = 255.0 if (c.size and float(np.nanmax(c)) <= 1.0) else 1.0
        c = np.clip(c * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(c[:, ::-1])


def render_cloud(
    points: np.ndarray,
    colors_bgr: Optional[np.ndarray],
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    dist: float,
    pan_x: float,
    pan_y: float,
    point_px: int = 2,
    max_points: int = 90000,
) -> np.ndarray:
    """把相机系点云（y 向下，米）画到黑底上。左键轨道、右键平移、滚轮缩放由调用方改参数。"""
    h, w = int(height), int(width)
    img = np.zeros((h, w, 3), np.uint8)
    if points is None or len(points) == 0 or h < 8 or w < 8:
        return img
    pts = np.asarray(points, np.float32)
    cols = None if colors_bgr is None else np.asarray(colors_bgr)
    n = len(pts)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
        pts = pts[idx]
        if cols is not None and len(cols) == n:
            cols = cols[idx]
    xyz = pts.copy()
    xyz[:, 1] *= -1.0
    xyz -= np.median(xyz, axis=0)
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x2 = cy * x + sy * z
    z2 = -sy * x + cy * z
    y2 = cp * y - sp * z2
    z3 = sp * y + cp * z2 + float(dist)
    ok = z3 > 1e-3
    if not np.any(ok):
        _draw_gizmo(img, yaw, pitch)
        return img
    focal = 0.9 * float(max(w, h))
    u = np.round(focal * x2[ok] / z3[ok] + w * 0.5 + pan_x).astype(np.int32)
    v = np.round(h * 0.5 - focal * y2[ok] / z3[ok] + pan_y).astype(np.int32)
    order = np.argsort(z3[ok])
    u, v = u[order], v[order]
    if cols is None or len(cols) != len(pts):
        cc = np.full((ok.sum(), 3), 190, np.uint8)[order]
    else:
        cc = cols[ok][order]
    m = (u >= 1) & (v >= 1) & (u < w - 1) & (v < h - 1)
    u, v, cc = u[m], v[m], cc[m]
    if len(u) == 0:
        _draw_gizmo(img, yaw, pitch)
        return img
    img[v, u] = cc
    if int(point_px) >= 2:
        img[v, np.clip(u + 1, 0, w - 1)] = cc
        img[np.clip(v + 1, 0, h - 1), u] = cc
    _draw_gizmo(img, yaw, pitch)
    return img


def _draw_gizmo(img: np.ndarray, yaw: float, pitch: float) -> None:
    """左下角的 RGB 坐标轴，跟着轨道旋转。X 红、Y 绿、Z 蓝。"""
    h, w = img.shape[:2]
    origin = np.array([46.0, h - 46.0], np.float32)
    length = 34.0
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    axes = ((1.0, 0.0, 0.0, (40, 40, 230), "X"),
            (0.0, 1.0, 0.0, (40, 200, 40), "Y"),
            (0.0, 0.0, 1.0, (230, 140, 40), "Z"))
    for x, y, z, col, name in axes:
        x2 = cy * x + sy * z
        z2 = -sy * x + cy * z
        y2 = cp * y - sp * z2
        end = origin + np.array([x2, -y2], np.float32) * length
        p0 = (int(origin[0]), int(origin[1]))
        p1 = (int(end[0]), int(end[1]))
        cv2.line(img, p0, p1, col, 2, cv2.LINE_AA)
        cv2.putText(img, name, (p1[0] + 3, p1[1] + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, col, 1, cv2.LINE_AA)


def _load_ply(path: str, max_points: int = 90000) -> Tuple[np.ndarray, np.ndarray]:
    """读本程序写的 ASCII PLY，或 Open3D 能打开的二进制 PLY。颜色返回 RGB uint8。"""
    with open(path, "rb") as fh:
        head = fh.read(256)
    if b"format ascii" in head:
        return _load_ply_ascii(path, max_points)
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, np.float32)
    cols = np.asarray(pcd.colors, np.float32)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts = pts[idx]
        if len(cols) == len(np.asarray(pcd.points)):
            cols = cols[idx]
    if len(cols) == len(pts) and len(pts):
        rgb = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.full((len(pts), 3), 180, np.uint8)
    return pts, rgb


def _load_ply_ascii(path: str, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        step = max(1, n // max_points) if n else 1
        pts: List[Tuple[float, float, float]] = []
        cols: List[Tuple[int, int, int]] = []
        for i in range(n):
            line = fh.readline()
            if i % step:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
            if len(parts) >= 6:
                cols.append((int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))))
            else:
                cols.append((180, 180, 180))
    if not pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.asarray(pts, np.float32), np.asarray(cols, np.uint8)


class ViewerApp(tk.Tk):
    """主窗口。串流在后台线程里启动和停止，界面线程只取最新帧。"""

    def __init__(self, dpi_scale: float = 1.0) -> None:
        super().__init__()
        if dpi_scale > 1.01:
            try:
                self.tk.call("tk", "scaling", dpi_scale * 1.3333)
            except Exception:  # noqa: BLE001
                pass
        self.font_family = _font_family(self)
        self.title("Stereo Viewer")
        self.geometry("1280x800")
        self.minsize(1100, 680)
        theme.apply(self)
        self._style_scale()
        self.configure(bg=BG)

        self.session = None
        self.q: "queue.Queue" = queue.Queue()
        self.busy = False
        self.paused = False
        self.mode = "2d"
        self.review: Optional[Dict[str, Any]] = None
        self.live_frame = None
        self.live_res = None
        self._frame_seq = -1
        self._photo = None
        self._hit: List[Dict[str, Any]] = []
        self._cursor = ""
        self._last_err = ""
        self._fit_pending = True
        self._param_job = None
        self._slider_lock = False
        self._pending_calib = False
        self._calib_proc: Optional[subprocess.Popen] = None
        self._closing = False
        self._result_paths: List[str] = []
        self.cloud_yaw = 0.55
        self.cloud_pitch = 0.28
        self.cloud_dist = 1.4
        self.cloud_pan = [0.0, 0.0]
        self._drag: Optional[Tuple[str, int, int]] = None
        self._view_dirty = True

        zmin, zmax, preset_label = self._defaults()
        self.zmin = zmin
        self.zmax = zmax
        self.point_px = 2

        self.show_left = tk.BooleanVar(value=True)
        self.show_right = tk.BooleanVar(value=False)
        self.show_depth = tk.BooleanVar(value=True)
        self.show_disp = tk.BooleanVar(value=False)
        self.preset_var = tk.StringVar(value=preset_label)
        self.status_var = tk.StringVar(value=t("未连接"))
        self.device_var = tk.StringVar(value=t("双目深度相机"))
        self.meta_var = tk.StringVar(value=t("分辨率 —    帧率 —"))
        self.lang_var = tk.StringVar(value=language_label())

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<space>", lambda e: self._toggle_pause())
        self.bind("s", lambda e: self._save_frame())
        self.bind("S", lambda e: self._save_frame())
        self.bind("1", lambda e: self._preset_view("color"))
        self.bind("2", lambda e: self._preset_view("depth"))
        self.bind("3", lambda e: self._preset_view("cloud"))
        self.after(40, self._tick)
        self.after(80, self._refresh_results)
        self._log(t("就绪。开始串流后可在 2D / 3D 之间切换。标定请先停止串流。"))

    # ------------------------------------------------------------------ 外观
    def _style_scale(self) -> None:
        """滑条用 ttk，滑块走青色，槽是深青。tk.Scale 的滑块在深色底上会糊成一条黑。"""
        st = ttk.Style(self)
        st.configure("Side.Horizontal.TScale",
                     background=ACCENT, troughcolor="#102833",
                     bordercolor=SEAM, lightcolor=ACCENT, darkcolor=ACCENT_DK,
                     sliderlength=16)
        st.map("Side.Horizontal.TScale",
               background=[("pressed", "#4ad6ee"), ("active", "#4ad6ee")])

    def _f(self, size: int = 10, bold: bool = False) -> Tuple:
        return (self.font_family, size, "bold" if bold else "normal")

    def _section(self, parent: tk.Misc, title: str) -> tk.Frame:
        box = tk.Frame(parent, bg=SIDE)
        row = tk.Frame(box, bg=SIDE)
        row.pack(fill="x", padx=14, pady=(12, 2))
        theme.accent_tick(row)
        tk.Label(row, text=title, bg=SIDE, fg=TEXT, font=self._f(9, True),
                 anchor="w").pack(side="left")
        return box

    def _button(self, parent: tk.Misc, text: str, command, primary: bool = False) -> tk.Button:
        if primary:
            btn = tk.Button(parent, text=text, command=command, bg=ACCENT, fg=ON_ACCENT,
                            activebackground="#4ad6ee", activeforeground=ON_ACCENT,
                            font=self._f(9, True), relief="flat", bd=0, padx=12, pady=6,
                            cursor="hand2", highlightthickness=0)
        else:
            btn = tk.Button(parent, text=text, command=command, bg=RAISED, fg=TEXT,
                            activebackground=HOVER, activeforeground=TEXT,
                            font=self._f(9), relief="flat", bd=0, padx=12, pady=6,
                            cursor="hand2", highlightthickness=1,
                            highlightbackground=LINE, highlightcolor=ACCENT)
        return btn

    def _paint_toggle(self, btn: tk.Button, on: bool) -> None:
        """2D / 3D 切换：选中是青底深字，未选中回到次级按钮。"""
        btn.configure(bg=ACCENT if on else RAISED,
                      fg=ON_ACCENT if on else TEXT,
                      activeforeground=ON_ACCENT if on else TEXT)

    # ------------------------------------------------------------------ 语言
    def _on_language(self, _event=None) -> None:
        """下拉框切换语言：重写整窗文案，并把选择写回 config.yaml。"""
        want = "en" if str(self.lang_var.get()).lower().startswith("en") else "zh"
        if want == get_language():
            return
        set_language(want)
        self._save_language(want)
        self._retranslate_ui()
        self._log(t("界面语言已切换为 English") if want == "en"
                  else t("界面语言已切换为中文"))

    def _save_language(self, lang: str) -> None:
        path = getattr(self, "config_path", "") or _find("config.yaml")
        if not path:
            return
        if not save_language(path, lang):
            self._log(tf("语言偏好没有写回 {path}", path=path))

    def _retranslate_ui(self) -> None:
        """把控件树上已有的文案按当前语言重写一遍。

        不重建窗口 —— 相机连接、串流状态、已加载的模型都保留，只换字。
        动态文本（帧率、坐标）走 tf() 生成，会在下一帧刷新时跟上。
        """
        def walk(widget: tk.Misc) -> None:
            for child in widget.winfo_children():
                try:
                    cur = child.cget("text")
                except Exception:               # noqa: BLE001
                    cur = None
                if isinstance(cur, str) and cur:
                    new = retranslate(cur)
                    if new != cur:
                        try:
                            child.configure(text=new)
                        except Exception:       # noqa: BLE001
                            pass
                walk(child)

        walk(self)

        # 档位下拉：显示翻译后的名字，内部键不变
        try:
            self.preset_box.configure(values=[t(a) for a, _ in PRESET_ITEMS])
            key = self._preset_key()
            zh = next((a for a, k in PRESET_ITEMS if k == key), PRESET_ITEMS[0][0])
            self.preset_var.set(t(zh))
        except Exception:                       # noqa: BLE001
            pass

        # 这几个走 StringVar，不在控件树上
        self.status_var.set(retranslate(self.status_var.get()))
        self.device_var.set(retranslate(self.device_var.get()))
        self.meta_var.set(retranslate(self.meta_var.get()))
        self.lang_var.set(language_label())

        # 视图标题与空态提示由 _paint_view 统一生成
        self._view_dirty = True
        try:
            self._paint_view()
        except Exception:                       # noqa: BLE001
            pass
        try:
            self._refresh_results()
        except Exception:                       # noqa: BLE001
            pass

    def _build(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        top = tk.Frame(self, bg=TOP, height=56)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_propagate(False)
        # 青线要先占住整行底边，否则会被左右控件的包裹切成中间一小段
        tk.Frame(top, bg=ACCENT, height=2, bd=0, highlightthickness=0).pack(side="bottom", fill="x")
        theme.brand_mark(top, TOP).pack(side="left", padx=(16, 10))
        tk.Label(top, text="STEREO", bg=TOP, fg=ACCENT,
                 font=self._f(12, True)).pack(side="left")
        tk.Label(top, text="VIEWER", bg=TOP, fg=TEXT,
                 font=self._f(12, True)).pack(side="left", padx=(6, 0))
        tk.Label(top, textvariable=self.device_var, bg=TOP, fg=MUTED,
                 font=self._f(9)).pack(side="left", padx=(16, 0))
        self.btn_stream = self._button(top, t("开始串流"), self._toggle_stream, primary=True)
        self.btn_stream.pack(side="right", padx=14, pady=10)

        # 语言选择器：中文 / English。切换后整窗重取文案，不重建窗口
        self.lang_box = ttk.Combobox(top, textvariable=self.lang_var, state="readonly",
                                     values=("中文", "English"), width=8, font=self._f(9))
        self.lang_box.pack(side="right", pady=10)
        self.lang_box.bind("<<ComboboxSelected>>", self._on_language)
        self.lang_title = tk.Label(top, text=t("界面语言"), bg=TOP, fg=MUTED,
                                   font=self._f(9))
        self.lang_title.pack(side="right", padx=(10, 6))

        body = tk.Frame(self, bg=SEAM)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._build_side(body)
        self._build_view(body)
        self._build_results(body)

        bar = tk.Frame(self, bg=TOP, height=30)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)
        tk.Frame(bar, bg=SEAM, height=1, bd=0, highlightthickness=0).pack(side="top", fill="x")
        tk.Label(bar, textvariable=self.status_var, bg=TOP, fg=TEXT,
                 font=self._f(9)).pack(side="left", padx=14)
        tk.Label(bar, textvariable=self.meta_var, bg=TOP, fg=MUTED,
                 font=self._f(9)).pack(side="right", padx=14)

    def _build_side(self, parent: tk.Misc) -> None:
        wrap = tk.Frame(parent, bg=SIDE, width=300)
        wrap.grid(row=0, column=0, sticky="nsew")
        wrap.grid_propagate(False)
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        canvas = tk.Canvas(wrap, bg=SIDE, highlightthickness=0, bd=0)
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        inner = tk.Frame(canvas, bg=SIDE)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", self._side_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        self.side_canvas = canvas

        dev = self._section(inner, t("设备"))
        dev.pack(fill="x")
        row = tk.Frame(dev, bg=SIDE)
        row.pack(fill="x", padx=14, pady=(0, 8))
        self.dot = tk.Canvas(row, width=12, height=12, bg=SIDE, highlightthickness=0)
        self.dot.pack(side="left", pady=4)
        self._paint_dot(MUTED)
        self.dev_text = tk.Label(row, text=t("未连接"), bg=SIDE, fg=TEXT, font=self._f(10, True))
        self.dev_text.pack(side="left", padx=8)
        self.dev_sub = tk.Label(dev, text=t("开始串流后显示分辨率和帧率"), bg=SIDE, fg=MUTED,
                                font=self._f(9), justify="left", anchor="w", wraplength=168)
        self.dev_sub.pack(fill="x", padx=14, pady=(0, 4))

        streams = self._section(inner, t("画面"))
        streams.pack(fill="x")
        self._check(streams, t("彩色（左）"), self.show_left)
        self._check(streams, t("彩色（右）"), self.show_right)
        self._check(streams, t("深度"), self.show_depth)
        self._check(streams, t("视差"), self.show_disp)
        tk.Label(streams, text=t("点云在中间的 3D 视图里。快捷键 1 彩色  2 深度  3 点云"),
                 bg=SIDE, fg=MUTED, font=self._f(8), wraplength=260, justify="left"
                 ).pack(fill="x", padx=14, pady=(2, 6))

        ctrl = self._section(inner, t("参数"))
        ctrl.pack(fill="x")
        self.scale_zmin = self._scale(ctrl, t("深度下限 (m)"), 0.05, 2.0, self.zmin, self._on_range)
        self.scale_zmax = self._scale(ctrl, t("深度上限 (m)"), 0.20, 8.0, self.zmax, self._on_range)
        self.scale_pt = self._scale(ctrl, t("点大小"), 1, 5, 2, self._on_point, resolution=1)
        tk.Label(ctrl, text=t("推理档位"), bg=SIDE, fg=MUTED, font=self._f(9)).pack(anchor="w", padx=14, pady=(8, 2))
        box = ttk.Combobox(ctrl, textvariable=self.preset_var, state="readonly",
                           values=[t(a) for a, _ in PRESET_ITEMS], width=12, font=self._f(9))
        box.pack(anchor="w", padx=14, pady=(0, 4))
        box.bind("<<ComboboxSelected>>", self._on_preset)
        self.preset_box = box
        tk.Label(ctrl, text=t("切换档位会重新加载模型，大约需要几秒。\n滑条只改显示范围，近处为红色，远处为蓝色。"),
                 bg=SIDE, fg=MUTED, font=self._f(8), wraplength=260, justify="left"
                 ).pack(fill="x", padx=14, pady=(0, 8))

        act = self._section(inner, t("采集与标定"))
        act.pack(fill="x", pady=(0, 12))
        self._button(act, t("保存当前帧"), self._save_frame).pack(fill="x", padx=14, pady=3)
        self._button(act, t("标定工作台"), self._open_calib, primary=True).pack(fill="x", padx=14, pady=3)
        self._button(act, t("环境自检"), self._run_selftest).pack(fill="x", padx=14, pady=3)
        self._button(act, t("模型自检"), self._run_model_test).pack(fill="x", padx=14, pady=(3, 8))

    def _check(self, parent: tk.Misc, text: str, var: tk.BooleanVar) -> None:
        ttk.Checkbutton(parent, text=text, variable=var, command=self._redraw,
                        style="Panel.TCheckbutton").pack(fill="x", padx=12, pady=1)

    def _scale(self, parent, title, lo, hi, value, cmd, resolution=0.01) -> ttk.Scale:
        holder = tk.Frame(parent, bg=SIDE)
        holder.pack(fill="x", padx=14, pady=(6, 0))
        head = tk.Frame(holder, bg=SIDE)
        head.pack(fill="x")
        tk.Label(head, text=title, bg=SIDE, fg=MUTED, font=self._f(9)).pack(side="left")
        shown = tk.StringVar()

        def text_of(raw) -> str:
            num = float(raw)
            if resolution >= 1:
                return str(int(round(num)))
            return f"{num:.2f}"

        shown.set(text_of(value))
        tk.Label(head, textvariable=shown, bg=SIDE, fg=ACCENT,
                 font=self._f(9)).pack(side="right")

        def wrapped(raw) -> None:
            shown.set(text_of(raw))
            cmd(raw)

        sc = ttk.Scale(holder, from_=lo, to=hi, orient="horizontal",
                       style="Side.Horizontal.TScale")
        sc.set(value)
        sc.configure(command=wrapped)
        sc.pack(fill="x", pady=(2, 0))
        return sc

    def _build_view(self, parent: tk.Misc) -> None:
        center = tk.Frame(parent, bg=BLACK)
        center.grid(row=0, column=1, sticky="nsew", padx=1, pady=0)
        center.grid_rowconfigure(1, weight=1)
        center.grid_columnconfigure(0, weight=1)

        bar = tk.Frame(center, bg=TOP, height=42)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)
        self.btn_2d = self._button(bar, "2D", lambda: self._set_mode("2d"), primary=True)
        self.btn_2d.pack(side="left", padx=(10, 4), pady=6)
        self.btn_3d = self._button(bar, "3D", lambda: self._set_mode("3d"))
        self.btn_3d.pack(side="left", padx=4, pady=6)
        self.view_title = tk.Label(bar, text=t("彩色") + "  |  " + t("深度"), bg=TOP, fg=TEXT, font=self._f(10))
        self.view_title.pack(side="left", padx=12)
        self.btn_back = self._button(bar, t("回到实时"), self._clear_review)
        self.btn_pause = self._button(bar, t("暂停"), self._toggle_pause)
        self.btn_pause.pack(side="right", padx=(4, 10), pady=6)
        self.btn_reset = self._button(bar, t("重置视角"), self._reset_cloud)
        self.btn_reset.pack(side="right", padx=4, pady=6)

        self.view_host = tk.Frame(center, bg=BLACK, width=640, height=420)
        self.view_host.grid(row=1, column=0, sticky="nsew")
        self.view_host.grid_propagate(False)
        self.view_host.grid_rowconfigure(0, weight=1)
        self.view_host.grid_columnconfigure(0, weight=1)
        self.view = tk.Label(self.view_host,
                             text=t("尚未串流\n\n右上角开始串流，查看彩色、深度和点云\n保存的采集与快照在右侧点选即可回看"),
                             bg=BLACK, fg=MUTED, font=self._f(11), justify="center")
        self.view.grid(row=0, column=0, sticky="nsew")
        self.view.bind("<ButtonPress-1>", self._press)
        self.view.bind("<B1-Motion>", self._drag_move)
        self.view.bind("<ButtonRelease-1>", self._release)
        self.view.bind("<ButtonPress-3>", self._press)
        self.view.bind("<B3-Motion>", self._drag_move)
        self.view.bind("<ButtonRelease-3>", self._release)
        self.view.bind("<Enter>", lambda e: self.view.bind_all("<MouseWheel>", self._view_wheel))
        self.view.bind("<Leave>", lambda e: self.view.unbind_all("<MouseWheel>"))
        self.view.bind("<Motion>", self._hover)

        hint = tk.Label(center, text=t("  3D  左键旋转    右键平移    滚轮缩放"
                                       "      深度图上移动鼠标可读距离      空格暂停      S 保存当前帧"),
                        bg=TOP, fg=MUTED, font=self._f(8), anchor="w")
        hint.grid(row=2, column=0, sticky="ew")

    def _build_results(self, parent: tk.Misc) -> None:
        right = tk.Frame(parent, bg=SIDE, width=320)
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_propagate(False)
        right.grid_rowconfigure(3, weight=1)
        right.grid_rowconfigure(6, weight=1)
        right.grid_columnconfigure(0, weight=1)

        head_cal = tk.Frame(right, bg=SIDE)
        head_cal.grid(row=0, column=0, sticky="ew", padx=12, pady=(14, 4))
        theme.accent_tick(head_cal)
        tk.Label(head_cal, text=t("标定结果"), bg=SIDE, fg=TEXT, font=self._f(9, True),
                 anchor="w").pack(side="left")
        self.calib_box = tk.Text(right, height=11, bg=BG, fg=TEXT, font=self._f(9),
                                 relief="flat", bd=0, wrap="word", padx=8, pady=6,
                                 insertbackground=TEXT, selectbackground=SELECT,
                                 selectforeground=TEXT,
                                 highlightthickness=1, highlightbackground=LINE,
                                 highlightcolor=ACCENT)
        self.calib_box.grid(row=1, column=0, sticky="ew", padx=12)
        self.calib_box.configure(state="disabled")

        head_snap = tk.Frame(right, bg=SIDE)
        head_snap.grid(row=2, column=0, sticky="nw", padx=12, pady=(12, 4))
        theme.accent_tick(head_snap)
        tk.Label(head_snap, text=t("采集与快照"), bg=SIDE, fg=TEXT, font=self._f(9, True),
                 anchor="w").pack(side="left")
        list_wrap = tk.Frame(right, bg=SIDE)
        list_wrap.grid(row=3, column=0, sticky="nsew", padx=12)
        self.result_list = tk.Listbox(list_wrap, bg=BG, fg=TEXT, font=self._f(9),
                                      relief="flat", bd=0, highlightthickness=1,
                                      highlightbackground=LINE, highlightcolor=ACCENT,
                                      selectbackground=SELECT, selectforeground=TEXT,
                                      activestyle="none")
        self.result_list.pack(side="left", fill="both", expand=True)
        self.result_list.bind("<<ListboxSelect>>", self._on_pick_result)
        sb = ttk.Scrollbar(list_wrap, orient="vertical", command=self.result_list.yview)
        sb.pack(side="right", fill="y")
        self.result_list.configure(yscrollcommand=sb.set)

        actions = tk.Frame(right, bg=SIDE)
        actions.grid(row=4, column=0, sticky="ew", padx=12, pady=6)
        self._button(actions, t("刷新"), self._refresh_results).pack(side="left", padx=(0, 6))
        self._button(actions, t("打开目录"), self._open_selected_dir).pack(side="left")

        head_log = tk.Frame(right, bg=SIDE)
        head_log.grid(row=5, column=0, sticky="ew", padx=12)
        theme.accent_tick(head_log)
        tk.Label(head_log, text=t("通知"), bg=SIDE, fg=TEXT, font=self._f(9, True),
                 anchor="w").pack(side="left")
        self.log = tk.Text(right, height=8, bg=BG, fg=MUTED, font=self._f(8),
                           relief="flat", bd=0, wrap="word", padx=8, pady=6,
                           insertbackground=TEXT, selectbackground=SELECT,
                           selectforeground=TEXT,
                           highlightthickness=1, highlightbackground=LINE,
                           highlightcolor=ACCENT)
        self.log.grid(row=6, column=0, sticky="nsew", padx=12, pady=(4, 12))
        self.log.configure(state="disabled")

    # ------------------------------------------------------------------ 小工具
    def _paint_dot(self, color: str) -> None:
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=color, outline=color)

    def _side_wheel(self, event) -> None:
        self.side_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _log(self, text: str) -> None:
        line = time.strftime("%H:%M:%S ") + str(text).rstrip() + "\n"
        self.log.configure(state="normal")
        self.log.insert("end", line)
        # 通知区只留最近一段，避免长时间串流把文本控件撑大
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 200:
            self.log.delete("1.0", f"{lines - 200}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _defaults(self) -> Tuple[float, float, str]:
        zmin, zmax, label = 0.15, 3.0, t("快速")
        self.config_path = _find("config.yaml")
        try:
            import yaml
            with open(self.config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            pc = cfg.get("pointcloud") or {}
            zmin = float(pc.get("zmin_m", zmin) or zmin)
            zmax = float(pc.get("zmax_m", zmax) or zmax)
            ffs = cfg.get("ffs") or {}
            preset = str(ffs.get("preset") or "")
            named = {b: a for a, b in PRESET_ITEMS}
            if preset in named:
                label = named[preset]
            else:
                scale = float(ffs.get("scale", 0.5) or 0.5)
                iters = int(ffs.get("valid_iters", 4) or 4)
                if scale >= 0.99 and iters >= 8:
                    label = t("精确")
                elif abs(scale - 0.75) < 0.05:
                    label = t("平衡")
        except Exception:  # noqa: BLE001
            pass
        return zmin, zmax, label

    def _preset_key(self) -> str:
        label = self.preset_var.get()
        for text, key in PRESET_ITEMS:
            if text == label or t(text) == label:
                return key
        return "fast"

    # ------------------------------------------------------------------ 视图
    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._paint_toggle(self.btn_2d, mode == "2d")
        self._paint_toggle(self.btn_3d, mode == "3d")
        if mode == "3d":
            self._fit_pending = True
        self._view_dirty = True
        self._redraw()

    def _preset_view(self, which: str) -> None:
        if which == "cloud":
            self._set_mode("3d")
            return
        self.show_left.set(which == "color")
        self.show_right.set(False)
        self.show_depth.set(which == "depth")
        self.show_disp.set(False)
        self._set_mode("2d")

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.btn_pause.configure(text=t("继续") if self.paused else t("暂停"))
        if self.session is not None and getattr(self.session, "started", False):
            try:
                if self.paused:
                    self.session.pause()
                else:
                    self.session.resume()
            except Exception:  # noqa: BLE001
                pass
        self.status_var.set(t("已暂停显示") if self.paused else self.status_var.get())

    def _reset_cloud(self) -> None:
        self.cloud_yaw, self.cloud_pitch = 0.55, 0.28
        self.cloud_pan = [0.0, 0.0]
        self._fit_pending = True
        self._redraw()

    def _clear_review(self) -> None:
        self.review = None
        self.btn_back.pack_forget()
        self._fit_pending = True
        self._log(t("已回到实时画面"))
        self._redraw()

    def _redraw(self) -> None:
        self._view_dirty = True
        self._paint_view()

    def _view_size(self) -> Tuple[int, int]:
        self.view_host.update_idletasks()
        return max(self.view_host.winfo_width(), 2), max(self.view_host.winfo_height(), 2)

    def _sources(self) -> Tuple[Any, Any]:
        if self.review is not None:
            return self.review.get("frame"), self.review.get("res")
        return self.live_frame, self.live_res

    def _paint_view(self) -> None:
        if self._closing:
            return
        w, h = self._view_size()
        if w < 40 or h < 40:
            return
        image = self._compose(w, h)
        names = []
        if self.mode == "3d":
            names = [t("点云")]
        else:
            if self.show_left.get():
                names.append(t("彩色"))
            if self.show_right.get():
                names.append(t("右彩色"))
            if self.show_depth.get():
                names.append(t("深度"))
            if self.show_disp.get():
                names.append(t("视差"))
        prefix = t("回看 · ") if self.review is not None else ""
        self.view_title.configure(text=prefix + ("  |  ".join(names) if names else t("未选择画面")))
        if image is None:
            self._photo = None
            if self.session is None and self.review is None and self.live_frame is None:
                self.view.configure(
                    image="",
                    text=t("尚未串流\n\n右上角开始串流，查看彩色、深度和点云\n保存的采集与快照在右侧点选即可回看"))
            else:
                self.view.configure(image="",
                                    text=t("没有可显示的画面\n在左侧勾选彩色、深度或视差，或切到 3D"))
            self._view_dirty = False
            return
        if not _HAVE_PIL:
            ok, buf = cv2.imencode(".png", image)
            if not ok:
                return
            import base64
            self._photo = tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.view.configure(image=self._photo, text="")
        self._view_dirty = False

    def _compose(self, w: int, h: int) -> Optional[np.ndarray]:
        self._hit = []
        if self.mode == "3d":
            pts, cols = self._cloud_arrays()
            img = np.zeros((h, w, 3), np.uint8)
            if pts is None or len(pts) == 0:
                cv2.putText(img, "No points in range", (24, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 1, cv2.LINE_AA)
                return img
            if self._fit_pending:
                self.cloud_dist = _suggest_dist(pts)
                self._fit_pending = False
            z = pts[:, 2]
            keep = np.isfinite(z) & (z >= self.zmin) & (z <= self.zmax)
            pts = pts[keep]
            cols = cols[keep] if cols is not None and len(cols) == len(keep) else cols
            if len(pts) == 0:
                cv2.putText(img, "No points in range", (24, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 1, cv2.LINE_AA)
                return img
            return render_cloud(pts, _rgb_uint8_to_bgr(cols, len(pts)), w, h,
                                self.cloud_yaw, self.cloud_pitch, self.cloud_dist,
                                self.cloud_pan[0], self.cloud_pan[1], point_px=self.point_px)
        tiles = self._collect_tiles()
        if not tiles:
            return None
        return self._layout(tiles, w, h)

    def _cloud_arrays(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        _frame, res = self._sources()
        if res is None or getattr(res, "points", None) is None or len(res.points) == 0:
            return None, None
        return res.points, res.colors

    def _collect_tiles(self) -> List[Dict[str, Any]]:
        frame, res = self._sources()
        tiles: List[Dict[str, Any]] = []
        if self.show_left.get() and frame is not None and getattr(frame, "left", None) is not None:
            tiles.append({"name": "Color L", "image": frame.left, "src_w": frame.left.shape[1]})
        if self.show_right.get() and frame is not None and getattr(frame, "right", None) is not None:
            tiles.append({"name": "Color R", "image": frame.right, "src_w": frame.right.shape[1]})
        if self.show_depth.get() and res is not None and getattr(res, "depth", None) is not None:
            depth = np.asarray(res.depth, np.float32)
            valid = np.asarray(res.valid, bool) if res.valid is not None else (depth > 0)
            show = valid & np.isfinite(depth) & (depth >= self.zmin) & (depth <= self.zmax)
            vis = attach_range_bar(colorize_depth(depth, show, self.zmin, self.zmax),
                                   self.zmin, self.zmax, unit="m")
            tiles.append({"name": "Depth", "image": vis, "src_w": int(depth.shape[1]),
                          "depth": depth, "valid": show})
        if self.show_disp.get() and res is not None and getattr(res, "disparity", None) is not None:
            vis, lo, hi = _colorize_disp(res.disparity)
            tiles.append({"name": "Disparity", "image": attach_range_bar(vis, hi, lo, unit="px"),
                          "src_w": int(res.disparity.shape[1])})
        return tiles

    def _layout(self, tiles: List[Dict[str, Any]], w: int, h: int) -> np.ndarray:
        n = len(tiles)
        gap = 6
        header = 26
        cell_w = max(1, (w - gap * (n - 1)) // n)
        body_h = max(1, h - header)
        canvas = np.zeros((h, w, 3), np.uint8)
        x = 0
        for tile in tiles:
            cell, ox, oy, scale = self._place(tile["image"], cell_w, body_h)
            canvas[header:header + body_h, x:x + cell_w] = cell
            cv2.rectangle(canvas, (x, 0), (x + cell_w, header), (20, 20, 20), -1)
            # 标题字号跟着格子宽度收缩：窄窗口下 "Color L"/"Disparity" 会顶出格子
            label = tile["name"]
            fscale, _ = 0.55, None
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)[0][0]
            if tw > cell_w - 20:
                fscale = max(0.35, fscale * (cell_w - 20) / float(tw))
            cv2.putText(canvas, label, (x + 10, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        fscale, (235, 235, 235), 1, cv2.LINE_AA)
            self._hit.append({
                "name": tile["name"].lower(),
                "x0": x + ox,
                "y0": header + oy,
                "scale": scale,
                "src_w": int(tile["src_w"]),
                "depth": tile.get("depth"),
                "valid": tile.get("valid"),
            })
            if tile["name"] == "Depth" and self._cursor:
                cv2.putText(canvas, self._cursor, (x + 10, h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            x += cell_w + gap
        return canvas

    @staticmethod
    def _place(image: np.ndarray, cell_w: int, cell_h: int):
        """把画面放进格子，返回 (画布, 左偏移, 上偏移, 缩放)。偏移用于把鼠标换算回像素。"""
        src = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        sh, sw = src.shape[:2]
        scale = min(cell_w / float(sw), cell_h / float(max(sh, 1)))
        nw = max(1, int(round(sw * scale)))
        nh = max(1, int(round(sh * scale)))
        nw = min(nw, cell_w)
        nh = min(nh, cell_h)
        resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((cell_h, cell_w, 3), np.uint8)
        x0 = (cell_w - nw) // 2
        y0 = (cell_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas, x0, y0, scale if scale > 0 else 1.0

    # ------------------------------------------------------------------ 鼠标
    def _press(self, event) -> None:
        kind = "pan" if event.num == 3 else "orbit"
        self._drag = (kind, event.x, event.y)

    def _drag_move(self, event) -> None:
        if self._drag is None or self.mode != "3d":
            return
        kind, x0, y0 = self._drag
        dx, dy = event.x - x0, event.y - y0
        self._drag = (kind, event.x, event.y)
        if kind == "orbit":
            self.cloud_yaw += dx * 0.01
            self.cloud_pitch = float(np.clip(self.cloud_pitch + dy * 0.01, -1.35, 1.35))
        else:
            self.cloud_pan[0] += dx
            self.cloud_pan[1] += dy
        self._paint_view()

    def _release(self, event) -> None:
        self._drag = None

    def _view_wheel(self, event) -> None:
        if self.mode != "3d":
            return
        self.cloud_dist = float(np.clip(self.cloud_dist * (0.9 if event.delta > 0 else 1.1), 0.05, 80.0))
        self._paint_view()

    def _hover(self, event) -> None:
        if self.mode != "2d":
            return
        text = ""
        for hit in self._hit:
            if hit["name"] != "depth" or hit.get("depth") is None:
                continue
            scale = float(hit["scale"] or 1.0)
            u = int((event.x - hit["x0"]) / scale)
            v = int((event.y - hit["y0"]) / scale)
            depth = hit["depth"]
            if u < 0 or v < 0 or u >= hit["src_w"] or v >= depth.shape[0] or u >= depth.shape[1]:
                continue
            valid = hit.get("valid")
            if valid is not None and not bool(valid[v, u]):
                text = "invalid"
            else:
                z = float(depth[v, u])
                text = f"{z:.3f} m" if z > 0 else "invalid"
        if text != self._cursor:
            self._cursor = text
            self._paint_view()

    # ------------------------------------------------------------------ 串流
    def _toggle_stream(self) -> None:
        if self.busy:
            self._log(t("正在连接或断开，请稍候"))
            return
        if self._calib_proc is not None and self._calib_proc.poll() is None:
            self._log(t("标定工作台还开着，相机会被它占用"))
            return
        if self.session is not None and getattr(self.session, "started", False):
            self._request_stop()
        else:
            self._request_start()

    def _request_start(self) -> None:
        self.busy = True
        self.btn_stream.configure(text=t("正在连接…"), state="disabled", bg=RAISED, fg=MUTED)
        self._paint_dot(WARN)
        self.dev_text.configure(text=t("正在连接"))
        self.status_var.set(t("正在打开相机并加载模型"))
        self._log(t("正在打开相机并加载模型…"))
        threading.Thread(target=self._start_worker, daemon=True).start()

    def _start_worker(self) -> None:
        try:
            from stereo_sdk.session import Session
            session = Session(self.config_path)
            session.start()
            self.q.put(("started", session))
        except Exception as exc:  # noqa: BLE001
            self.q.put(("start_failed", f"{type(exc).__name__}: {exc}"))

    def _request_stop(self) -> None:
        self.busy = True
        self.btn_stream.configure(text=t("正在停止…"), state="disabled", bg=RAISED, fg=MUTED)
        self.status_var.set(t("正在停止串流"))
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self) -> None:
        session = self.session
        err = ""
        try:
            if session is not None:
                session.stop()
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        self.q.put(("stopped", err))

    def _on_started(self, session) -> None:
        self.session = session
        self.busy = False
        self.review = None
        self.btn_back.pack_forget()
        self._fit_pending = True
        self.btn_stream.configure(text=t("停止串流"), state="normal", bg=DANGER, fg="#ffffff",
                                  activebackground="#d44854", activeforeground="#ffffff")
        self._paint_dot(OK)
        self.dev_text.configure(text=t("串流中"))
        self.status_var.set(t("串流中"))
        try:
            params = session.get_params()
            self._slider_lock = True
            self.scale_zmin.set(float(params.get("zmin") or self.zmin))
            self.scale_zmax.set(float(params.get("zmax") or self.zmax))
            self._slider_lock = False
            self.zmin = float(self.scale_zmin.get())
            self.zmax = float(self.scale_zmax.get())
            self._push_range()
        except Exception:  # noqa: BLE001
            self._slider_lock = False
        self._log(t("串流已开始"))
        err = session.last_error() if hasattr(session, "last_error") else None
        if err:
            self._log(err)

    def _on_stopped(self, err: str) -> None:
        self.session = None
        self.live_frame = None
        self.live_res = None
        self.busy = False
        self.btn_stream.configure(text=t("开始串流"), state="normal", bg=ACCENT, fg=ON_ACCENT,
                                  activebackground="#4ad6ee", activeforeground=ON_ACCENT)
        self._paint_dot(MUTED)
        self.dev_text.configure(text=t("未连接"))
        self.dev_sub.configure(text=t("开始串流后显示分辨率和帧率"))
        self.status_var.set(t("已停止"))
        self.meta_var.set(t("分辨率 —    帧率 —"))
        self._photo = None
        self.view.configure(image="",
                            text=t("串流已停止\n\n右上角可以再次开始\n右侧结果仍可点开回看"))
        if err:
            self._log(tf("停止时出错：{exc}", exc=err))
        else:
            self._log(t("串流已停止"))
        if self._pending_calib:
            self._pending_calib = False
            self._spawn_calib()
        if self._closing:
            self.destroy()

    def _on_range(self, _value=None) -> None:
        if self._slider_lock:
            return
        lo = float(self.scale_zmin.get())
        hi = float(self.scale_zmax.get())
        if hi < lo + 0.05:
            hi = lo + 0.05
            self._slider_lock = True
            self.scale_zmax.set(hi)
            self._slider_lock = False
        self.zmin, self.zmax = lo, hi
        if self._param_job is not None:
            try:
                self.after_cancel(self._param_job)
            except Exception:  # noqa: BLE001
                pass
        self._param_job = self.after(150, self._push_range)
        self._paint_view()

    def _push_range(self) -> None:
        self._param_job = None
        session = self.session
        if session is None or not getattr(session, "started", False):
            return
        try:
            session.set_params(zmin=self.zmin, zmax=self.zmax)
            builder = getattr(session, "_builder", None)
            if builder is not None:
                builder.zmin_m = float(self.zmin)
                builder.zmax_m = float(self.zmax)
        except Exception as exc:  # noqa: BLE001
            self._log(f"深度范围没有生效：{exc}")

    def _on_point(self, value) -> None:
        self.point_px = max(1, int(float(value)))
        if self.mode == "3d":
            self._paint_view()

    def _on_preset(self, _event=None) -> None:
        session = self.session
        if session is None or not getattr(session, "started", False):
            return
        if self.busy:
            self._log(t("正在忙，档位稍后再切"))
            return
        name = self._preset_key()
        try:
            current = str(session.get_params().get("preset") or "")
        except Exception:  # noqa: BLE001
            current = ""
        if name == current:
            return
        self.busy = True
        self.status_var.set(tf("正在切换到{name}", name=self.preset_var.get()))
        self._log(tf("正在切换到{name}，模型会重新加载", name=self.preset_var.get()))
        threading.Thread(target=self._preset_worker, args=(name,), daemon=True).start()

    def _preset_worker(self, name: str) -> None:
        session = self.session
        try:
            got = session.set_params(preset=name) if session is not None else {}
            self.q.put(("preset_done", got))
        except Exception as exc:  # noqa: BLE001
            self.q.put(("preset_done", {"error": str(exc)}))

    def _save_frame(self) -> None:
        session = self.session
        if session is None or not getattr(session, "started", False):
            self._log(t("还没有串流，无法保存"))
            return
        def work():
            try:
                path = session.save_snapshot()
                self.q.put(("saved", path or ""))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("saved", f"ERROR {exc}"))
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ 标定与结果
    def _open_calib(self) -> None:
        if self._calib_proc is not None and self._calib_proc.poll() is None:
            self._log(t("标定工作台已经开着"))
            return
        if self.busy:
            self._log(t("正在连接或断开，请稍候再打开标定"))
            return
        if self.session is not None and getattr(self.session, "started", False):
            self._pending_calib = True
            self._log(t("先停止串流，把相机交给标定工作台"))
            self._request_stop()
            return
        self._spawn_calib()

    def _spawn_calib(self) -> None:
        try:
            import app_paths
            frozen = app_paths.is_frozen()
        except Exception:  # noqa: BLE001
            frozen = bool(getattr(sys, "frozen", False))
        if frozen:
            cmd = [sys.executable, "--gui"]
        else:
            gui = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_gui.py")
            cmd = [sys.executable, gui]
        try:
            self._calib_proc = subprocess.Popen(cmd)
        except Exception as exc:  # noqa: BLE001
            self._log(f"打不开标定工作台：{exc}")
            return
        self._log(t("已打开标定工作台。采集、标定、检查、保存都在那个窗口里。"))
        self.status_var.set(t("标定工作台已打开"))
        self.after(600, self._poll_calib)

    def _poll_calib(self) -> None:
        proc = self._calib_proc
        if proc is not None and proc.poll() is None:
            self.after(600, self._poll_calib)
            return
        self._calib_proc = None
        self.status_var.set(t("已停止"))
        self._log(t("标定工作台已关闭，正在刷新结果"))
        self._refresh_results()

    def _refresh_results(self) -> None:
        self._set_text(self.calib_box, self._calib_summary())
        items: List[Tuple[str, str]] = []
        root = _data("output")
        for sub, kind in (("calib", t("采集")), ("captures", t("快照")), ("snapshots", t("快照"))):
            base = os.path.join(root, sub)
            if not os.path.isdir(base):
                continue
            dirs = []
            for name in os.listdir(base):
                path = os.path.join(base, name)
                if os.path.isdir(path):
                    dirs.append(path)
            dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for path in dirs[:30]:
                items.append((f"[{kind}]  {os.path.basename(path)}", path))
        self._result_paths = [p for _, p in items]
        self.result_list.delete(0, "end")
        for label, _path in items:
            self.result_list.insert("end", label)
        if not items:
            self.result_list.insert("end", t("（还没有采集或快照）"))
            self._result_paths = []

    def _calib_summary(self) -> str:
        path = _find("calibration/stereo_calib.yaml")
        if not os.path.isfile(path):
            return t("当前没有生效的标定。\n打开标定工作台，完成采集、标定并保存后，这里会显示基线和 RMS。")
        try:
            from calibration import StereoCalibration
            cal = StereoCalibration.load(path)
        except Exception as exc:  # noqa: BLE001
            return tf("标定文件读不了：\n{exc}\n{path}", exc=exc, path=path)
        date = ""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            date = str(raw.get("date") or "")
        except Exception:  # noqa: BLE001
            date = ""
        lines = [
            os.path.basename(path),
            tf("日期      {date}", date=date or t("未知")),
            tf("图像      {w} x {h}", w=cal.image_size[0], h=cal.image_size[1]),
            tf("基线      {mm:.2f} mm", mm=cal.baseline_mm),
            tf("双目 RMS  {rms:.3f} px", rms=cal.rms_stereo),
            tf("单目 RMS  左 {left:.3f} / 右 {right:.3f} px", left=cal.rms_left, right=cal.rms_right),
            tf("校正焦距  {fx:.1f} px", fx=cal.rect_fx),
        ]
        try:
            import calib_manage
            rows = list(calib_manage.list_archives())
        except Exception:  # noqa: BLE001
            rows = []
        if rows:
            lines.append("")
            lines.append(tf("历史归档 {n} 份", n=len(rows)))
            for row in rows[:5]:
                when = str(row.get("archived_at") or row.get("date") or "")
                base = row.get("baseline_mm")
                rms = row.get("rms_stereo")
                extra = ""
                if isinstance(base, (int, float)) and isinstance(rms, (int, float)):
                    extra = f"   {float(base):.1f} mm   RMS {float(rms):.3f}"
                lines.append(f"· {when}{extra}")
        return "\n".join(lines)

    def _on_pick_result(self, _event=None) -> None:
        sel = self.result_list.curselection()
        if not sel or sel[0] >= len(self._result_paths):
            return
        path = self._result_paths[sel[0]]
        self._log(tf("正在读取 {name}", name=os.path.basename(path)))
        threading.Thread(target=self._load_review, args=(path,), daemon=True).start()

    def _load_review(self, path: str) -> None:
        try:
            payload = _read_capture(path)
        except Exception as exc:  # noqa: BLE001
            self.q.put(("review_failed", f"{type(exc).__name__}: {exc}"))
            return
        payload["label"] = os.path.basename(path)
        payload["path"] = path
        self.q.put(("review", payload))

    def _open_selected_dir(self) -> None:
        sel = self.result_list.curselection()
        if sel and sel[0] < len(self._result_paths):
            path = self._result_paths[sel[0]]
        else:
            path = _data("output")
        if not os.path.isdir(path):
            self._log(t("目录不存在：") + path)
            return
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            self._log(tf("打不开目录：{exc}", exc=exc))

    def _run_selftest(self) -> None:
        if self.busy:
            return
        self._log(t("正在做环境自检…"))
        threading.Thread(target=self._selftest_worker, args=("env",), daemon=True).start()

    def _run_model_test(self) -> None:
        if self.session is not None and getattr(self.session, "started", False):
            self._log(t("请先停止串流，再做模型自检（显存不够两件事一起做）"))
            return
        self._log(t("正在做模型自检，第一次会比较久…"))
        threading.Thread(target=self._selftest_worker, args=("model",), daemon=True).start()

    def _selftest_worker(self, kind: str) -> None:
        import io
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf  # type: ignore[assignment]
        rc = 1
        try:
            import app_entry
            if kind == "env":
                rc = int(app_entry._selftest() or 0)
            else:
                rc = int(app_entry._model_selftest() or 0)
        except Exception as exc:  # noqa: BLE001
            buf.write(f"\n{type(exc).__name__}: {exc}\n")
        finally:
            sys.stdout = old
        self.q.put(("report", (kind, rc, buf.getvalue())))

    # ------------------------------------------------------------------ 循环
    def _tick(self) -> None:
        if self._closing and self.session is None and not self.busy:
            return
        try:
            while True:
                kind, payload = self.q.get_nowait()
                self._on_queue(kind, payload)
        except queue.Empty:
            pass
        session = self.session
        if session is not None and getattr(session, "started", False) and not self.paused:
            try:
                frame, seq = session.latest_frame()
                if frame is not None and seq != self._frame_seq:
                    self.live_frame = frame
                    self._frame_seq = seq
                    if self.review is None and self.mode == "2d":
                        self._view_dirty = True
                if session.has_new_result():
                    res = session.take_result()
                    if res is not None:
                        self.live_res = res
                        if self.review is None:
                            self._view_dirty = True
                err = session.last_error() if hasattr(session, "last_error") else None
                if err and err != self._last_err:
                    self._last_err = err
                    self._log(err)
                st = session.stats()
                eye = ""
                if self.live_frame is not None:
                    eye = f"{self.live_frame.left.shape[1]}x{self.live_frame.left.shape[0]}"
                med = float(st.get("median_m") or 0.0)
                self.meta_var.set(tf(
                    "{eye}    采集 {cap:.1f} fps    推理 {inf:.2f} fps"
                    "    中位 {med:.3f} m    {preset}",
                    eye=eye or "—",
                    cap=float(st.get("capture_fps") or 0),
                    inf=float(st.get("infer_fps") or 0),
                    med=med, preset=st.get("preset") or ""))
                self.dev_sub.configure(text=tf(
                    "{eye}    有效 {valid:.0f}%    点数 {pts}",
                    eye=eye or t("分辨率未知"),
                    valid=float(st.get("valid_ratio") or 0),
                    pts=int(st.get("points") or 0)))
            except Exception as exc:  # noqa: BLE001
                self._log(f"刷新画面失败：{exc}")
        if self._view_dirty and not self.paused:
            try:
                self._paint_view()
            except Exception as exc:  # noqa: BLE001
                self._view_dirty = False
                self._log(f"绘制失败：{exc}")
        if not (self._closing and self.session is None and not self.busy):
            self.after(40, self._tick)

    def _on_queue(self, kind: str, payload: Any) -> None:
        if kind == "started":
            self._on_started(payload)
        elif kind == "start_failed":
            self.busy = False
            self.session = None
            self.btn_stream.configure(text=t("开始串流"), state="normal", bg=ACCENT)
            self._paint_dot(DANGER)
            self.dev_text.configure(text=t("连接失败"))
            self.status_var.set(t("连接失败"))
            self._log(str(payload))
        elif kind == "stopped":
            self._on_stopped(str(payload or ""))
        elif kind == "preset_done":
            self.busy = False
            self.status_var.set(t("串流中"))
            err = ""
            if isinstance(payload, dict):
                err = str(payload.get("error") or "")
                if not payload.get("applied") and not err:
                    err = t("档位没有切换")
            if err:
                self._log(err)
            else:
                self._log(tf("已切换到{name}", name=self.preset_var.get()))
                self._fit_pending = True
        elif kind == "saved":
            text = str(payload or "")
            if text.startswith("ERROR") or not text:
                self._log(text or t("还没有结果，未保存"))
            else:
                self._log(t("已保存 ") + text)
                self._refresh_results()
        elif kind == "review":
            self.review = _wrap_review(payload)
            self.btn_back.pack(side="right", padx=4, pady=6)
            self._fit_pending = True
            self._cursor = ""
            label = str(payload.get("label") or "")
            bits = []
            if payload.get("has_color"):
                bits.append(t("彩色"))
            if payload.get("has_depth"):
                bits.append(t("深度"))
            if payload.get("has_cloud"):
                bits.append(t("点云"))
            self._log(f"正在回看 {label}（{'、'.join(bits) or '没有可显示的图像'}）")
            if payload.get("has_cloud"):
                self._set_mode("3d")
            elif payload.get("has_depth"):
                self.show_depth.set(True)
                self._set_mode("2d")
            else:
                self.show_left.set(True)
                self.show_right.set(bool(payload.get("has_right")))
                self._set_mode("2d")
        elif kind == "review_failed":
            self._log(t("读取失败：") + str(payload))
        elif kind == "report":
            kind_name, rc, text = payload
            title = t("环境自检") if kind_name == "env" else t("模型自检")
            self._log(f"{title}结束，退出码 {rc}")
            self._set_text(self.calib_box, text.strip() or t("（没有输出）"))

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        proc = self._calib_proc
        if proc is not None and proc.poll() is None:
            self._log(t("标定工作台还开着，请先关掉它再退出"))
            self._closing = False
            return
        if self.session is not None and getattr(self.session, "started", False):
            self._request_stop()
            return
        if self.busy:
            self.after(200, self._on_close)
            return
        self.destroy()


def _colorize_disp(disp: np.ndarray) -> Tuple[np.ndarray, float, float]:
    d = np.asarray(disp, np.float32)
    ok = np.isfinite(d) & (d > 0.5)
    vis = np.zeros((*d.shape, 3), np.uint8)
    if not ok.any():
        return vis, 0.0, 1.0
    lo = float(np.percentile(d[ok], 2))
    hi = float(np.percentile(d[ok], 98))
    span = max(hi - lo, 1e-6)
    norm = np.clip((d - lo) / span, 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    vis[ok] = colored[ok]
    return vis, lo, hi


class _Img:
    """回看时凑一个和 FramePair 一样能取 .left / .right 的对象。"""

    def __init__(self, left: Optional[np.ndarray], right: Optional[np.ndarray]) -> None:
        self.left = left
        self.right = right


class _Res:
    def __init__(self, depth=None, valid=None, disparity=None, points=None, colors=None) -> None:
        self.depth = depth
        self.valid = valid
        self.disparity = disparity
        self.points = points if points is not None else np.zeros((0, 3), np.float32)
        self.colors = colors if colors is not None else np.zeros((0, 3), np.uint8)


def _wrap_review(payload: Dict[str, Any]) -> Dict[str, Any]:
    frame = None
    if payload.get("left") is not None or payload.get("right") is not None:
        frame = _Img(payload.get("left"), payload.get("right"))
    res = None
    if payload.get("has_depth") or payload.get("has_disp") or payload.get("has_cloud"):
        res = _Res(payload.get("depth"), payload.get("valid"), payload.get("disparity"),
                   payload.get("points"), payload.get("colors"))
    return {"frame": frame, "res": res, "label": payload.get("label")}


def _read_capture(path: str) -> Dict[str, Any]:
    left = _imread(os.path.join(path, "left_raw.png"))
    right = _imread(os.path.join(path, "right_raw.png"))
    if left is None:
        lefts = sorted(glob.glob(os.path.join(path, "left_*.png")))
        rights = sorted(glob.glob(os.path.join(path, "right_*.png")))
        if lefts:
            left = _imread(lefts[0])
        if rights:
            right = _imread(rights[0])
    depth = valid = disparity = None
    dp = os.path.join(path, "depth_meter.npy")
    if os.path.isfile(dp):
        depth = np.load(dp)
        vp = os.path.join(path, "valid.npy")
        if os.path.isfile(vp):
            valid = np.load(vp).astype(bool)
        else:
            valid = np.isfinite(depth) & (depth > 0)
    disp_path = os.path.join(path, "disparity.npy")
    if os.path.isfile(disp_path):
        disparity = np.load(disp_path)
    points = colors = None
    ply = os.path.join(path, "cloud.ply")
    if os.path.isfile(ply):
        points, colors = _load_ply(ply)
    return {
        "left": left, "right": right, "depth": depth, "valid": valid,
        "disparity": disparity, "points": points, "colors": colors,
        "has_color": left is not None, "has_right": right is not None,
        "has_depth": depth is not None, "has_disp": disparity is not None,
        "has_cloud": points is not None and len(points) > 0,
    }


def _selftest() -> int:
    """不连相机：建窗口、画一帧深度和点云、关掉。"""
    ok = True
    depth = np.linspace(0.2, 2.0, 90 * 120, dtype=np.float32).reshape(90, 120)
    valid = np.ones(depth.shape, dtype=bool)
    valid[:5, :] = False
    vis = attach_range_bar(colorize_depth(depth, valid, 0.2, 2.0), 0.2, 2.0, unit="m")
    if vis.ndim != 3 or vis.shape[1] <= depth.shape[1] or vis.dtype != np.uint8:
        print("  [FAIL] 深度色标尺寸不对", vis.shape)
        ok = False
    else:
        print(f"  [OK] 深度色标 {vis.shape[1]}x{vis.shape[0]}")
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 0.05, (800, 3)).astype(np.float32)
    pts[:, 2] += 0.8
    cols = np.full((800, 3), (40, 180, 220), np.uint8)
    cloud = render_cloud(pts, cols, 320, 240, 0.4, 0.2, 1.2, 0, 0, 2)
    if cloud.shape != (240, 320, 3) or int(cloud.sum()) == 0:
        print("  [FAIL] 点云没有画出来", cloud.shape, int(cloud.sum()))
        ok = False
    else:
        print(f"  [OK] 点云画面 {cloud.shape[1]}x{cloud.shape[0]}")
    app = ViewerApp(dpi_scale=1.0)
    app.after(250, app.destroy)
    app.mainloop()
    print("  [OK] 主窗口可创建并关闭")
    print("  " + ("[OK] Viewer 自检通过" if ok else "[FAIL] Viewer 自检失败"))
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Stereo Viewer")
    ap.add_argument("--selftest", action="store_true", help="只建界面并画一帧测试图，然后退出")
    ap.add_argument("--lang", default="", choices=("", "zh", "en"), help="界面语言 zh / en")
    args = ap.parse_args(argv)
    # 命令行优先；否则读 config.yaml 的 ui.language（app_entry 里可能已经定过）
    import i18n
    i18n.resolve_language(_find("config.yaml"), args.lang)
    if args.selftest:
        scale = 1.0
    else:
        scale = _enable_dpi()
    if args.selftest:
        return _selftest()
    app = ViewerApp(dpi_scale=scale)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
