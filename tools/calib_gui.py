"""calib_gui.py —— 双目标定四页 GUI（tkinter，零第三方界面依赖）
=============================================================================
【这四页对应你要求的四个动作】
    ① 采集  实时预览左右图 + 角点叠加 + 质量指标 + 采集清单，合格才允许保存
    ② 标定  选采集目录 → 后台线程算（界面不卡）→ 显示 RMS/基线/逐帧最差
    ③ 检查  校正后上下叠放 + 水平参考线、红青、差值、棋盘混合、覆盖热力图、分区 |dy|
    ④ 保存  写 YAML → 自动归档 → 列出历史 → 激活 / 对比 / 结构校验

【设计约束（都是刻意选的）】
    1) 界面线程只做显示：相机读取与标定计算都在后台线程，用"只保留最新"的槽位交接，
       不排队、不阻塞（与 pipeline_threaded.py 同一套原则）。
    2) 质检判据不在这里重写：全部来自 capture_calib（analyze_pair / gate_messages），
       这里只负责把结果显示出来。判据改一处，CLI 与 GUI 同步生效。
    3) 中文只出现在 tkinter 控件里，画到图像上的文字一律英文（cv2.putText 不支持中文）。

【依赖】
    Python 3.10+（标准安装自带 tkinter）、opencv-python、numpy、pyyaml、pillow
    没有相机也能启动：点"打开相机"失败时，其余三页仍可用已有采集目录跑。

【用法】
    python tools\\calib_gui.py
    python tools\\calib_gui.py --dir output\\calib\\20261013_151253
    python tools\\calib_gui.py --selftest        REM 只建界面不连相机（用于确认环境）
=============================================================================
"""

from __future__ import annotations

import base64
import glob
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# ---- 依赖检查：给出人话提示，而不是一串 ImportError 栈 ----
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:                        # pragma: no cover
    print("[错误] 当前 Python 没有 tkinter。请安装官方 python.org 的 CPython 发行版"
          "（安装时勾选 tcl/tk），而不是精简版/绿色版解释器。")
    raise SystemExit(1)

# pillow 只是「更快的显示通道」，不是必需品。
# 缺少 pillow 时退回 Tk 自带的 PNG 解码（Tk >= 8.6），功能完全一致，只是每帧多一次编码。
try:
    from PIL import Image, ImageTk
    _HAVE_PIL = True
except ImportError:                        # pragma: no cover
    Image = ImageTk = None                 # type: ignore
    _HAVE_PIL = False

import cv2
import numpy as np

# ---- 路径解析（源码运行 = 项目根；打包成 EXE 后 = exe 所在目录）----
# 打包后 __file__ 在 _MEIPASS 里，用 dirname(__file__) 会让 output/ 与 calibration/
# 落到临时目录（退出即删）。统一交给 app_paths 判断，见该文件头部的说明。
try:
    from app_paths import data_root as _data_root, find as _find, writable_dir as _writable_dir
except ImportError:                    # tools/ 被单独拷走时的兜底，行为与改造前一致
    def _data_root() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _find(rel: str) -> str:
        return os.path.join(_data_root(), rel)

    def _writable_dir(rel: str) -> str:
        p = os.path.join(_data_root(), rel)
        os.makedirs(p, exist_ok=True)
        return p

PROJECT_ROOT = _data_root()
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import calib_check as chk              # noqa: E402
import calib_core as core              # noqa: E402
import calib_manage as manage          # noqa: E402
import gui_state as gstate             # noqa: E402
import gui_theme as theme               # noqa: E402

#: 系统缩放比，由 main() 里的 _enable_dpi_awareness() 填入（1.0 = 100%）
_DPI_SCALE = 1.0


# -----------------------------------------------------------------------------
# 显示通道：有 pillow 走 pillow，没有就走 Tk 自带 PNG 解码（Tk >= 8.6）
# -----------------------------------------------------------------------------
def _make_photo(bgr: np.ndarray):
    """BGR ndarray -> Tk 可显示的 PhotoImage；失败返回 None。"""
    if _HAVE_PIL:
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    if not ok:
        return None
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))


# -----------------------------------------------------------------------------
# 后台线程：相机预览 + 逐帧分析
# -----------------------------------------------------------------------------
class CameraWorker(threading.Thread):
    """独占相机，循环"读取 → 分析"；只把**最新一帧**交给界面，从不排队。"""

    def __init__(self, cam_cfg: Dict[str, Any], spec: Any, thr: Dict[str, float],
                 preview_fps: float = 8.0, cv_threads: int = 1) -> None:
        super().__init__(daemon=True, name="calib-camera")
        self.cfg = cam_cfg
        self.spec = spec
        self.thr = dict(thr)
        # 预览分析帧率上限与 OpenCV 线程数：见 run() 里为什么必须限制
        self.preview_fps = float(preview_fps or 0.0)
        self.cv_threads = int(cv_threads or 1)
        self.error = ""
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._payload: Optional[Tuple[Any, Dict[str, Any], Optional[Dict[str, Any]], List[str], float]] = None
        self._seq = 0
        self._taken = 0
        self.cam = None

    # ---------------------------------------------------------------- 线程体
    def run(self) -> None:
        try:
            from camera import StereoCamera
            cam = StereoCamera(self.cfg)
            cam.open()
        except Exception as exc:            # noqa: BLE001
            self.error = f"打开相机失败：{type(exc).__name__}: {exc}"
            self._stop.set()
            return

        self.cam = cam
        cc = core._cc()
        stamps: deque = deque(maxlen=15)
        # 并发控制：OpenCV 默认按核数开线程池（本机 24 个）。实测 analyze_pair 用
        # 1 线程与 24 线程的**耗时相同**（107.8 vs 108.1 ms），但 24 线程要多烧
        # 约 0.5 个核（CPU 152% → 97%）。多出来的线程只是在空转抢核，会把整机拖卡。
        prev_threads = cv2.getNumThreads()
        try:
            cv2.setNumThreads(max(int(self.cv_threads), 1))
        except Exception:                   # noqa: BLE001
            prev_threads = 0
        interval = (1.0 / self.preview_fps) if self.preview_fps > 0 else 0.0
        next_t = 0.0
        try:
            while not self._stop.is_set():
                if interval > 0:            # 限速：相机按需读，不必空转解码
                    now = time.perf_counter()
                    if now < next_t:
                        time.sleep(min(next_t - now, 0.02))
                        continue
                    next_t = now + interval
                try:
                    pair = cam.read()
                except Exception as exc:    # noqa: BLE001
                    self.error = f"读取失败：{type(exc).__name__}: {exc}"
                    break
                try:
                    res = cc.analyze_pair(pair.left, pair.right, self.spec, strict=True)
                    pose = cc.classify_pose(res, pair.left.shape[:2], self.spec)
                    reasons = [m[0] for m in cc.gate_messages(res, self.thr)]
                except Exception as exc:    # noqa: BLE001
                    self.error = f"分析失败：{type(exc).__name__}: {exc}"
                    break
                now = time.perf_counter()
                stamps.append(now)
                fps = 0.0
                if len(stamps) > 1 and stamps[-1] > stamps[0]:
                    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
                with self._lock:
                    self._payload = (pair, res, pose, reasons, fps)
                    self._seq += 1
        finally:
            if prev_threads:
                try:
                    cv2.setNumThreads(prev_threads)
                except Exception:           # noqa: BLE001
                    pass
            try:
                cam.close()
            except Exception:               # noqa: BLE001
                pass

    # ---------------------------------------------------------------- 取数据
    def take(self):
        with self._lock:
            if self._payload is None or self._seq == self._taken:
                return None
            self._taken = self._seq
            return self._payload

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2.5)


# -----------------------------------------------------------------------------
# 后台线程：标定计算 / 检查图生成
# -----------------------------------------------------------------------------
class Job(threading.Thread):
    """跑一个阻塞任务，把进度与结果丢进队列，由界面线程消费。"""

    def __init__(self, kind: str, q, fn, *args, **kwargs) -> None:
        super().__init__(daemon=True, name=f"calib-{kind}")
        self.kind = kind
        self.q = q
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            out = self.fn(*self.args, **self.kwargs)
            self.q.put(("done", (self.kind, out)))
        except core.Cancelled:              # 用户点了取消，不是失败
            self.q.put(("cancelled", self.kind))
        except Exception as exc:            # noqa: BLE001
            self.q.put(("error", (self.kind, f"{type(exc).__name__}: {exc}")))


# -----------------------------------------------------------------------------
# 帧缩略图网格
# -----------------------------------------------------------------------------
class FrameGrid(tk.Toplevel):
    """把已保存的每一帧摆成网格：一眼看出哪张姿势不对、哪张糊了、板子偏了。

    为什么必须看图而不是看数字：角点数 48 的两帧，可能一帧板子在中央、一帧贴边被裁。
    数字一样，价值完全不同。缩略图是唯一能立刻分辨的画面证据。

    items: [{"idx", "left", "right", "desc", "n"}]；on_delete(rec_idx) 由调用方提供时
    才允许删除（本次会话的内存记录才有）；否则只能看。
    """

    COLS = 4
    TH_W, TH_H = 168, 126

    def __init__(self, master: tk.Misc, items, on_delete=None,
                 title: str = "帧缩略图") -> None:
        super().__init__(master)
        # Toplevel 自身的底色不在 option 数据库管辖范围内（那是按控件类匹配的），
        # 必须显式设一次，否则对话框边缘会露出一圈系统灰
        self.configure(background=theme.BG)
        self.title(f"{title}（共 {len(items)} 帧）")
        self.geometry("780x620")
        self.transient(master)
        self.items = list(items)
        self.on_delete = on_delete
        self._photos: Dict[int, Any] = {}
        self._sel: int = -1
        self._cells: List[tk.Frame] = []

        hint = ("单击选中，双击看大图；"
                + ("选中后按 Delete 删除（连磁盘文件一起删）。" if on_delete else "本目录来自磁盘，只读。"))
        tk.Label(self, text=hint, anchor="w", background=theme.BG, foreground=theme.MUTED,
                 font=theme.font(9)).pack(fill="x", padx=12, pady=(10, 6))

        wrap = tk.Frame(self, background=theme.BG)
        wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.canvas = tk.Canvas(wrap, highlightthickness=0, background=theme.BG)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        sb.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=sb.set)

        self.inner = tk.Frame(self.canvas, background=theme.BG)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self._win, width=e.width))

        self.canvas.bind("<Enter>", lambda _e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda _e: self.canvas.unbind_all("<MouseWheel>"))
        self.bind("<Delete>", self._delete_sel)
        self.bind("<Escape>", lambda _e: self.destroy())

        self._build()

    # ------------------------------------------------------------------ 构建
    def _build(self) -> None:
        for c in self.inner.winfo_children():
            c.destroy()
        self._cells = []
        for pos, it in enumerate(self.items):
            r, c = divmod(pos, self.COLS)
            cell = tk.Frame(self.inner, background=theme.CARD, padx=4, pady=4,
                            highlightthickness=1, highlightbackground=theme.BORDER)
            cell.grid(row=r, column=c, padx=5, pady=5, sticky="n")

            img = cv2.imread(it["left"])
            if img is None:
                thumb = np.zeros((self.TH_H, self.TH_W, 3), np.uint8)
                cv2.putText(thumb, "LOAD FAIL", (10, self.TH_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                h, w = img.shape[:2]
                s = min(self.TH_W / float(w), self.TH_H / float(h))
                thumb = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)),
                                   interpolation=cv2.INTER_AREA)
            ph = _make_photo(thumb)
            self._photos[it["idx"]] = ph
            lab = tk.Label(cell, image=ph, background=theme.CARD, borderwidth=0)
            lab.pack()
            txt = f"#{it['idx']:02d}  {it['desc']}\n角点 {it['n'] if it['n'] >= 0 else '—'}"
            cap = tk.Label(cell, text=txt, background=theme.CARD, foreground=theme.TEXT,
                           font=theme.font(8, mono=True), justify="center")
            cap.pack()
            for wdg in (cell, lab, cap):
                wdg.bind("<Button-1>", lambda _e, p=pos: self._select(p))
                wdg.bind("<Double-Button-1>", lambda _e, p=pos: self._preview(p))
            self._cells.append(cell)

    # ------------------------------------------------------------------ 交互
    def _wheel(self, evt) -> None:
        self.canvas.yview_scroll(int(-1 * (evt.delta / 120.0)), "units")

    def _select(self, pos: int) -> None:
        # 选中态靠"底色 + 描边"两层信号：只换底色在深色下对比不够，容易看不出选中谁
        for i, c in enumerate(self._cells):
            hit = i == pos
            bg = "#1f3b5e" if hit else theme.CARD
            c.configure(background=bg,
                        highlightbackground=theme.ACCENT if hit else theme.BORDER)
            for w in c.winfo_children():
                if isinstance(w, tk.Label) and w.cget("text"):
                    w.configure(background=bg)
        self._sel = pos

    def _preview(self, pos: int) -> None:
        it = self.items[pos]
        img = cv2.imread(it["left"])
        if img is None:
            return
        win = tk.Toplevel(self)
        win.title(f"第 {it['idx']:02d} 帧 · {it['desc']} · 角点 {it['n']}")
        lbl = tk.Label(win)
        lbl.pack()
        self._show_in(lbl, img, "prev", max_w=980, max_h=720)
        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=6)

    def _show_in(self, label: tk.Label, img: np.ndarray, key: str,
                 max_w: int, max_h: int) -> None:
        h, w = img.shape[:2]
        s = min(max_w / float(w), max_h / float(h), 1.0)
        if s < 0.999:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ph = _make_photo(img)
        if ph is not None:
            label.configure(image=ph)
            label.image = ph
            self._photos[f"{key}"] = ph

    def _delete_sel(self, _evt=None) -> None:
        if self.on_delete is None or not (0 <= self._sel < len(self.items)):
            return
        it = self.items[self._sel]
        if not messagebox.askyesno("确认删除",
                                   f"删除第 {it['idx']:02d} 帧？\n\n{it['left']}\n{it['right']}\n\n"
                                   "这两个文件会从磁盘删除，不可撤销。", parent=self):
            return
        self.on_delete(it["idx"])
        self.destroy()


# -----------------------------------------------------------------------------
# 主界面
# -----------------------------------------------------------------------------
class CalibApp(tk.Tk):

    def __init__(self, start_dir: str = "") -> None:
        super().__init__()
        # DPI 与主题必须在建任何控件之前完成：控件一旦创建就已经按当时的字体
        # 与默认调色板渲染，事后再改 Style 只能改到一部分，会留下深浅混搭。
        if _DPI_SCALE > 1.01:
            try:
                self.tk.call("tk", "scaling", _DPI_SCALE * 1.3333)
            except Exception:               # noqa: BLE001
                pass
        theme.apply(self)
        self.title("双目标定工作台 —— 采集 / 标定 / 检查 / 保存")
        # 记忆：窗口尺寸与位置、上次目录、阈值、标定板参数（见 tools/gui_state.py）
        self._has_state = os.path.exists(gstate.state_path())
        self.state = gstate.load()
        self.geometry(self.state.get("geometry") or "1280x880")
        self.minsize(1040, 720)

        # ---- 配置与状态
        self.spec, self.cfg_cal = core.spec_from_config()
        self.session: Optional[core.CaptureSession] = None
        self.worker: Optional[CameraWorker] = None
        self.job: Optional[Job] = None
        self.cancel_evt: Optional[threading.Event] = None
        self.calib_out: Optional[Dict[str, Any]] = None
        self.check_out: Optional[Dict[str, Any]] = None
        self.frames_dir = start_dir or self.state.get("dir", "")
        self._photos: Dict[str, Any] = {}        # 防止 PhotoImage 被回收
        self._last_img: Dict[str, np.ndarray] = {}   # 每个显示区的原图，供缩放重绘
        self._fit_job = None                     # 尺寸变化后的重绘防抖句柄
        self._last_frame = None                  # 最近一帧（保存按钮要用，take() 会消费掉）
        self.q: "queue.Queue" = queue.Queue()

        self._build_vars()
        self._build_ui()
        # 窗口/面板尺寸一变就按新尺寸重画：不带这个绑定，最大化后图仍是建窗时的大小，
        # 四周留大片黑边（实测标签 1069x1268、图只有 430x322，仅占 10.2% 面积）
        for _lbl in (self.lbl_left, self.lbl_right, self.lbl_check):
            _lbl.bind("<Configure>", self._on_image_resize, add="+")
        self._refresh_dirs()
        self._refresh_archives()
        self._update_check_hint()
        # 恢复上次停留的页（越界时 _build_vars 已夹紧到 0~3）
        try:
            self.nb.select(self.state.get("tab", 0))
        except Exception:                   # noqa: BLE001
            pass
        self.var_mode.trace_add("write", lambda *a: self._render_check())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick_id: Optional[str] = self.after(30, self._tick)
        self.bind("<space>", lambda e: self._on_hotkey("save", e))
        # 大写 keysym 要单独绑：Tk 按 keysym 匹配，按住 Shift 时 keysym 变成 "F"/"U"
        for _key, _act in (("<Key-f>", "force"), ("<Key-F>", "force"),
                           ("<Key-u>", "undo"), ("<Key-U>", "undo")):
            self.bind(_key, lambda e, a=_act: self._on_hotkey(a, e))
        # 切页时补齐该页的"进入动作"：不补的话③页会显示上次的旧结论，
        # 用户以为标定结果没更新
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._refresh_checklist()
        # tkinter 回调里的异常默认只打到 stderr。打包成 EXE（--noconsole）后 stderr 是 None，
        # 异常会**静默消失**，界面看起来"点了没反应"。这里改成弹窗 + 写日志。
        self.report_callback_exception = self._on_tk_error

    # -------------------------------------------------------------- 异常与诊断
    def _on_tk_error(self, exc_type, exc, tb) -> None:
        """界面回调里未捕获的异常：落盘 + 弹窗（无控制台时这是唯一线索）。"""
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log_path = ""
        try:
            log_path = os.path.join(_writable_dir("logs"), "gui-errors.log")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] pid={os.getpid()}\n{text}")
        except OSError:
            log_path = ""
        try:
            messagebox.showerror(
                "操作失败",
                f"{exc_type.__name__}: {exc}\n\n"
                + (f"详情已写入：\n{log_path}" if log_path else "（日志写入失败）"))
        except Exception:                      # noqa: BLE001 —— 弹窗本身失败不能再抛
            pass

    def _show_env(self) -> None:
        """环境信息：打包后用户自查的第一步（路径错是最常见的"为什么存不了"）。"""
        try:
            import app_paths
            body = app_paths.describe_text()
            body += "\n\n搜索顺序（按优先级）：\n" + "\n".join(
                f"  {i}. {p}" for i, p in enumerate(app_paths.search_roots(), 1))
        except Exception as exc:               # noqa: BLE001
            body = f"app_paths 不可用：{exc}\n\n项目根：{PROJECT_ROOT}"
        body += (f"\n\n采集根目录：{core.CAPTURE_ROOT}"
                 f"\n标定文件  ：{manage.CURRENT}"
                 f"\n归档目录  ：{manage.ARCHIVE_DIR}"
                 f"\n日志目录  ：{_writable_dir('logs')}"
                 f"\n界面记忆  ：{gstate.state_path()}")

        win = tk.Toplevel(self)
        win.title("环境信息")
        theme_pad = 10
        win.configure(background=theme.BG)
        txt = tk.Text(win, wrap="none", width=96, height=20, font=("Consolas", 9))
        theme.style_text(txt)
        txt.pack(fill="both", expand=True, padx=theme_pad, pady=(theme_pad, 6))
        txt.insert("1.0", body)
        txt.configure(state="disabled")
        row = tk.Frame(win, background=theme.BG)
        row.pack(fill="x", padx=theme_pad, pady=(0, theme_pad))

        def _copy() -> None:
            self.clipboard_clear()
            self.clipboard_append(body)
            self.var_status.set("环境信息已复制到剪贴板")

        ttk.Button(row, text="复制全部", style="Primary.TButton", width=0,
                   command=_copy).pack(side="left")
        ttk.Button(row, text="关闭", style="Ghost.TButton", width=0,
                   command=win.destroy).pack(side="right")
        win.transient(self)

    # ------------------------------------------------------------------ 变量
    def _build_vars(self) -> None:
        cal = self.cfg_cal
        st = self.state
        thr = st.get("thresholds", {})
        brd = st.get("board", {})

        def pick(state_val, cfg_val):
            """有记忆文件就用记忆值，否则用 config.yaml（首次运行按配置走）。"""
            return state_val if self._has_state else cfg_val

        self.var_cols = tk.IntVar(value=int(pick(brd.get("cols", 12), cal.get("board_cols", 12))))
        self.var_rows = tk.IntVar(value=int(pick(brd.get("rows", 9), cal.get("board_rows", 9))))
        self.var_square = tk.DoubleVar(
            value=float(pick(brd.get("square", 3.0), cal.get("square_mm", 3.0))))
        self.var_target = tk.IntVar(
            value=int(pick(thr.get("target", 26), cal.get("target_frames", 26))))
        self.var_px = tk.DoubleVar(
            value=float(pick(thr.get("px", 8.0), cal.get("min_px_per_square", 8.0))))
        self.var_sharp = tk.DoubleVar(
            value=float(pick(thr.get("sharp", 100.0), cal.get("min_sharpness", 100.0))))
        self.var_contrast = tk.DoubleVar(
            value=float(pick(thr.get("contrast", 120.0), cal.get("min_contrast", 120.0))))

        self.var_dir = tk.StringVar(value=self.frames_dir)
        self.var_fix = tk.BooleanVar(value=bool(pick(st.get("fix_intrinsic", True), True)))
        self.var_mode = tk.StringVar(value=st.get("check_mode") or gstate.CHECK_MODES[0])
        self.var_alpha = tk.DoubleVar(value=float(st.get("alpha", 0.0)))
        self.var_save_path = tk.StringVar(value=manage.CURRENT)
        self.var_note = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="就绪")
        self.var_ready = tk.StringVar(value="相机未打开")
        self.var_chk_hint = tk.StringVar(value="")
        self.var_short = tk.StringVar(value="还没开始采集")

    # ------------------------------------------------------------------ 界面
    @staticmethod
    def _dep_text() -> str:
        """顶栏右侧那一行：让人一眼知道"这界面是用哪套 OpenCV 跑的"。

        装了两个 OpenCV（系统 + venv）导致结果不一致，是这类项目最常见的求助原因，
        把版本放在窗口上比藏在"关于"里有用。
        """
        bits = [f"cv2 {cv2.__version__}", f"numpy {np.__version__}"]
        bits.append("pillow" if _HAVE_PIL else "Tk-PNG")
        return "   ·   ".join(bits)

    def _build_ui(self) -> None:
        # ---- 顶栏：标题 + 流程提示 + 依赖版本
        head = tk.Frame(self, background=theme.BG)
        head.pack(side="top", fill="x", padx=12, pady=(10, 0))
        tk.Label(head, text="双目标定工作台", background=theme.BG, foreground=theme.TEXT,
                 font=theme.font(14, True)).pack(side="left")
        tk.Label(head, text="① 采集　→　② 标定　→　③ 检查　→　④ 保存",
                 background=theme.BG, foreground=theme.MUTED,
                 font=theme.font(9)).pack(side="left", padx=(14, 0))
        self.lbl_dep = tk.Label(head, text=self._dep_text(), background=theme.BG,
                                foreground=theme.MUTED, font=theme.font(9))
        self.lbl_dep.pack(side="right")
        # 打包成 EXE 后没有控制台，"东西到底存到哪了"只能靠这个按钮自查
        ttk.Button(head, text="环境信息", style="Ghost.TButton", width=0,
                   command=self._show_env).pack(side="right", padx=(0, 8))
        theme.hline(self, pady=(8, 0))

        # ---- 底栏先 pack：底部状态条必须在 Notebook 抢走纵向空间之前占位，
        #      否则窗口一缩小状态条会先被挤掉
        bar = tk.Frame(self, background=theme.PANEL, bd=0, highlightthickness=1,
                       highlightbackground=theme.BORDER)
        bar.pack(side="bottom", fill="x", padx=12, pady=(6, 10))
        # 采集进度提示（还差几张）：常驻底栏，因为它是"现在该不该停手"的唯一依据
        self.lbl_short = theme.pill(bar, self.var_short)
        self.lbl_short.pack(side="left", padx=(8, 10), pady=6)
        tk.Label(bar, textvariable=self.var_status, anchor="w", background=theme.PANEL,
                 foreground=theme.MUTED, font=theme.font(9)
                 ).pack(side="left", fill="x", expand=True, padx=(4, 8))
        self.lbl_ready = tk.Label(bar, textvariable=self.var_ready, anchor="e",
                                  background=theme.PANEL, foreground=theme.MUTED,
                                  font=theme.font(9))
        self.lbl_ready.pack(side="right", padx=(6, 10))

        # ---- 四页
        self.nb = ttk.Notebook(self)
        self.nb.pack(side="top", fill="both", expand=True, padx=12, pady=(6, 0))
        self.tab_cap = ttk.Frame(self.nb, padding=(8, 10, 8, 8))
        self.tab_cal = ttk.Frame(self.nb, padding=(8, 10, 8, 8))
        self.tab_chk = ttk.Frame(self.nb, padding=(8, 10, 8, 8))
        self.tab_sav = ttk.Frame(self.nb, padding=(8, 10, 8, 8))
        self.nb.add(self.tab_cap, text="① 采集")
        self.nb.add(self.tab_cal, text="② 标定")
        self.nb.add(self.tab_chk, text="③ 检查校正效果")
        self.nb.add(self.tab_sav, text="④ 保存参数")
        self._build_capture_tab()
        self._build_calibrate_tab()
        self._build_check_tab()
        self._build_save_tab()

    # ------------------------------------------------------------ 页① 采集
    def _build_capture_tab(self) -> None:
        f = self.tab_cap
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        top = tk.Frame(f, background=theme.BG)
        top.grid(row=0, column=0, sticky="nsew")

        # 先放右侧面板再放图像区：pack 的顺序会影响可用宽度分配，这个顺序最稳
        side = tk.Frame(top, background=theme.BG, width=356)
        side.pack(side="right", fill="y", padx=(10, 0))
        side.pack_propagate(False)
        # 卡片容器单独一层：否则 grid 子控件会把 side 的请求宽度顶到 700 多，
        # 图像区被挤成一条；直接关 grid_propagate 又会冻住行高度的再分配
        holder = tk.Frame(side, background=theme.BG)
        holder.pack(fill="both", expand=True)
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(1, weight=1)       # 只有清单吃剩余高度

        imgs = tk.Frame(top, background=theme.BG)
        imgs.pack(side="left", fill="both", expand=True)
        # 图像标签必须是 tk.Label：ttk.Label 不接受 background，
        # 深色底下会露出一块系统灰底，画面边界就糊了
        box_l, self.lbl_left = theme.image_panel(imgs, "左眼 · 角点与判定叠加",
                                              "LEFT（相机未打开）")
        box_l.pack(side="left", fill="both", expand=True, padx=(0, 4))
        box_r, self.lbl_right = theme.image_panel(imgs, "右眼 · 角点与判定叠加", "RIGHT")
        box_r.pack(side="left", fill="both", expand=True, padx=(4, 0))

        # 三张卡片用 grid 排在 holder 里：pack 在空间不足时会把最后放的那张整个裁掉，
        # 而质量门恰恰是最需要常驻可见的一块（它是"现在能不能存"的唯一依据）

        # ---- 卡片一：本帧三项硬指标
        c_m, b_m = theme.card(holder, "本帧指标", "过门为绿")
        c_m.grid(row=0, column=0, sticky="ew")
        self.txt_metrics = tk.Text(b_m, height=5, wrap="word", state="disabled")
        theme.style_text(self.txt_metrics)
        self.txt_metrics.pack(fill="both", expand=True)

        # ---- 卡片二：采集清单（吃掉中段剩余高度）
        c_l, b_l = theme.card(holder, "采集清单", "姿态覆盖")
        c_l.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.txt_list = tk.Text(b_l, height=6, wrap="none", state="disabled")
        theme.style_text(self.txt_list)
        self.txt_list.pack(fill="both", expand=True)

        # ---- 卡片三：质量门（唯一允许随手改的地方）
        c_t, b_t = theme.card(holder, "质量门（可改）", "低于下限即拒绝")
        c_t.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        grid = tk.Frame(b_t, background=theme.PANEL)
        grid.pack(fill="x")
        for i, (label, var) in enumerate((("每格 px", self.var_px),
                                          ("清晰度", self.var_sharp),
                                          ("对比度", self.var_contrast),
                                          ("目标对数", self.var_target))):
            r, c = divmod(i, 2)
            tk.Label(grid, text=label, background=theme.PANEL, foreground=theme.MUTED,
                     font=theme.font(9), anchor="w"
                     ).grid(row=r, column=c * 2, sticky="w",
                            padx=(0 if c == 0 else 12, 4), pady=2)
            ttk.Entry(grid, textvariable=var, width=7
                      ).grid(row=r, column=c * 2 + 1, sticky="w", pady=2)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        # ---- 底部动作条：主操作在左，查看类靠右，流程推进单列最右
        btns = tk.Frame(f, background=theme.BG)
        btns.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.btn_open = ttk.Button(btns, text="打开相机", command=self._toggle_camera,
                                   style="Primary.TButton")
        self.btn_open.pack(side="left")
        ttk.Button(btns, text="保存这一帧", command=self._save_frame,
                   style="Primary.TButton").pack(side="left", padx=(8, 0))
        theme.keycap(btns, "空格").pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="强制保存",
                   command=lambda: self._save_frame(force=True)
                   ).pack(side="left", padx=(10, 0))
        theme.keycap(btns, "F").pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="撤销", command=self._undo
                   ).pack(side="left", padx=(10, 0))
        theme.keycap(btns, "U").pack(side="left", padx=(6, 0))

        self.btn_finish = ttk.Button(btns, text="完成采集 → 去标定",
                                     command=self._finish_capture,
                                     style="Primary.TButton")
        self.btn_finish.pack(side="right")
        ttk.Button(btns, text="覆盖热力图", command=self._show_coverage,
                   style="Ghost.TButton").pack(side="right", padx=(0, 8))
        ttk.Button(btns, text="删帧…", command=self._delete_frame_dialog,
                   style="Ghost.TButton").pack(side="right", padx=(0, 6))
        ttk.Button(btns, text="缩略图…", command=self._open_thumbnails,
                   style="Ghost.TButton").pack(side="right", padx=(0, 6))

    # ------------------------------------------------------------ 页② 标定
    def _build_calibrate_tab(self) -> None:
        f = self.tab_cal
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)
        f.rowconfigure(2, weight=1)

        # ---- 上半：左右两张参数卡（数据源 / 板与动作）
        head = tk.Frame(f, background=theme.BG)
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=3)
        head.columnconfigure(1, weight=2)

        c_src, b_src = theme.card(head, "数据源", "带对数的是可用目录")
        c_src.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        line1 = tk.Frame(b_src, background=theme.PANEL)
        line1.pack(fill="x")
        self.cmb_dir = ttk.Combobox(line1, textvariable=self.var_dir, width=34)
        self.cmb_dir.pack(side="left", fill="x", expand=True)
        ttk.Button(line1, text="刷新", command=self._refresh_dirs,
                   style="Ghost.TButton").pack(side="left", padx=(6, 0))
        ttk.Button(line1, text="浏览…", command=self._browse_dir,
                   style="Ghost.TButton").pack(side="left", padx=(6, 0))
        ttk.Checkbutton(b_src, text="固定内参（推荐：只用外参重估）",
                        variable=self.var_fix).pack(anchor="w", pady=(8, 0))

        c_cal, b_cal = theme.card(head, "标定板与动作", "板参数必须与实物一致")
        c_cal.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        line2 = tk.Frame(b_cal, background=theme.PANEL)
        line2.pack(fill="x")
        for label, var, width in (("列", self.var_cols, 5), ("行", self.var_rows, 5),
                                  ("格边长 mm", self.var_square, 7)):
            tk.Label(line2, text=label, background=theme.PANEL, foreground=theme.MUTED,
                     font=theme.font(9)).pack(side="left", padx=(0, 4))
            ttk.Entry(line2, textvariable=var, width=width).pack(side="left", padx=(0, 10))

        acts = tk.Frame(b_cal, background=theme.PANEL)
        acts.pack(fill="x", pady=(10, 0))
        self.btn_cal = ttk.Button(acts, text="开始标定", command=self._start_calibration,
                                  style="Primary.TButton")
        self.btn_cal.pack(side="left")
        # 取消：只清除"继续算"的意愿，不强杀线程。主计算是一次原生调用，打断不了，
        # 所以在阶段边界生效（详见 calib_core.check_cancel 的说明）。
        self.btn_cancel = ttk.Button(acts, text="取消", command=self._cancel_job,
                                     state="disabled", style="Danger.TButton")
        self.btn_cancel.pack(side="left", padx=(8, 0))
        self.btn_check = ttk.Button(acts, text="生成检查图", command=self._start_check,
                                    state="disabled")
        self.btn_check.pack(side="left", padx=(8, 0))

        # ---- 下半：日志与结果各占一半，都能纵向拉伸
        c_log, b_log = theme.card(f, "运行日志", "逐帧检出与被拒原因")
        c_log.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.txt_cal_log = tk.Text(b_log, height=7, wrap="none", state="disabled")
        theme.style_text(self.txt_cal_log)
        self.txt_cal_log.pack(fill="both", expand=True)

        c_res, b_res = theme.card(f, "标定结果与体检", "RMS 之外还查基线量级 / T 方向 / 尺寸一致性")
        c_res.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.txt_cal_res = tk.Text(b_res, height=8, wrap="none", state="disabled")
        theme.style_text(self.txt_cal_res)
        self.txt_cal_res.pack(fill="both", expand=True)

    # ------------------------------------------------------------ 页③ 检查
    def _build_check_tab(self) -> None:
        f = self.tab_chk
        f.columnconfigure(0, weight=1)
        f.rowconfigure(2, weight=1)

        bar = tk.Frame(f, background=theme.BG)
        bar.grid(row=0, column=0, sticky="ew")
        tk.Label(bar, text="显示方式", background=theme.BG, foreground=theme.MUTED,
                 font=theme.font(9)).pack(side="left")
        ttk.Combobox(bar, textvariable=self.var_mode, width=24, state="readonly",
                     values=list(gstate.CHECK_MODES)
                     ).pack(side="left", padx=(6, 0))
        tk.Label(bar, text="alpha", background=theme.BG, foreground=theme.MUTED,
                 font=theme.font(9)).pack(side="left", padx=(14, 4))
        ttk.Spinbox(bar, from_=0.0, to=1.0, increment=0.1, width=6,
                    textvariable=self.var_alpha).pack(side="left")
        ttk.Button(bar, text="重新生成", command=self._start_check,
                   style="Primary.TButton").pack(side="left", padx=(12, 0))
        ttk.Button(bar, text="另存图片…", command=self._save_check_image
                   ).pack(side="left", padx=(8, 0))

        # 无标定时的引导条：直接告诉人去哪一步，并允许载入磁盘上已有的标定文件。
        # 用暖色描边而不是红：这是"还差一步"，不是"出错了"
        self.frm_hint = tk.Frame(f, background=theme.PANEL, bd=0, highlightthickness=1,
                                 highlightbackground=theme.WARN)
        self.frm_hint.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        # 用 grid 而不是 pack：pack 下长文本会先占满宽度再把按钮挤到窗口外，
        # 实测提示文字直接压在了按钮上面
        self.frm_hint.columnconfigure(0, weight=1)
        self.lbl_hint = tk.Label(self.frm_hint, textvariable=self.var_chk_hint,
                                 anchor="w", justify="left", background=theme.PANEL,
                                 foreground="#f0e6d2", font=theme.font(9))
        self.lbl_hint.grid(row=0, column=0, sticky="ew", padx=(10, 8), pady=7)
        self.btn_hint_cal = ttk.Button(self.frm_hint, text="去第②页标定",
                                       command=lambda: self.nb.select(self.tab_cal),
                                       style="Ghost.TButton")
        self.btn_hint_cal.grid(row=0, column=1, sticky="e", padx=(0, 8), pady=5)
        self.btn_hint_load = ttk.Button(self.frm_hint, text="载入已有标定",
                                        command=self._load_current_calib,
                                        style="Ghost.TButton")
        self.btn_hint_load.grid(row=0, column=2, sticky="e", padx=(0, 10), pady=5)
        # 换行宽度跟着窗口走：写死会在窄窗口下让文字重新压到按钮上
        self.frm_hint.bind("<Configure>", self._on_hint_resize)

        img_outer, self.lbl_check = theme.image_panel(f, "校正结果预览",
                                                   "先完成标定，再生成检查图")
        img_outer.grid(row=2, column=0, sticky="nsew")

        c_chk, b_chk = theme.card(f, "复检数据", "极线残差、分区 |dy| 与结论")
        c_chk.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.txt_chk = tk.Text(b_chk, height=6, wrap="none", state="disabled")
        theme.style_text(self.txt_chk)
        self.txt_chk.pack(fill="x")

    # ------------------------------------------------------------ 页④ 保存
    def _build_save_tab(self) -> None:
        f = self.tab_sav
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

        c_sav, b_sav = theme.card(f, "写出标定文件",
                               "只新增 calibration/*.yaml，不改 config.yaml")
        c_sav.grid(row=0, column=0, sticky="ew")
        r1 = tk.Frame(b_sav, background=theme.PANEL)
        r1.pack(fill="x")
        tk.Label(r1, text="保存到", background=theme.PANEL, foreground=theme.MUTED,
                 font=theme.font(9)).pack(side="left")
        ttk.Entry(r1, textvariable=self.var_save_path, width=48
                  ).pack(side="left", padx=(6, 0), fill="x", expand=True)
        ttk.Button(r1, text="浏览…", command=self._browse_save,
                   style="Ghost.TButton").pack(side="left", padx=(6, 0))

        r2 = tk.Frame(b_sav, background=theme.PANEL)
        r2.pack(fill="x", pady=(8, 0))
        tk.Label(r2, text="备注", background=theme.PANEL, foreground=theme.MUTED,
                 font=theme.font(9)).pack(side="left")
        ttk.Entry(r2, textvariable=self.var_note, width=26).pack(side="left", padx=(6, 0))
        ttk.Button(r2, text="保存并归档", command=self._save_calib,
                   style="Primary.TButton").pack(side="left", padx=(10, 0))
        ttk.Button(r2, text="校验当前文件", command=self._validate_current
                   ).pack(side="left", padx=(8, 0))
        ttk.Button(r2, text="与上一份对比", command=self._diff_with_last
                   ).pack(side="left", padx=(8, 0))

        c_hist, b_hist = theme.card(f, "归档历史",
                                 "选中后可激活：自动备份当前那份，可回滚")
        c_hist.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        mid = tk.Frame(b_hist, background=theme.PANEL)
        mid.pack(fill="both", expand=True)
        mid.columnconfigure(0, weight=1)
        mid.rowconfigure(0, weight=1)
        self.lst_arch = tk.Listbox(mid, height=7)
        theme.style_listbox(self.lst_arch)
        self.lst_arch.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(mid, orient="vertical", command=self.lst_arch.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.lst_arch.configure(yscrollcommand=sb.set)
        acts = tk.Frame(mid, background=theme.PANEL)
        acts.grid(row=0, column=2, sticky="n", padx=(8, 0))
        ttk.Button(acts, text="刷新", command=self._refresh_archives,
                   style="Ghost.TButton").pack(fill="x")
        ttk.Button(acts, text="激活选中", command=self._activate_selected
                   ).pack(fill="x", pady=(6, 0))

        c_out, b_out = theme.card(f, "操作结果", "校验 / 对比的输出都在这里")
        c_out.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.txt_sav = tk.Text(b_out, height=6, wrap="none", state="disabled")
        theme.style_text(self.txt_sav)
        self.txt_sav.pack(fill="x")

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    @staticmethod
    def _fit_size(src_w: int, src_h: int, box_w: int, box_h: int) -> Tuple[int, int]:
        """等比缩放到刚好放进 box 的尺寸：可放大也可缩小，宽高比不变。"""
        if src_w <= 0 or src_h <= 0 or box_w <= 0 or box_h <= 0:
            return max(src_w, 1), max(src_h, 1)
        s = min(box_w / float(src_w), box_h / float(src_h))
        return max(int(round(src_w * s)), 1), max(int(round(src_h * s)), 1)

    def _on_image_resize(self, _evt=None) -> None:
        """面板尺寸变化：防抖 120 ms 后按新尺寸重画。

        不防抖的话，拖动窗口边缘会每一像素都重算一次缩放+PNG 编码，界面直接卡住。
        """
        if self._fit_job is not None:
            try:
                self.after_cancel(self._fit_job)
            except Exception:               # noqa: BLE001
                pass
        self._fit_job = self.after(120, self._redraw_images)

    def _redraw_images(self) -> None:
        """用保存下来的原图按当前尺寸重画三个常驻显示区。"""
        self._fit_job = None
        for key, label in (("live_l", self.lbl_left), ("live_r", self.lbl_right),
                           ("chk", self.lbl_check)):
            img = self._last_img.get(key)
            if img is None:
                continue
            # 尺寸没变就不重编码：Configure 事件会连发，重复重画既费时又会闪
            h, w = img.shape[:2]
            tw, th = self._fit_size(w, h, label.winfo_width(), label.winfo_height())
            ph = self._photos.get(key)
            if ph is not None and (ph.width(), ph.height()) == (tw, th):
                continue
            try:
                self._show(img, label, key, fit=True)
            except Exception:               # noqa: BLE001
                pass

    def _show(self, bgr: Optional[np.ndarray], label: ttk.Label, key: str,
              max_w: int = 560, max_h: int = 420, fit: bool = False) -> None:
        """把 BGR 图贴到 label 上。

        fit=True ：按 label 的当前像素尺寸等比缩放，窗口最大化后自动铺满可用区；
        fit=False：按 max_w/max_h 缩略且**不放大**（弹窗、缩略图这类固定尺寸场合）。
        """
        if bgr is None:
            return
        img = bgr
        h, w = img.shape[:2]
        if h <= 1 or w <= 1:
            return
        self._last_img[key] = img               # 原图留一份，窗口缩放时不必重新采集
        if fit:
            bw, bh = label.winfo_width(), label.winfo_height()
            if bw < 40 or bh < 40:              # 布局还没完成，退回调用方给的框
                bw, bh = max_w, max_h
            tw, th = self._fit_size(w, h, bw, bh)
        else:
            s = min(max_w / float(w), max_h / float(h), 1.0)
            tw, th = max(int(w * s), 1), max(int(h * s), 1)
        if (tw, th) != (w, h):
            interp = cv2.INTER_AREA if tw < w else cv2.INTER_LINEAR
            img = cv2.resize(img, (tw, th), interpolation=interp)
        ph = _make_photo(img)
        if ph is None:
            return
        label.configure(image=ph, text="")
        label.image = ph                        # 必须保留引用，否则图会被回收成空白
        self._photos[key] = ph

    # ------------------------------------------------------ 采集进度与"还差几张"
    def _refresh_checklist(self) -> None:
        """刷新采集清单与底栏进度提示。所有会改动采集结果的地方都必须调它。

        文本只在"记录数 / 目标数 / 各项达成数"变化时重写——相机 30 帧/秒地刷同一个
        文本控件会让滚动位置乱跳，而且纯属浪费。
        """
        if self.session is None:
            if getattr(self, "_list_sig", None) != ("empty",):
                self._set_text(self.txt_list, "还没有保存任何一帧")
                self._list_sig = ("empty",)
            self._apply_short(None)
            return

        prog = self.session.checklist_progress()
        sig = (self.session.count, self.session.target, tuple(prog.items()))
        plan = self.session.shortage()
        if sig != getattr(self, "_list_sig", None):
            lines = self.session.checklist_lines()
            lines += ["", f"进度：{plan['text']}"]
            if plan["advice"]:
                lines += ["下一步："] + [f"  · {a}" for a in plan["advice"]]
            self._set_text(self.txt_list, "\n".join(lines))
            self._list_sig = sig
        self._apply_short(plan)

    def _apply_short(self, plan: Optional[Dict[str, Any]]) -> None:
        """底栏那一格：短到一眼能读完，颜色直接表示"该继续拍 / 该去标定"。"""
        if plan is None:
            self.var_short.set("还没开始采集")
            theme.set_pill(self.lbl_short, "idle")
            return
        if plan["done"]:
            self.var_short.set("采集完成：数量与姿态都达标")
            theme.set_pill(self.lbl_short, "ok")
            return
        bits: List[str] = []
        if plan["short"] > 0:
            bits.append(f"还差 {plan['short']} 对")
        if plan["missing"]:
            bits.append(f"缺 {len(plan['missing'])} 类姿态")
        self.var_short.set(" · ".join(bits) if bits else "继续采集")
        # 数量不够时给暖色：这是唯一需要用户立刻行动的信号
        theme.set_pill(self.lbl_short, "warn" if plan["short"] > 0 else "idle")

    # ------------------------------------------------------------ 帧缩略图
    def _thumb_items(self) -> List[Dict[str, Any]]:
        """本次会话的内存记录优先；没有会话就读磁盘上的采集目录（只读）。"""
        if self.session is not None and self.session.count:
            descs = {r[0]: r[1] for r in self.session.frame_rows()}
            return [{"idx": r.idx, "left": r.left_path, "right": r.right_path,
                     "desc": descs.get(r.idx, "?"),
                     "n": int(len(r.corners)) if r.corners is not None else -1}
                    for r in self.session.records]
        d = self._current_frames_dir()
        if not d:
            return []
        out: List[Dict[str, Any]] = []
        for lp in sorted(glob.glob(os.path.join(d, "left_*.png"))):
            base = os.path.basename(lp)[5:-4]
            if not base.isdigit():
                continue
            rp = os.path.join(d, f"right_{int(base):02d}.png")
            out.append({"idx": int(base), "left": lp, "right": rp,
                        "desc": "磁盘", "n": -1})
        return out

    def _open_thumbnails(self) -> None:
        items = self._thumb_items()
        if not items:
            messagebox.showinfo("提示", "还没有可看的帧：先在①页采集，或在②页选一个采集目录")
            return
        on_del = self._delete_frame_by_idx if self.session is not None else None
        FrameGrid(self, items, on_delete=on_del, title="帧缩略图")

    def _delete_frame_by_idx(self, idx: int) -> None:
        if self.session is None:
            return
        rec = next((r for r in self.session.records if r.idx == idx), None)
        if rec is None:
            return
        ok = self.session.remove(rec)
        self._refresh_checklist()
        self.var_status.set(f"已删除第 {idx:02d} 帧" if ok else f"第 {idx:02d} 帧已不在列表里")

    # ------------------------------------------------------------------ 主循环
    def _tick(self) -> None:
        self._drain_queue()
        if self.worker is not None:
            p = self.worker.take()
            if p is not None:
                self._on_frame(*p)
            elif self.worker.error:
                self.var_ready.set(self.worker.error)
        self._tick_id = self.after(30, self._tick)

    def destroy(self) -> None:
        """先撤掉定时回调再销毁。

        不这么做的话，窗口销毁后那条 after 仍会触发，Tk 会打印
        `invalid command name "..._tick"` —— 不影响功能，但每次退出都报一行错，
        会让人以为程序出错了。
        """
        try:
            if getattr(self, "_tick_id", None):
                self.after_cancel(self._tick_id)
                self._tick_id = None
        except Exception:                   # noqa: BLE001
            pass
        super().destroy()

    def _drain_queue(self) -> None:
        while True:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                return
            except Exception:               # noqa: BLE001
                return
            if kind == "progress":
                stage, detail = payload
                self._log(self.txt_cal_log, f"[{stage}] {detail}")
                self.var_status.set(f"{stage}…")
            elif kind == "done":
                job_kind, out = payload
                self._set_busy(None)
                if job_kind == "calib":
                    self._on_calibrated(out)
                elif job_kind == "check":
                    self._on_checked(out)
            elif kind == "cancelled":
                self._set_busy(None)
                self.var_status.set("已取消（未改动任何文件）")
                target = self.txt_cal_log if payload == "calib" else self.txt_chk
                self._log(target, "[取消] 用户中止，本次结果未采用")
            elif kind == "error":
                job_kind, msg = payload
                self._set_busy(None)
                self.var_status.set(f"{job_kind} 失败")
                if job_kind == "calib":
                    self._log(self.txt_cal_log, f"[失败] {msg}")
                    messagebox.showerror("标定失败", msg)
                else:
                    self._log(self.txt_chk, f"[失败] {msg}")
                    messagebox.showerror("检查图生成失败", msg)

    def _set_busy(self, kind: Optional[str]) -> None:
        """统一管理"计算中"的按钮状态，避免取消/完成/失败三条路径各写一遍。"""
        busy = kind is not None
        self.btn_cal.configure(state="disabled" if busy else "normal")
        self.btn_cancel.configure(state="normal" if busy else "disabled")
        ok = bool(self.calib_out) and not busy
        self.btn_check.configure(state="normal" if ok else "disabled")
        if not busy:
            self.cancel_evt = None

    def _cancel_job(self) -> None:
        if self.cancel_evt is None:
            return
        self.cancel_evt.set()
        self.btn_cancel.configure(state="disabled")
        self.var_status.set("正在取消…（主计算是一整次原生调用，需等它算完，通常 1~3 秒）")

    @staticmethod
    def _log(widget: tk.Text, line: str) -> None:
        widget.configure(state="normal")
        widget.insert("end", line + "\n")
        widget.see("end")
        widget.configure(state="disabled")

    # ------------------------------------------------------------------ 相机
    def _cam_cfg(self) -> Dict[str, Any]:
        try:
            import yaml
            # config.yaml 是只读资源：打包后可能在 _MEIPASS 里，也可能被用户改在 exe 旁边，
            # 所以用 find() 按"可写根 → 只读根 → cwd"顺序找，而不是拼死路径
            with open(_find("config.yaml"), "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            return dict(cfg.get("camera", {}) or {})
        except Exception:                   # noqa: BLE001
            return {}

    def _gui_perf(self) -> Dict[str, float]:
        """读 config.yaml 的 gui 段（预览帧率上限、OpenCV 线程数）。

        单独一个函数而不是并进 _cam_cfg()：相机配置会原样交给 StereoCamera，
        多塞键会被当成未知字段；这两个参数只与界面负载有关。
        读不到就用默认值，绝不让配置问题阻断"打开相机"。
        """
        out: Dict[str, float] = {"preview_fps": 8.0, "cv_threads": 1}
        try:
            import yaml
            with open(_find("config.yaml"), "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            g = dict(cfg.get("gui", {}) or {})
            if "preview_fps" in g:
                out["preview_fps"] = float(g.get("preview_fps") or 0)
            if "cv_threads" in g:
                out["cv_threads"] = max(int(g.get("cv_threads") or 1), 1)
        except Exception:                   # noqa: BLE001
            pass
        return out

    def _toggle_camera(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.btn_open.configure(text="打开相机")
            self.var_ready.set("相机已关闭")
            return
        self.spec = core.default_spec(self.var_cols.get(), self.var_rows.get(),
                                      self.var_square.get())
        thr = {"min_px_per_square": self.var_px.get(),
               "min_sharpness": self.var_sharp.get(),
               "min_contrast": self.var_contrast.get()}
        if self.session is None:
            self.session = core.CaptureSession(self.spec, target=self.var_target.get(), thr=thr)
        perf = self._gui_perf()
        self.worker = CameraWorker(self._cam_cfg(), self.spec, thr,
                                   preview_fps=perf["preview_fps"],
                                   cv_threads=int(perf["cv_threads"]))
        self.worker.start()
        self.btn_open.configure(text="关闭相机")
        self.var_ready.set("相机正在打开…（首次打开约需 2~5 秒）")

    def _on_frame(self, pair, res, pose, reasons, fps) -> None:
        self._last_frame = (pair, res, pose, reasons, fps)
        # 左图叠左眼角点、右图叠右眼角点（两处用同一套左眼坐标会整体偏移一个视差）
        try:
            cc = core._cc()
            left_vis = cc.draw_overlay(pair.left, res, self.spec, side="left")
            right_vis = cc.draw_overlay(pair.right, res, self.spec, side="right")
        except Exception:                   # noqa: BLE001
            left_vis, right_vis = pair.left, pair.right
        self._show(left_vis, self.lbl_left, "live_l", fit=True)
        self._show(right_vis, self.lbl_right, "live_r", fit=True)

        level = res.get("level", "bad")
        tag = {"ok": "可保存", "warn": "勉强可用", "bad": "不可用"}.get(level, level)
        self.var_ready.set(f"{tag}  |  {fps:.1f} 帧/秒  |  "
                           f"每格 {res.get('px_per_square', 0):.1f} px")

        dup = False
        if pose is not None and self.session is not None:
            try:
                dup = core._cc()._pose_signature(pose) in self.session._sigs
            except Exception:               # noqa: BLE001
                dup = False
        pose_txt = ("-" if pose is None else
                    f"{pose['region']} 倾斜 {pose['tilt_deg']:.0f}° {pose['tilt_dir']}")
        lines = [
            f"角点  左 {'√' if res.get('found_l') else '×'}   "
            f"右 {'√' if res.get('found_r') else '×'}   "
            f"{'双眼可用' if (res.get('found_l') and res.get('found_r')) else '仅单眼'}",
            f"每格像素      {res.get('px_per_square', 0):.1f}",
            f"清晰度        {res.get('sharpness', 0):.0f}",
            f"对比度(最差)  {res.get('pattern_contrast_min', 0):.0f}",
            f"姿态          {pose_txt}",
            f"已保存        {self.session.count if self.session else 0}"
            f" / {self.var_target.get()}（双眼 "
            f"{self.session.stereo_count if self.session else 0}）",
        ]
        if reasons:
            lines.append("")
            lines += ["拦下：" + r for r in reasons]
        elif dup:
            lines.append("")
            lines.append("拦下：与已保存的某张姿态过于相似")
        else:
            lines.append("")
            lines.append("质检通过，可以保存（空格）")
        self._set_text(self.txt_metrics, "\n".join(lines))
        if self.session is not None:
            self._refresh_checklist()

    # ------------------------------------------------------------------ 录制
    def _save_frame(self, force: bool = False) -> None:
        if self.worker is None:
            messagebox.showinfo("提示", "先打开相机")
            return
        p = self._last_frame
        if p is None:
            self.var_status.set("还没有可用画面，等相机出一帧再试")
            return
        pair, res, pose, reasons, _ = p
        thr = {"min_px_per_square": self.var_px.get(),
               "min_sharpness": self.var_sharp.get(),
               "min_contrast": self.var_contrast.get()}
        if self.session is None:
            self.session = core.CaptureSession(self.spec, target=self.var_target.get(), thr=thr)
        else:
            self.session.thr = dict(thr)     # 让界面上改过的阈值立即生效
            self.session.target = int(self.var_target.get())
        try:
            rec = self.session.record(pair.left, pair.right, res, pose, force=force)
        except core.CaptureRejected as exc:
            self.var_status.set(f"未保存：{exc}")
            return
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("保存失败", f"{type(exc).__name__}: {exc}")
            return
        self.var_status.set(f"已保存第 {rec.idx} 帧 → {self.session.dir}")
        self._refresh_checklist()

    def _on_tab_changed(self, _evt=None) -> None:
        """切页时刷新该页依赖的数据（目录 / 标定结论 / 归档列表）。

        这些刷新原先只在启动和标定完成时各做一次，切回旧页看到的可能是几分钟前的
        快照——比如在文件管理器里手工删了几帧，①页的清单不会跟着变。
        """
        try:
            idx = self.nb.index(self.nb.select())
        except Exception:                   # noqa: BLE001
            return
        if idx == 0:
            self._refresh_checklist()
        elif idx == 1:
            self._refresh_dirs()
        elif idx == 2:
            self._update_check_hint()
        elif idx == 3:
            self._refresh_archives()

    def _typing(self) -> bool:
        """焦点是否在可输入控件里。

        根窗口绑定会连带输入框一起触发：在"备注"里打一个 f，不应该顺手强制保存
        一帧。所以单键快捷键一律先过这道闸。
        """
        w = self.focus_get()
        return isinstance(w, (tk.Entry, tk.Text, tk.Spinbox, tk.Listbox,
                              ttk.Entry, ttk.Combobox, ttk.Spinbox))

    def _on_hotkey(self, action: str, _evt=None) -> None:
        """第①页的单键快捷键：空格=保存 / F=强制保存 / U=撤销。

        只在采集页生效：其它页上按空格没有"保存一帧"的语义，误触发会污染数据。
        """
        if self._typing():
            return
        if self.nb.index(self.nb.select()) != 0:
            return
        if action == "force":
            self._save_frame(force=True)
        elif action == "undo":
            self._undo()
        else:
            self._save_frame()

    def _undo(self) -> None:
        if self.session is None:
            return
        rec = self.session.undo()
        if rec is None:
            self.var_status.set("没有可撤销的帧")
            return
        self.var_status.set(f"已撤销第 {rec.idx} 帧")
        self._refresh_checklist()

    def _delete_frame_dialog(self) -> None:
        """列出本次会话已存的帧，允许删掉任意一帧（不只是最后一帧）。"""
        if self.session is None or self.session.count == 0:
            messagebox.showinfo("提示", "本次会话还没有保存任何一帧")
            return
        rows = self.session.frame_rows()

        win = tk.Toplevel(self)
        win.title(f"删除帧（共 {len(rows)} 帧）")
        win.transient(self)
        win.geometry("440x380")
        ttk.Label(win, text="选中一帧后按「删除选中」或 Delete 键；文件会一并从磁盘删除。",
                  wraplength=410, justify="left").pack(anchor="w", padx=10, pady=(8, 4))

        box = ttk.Frame(win)
        box.pack(fill="both", expand=True, padx=10)
        lb = tk.Listbox(box, font=("Consolas", 10), activestyle="dotbox")
        lb.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(box, orient="vertical", command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.configure(yscrollcommand=sb.set)
        for idx, desc, n in rows:
            lb.insert("end", f"#{idx:02d}   {desc:<12} 角点 {n:3d}")
        if rows:
            lb.selection_set(0)

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=10, pady=8)

        def do_delete(_evt=None) -> None:
            sel = lb.curselection()
            if not sel:
                return
            idx = rows[int(sel[0])][0]
            rec = next((r for r in self.session.records if r.idx == idx), None)
            if rec is None:
                return
            if not messagebox.askyesno(
                    "确认删除",
                    f"删除第 {idx:02d} 帧？\n\n{rec.left_path}\n{rec.right_path}\n\n"
                    "这两个文件会从磁盘删除，不可撤销。",
                    parent=win):
                return
            ok = self.session.remove(rec)
            self._refresh_checklist()
            self.var_status.set(f"已删除第 {idx:02d} 帧" if ok else f"第 {idx:02d} 帧已不在列表里")
            win.destroy()

        ttk.Button(btns, text="删除选中", command=do_delete).pack(side="left")
        ttk.Button(btns, text="取消", command=win.destroy).pack(side="right")
        lb.bind("<Delete>", do_delete)
        lb.bind("<Double-Button-1>", do_delete)
        lb.focus_set()

    def _show_coverage(self) -> None:
        if self.session is None or self.session.count == 0:
            messagebox.showinfo("提示", "还没有采集任何一帧")
            return
        rec = self.session.records[-1]
        ref = cv2.imread(rec.left_path)
        if ref is None:
            return
        img = chk.coverage_overlay(ref, self.session.grid_counts())
        win = tk.Toplevel(self)
        win.title(f"覆盖热力图（{self.session.coverage_ratio() * 100:.0f}% 的格子有角点）")
        lbl = ttk.Label(win)
        lbl.pack()
        self._show(img, lbl, "cov", max_w=760, max_h=560)

    def _finish_capture(self) -> None:
        if self.session is None or self.session.count == 0:
            messagebox.showinfo("提示", "还没有采集任何一帧")
            return
        try:
            paths = self.session.finalize()
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("写报告失败", str(exc))
            return
        self.var_dir.set(self.session.dir)
        self._refresh_dirs()
        self.nb.select(self.tab_cal)
        self._log(self.txt_cal_log, f"采集完成：{self.session.count} 对，报告 {paths['txt']}")

    # ------------------------------------------------------------------ 目录
    def _refresh_dirs(self) -> None:
        items = core.list_capture_dirs()
        names = [f"{d['name']}  ({d['pairs']} 对)" for d in items]
        self.cmb_dir.configure(values=names)
        self._dirs = items
        if names and not self.var_dir.get():
            self.var_dir.set(items[0]["dir"])

    def _browse_dir(self) -> None:
        p = filedialog.askdirectory(initialdir=core.CAPTURE_ROOT)
        if p:
            self.var_dir.set(p)

    def _current_frames_dir(self) -> str:
        t = self.var_dir.get().strip()
        if os.path.isdir(t):
            return t
        # 下拉框里显示的是 "名字 (n 对)"，取回真实路径
        for d in getattr(self, "_dirs", []):
            if t.startswith(d["name"]):
                return d["dir"]
        return ""

    # ------------------------------------------------------------------ 标定
    def _start_calibration(self) -> None:
        if self.job is not None and self.job.is_alive():
            messagebox.showinfo("提示", "上一项任务还在运行")
            return
        frames_dir = self._current_frames_dir()
        if not frames_dir:
            messagebox.showwarning("提示", "请先选择一个采集目录")
            return
        self.spec = core.default_spec(self.var_cols.get(), self.var_rows.get(),
                                      self.var_square.get())
        self._set_text(self.txt_cal_log, "")
        self._set_text(self.txt_cal_res, "")
        self._log(self.txt_cal_log, f"目录：{frames_dir}")
        self._log(self.txt_cal_log, f"标定板：{self.spec.describe()}")
        self.var_status.set("标定计算中…（可点「取消」）")
        self.cancel_evt = threading.Event()
        self._set_busy("calib")
        self.job = Job("calib", self.q, core.calibrate_from_dir, frames_dir, self.spec,
                       fix_intrinsic=bool(self.var_fix.get()),
                       progress=self._progress, cancel=self.cancel_evt)
        self.job.start()

    def _progress(self, stage: str, detail: str) -> None:
        self.q.put(("progress", (stage, detail)))

    def _on_calibrated(self, out: Dict[str, Any]) -> None:
        self.calib_out = out
        calib = out["calib"]
        self._log(self.txt_cal_log, calib.summary())
        self._log(self.txt_cal_log, f"耗时 {out['elapsed_s']:.1f} s，"
                                    f"图像尺寸 {out['image_size'][0]}x{out['image_size'][1]}")
        pre = out.get("pre_dy") or []
        post = core.summarize_post(out.get("post_rows") or [])
        if pre:
            self._log(self.txt_cal_log,
                      f"校正前 |dy| 平均 {float(np.mean(pre)):.2f} px  "
                      f"最大 {float(np.max(pre)):.2f} px")
        if post["n"]:
            self._log(self.txt_cal_log,
                      f"校正后 |dy| 平均 {post['dy_mean']:.3f} px  "
                      f"最大 {post['dy_max']:.3f} px（{post['n']} 帧复检）")
        items = core.validate(calib, out["image_size"], out.get("post_rows") or [])
        self._set_text(self.txt_cal_res, core.validate_text(items))
        self.btn_check.configure(state="normal")
        self._update_check_hint()
        self.var_status.set("标定完成，可生成检查图")
        self.nb.select(self.tab_chk)

    # ------------------------------------------------------------------ 检查
    def _start_check(self) -> None:
        if self.calib_out is None:
            self._update_check_hint()
            messagebox.showinfo("还没有标定结果",
                                "这一页要在标定之后才能画。两条路：\n\n"
                                "· 去第②页做一次标定（会自动跳到本页）\n"
                                "· 若磁盘上已有 calibration/stereo_calib.yaml，"
                                "点上方「载入已有标定文件」直接看")
            return
        if not self._current_frames_dir() or not glob.glob(
                os.path.join(self._current_frames_dir(), "left_*.png")):
            self.var_status.set("请先在第①页选一个采集目录")
            messagebox.showinfo("没有可用的采集图",
                                "检查图需要一对真实的左右原图来校正。\n"
                                "请在第①页顶部的目录下拉框里选一个有 left_*.png 的会话，"
                                "或先在①页采集几对。")
            return
        if self.job is not None and self.job.is_alive():
            messagebox.showinfo("提示", "上一项任务还在运行")
            return
        calib = self.calib_out["calib"]
        frames_dir = self._current_frames_dir()
        alpha = float(self.var_alpha.get())
        spec = self.spec
        self.var_status.set("生成检查图…（可点「取消」）")
        self.cancel_evt = threading.Event()
        self._set_busy("check")
        cancel = self.cancel_evt

        def work():
            # alpha 只影响本页预览用的映射，不修改已标定结果
            use = calib
            if abs(alpha) > 1e-6:
                import copy
                use = copy.deepcopy(calib)
                R1, R2, P1, P2, _Q, _r1, _r2 = cv2.stereoRectify(
                    use.K1, use.D1, use.K2, use.D2, use.image_size,
                    use.R, use.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
                use.R1, use.R2, use.P1, use.P2 = R1, R2, P1, P2
                use._maps = None
            out = chk.render_all(use, frames_dir, spec, out_dir="", grid=(3, 4),
                                 cancel=cancel)
            out["calib_used"] = use
            return out

        self.job = Job("check", self.q, work)
        self.job.start()

    def _update_check_hint(self) -> None:
        """按当前状态刷新③页顶部的引导条：有标定就收起，没标定就说清去哪一步。"""
        if self.calib_out is not None:
            calib = self.calib_out.get("calib")
            src = self.calib_out.get("loaded_from") or "本次标定结果"
            try:
                info = (f"当前用：{os.path.basename(str(src))} · "
                        f"基线 {calib.baseline_mm:.3f} mm · "
                        f"rect fx {calib.rect_fx:.2f} px")
                if getattr(calib, "rms_stereo", 0):
                    info += f" · RMS {calib.rms_stereo:.3f} px"
            except Exception:              # noqa: BLE001
                info = f"当前用：{src}"
            self.var_chk_hint.set(info)
            self.btn_hint_cal.configure(state="disabled")
            self.btn_hint_load.configure(state="disabled")
            self.frm_hint.grid()
            self._hint_tone("ok")
            return

        exists = os.path.exists(manage.CURRENT)
        if exists:
            self.var_chk_hint.set(
                "还没有载入标定结果。磁盘上已有一份 "
                f"{os.path.basename(manage.CURRENT)}，可直接载入查看校正效果；"
                "也可以回第②页重新标定。")
            self.btn_hint_load.configure(state="normal")
            self.btn_hint_cal.configure(state="normal")
        else:
            self.var_chk_hint.set(
                "还没有任何标定结果。请先去第②页选采集目录并执行标定"
                "（没有标定文件时本页无法生成校正图）。")
            self.btn_hint_load.configure(state="disabled")
            self.btn_hint_cal.configure(state="normal")
        self.frm_hint.grid()
        self._hint_tone("warn")

    def _hint_tone(self, kind: str) -> None:
        """③页引导条的两种语义：ok=已经载入标定（绿），warn=还差一步（暖黄）。

        只用底色与 1px 描边表达，不改文字——文字负责说"做什么"，颜色负责说"急不急"。
        """
        bg, border = ((theme.PILL_OK_BG, theme.OK) if kind == "ok"
                      else (theme.PILL_WARN_BG, theme.WARN))
        self.frm_hint.configure(background=bg, highlightbackground=border)
        self.lbl_hint.configure(background=bg)

    def _on_hint_resize(self, evt) -> None:
        """引导条变宽就重算换行宽度：预留两个按钮占的横向空间（约 330 px）。"""
        self.lbl_hint.configure(wraplength=max(200, evt.width - 330))

    def _load_current_calib(self) -> None:
        """载入磁盘上已有的 stereo_calib.yaml，跳过重新标定直接看校正效果。"""
        path = manage.CURRENT
        if not os.path.exists(path):
            messagebox.showinfo("没有标定文件", f"未找到：\n{path}")
            return
        try:
            calib = core._cal().StereoCalibration.load(path)
        except Exception as exc:           # noqa: BLE001
            messagebox.showerror("载入失败", f"{type(exc).__name__}: {exc}")
            return
        self.calib_out = {"calib": calib, "loaded_from": path,
                          "image_size": tuple(calib.image_size), "post_rows": []}
        self.btn_check.configure(state="normal")
        self._log(self.txt_chk, f"已载入 {path}\n{calib.summary()}")
        self._update_check_hint()
        self.var_status.set("已载入标定文件，正在生成检查图…")
        self._start_check()

    def _on_checked(self, out: Dict[str, Any]) -> None:
        self.check_out = out
        self._render_check()
        stats = out["stats"]
        self._set_text(self.txt_chk,
                       chk.region_stats_text(stats) + "\n\n" + chk.alpha_text(out["alpha"]))
        self.var_status.set(f"检查图已生成（用第 {out['frame_index']} 帧）")

    def _render_check(self) -> None:
        if self.check_out is None:
            return
        imgs = self.check_out["images"]
        key = self.var_mode.get()
        pick = {
            "上下叠放 + 水平线": "stack",
            "并排 + 参考线": "sbs",
            "红青立体": "anaglyph",
            "左右差值": "diff",
            "棋盘混合": "checker",
            "覆盖热力图": "coverage",
            "分区 |dy| 热力图": "region",
        }.get(key, "stack")
        self._show(imgs.get(pick), self.lbl_check, "chk", fit=True)

    def _save_check_image(self) -> None:
        if self.check_out is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                        filetypes=[("PNG", "*.png")])
        if not p:
            return
        key = self.var_mode.get()
        pick = {"上下叠放 + 水平线": "stack", "并排 + 参考线": "sbs", "红青立体": "anaglyph",
                "左右差值": "diff", "棋盘混合": "checker", "覆盖热力图": "coverage",
                "分区 |dy| 热力图": "region"}.get(key, "stack")
        img = self.check_out["images"].get(pick)
        if img is not None and cv2.imwrite(p, img):
            self.var_status.set(f"已保存 {p}")

    # ------------------------------------------------------------------ 保存
    def _browse_save(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".yaml",
                                        initialdir=manage.CALIB_DIR,
                                        filetypes=[("YAML", "*.yaml")])
        if p:
            self.var_save_path.set(p)

    def _save_calib(self) -> None:
        if self.calib_out is None:
            messagebox.showinfo("提示", "还没有可保存的标定结果")
            return
        calib = self.calib_out["calib"]
        path = self.var_save_path.get().strip() or manage.CURRENT
        note = self.var_note.get().strip()
        lines: List[str] = []
        try:
            if os.path.abspath(path) == os.path.abspath(manage.CURRENT) and os.path.exists(path):
                b = manage.archive(path, note=note or "保存新标定前自动备份")
                lines.append(f"已备份旧文件：{b}")
            calib.save(path)
            lines.append(f"已保存：{path}")
            if os.path.abspath(path) == os.path.abspath(manage.CURRENT):
                b2 = manage.archive(path, note=note or "本次保存的标定")
                lines.append(f"已归档：{b2}")
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("保存失败", f"{type(exc).__name__}: {exc}")
            return
        lines += ["", core.validate_text(core.validate(
            calib, calib.image_size, (self.calib_out or {}).get("post_rows") or []))]
        self._set_text(self.txt_sav, "\n".join(lines))
        self._refresh_archives()
        self.var_status.set("标定文件已保存")

    def _refresh_archives(self) -> None:
        self.lst_arch.delete(0, "end")
        rows = manage.list_archives()
        self._arch_rows = rows
        for r in rows:
            rms = r.get("rms_stereo")
            rms_s = f"{rms:.4f}" if isinstance(rms, (int, float)) else "-"
            base = r.get("baseline_mm")
            base_s = f"{base:.3f}" if isinstance(base, (int, float)) else "-"
            self.lst_arch.insert("end", f"{r.get('file', '-')}  {base_s} mm  RMS {rms_s}  "
                                        f"{r.get('archived_at', '')}  {r.get('note', '')}")

    def _activate_selected(self) -> None:
        sel = self.lst_arch.curselection()
        if not sel:
            messagebox.showinfo("提示", "先选中一条归档")
            return
        rows = getattr(self, "_arch_rows", [])
        if not (0 <= sel[0] < len(rows)):
            return
        target = rows[sel[0]].get("path", "")
        if not os.path.exists(target):
            messagebox.showerror("错误", f"归档文件不存在：{target}")
            return
        if not messagebox.askyesno("确认", "把这份归档设为当前生效标定？\n"
                                           "当前文件会先自动备份，可随时退回。"):
            return
        try:
            res = manage.activate(target, self.var_note.get().strip())
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("激活失败", str(exc))
            return
        msg = [f"已激活：{res['activated']}", f"当前文件：{res['current']}"]
        if res["backup"]:
            msg.append(f"原文件备份：{res['backup']}")
        msg.append("")
        msg.append("提示：运行中的程序需要调用 SDK 的 reload_calibration()，"
                   "或重启程序，才会用上新的标定参数。")
        self._set_text(self.txt_sav, "\n".join(msg))
        self._refresh_archives()

    def _validate_current(self) -> None:
        path = self.var_save_path.get().strip() or manage.CURRENT
        self._set_text(self.txt_sav, f"校验：{path}\n\n" +
                       manage.validate_text(manage.validate_file(path)))

    def _diff_with_last(self) -> None:
        rows = getattr(self, "_arch_rows", None)
        if not rows:
            rows = manage.list_archives()
        if not rows:
            messagebox.showinfo("提示", "还没有归档可比")
            return
        b = rows[-1].get("path", "")
        a = self.var_save_path.get().strip() or manage.CURRENT
        if not os.path.exists(a) or not os.path.exists(b):
            messagebox.showinfo("提示", "文件不存在，无法对比")
            return
        try:
            text = manage.diff_text(a, b)
        except Exception as exc:            # noqa: BLE001
            messagebox.showerror("对比失败", str(exc))
            return
        self._set_text(self.txt_sav, f"A = {a}\nB = {b}\n\n{text}")

    # ------------------------------------------------------------------ 关闭
    @staticmethod
    def _safe(var, default):
        """从 Tk 变量取值：输入框里是空或非法内容时返回默认值，绝不让退出流程崩掉。"""
        try:
            v = var.get()
            return default if v is None or v == "" else v
        except Exception:                   # noqa: BLE001 —— TclError 等
            return default

    def _save_state(self) -> bool:
        """把窗口尺寸、当前页、上次目录、阈值、标定板参数写进记忆文件。"""
        try:
            geo = self.geometry()
        except Exception:                   # noqa: BLE001
            geo = ""
        state = {
            "geometry": geo,
            "tab": int(self.nb.index(self.nb.select())),
            "dir": str(self._safe(self.var_dir, "") or ""),
            "thresholds": {
                "px": self._safe(self.var_px, 8.0),
                "sharp": self._safe(self.var_sharp, 100.0),
                "contrast": self._safe(self.var_contrast, 120.0),
                "target": self._safe(self.var_target, 26),
            },
            "board": {
                "cols": self._safe(self.var_cols, 12),
                "rows": self._safe(self.var_rows, 9),
                "square": self._safe(self.var_square, 3.0),
            },
            "check_mode": self.var_mode.get(),
            "alpha": self._safe(self.var_alpha, 0.0),
            "fix_intrinsic": bool(self.var_fix.get()),
        }
        return gstate.save(state)

    def _on_close(self) -> None:
        # 先记配置：即使后面写报告弹窗被取消，窗口尺寸也已经存下来了
        ok = False
        try:
            ok = self._save_state()
        except Exception:                   # noqa: BLE001
            pass
        if self.cancel_evt is not None:
            self.cancel_evt.set()           # 让后台计算尽早退出，避免拖住进程
        if self.worker is not None:
            self.worker.stop()
        if self.session is not None and self.session.count and not self.session._finalized:
            if messagebox.askyesno("退出", "本次采集还没写报告，现在写吗？"):
                try:
                    self.session.finalize()
                except Exception:           # noqa: BLE001
                    pass
        if not ok:
            # 记忆写不进去不影响使用，但要让人知道下次不会保持这次的布局
            print(f"[提示] 界面记忆未能写入：{gstate.state_path()}", file=sys.stderr)
        self.destroy()


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
def _enable_dpi_awareness() -> float:
    """让 Windows 缩放 125%/150% 下界面不发虚。

    返回系统缩放比（1.0 = 100%）。非 Windows 或无权限时安全返回 1.0。
    必须在创建 Tk 根窗口**之前**调用，否则 Windows 已经按位图拉伸渲染，改不回来。
    """
    if sys.platform != "win32":
        return 1.0
    import ctypes
    scale = 1.0
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2（Win8.1+）；失败再退回 system aware
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:                  # noqa: BLE001
            ctypes.windll.user32.SetProcessDPIAware()
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)   # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        if dpi:
            scale = float(dpi) / 96.0
    except Exception:                      # noqa: BLE001
        scale = 1.0
    return scale


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="双目标定 GUI（采集/标定/检查/保存）")
    ap.add_argument("--dir", default="", help="预填采集目录")
    ap.add_argument("--selftest", action="store_true", help="只建界面，300ms 后自动退出")
    args = ap.parse_args(argv)

    scale = _enable_dpi_awareness()
    # 缩放比交给 CalibApp 在 __init__ 里应用：必须在建立任何控件之前生效，
    # 否则控件先按 100% 排版，之后再改 scaling 会出现字号与间距不同步
    global _DPI_SCALE
    _DPI_SCALE = scale
    app = CalibApp(start_dir=args.dir)
    if args.selftest:
        app.after(300, app.destroy)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
