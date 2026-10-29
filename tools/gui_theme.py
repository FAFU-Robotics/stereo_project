"""gui_theme.py —— 科技黑主题与可复用界面构件（纯 tkinter，零新依赖）
=============================================================================
主界面（viewer_shell）和标定工作台共用这一份调色板：近黑底、冷青强调、
细描边。改颜色只改这里，两处窗口会一起变。

【为什么单独一个文件】
    颜色和字体原本散落在 calib_gui.py 各处（"#111111"、"#2a2a2a"、"#1f3320"…），
    同一个"警告黄"在不同页深浅不一，看着像几个不同的人拼出来的。这里集中定义：
    调色板 → ttk 样式 → 复用构件，三样东西只在这一个文件里改。

【技术前提：必须切到 clam 主题】
    Windows 默认的 vista/xpnative 主题由系统绘制，很多颜色选项会被直接忽略
    （给 TButton 设 background 无效、Combobox 的 fieldbackground 改不动）。
    clam 是纯 Tk 自绘主题，颜色字段全部生效，是 ttk 做深色界面的唯一可行基础。
    代价：clam 的控件边缘比原生更"方"，用 bordercolor + flat relief 收敛即可。

【tk 与 ttk 的分工（踩过坑）】
    ttk 控件走 Style；tk 原生控件（Text / Listbox / Label / Frame）不吃 ttk.Style，
    必须用 option_add 写选项数据库或逐个 configure。两者都做，否则会出现
    "按钮是深色、文本框是白色"的割裂感。
=============================================================================
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# 调色板：科技黑。近黑底，强调色只用青，语义色只表达合格 / 注意 / 不合格。
# -----------------------------------------------------------------------------
BG = "#06080c"          # 窗口底色
TOP = "#080c12"         # 顶栏、底栏
PANEL = "#0c121a"       # 侧栏 / 卡片 / 输入框
PANEL_HI = "#15202c"    # 次级按钮、悬停
CARD = "#090e14"        # 略凹一层（缩略图格子）
BORDER = "#1c2c3c"      # 1px 描边，带一点青
SEAM = "#0e3c4a"        # 面板之间的冷色接缝
TEXT = "#e7f3f8"        # 主文字（冷白）
MUTED = "#7e8d9c"       # 次要文字
ACCENT = "#1ec8e6"      # 主强调青
ACCENT_DK = "#0e90aa"   # 按下 / 深一档青
ON_ACCENT = "#04141c"   # 青底上的字。青太亮，白字会发灰
HOVER = "#1a2838"
SELECT = "#0c4252"      # 列表、文本的选中底
IMG_BG = "#030508"      # 图像井（比窗口更黑，画面边界自己浮出来）

# 语义色
OK = "#3ee0a2"
WARN = "#e6b450"
BAD = "#ff5d6c"

# 底栏进度格：深底浅字，避免整条底栏跳色
PILL_IDLE_BG, PILL_IDLE_FG = "#121a24", MUTED
PILL_WARN_BG, PILL_WARN_FG = "#2c2412", "#f3e6c4"
PILL_OK_BG, PILL_OK_FG = "#10281e", "#b7f3d6"

#: 字体族名。None 表示探测失败，交给 Tk 默认（中文会落到系统中文字体）
FONT_UI: str = ""
FONT_MONO: str = ""

#: 尺寸（像素）。DPI 缩放由 calib_gui 的 tk scaling 统一放大，这里不乘系数
PAD = 8
GAP = 6

_CJK_CANDIDATES = ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
                   "Noto Sans CJK SC", "Source Han Sans SC", "SimHei")
_MONO_CANDIDATES = ("Cascadia Mono", "Consolas", "JetBrains Mono",
                    "DejaVu Sans Mono", "Courier New")


def _pick(candidates: Tuple[str, ...], fallback: str = "") -> str:
    """挑第一个本机真实存在的字体族。

    为什么要挑：直接写死 "Microsoft YaHei UI"，在 Linux 上会落到无中文字形的
    字体，整个界面变成一排方框；而 Windows 上换机器也可能缺某个字体。
    """
    try:
        have = set(tkfont.families())
    except Exception:                       # noqa: BLE001
        return fallback
    for name in candidates:
        if name in have:
            return name
    return fallback


def font(size: int = 10, bold: bool = False, mono: bool = False) -> Tuple:
    """返回可直接传给 tk/ttk 控件的字体元组。"""
    family = (FONT_MONO or "Courier New") if mono else (FONT_UI or "TkDefaultFont")
    weight = "bold" if bold else "normal"
    return (family, size, weight)


# -----------------------------------------------------------------------------
# ttk 样式
# -----------------------------------------------------------------------------
def _style(root: tk.Misc) -> ttk.Style:
    st = ttk.Style(root)
    try:
        st.theme_use("clam")               # 颜色选项只有 clam 会真正生效
    except tk.TclError:                    # pragma: no cover
        pass
    return st


def _apply_ttk(st: ttk.Style) -> None:
    # ---- 容器
    st.configure(".", background=BG, foreground=TEXT, font=font(10),
                 bordercolor=BORDER, darkcolor=PANEL, lightcolor=PANEL,
                 troughcolor=PANEL, focuscolor=BG)
    st.configure("TFrame", background=BG)
    st.configure("TLabel", background=BG, foreground=TEXT)
    st.configure("Muted.TLabel", background=BG, foreground=MUTED)
    st.configure("H1.TLabel", background=BG, foreground=TEXT, font=font(13, True))
    st.configure("H2.TLabel", background=BG, foreground=MUTED, font=font(10, True))

    # ---- 分组框：1px 描边 + 灰标题，不用系统凸起边框
    st.configure("TLabelframe", background=BG, bordercolor=BORDER,
                 borderwidth=1, relief="solid")
    st.configure("TLabelframe.Label", background=BG, foreground=MUTED,
                 font=font(10, True))

    # ---- 页签
    # 选中页靠两层信号：底比未选中亮一档，字从灰变成青。
    # 只靠亮度差不够（深色里相邻灰阶几乎分不出来）。
    st.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(2, 8, 2, 0),
                 bordercolor=BG, lightcolor=BG, darkcolor=BG)
    st.configure("TNotebook.Tab", background=PANEL, foreground=MUTED,
                 padding=(18, 8), borderwidth=0, font=font(10),
                 bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL,
                 focuscolor=BG)
    st.map("TNotebook.Tab",
           background=[("selected", "#142433"), ("active", PANEL_HI)],
           foreground=[("selected", ACCENT), ("active", TEXT)],
           # clam 默认给选中页签做"放大"，会让整排页签抖动，这里抹平
           expand=[("selected", (0, 0, 0, 0))])

    # ---- 按钮：三档变体，都靠 bordercolor 表达层级
    # width=0 不可省：clam 给 TButton 带了字符数默认宽度，中文按钮会被按字符截断
    # （实测"保存并归档"显示成"存并归档"、"删帧…"显示成"删帧..."）
    base = dict(background=PANEL, foreground=TEXT, bordercolor=BORDER,
                lightcolor=PANEL, darkcolor=PANEL,
                focuscolor=BG, relief="flat", padding=(12, 6), width=0,
                font=font(10), anchor="center")
    st.configure("TButton", **base)
    st.map("TButton",
           background=[("disabled", "#10161e"), ("pressed", "#1c2838"),
                       ("active", HOVER)],
           foreground=[("disabled", "#5c6a78")],
           bordercolor=[("active", ACCENT), ("disabled", "#1a222c")])

    st.configure("Primary.TButton", **{**base, "background": ACCENT,
                                       "foreground": ON_ACCENT,
                                       "bordercolor": ACCENT,
                                       "lightcolor": ACCENT, "darkcolor": ACCENT,
                                       "font": font(10, True)})
    st.map("Primary.TButton",
           background=[("disabled", "#0d4e60"), ("pressed", ACCENT_DK),
                       ("active", "#4ad6ee")],
           foreground=[("disabled", "#7aafbc"), ("pressed", ON_ACCENT),
                       ("active", ON_ACCENT)],
           bordercolor=[("disabled", "#0d4e60"), ("active", "#4ad6ee"),
                        ("pressed", ACCENT_DK)])

    st.configure("Danger.TButton", **base)
    st.map("Danger.TButton",
           bordercolor=[("active", BAD), ("disabled", "#1a222c")],
           foreground=[("active", "#ffc2c8")],
           background=[("active", "#2a1418")])

    st.configure("Ghost.TButton", **{**base, "background": BG, "bordercolor": BG})
    st.map("Ghost.TButton",
           background=[("active", PANEL_HI)],
           bordercolor=[("active", BORDER)])

    # ---- 输入类
    entry = dict(fieldbackground=PANEL, background=PANEL, foreground=TEXT,
                 bordercolor=BORDER, lightcolor=PANEL, darkcolor=PANEL,
                 insertcolor=TEXT, padding=(7, 5), relief="flat")
    st.configure("TEntry", **entry)
    st.map("TEntry", bordercolor=[("focus", ACCENT)])
    st.configure("TSpinbox", **entry, arrowcolor=MUTED)
    st.map("TSpinbox", bordercolor=[("focus", ACCENT)])

    st.configure("TCombobox", **{**entry, "arrowcolor": MUTED})
    st.map("TCombobox",
           bordercolor=[("focus", ACCENT), ("active", ACCENT)],
           # readonly 状态下 clam 会把底色刷回浅色，必须显式压住
           fieldbackground=[("readonly", PANEL), ("disabled", "#10161e")],
           selectbackground=[("readonly", PANEL)],
           selectforeground=[("readonly", TEXT)])

    st.configure("TCheckbutton", background=BG, foreground=TEXT,
                 indicatorbackground=PANEL_HI, indicatorforeground=ACCENT,
                 bordercolor=BORDER, focusthickness=0, padding=(2, 3))
    st.map("TCheckbutton",
           background=[("active", BG)],
           indicatorbackground=[("selected", ACCENT), ("active", HOVER)],
           indicatorforeground=[("selected", ON_ACCENT)])
    # 卡片 / 侧栏底色是 PANEL，勾选框若仍用窗口底色 BG，会露出一块更深的补丁
    st.configure("Panel.TCheckbutton", background=PANEL, foreground=TEXT,
                 indicatorbackground=PANEL_HI, indicatorforeground=ACCENT,
                 bordercolor=BORDER, focusthickness=0, padding=(2, 3))
    st.map("Panel.TCheckbutton",
           background=[("active", PANEL)],
           indicatorbackground=[("selected", ACCENT), ("active", HOVER)],
           indicatorforeground=[("selected", ON_ACCENT)])

    # ---- 滚动条：细、无箭头高光，融入深色底
    for orient, name in (("vertical", "Vertical.TScrollbar"),
                         ("horizontal", "Horizontal.TScrollbar")):
        st.configure(name, background=PANEL_HI, troughcolor=BG,
                     bordercolor=BG, arrowcolor=MUTED, relief="flat",
                     width=11, arrowsize=11)
        st.map(name, background=[("active", "#243448"), ("pressed", ACCENT)])

    st.configure("TProgressbar", background=ACCENT, troughcolor=PANEL,
                 bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
    st.configure("TSeparator", background=BORDER)


def _apply_tk_options(root: tk.Misc) -> None:
    """tk 原生控件不读 ttk.Style，用选项数据库统一压成深色。

    匹配的是**类名**（Text / Listbox / Frame），所以对所有同名控件生效，
    包括运行期才创建的对话框；个别需要特殊底色的（如图像区）再单独 configure。
    """
    db = root.option_add
    for cls in ("Text", "Listbox"):
        db(f"*{cls}.background", PANEL)
        db(f"*{cls}.foreground", TEXT)
        db(f"*{cls}.relief", "flat")
        db(f"*{cls}.borderwidth", 0)
        db(f"*{cls}.highlightthickness", 1)
        db(f"*{cls}.highlightbackground", BORDER)
        db(f"*{cls}.highlightcolor", ACCENT)
        db(f"*{cls}.selectbackground", SELECT)
        db(f"*{cls}.selectforeground", TEXT)
    db("*Text.insertBackground", TEXT)
    db("*Listbox.activestyle", "none")
    # 下拉列表是独立弹出的 Toplevel，不受 TCombobox 样式管辖
    db("*TCombobox*Listbox.background", PANEL)
    db("*TCombobox*Listbox.foreground", TEXT)
    db("*TCombobox*Listbox.selectBackground", ACCENT)
    db("*TCombobox*Listbox.selectForeground", ON_ACCENT)
    # 消息框 / 文件对话框的底色
    db("*Menu.background", PANEL)
    db("*Menu.foreground", TEXT)
    db("*Menu.activeBackground", ACCENT)
    db("*Menu.activeForeground", ON_ACCENT)


def apply(root: tk.Misc) -> ttk.Style:
    """在 Tk 根窗口创建后、搭建控件前调用一次。"""
    global FONT_UI, FONT_MONO
    FONT_UI = _pick(_CJK_CANDIDATES, "")
    FONT_MONO = _pick(_MONO_CANDIDATES, "")
    st = _style(root)
    _apply_tk_options(root)
    _apply_ttk(st)
    try:
        root.configure(background=BG)
    except Exception:                       # noqa: BLE001
        pass
    return st


# -----------------------------------------------------------------------------
# 复用构件
# -----------------------------------------------------------------------------
def brand_mark(parent: tk.Misc, bg: str = "") -> tk.Canvas:
    """两个交叠的青圈。主界面和标定工作台顶栏用同一个标记。"""
    bg = bg or TOP
    c = tk.Canvas(parent, width=26, height=16, background=bg,
                  highlightthickness=0, bd=0)
    c.create_oval(1, 1, 15, 15, outline=ACCENT, width=2)
    c.create_oval(10, 1, 24, 15, outline=ACCENT, width=2)
    return c


def accent_tick(parent: tk.Misc) -> tk.Frame:
    """标题左侧 3px 青条。空 Frame 要关掉尺寸传播，否则高度会被压成 0。"""
    tick = tk.Frame(parent, background=ACCENT, width=3, height=12,
                    bd=0, highlightthickness=0)
    tick.pack_propagate(False)
    tick.pack(side="left", padx=(0, 8))
    return tick


def hline(parent: tk.Misc, pady: int = 6) -> tk.Frame:
    """1px 水平分隔线（ttk.Separator 在深色下会带阴影，用 Frame 更干净）。"""
    line = tk.Frame(parent, height=1, background=BORDER, bd=0, highlightthickness=0)
    line.pack(fill="x", pady=pady)
    return line


def card(parent: tk.Misc, title: str, subtitle: str = "",
         pad: int = 10) -> Tuple[tk.Frame, tk.Frame]:
    """带标题的卡片容器，返回 (外框, 内容框)。

    用法：
        outer, body = card(parent, "本帧指标", "数值越低越该怀疑")
        ttk.Label(body, text="...").pack(anchor="w")

    tk.Frame + 内嵌 tk.Label 实现：ttk.Labelframe 的标题会压在边框上、
    深色下边框还会出现双线，控制不住。卡片是"底色差 + 1px 描边"，更接近目标观感。
    """
    outer = tk.Frame(parent, background=PANEL, bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=BORDER)
    head = tk.Frame(outer, background=PANEL, bd=0, highlightthickness=0)
    head.pack(fill="x", padx=pad, pady=(pad - 2, 0))
    accent_tick(head)
    tk.Label(head, text=title, background=PANEL, foreground=TEXT,
             font=font(10, True), anchor="w").pack(side="left")
    if subtitle:
        tk.Label(head, text=subtitle, background=PANEL, foreground=MUTED,
                 font=font(9), anchor="e").pack(side="right")
    body = tk.Frame(outer, background=PANEL, bd=0, highlightthickness=0)
    body.pack(fill="both", expand=True, padx=pad, pady=(6, pad))
    return outer, body


def image_panel(parent: tk.Misc, title: str, text: str) -> Tuple[tk.Frame, tk.Label]:
    """图像显示区：上方一行标题（左标题 + 右侧空位），下方是贴满的 tk.Label。

    返回 (外框, 图像 Label)。图像 Label 必须由调用方持有引用，否则 PhotoImage
    被垃圾回收后画面会变白。
    """
    outer = tk.Frame(parent, background=BG, bd=0, highlightthickness=0)
    head = tk.Frame(outer, background=BG, bd=0, highlightthickness=0)
    head.pack(fill="x", pady=(0, 4))
    accent_tick(head)
    tk.Label(head, text=title, background=BG, foreground=TEXT,
             font=font(9, True), anchor="w").pack(side="left")
    frame = tk.Frame(outer, background=IMG_BG, bd=0, highlightthickness=1,
                     highlightbackground=SEAM)
    frame.pack(fill="both", expand=True)
    lbl = tk.Label(frame, text=text, anchor="center", background=IMG_BG,
                   foreground=MUTED, bd=0, highlightthickness=0)
    lbl.pack(fill="both", expand=True)
    # 关键：切断"图片尺寸 → 面板请求尺寸"的传播。
    # tk.Label 的请求尺寸等于它显示的图片尺寸。图片一旦按面板放大，面板请求尺寸也跟着
    # 变大，pack/grid 又按新请求重新分配空间 —— 于是左面板越撑越宽、右面板被挤扁，
    # 而且窗口缩小时左面板不肯缩回去（实测窗口缩到 1100 时 639 vs 39）。
    # 关掉传播后，图片只影响绘制，不再参与布局计算。
    frame.pack_propagate(False)
    return outer, lbl


def pill(parent: tk.Misc, textvar: tk.StringVar, width: int = 0) -> tk.Label:
    """底栏的状态胶囊：深底浅字，由调用方按语义改 background/foreground。"""
    lbl = tk.Label(parent, textvariable=textvar, anchor="w", padx=10, pady=3,
                   background=PILL_IDLE_BG, foreground=PILL_IDLE_FG,
                   font=font(9, True), bd=0, highlightthickness=1,
                   highlightbackground=BORDER, highlightcolor=BORDER)
    if width:
        lbl.configure(width=width)
    return lbl


def set_pill(lbl: tk.Label, kind: str) -> None:
    """按语义给胶囊上色：idle / warn / ok / bad。"""
    table = {"idle": (PILL_IDLE_BG, PILL_IDLE_FG),
             "warn": (PILL_WARN_BG, PILL_WARN_FG),
             "ok": (PILL_OK_BG, PILL_OK_FG),
             "bad": ("#2a1218", "#ffc2c8")}
    bg, fg = table.get(kind, table["idle"])
    lbl.configure(background=bg, foreground=fg)


def keycap(parent: tk.Misc, keys: str) -> tk.Label:
    """快捷键小胶囊，青字深底，和主按钮拉开层级。"""
    return tk.Label(parent, text=keys, background=PANEL_HI, foreground=ACCENT,
                    font=font(8, True), padx=6, pady=1, bd=0,
                    highlightthickness=1, highlightbackground=BORDER,
                    highlightcolor=BORDER)


def style_text(widget: tk.Text, mono: bool = True, size: int = 9) -> None:
    """把 tk.Text 调成深色等宽控制台观感（选项数据库给了底色，这里补内边距）。"""
    widget.configure(font=font(size, mono=mono), padx=8, pady=6,
                     spacing1=1, spacing3=1, wrap=widget.cget("wrap"))


def style_listbox(widget: tk.Listbox) -> None:
    widget.configure(font=font(9, mono=True), bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT,
                     activestyle="none", selectbackground=SELECT,
                     selectforeground=TEXT)
