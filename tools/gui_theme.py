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

import ctypes
import locale
import os
import subprocess
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

#: 界面实际使用的字体族。空字符串表示还没探测；不要把 "TkDefaultFont" 填进来。
FONT_UI: str = ""
FONT_MONO: str = ""

#: 尺寸（像素）。DPI 缩放由 calib_gui 的 tk scaling 统一放大，这里不乘系数
PAD = 8
GAP = 6

# 随包带的单文件字体。族名必须和字体内部 name 表一致，Tk 才能按这个名字点到它。
BUNDLED_FAMILY = "Noto Sans SC"
_BUNDLED_FILES = ("NotoSansSC-Regular.otf", "NotoSansSC-Bold.otf")

# 顺序就是优先级。Windows / macOS 用系统字体（GDI、CoreText 能读 TTC）。
# Linux 上若随包字体已经注册成功，会优先用它，不走这份列表。
_CJK_CANDIDATES = (
    "Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑",
    "Microsoft JhengHei UI",
    "PingFang SC", "苹方-简",
    "Hiragino Sans GB", "冬青黑体简体中文",
    "Heiti SC", "黑体-简", "Songti SC",
    "Noto Sans SC", "Noto Sans CJK SC",
    "Source Han Sans SC", "Source Han Sans CN", "思源黑体",
    "WenQuanYi Micro Hei", "文泉驿微米黑",
    "WenQuanYi Zen Hei", "文泉驿正黑",
    "Droid Sans Fallback",
    "AR PL UMing CN", "AR PL UKai CN",
    "SimHei", "黑体", "SimSun", "宋体", "NSimSun", "新宋体",
    "DengXian", "等线",
)
# 等宽字体里只有带中日韩字形的才能用来画中文。Consolas / DejaVu Sans Mono
# 没有这些字，指定它们之后缺字不会再回退，中文会变成方框。
_MONO_CJK_CANDIDATES = (
    "Sarasa Mono SC", "Sarasa Gothic SC",
    "Noto Sans Mono CJK SC", "Source Han Mono SC",
)
_NAMED_FONTS = (
    "TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont",
    "TkHeadingFont", "TkCaptionFont", "TkTooltipFont",
)
_WIN_FILES = {
    "microsoftyaheiui": r"C:\Windows\Fonts\msyh.ttc",
    "microsoftyahei": r"C:\Windows\Fonts\msyh.ttc",
    "microsoftjhengheiui": r"C:\Windows\Fonts\msjh.ttc",
    "simhei": r"C:\Windows\Fonts\simhei.ttf",
    "simsun": r"C:\Windows\Fonts\simsun.ttc",
    "nsimsun": r"C:\Windows\Fonts\simsun.ttc",
    "dengxian": r"C:\Windows\Fonts\Deng.ttf",
    "宋体": r"C:\Windows\Fonts\simsun.ttc",
    "黑体": r"C:\Windows\Fonts\simhei.ttf",
    "等线": r"C:\Windows\Fonts\Deng.ttf",
    "微软雅黑": r"C:\Windows\Fonts\msyh.ttc",
}

_PREPARED = False
_BUNDLED_REGULAR = ""
_CHOSEN_FILE = ""
_BOLD_SAFE: Dict[str, bool] = {}


def _norm(name: str) -> str:
    return "".join(str(name or "").casefold().split())


def parse_fc_list(text: str) -> List[Tuple[str, str]]:
    """解析 `fc-list file family` 的输出。同一行逗号后的名字是同一文件的别名。"""
    faces: List[Tuple[str, str]] = []
    for line in str(text or "").splitlines():
        if ":" not in line:
            continue
        path, rest = line.split(":", 1)
        path = path.strip()
        if not path:
            continue
        for part in rest.split(","):
            fam = part.split(":")[0].strip()
            if fam:
                faces.append((fam, path))
    return faces


def match_family(available: Iterable[str], candidates: Iterable[str]) -> str:
    """在 Tk 报出来的族名里找候选。忽略大小写和空格，避开假族名 TkDefaultFont。"""
    index: Dict[str, str] = {}
    for name in available:
        text = str(name or "")
        if not text or text.startswith("@") or _norm(text) == "tkdefaultfont":
            continue
        index.setdefault(_norm(text), text)
    for cand in candidates:
        hit = index.get(_norm(cand))
        if hit:
            return hit
    return ""


def choose_ui_family(available: Iterable[str],
                     faces: Sequence[Tuple[str, str]] = (),
                     platform: str = "",
                     bundled_family: str = "") -> str:
    """决定界面字体族。返回值必须是 `available` 里的真实族名，找不到就返回空串。

    Linux 上优先随包的单文件字体：系统里的中文字体经常是 TTC/OTC，
    Tk/Xft 会点中集合里没有所需字形的那一副，中文就变成方框。
    Windows / macOS 的系统字体栈能正确读集合字体，所以继续用雅黑、苹方。
    """
    index: Dict[str, str] = {}
    for name in available:
        text = str(name or "")
        if not text or text.startswith("@") or _norm(text) == "tkdefaultfont":
            continue
        index.setdefault(_norm(text), text)

    groups: Dict[str, List[str]] = {}
    for fam, path in faces:
        fam = str(fam or "").split(":")[0].strip()
        path = str(path or "").strip()
        if not fam or not path:
            continue
        bucket = groups.setdefault(path, [])
        if fam not in bucket:
            bucket.append(fam)

    def listed(name: str) -> str:
        hit = index.get(_norm(name))
        if hit:
            return hit
        for names in groups.values():
            if not any(_norm(item) == _norm(name) for item in names):
                continue
            for alias in names:
                hit = index.get(_norm(alias))
                if hit:
                    return hit
        return ""

    def paths_for(name: str) -> List[str]:
        key = _norm(name)
        return [path for path, names in groups.items()
                if any(_norm(item) == key for item in names)]

    def collection_only(name: str) -> bool:
        paths = paths_for(name)
        if not paths:
            return False
        return all(path.lower().endswith((".ttc", ".otc")) for path in paths)

    def first(names: Iterable[str], skip_collection: bool) -> str:
        for name in names:
            hit = listed(name)
            if not hit:
                continue
            if skip_collection and collection_only(hit):
                continue
            return hit
        return ""

    # 随包字体是单文件 OTF。Linux 上只要 Tk 已经能看见它，就用它，避免点到 TTC。
    if platform not in ("win32", "darwin") and bundled_family:
        hit = listed(bundled_family)
        if hit:
            return hit

    skip = platform not in ("win32", "darwin")
    hit = first(_CJK_CANDIDATES, skip_collection=skip)
    if hit:
        return hit
    fc_names: List[str] = []
    for names in groups.values():
        for name in names:
            if name not in fc_names:
                fc_names.append(name)
    hit = first(fc_names, skip_collection=skip)
    if hit:
        return hit
    if bundled_family:
        hit = listed(bundled_family)
        if hit:
            return hit
    hit = first(_CJK_CANDIDATES, skip_collection=False)
    if hit:
        return hit
    return first(fc_names, skip_collection=False)


def _bundled_paths() -> List[str]:
    """找到随包字体。打包后走 app_paths，源码运行走仓库里的 assets/fonts。"""
    root = ""
    rel = os.path.join("assets", "fonts", _BUNDLED_FILES[0])
    try:
        import app_paths
        found = app_paths.find(rel)
        if found and os.path.isfile(found):
            root = os.path.dirname(found)
    except Exception:                       # noqa: BLE001
        root = ""
    if not root:
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "assets", "fonts")
        if os.path.isdir(here):
            root = here
    out: List[str] = []
    if not root:
        return out
    for name in _BUNDLED_FILES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            out.append(path)
    return out


def _register_fontconfig(path: str) -> bool:
    try:
        fc = ctypes.CDLL("libfontconfig.so.1")
    except OSError:
        return False
    try:
        fc.FcInit.restype = ctypes.c_int
        fc.FcInit()
        fc.FcConfigGetCurrent.restype = ctypes.c_void_p
        fc.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        fc.FcConfigAppFontAddFile.restype = ctypes.c_int
        fc.FcConfigBuildFonts.argtypes = [ctypes.c_void_p]
        cfg = fc.FcConfigGetCurrent()
        ok = fc.FcConfigAppFontAddFile(cfg, os.fsencode(path))
        if ok:
            fc.FcConfigBuildFonts(cfg)
        return bool(ok)
    except Exception:                       # noqa: BLE001
        return False


def _register_macos(path: str) -> bool:
    try:
        cf = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        ct = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreText.framework/CoreText")
        data = os.fsencode(path)
        cf.CFURLCreateFromFileSystemRepresentation.restype = ctypes.c_void_p
        cf.CFURLCreateFromFileSystemRepresentation.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
        url = cf.CFURLCreateFromFileSystemRepresentation(None, data, len(data), False)
        if not url:
            return False
        # kCTFontManagerScopeProcess = 1，只对当前进程生效
        ct.CTFontManagerRegisterFontsForURL.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        ct.CTFontManagerRegisterFontsForURL.restype = ctypes.c_bool
        return bool(ct.CTFontManagerRegisterFontsForURL(url, 1, None))
    except Exception:                       # noqa: BLE001
        return False


def _register_font_file(path: str) -> bool:
    """把字体注册进当前进程。必须在第一次创建 Tk 之前调用，Xft 才会看见它。"""
    if not path or not os.path.isfile(path):
        return False
    if sys.platform == "win32":
        try:
            # FR_PRIVATE = 0x10：不装进系统，只给本进程的 GDI / Tk 用。
            # 不声明 argtypes 时，str 会被当成多字节路径，含空格或非英文的目录会注册失败。
            add = ctypes.windll.gdi32.AddFontResourceExW
            add.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p]
            add.restype = ctypes.c_int
            return int(add(path, 0x10, None)) > 0
        except Exception:                   # noqa: BLE001
            return False
    if sys.platform == "darwin":
        return _register_macos(path)
    return _register_fontconfig(path)


def _ensure_posix_utf8() -> None:
    """POSIX 区域若是 C/POSIX，fontconfig 会按非 UTF-8 区域挑默认字体。

    不改 Windows：那边的系统编码是代码页，强行改成 utf-8 会让文件对话框乱码。
    已经是 UTF-8 时什么都不做。
    """
    if sys.platform == "win32":
        return
    try:
        enc = (locale.getpreferredencoding(False) or "").lower()
    except Exception:                       # noqa: BLE001
        enc = ""
    if "utf" in enc:
        return
    for loc in ("C.UTF-8", "en_US.UTF-8", "zh_CN.UTF-8"):
        try:
            locale.setlocale(locale.LC_CTYPE, loc)
        except locale.Error:
            continue
        os.environ["LC_CTYPE"] = loc
        lang = os.environ.get("LANG", "")
        if lang.lower() in ("", "c", "posix"):
            os.environ["LANG"] = loc
        return


def prepare_fonts() -> str:
    """注册随包中文字体。返回正文字体路径；没有文件时返回空串。

    可以重复调用。要在创建 Tk 根窗口之前调用一次，否则 Linux 上
    字体列表已经缓存，后注册的文件不会出现在界面里。
    """
    global _PREPARED, _BUNDLED_REGULAR
    if _PREPARED:
        return _BUNDLED_REGULAR
    _ensure_posix_utf8()
    paths = _bundled_paths()
    regular = ""
    for path in paths:
        if os.path.basename(path).lower().startswith("notosanssc-regular"):
            regular = path
            break
    if not regular and paths:
        regular = paths[0]
    for path in paths:
        _register_font_file(path)
    _BUNDLED_REGULAR = regular
    _PREPARED = True
    return _BUNDLED_REGULAR


def _fontconfig_faces() -> List[Tuple[str, str]]:
    if sys.platform == "win32":
        return []
    try:
        proc = subprocess.run(
            ["fc-list", ":lang=zh", "file", "family"],
            check=False, capture_output=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_fc_list(proc.stdout.decode("utf-8", "replace"))


def _file_for(family: str, faces: Sequence[Tuple[str, str]], bundled: str) -> str:
    if family and _norm(family) == _norm(BUNDLED_FAMILY) and bundled and os.path.isfile(bundled):
        return bundled
    single = ""
    collection = ""
    for fam, path in faces:
        if _norm(fam) != _norm(family) or not path or not os.path.isfile(path):
            continue
        if path.lower().endswith((".ttc", ".otc")):
            collection = collection or path
        else:
            single = single or path
    if single:
        return single
    win = _WIN_FILES.get(_norm(family), "")
    if win and os.path.isfile(win):
        return win
    if collection:
        return collection
    if bundled and os.path.isfile(bundled):
        return bundled
    return ""


def _families(root: tk.Misc) -> List[str]:
    try:
        return [str(name) for name in tkfont.families(root)]
    except Exception:                       # noqa: BLE001
        return []


def _bold_keeps_family(family: str) -> bool:
    """粗体如果会把族名换成另一套拉丁字体，中文标题会单独变成方框。"""
    try:
        got = tkfont.Font(family=family, size=12, weight="bold").actual("family")
    except tk.TclError:
        return False
    return _norm(str(got)) == _norm(family)


def _point_named_fonts(family: str) -> None:
    """没单独指定字体的控件（消息框、菜单）也走这套中文字体。"""
    if not family:
        return
    for name in _NAMED_FONTS:
        try:
            tkfont.nametofont(name).configure(family=family)
        except tk.TclError:
            continue


def install_cjk(root: tk.Misc) -> str:
    """探测并装上界面字体。返回最终族名；探测失败时返回空串。"""
    global FONT_UI, FONT_MONO, _CHOSEN_FILE
    prepare_fonts()
    faces = _fontconfig_faces()
    bundled_name = BUNDLED_FAMILY if _BUNDLED_REGULAR else ""
    FONT_UI = choose_ui_family(_families(root), faces, sys.platform, bundled_name)
    FONT_MONO = match_family(_families(root), _MONO_CJK_CANDIDATES)
    _CHOSEN_FILE = _file_for(FONT_UI, faces, _BUNDLED_REGULAR)
    if FONT_UI:
        _BOLD_SAFE[FONT_UI] = _bold_keeps_family(FONT_UI)
        _point_named_fonts(FONT_UI)
    return FONT_UI


def ui_family() -> str:
    return FONT_UI


def cjk_font_file() -> str:
    """给 PIL 往画面上写中文用。优先本次界面选中的文件。"""
    if _CHOSEN_FILE and os.path.isfile(_CHOSEN_FILE):
        return _CHOSEN_FILE
    if _BUNDLED_REGULAR and os.path.isfile(_BUNDLED_REGULAR):
        return _BUNDLED_REGULAR
    faces = _fontconfig_faces()
    for _fam, path in faces:
        if path.lower().endswith((".ttf", ".otf")) and os.path.isfile(path):
            return path
    for _fam, path in faces:
        if os.path.isfile(path):
            return path
    for path in _WIN_FILES.values():
        if os.path.isfile(path):
            return path
    return ""


def _fallback_family() -> str:
    """还没探测成功时，用命名字体真正的族名，而不是字符串 "TkDefaultFont"。"""
    try:
        name = str(tkfont.nametofont("TkDefaultFont").cget("family"))
    except tk.TclError:
        return "sans-serif"
    if not name or _norm(name) == "tkdefaultfont":
        return "sans-serif"
    return name


def font(size: int = 10, bold: bool = False, mono: bool = False) -> Tuple:
    """返回可直接传给 tk/ttk 控件的字体元组。

    粗体只有在不会换掉字体族时才加。等宽候选不含中日韩字形时退回界面字体，
    否则日志、列表里的中文会变成方框，而旁边的标签是正常的。
    """
    if mono and FONT_MONO:
        family = FONT_MONO
    else:
        family = FONT_UI or _fallback_family()
    weight = "bold" if bold and _BOLD_SAFE.get(family, False) else "normal"
    return (family, int(size), weight)


def _option_font(size: int) -> str:
    family = FONT_UI
    if not family:
        return ""
    if any(ch.isspace() for ch in family) or any(ch in family for ch in "{}\\"):
        family = "{" + family.replace("}", "") + "}"
    return "%s %d" % (family, int(size))


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
    spec = _option_font(10)
    if spec:
        # 没写 font= 的控件、以及下拉列表，都从选项数据库拿默认字体
        db("*Font", spec)
        db("*Text.font", spec)
        db("*Listbox.font", spec)
        db("*Menu.font", spec)
        db("*TCombobox*Listbox.font", spec)
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
    install_cjk(root)
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
    """把 tk.Text 调成深色控制台观感（选项数据库给了底色，这里补内边距）。

    没有中日韩等宽字体时会退回界面字体，避免中文日志变成方框。
    """
    widget.configure(font=font(size, mono=mono), padx=8, pady=6,
                     spacing1=1, spacing3=1, wrap=widget.cget("wrap"))


def style_listbox(widget: tk.Listbox) -> None:
    widget.configure(font=font(9, mono=True), bd=0, highlightthickness=1,
                     highlightbackground=BORDER, highlightcolor=ACCENT,
                     activestyle="none", selectbackground=SELECT,
                     selectforeground=TEXT)


# 模块导入时就注册随包字体。viewer / 标定窗口都是 import 本模块之后才创建 Tk，
# 这样 Linux 的 fontconfig 能在第一次打开界面之前看见这个文件。
prepare_fonts()
