"""中文界面字体：族名必须真实存在，不能把 TkDefaultFont 当成族名。

用法：
    python tests/test_cjk_font.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for _p in (ROOT, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gui_theme as theme  # noqa: E402


def test_parse_fc_list_aliases() -> None:
    text = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc: "
        "Noto Sans CJK SC,Noto Sans CJK SC Regular:style=Regular\n"
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc: "
        "WenQuanYi Micro Hei,文泉驿微米黑\n"
    )
    faces = theme.parse_fc_list(text)
    families = {fam for fam, _path in faces}
    assert "Noto Sans CJK SC" in families
    assert "文泉驿微米黑" in families
    assert all(not fam.startswith("style=") for fam, _path in faces)
    assert all(not path.endswith(":style=Regular") for _fam, path in faces)


def test_linux_prefers_bundled_face_over_collection() -> None:
    """Linux 上系统只有 TTC 时，用随包的单文件字体，避免 Tk 点错集合里的字面。"""
    available = ["DejaVu Sans", "Noto Sans CJK SC", "Noto Sans SC"]
    faces = [("Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    got = theme.choose_ui_family(available, faces, "linux", theme.BUNDLED_FAMILY)
    assert got == "Noto Sans SC"


def test_linux_uses_single_face_system_font_when_bundle_missing() -> None:
    available = ["DejaVu Sans", "WenQuanYi Micro Hei"]
    faces = [("WenQuanYi Micro Hei", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf")]
    got = theme.choose_ui_family(available, faces, "linux", "")
    assert got == "WenQuanYi Micro Hei"


def test_linux_alias_matches_localized_family() -> None:
    available = ["DejaVu Sans", "文泉驿微米黑"]
    faces = [
        ("WenQuanYi Micro Hei", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf"),
        ("文泉驿微米黑", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttf"),
    ]
    got = theme.choose_ui_family(available, faces, "linux", "")
    assert got == "文泉驿微米黑"


def test_linux_falls_back_to_collection_when_nothing_else_exists() -> None:
    available = ["DejaVu Sans", "Noto Sans CJK SC"]
    faces = [("Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    got = theme.choose_ui_family(available, faces, "linux", "")
    assert got == "Noto Sans CJK SC"


def test_windows_prefers_yahei_over_bundled() -> None:
    available = ["Segoe UI", "Noto Sans SC", "Microsoft YaHei UI"]
    got = theme.choose_ui_family(available, (), "win32", theme.BUNDLED_FAMILY)
    assert got == "Microsoft YaHei UI"


def test_windows_uses_bundled_when_no_system_cjk() -> None:
    available = ["Segoe UI", "Calibri", "Noto Sans SC"]
    got = theme.choose_ui_family(available, (), "win32", theme.BUNDLED_FAMILY)
    assert got == "Noto Sans SC"


def test_never_returns_tkdefaultfont_as_family() -> None:
    available = ["TkDefaultFont", "DejaVu Sans", "Ubuntu"]
    got = theme.choose_ui_family(available, (), "linux", "")
    assert got == ""
    assert theme.match_family(available, theme._CJK_CANDIDATES) == ""


def test_case_insensitive_match() -> None:
    got = theme.choose_ui_family(["noto sans sc"], (), "linux", "Noto Sans SC")
    assert got == "noto sans sc"


def test_bundled_font_draws_chinese() -> None:
    path = os.path.join(ROOT, "assets", "fonts", "NotoSansSC-Regular.otf")
    bold = os.path.join(ROOT, "assets", "fonts", "NotoSansSC-Bold.otf")
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 1_000_000
    assert os.path.isfile(bold)
    from PIL import ImageFont
    face = ImageFont.truetype(path, 48)
    assert face.getname()[0] == "Noto Sans SC"
    box = face.getmask("尚未串流，查看彩色、深度和点云").getbbox()
    assert box is not None and (box[2] - box[0]) > 80
    assert theme._register_font_file(path)


def test_tk_ui_font_is_a_real_cjk_family() -> None:
    import tkinter as tk
    import tkinter.font as tkfont
    try:
        root = tk.Tk()
    except tk.TclError:
        return
    try:
        root.withdraw()
        theme.apply(root)
        family = theme.ui_family()
        assert family and family != "TkDefaultFont"
        spec = theme.font(16, bold=True)
        assert spec[0] != "TkDefaultFont"
        face = tkfont.Font(root=root, family=spec[0], size=spec[1], weight=spec[2])
        actual = str(face.actual("family"))
        assert theme._norm(actual) == theme._norm(spec[0])
        ui = tkfont.Font(root=root, family=family, size=32)
        assert ui.measure("中") > ui.measure("i")
        sample = "尚未串流"
        label = tk.Label(root, text=sample, font=theme.font(18))
        label.update_idletasks()
        assert label.cget("text") == sample
    finally:
        root.destroy()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}")
        else:
            print(f"[OK] {fn.__name__}")
    if failed:
        raise SystemExit(1)
    print(f"[OK] {len(tests)} passed")
