"""i18n.py —— 界面中英切换（中文原文作 key，缺翻译时回退中文）

用法：
    from i18n import t, tf, set_language, resolve_language
    label = t("开始串流")                     # 静态文案
    msg = tf("正在切换到{name}", name=label)   # 带占位符的文案
    set_language("en")                        # "en" / "zh"

为什么用中文原文当 key，而不是 key_001：
    1) 代码里直接读得懂，新增文案不必先去登记编号；
    2) 漏翻时回退成中文原文，界面不会出现 "viewer.start_stream" 这种原始 key；
    3) 同一句中文在多处复用，只翻一次。

语言解析优先级：命令行 --lang > config.yaml 的 ui.language > 默认 zh。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

_LANG = "zh"
_EXPLICIT = False        # 命令行 --lang 指定过，就不再被配置文件覆盖

#: 英文翻译表。键必须与源码里的中文原文完全一致（含空格与换行）。
_EN: Dict[str, str] = {
    # ---- 语言选择器 ----
    "界面语言": "Language",
    "中文": "中文",              # 下拉框里两种语言始终用各自母语显示
    "界面语言已切换为中文": "Language switched to Chinese",
    "界面语言已切换为 English": "Language switched to English",

    # ---- 启动菜单 ----
    "选择要打开的功能。关闭本窗口即退出程序。": "Choose what to open. Closing this window exits the app.",
    "主界面（设备 / 深度 / 点云 / 采集 / 标定）": "Main window (device / depth / cloud / capture / calibration)",
    "旧版标定工作台（采集 / 标定 / 检查 / 保存）": "Legacy calibration console (capture / calibrate / inspect / save)",
    "实时预览（左右相机）": "Live preview (left / right)",
    "实时深度（视差图）": "Live depth (disparity)",
    "实时点云（深度图 + 3D）": "Live point cloud (depth + 3D)",
    "环境自检（不连相机）": "Environment check (no camera)",
    "模型自检（加载权重并推理一帧，不连相机）": "Model check (load weights, one inference, no camera)",
    "退出": "Exit",

    # ---- 顶栏 / 设备 ----
    "双目深度相机": "Stereo Depth Camera",
    "未连接": "Offline",
    "分辨率 —    帧率 —": "Resolution —    FPS —",
    "分辨率未知": "resolution unknown",
    "就绪。开始串流后可在 2D / 3D 之间切换。标定请先停止串流。":
        "Ready. Start streaming to switch between 2D and 3D. Stop streaming before calibration.",
    "开始串流": "Start streaming",
    "停止串流": "Stop streaming",
    "正在连接…": "Connecting…",
    "正在连接": "Connecting",
    "正在连接或断开，请稍候": "Connecting or disconnecting, please wait",
    "正在打开相机并加载模型": "Opening camera and loading model",
    "正在打开相机并加载模型…": "Opening camera and loading model…",
    "正在停止…": "Stopping…",
    "正在停止串流": "Stopping stream",
    "串流中": "Streaming",
    "串流已开始": "Streaming started",
    "已停止": "Stopped",
    "连接失败": "Connection failed",
    "设备": "Device",
    "开始串流后显示分辨率和帧率": "Resolution and FPS appear after streaming starts",

    # ---- 画面 / 视图 ----
    "画面": "Views",
    "彩色（左）": "Color (L)",
    "彩色（右）": "Color (R)",
    "深度": "Depth",
    "视差": "Disparity",
    "点云": "Point cloud",
    "彩色": "Color",
    "右彩色": "Color R",
    "点云在中间的 3D 视图里。快捷键 1 彩色  2 深度  3 点云":
        "Point cloud is in the 3D view. Shortcuts: 1 color, 2 depth, 3 point cloud",
    "回到实时": "Live",
    "暂停": "Pause",
    "继续": "Resume",
    "重置视角": "Reset view",
    "已暂停显示": "Display paused",
    "已回到实时画面": "Back to live view",
    "未选择画面": "No view selected",
    "串流已停止\n\n右上角可以再次开始\n右侧结果仍可点开回看":
        "Streaming stopped\n\nYou can start again from the top right\n"
        "Results on the right can still be opened",
    "串流已停止": "Streaming stopped",
    "停止时出错：": "Error while stopping: ",
    "已打开标定工作台。采集、标定、检查、保存都在那个窗口里。":
        "Calibration console opened. Capture, calibrate, inspect and save happen in that window.",
    "未知": "unknown",
    "正在读取 ": "Reading ",
    "没有可显示的画面\n在左侧勾选彩色、深度或视差，或切到 3D":
        "Nothing to show\nEnable Color, Depth or Disparity on the left, or switch to 3D",
    "尚未串流\n\n右上角开始串流，查看彩色、深度和点云\n":
        "Not streaming\n\nClick \"Start streaming\" at the top right to see color, depth and point cloud\n",
    "尚未串流\n\n右上角开始串流，查看彩色、深度和点云\n保存的采集与快照在右侧点选即可回看":
        "Not streaming\n\nClick \"Start streaming\" at the top right.\n"
        "Saved captures and snapshots can be opened from the right panel.",
    "保存的采集与快照在右侧点选即可回看":
        "Saved captures and snapshots can be opened from the right panel",
    "  3D  左键旋转    右键平移    滚轮缩放      深度图上移动鼠标可读距离      空格暂停      S 保存当前帧":
        "  3D: drag / wheel to orbit · hover depth for distance · Space pause · S save",
    "  3D  左键旋转    右键平移    滚轮缩放":
        "  3D  Drag to orbit · Right-drag to pan · Wheel to zoom",
    "      深度图上移动鼠标可读距离      空格暂停      S 保存当前帧":
        "      Move over the depth view to read distance      Space to pause      S to save",

    # ---- 参数 ----
    "参数": "Parameters",
    "深度下限 (m)": "Depth min (m)",
    "深度上限 (m)": "Depth max (m)",
    "点大小": "Point size",
    "推理档位": "Inference preset",
    "切换档位会重新加载模型，大约需要几秒。\n滑条只改显示范围，近处为红色，远处为蓝色。":
        "Changing the preset reloads the model (a few seconds).\n"
        "The slider only changes the display range: near is red, far is blue.",
    "切换档位会重新加载模型，大约需要几秒。\\n滑条只改显示范围，近处为红色，远处为蓝色。":
        "Changing the preset reloads the model (a few seconds).\\n"
        "The slider only changes the display range: near is red, far is blue.",
    "精确": "Precise",
    "平衡": "Balanced",
    "快速": "Fast",
    "正在忙，档位稍后再切": "Busy — change the preset later",
    "档位没有切换": "Preset not changed",

    # ---- 采集 / 标定 ----
    "采集与标定": "Capture & calibration",
    "保存当前帧": "Save frame",
    "标定工作台": "Calibration console",
    "环境自检": "Environment check",
    "模型自检": "Model check",
    "还没有串流，无法保存": "Not streaming — nothing to save",
    "还没有结果，未保存": "No result to save",
    "已保存 ": "Saved ",
    "标定工作台还开着，相机会被它占用":
        "Calibration console is still open and holds the camera",
    "标定工作台已经开着": "Calibration console is already open",
    "正在连接或断开，请稍候再打开标定":
        "Connecting or disconnecting — please wait before opening the console",
    "先停止串流，把相机交给标定工作台":
        "Stop streaming first to hand the camera over to the calibration console",
    "标定工作台已打开": "Calibration console opened",
    "标定工作台已关闭，正在刷新结果":
        "Calibration console closed — refreshing results",
    "正在做环境自检…": "Running environment check…",
    "请先停止串流，再做模型自检（显存不够两件事一起做）":
        "Stop streaming before the model check (not enough VRAM for both)",
    "正在做模型自检，第一次会比较久…":
        "Running model check — the first run takes longer…",
    "标定工作台还开着，请先关掉它再退出":
        "Calibration console is still open — close it before exiting",

    # ---- 右侧结果 ----
    "标定结果": "Calibration",
    "采集与快照": "Captures & snapshots",
    "刷新": "Refresh",
    "打开目录": "Open folder",
    "通知": "Notifications",
    "采集": "Capture",
    "快照": "Snapshot",
    "（还没有采集或快照）": "(no captures or snapshots yet)",
    "回看 · ": "Playback · ",
    "当前没有生效的标定。\n打开标定工作台，完成采集、标定并保存后，这里会显示基线和 RMS。":
        "No active calibration.\nOpen the calibration console, capture, calibrate and save; "
        "baseline and stereo RMS will appear here.",
    "目录不存在：": "Folder not found: ",
    "读取失败：": "Read failed: ",
    "（没有输出）": "(no output)",
    "[错误] 当前 Python 没有 tkinter。请安装官方 CPython，并勾选 tcl/tk。":
        "[ERROR] This Python has no tkinter. Install official CPython with tcl/tk enabled.",
}

#: 带占位符的模板（用 tf() 调用）
_EN_TF: Dict[str, str] = {
    "正在切换到{name}": "Switching to {name}",
    "正在切换到{name}，模型会重新加载": "Switching to {name} — the model will reload",
    "已切换到{name}": "Switched to {name}",
    "深度范围没有生效：{exc}": "Depth range not applied: {exc}",
    "打不开标定工作台：{exc}": "Cannot open the calibration console: {exc}",
    "标定文件读不了：\n{exc}\n{path}": "Cannot read the calibration file:\n{exc}\n{path}",
    "日期      {date}": "Date       {date}",
    "图像      {w} x {h}": "Image      {w} x {h}",
    "基线      {mm:.2f} mm": "Baseline   {mm:.2f} mm",
    "双目 RMS  {rms:.3f} px": "Stereo RMS {rms:.3f} px",
    "单目 RMS  左 {left:.3f} / 右 {right:.3f} px": "Mono RMS   L {left:.3f} / R {right:.3f} px",
    "校正焦距  {fx:.1f} px": "Rect. fx   {fx:.1f} px",
    "历史归档 {n} 份": "Archive    {n} files",
    "正在回看 {label}（{bits}）": "Playing back {label} ({bits})",
    "{title}结束，退出码 {rc}": "{title} exited with code {rc}",
    "正在打开 {name}…": "Opening {name}…",
    "语言偏好没有写回 {path}": "Could not save the language preference to {path}",
    "{eye}    采集 {cap:.1f} fps    推理 {inf:.2f} fps    中位 {med:.3f} m    {preset}":
        "{eye}    capture {cap:.1f} fps    inference {inf:.2f} fps    median {med:.3f} m    {preset}",
    "{eye}    有效 {valid:.0f}%    点数 {pts}":
        "{eye}    valid {valid:.0f}%    points {pts}",
    "正在读取 {name}": "Reading {name}",
    "停止时出错：{exc}": "Error while stopping: {exc}",
    "打不开目录：{exc}": "Cannot open folder: {exc}",
}


def set_language(lang: Optional[str], explicit: bool = False) -> str:
    """设置界面语言：'en' / 'zh'（其它值按中文处理）。返回生效值。

    explicit=True 表示来自命令行 --lang，后续 resolve_language 不再用配置覆盖它。
    """
    global _LANG, _EXPLICIT
    _LANG = "en" if str(lang or "").lower().startswith("en") else "zh"
    if explicit:
        _EXPLICIT = True
    return _LANG


def get_language() -> str:
    return _LANG


def is_english() -> bool:
    return _LANG == "en"


def t(text: str) -> str:
    """翻译静态文案；中文模式或未收录时原样返回。

    比对时把字面量 "\\n"（反斜杠+n）与真实换行视作同一串：
    源码里有的地方写的是转义形式，字典里存的是真实换行，不归一化会漏翻。
    """
    if _LANG != "en":
        return text
    hit = _EN.get(text)
    if hit is not None:
        return hit
    hit = _EN.get(text.replace("\\n", "\n"))
    return text if hit is None else hit


def tf(template: str, **kwargs: Any) -> str:
    """翻译带占位符的文案并格式化。格式化失败时退回未格式化的译文，避免抛异常。"""
    text = _EN_TF.get(template, template) if _LANG == "en" else template
    try:
        return text.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return text


def resolve_language(config_path: str = "", cli_lang: str = "") -> str:
    """按「命令行 > 配置文件 ui.language > zh」决定语言并设置。"""
    if cli_lang:
        return set_language(cli_lang, explicit=True)
    if _EXPLICIT:                       # 命令行已经定过，别被配置盖掉
        return _LANG
    lang = ""
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            lang = str((cfg.get("ui", {}) or {}).get("language", "") or "")
        except Exception:                   # noqa: BLE001
            lang = ""
    return set_language(lang)


#: 英文译文 → 中文原文。界面切换语言时要把控件上现有的旧文本换掉，
#: 手上只有"当前显示的字"，需要靠它倒推回 key。
_ZH_BY_EN: Dict[str, str] = {}
for _k, _v in _EN.items():
    _ZH_BY_EN.setdefault(_v, _k)


def language_label(lang: Optional[str] = None) -> str:
    """给下拉框用的显示名：中文 / English。"""
    return "English" if (lang or _LANG) == "en" else "中文"


def retranslate(text: str) -> str:
    """把控件上已有的文本切换到当前语言；不属于词表的原样返回。

    两种输入都接受：
      - 中文原文（刚建好界面时）
      - 上一次翻译出来的英文（再次切换时）
    """
    if not text:
        return text
    if text in _EN:
        zh = text
    else:
        zh = _ZH_BY_EN.get(text, "")
        if not zh:
            return text
    return _EN.get(zh, zh) if _LANG == "en" else zh


def save_language(config_path: str, lang: str) -> bool:
    """把语言写回 config.yaml 的 ui.language，保留注释与原有排版。

    失败（文件只读、格式异常）不抛异常，只返回 False —— 界面上切换语言
    不该因为写配置失败而中断。
    """
    lang = "en" if str(lang or "").lower().startswith("en") else "zh"
    if not config_path or not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return False

    out = []
    in_ui = False
    wrote = False
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if stripped.startswith("ui:"):
            in_ui = True
            out.append(line)
            continue
        # 顶层新键（顶格且不是注释）意味着 ui 段结束
        if in_ui and stripped and indent == 0 and not stripped.startswith("#"):
            if not wrote:
                out.append(f"  language: {lang}\n")
                wrote = True
            in_ui = False

        if in_ui and indent > 0 and stripped.startswith("language:"):
            # 保留行尾注释与原有对齐，别把 `language: zh  # zh/en` 的说明写没了
            idx = line.find("#")
            gap, tail = "", ""
            if idx >= 0:
                tail = line[idx:].rstrip("\r\n")
                gap = line[:idx]
                gap = gap[len(gap.rstrip()):] or "        "
            out.append(f"  language: {lang}{gap}{tail}\n")
            wrote = True
            continue
        out.append(line)

    if not wrote:
        if in_ui:                       # ui 段在文件末尾
            out.append(f"  language: {lang}\n")
        else:
            out.append("\nui:\n  language: %s\n" % lang)

    try:
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.writelines(out)
    except OSError:
        return False
    return True
