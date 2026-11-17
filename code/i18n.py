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
    "正在打开左右相机": "Opening left and right cameras",
    "左右相机已打开": "Left and right cameras are open",
    "相机已就绪，可以开始串流": "Cameras are ready. You can start streaming",
    "相机与模型已就绪，开始串流会很快": "Cameras and model are ready. Streaming will start quickly",
    "相机正在打开，完成后立即开始串流": "Cameras are opening. Streaming starts as soon as they are ready",
    "正在启动串流": "Starting stream",
    "相机和模型已就绪，正在启动串流": "Cameras and model are ready. Starting stream",
    "未自动打开相机。点开始串流时再连接。": "Cameras stay closed until you start streaming.",
    "左相机已断开": "Left camera disconnected",
    "右相机已断开": "Right camera disconnected",
    "相机已断开": "Camera disconnected",
    "相机打开失败": "Couldn't open the cameras",
    "正在确认相机是否断开": "Checking whether the camera disconnected",
    "USB 设备已断开，串流已停止": "USB device removed. Streaming stopped",
    "串流已停止，相机仍保持打开": "Streaming stopped. Cameras stay open",
    "左相机 USB 已断开": "Left camera USB disconnected",
    "右相机 USB 已断开": "Right camera USB disconnected",
    "左右相机 USB 已断开": "Left and right camera USB disconnected",
    "USB 设备断开，读帧确认相机已掉线": "USB removed. A failed read confirmed the camera is gone",
    "相机无画面，已停止串流": "No frames from the camera. Streaming stopped",
    "相机已重新接上，正在打开左右相机": "Camera plugged back in. Opening left and right cameras",
    "先释放相机，把设备交给标定工作台": "Releasing the cameras so calibration can use them",
    "标定前会先释放相机。": "Cameras are released before calibration.",
    "就绪。开始串流后可在 2D / 3D 之间切换。标定前会先释放相机。":
        "Ready. Start streaming to switch between 2D and 3D. Cameras are released before calibration.",
    "串流已停止\n\n相机仍开着，再次开始会很快\n右侧结果仍可点开回看":
        "Streaming stopped.\n\nCameras stay open, so the next start is quick.\nSaved results on the right can still be opened.",
    "相机已断开\n\n重新接上后可以再次开始串流":
        "Camera disconnected.\n\nPlug it back in, then start streaming again",

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
    "\n\n搜索顺序（按优先级）：\n": "\n\nSearch order (by priority):\n",
    "[错误] 当前 Python 没有 tkinter。请安装官方 CPython，并勾选 tcl/tk。":
        "[ERROR] This Python has no tkinter. Install official CPython with tcl/tk enabled.",
    # ---- 标定工作台（calib_gui）----
    '帧缩略图': 'Frame thumbnails',
    '[错误] 当前 Python 没有 tkinter。请安装官方 python.org 的 CPython 发行版（安装时勾选 tcl/tk），而不是精简版/绿色版解释器。': '[ERROR] This Python has no tkinter. Install the official CPython from python.org (enable tcl/tk during setup), not a slim/portable build.',
    '单击选中，双击看大图；': 'Click to select, double-click to enlarge;',
    '双目标定工作台 —— 采集 / 标定 / 检查 / 保存': 'Stereo Calibration Console — Capture / Calibrate / Inspect / Save',
    '环境信息': 'Env info',
    '左眼 · 角点与判定叠加': 'Left eye · corners and verdict overlay',
    'LEFT（相机未打开）': 'LEFT (camera not open)',
    '右眼 · 角点与判定叠加': 'Right eye · corners and verdict overlay',
    '本帧指标': "This frame's metrics",
    '过门为绿': 'Green = passed the gates',
    '采集清单': 'Capture checklist',
    '姿态覆盖': 'Pose coverage',
    '质量门（可改）': 'Quality gates (editable)',
    '低于下限即拒绝': 'Below a limit = rejected',
    '数据源': 'Data source',
    '带对数的是可用目录': 'Entries with a pair count are usable folders',
    '标定板与动作': 'Board',
    '板参数必须与实物一致': 'Board parameters must match the physical board',
    '运行日志': 'Run log',
    '逐帧检出与被拒原因': 'Per-frame detections and rejection reasons',
    '标定结果与体检': 'Result & checks',
    'RMS 之外还查基线量级 / T 方向 / 尺寸一致性': 'Beyond RMS: baseline scale, T direction and size consistency',
    '校正结果预览': 'Rectified preview',
    '先完成标定，再生成检查图': 'Calibrate first, then generate check images',
    '复检数据': 'Recheck data',
    '极线残差、分区 |dy| 与结论': 'Epipolar residuals, per-region |dy| and verdict',
    '写出标定文件': 'Write calibration file',
    '只新增 calibration/*.yaml，不改 config.yaml': 'Only adds calibration/*.yaml; config.yaml is never touched',
    '归档历史': 'Archive history',
    '选中后可激活：自动备份当前那份，可回滚': 'Select one to activate: the current file is backed up automatically and can be rolled back',
    '操作结果': 'Operation output',
    '校验 / 对比的输出都在这里': 'Validation and comparison output appears here',
    '正在取消…（主计算是一整次原生调用，需等它算完，通常 1~3 秒）': 'Cancelling… (the main computation is one native call; it must finish, usually 1-3 s)',
    '相机正在打开…（首次打开约需 2~5 秒）': 'Opening camera… (first open takes about 2-5 s)',
    '标定计算中…（可点「取消」）': 'Calibrating… (you can click Cancel)',
    '标定完成，可生成检查图': 'Calibration done — check images can be generated',
    '生成检查图…（可点「取消」）': 'Generating check images… (you can click Cancel)',
    '已载入标定文件，正在生成检查图…': 'Calibration file loaded — generating check images…',
    '标定文件已保存': 'Calibration file saved',
    '提示：运行中的程序需要调用 SDK 的 reload_calibration()，或重启程序，才会用上新的标定参数。': "Note: a running program picks up the new calibration only after calling the SDK's reload_calibration() or restarting.",
    '双目标定 GUI（采集/标定/检查/保存）': 'Stereo calibration GUI (capture/calibrate/inspect/save)',
    '预填采集目录': 'Prefill capture folder',
    '只建界面，300ms 后自动退出': 'Build the UI only; auto-exits after 300 ms',
    '选中后按 Delete 删除（连磁盘文件一起删）。': 'Select a frame and press Delete to remove it (files are deleted from disk too).',
    '本目录来自磁盘，只读。': 'This folder was loaded from disk and is read-only.',
    '确认删除': 'Confirm delete',
    '操作失败': 'Operation failed',
    '搜索顺序（按优先级）：': 'Search order (by priority):',
    '环境信息已复制到剪贴板': 'Environment info copied to clipboard',
    '就绪': 'Ready',
    '相机未打开': 'Camera not open',
    '还没开始采集': 'Capturing has not started',
    '① 采集': '① Capture',
    '② 标定': '② Calibrate',
    '③ 检查校正效果': '③ Check rectification',
    '④ 保存参数': '④ Save',
    '打开相机': 'Open camera',
    '完成采集 → 去标定': 'Finish capture → Calibrate',
    '列': 'Cols',
    '行': 'Rows',
    '格边长 mm': 'Square size mm',
    '开始标定': 'Start calibration',
    '取消': 'Cancel',
    '生成检查图': 'Generate check images',
    '去第②页标定': 'Go to step ② to calibrate',
    '载入已有标定': 'Load existing calibration',
    '采集完成：数量与姿态都达标': 'Capture complete: count and pose coverage met',
    '继续采集': 'Keep capturing',
    '提示': 'Note',
    '还没有可看的帧：先在①页采集，或在②页选一个采集目录': 'No frames to view yet: capture on step ① or pick a capture folder on step ②',
    '相机已关闭': 'Camera closed',
    '关闭相机': 'Close camera',
    '先打开相机': 'Open the camera first',
    '还没有可用画面，等相机出一帧再试': 'No frame available yet — wait for the camera to deliver one',
    '没有可撤销的帧': 'Nothing to undo',
    '本次会话还没有保存任何一帧': 'No frames saved in this session yet',
    '还没有采集任何一帧': 'No frames captured yet',
    '还没有采集会话：去①页采集，或点「浏览…」选一个含 left_*.png 的文件夹': 'No capture sessions yet: capture on step ①, or click "Browse…" to pick a folder containing left_*.png',
    '上一项任务还在运行': 'The previous task is still running',
    '请先选择一个采集目录': 'Pick a capture folder first',
    '还没有标定结果': 'No calibration result yet',
    '这一页要在标定之后才能画。两条路：\n\n· 去第②页做一次标定（会自动跳到本页）\n· 若磁盘上已有 calibration/stereo_calib.yaml，点上方「载入已有标定文件」直接看': 'This page needs a calibration first. Two ways:\n\n· Run a calibration on step ② (it jumps here automatically)\n· If calibration/stereo_calib.yaml already exists on disk, click "Load existing calibration" above',
    '请先在②页「数据源」选一个有 left_*.png 的会话': 'Pick a session with left_*.png in "Data source" on step ② first',
    '没有可用的采集图': 'No capture images available',
    '检查图需要一对真实的左右原图来校正。\n\n当前没有选中含 left_*.png 的采集目录。\n现在就选一个文件夹吗？\n（也可以先去②页「数据源」下拉框选，或在①页采集几对）': 'Check images need a real left/right pair to rectify.\n\nNo capture folder with left_*.png is selected.\nPick one now?\n(You can also choose from "Data source" on step ②, or capture a few pairs on step ①)',
    '选择包含 left_*.png / right_*.png 的采集目录': 'Choose a capture folder containing left_*.png / right_*.png',
    '本次标定结果': "This calibration's result",
    '还没有任何标定结果。请先去第②页选采集目录并执行标定（没有标定文件时本页无法生成校正图）。': 'No calibration result yet. Run a calibration on step ② first (this page cannot rectify without a calibration file).',
    '没有标定文件': 'No calibration file',
    '还没有可保存的标定结果': 'No calibration result to save',
    '先选中一条归档': 'Select an archive entry first',
    '错误': 'Error',
    '确认': 'Confirm',
    '把这份归档设为当前生效标定？\n当前文件会先自动备份，可随时退回。': 'Make this archive the active calibration?\nThe current file is backed up first and can be restored anytime.',
    '还没有归档可比': 'No archive to compare with',
    '文件不存在，无法对比': 'File not found — cannot compare',
    '本次采集还没写报告，现在写吗？': 'This session has no report yet. Write it now?',
    '每格 px': 'px per square',
    '清晰度': 'Sharpness',
    '对比度': 'Contrast',
    '目标对数': 'Target pairs',
    '空格': 'Space',
    '还没有保存任何一帧': 'No frames saved yet',
    '磁盘': 'Disk',
    '可保存': 'OK to save',
    '勉强可用': 'Marginal',
    '不可用': 'Unusable',
    '拦下：': 'Rejected: ',
    '拦下：与已保存的某张姿态过于相似': 'Rejected: too similar to a saved pose',
    '质检通过，可以保存（空格）': 'Quality gates passed — save with Space',
    '保存失败': 'Save failed',
    '写报告失败': 'Report write failed',
    '载入失败': 'Load failed',
    '激活失败': 'Activation failed',
    '对比失败': 'Comparison failed',
    '关闭': 'Close',
    '（日志写入失败）': '(log write failed)',
    '复制全部': 'Copy all',
    # 显示方式下拉的值（gui_state.CHECK_MODES；内部记忆用中文名）
    '上下叠放 + 水平线': 'Stacked + epipolar lines',
    '并排 + 参考线': 'Side by side + guides',
    '红青立体': 'Red-cyan anaglyph',
    '左右差值': 'L-R difference',
    '棋盘混合': 'Checker blend',
    '分区 |dy| 热力图': 'Per-region |dy| heatmap',
    '① 采集    →    ② 标定    →    ③ 检查    →    ④ 保存': '① → ② → ③ → ④',
    '保存这一帧': 'Save this frame',
    '强制保存': 'Force save',
    '撤销': 'Undo',
    '覆盖热力图': 'Coverage heatmap',
    '删帧…': 'Delete frames…',
    '缩略图…': 'Thumbnails…',
    '浏览…': 'Browse',
    '固定内参（推荐：只用外参重估）': 'Fix intrinsics (recommended: re-estimate extrinsics only)',
    '显示方式': 'Display mode',
    '重新生成': 'Regenerate',
    '另存图片…': 'Save image as…',
    '保存到': 'Save to',
    '备注': 'Note',
    '保存并归档': 'Save & archive',
    '校验当前文件': 'Validate current file',
    '与上一份对比': 'Compare with previous',
    '激活选中': 'Activate selected',
    '下一步：': 'Next:',
    '选中一帧后按「删除选中」或 Delete 键；文件会一并从磁盘删除。': 'Select a frame, then click "Delete selected" or press Delete; files are removed from disk too.',
    '删除选中': 'Delete selected',
    '已取消（未改动任何文件）': 'Cancelled (no files changed)',
    '[取消] 用户中止，本次结果未采用': '[CANCELLED] Stopped by user — this result was discarded',
    '保存新标定前自动备份': 'Auto-backup before saving a new calibration',
    '本次保存的标定': 'The calibration just saved',
    '标定失败': 'Calibration failed',
    '检查图生成失败': 'Check image generation failed',
    '双眼可用': 'Both eyes usable',
    '仅单眼': 'One eye only',
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
    "{eye}\n有效 {valid:.0f}%\n点数 {pts}":
        "{eye}\nvalid {valid:.0f}%\npoints {pts}",
    "正在读取 {name}": "Reading {name}",
    "停止时出错：{exc}": "Error while stopping: {exc}",
    "\n\n采集根目录：{p0}\n标定文件  ：{p1}\n归档目录  ：{p2}\n日志目录  ：{p3}\n界面记忆  ：{p4}":
        "\n\nCapture root : {p0}\nCalibration  : {p1}\nArchive dir  : {p2}\nLog dir      : {p3}\nUI state     : {p4}",
    "#{p0:02d}  {p1}\n角点 {p2}": "#{p0:02d}  {p1}\ncorners {p2}",
    "app_paths 不可用：{p0}\n\n项目根：{p1}": "app_paths unavailable: {p0}\n\nProject root: {p1}",
    "已载入 {p0}\n{p1}": "Loaded {p0}\n{p1}",
    "校验：{p0}\n\n": "Validation: {p0}\n\n",
    "打不开目录：{exc}": "Cannot open folder: {exc}",
    # ---- 标定工作台（calib_gui）----
    '采集根目录：{p0}': 'Capture root : {p0}',
    '标定文件  ：{p1}': 'Calibration  : {p1}',
    '归档目录  ：{p2}': 'Archive dir  : {p2}',
    '日志目录  ：{p3}': 'Log dir      : {p3}',
    '界面记忆  ：{p4}': 'UI state     : {p4}',
    '{p0}（共 {p1} 帧）': '{p0} ({p1} frames)',
    '#{p0:02d}  {p1}': '#{p0:02d}  {p1}',
    '角点 {p2}': 'corners {p2}',
    '第 {p0:02d} 帧 · {p1} · 角点 {p2}': 'Frame {p0:02d} · {p1} · corners {p2}',
    '{p0}  |  {p1:.1f} 帧/秒  |  每格 {p2:.1f} px': '{p0}  |  {p1:.1f} fps  |  {p2:.1f} px/sq',
    '{p0} 倾斜 {p1:.0f}° {p2}': '{p0} tilt {p1:.0f}° {p2}',
    '角点  左 {p0}   右 {p1}   {p2}': 'Corners  L {p0}   R {p1}   {p2}',
    '每格像素      {p0:.1f}': 'px per square {p0:.1f}',
    '清晰度        {p0:.0f}': 'Sharpness      {p0:.0f}',
    '对比度(最差)  {p0:.0f}': 'Contrast(min)  {p0:.0f}',
    '姿态          {p0}': 'Pose           {p0}',
    '已保存        {p0} / {p1}（双眼 {p2}）': 'Saved         {p0} / {p1} (stereo {p2})',
    '已保存第 {p0} 帧 → {p1}': 'Saved frame {p0} → {p1}',
    '已撤销第 {p0} 帧': 'Undid frame {p0}',
    '删除帧（共 {p0} 帧）': 'Delete frames ({p0} total)',
    '覆盖热力图（{p0:.0f}% 的格子有角点）': 'Coverage heatmap ({p0:.0f}% of squares have corners)',
    '采集完成：{p0} 对，报告 {p1}': 'Capture done: {p0} pairs, report {p1}',
    '{p0}  ({p1} 对)': '{p0}  ({p1} pairs)',
    '目录：{p0}': 'Folder: {p0}',
    '标定板：{p0}': 'Board: {p0}',
    '耗时 {p0:.1f} s，图像尺寸 {p1}x{p2}': 'Took {p0:.1f} s, image size {p1}x{p2}',
    '已载入 {p0}': 'Loaded {p0}',
    '{p1}': '{p1}',
    '载入已有标定：{p0}': 'Load existing calibration: {p0}',
    '检查图已生成（用第 {p0} 帧）': 'Check image generated (using frame {p0})',
    '已激活：{p0}': 'Activated: {p0}',
    '当前文件：{p0}': 'Current file: {p0}',
    '打开相机失败：{p0}: {p1}': 'Opening camera failed: {p0}: {p1}',
    '删除第 {p0:02d} 帧？\n\n{p1}\n{p2}\n\n这两个文件会从磁盘删除，不可撤销。': 'Delete frame {p0:02d}?\n\n{p1}\n{p2}\n\nBoth files are deleted from disk. This cannot be undone.',
    'app_paths 不可用：{p0}': 'app_paths unavailable: {p0}',
    '项目根：{p1}': 'Project root: {p1}',
    '进度：{p0}': 'Progress: {p0}',
    '还差 {p0} 对': '{p0} pairs to go',
    '缺 {p0} 类姿态': 'missing {p0} pose types',
    '已删除第 {p0:02d} 帧': 'Deleted frame {p0:02d}',
    '第 {p0:02d} 帧已不在列表里': 'Frame {p0:02d} is no longer in the list',
    '#{p0:02d}   {p1:<12} 角点 {p2:3d}': '#{p0:02d}   {p1:<12} corners {p2:3d}',
    '校正前 |dy| 平均 {p0:.2f} px  最大 {p1:.2f} px': 'Before rect: |dy| mean {p0:.2f} px  max {p1:.2f} px',
    '校正后 |dy| 平均 {p0:.3f} px  最大 {p1:.3f} px（{p2} 帧复检）': 'After rect:  |dy| mean {p0:.3f} px  max {p1:.3f} px ({p2} frames rechecked)',
    '当前用：{p0} · 基线 {p1:.3f} mm · rect fx {p2:.2f} px': 'Active: {p0} · baseline {p1:.3f} mm · rect fx {p2:.2f} px',
    '还没有载入标定结果。磁盘上已有一份 {p0}，可直接载入查看校正效果；也可以回第②页重新标定。': 'No calibration loaded yet. {p0} already exists on disk — load it to see the rectified view, or calibrate again on step ②.',
    '未找到：\n{p0}': 'Not found:\n{p0}',
    '已保存 {p0}': 'Saved {p0}',
    '已保存：{p0}': 'Saved: {p0}',
    '归档文件不存在：{p0}': 'Archive file not found: {p0}',
    '原文件备份：{p0}': 'Original backed up: {p0}',
    '校验：{p0}': 'Validation: {p0}',
    '[提示] 界面记忆未能写入：{p0}': '[NOTE] Could not write the UI state file: {p0}',
    '未保存：{p0}': 'Not saved: {p0}',
    '当前用：{p0}': 'Active: {p0}',
    '已备份旧文件：{p0}': 'Previous file backed up: {p0}',
    '已归档：{p0}': 'Archived: {p0}',
    '读取失败：{p0}: {p1}': 'Read failed: {p0}: {p1}',
    '分析失败：{p0}: {p1}': 'Analysis failed: {p0}: {p1}',
    '详情已写入：\n{p0}': 'Details written to:\n{p0}',
    '{p0} 失败': '{p0} failed',
    '[失败] {p0}': '[FAILED] {p0}',
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


def to_zh(text: str) -> str:
    """不管输入是中文原文还是英文译文，都归一到中文 key。

    与 retranslate() 的区别：retranslate 返回「当前语言」的显示文本，
    to_zh 永远给中文原键 —— 用于内部取值/写盘（如 CHECK_MODES 查表、gui_state 记忆）。
    """
    if not text:
        return text
    if text in _EN:
        return text
    return _ZH_BY_EN.get(text, text)


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
