# -*- mode: python ; coding: utf-8 -*-
"""StereoCamera onedir 打包规格。

在 stereo_project 目录执行：
    packaging\\build_exe.ps1

产物：release\\StereoCamera\\StereoCamera.exe
不要改成 onefile：torch 的 CUDA 库约 4 GB，单文件每次启动都要解压。
不要开 UPX：压缩过的 CUDA DLL 在 Windows 上经常加载失败。
"""
from __future__ import annotations

import os

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
TOOLS = os.path.join(ROOT, "tools")


def _tree(src: str, dest: str, skip_dirs=None):
    """把目录收成 (文件, 目标目录) 列表，跳过缓存和用不到的脚本。"""
    skip_dirs = set(skip_dirs or ())
    out = []
    if not os.path.isdir(src):
        return out
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and d != "__pycache__"]
        rel = os.path.relpath(dirpath, src)
        dest_dir = dest if rel == "." else os.path.join(dest, rel)
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            out.append((os.path.join(dirpath, name), dest_dir))
    return out


datas = []
binaries = []
hiddenimports = [
    "app_paths",
    "app_entry",
    "main",
    "camera",
    "camera_guard",
    "viewer",
    "calibration",
    "stereo_depth",
    "depth_utils",
    "pointcloud",
    "pipeline_threaded",
    "runtime",
    "profiler",
    "logger",
    "monitor",
    "view3d",
    "calib_gui",
    "calib_core",
    "calib_check",
    "calib_manage",
    "gui_state",
    "gui_theme",
    "viewer_shell",
    "stereo_sdk",
    "stereo_sdk.session",
    "stereo_sdk.types",
    "stereo_sdk.calib",
    "stereo_sdk.export",
    "stereo_sdk.recorder",
    "stereo_sdk.mock",
    "yaml",
    "PIL",
    "PIL.Image",
    "PIL.ImageTk",
    "imageio",
    "imageio.plugins",
    "imageio.v2",
    "imageio.v3",
    "omegaconf",
    "omegaconf.base",
    "omegaconf.dictconfig",
    "omegaconf.listconfig",
    "omegaconf.nodes",
    "einops",
    "timm",
    "timm.layers.mlp",
    "timm.layers.norm",
    "timm.models.edgenext",
    "open3d",
    "open3d.visualization",
    "plotly",
    "dash",
    "flask",
    "werkzeug",
    "cv2",
    "numpy",
]

# 权重、配置、标定、FFS 源码必须是磁盘上的真实文件。
# FFS 在运行时把仓库根插进 sys.path 再 import core / Utils，不能只打进 PYZ。
datas += [(os.path.join(ROOT, "config.yaml"), ".")]
datas += _tree(os.path.join(ROOT, "calibration"), "calibration")
datas += _tree(os.path.join(ROOT, "weights"), "weights")
datas += _tree(
    os.path.join(ROOT, "third_party", "Fast-FoundationStereo"),
    os.path.join("third_party", "Fast-FoundationStereo"),
    skip_dirs={"scripts", "cpp", "demo_data", ".git"},
)

# torch / torchvision / open3d / timm：官方 hook 经常漏掉 CUDA DLL 或动态导入。
# collect_all 体积大，但这是这条链路上能稳定启动的做法。
for pkg in ("torch", "torchvision", "open3d", "timm", "imageio", "omegaconf", "plotly", "dash"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

excludes = [
    "pytest",
    "IPython",
    "notebook",
    "matplotlib",
    "pandas",
    "sklearn",
    "tensorboard",
    "tkinter.test",
    "triton",
    "tensorrt",
]

a = Analysis(
    [os.path.join(ROOT, "app_entry.py")],
    pathex=[ROOT, TOOLS],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SPECPATH, "rthook_stereo.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StereoCamera",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=True,
    argv_emulation=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="StereoCamera",
)
