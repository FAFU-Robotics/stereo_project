# PyInstaller 运行时钩子：在 app_entry 之前执行。
# 只做环境变量和 DLL 搜索路径，不 import torch / cv2。
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

if getattr(sys, "frozen", False):
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    try:
        os.chdir(exe_dir)
    except OSError:
        pass
    base = getattr(sys, "_MEIPASS", exe_dir)
    for sub in ("", "torch/lib", "torchvision", "cv2", "numpy.libs", "open3d", "open3d/cpu"):
        folder = base if not sub else os.path.join(base, sub)
        if folder and os.path.isdir(folder):
            try:
                os.add_dll_directory(folder)
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
