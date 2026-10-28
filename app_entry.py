"""app_entry.py —— 唯一启动入口（源码运行 & 打包成 EXE 都用它）
=============================================================================
【为什么需要一个单独入口】

    打包器只认"一个入口函数"。而本项目有三个可执行面：

        标定 GUI       tools/calib_gui.py
        实时深度/点云   main.py --mode preview|depth|pointcloud
        SDK            stereo_sdk（**不能被包进 EXE**，见 README）

    没有统一入口时，只能打三个 EXE，或者打一个"只有 GUI"的 EXE。
    这里用一个入口 + 子命令覆盖前两者，产物仍然只有一个 exe：

        StereoApp.exe                    → 标定 GUI（默认，不需要 torch）
        StereoApp.exe --mode depth       → 实时深度（需要 torch + 权重）
        StereoApp.exe --mode pointcloud  → 实时点云（需要 open3d）
        StereoApp.exe --where            → 打印路径解析结果（排错第一条命令）
        StereoApp.exe --selftest         → 无相机/无模型自检

【这个文件解决了三个"只有打包后才会暴露"的问题】

    1) torch 环境变量时序
       stereo_depth.py 在 import torch 之前设 TORCHDYNAMO_DISABLE=1。
       但打包后 bootloader 可能先触发某个 hook 间接 import torch，
       那时再设就晚了（Windows 无 Triton，会报 torch.compile 相关错误）。
       → 本模块在**第一行可执行代码**就设，早于一切 import。

    2) windowed 模式没有 stdout
       PyInstaller --noconsole 下 sys.stdout / sys.stderr 是 None，
       任何 print() 都会抛 AttributeError，而且抛在 tkinter 回调里时界面直接白屏。
       → setup_streams() 把 None 换成写日志文件的替身。

    3) 未捕获异常在无控制台时完全不可见
       → 装 sys.excepthook，写 logs/ 并尽量弹窗。

【用法】
    python app_entry.py                       # 源码运行 GUI
    python app_entry.py --where               # 看路径解析
    python app_entry.py --selftest            # 自检
    python app_entry.py --mode depth --scale 0.5   # 其余参数原样转给 main.py
=============================================================================
"""

from __future__ import annotations

# ==== 必须排在所有 import 之前：Windows 无 Triton，dynamo 会在 CUDA 上失败 ====
import os as _os

if str(_os.environ.get("TORCHDYNAMO_DISABLE", "")).strip() not in ("1", "true", "True"):
    _os.environ["TORCHDYNAMO_DISABLE"] = "1"
# 同一类问题的另一个开关：某些 torch 版本在没有 Triton 时会尝试走 inductor
_os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import argparse
import datetime as _dt
import sys
import traceback
from typing import Any, List, Optional, TextIO

APP_TITLE = "双目深度相机 · 标定与实时工具"

#: 版本号只有一个来源：stereo_sdk/types.py 的 SDK_VERSION（pyproject 也从这里读）。
#: 入口再硬编码一个数字，迟早出现"exe 显示 0.4、pip 包显示 0.3"的不一致。
try:
    from stereo_sdk.types import SDK_VERSION as APP_VERSION   # noqa: E402
except Exception:                                            # noqa: BLE001
    APP_VERSION = "0.3.0"

CLI_MODES = ("preview", "depth", "pointcloud")


# --------------------------------------------------------------------------- 基础
def _root() -> str:
    """项目根 / 只读资源根。延迟 import：app_paths 自身也可能被 pyinstaller 重定位。"""
    import app_paths
    return app_paths.bundle_root()


def _bootstrap_syspath() -> None:
    """让 `import camera` / `import calib_core` 这类平铺模块可用。

    源码运行时项目根已在 sys.path（脚本所在目录），但打包产物不会自动加；
    tools/ 下的模块（calib_core 等）是平铺导入，也需要它进 sys.path。
    pip 安装（site-packages）时本文件所在目录不是项目根，所以用 search_roots()
    把所有可能的位置都试一遍，而不是只认 __file__。
    """
    import app_paths

    for root in app_paths.search_roots():
        for p in (root, _os.path.join(root, "tools")):
            if _os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)


class _StreamTee:
    """把 print 重定向到日志文件（windowed 模式下 sys.stdout 是 None）。"""

    def __init__(self, fh: TextIO) -> None:
        self._fh = fh

    def write(self, s: str) -> int:
        try:
            self._fh.write(s)
            self._fh.flush()
        except Exception:      # noqa: BLE001 —— 日志失败绝不能影响主流程
            pass
        return len(s or "")

    def flush(self) -> None:
        try:
            self._fh.flush()
        except Exception:      # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False


def setup_streams() -> Optional[TextIO]:
    """确保 stdout/stderr 可写；返回已打开的日志句柄（调用方负责关闭）。"""
    import app_paths

    log_dir = app_paths.writable_dir("logs")
    stamp = _dt.datetime.now().strftime("%Y%m%d")
    path = _os.path.join(log_dir, f"app-{stamp}.log")
    try:
        fh = open(path, "a", encoding="utf-8", buffering=1)
        fh.write(f"\n{'=' * 70}\n[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] 启动 v{APP_VERSION}\n")
        fh.write(app_paths.describe_text() + "\n")
        if sys.stdout is None or getattr(sys.stdout, "closed", False):
            sys.stdout = _StreamTee(fh)      # type: ignore[assignment]
        if sys.stderr is None or getattr(sys.stderr, "closed", False):
            sys.stderr = _StreamTee(fh)      # type: ignore[assignment]
        return fh
    except OSError:
        return None


def install_excepthook(log_fh: Optional[TextIO]) -> None:
    """未捕获异常 → 日志 + 弹窗（无控制台时这是唯一可见的反馈）。"""

    def _hook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            if log_fh is not None:
                log_fh.write("\n[未捕获异常]\n" + text)
        except Exception:      # noqa: BLE001
            pass
        try:
            sys.__stderr__.write(text)      # 有控制台时照常显示
        except Exception:      # noqa: BLE001
            pass
        try:
            import app_paths
            log_hint = app_paths.writable_dir("logs")
        except Exception:      # noqa: BLE001
            log_hint = "(日志目录不可用)"
        try:
            import tkinter as tk
            from tkinter import messagebox
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror(
                "程序遇到未处理的错误",
                f"{exc_type.__name__}: {exc}\n\n"
                f"完整堆栈已写入日志：\n{log_hint}\n\n"
                "把该日志发给开发者可定位问题。")
            r.destroy()
        except Exception:      # noqa: BLE001
            pass

    sys.excepthook = _hook


def seed_user_data() -> None:
    """首次运行把只读资源里的初始配置复制到可写根。

    只复制"用户可以改"的东西：config.yaml 与 calibration/。
    weights/ 与 third_party/ 有 100 MB 量级，绝不复制（只读引用）。
    已存在同名文件时不覆盖 —— 升级包不能冲掉用户标定好的参数。
    """
    import app_paths

    if not app_paths.is_frozen():
        return                     # 源码运行：只读根 == 可写根，无需播种
    if app_paths.data_root() == app_paths.bundle_root():
        return
    app_paths.seed_file("config.yaml")
    app_paths.seed_dir("calibration")
    app_paths.ensure_dir(app_paths.data("output"))


# --------------------------------------------------------------------------- 分发
def _run_gui(extra: List[str], gui_selftest: bool = False) -> int:
    """标定 GUI。**不 import torch**，所以没有 GPU 也能用。

    extra 里只挑 GUI 自己认的参数（--dir），其余忽略并写进日志 ——
    否则 argparse 会直接以"unrecognized arguments"退出，而用户看不到原因。
    """
    _bootstrap_syspath()
    try:
        # 供打包器静态分析（PyInstaller 只看 import 语句，动态 find_spec 抓不到）
        if False:                               # pragma: no cover
            import calib_check                  # noqa: F401
            import calib_core                   # noqa: F401
            import calib_manage                 # noqa: F401
            import gui_state                    # noqa: F401
            import gui_theme                    # noqa: F401
        import calib_gui
    except ImportError as exc:
        # pip 安装到 site-packages 后，tools/ 不在 wheel 里（它依赖 tkinter/CV 全链路，
        # 不适合当库分发）。这里给出一句能直接照做的话，而不是一串 ModuleNotFound。
        print(f"[错误] 找不到标定 GUI 模块：{exc}\n"
              f"        请从项目源码目录运行：python tools\\calib_gui.py\n"
              f"        或设置 STEREO_HOME 指向项目根后重试。")
        return 2

    argv: List[str] = []
    dropped: List[str] = []
    i = 0
    while i < len(extra):
        item = str(extra[i])
        if item == "--dir" and i + 1 < len(extra):
            argv += ["--dir", extra[i + 1]]
            i += 2
            continue
        if item.startswith("--dir="):
            argv.append(item)
        else:
            dropped.append(item)
        i += 1
    if gui_selftest:
        argv.append("--selftest")
    if dropped:
        print(f"[提示] 标定 GUI 不认识这些参数，已忽略：{' '.join(dropped)}")
    return int(calib_gui.main(argv) or 0)


def _run_cli(mode: str, extra: List[str]) -> int:
    """实时预览/深度/点云：转给 main.py，参数原样透传。"""
    _bootstrap_syspath()
    import main as cli
    argv = ["--mode", mode] + list(extra or [])
    return int(cli.main(argv) or 0)


def _selftest() -> int:
    """不需要相机与 GPU 的自检：路径 + 依赖 + GUI 构建 + 版本一致性。"""
    _bootstrap_syspath()
    import app_paths

    ok = True
    print(f"== {APP_TITLE} v{APP_VERSION} 自检 ==")
    print(app_paths.describe_text())
    print()

    # 1) 版本一致性（pyproject 用 dynamic 从 SDK_VERSION 读，这里不重复维护）
    try:
        from stereo_sdk.types import SDK_VERSION
        same = SDK_VERSION == APP_VERSION
        print(f"  {'✅' if same else '⚠️ '} SDK_VERSION={SDK_VERSION}  APP_VERSION={APP_VERSION}")
        if not same:
            print("       （两者不一致不影响运行；打包脚本会以 SDK_VERSION 为准）")
    except Exception as exc:                    # noqa: BLE001
        print(f"  ❌ 读不到 SDK_VERSION：{exc}")
        ok = False

    # 2) 关键文件
    for rel, need in (("config.yaml", True),
                      ("calibration/stereo_calib.yaml", False),
                      ("third_party/Fast-FoundationStereo/core/foundation_stereo.py", False)):
        p = app_paths.find(rel)
        got = _os.path.exists(p) if need else _os.path.isfile(p)
        mark = "✅" if got else ("❌" if need else "⚠️ ")
        print(f"  {mark} {rel} -> {p}")
        if need:
            ok &= bool(got)

    # 3) 依赖
    for mod, need in (("cv2", True), ("numpy", True), ("yaml", True),
                      ("tkinter", True), ("PIL", False), ("torch", False)):
        try:
            __import__(mod)
            print(f"  ✅ {mod}")
        except Exception as exc:                # noqa: BLE001
            print(f"  {'❌' if need else '⚠️ '} {mod} 未安装（{type(exc).__name__}）")
            if need:
                ok = False

    # 4) torch 能不能真的用 GPU（有 torch 才查）
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if cuda else "-"
        print(f"  {'✅' if cuda else '⚠️ '} CUDA 可用={cuda}  设备={name}  torch={torch.__version__}")
    except Exception:                           # noqa: BLE001
        print("  ⚠️  未安装 torch：仅 GUI 与标定可用，实时深度/点云不可用")

    # 5) GUI 能否构建（不连相机）
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        r.destroy()
        print("  ✅ tkinter 可建窗口")
    except Exception as exc:                    # noqa: BLE001
        print(f"  ❌ tkinter 建窗口失败：{exc}")
        ok = False

    print("\n" + ("✅ 自检通过" if ok else "❌ 自检有失败项"))
    return 0 if ok else 1


def cli_main(argv: Optional[List[str]] = None) -> int:
    """入口函数（`[project.scripts]` 与 PyInstaller 都指向这里）。"""
    raw = list(sys.argv[1:] if argv is None else argv)

    # 第一优先级：让 stdout/stderr 能写（windowed 模式下它们是 None）
    _bootstrap_syspath()
    log_fh = setup_streams()
    install_excepthook(log_fh)
    seed_user_data()

    ap = argparse.ArgumentParser(
        prog="stereo-app", add_help=True,
        description=APP_TITLE,
        epilog="其余未识别参数会原样转给 main.py（例如 --scale 0.5 --threaded on）。")
    ap.add_argument("--gui", action="store_true", help="启动标定 GUI（默认行为）")
    ap.add_argument("--mode", choices=("gui",) + CLI_MODES, default="gui",
                    help="gui=标定界面；其余三种进实时链路（需要 torch）")
    ap.add_argument("--config", default="", help="改用别的 config.yaml")
    ap.add_argument("--where", action="store_true", help="打印路径解析结果后退出")
    ap.add_argument("--version", action="store_true", help="打印版本后退出")
    ap.add_argument("--selftest", action="store_true", help="环境自检（不需要相机）")
    ap.add_argument("--gui-selftest", action="store_true",
                    help="只把 GUI 建出来再自动关掉（验证界面能在本机启动）")
    # parse_known_args：让 main.py 的 --scale/--threaded/--list 等参数照样能用
    args, extra = ap.parse_known_args(raw)

    if args.version:
        try:
            from stereo_sdk.types import SDK_VERSION
            print(f"{APP_TITLE} {SDK_VERSION}")
        except Exception:                        # noqa: BLE001
            print(f"{APP_TITLE} {APP_VERSION}")
        return 0

    if args.where:
        import app_paths
        print(app_paths.describe_text())
        print("\n搜索顺序：")
        for i, p in enumerate(app_paths.search_roots(), 1):
            print(f"  {i}. {p}")
        return 0

    if args.selftest:
        return _selftest()

    if args.config:
        extra = ["--config", args.config] + extra

    try:
        if args.gui_selftest:
            return _run_gui(extra, gui_selftest=True)
        if args.mode == "gui" or args.gui:
            return _run_gui(extra)
        return _run_cli(args.mode, extra)
    finally:
        if log_fh is not None:
            try:
                log_fh.write(f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] 退出\n")
                log_fh.close()
            except Exception:                    # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(cli_main())
