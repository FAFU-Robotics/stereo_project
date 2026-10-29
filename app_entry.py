"""app_entry.py —— 唯一启动入口（源码运行 & 打包成 EXE 都用它）
=============================================================================
【为什么需要一个单独入口】

    打包器只认"一个入口函数"。而本项目有三个可执行面：

        标定 GUI       tools/calib_gui.py
        实时深度/点云   main.py --mode preview|depth|pointcloud
        SDK            stereo_sdk（**不能被包进 EXE**，见 README）

    没有统一入口时，只能打三个 EXE，或者打一个"只有 GUI"的 EXE。
    这里用一个入口 + 子命令覆盖前两者，产物仍然只有一个 exe：

        StereoApp.exe                    → Stereo Viewer（默认，不需要先开相机）
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
    python app_entry.py                       # 打开 Stereo Viewer
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
# numpy 与 torch 各带一份 OpenMP 时，Windows 上会直接中止进程
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("PYTHONUTF8", "1")
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import datetime as _dt
import sys
import traceback
from typing import Any, Dict, List, Optional, TextIO


def _prepare_frozen_runtime() -> None:
    """冻结运行时：切到 exe 目录，并把 CUDA / OpenCV 的 DLL 目录加进搜索路径。

    双击 EXE 时，Windows 的工作目录不一定是 exe 所在目录（快捷方式、资源管理器
    从别的文件夹启动都可能如此）。先 chdir，相对路径才不会漂。
    torch 的 cublas/cudnn 在 torch/lib 里，不在系统 PATH；不注册这个目录，
    打包后第一次 import torch 就会报找不到 DLL。显卡驱动本身不在包里。
    """
    if not bool(getattr(sys, "frozen", False)):
        return
    exe_directory = _os.path.dirname(_os.path.abspath(sys.executable))
    try:
        _os.chdir(exe_directory)
    except OSError:
        pass
    base = getattr(sys, "_MEIPASS", exe_directory)
    for sub in ("", "torch/lib", "torchvision", "cv2", "numpy.libs", "open3d", "open3d/cpu"):
        folder = base if not sub else _os.path.join(base, sub)
        if not folder or not _os.path.isdir(folder):
            continue
        try:
            _os.add_dll_directory(folder)
        except (OSError, AttributeError):
            pass
        _os.environ["PATH"] = folder + _os.pathsep + _os.environ.get("PATH", "")


_prepare_frozen_runtime()

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


class _SafeStream:
    """同时写日志和原控制台，并且永远不因编码失败抛异常。

    中文 Windows 的控制台是 GBK。自检里如果打印 ✅/❌，GBK 编不了，
    print 会抛 UnicodeEncodeError，异常处理里再打印一次就连对话框也出两层。
    """

    def __init__(self, underlying: Optional[TextIO], log_fh: Optional[TextIO]) -> None:
        self._u = underlying
        self._log = log_fh

    def write(self, s: str) -> int:
        text = "" if s is None else str(s)
        if self._log is not None:
            try:
                self._log.write(text)
                self._log.flush()
            except Exception:  # noqa: BLE001
                pass
        stream = self._u
        if stream is None:
            return len(text)
        try:
            stream.write(text)
        except UnicodeEncodeError:
            enc = getattr(stream, "encoding", None) or "mbcs"
            try:
                stream.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
        return len(text)

    def flush(self) -> None:
        for stream in (self._log, self._u):
            if stream is None:
                continue
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
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
        out = None if sys.stdout is None or getattr(sys.stdout, "closed", False) else sys.stdout
        err = None if sys.stderr is None or getattr(sys.stderr, "closed", False) else sys.stderr
        sys.stdout = _SafeStream(out, fh)    # type: ignore[assignment]
        sys.stderr = _SafeStream(err, fh)    # type: ignore[assignment]
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


def _model_selftest() -> int:
    """不连相机：加载权重，对一对合成图推理一帧。用来确认打包后的 GPU 链路。"""
    _bootstrap_syspath()
    import app_paths
    from main import load_config
    from stereo_depth import FFSDepthEstimator

    cfg_path = app_paths.find("config.yaml")
    print(f"  config = {cfg_path}")
    if not _os.path.isfile(cfg_path):
        print("  [FAIL] 找不到 config.yaml")
        return 1
    cfg = load_config(cfg_path)
    calib = app_paths.find("calibration/stereo_calib.yaml")
    weights = app_paths.find("weights/23-36-37/model_best_bp2_serialize.pth")
    repo = str(cfg.get("project", {}).get("ffs_repo", ""))
    print(f"  calib  = {calib}  exists={_os.path.isfile(calib)}")
    print(f"  weights= {weights}  exists={_os.path.isfile(weights)}")
    print(f"  ffs    = {repo}  exists={_os.path.isdir(repo)}")
    est = FFSDepthEstimator(cfg["project"], cfg.get("ffs", {}))
    ok, msg = est.weights_ready()
    print(f"  [权重] {msg}")
    if not ok:
        return 3
    try:
        est.load()
        import numpy as np
        left = np.zeros((480, 640, 3), np.uint8)
        left[:, :] = (40, 80, 120)
        right = left.copy()
        est.warmup(1, left=left, right=right)
        out = est.infer(left, right)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] 模型自检失败：{type(exc).__name__}: {exc}")
        return 4
    finally:
        try:
            est.close()
        except Exception:  # noqa: BLE001
            pass
    infer_ms = float(out["timings_ms"]["infer"])
    print(f"  [OK] 推理完成  {infer_ms:.1f} ms  disp={tuple(out['disp'].shape)}  scale={out['scale']}")
    return 0


def _show_text(title: str, text: str) -> None:
    import tkinter as tk
    from tkinter import scrolledtext
    win = tk.Tk()
    win.title(title)
    win.geometry("720x480")
    box = scrolledtext.ScrolledText(win, wrap="word", font=("Consolas", 10))
    box.pack(fill="both", expand=True)
    box.insert("1.0", text)
    box.configure(state="disabled")
    tk.Button(win, text="关闭", command=win.destroy).pack(pady=6)
    win.mainloop()


def _notify_failure(rc: int) -> None:
    """模式异常退出时给一个能看懂的提示。详细堆栈在 logs/。"""
    import app_paths
    log_dir = app_paths.writable_dir("logs")
    tail = ""
    try:
        logs = [f for f in _os.listdir(log_dir) if f.startswith("app-") and f.endswith(".log")]
        if logs:
            path = _os.path.join(log_dir, sorted(logs)[-1])
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            tail = "".join(lines[-25:])
    except OSError:
        path = ""
    try:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk()
        r.withdraw()
        messagebox.showwarning(
            "这一项没有正常结束",
            f"退出码 {rc}。\n\n日志目录：\n{log_dir}\n\n最近日志：\n{tail[-800:]}")
        r.destroy()
    except Exception:  # noqa: BLE001
        print(tail)


def _run_viewer(selftest: bool = False) -> int:
    """RealSense Viewer 式主界面。标定工作台从这里再打开，不在这里 import torch。"""
    _bootstrap_syspath()
    if False:                               # pragma: no cover  供打包器看到这个模块
        import viewer_shell                 # noqa: F401
    import viewer_shell
    argv = ["--selftest"] if selftest else []
    return int(viewer_shell.main(argv) or 0)


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
        print(f"  {'[OK]' if same else '[WARN]'} SDK_VERSION={SDK_VERSION}  APP_VERSION={APP_VERSION}")
        if not same:
            print("       （两者不一致不影响运行；打包脚本会以 SDK_VERSION 为准）")
    except Exception as exc:                    # noqa: BLE001
        print(f"  [FAIL] 读不到 SDK_VERSION：{exc}")
        ok = False

    # 2) 关键文件
    for rel, need in (("config.yaml", True),
                      ("calibration/stereo_calib.yaml", False),
                      ("third_party/Fast-FoundationStereo/core/foundation_stereo.py", False)):
        p = app_paths.find(rel)
        got = _os.path.exists(p) if need else _os.path.isfile(p)
        mark = "[OK]" if got else ("[FAIL]" if need else "[WARN]")
        print(f"  {mark} {rel} -> {p}")
        if need:
            ok &= bool(got)

    # 3) 依赖
    for mod, need in (("cv2", True), ("numpy", True), ("yaml", True),
                      ("tkinter", True), ("PIL", False), ("torch", False)):
        try:
            __import__(mod)
            print(f"  [OK] {mod}")
        except Exception as exc:                # noqa: BLE001
            print(f"  {'[FAIL]' if need else '[WARN]'} {mod} 未安装（{type(exc).__name__}）")
            if need:
                ok = False

    # 4) torch 能不能真的用 GPU（有 torch 才查）
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if cuda else "-"
        print(f"  {'[OK]' if cuda else '[WARN]'} CUDA 可用={cuda}  设备={name}  torch={torch.__version__}")
    except Exception:                           # noqa: BLE001
        print("  [WARN] 未安装 torch：仅 GUI 与标定可用，实时深度/点云不可用")

    # 5) GUI 能否构建（不连相机）
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        r.destroy()
        print("  [OK] tkinter 可建窗口")
    except Exception as exc:                    # noqa: BLE001
        print(f"  [FAIL] tkinter 建窗口失败：{exc}")
        ok = False

    print("\n" + ("[OK] 自检通过" if ok else "[FAIL] 自检有失败项"))
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
    ap.add_argument("--gui", action="store_true",
                    help="启动标定工作台（四页流程）；不带参数时默认进主界面 Stereo Viewer")
    ap.add_argument("--mode", choices=("gui",) + CLI_MODES, default="gui",
                    help="gui=主界面/标定；其余三种进实时链路（需要 torch）")
    ap.add_argument("--config", default="", help="改用别的 config.yaml")
    ap.add_argument("--lang", default="", choices=("", "zh", "en"),
                    help="界面语言：zh / en；留空读 config.yaml 的 ui.language")
    ap.add_argument("--where", action="store_true", help="打印路径解析结果后退出")
    ap.add_argument("--version", action="store_true", help="打印版本后退出")
    ap.add_argument("--selftest", action="store_true", help="环境自检（不需要相机）")
    ap.add_argument("--gui-selftest", action="store_true",
                    help="只把主界面（Stereo Viewer）建出来再自动关掉")
    ap.add_argument("--calib-selftest", action="store_true",
                    help="只把旧版标定工作台建出来再自动关掉")
    ap.add_argument("--launcher", action="store_true",
                    help="打开 Stereo Viewer（打包后的 EXE 双击时默认就是这个）")
    ap.add_argument("--viewer-selftest", action="store_true",
                    help="只把 Viewer 建出来再自动关掉")
    ap.add_argument("--model-selftest", action="store_true",
                    help="不连相机，只加载权重并对合成图推理一帧")
    # parse_known_args：让 main.py 的 --scale/--threaded/--list 等参数照样能用
    args, extra = ap.parse_known_args(raw)

    try:
        import i18n
        cfg_path = args.config or ""
        if not cfg_path:
            try:
                import app_paths
                cfg_path = app_paths.find("config.yaml")
            except Exception:                   # noqa: BLE001
                cfg_path = ""
        i18n.resolve_language(cfg_path, args.lang)
    except Exception:                           # noqa: BLE001
        pass

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

    if args.model_selftest:
        return _model_selftest()

    if args.viewer_selftest:
        return _run_viewer(selftest=True)

    import multiprocessing
    multiprocessing.freeze_support()

    # 不带参数（源码或双击 EXE）打开 Viewer。--gui 仍直接进标定工作台，
    # Viewer 里的「标定工作台」按钮走的就是这条，避免再套一层窗口。
    if args.config:
        extra = ["--config", args.config] + extra

    open_viewer = (not args.gui and not args.gui_selftest and not args.calib_selftest
                   and args.mode == "gui" and (args.launcher or not extra))
    try:
        if open_viewer:
            return _run_viewer()
        if args.gui_selftest:
            return _run_viewer(selftest=True)
        if args.calib_selftest:
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
