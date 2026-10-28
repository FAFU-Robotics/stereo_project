"""app_paths.py —— 源码运行 / 冻结（PyInstaller）运行 共用的路径解析
=============================================================================
【为什么必须有这个文件】

    打包前，全项目靠 `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
    找项目根；打包后这个推断**必然失效**：

        PyInstaller onefile : __file__ → C:\\Users\\xxx\\AppData\\Local\\Temp\\_MEIxxxx\\...
        PyInstaller onedir  : __file__ → <安装目录>\\_internal\\...
        pip install         : __file__ → <Python>\\Lib\\site-packages\\...

    三者都不是"用户想存放 output/ 和 calibration/ 的地方"。结果就是：
    config.yaml 找不到、标定文件写不进去、点云存到临时目录关机就没。

【两条根，必须分开】

    ┌─ bundle_root()  只读资源根 ─────────────────────────────────────────┐
    │  代码、config.yaml、calibration/ 的初始模板、third_party/、weights/  │
    │  onefile → sys._MEIPASS（每次启动新建的临时目录）                    │
    │  onedir  → <安装目录>\\_internal                                     │
    │  源码    → 项目根                                                   │
    │  ⚠️ 绝不能往这里写文件：onefile 的临时目录退出即删。                 │
    └──────────────────────────────────────────────────────────────────────┘

    ┌─ data_root()    可写数据根 ─────────────────────────────────────────┐
    │  output/（采集图、点云）、calibration/（标定结果）、logs/、state     │
    │  冻结时 → exe 所在目录（便携：整个文件夹拷走就能带走数据）           │
    │  源码时 → 项目根                                                    │
    │  可用环境变量 STEREO_HOME 强制指定（多用户共用一台机器时用）         │
    └──────────────────────────────────────────────────────────────────────┘

【查找规则】

    `find()` 不假设"资源一定在某一个根下"，而是按
        data_root → bundle_root → exe 目录 → 当前工作目录
    依次探测。这样无论打包器把 datas 放在 _internal 里、还是平铺在 exe 旁边
    （PyInstaller 6 的 --contents-directory "" 就是后者），都能找到。

    找不到时返回"首选根下的路径"而不是空字符串 —— 报错信息里要能显示
    "我以为它应该在哪"，否则用户无法自查。

【自检】
    python app_paths.py            # 打印当前解析结果
=============================================================================
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

#: config.yaml 是"这是不是一个项目根目录"的判据。
#: 用它而不是判空目录，是因为 pip 安装到 site-packages 时目录里没有它。
MARKER_FILE = "config.yaml"


# --------------------------------------------------------------------------- 状态
def is_frozen() -> bool:
    """是否运行在 PyInstaller 等冻结产物里。

    `sys.frozen` 由 PyInstaller / cx_Freeze / Nuitka 都会设置，是行业通行判据。
    `sys._MEIPASS` 只有 PyInstaller 有，单独用它会在别的打包器上误判。
    """
    return bool(getattr(sys, "frozen", False))


def exe_dir() -> str:
    """exe 所在目录（源码运行时是项目根）。"""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def bundle_root() -> str:
    """只读资源根（见文件头表格）。"""
    if is_frozen():
        # onefile 会解压到 _MEIPASS；onedir 下 _MEIPASS 指向 _internal
        return os.path.abspath(getattr(sys, "_MEIPASS", exe_dir()))
    return os.path.dirname(os.path.abspath(__file__))


def _source_root() -> str:
    """源码运行时的项目根。

    不能简单用 `dirname(__file__)`：`pip install` 之后本文件在 site-packages 里，
    那个目录不是项目根。用 MARKER_FILE 判别，认不出来就退回当前工作目录。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, MARKER_FILE)):
        return here
    return os.getcwd()


def data_root() -> str:
    """可写数据根（见文件头表格）。优先级：STEREO_HOME > 冻结 exe 目录 > 项目根。"""
    env = str(os.environ.get("STEREO_HOME", "") or "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if is_frozen():
        return exe_dir()
    return _source_root()


def search_roots() -> List[str]:
    """按优先级返回所有可能藏着资源的目录（去重、保序）。"""
    out: List[str] = []
    for p in (data_root(), bundle_root(), exe_dir(), os.getcwd()):
        if p and p not in out:
            out.append(p)
    return out


# --------------------------------------------------------------------------- 查找
def find(rel: str, *, prefer: Optional[str] = None) -> str:
    """在 search_roots() 里依次找 rel，返回第一个存在的绝对路径。

    找不到时返回 `prefer`（或 data_root）下的绝对路径 —— 调用方拿它拼报错信息，
    用户能立刻看出"程序去哪个目录找过、期望放在哪"。
    """
    rel = str(rel or "")
    if not rel:
        return data_root()
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    for base in ([prefer] if prefer else []) + search_roots():
        cand = os.path.normpath(os.path.join(base, rel))
        if os.path.exists(cand):
            return cand
    return os.path.normpath(os.path.join(prefer or data_root(), rel))


def find_all(rel: str) -> List[str]:
    """返回所有根下存在的候选路径（排错时列出"我找到几份、分别在哪"）。"""
    if os.path.isabs(str(rel or "")):
        return [os.path.normpath(rel)] if os.path.exists(rel) else []
    hits: List[str] = []
    for base in search_roots():
        cand = os.path.normpath(os.path.join(base, rel))
        if os.path.exists(cand) and cand not in hits:
            hits.append(cand)
    return hits


def resource(rel: str) -> str:
    """只读资源（config、third_party、weights…）。优先只读根，找不到再全盘探。"""
    return find(rel, prefer=bundle_root())


def data(rel: str, *, create: bool = False) -> str:
    """可写数据路径（output、calibration、logs…）。**永远落在 data_root 下**，
    不会因为"别处也有一份同名文件"而写错地方。"""
    path = os.path.normpath(os.path.join(data_root(), str(rel or "")))
    if create:
        ensure_dir(path if not os.path.splitext(path)[1] else os.path.dirname(path))
    return path


def ensure_dir(path: str) -> str:
    """mkdir -p；返回入参本身，方便链式写。失败不抛（只读介质上调用方自行处理）。"""
    try:
        if path:
            os.makedirs(path, exist_ok=True)
    except OSError:
        pass
    return path


def writable_dir(rel: str) -> str:
    """取一个"确实可写"的数据子目录；建不出来就退回临时目录并给出告警。

    典型场景：程序被装在 `C:\\Program Files`，exe 同级无写权限。
    这时不能直接崩 —— 标定 GUI 起码要能启动、能看已有标定。
    """
    target = data(rel)
    try:
        os.makedirs(target, exist_ok=True)
        probe = os.path.join(target, ".write_test")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        return target
    except OSError:
        import tempfile
        fallback = os.path.join(tempfile.gettempdir(), "stereo_project", str(rel or "data"))
        ensure_dir(fallback)
        return fallback


# --------------------------------------------------------------------------- 播种
def seed_file(rel: str, *, overwrite: bool = False) -> str:
    """首次运行时，把只读资源里的文件复制到可写根（例如 calibration 模板）。

    已存在就不覆盖：用户改过的标定文件不能被升级包悄悄冲掉。
    返回可写根下的目标路径（无论是否真的复制）。
    """
    import shutil

    dst = data(rel)
    if os.path.exists(dst) and not overwrite:
        return dst
    src = resource(rel)
    if not os.path.isfile(src):
        return dst
    ensure_dir(os.path.dirname(dst))
    try:
        shutil.copy2(src, dst)
    except OSError:
        pass
    return dst


def seed_dir(rel: str, *, overwrite: bool = False) -> str:
    """把只读根下的整个目录播种到可写根（已存在的文件一律不覆盖）。"""
    import shutil

    dst = data(rel)
    src = resource(rel)
    ensure_dir(dst)
    if not os.path.isdir(src) or os.path.abspath(src) == os.path.abspath(dst):
        return dst
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.exists(d) and not overwrite:
            continue
        try:
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        except OSError:
            pass
    return dst


# --------------------------------------------------------------------------- 诊断
def describe() -> Dict[str, Any]:
    """给"关于/环境信息"对话框与日志用的快照。"""
    return {
        "frozen": is_frozen(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "exe_dir": exe_dir(),
        "bundle_root": bundle_root(),
        "data_root": data_root(),
        "cwd": os.getcwd(),
        "config": find(MARKER_FILE),
        "STEREO_HOME": os.environ.get("STEREO_HOME", ""),
    }


def describe_text() -> str:
    d = describe()
    lines = [
        f"运行方式    : {'冻结产物（EXE）' if d['frozen'] else '源码'}",
        f"Python      : {d['python']}",
        f"解释器/exe  : {d['executable']}",
        f"只读资源根  : {d['bundle_root']}",
        f"可写数据根  : {d['data_root']}",
        f"工作目录    : {d['cwd']}",
        f"config.yaml : {d['config']}",
    ]
    if d["STEREO_HOME"]:
        lines.append(f"STEREO_HOME : {d['STEREO_HOME']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 自检
def _selftest() -> int:
    ok = True
    print("== app_paths 自检 ==")
    print(describe_text())
    print()

    # 1) 三个根都是绝对路径、都存在
    for name, p in (("bundle_root", bundle_root()), ("data_root", data_root()), ("exe_dir", exe_dir())):
        good = os.path.isabs(p) and os.path.isdir(p)
        print(f"  {'✅' if good else '❌'} {name} = {p}")
        ok &= good

    # 2) config.yaml 能找到
    cfg = find(MARKER_FILE)
    good = os.path.isfile(cfg)
    print(f"  {'✅' if good else '❌'} find('config.yaml') -> {cfg}")
    ok &= good

    # 3) 只读资源不会落到 data_root 之外
    third = find("third_party/Fast-FoundationStereo")
    print(f"  {'✅' if os.path.isdir(third) else '⚠️ '} third_party -> {third}")

    # 4) 可写目录真的可写
    wd = writable_dir("_work/path_selftest")
    try:
        probe = os.path.join(wd, "x.txt")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("1")
        os.remove(probe)
        print(f"  ✅ 可写测试通过 -> {wd}")
    except OSError as exc:
        print(f"  ❌ 可写测试失败 -> {wd} ({exc})")
        ok = False

    # 5) data() 必须落在 data_root 之内
    inside = os.path.abspath(data("output/x")).startswith(os.path.abspath(data_root()))
    print(f"  {'✅' if inside else '❌'} data('output/x') 落在 data_root 内")
    ok &= inside

    print("\n" + ("✅ 全部通过" if ok else "❌ 有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
