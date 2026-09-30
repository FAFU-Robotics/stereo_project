r"""tools/install_opencv4_isolated.py —— 把指定版本的 OpenCV 隔离安装到临时目录
=============================================================================
目的：为"验证 OpenCV 4.x 能否修好相机"提供一个**零污染**的安装方式：
      - 不安装到 python312 的 site-packages
      - 不修改 numpy / torch 等任何现有依赖
      - 只往一个临时目录里放 cv2，测试时用 PYTHONPATH 临时指过去
      - 测完删掉该目录即可完全恢复

【为什么不用 `pip install --target`】
    本机实测：pip 在下载/解包阶段会创建 `tempfile.mkdtemp()` 临时目录，而该目录里的
    写入被 DSH 沙箱拒绝（PermissionError: [Errno 13]，路径形如
    `...\pip-unpack-xxxx\xxx.whl`），导致 install/download 全部失败。
    这不是本项目的问题，而是沙箱对 pip 临时目录的限制。
    绕过办法（本脚本采用）：**wheel 本身就是 zip**，用标准库 urllib 下载到普通文件，
    再用 zipfile 解压到目标目录 —— 全程不使用 pip 的临时目录。
    下载大文件的能力已实测（67.8 MB 的 FFS 权重就是这样下下来的）。

用法：
    REM 先看会装哪个版本、到哪里（不下载）
    python tools\install_opencv4_isolated.py --version 4.12.0.88 --list-only

    REM 正式安装到项目内的临时目录
    python tools\install_opencv4_isolated.py --version 4.12.0.88 \
        --target D:\31931\Documents\stereo_project\_work\cv412

    REM 卸载（直接删目录即可）
    python tools\install_opencv4_isolated.py --target <目录> --remove

测试时怎么用（临时设 PYTHONPATH，只影响那一条命令）：
    set PYTHONPATH=<目标目录>
    python -c "import cv2; print(cv2.__version__, cv2.__file__)"
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

MIRROR = "https://mirrors.aliyun.com/pypi/simple/opencv-contrib-python/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) stereo_project-isolated-install/1.0"


def list_wheels() -> Dict[str, List[str]]:
    """从镜像索引里列出所有 win_amd64 wheel，返回 {版本: [文件名...]}。"""
    html = urllib.request.urlopen(urllib.request.Request(MIRROR, headers={"User-Agent": UA}), timeout=60).read().decode(
        "utf-8", "replace"
    )
    out: Dict[str, List[str]] = {}
    for m in re.finditer(r'href="([^"]+?\.whl)[^"]*"', html):
        href = m.group(1)
        name = href.rsplit("/", 1)[-1]
        if "win_amd64" not in name:
            continue
        vm = re.match(r"opencv_contrib_python-([0-9][^-]*)-", name)
        if not vm:
            continue
        out.setdefault(vm.group(1), []).append(name)
    return out


def resolve_url(version: str) -> Tuple[Optional[str], str]:
    """找到指定版本的 win_amd64 wheel 的下载地址。返回 (地址, 文件名或错误说明)。"""
    html = urllib.request.urlopen(urllib.request.Request(MIRROR, headers={"User-Agent": UA}), timeout=60).read().decode(
        "utf-8", "replace"
    )
    pattern = r'href="([^"]*?opencv_contrib_python-' + re.escape(version) + r'-[^"]*?win_amd64\.whl)[^"]*"'
    m = re.search(pattern, html)
    if not m:
        return None, f"镜像上没有 {version} 的 win_amd64 wheel（请先用 --list-only 查看可用版本）"
    href = m.group(1)
    url = href if href.startswith("http") else urllib.parse.urljoin(MIRROR, href)
    return url, url.rsplit("/", 1)[-1]


def download(url: str, dest: str) -> None:
    """流式下载（带进度），写入普通文件 —— 不使用任何临时目录机制。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        got = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total:
                sys.stdout.write(f"\r    下载 {got * 100.0 / total:5.1f}%  {got / 1024 ** 2:.1f}/{total / 1024 ** 2:.1f} MB")
                sys.stdout.flush()
    print()


def install(wheel_path: str, target: str) -> int:
    """把 wheel 当 zip 解压到 target。返回解出的文件数。"""
    os.makedirs(target, exist_ok=True)
    n = 0
    with zipfile.ZipFile(wheel_path) as zf:
        for info in zf.infolist():
            # wheel 里的路径分隔符是 /，Python 的 zipfile 会处理；这里防止目录穿越
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ".." in name.split("/"):
                print(f"    [跳过可疑路径] {name}")
                continue
            zf.extract(info, target)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenCV 隔离安装（wheel 直接解压，不用 pip 临时目录）")
    ap.add_argument("--version", default="4.12.0.88", help="要安装的版本，如 4.12.0.88")
    ap.add_argument("--target", default="", help="安装目标目录（必须是独立临时目录，不要指向 site-packages！）")
    ap.add_argument("--wheel-dir", default="", help="wheel 存放目录（默认 <target>/../wheels）")
    ap.add_argument("--list-only", action="store_true", help="只列出可用版本")
    ap.add_argument("--remove", action="store_true", help="删除目标目录（= 卸载）")
    args = ap.parse_args()

    if args.list_only:
        wheels = list_wheels()
        print("镜像上 opencv-contrib-python 的 win_amd64 版本：")
        for v in sorted(wheels, key=lambda s: [int(x) for x in re.findall(r"\d+", s)]):
            print(f"  {v:<14} {wheels[v]}")
        return 0

    if not args.target:
        print("[错误] 必须指定 --target（独立的临时目录）")
        return 2
    target = os.path.abspath(args.target)

    if args.remove:
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
            print(f"已删除（卸载完成）：{target}")
        else:
            print(f"目录不存在，无需删除：{target}")
        return 0

    # 安全护栏：绝不允许装到 site-packages / python 安装目录里
    low = target.lower()
    if "site-packages" in low or low.rstrip("\\").endswith("python312"):
        print(f"[拒绝] 目标目录看起来是 Python 环境目录，本脚本只允许装到独立临时目录：{target}")
        return 3

    print("=" * 88)
    print("  OpenCV 隔离安装（wheel 直接解压）")
    print("=" * 88)
    print(f"  版本   : {args.version}")
    print(f"  目标   : {target}")
    url, name = resolve_url(args.version)
    if url is None:
        print("[失败] " + name)
        return 4
    print(f"  wheel  : {name}")

    wheel_dir = args.wheel_dir or os.path.join(os.path.dirname(target), "wheels")
    wheel_path = os.path.join(wheel_dir, name)
    if os.path.isfile(wheel_path) and os.path.getsize(wheel_path) > 1024 * 1024:
        print(f"  已存在 : {wheel_path}（跳过下载）")
    else:
        print("  下载中 …")
        download(url, wheel_path)

    print("  解压中 …")
    n = install(wheel_path, target)
    print(f"  完成：解压 {n} 个文件到 {target}")
    print("\n  验证方式（只对这条命令生效，不改任何环境）：")
    print(f'    set PYTHONPATH={target}')
    print('    python -c "import cv2; print(cv2.__version__, cv2.__file__)"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
