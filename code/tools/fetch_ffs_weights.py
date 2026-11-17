r"""tools/fetch_ffs_weights.py —— 下载 Fast-FoundationStereo 官方权重（研究版）
=============================================================================
用途：把官方 Google Drive 上的 23-36-37 研究版 checkpoint 下载到本项目
      weights/23-36-37/ 下（该目录必须同时含 model_best_bp2_serialize.pth 与 cfg.yaml）。

【为什么不用 gdown / 浏览器】
    1) 本项目不想为了下载再引入一个新依赖；
    2) 本机实测：PowerShell/.NET 与 git 的 HTTPS 通道被系统凭据服务限制
       （schannel: SEC_E_NO_CREDENTIALS），但 **Python 自带的 OpenSSL 通道正常**，
       所以用 Python 标准库 urllib 下载最稳。
    3) Google Drive 对大文件会插一层"确认下载"的 HTML 页面，本脚本按官方做法
       解析 confirm 令牌与 uuid 后重新请求，纯标准库实现。

【许可提醒（重要）】
    研究版 checkpoint 适用 NVIDIA Source Code License（**研究与评估用途**）。
    若要商用，请改用 commercial 版本 C-Fast-FoundationStereo
    （HuggingFace: nvidia/c-fast-foundationstereo，需同意 NVIDIA Open Model Agreement）。

用法：
    python tools\fetch_ffs_weights.py                 # 下载到 weights/23-36-37/
    python tools\fetch_ffs_weights.py --dest D:\somewhere\23-36-37
    python tools\fetch_ffs_weights.py --only cfg.yaml # 只下其中一个
    python tools\fetch_ffs_weights.py --list          # 只打印已知的文件 ID 与目标路径

输出：下载完成后自动做完整性检查（文件大小 + torch.load 能否成功 + 打印模型参数键）。
=============================================================================
"""

from __future__ import annotations

import argparse
import http.cookiejar
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 官方 readme 里的 Google Drive 文件夹（父目录 id = 1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap）
# 23-36-37 子目录 id 与文件 id 是实测列出得到的（见 tools/fetch_ffs_weights.py 的说明）
DRIVE_FOLDER_ID = "1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap"
FILES: Dict[str, str] = {
    # 文件名 -> Google Drive 文件 id
    "cfg.yaml": "1GDBRYL-ZaLpXEtWfGFRJvkBc_2sywjgj",
    "model_best_bp2_serialize.pth": "1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E",
}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) stereo_project-fetch/1.0"


def _opener() -> urllib.request.OpenerDirector:
    """带 cookie 的 opener（Google Drive 的确认流程依赖 cookie）。"""
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def _human(n: float) -> str:
    return f"{n / 1024 ** 2:.1f} MB"


def download_drive_file(file_id: str, dest: str, opener: Optional[urllib.request.OpenerDirector] = None) -> Tuple[bool, str]:
    """下载 Google Drive 上的单个文件到 dest。

    输入：
        file_id : Drive 文件 id
        dest    : 本地目标路径（会被覆盖）
        opener  : 可复用的 opener（保留 cookie）
    输出：(成功?, 说明文字)
    """
    opener = opener or _opener()
    base = "https://drive.google.com/uc?export=download&id=" + urllib.parse.quote(file_id)

    def _open(url: str):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        return opener.open(req, timeout=60)

    try:
        resp = _open(base)
    except Exception as exc:  # noqa: BLE001
        return False, f"请求失败：{type(exc).__name__}: {exc}"

    ctype = resp.headers.get("Content-Type", "")
    # 情况一：直接就是文件
    if "text/html" not in ctype:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        return _stream_to_file(resp, dest, total)

    # 情况二：返回的是 HTML —— 可能是"确认下载"页，也可能是"配额超限"页
    html = resp.read().decode("utf-8", "replace")
    if re.search(r"quota|已超出|exceeded", html, re.I) and "confirm" not in html:
        return False, (
            "Google Drive 返回配额超限页面。解决办法：\n"
            "        1) 稍后重试；2) 登录自己的 Google 账号后在浏览器下载；\n"
            "        3) 把文件放到自己的 Drive 再分享下载。"
        )

    # 解析确认表单：**按表单原样重建请求**（不要自己拼字段，实测漏字段会返回 HTTP 400）。
    # 实测该页面的结构为：
    #   <form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
    #     <input type="hidden" name="id"      value="<文件id>">
    #     <input type="hidden" name="export"  value="download">
    #     <input type="hidden" name="confirm" value="t">
    #     <input type="hidden" name="uuid"    value="<uuid>">
    m_form = re.search(r'<form[^>]*action="([^"]+)"[^>]*>(.*?)</form>', html, re.S)
    if not m_form:
        return False, (
            "无法解析 Google Drive 的确认页面（页面结构可能又变了，或该文件需要登录）。\n"
            "        建议：在浏览器里手动下载，放到目标目录即可（文件名必须一致）。"
        )
    action = m_form.group(1).replace("&amp;", "&").strip()
    # 收集表单里所有带 name/value 的字段（含隐藏字段）
    fields = re.findall(r'name="([^"]+)"\s+value="([^"]*)"', m_form.group(2))
    if not fields:
        return False, "确认页面里没有可用的表单字段，无法继续（请在浏览器手动下载）。"
    url2 = action + ("&" if "?" in action else "?") + urllib.parse.urlencode(fields)

    try:
        resp2 = _open(url2)
    except Exception as exc:  # noqa: BLE001
        return False, f"确认后请求失败：{type(exc).__name__}: {exc}\n        请求地址：{url2[:160]}"
    total = int(resp2.headers.get("Content-Length", 0) or 0)
    return _stream_to_file(resp2, dest, total)


def _stream_to_file(resp, dest: str, total: int) -> Tuple[bool, str]:
    """把 HTTP 响应流式写入文件，并打印进度。"""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    got = 0
    t0 = time.perf_counter()
    tmp = dest + ".part"
    try:
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if total > 0:
                    pct = got * 100.0 / total
                    speed = got / max(time.perf_counter() - t0, 1e-6) / 1024 ** 2
                    sys.stdout.write(f"\r    {pct:5.1f}%  {_human(got)} / {_human(total)}  {speed:5.1f} MB/s")
                else:
                    sys.stdout.write(f"\r    {_human(got)} 已下载")
                sys.stdout.flush()
        print()
        os.replace(tmp, dest)
    except Exception as exc:  # noqa: BLE001
        print()
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, f"写入失败：{type(exc).__name__}: {exc}"
    if total > 0 and got < total * 0.999:
        return False, f"下载不完整：只拿到 {_human(got)} / {_human(total)}（网络中断？请重试）"
    return True, f"完成：{dest}（{_human(got)}）"


def verify(dest_dir: str) -> bool:
    """完整性检查：文件存在 + 大小合理 + torch.load 能成功（并打印模型 args 键）。

    为什么值得做：权重下载不完整是最常见的坑，torch.load 一次就能提前暴露问题。
    """
    print("\n" + "=" * 88)
    print("  权重完整性检查")
    print("=" * 88)
    ok = True
    for name in FILES:
        path = os.path.join(dest_dir, name)
        if not os.path.isfile(path):
            print(f"  [缺失] {path}")
            ok = False
            continue
        print(f"  [存在] {name:<32} {_human(os.path.getsize(path))}")

    model_path = os.path.join(dest_dir, "model_best_bp2_serialize.pth")
    if ok and os.path.isfile(model_path):
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        try:
            import torch
            # 关键：官方模型是"整个对象"序列化的，反序列化时必须能 import 到
            # core.foundation_stereo.FastFoundationStereo，即官方仓库根目录要在 sys.path 上。
            # 少了这一步就会报 ModuleNotFoundError: No module named 'core'（实测踩过）。
            repo_root = os.path.join(PROJECT_ROOT, "third_party", "Fast-FoundationStereo")
            if os.path.isdir(repo_root) and repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            t0 = time.perf_counter()
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            dt = time.perf_counter() - t0
            args = getattr(model, "args", None)
            n_param = sum(p.numel() for p in model.parameters()) / 1e6 if hasattr(model, "parameters") else 0
            print(f"  [成功] torch.load 通过（{dt:.2f}s），类型={type(model).__name__}，参数={n_param:.1f}M")
            if hasattr(args, "keys"):
                keys = sorted(args.keys())
                interesting = [k for k in keys if k in ("max_disp", "valid_iters", "cv_group", "normalize",
                                                        "mixed_precision", "corr_radius", "n_gru_layers",
                                                        "hidden_dims", "volume_dim", "low_memory")]
                print(f"         模型参数字段(关键项): " + ", ".join(f"{k}={args[k]}" for k in interesting))
        except Exception as exc:  # noqa: BLE001
            print(f"  [失败] torch.load 报错：{type(exc).__name__}: {exc}")
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 FFS 官方研究版权重（Google Drive）")
    ap.add_argument("--dest", default=os.path.join(PROJECT_ROOT, "weights", "23-36-37"),
                    help="目标目录（默认 weights/23-36-37）")
    ap.add_argument("--only", default="", help="只下载指定文件名")
    ap.add_argument("--list", action="store_true", help="只打印文件 ID 与目标路径")
    ap.add_argument("--skip-verify", action="store_true", help="跳过下载后的完整性检查")
    args = ap.parse_args()

    print("=" * 88)
    print("  Fast-FoundationStereo 研究版权重下载（23-36-37）")
    print("=" * 88)
    print(f"  来源文件夹 : https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}")
    print(f"  目标目录   : {args.dest}")
    print("  许可提醒   : 研究版适用 NVIDIA Source Code License（研究/评估用途）；商用请用 C-Fast-FoundationStereo")
    for name, fid in FILES.items():
        print(f"    {name:<32} id={fid}")

    if args.list:
        return 0

    targets = [args.only] if args.only else list(FILES.keys())
    opener = _opener()
    all_ok = True
    for name in targets:
        if name not in FILES:
            print(f"  [跳过] 未知文件名：{name}")
            all_ok = False
            continue
        dest = os.path.join(args.dest, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print(f"\n  [已存在] {dest}（{_human(os.path.getsize(dest))}），跳过下载")
            continue
        print(f"\n  正在下载 {name} …")
        ok, msg = download_drive_file(FILES[name], dest, opener)
        print(f"    {msg}")
        all_ok = all_ok and ok
        if not ok:
            print("    提示：若本机无法下载，可在浏览器里手动下载后放到该目录，文件名必须一致。")

    if not args.skip_verify and all_ok:
        verify(args.dest)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
