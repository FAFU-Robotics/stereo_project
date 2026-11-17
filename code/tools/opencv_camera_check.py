"""opencv_camera_check.py —— 最小化 · 独立的 OpenCV 相机可用性自检
=============================================================================
这个程序的唯一目的：
    用**最纯粹的 OpenCV**（不加任何项目封装）去打开摄像头，看看到底行不行，
    并把"为什么不行"的信息一次打印清楚。

为什么要有它（本机背景）：
    2026-09-30 曾出现"物理相机只对管理员可用"的现象；2026-10-21 复核发现
    **普通用户（Medium 完整性）已经可以正常打开**，真正会被拒绝的是
    **低完整性 / 沙箱 / 受限会话**里的进程（E_ACCESSDENIED）。
    本工具刻意不用项目的任何模块，只用 cv2 + numpy + 系统自带模块，
    用来快速判断"当前这个终端/进程"能不能打开相机，以及是哪个索引/后端。
    注意：索引 2 = HP 虚拟摄像头，isOpened()/read() 都返回 True，但画面全黑，
    不能拿它当"双目相机正常"的证据；验收请用索引 1。

它会做这些事：
    1) 打印**自己是否以管理员身份运行**（仅供参考：普通用户不需要管理员）；
    2) 打印 Python / OpenCV 版本、解释器路径；
    3) 从注册表读取在线的 UVC 相机名字（只读，用来和索引交叉印证）；
    4) 对每个索引 × 每种后端(dshow/msmf) 尝试打开，成功则：
         读取 N 帧 → 统计真实分辨率/帧率/像素格式 → 判断"真实画面/全黑/无信号"
         → 有真实画面时保存一张快照作为证据；
    5) 最后给出**总结表 + 明确的结论与下一步建议**。

用法（把 D:\\31931\\Documents\\python312\\python.exe 当解释器）：
    python tools\\opencv_camera_check.py                     # 默认扫描索引 0~2，两种后端
    python tools\\opencv_camera_check.py --quick             # 快速全扫 0~9（原来的 Camara_test.py 已并入此模式）
    python tools\\opencv_camera_check.py --index 1           # 只测索引 1
    python tools\\opencv_camera_check.py --max-index 4       # 扫描 0~4
    python tools\\opencv_camera_check.py --backends dshow    # 只用 DirectShow
    python tools\\opencv_camera_check.py --frames 30 --res 1280x480 --fourcc MJPG
    python tools\\opencv_camera_check.py --open-params       # 改成"打开时带分辨率/格式"

关于 --quick 模式（与用户原来的 Camara_test.py 的关系）：
    Camara_test.py 做的是"0~9 全部索引、只用 DSHOW、每路读 1 帧、一行一个结果"，
    本工具的 --quick **完全覆盖**这套行为，并额外提供三件它没有的东西：
      ① 打印【是否以管理员身份运行】（仅参考，普通用户不需要管理员）；
      ② 把结果翻译成人话（打不开 / 能打开但读不到帧 / 全黑 / 拿到真实画面）；
      ③ 结尾给出总结与下一步建议。
    该脚本已并入本工具，不再作为独立文件存在（如需独立版本，本文件就是）。

如何做对照实验（关键）：
    ① 普通权限：直接双击 tools\\相机OpenCV自检.bat（推荐，也是验收方式）
    ② 管理员权限：仅当普通权限被拒、怀疑是受限会话时，才用 tools\\启动_管理员.bat → 菜单 [10]
    如果普通权限和本工具都拿不到图，先确认当前终端是不是沙箱/低完整性，
    而不是直接改用管理员运行。

退出码：0=至少一路拿到真实画面；3=能打开但全是黑屏/无信号；5=没有任何一路能打开。
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 与项目一致的"无有效信号"判据（来自阶段 1 实测：真画面 mean≈95~107、std≈27~39）
NO_SIGNAL_MEAN = 5.0
NO_SIGNAL_STD = 12.0

BACKENDS: List[Tuple[str, int]] = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)]


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------
def is_admin() -> bool:
    """返回当前进程是否以管理员（高完整性）身份运行。

    仅作为诊断信息：本机普通用户（Medium）已经能打开相机，不需要管理员。
    真正影响物理相机访问的是"低完整性 / 沙箱 / 受限会话"，不是 UAC 提权。
    """
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def fourcc_of(cap: cv2.VideoCapture) -> str:
    """把 CAP_PROP_FOURCC 的数字还原成可读的四个字符（读不到返回 '?'）。"""
    try:
        v = int(cap.get(cv2.CAP_PROP_FOURCC))
    except Exception:
        return "?"
    if v <= 0:
        return "?"
    chars = [chr((v >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in chars)


def list_camera_names_from_registry() -> List[str]:
    """从注册表读在线 UVC 相机名字（只读，失败返回空列表）。

    注意：注册表里的顺序**不等于** OpenCV 的索引顺序，这里只用于"这台机器上有哪些相机"
    的交叉印证，不要据此推断索引。
    """
    names: List[str] = []
    try:
        import winreg

        root = r"SYSTEM\CurrentControlSet\Enum\USB"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root) as usb_root:
            for i in range(winreg.QueryInfoKey(usb_root)[0]):
                vid_pid = winreg.EnumKey(usb_root, i)
                try:
                    with winreg.OpenKey(usb_root, vid_pid) as vk:
                        for j in range(winreg.QueryInfoKey(vk)[0]):
                            inst = winreg.EnumKey(vk, j)
                            with winreg.OpenKey(vk, inst) as ik:
                                vals = {
                                    winreg.EnumValue(ik, k)[0]: winreg.EnumValue(ik, k)[1]
                                    for k in range(winreg.QueryInfoKey(ik)[1])
                                }
                            if str(vals.get("Service", "")).lower() != "usbvideo":
                                continue
                            name = str(vals.get("FriendlyName", "") or vals.get("DeviceDesc", ""))
                            if ";" in name:
                                name = name.split(";")[-1]
                            if name and name not in names:
                                names.append(name)
                except OSError:
                    continue
    except Exception:
        pass
    return names


def open_camera(
    index: int,
    backend: int,
    req_w: int,
    req_h: int,
    fourcc: str,
    with_params: bool,
) -> Tuple[Optional[cv2.VideoCapture], str]:
    """尝试打开一个相机索引，返回 (cap 或 None, 说明文字)。

    with_params=True  —— 在**打开时**就把分辨率/格式传进去（对只声明宽格式的相机更可靠）
    with_params=False —— 先打开，再用 cap.set() 设置（OpenCV 最常见用法）
    """
    params: List[int] = []
    if with_params:
        if req_w > 0:
            params += [cv2.CAP_PROP_FRAME_WIDTH, req_w]
        if req_h > 0:
            params += [cv2.CAP_PROP_FRAME_HEIGHT, req_h]
        if fourcc:
            params += [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)]

    t0 = time.perf_counter()
    try:
        cap = cv2.VideoCapture(index, backend, params) if params else cv2.VideoCapture(index, backend)
    except Exception as exc:  # 极少见：驱动层直接抛异常
        return None, f"构造 VideoCapture 抛出异常: {exc}"
    dt = (time.perf_counter() - t0) * 1000.0

    if not cap.isOpened():
        cap.release()
        return None, f"打不开（耗时 {dt:.0f} ms）"

    if not with_params:
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if req_w > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        if req_h > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 只保留最新帧，避免读到旧数据
    except Exception:
        pass
    return cap, f"已打开（耗时 {dt:.0f} ms）"


def grab_frames(cap: cv2.VideoCapture, n: int) -> Dict[str, object]:
    """连读 n 帧并统计。返回统计字典（不抛异常）。"""
    ok_n = fail_n = blank_n = nosig_n = 0
    real_frames: List[np.ndarray] = []
    w = h = 0
    t0 = time.perf_counter()
    for _ in range(n):
        ok, frame = cap.read()
        if not ok or frame is None:
            fail_n += 1
            continue
        ok_n += 1
        h, w = frame.shape[:2]
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, std = float(gray.mean()), float(gray.std())
        if mean < 1.0 and std < 1.0:
            blank_n += 1
        elif mean < NO_SIGNAL_MEAN and std < NO_SIGNAL_STD:
            nosig_n += 1
        else:
            real_frames.append(frame)
    dt = max(time.perf_counter() - t0, 1e-9)

    detail = ""
    if real_frames:
        g = cv2.cvtColor(real_frames[-1], cv2.COLOR_BGR2GRAY)
        detail = f"灰度 mean={g.mean():.1f} std={g.std():.1f} 唯一灰度={len(np.unique(g))}"

    return {
        "read_ok": ok_n,
        "read_fail": fail_n,
        "blank": blank_n,
        "nosignal": nosig_n,
        "real": len(real_frames),
        "fps": ok_n / dt,
        "w": w,
        "h": h,
        "last_real": real_frames[-1] if real_frames else None,
        "detail": detail,
    }


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="OpenCV 相机可用性自检（独立、最小化）")
    p.add_argument("--index", type=int, default=-1, help="只测这一个索引（默认 -1 = 扫描一段）")
    p.add_argument("--max-index", type=int, default=2, help="扫描范围 0..N（默认 2）")
    p.add_argument("--backends", default="dshow,msmf", help="要测的后端，逗号分隔：dshow,msmf")
    p.add_argument("--frames", type=int, default=20, help="每个组合尝试读取的帧数（默认 20）")
    p.add_argument("--res", default="1280x480", help="请求分辨率 WxH（默认 1280x480，即本机双目拼接尺寸）")
    p.add_argument("--fourcc", default="MJPG", help="请求像素格式（默认 MJPG；空字符串=不设置）")
    p.add_argument("--open-params", action="store_true", help="改为在打开时传入分辨率/格式")
    p.add_argument("--save-dir", default="", help="快照保存目录（默认 output/probe）")
    # --quick：快速全扫模式（等价于用户原来的 Camara_test.py：0~9 全部索引、只用 DSHOW、每路读 1 帧、一行一个结果）
    p.add_argument("--quick", action="store_true",
                   help="快速全扫：索引 0~9、只用 dshow、每路读 1 帧、每个索引一行结果"
                        "（原来的 Camara_test.py 已并入本模式；区别是本工具会先打印是否以管理员运行。"
                        "与 --index 同时使用时以 --quick 为准）")
    args = p.parse_args(argv)

    if args.quick:
        # 快速模式固定这些参数：与 Camara_test.py 的行为保持一致，但换成更清晰的输出
        args.max_index = 9
        args.index = -1
        args.backends = "dshow"
        args.frames = 1

    try:
        req_w, req_h = (int(x) for x in str(args.res).lower().split("x"))
    except Exception:
        print(f"[参数错误] --res 应形如 1280x480，当前是 {args.res!r}")
        return 1

    # 中文 Windows 控制台默认 GBK；✅/❌ 等字符可能触发 UnicodeEncodeError。
    # 保留当前编码，只把无法编码的字符替换掉，确保诊断脚本不会因打印而崩溃。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

    admin = is_admin()
    print("=" * 88)
    print("  OpenCV 相机可用性自检（独立最小化版本）" + ("  [--quick 快速全扫 0~9]" if args.quick else ""))
    print("=" * 88)
    print(f"  管理员令牌      : {'是（高完整性）' if admin else '否（普通用户）'}   <-- 仅参考，普通用户无需")
    print(f"  Python          : {sys.version.split()[0]}   {sys.executable}")
    print(f"  OpenCV          : {cv2.__version__}")
    print(f"  NumPy           : {np.__version__}")
    print(f"  请求参数        : {req_w}x{req_h} {args.fourcc or '(不设置格式)'}  "
          f"打开方式={'打开时传参' if args.open_params else '打开后 cap.set'}")
    names = list_camera_names_from_registry()
    print(f"  注册表相机名字  : {', '.join(names) if names else '（读不到）'}")
    print("  提示            : 注册表顺序 ≠ OpenCV 索引顺序，名称仅用于交叉印证")
    print("  说明            : OpenCV 自己可能会打印 WARN:0 / ERROR:0 之类的信息，")
    print("                    那是它内部的诊断输出（不隐藏），不影响本报告结论。")
    print("-" * 88)

    want = [b.strip().lower() for b in str(args.backends).split(",") if b.strip()]
    chosen = [(n, c) for n, c in BACKENDS if n in want]
    if not chosen:
        print(f"[参数错误] --backends 只支持 dshow / msmf，当前 {args.backends!r}")
        return 1

    indices = [args.index] if args.index >= 0 else list(range(0, args.max_index + 1))
    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "probe"
    )

    results: List[Dict[str, object]] = []
    for idx in indices:
        for bname, bcode in chosen:
            line = f"  索引 {idx} [{bname:5s}] "
            cap, msg = open_camera(idx, bcode, req_w, req_h, args.fourcc, args.open_params)
            if cap is None:
                print(line + "❌ " + msg)
                results.append({"index": idx, "backend": bname, "opened": False, "msg": msg})
                continue

            st = grab_frames(cap, args.frames)
            real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            fcc = fourcc_of(cap)
            cap.release()

            wh = f"{real_w}x{real_h}" if real_w > 0 and real_h > 0 else "?x?"
            verdict = ("✅ 拿到真实画面" if st["real"] > 0 else
                       ("⚠️ 能打开但全是黑屏" if st["blank"] > 0 else
                        ("⚠️ 能打开但无有效信号" if st["read_ok"] > 0 else "⚠️ 能打开但读不到帧")))

            if args.quick:
                # 快速模式：一行一个索引（Camara_test.py 的风格），但把关键信息都带上
                brief = (f"{wh} {fcc} {st['fps']:.1f}fps" if st["real"] > 0
                         else f"{wh} {fcc} 成功{st['read_ok']}/{args.frames}")
                print(f"  索引 {idx} [{bname:5s}] {verdict}   {brief}")
                results.append({"index": idx, "backend": bname, "opened": True,
                                "real": st["real"], "w": real_w, "h": real_h, "fcc": fcc,
                                "fps": st["fps"], "saved": ""})
                continue

            print(line + f"{msg}  {wh} {fcc} 标称{fps:.1f}fps  "
                         f"实测{st['fps']:.1f}fps  成功{st['read_ok']}/{args.frames} "
                         f"失败{st['read_fail']} 黑{st['blank']} 无信号{st['nosignal']}")
            print(f"  {'':16s}{verdict}" + (f"   {st['detail']}" if st["detail"] else ""))

            # 读不到帧 + 我们请求过格式/分辨率 -> 退回"默认设置"再试一次，
            # 用来区分"格式不匹配"和"设备本身不出流"（实测虚拟摄像头属于前者）
            # 快速模式跳过这一步：它追求"一行一个索引"的简洁输出
            if (not args.quick) and st["read_ok"] == 0 and (req_w > 0 or args.fourcc):
                cap2, msg2 = open_camera(idx, bcode, 0, 0, "", False)
                if cap2 is not None:
                    st2 = grab_frames(cap2, args.frames)
                    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    f2 = fourcc_of(cap2)
                    cap2.release()
                    v2 = ("✅ 拿到真实画面" if st2["real"] > 0 else
                          ("⚠️ 全是黑屏" if st2["blank"] > 0 else
                           ("⚠️ 无有效信号" if st2["read_ok"] > 0 else "⚠️ 仍读不到帧")))
                    wh2 = f"{w2}x{h2}" if w2 > 0 and h2 > 0 else "?x?"
                    print(f"  {'':16s}↻ 退回默认设置重试: {wh2} {f2}  "
                          f"成功{st2['read_ok']}/{args.frames} 失败{st2['read_fail']} "
                          f"黑{st2['blank']} 无信号{st2['nosignal']}  -> {v2}")
                    if st2["real"] > 0 and st2["last_real"] is not None:
                        try:
                            os.makedirs(save_dir, exist_ok=True)
                            sv = os.path.join(save_dir, f"opencv_check_idx{idx}_{bname}_default.png")
                            cv2.imwrite(sv, st2["last_real"])
                            print(f"  {'':16s}已保存快照: {sv}")
                        except Exception:
                            pass
                else:
                    print(f"  {'':16s}↻ 退回默认设置重试: {msg2}")

            saved = ""
            if st["real"] > 0 and st["last_real"] is not None:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    saved = os.path.join(save_dir, f"opencv_check_idx{idx}_{bname}.png")
                    cv2.imwrite(saved, st["last_real"])
                    print(f"  {'':16s}已保存快照: {saved}")
                except Exception as exc:  # noqa: BLE001
                    saved = f"（保存失败: {exc}）"
            results.append({"index": idx, "backend": bname, "opened": True,
                            "real": st["real"], "w": real_w, "h": real_h, "fcc": fcc,
                            "fps": st["fps"], "saved": saved})

    # ---------------------------------------------------------------- 总结
    print("=" * 88)
    ok_any = [r for r in results if r.get("real")]
    opened_any = [r for r in results if r.get("opened")]
    print("  总结")
    print(f"    尝试组合数        : {len(results)}")
    print(f"    能打开的          : {len(opened_any)}")
    print(f"    拿到真实画面的    : {len(ok_any)}")
    for r in ok_any:
        print(f"      ✅ 索引 {r['index']} [{r['backend']}]  {r['w']}x{r['h']} {r['fcc']}  {r['fps']:.1f} fps")
    print("-" * 88)

    if ok_any:
        print("  【结论】OpenCV 能正常取到真实画面，相机链路可用。")
        print("         普通用户权限即可，不需要管理员；可在普通 PowerShell 里复跑确认。")
        print("         验收请以索引 1（DECXIN）的真实画面为准；索引 2 全黑不算通过。")
        return 0

    if opened_any:
        print("  【结论】能打开设备，但拿不到有效画面（黑屏/无信号）。")
        print("         本机索引 2 = HP 虚拟摄像头：isOpened()/read() 都返回 True，但画面全黑；")
        print("         请改用索引 1（DECXIN）复测。其他原因：格式/分辨率不匹配、镜头盖、光照不足。")
        return 3

    print("  【结论】所有组合都打不开相机。")
    print("         先确认当前进程不是低完整性 / 沙箱 / 受限会话 ——")
    print("         本机只有这种会话会被 Windows 以 E_ACCESSDENIED 拒绝。")
    print("         请在普通桌面 PowerShell 里复跑本工具；仍失败再检查：")
    print("         设备是否在位、是否被别的程序占用、索引是否因拔插变化。")
    print("         不建议直接把“管理员运行”当作解决方案。")
    return 5


if __name__ == "__main__":
    sys.exit(main())
