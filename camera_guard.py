"""camera_guard.py —— 相机占用重试 + 单实例锁（稳定性加固 ②）

=============================================================================
【要解决的两个真实故障】
    1) 相机被别的程序占用（RealSense Viewer / 另一个 main.py）时，
       cv2.VideoCapture 直接失败，原来只报一句含糊的错误，用户不知道该怎么办。
    2) 同一台相机只能被一个进程打开；第二个进程去抢，报错同样难以理解，
       而且两边都"时好时坏"。

【做法】
    · open_with_retry(): 打开失败重试 N 次（默认 5 次 × 300 ms 退避），
      最终失败时抛 CameraBusyError，消息里直接列出常见原因和排查动作。
    · acquire_camera_lock() / release_camera_lock(): 在 output/.camera.lock 里写 PID；
      文件已存在时判断那个 PID 是否还活着：
          - 活着  -> 抛 CameraBusyError("另一个实例正在使用相机（PID=…）")
          - 已死  -> 视为陈旧锁，接管
      判断存活用 Windows API OpenProcess(SYNCHRONIZE)，**绝不能**用 os.kill(pid, 0)
      —— 在 Windows 上 os.kill 会真的去终止那个进程（踩过这个概念坑）。
    · 锁文件写入 PID + 启动时间，便于人工排查。

【为什么单独一个模块】
    camera.py 是核心模块，尽量不动；这里只做"包一层"，
    Session 想用就调，不用就完全不受影响。
=============================================================================
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

_LOCK_PATH: Optional[str] = None


class CameraBusyError(RuntimeError):
    """相机打不开或被占用（消息本身是给用户看的）。"""


def _default_lock_path() -> str:
    # 必须落在**可写**的数据根：打包成 onefile 后 __file__ 位于 _MEIPASS 临时目录，
    # 那里退出即删且不可写，锁文件写不进去会变成"永远抢不到锁"。
    try:
        from app_paths import data_root
        root = data_root()
    except Exception:                     # noqa: BLE001 —— 单独拿本模块用时仍可跑
        root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "output", ".camera.lock")


def _pid_alive(pid: int) -> bool:
    """Windows 下安全的进程存活判断（不使用 os.kill）。"""
    if pid <= 0:
        return False
    try:
        import ctypes
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if not h:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 0x102  # WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    except Exception:  # noqa: BLE001
        return False


def _read_lock(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("pid="):
                    return int(line[4:])
    except (OSError, ValueError):
        return None
    return None


def acquire_camera_lock(lock_path: Optional[str] = None, wait_s: float = 0.0) -> str:
    """获取单实例锁；失败抛 CameraBusyError（消息可直接展示给用户）。"""
    global _LOCK_PATH
    path = lock_path or _default_lock_path()
    _LOCK_PATH = path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    deadline = time.perf_counter() + max(wait_s, 0.0)
    while True:
        pid = _read_lock(path) if os.path.exists(path) else None
        if pid is not None and not _pid_alive(pid):
            try:
                os.remove(path)          # 陈旧锁：上一个进程已退出
            except OSError:
                pass
            pid = None
        if pid is None:
            try:
                with open(path, "x", encoding="utf-8") as fh:   # 'x' = 不存在才创建
                    fh.write(f"pid={os.getpid()}\nstarted={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                return path
            except FileExistsError:
                pass                  # 竞态：别人刚建了，下一轮再判断
        elif time.perf_counter() < deadline:
            time.sleep(0.1)
            continue
        raise CameraBusyError(
            f"另一个程序正在使用这台相机（PID={pid}）。\n"
            f"  处理：关掉其它相机程序（RealSense Viewer、另一个 main.py、会议软件），"
            f"或等它退出后重试。\n"
            f"  若确认没有其它程序，可删除锁文件后重试：{path}")


def release_camera_lock() -> None:
    """释放锁（幂等）。只删自己建的锁。"""
    path = _LOCK_PATH or _default_lock_path()
    try:
        if os.path.exists(path) and _read_lock(path) in (os.getpid(), None):
            os.remove(path)
    except OSError:
        pass


def lock_held_by_self(lock_path: Optional[str] = None) -> bool:
    """当前进程是不是已经持有相机锁。

    启动时 prime() 会先抢锁并打开相机，随后 start() 再进来不能再抢一次：
    acquire 对同一 PID 会故意失败。这里让第二次调用发现「锁就是我的」然后继续。
    """
    path = lock_path or _LOCK_PATH or _default_lock_path()
    if not path or not os.path.exists(path):
        return False
    return _read_lock(path) == os.getpid()


@contextmanager
def camera_lock(lock_path: Optional[str] = None) -> Iterator[str]:
    try:
        yield acquire_camera_lock(lock_path)
    finally:
        release_camera_lock()


def open_with_retry(cam: Any, retries: int = 5, delay: float = 0.3) -> float:
    """打开相机，失败重试。返回总耗时（秒）；最终失败抛 CameraBusyError。"""
    last: Optional[BaseException] = None
    t0 = time.perf_counter()
    for i in range(max(int(retries), 1)):
        try:
            cam.open()
            return time.perf_counter() - t0
        except BaseException as exc:  # noqa: BLE001 —— 相机层异常类型不统一
            last = exc
            if i + 1 < retries:
                time.sleep(max(delay, 0.0))
    raise CameraBusyError(
        f"打不开相机（已重试 {retries} 次）：{type(last).__name__}: {last}\n"
        f"  按这个顺序排查：\n"
        f"   ① 关掉所有可能占用相机的程序（RealSense Viewer / 另一个 main.py / 会议软件）\n"
        f"   ② 拔插一次 USB，确认设备管理器里没有黄色感叹号\n"
        f"   ③ 确认本程序的权限正常（曾遇到 python.exe 被标成 Low 完整性级别导致拒绝访问）\n"
        f"   ④ 换一个后端试试：config.yaml 的 camera.backend 改为 msmf / dshow / any")


# -----------------------------------------------------------------------------
def _selftest() -> int:
    import sys
    import tempfile

    root = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(root, "_work", f"locktest_{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    lock = os.path.join(d, ".camera.lock")
    ok = True

    print("=" * 68)
    print("  camera_guard 自检")
    print("=" * 68)

    # ① 首次获取成功
    acquire_camera_lock(lock)
    print("  ✅ ① 首次获取锁成功")
    # ② 重复获取应失败（同进程也要能识别"已被占用"）
    try:
        acquire_camera_lock(lock)
        print("  ❌ ② 重复获取锁竟然成功了")
        ok = False
    except CameraBusyError as exc:
        print(f"  ✅ ② 重复获取被拒绝：{str(exc).splitlines()[0][:52]}…")
    release_camera_lock()
    print("  ✅ ③ 释放后锁文件已删除" if not os.path.exists(lock) else "  ❌ ③ 释放后锁仍在")
    # ④ 陈旧锁（一个肯定不存在的 PID）应被接管
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write("pid=999999\n")
    try:
        acquire_camera_lock(lock)
        print("  ✅ ④ 陈旧锁被接管")
        release_camera_lock()
    except CameraBusyError:
        print("  ❌ ④ 陈旧锁没有被接管")
        ok = False
    # ⑤ 打开重试：前 2 次失败、第 3 次成功
    class FlakyCam:
        def __init__(self):
            self.n = 0

        def open(self):
            self.n += 1
            if self.n < 3:
                raise OSError("device busy (simulated)")

    fc = FlakyCam()
    cost = open_with_retry(fc, retries=5, delay=0.01)
    if fc.n == 3:
        print(f"  ✅ ⑤ 重试生效：第 3 次成功（耗时 {cost * 1000:.0f} ms）")
    else:
        print(f"  ❌ ⑤ 重试次数异常：{fc.n}")
        ok = False
    # ⑥ 全部失败时抛出可读错误
    class DeadCam:
        def open(self):
            raise OSError("always fail")

    try:
        open_with_retry(DeadCam(), retries=2, delay=0.01)
        print("  ❌ ⑥ 全失败却没有抛异常")
        ok = False
    except CameraBusyError as exc:
        msg = str(exc)
        good = ("重试 2 次" in msg) and ("RealSense Viewer" in msg) and ("msmf" in msg)
        print("  ✅ ⑥ 全失败抛出可读错误（含重试次数 + 排查步骤）" if good else "  ❌ ⑥ 错误消息不够可读")
        ok = ok and good

    import shutil
    shutil.rmtree(d, ignore_errors=True)
    print("\n  " + ("✅ 自检全部通过" if ok else "❌ 自检有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
