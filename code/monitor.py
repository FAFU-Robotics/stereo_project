"""monitor.py —— 内存 / 显存采样（稳定性加固 ④）

=============================================================================
【为什么需要】
    10 分钟长跑只证明了"帧率不衰减、没有断流"，但**说不出内存有没有泄漏**。
    原因是没做资源采样。本模块提供两个数，配合 profiler.to_csv() 就能画出
    长跑曲线，把性能报告第 7 节那三条"未测"补掉。

【两个数的取法】
    · rss_mb  : 进程常驻内存。优先 psutil；没装就用 Windows API
                (psapi.GetProcessMemoryInfo)，纯 ctypes、无额外依赖。
    · vram_mb : torch 已分配显存。没装 torch / 没有 CUDA 时返回 nan（不报错）。

【设计原则】
    · 采样本身必须**极便宜**（每 5 秒一次，绝不能拖慢主循环）
    · 任何异常都吞掉并返回 nan —— 监控代码不允许拖垮主程序
=============================================================================
"""

from __future__ import annotations

import math
import os
from typing import Dict

_PID = os.getpid()


def rss_mb() -> float:
    """当前进程常驻内存（MB）；取不到返回 nan。"""
    # 1) psutil（最省事）
    try:
        import psutil  # type: ignore
        return float(psutil.Process(_PID).memory_info().rss) / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    # 2) Windows API（ctypes 兜底，无需任何依赖）
    try:
        import ctypes
        import ctypes.wintypes as wt

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(pmc), pmc.cb):
            return float(pmc.WorkingSetSize) / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def vram_mb() -> float:
    """torch 当前已分配显存（MB）；无 torch / 无 CUDA 返回 nan。"""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated()) / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def peak_vram_mb() -> float:
    """torch 峰值显存（MB）；取不到返回 nan。"""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 1048576.0
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def _r1(v: float) -> float:
    return v if (isinstance(v, float) and math.isnan(v)) else round(float(v), 1)


def sample() -> Dict[str, float]:
    """一行采样结果，可直接塞进 profiler.to_csv(extra=...)。"""
    return {"rss_mb": _r1(rss_mb()), "vram_mb": _r1(vram_mb())}


def _selftest() -> int:
    import sys
    import time
    print("=" * 64)
    print("  monitor 自检")
    print("=" * 64)
    a = rss_mb()
    print(f"  rss_mb  = {a:.1f} MB")
    v = vram_mb()
    print(f"  vram_mb = {v} MB" + ("  (无 torch/CUDA，返回 nan 属正常)" if isinstance(v, float) and math.isnan(v) else ""))
    frames = []
    for _ in range(3):
        frames.append(sample())
        time.sleep(0.05)
    print(f"  sample() x3 = {frames}")
    checks = [
        ("rss_mb 是有限正数", a == a and a > 1.0),
        ("sample() 含 rss_mb/vram_mb", set(frames[0]) == {"rss_mb", "vram_mb"}),
        ("采样足够便宜（1000 次 < 200 ms）", _timing_ok()),
    ]
    for name, good in checks:
        print(f"  {'✅' if good else '❌'} {name}")
    return 0 if all(g for _, g in checks) else 1


def _timing_ok() -> bool:
    import time
    t0 = time.perf_counter()
    for _ in range(1000):
        rss_mb()
    return (time.perf_counter() - t0) < 0.2


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
