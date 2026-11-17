"""profiler.py —— 统一的阶段耗时 / FPS 统计（任务 1 的「统计」交付物）

=============================================================================
【为什么需要它】
    改代码前必须先能量出"每一段到底多慢"。项目里原来是零散的：
    rec_infer_ms（只统计推理）、est.timings_ms（推理内部）、还有各处手算 FPS。
    统一到本模块后：控制台一行摘要 + 落 CSV，就是"性能测试结果"的数据源。

【设计要点】
  · 用 deque 保留最近 N 次，report 时给 中位数/P95/最大 —— 只看均值会被冷启动带偏
    （FFS 第一帧实测 1.2~1.7 秒，均值会被它拉高）。
  · 每个 stage 独立计数，缺数据时返回 None 而不是 0（0 会被误读成"很快"）。
  · `NullProfiler` 提供同接口的空实现，方便把统计做成可选（生产可关掉，零开销）。
  · 落 CSV 用**追加**模式：一次长跑每 N 秒写一行，事后可直接画曲线。

【用法】
    p = Profiler(window=60)
    with p.span("cloud"):          # 自动计时
        builder.build(...)
    p.tick("infer", 95.0)          # 已知耗时（毫秒）
    p.mark("frame")                # 事件计数（用于算 FPS）
    print(p.summary_text())
    p.to_csv("output/logs/perf.csv")

    # 阶段名固定为 STAGES，GUI/报告按这些键取，不要随便改名
=============================================================================
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

STAGES = ("capture", "rectify", "infer", "cloud", "display")


class Profiler:
    """滚动窗口的阶段耗时统计。"""

    def __init__(self, window: int = 60) -> None:
        self.window = max(int(window), 4)
        self._ms: Dict[str, deque] = {s: deque(maxlen=self.window) for s in STAGES}
        self._counts: Dict[str, int] = {}
        self._t0 = time.perf_counter()
        self._last_event: Dict[str, float] = {}
        self._marks: Dict[str, deque] = {}

    # ------------------------------------------------------------------ 记录
    def tick(self, stage: str, ms: Optional[float]) -> None:
        """记录一次耗时（毫秒）。stage 未知时自动建桶，方便临时加阶段。"""
        if ms is None:
            return
        try:
            v = float(ms)
        except (TypeError, ValueError):
            return
        if v < 0 or v != v:            # 负数/NaN 直接丢弃
            return
        self._ms.setdefault(stage, deque(maxlen=self.window)).append(v)
        self._counts[stage] = self._counts.get(stage, 0) + 1

    @contextmanager
    def span(self, stage: str) -> Iterator[None]:
        """with 块自动计时。"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.tick(stage, (time.perf_counter() - t0) * 1000.0)

    def mark(self, key: str) -> None:
        """打一个时间戳（用于算 FPS）。"""
        now = time.perf_counter()
        self._marks.setdefault(key, deque(maxlen=self.window + 1)).append(now)
        self._last_event[key] = now

    def add_frame(self) -> None:
        """等价于 mark("frame")。"""
        self.mark("frame")

    # ------------------------------------------------------------------ 读取
    def fps(self, key: str = "frame") -> Optional[float]:
        """最近 window 次 mark 的速率（次/秒）；样本不足 2 个返回 None。"""
        ts = self._marks.get(key)
        if not ts or len(ts) < 2:
            return None
        span = ts[-1] - ts[0]
        return (len(ts) - 1) / span if span > 1e-9 else None

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """{stage: {n, mean, p50, p95, max}}；无数据的 stage 不出现。"""
        out: Dict[str, Dict[str, Any]] = {}
        for stage, buf in self._ms.items():
            if not buf:
                continue
            arr = sorted(buf)
            n = len(arr)

            def pct(q: float) -> float:
                idx = min(max(int(round(q * (n - 1))), 0), n - 1)
                return arr[idx]

            out[stage] = {"n": n, "mean": sum(arr) / n, "p50": pct(0.5),
                          "p95": pct(0.95), "max": arr[-1]}
        return out

    def summary_text(self, keys: Optional[tuple] = None) -> str:
        """一行摘要，适合直接 print。"""
        keys = keys or STAGES
        s = self.summary()
        parts = []
        for k in keys:
            d = s.get(k)
            parts.append(f"{k}=--" if d is None else f"{k}={d['p50']:.1f}ms")
        fps = self.fps("frame")
        parts.append("fps=--" if fps is None else f"fps={fps:.2f}")
        return " ".join(parts)

    def format_table(self) -> str:
        """多行表格（贴报告用）。"""
        s = self.summary()
        lines = [f"{'stage':<10}{'n':>6}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}"]
        for k in STAGES:
            if k not in s:
                continue
            d = s[k]
            lines.append(f"{k:<10}{d['n']:>6}{d['mean']:>9.2f}{d['p50']:>9.2f}"
                         f"{d['p95']:>9.2f}{d['max']:>9.2f}")
        for k in s:
            if k in STAGES:
                continue
            d = s[k]
            lines.append(f"{k:<10}{d['n']:>6}{d['mean']:>9.2f}{d['p50']:>9.2f}"
                         f"{d['p95']:>9.2f}{d['max']:>9.2f}  (临时阶段)")
        fps = self.fps("frame")
        if fps is not None:
            lines.append(f"fps={fps:.2f}  elapsed={self.elapsed():.1f}s")
        return "\n".join(lines)

    def to_csv(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """追加一行到 CSV（首次自动写表头）。适合长跑时定期调用。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        s = self.summary()
        cols = ["timestamp", "elapsed_s", "fps"] + [f"{k}_p50" for k in STAGES] \
               + [f"{k}_p95" for k in STAGES]
        if extra:
            cols += list(extra.keys())
        need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        row = [time.strftime("%Y-%m-%d %H:%M:%S"), f"{self.elapsed():.1f}",
               "" if self.fps("frame") is None else f"{self.fps('frame'):.2f}"]
        row += ["" if k not in s else f"{s[k]['p50']:.2f}" for k in STAGES]
        row += ["" if k not in s else f"{s[k]['p95']:.2f}" for k in STAGES]
        if extra:
            row += [str(extra[k]) for k in extra]
        with open(path, "a", encoding="utf-8", newline="") as fh:
            if need_header:
                fh.write(",".join(cols) + "\n")
            fh.write(",".join(row) + "\n")
        return path

    def reset(self) -> None:
        for buf in self._ms.values():
            buf.clear()
        self._counts.clear()
        self._marks.clear()
        self._t0 = time.perf_counter()


class NullProfiler(Profiler):
    """空实现：所有记录方法都变成 no-op，用于关闭统计时零开销。"""

    def tick(self, stage: str, ms: Optional[float]) -> None:
        pass

    def mark(self, key: str) -> None:
        pass

    def to_csv(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        return ""


# -----------------------------------------------------------------------------
def _selftest() -> int:
    import math
    print("=" * 70)
    print("  profiler 自检")
    print("=" * 70)
    p = Profiler(window=8)
    # ① span 自动计时
    with p.span("rectify"):
        time.sleep(0.005)
    # ② tick 已知耗时（含异常值，看会不会被 p50 挡住）
    for i in range(10):
        p.tick("infer", 100.0 if i < 9 else 1000.0)
    # ③ mark → fps
    for _ in range(6):
        p.mark("frame")
        time.sleep(0.01)
    s = p.summary()
    ok = True
    if not (4.0 <= s["rectify"]["p50"] <= 7.0):
        print(f"  ❌ rectify p50 异常: {s['rectify']['p50']:.2f}"); ok = False
    if abs(s["infer"]["p50"] - 100.0) > 1e-6:
        print(f"  ❌ infer p50 应为 100（window=8 只保留最近 8 个）：{s['infer']['p50']}"); ok = False
    if s["infer"]["max"] != 1000.0:
        print(f"  ❌ infer max 应为 1000（异常值应保留在 max 里）: {s['infer']['max']}"); ok = False
    fps = p.fps("frame")
    if fps is None or not (1.0 < fps < 200.0):
        print(f"  ❌ fps 异常: {fps}"); ok = False
    # ④ 非法值不能污染统计
    p.tick("cloud", None); p.tick("cloud", float("nan")); p.tick("cloud", -1)
    if "cloud" in p.summary():
        print("  ❌ NaN/负数被记进了统计"); ok = False
    # ⑤ NullProfiler 零副作用
    n = NullProfiler()
    with n.span("infer"):
        pass
    n.mark("frame"); n.tick("infer", 1.0)
    if n.summary() or n.fps("frame") is not None:
        print("  ❌ NullProfiler 不应产生任何数据"); ok = False
    # ⑥ CSV
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_prof_test.csv")
    p.to_csv(path, {"note": "selftest"})
    p.to_csv(path, {"note": "selftest2"})
    txt = open(path, encoding="utf-8").read().strip().splitlines()
    if len(txt) != 3:
        print(f"  ❌ CSV 应为 1 表头 + 2 行，实际 {len(txt)} 行"); ok = False
    os.remove(path)
    print("\n" + p.format_table())
    print("\n  摘要:", p.summary_text())
    print("  " + ("✅ 自检全部通过" if ok else "❌ 自检有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
