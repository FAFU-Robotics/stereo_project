"""pipeline_threaded.py —— 采集 / 推理 解耦的三线程流水线（阶段 7.2）
=============================================================================
【为什么需要】
    单线程串行时，帧周期 = 采集 + 校正 + 推理 + 点云 + 显示（实测 ≈ 220 ms → 4.99 FPS），
    而其中推理独占 164 ms（82%）。相机本身能跑 51.6 FPS，却只能跟着推理走 5 FPS。
    把"采集"和"推理"放到后台线程后：
        - 画面可以用**最新帧**刷新（20~30 FPS，不再等推理）
        - 点云/深度的显示与推理**并行**（不再串行相加）

【设计原则（都是踩过坑的结论）】
    1) **只保留最新，不排队**：队列会越积越多、越来越"旧"，实时演示必须丢弃旧帧。
    2) **推理固定在一个线程里**：PyTorch 的 CUDA 上下文有线程归属，跨线程乱调会出问题。
    3) **线程出错要能被主线程看到**：任何后台异常都记到 .error 并置停止位，
       主线程负责收尾（不要让线程静默死掉、主循环还在空转）。
    4) **共享数据只传引用**：numpy 帧与结果字典写入后不再修改，所以持锁只做引用交换，
       不拷贝大数组（640x480x3 ≈ 0.9 MB，每帧拷贝会白花几毫秒）。
    5) **两者速率解耦**：显示端自己控制刷新节奏（display_fps），推理端尽力而为。

【用法】
    pipe = ThreadedPipeline(cam, est, rectifier=rectifier)
    pipe.start()
    while True:
        pair, seq = pipe.latest_frame()          # 最新一帧（不阻塞、可为 None）
        if pipe.has_new_result():
            frame, out = pipe.take_result()      # 最新一次推理结果（含配对帧）
        pipe.stats()                             # {"capture_fps","infer_fps",...}
        ...
    pipe.stop()
=============================================================================
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

from camera import FramePair


class ThreadedPipeline:
    """采集 + 推理 两个后台线程；主线程按自己的节奏取最新帧/最新结果显示。"""

    def __init__(self, cam: Any, est: Any, rectifier: Any = None,
                 log: Optional[Any] = None) -> None:
        self.cam = cam
        self.est = est
        self.rectifier = rectifier
        self.log = log or (lambda *a, **k: None)

        self._lock = threading.Lock()
        self._frame: Optional[FramePair] = None      # 最新采集帧（原始）
        self._frame_seq = 0                          # 采集序号（递增）
        self._result: Optional[Dict[str, Any]] = None
        self._result_frame: Optional[FramePair] = None   # 与结果配对的（已校正）帧
        self._result_seq = 0
        self._consumed = 0

        self._stop = threading.Event()
        self._threads = []
        self.error: Optional[BaseException] = None

        # 速率统计（保留最近 31 个时间戳，用首尾差算速率，抗抖动）
        self._cap_ts = []
        self._inf_ts = []

    # ------------------------------------------------------------------ 线程体
    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                pair = self.cam.read()
            except BaseException as exc:      # noqa: BLE001 —— 必须让主线程看到
                self.error = exc
                self._stop.set()
                return
            now = time.perf_counter()
            with self._lock:
                self._frame = pair
                self._frame_seq += 1
                self._cap_ts.append(now)
                if len(self._cap_ts) > 31:
                    self._cap_ts.pop(0)

    def _infer_loop(self) -> None:
        last_done = 0
        while not self._stop.is_set():
            with self._lock:
                pair = self._frame
                seq = self._frame_seq
            if pair is None or seq == last_done:
                time.sleep(0.002)             # 没有新帧，稍等 2ms（不要忙等）
                continue
            try:
                left, right = pair.left, pair.right
                if self.rectifier is not None:
                    left, right = self.rectifier.rectify_pair(left, right)
                res = self.est.infer(left, right)
            except BaseException as exc:      # noqa: BLE001
                self.error = exc
                self._stop.set()
                return
            last_done = seq
            now = time.perf_counter()
            with self._lock:
                self._result = res
                self._result_frame = FramePair(left, right, pair.t_left, pair.t_right,
                                               pair.seq, pair.mode, pair.synced)
                self._result_seq += 1
                self._inf_ts.append(now)
                if len(self._inf_ts) > 31:
                    self._inf_ts.pop(0)

    # ------------------------------------------------------------------ 控制
    def start(self) -> "ThreadedPipeline":
        t_cap = threading.Thread(target=self._capture_loop, name="dsh-capture", daemon=True)
        t_inf = threading.Thread(target=self._infer_loop, name="dsh-infer", daemon=True)
        self._threads = [t_cap, t_inf]
        for t in self._threads:
            t.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads = []

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ------------------------------------------------------------------ 取数据
    def latest_frame(self) -> Tuple[Optional[FramePair], int]:
        """最新采集帧与序号（不阻塞；还没采到则返回 (None, 0)）。"""
        with self._lock:
            return self._frame, self._frame_seq

    def has_new_result(self) -> bool:
        """是否有"还没被消费过"的新推理结果。"""
        with self._lock:
            return self._result is not None and self._result_seq > self._consumed

    def take_result(self) -> Tuple[Optional[FramePair], Optional[Dict[str, Any]]]:
        """取最新推理结果（标记为已消费），返回 (配对帧, 结果)。"""
        with self._lock:
            self._consumed = self._result_seq
            return self._result_frame, self._result

    def peek_result(self) -> Tuple[Optional[FramePair], Optional[Dict[str, Any]]]:
        """看一眼最新结果但**不**标记消费（重绘时用）。"""
        with self._lock:
            return self._result_frame, self._result

    def stats(self) -> Dict[str, float]:
        def rate(ts) -> float:
            if len(ts) < 2:
                return 0.0
            span = ts[-1] - ts[0]
            return (len(ts) - 1) / span if span > 1e-9 else 0.0

        with self._lock:
            return {
                "capture_fps": rate(self._cap_ts),
                "infer_fps": rate(self._inf_ts),
                "frames": float(self._frame_seq),
                "results": float(self._result_seq),
                "pending": float(max(self._frame_seq - self._result_seq, 0)),
            }
