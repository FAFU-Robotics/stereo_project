"""stereo_sdk —— 双目深度相机 Python SDK（GUI / 外部程序统一入口）

=============================================================================
【这一层是干什么的】
    把"相机采集 / 校正 / 推理 / 点云"整套东西，包装成一个**稳定、非阻塞**的接口，
    让 GUI 或别的程序**不需要碰** camera.py / stereo_depth.py 这些内部模块。
    内部怎么重构，只要这一层的 API 不变，GUI 代码就不用改。

【最重要的一条】
    本 SDK **故意不提供**同步的 infer()。
    推理一次要 47~164 ms，如果在 UI 线程里同步调用，界面必卡死。
    所以只提供：
        latest_frame()   取最新一帧（永不阻塞）
        has_new_result() 有没有新结果
        take_result()    取最新结果（永不阻塞）
    GUI 只需要一个定时器（建议 30 Hz）轮询这两个方法即可。

【快速开始（没有相机也能跑）】
    from stereo_sdk.mock import MockSession
    s = MockSession(); s.start()
    frame, seq = s.latest_frame()
    if s.has_new_result():
        r = s.take_result()          # r.depth / r.points / r.timings ...

【有相机之后】
    from stereo_sdk import Session
    s = Session("config.yaml"); s.start()   # 记得在后台线程里 start ⚠️

【约定（务必遵守）】
    图像 = BGR uint8；深度 = float32 米（0=无效，另给 valid 掩码）；
    点云 = float32 米，相机系 **y 向下**；时间 = time.perf_counter() 秒。
=============================================================================
"""

from __future__ import annotations

from .types import (HOT_PARAMS, RESTART_PARAMS, SDK_VERSION, TIMING_KEYS, DepthResult,
                    FramePair, check_frame, check_result)

__all__ = [
    "SDK_VERSION", "TIMING_KEYS", "FramePair", "DepthResult",
    "HOT_PARAMS", "RESTART_PARAMS", "check_frame", "check_result",
    "Session", "MockSession", "get_session",
]


def get_session(config_path: str = "config.yaml", mock: bool = False):
    """按需返回真 Session 或 MockSession（推荐 GUI 开发者先用 mock=True）。"""
    if mock:
        from .mock import MockSession
        return MockSession(config_path)
    from .session import Session
    return Session(config_path)


def __getattr__(name: str):  # 惰性导入：只有真的用到真 Session 才加载 torch 等重依赖
    if name == "Session":
        from .session import Session as _S
        return _S
    if name == "MockSession":
        from .mock import MockSession as _M
        return _M
    raise AttributeError(f"module 'stereo_sdk' has no attribute {name!r}")
