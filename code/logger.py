"""logger.py —— 日志落盘（稳定性加固 ③）

=============================================================================
【为什么需要】
    现在所有信息都是 print 到控制台。跑几分钟没事，但长时间运行后崩溃时，
    控制台可能已被滚掉、或者程序被直接关掉 —— **事后无从查起**。
    本项目已经吃过这个亏：多个 bug 是靠离线自检复现才定位的。

【设计】
    · 文件：output/logs/session_<时间戳>.log，5 MB × 3 轮转（不会无限长）
    · 同时输出到控制台（不改变现在的使用习惯）
    · 幂等：重复调用 setup() 不会重复加 handler
    · 只依赖标准库

【用法】
    from logger import setup, get_logger
    log = setup("output/logs")            # 程序启动时一次
    log.info("相机打开: %s", cam.describe())
    log.warning("画面清晰度偏低: %.1f", sharp)
    try:
        ...
    except Exception:
        log.exception("推理失败")          # 带 traceback
=============================================================================
"""

from __future__ import annotations

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOGGER_NAME = "stereo"
_inited = False
_log_path: Optional[str] = None


def setup(log_dir: str = "output/logs", level: int = logging.INFO,
          session_tag: Optional[str] = None) -> logging.Logger:
    """初始化日志（幂等）。返回 logger。"""
    global _inited, _log_path
    logger = logging.getLogger(_LOGGER_NAME)
    if _inited:
        return logger
    try:
        os.makedirs(log_dir, exist_ok=True)
        tag = session_tag or time.strftime("%Y%m%d_%H%M%S")
        _log_path = os.path.join(log_dir, f"session_{tag}.log")
        fh = RotatingFileHandler(_log_path, maxBytes=5 * 1024 * 1024, backupCount=3,
                                 encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(level)
        logger.propagate = False
        _inited = True
        logger.info("=== 日志开始：%s ===", _log_path)
    except Exception as exc:  # noqa: BLE001 —— 日志系统本身不能拖垮程序
        logging.getLogger(_LOGGER_NAME).addHandler(logging.NullHandler())
        print(f"[日志] 初始化失败（不影响运行）：{exc}")
    return logger


def get_logger() -> logging.Logger:
    """取 logger（未初始化时返回一个不会报错的对象）。"""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def log_path() -> Optional[str]:
    return _log_path


def _selftest() -> int:
    # 自检目录放在项目内 _work/ 下：某些环境（如受限沙箱）系统临时目录不可写
    _root = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(_root, "_work", f"logtest_{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    log = setup(d, session_tag="selftest")
    log.info("启动参数 %s=%s", "scale", 0.75)
    log.warning("清晰度偏低 %.1f", 11.3)
    try:
        raise RuntimeError("模拟异常")
    except RuntimeError:
        log.exception("推理失败")
    for h in logging.getLogger(_LOGGER_NAME).handlers:
        h.flush()
    path = log_path()
    ok = path is not None and os.path.exists(path)
    txt = open(path, encoding="utf-8").read() if ok else ""
    checks = [
        ("日志文件已创建", ok),
        ("包含启动行", "启动参数" in txt),
        ("包含告警", "清晰度偏低" in txt),
        ("包含 traceback", "Traceback" in txt and "模拟异常" in txt),
        ("幂等（不会重复加 handler）", len(logging.getLogger(_LOGGER_NAME).handlers) == 2),
    ]
    print("=" * 60)
    print("  logger 自检")
    print("=" * 60)
    for name, good in checks:
        print(f"  {'✅' if good else '❌'} {name}")
    print(f"  文件: {path}")
    return 0 if all(g for _, g in checks) else 1


if __name__ == "__main__":
    sys.exit(_selftest())
