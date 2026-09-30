"""viewer.py —— 显示模块（图像 / 状态信息；后续阶段加入深度图与点云）
=============================================================================
职责：
    1) 把多幅图像按"平铺面板"的方式显示在一个窗口里（左图 | 右图 | …）；
    2) 在每幅面板上叠加标题、分辨率、帧率等状态信息；
    3) 收集键盘输入并返回给 main.py（viewer 只负责显示，不做业务动作）；
    4) 可选：把当前画面保存成截图（文件名由调用方给出）。

为什么单独一个模块：
    main.py 只负责"调度"，显示细节（窗口名、缩放、文字位置、按键）都在这里，
    这样以后加"视差图面板""深度图面板"不用动 main.py 以外的逻辑。

文字说明（重要）：
    OpenCV 的 cv2.putText 只能画 ASCII 字符，**画不出中文**（会变成一串问号）。
    因此叠加在画面上的文字统一用英文；中文提示一律打印到控制台。
    中文字幕渲染需要 PIL + 中文字体，留到阶段 7（界面优化）再做。
=============================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 键盘按键定义（统一在这里，避免各处硬编码数字）
KEY_QUIT = (27, ord("q"))          # ESC 或 q：退出
KEY_PAUSE = (ord("p"), ord(" "))   # p 或 空格：暂停/恢复
KEY_SNAPSHOT = (ord("s"),)         # s：保存当前数据
KEY_INFO = (ord("i"),)             # i：打印详细信息（控制台）
KEY_HELP = (ord("h"),)             # h：显示/隐藏帮助


class StereoViewer:
    """平铺式图像显示窗口。

    输入：
        window_name   : 窗口标题（也是 cv2 的窗口句柄名，必须唯一）
        display_scale : 显示缩放系数（1.0 = 原始像素大小；屏幕放不下时改小）
        show_help     : 启动时是否在画面上显示按键帮助
    """

    HELP_LINES = [
        "keys: q/ESC=quit  p/space=pause  s=save  i=info  h=help",
    ]

    def __init__(
        self,
        window_name: str = "stereo_project",
        display_scale: float = 1.0,
        show_help: bool = True,
    ) -> None:
        self.window_name = window_name
        self.display_scale = float(display_scale) if display_scale else 1.0
        self.show_help = bool(show_help)
        self._created = False
        self._last_key: int = -1

    # ------------------------------------------------------------------ 基础
    def _ensure_window(self) -> None:
        if not self._created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._created = True

    def close(self) -> None:
        """关闭窗口。可重复调用。"""
        if self._created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            try:
                cv2.waitKey(1)  # 让窗口事件真正被处理掉
            except Exception:
                pass
            self._created = False

    # -------------------------------------------------------------- 绘制工具
    @staticmethod
    def _put_text(
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        scale: float = 0.6,
        thickness: int = 1,
        shadow: bool = True,
    ) -> None:
        """在图像上写字（先画黑色描边，保证亮/暗背景都看得清）。原地修改 img。"""
        if shadow:
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _decorate(
        self,
        img: np.ndarray,
        title: str,
        extra: Sequence[str] = (),
    ) -> np.ndarray:
        """给一幅面板叠加标题和附加信息（返回副本，不改动原图）。"""
        out = img.copy()
        self._put_text(out, title, (10, 26), (0, 255, 255), 0.7, 2)
        y = 52
        for line in extra:
            self._put_text(out, line, (10, y), (0, 255, 0), 0.55, 1)
            y += 24
        return out

    # ------------------------------------------------------------------ 显示
    def show_panels(
        self,
        panels: List[Tuple[str, np.ndarray, Sequence[str]]],
        status_lines: Sequence[str] = (),
        paused: bool = False,
    ) -> int:
        """把若干面板横向平铺显示，并返回本帧按下的键（无按键返回 -1）。

        输入：
            panels       : list[(标题, 图像(BGR/gray), 附加信息行)]
                           图像高度可以不同，本函数会自动补齐到相同高度（补黑边）
            status_lines : 底部状态栏文字（英文）
            paused       : 是否处于暂停状态（会在画面上打 PAUSED 标记）
        输出：
            int —— 按下的按键 ASCII 码，或 -1
        """
        if not panels:
            return -1

        # 统一高度（防止不同面板的高度不一致导致 hconcat 失败）
        max_h = max(p[1].shape[0] for p in panels)
        rows: List[np.ndarray] = []
        for title, img, extra in panels:
            canvas = img
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            if canvas.shape[0] < max_h:
                pad = np.zeros((max_h - canvas.shape[0], canvas.shape[1], 3), dtype=canvas.dtype)
                canvas = np.vstack([canvas, pad])
            rows.append(self._decorate(canvas, title, extra))

        frame = np.hstack(rows)

        # 暂停标记
        if paused:
            self._put_text(frame, "PAUSED", (frame.shape[1] // 2 - 80, 40), (0, 0, 255), 1.0, 2)

        # 底部状态栏（黑底 + 多行文字）
        if status_lines:
            bar_h = 26 * len(status_lines) + 8
            bar = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)
            y = 22
            for line in status_lines:
                self._put_text(bar, line, (10, y), (200, 200, 200), 0.55, 1, shadow=False)
                y += 26
            frame = np.vstack([frame, bar])

        # 帮助行
        if self.show_help:
            help_bar = np.zeros((28, frame.shape[1], 3), dtype=np.uint8)
            self._put_text(help_bar, self.HELP_LINES[0], (10, 20), (0, 200, 255), 0.5, 1, shadow=False)
            frame = np.vstack([frame, help_bar])

        if self.display_scale != 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(frame.shape[1] * self.display_scale)), max(1, int(frame.shape[0] * self.display_scale))),
                interpolation=cv2.INTER_AREA,
            )

        self._ensure_window()
        cv2.imshow(self.window_name, frame)
        # waitKey(1) 既刷新窗口又取键；无按键返回 -1
        self._last_key = cv2.waitKey(1) & 0xFF
        return self._last_key if self._last_key != 255 else -1

    def toggle_help(self) -> bool:
        """切换帮助行显示，返回切换后的状态。"""
        self.show_help = not self.show_help
        return self.show_help

    @staticmethod
    def is_quit(key: int) -> bool:
        return key in KEY_QUIT

    @staticmethod
    def is_pause(key: int) -> bool:
        return key in KEY_PAUSE

    @staticmethod
    def is_snapshot(key: int) -> bool:
        return key in KEY_SNAPSHOT

    @staticmethod
    def is_info(key: int) -> bool:
        return key in KEY_INFO

    @staticmethod
    def is_help(key: int) -> bool:
        return key in KEY_HELP
