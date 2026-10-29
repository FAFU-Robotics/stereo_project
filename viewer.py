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
KEY_VIEW3D = (ord("o"),)           # o：重新打开被关掉的 3D 点云窗口（演示不中断）


class DisparityColorizer:
    """把视差图（float，单位：像素）变成稳定的伪彩图，用于实时显示。

    为什么不用"整幅 min/max"：
        实时视频里只要有一两个异常像素，min/max 就会剧烈跳变 → 画面颜色整体闪烁。
        这里改用【百分位裁剪 + 指数滑动平均(EMA)】：
            - 百分位（默认 2%~98%）忽略极端离群像素；
            - EMA 让显示范围随时间缓慢变化，颜色稳定不闪。
    无效像素（<=0 或非有限值）统一涂黑，一眼看出"这里没有有效视差"。

    输入：
        disp  : np.ndarray (H,W)，float，单位像素
        valid : 可选 bool 数组，True=有效；默认用 disp > 0
    输出：
        (bgr_uint8 (H,W,3), (lo, hi)) —— 同时返回当前显示范围，便于打印到状态栏
    """

    def __init__(
        self,
        colormap: int = cv2.COLORMAP_TURBO,
        percentile: Tuple[float, float] = (2.0, 98.0),
        ema_alpha: float = 0.25,
        invalid_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.colormap = colormap
        self.percentile = percentile
        self.ema_alpha = float(ema_alpha)
        self.invalid_color = invalid_color
        self._lo: Optional[float] = None
        self._hi: Optional[float] = None

    def reset(self) -> None:
        """清空平滑状态（暂停恢复后、或换了场景时可调用）。"""
        self._lo = None
        self._hi = None

    def colorize(self, disp: np.ndarray, valid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Tuple[float, float]]:
        d = np.asarray(disp, dtype=np.float32)
        if d.ndim != 2:
            raise ValueError(f"视差图必须是 2D，当前形状 {d.shape}")
        if valid is None:
            valid = np.isfinite(d) & (d > 0)
        else:
            valid = np.asarray(valid, dtype=bool) & np.isfinite(d)

        out = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
        if not valid.any():
            return out, (0.0, 0.0)

        lo, hi = np.percentile(d[valid], self.percentile)
        lo, hi = float(lo), float(hi)
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        if self._lo is None:
            self._lo, self._hi = lo, hi
        else:
            a = self.ema_alpha
            self._lo = (1 - a) * self._lo + a * lo
            self._hi = (1 - a) * self._hi + a * hi

        span = max(self._hi - self._lo, 1e-6)
        norm = np.clip((d - self._lo) / span, 0.0, 1.0) * 255.0
        colored = cv2.applyColorMap(norm.astype(np.uint8), self.colormap)
        out[valid] = colored[valid]
        out[~valid] = self.invalid_color
        return out, (self._lo, self._hi)


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
        window_name: str = "Stereo Viewer",
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

    def _tile(self, img: np.ndarray, title: str, extra: Sequence[str] = ()) -> np.ndarray:
        """单路画面：顶上一条深色标题（流名称 + 分辨率），画面本身不再盖字。"""
        h, w = img.shape[:2]
        head = np.full((28, w, 3), 28, dtype=np.uint8)
        cv2.line(head, (0, 27), (w, 27), (90, 90, 90), 1)
        self._put_text(head, str(title)[:42], (8, 19), (235, 235, 235), 0.5, 1, shadow=False)
        if extra:
            side = str(extra[0])[:28]
            self._put_text(head, side, (max(8, w - 8 - int(9 * len(side))), 19),
                           (170, 170, 170), 0.45, 1, shadow=False)
        return np.vstack([head, img])

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

        # 统一高度后做成 RealSense 那种「标题条 + 画面」磁贴，中间留深色缝
        max_h = max(p[1].shape[0] for p in panels)
        rows: List[np.ndarray] = []
        for title, img, extra in panels:
            canvas = img
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            if canvas.shape[0] < max_h:
                pad = np.zeros((max_h - canvas.shape[0], canvas.shape[1], 3), dtype=canvas.dtype)
                canvas = np.vstack([canvas, pad])
            rows.append(self._tile(canvas, title, extra))

        gap_h = rows[0].shape[0]
        gap = np.full((gap_h, 8, 3), 18, dtype=np.uint8)
        pieces: List[np.ndarray] = []
        for i, row in enumerate(rows):
            if i:
                pieces.append(gap)
            pieces.append(row)
        frame = np.hstack(pieces)
        margin = np.full((frame.shape[0], 8, 3), 18, dtype=np.uint8)
        frame = np.hstack([margin, frame, margin])

        if paused:
            self._put_text(frame, "PAUSED", (frame.shape[1] // 2 - 70, 22), (80, 80, 255), 0.7, 2)

        if status_lines:
            bar_h = 22 * len(status_lines) + 10
            bar = np.full((bar_h, frame.shape[1], 3), 16, dtype=np.uint8)
            cv2.line(bar, (0, 0), (frame.shape[1], 0), (239, 174, 0), 2)
            y = 18
            for line in status_lines:
                self._put_text(bar, line, (12, y), (210, 210, 210), 0.48, 1, shadow=False)
                y += 22
            frame = np.vstack([frame, bar])

        if self.show_help:
            help_bar = np.full((24, frame.shape[1], 3), 16, dtype=np.uint8)
            self._put_text(help_bar, self.HELP_LINES[0], (12, 16), (160, 160, 160), 0.45, 1, shadow=False)
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

    @staticmethod
    def is_view3d(key: int) -> bool:
        """是否按下了"重新打开 3D 点云窗口"键（o）。"""
        return key in KEY_VIEW3D
