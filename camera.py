"""camera.py —— 双目相机采集模块
=============================================================================
职责（单一）：
    1) 枚举/探测本机可用摄像头，并读取注册表里的**真实设备名**（Windows）；
    2) 按 config.yaml 打开相机，支持两种模式：
         - side_by_side ：一个 USB 设备输出左右拼接画面，切成左眼/右眼
         - dual         ：两个独立 UVC 设备，各自一个 OpenCV 索引
    3) 提供 read() 取得"一对"左右帧，并记录采集时间戳；
    4) 检测读取失败、黑屏、设备占用，并给出可读的错误信息；
    5) 安全释放设备（close / with 语句），保证退出后相机不残留占用。

明确不做的事：
    - 不做任何"同步"承诺。两个独立 read() 不是严格同步（见 StereoCamera.read 文档）。
    - 不做去畸变/校正（属于 calibration.py）。
    - 不做深度计算（属于 depth_utils.py）。

数据结构：
    FramePair : 一对左右帧 + 时间戳 + 序号（见下方 dataclass）

术语：
    "眼" = 左相机或右相机；"拼接画面" = 一只眼占左半幅、另一只眼占右半幅的单幅图像。
=============================================================================
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# OpenCV 后端的可读名字 -> OpenCV 常量
BACKENDS: Dict[str, int] = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}

# 默认回退顺序：DSHOW 是本机唯一可靠的后端（MSMF 在本机按索引打不开物理相机）
DEFAULT_BACKEND_ORDER: Tuple[str, ...] = ("dshow", "msmf", "any")


def _normalize_backend_order(value: Any, primary: str) -> List[str]:
    """把 "dshow,msmf,any" / ["dshow", ...] 规范成候选后端顺序。

    规则：配置的 primary 永远排第一；随后是 backend_fallbacks；最后补齐默认项。
    只保留 BACKENDS 里认识的键，并去重；这样即使 config 写错也不会静默变成 CAP_ANY。
    """
    if isinstance(value, str):
        items = [v.strip().lower() for v in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(v).strip().lower() for v in value]
    else:
        items = []
    ordered = [str(primary).strip().lower()] + items + list(DEFAULT_BACKEND_ORDER)
    out: List[str] = []
    for name in ordered:
        if name and name in BACKENDS and name not in out:
            out.append(name)
    return out or ["dshow"]


class CameraError(RuntimeError):
    """相机打开/配置失败（设备不存在、被占用、后端不可用等）。"""


class CameraReadError(RuntimeError):
    """相机在重试后仍无法返回有效帧。"""


# -----------------------------------------------------------------------------
# 设备枚举（Windows 注册表读取真实设备名）
# -----------------------------------------------------------------------------
def _format_vid_pid(instance_key: str) -> str:
    """把注册表键 "VID_0BDA&PID_5883&MI_00" 格式化成 "0BDA:5883"。

    直接 replace("&", "") 会得到 "0BDA:5883MI_00" 这种带接口后缀的字符串；
    这里只取 VID_/PID_ 两段，解析失败时原样返回，绝不抛异常。
    """
    vid = pid = ""
    for part in str(instance_key).split("&"):
        if part.startswith("VID_"):
            vid = part[4:]
        elif part.startswith("PID_"):
            pid = part[4:]
    return f"{vid}:{pid}" if vid and pid else str(instance_key)


def list_video_devices_from_registry() -> List[Dict[str, str]]:
    """从 Windows 注册表读取当前**在线**的 UVC 视频设备真实名字。

    为什么需要它：
        OpenCV 在 Windows 上只能用"索引"打开相机，**拿不到设备名**，而本机同时存在
        笔记本内置摄像头和 HP 的虚拟摄像头，光看索引无法判断哪个是双目相机。
        注册表里能看到 FriendlyName（如 "USB Camera"），用来和 OpenCV 索引做交叉印证。

    输入：无
    输出：list[dict]，每项形如
        {"instance_id": "USB\\VID_0BDA&PID_5883&MI_00\\6&d574a2&0&0000",
         "friendly_name": "USB Camera",
         "vid_pid": "0BDA:5883"}
    说明：非 Windows 或读取失败时返回空列表，不抛异常（探测功能不应让主程序崩）。
    """
    devices: List[Dict[str, str]] = []
    try:
        import winreg  # 仅 Windows 提供
    except ImportError:
        return devices

    # 先用 pnputil 拿到"当前真正连接"的设备集合，用来标记注册表里的历史残留
    connected = list_connected_camera_devices()

    enum_root = r"SYSTEM\CurrentControlSet\Enum\USB"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, enum_root) as usb_root:
            n_sub, _, _ = winreg.QueryInfoKey(usb_root)
            for i in range(n_sub):
                vid_pid = winreg.EnumKey(usb_root, i)
                try:
                    with winreg.OpenKey(usb_root, vid_pid) as vid_key:
                        n_inst, _, _ = winreg.QueryInfoKey(vid_key)
                        for j in range(n_inst):
                            inst = winreg.EnumKey(vid_key, j)
                            with winreg.OpenKey(vid_key, inst) as inst_key:
                                vals = {
                                    winreg.EnumValue(inst_key, k)[0]: winreg.EnumValue(inst_key, k)[1]
                                    for k in range(winreg.QueryInfoKey(inst_key)[1])
                                }
                            # 只保留 UVC 视频接口（Service=usbvideo）
                            if str(vals.get("Service", "")).lower() != "usbvideo":
                                continue
                            name = str(vals.get("FriendlyName", "") or vals.get("DeviceDesc", ""))
                            # DeviceDesc 形如 "@usbvideo.inf,%usbvideo.devicedesc%;USB Video Device"
                            if ";" in name:
                                name = name.split(";")[-1]
                            devices.append(
                                {
                                    "instance_id": f"USB\\{vid_pid}\\{inst}",
                                    "friendly_name": name,
                                    "vid_pid": _format_vid_pid(vid_pid),
                                    # present=False 表示"历史上插过但现在没插"（注册表残留）
                                    "present": "是" if f"USB\\{vid_pid}\\{inst}" in connected else "否",
                                }
                            )
                except OSError:
                    continue
    except OSError:
        return devices
    return devices


def list_connected_camera_devices() -> Dict[str, Dict[str, str]]:
    """列出**当前真正连接且已启动**的相机类设备（用系统自带 pnputil，只读）。

    为什么需要它：
        注册表 Enum 分支里同时保存着"历史上插过、现在没插"的设备
        （本机就有 2 个 Intel RealSense D405 是这种残留），
        只看注册表会把不存在的相机也算进去，误导人。pnputil 会给出 Status 字段
        （Started / Disconnected），据此过滤才是真实的设备清单。

    输入：无
    输出：dict，键为设备实例 ID，值为
        {"description": "USB Camera", "status": "Started", "driver": "usbvideo.inf"}
    说明：pnputil 输出在中文系统上可能是 GBK 编码，这里用 errors='replace' 容错，
          并且只解析 ASCII 字段（Instance ID / Status），不依赖中文。
    """
    out: Dict[str, Dict[str, str]] = {}
    try:
        proc = subprocess.run(
            ["pnputil", "/enum-devices", "/connected", "/class", "Camera"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except Exception:
        return out

    cur: Optional[str] = None
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip()
        if line.startswith("Instance ID:"):
            cur = line.split(":", 1)[1].strip()
            out[cur] = {"description": "", "status": "", "driver": ""}
        elif cur and line.startswith("Device Description:"):
            out[cur]["description"] = line.split(":", 1)[1].strip()
        elif cur and line.startswith("Status:"):
            out[cur]["status"] = line.split(":", 1)[1].strip()
        elif cur and line.startswith("Driver Name:"):
            out[cur]["driver"] = line.split(":", 1)[1].strip()
    return out


def quick_probe_index(
    index: int,
    backend: str = "dshow",
    fourcc: str = "",
    width: int = 1280,
    height: int = 480,
    warmup: int = 5,
    measure_frames: int = 10,
) -> Dict[str, Any]:
    """快速探测单个 OpenCV 索引：能否打开、真实分辨率、实测帧率、画面是否黑屏。

    输入：
        index           : OpenCV 相机索引（int）
        backend         : "dshow" / "msmf" / "any"
        fourcc          : "MJPG" / "YUY2" / ""（不设置）
        width, height   : 请求分辨率
        warmup          : 丢弃的前若干帧
        measure_frames  : 用来测帧率的帧数
    输出：dict
        {"index":2, "opened":True, "width":1280, "height":480, "fps":29.8,
         "mean":83.2, "std":41.0, "blank":False, "error":""}
    注意：本函数会**打开并释放**相机；调用方负责不要在别处同时占用同一设备。
    """
    result: Dict[str, Any] = {
        "index": index,
        "backend": backend,
        "fourcc": fourcc,
        "opened": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "blank": True,
        "error": "",
    }
    cap: Optional[cv2.VideoCapture] = None
    try:
        cap = cv2.VideoCapture(index, BACKENDS.get(backend, cv2.CAP_ANY))
        if not cap.isOpened():
            result["error"] = "打不开（设备不存在 / 被占用 / 后端不支持按索引打开）"
            return result
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width > 0 and height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 降低延迟：只保留最新帧
        except Exception:
            pass
        for _ in range(max(0, warmup)):
            cap.read()
        ok_count = 0
        last: Optional[np.ndarray] = None
        t0 = time.perf_counter()
        for _ in range(max(1, measure_frames)):
            ok, frame = cap.read()
            if ok and frame is not None:
                ok_count += 1
                last = frame
        dt = max(time.perf_counter() - t0, 1e-9)
        if last is None:
            result["error"] = "能打开但读不到任何帧（驱动/格式不匹配或设备异常）"
            return result
        result["opened"] = True
        result["height"], result["width"] = int(last.shape[0]), int(last.shape[1])
        result["fps"] = round(ok_count / dt, 1)
        gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY) if last.ndim == 3 else last
        result["mean"] = round(float(gray.mean()), 2)
        result["std"] = round(float(gray.std()), 2)
        result["blank"] = bool(gray.mean() < 1.0 and gray.std() < 1.0)
        return result
    except Exception as exc:  # noqa: BLE001 —— 探测函数不应把异常抛给调用方
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if cap is not None:
            cap.release()


def frame_sharpness(img: np.ndarray) -> float:
    """返回图像清晰度评分（拉普拉斯方差），用来判断"能不能做立体匹配"。

    为什么需要它（实测教训）：
        立体匹配（无论经典算法还是 FFS 这类学习模型）都依赖图像里有**可匹配的纹理**。
        实测本机相机在"失焦 + 欠曝"时，拉普拉斯方差只有 14 左右，
        FFS 会输出几乎恒定的视差（失效），而经典 SGBM 只有 6~17% 像素能匹配上。
        正常清晰、有纹理的画面该值通常在 100 以上；低于 30 基本无法可靠匹配。

    输入：BGR 或灰度图（np.uint8）
    输出：float（越大越清晰；同一台相机、同一场景下可横向比较）
    """
    a = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(a, cv2.CV_64F).var())


# -----------------------------------------------------------------------------
# 一帧数据
# -----------------------------------------------------------------------------
@dataclass
class FramePair:
    """一对左右帧及其采集时间信息。

    字段：
        left / right : np.ndarray, BGR, uint8, 形状 (H, W, 3) —— 已切成单眼大小
        t_left       : float, time.perf_counter() 时刻（左眼帧读取返回的瞬间）
        t_right      : float, 右眼帧时刻；side_by_side 模式下与 t_left 相同
        seq          : int, 递增序号
        mode         : str, "side_by_side" 或 "dual"
        synced       : bool, 该帧对是否来自同一个硬件时间基准
                        - side_by_side: True（同一幅图像内，由相机内部同步）
                        - dual        : False（两次独立 read，只能算"近似同时"）
    """

    left: np.ndarray
    right: np.ndarray
    t_left: float
    t_right: float
    seq: int
    mode: str
    synced: bool


# -----------------------------------------------------------------------------
# 双路相机
# -----------------------------------------------------------------------------
class StereoCamera:
    """双目相机采集器（side_by_side 与 dual 两种模式统一接口）。

    典型用法：
        cam = StereoCamera(cfg["camera"])
        cam.open()
        try:
            while True:
                pair = cam.read()
                ...  # 用 pair.left / pair.right
        finally:
            cam.close()

    输入：cfg —— config.yaml 中 camera 段的 dict（见 config.yaml 注释）
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg or {})
        self.mode: str = str(self.cfg.get("mode", "side_by_side")).lower()
        if self.mode not in ("side_by_side", "dual"):
            raise ValueError(f"camera.mode 只能是 side_by_side 或 dual，当前为 {self.mode!r}")

        self.backend: str = str(self.cfg.get("backend", "dshow")).lower()
        if self.backend not in BACKENDS:
            raise ValueError(f"camera.backend 只能是 {sorted(BACKENDS)}，当前为 {self.backend!r}")
        # 打开策略：主后端优先，失败后按 backend_fallbacks 依次回退。
        # 普通用户（Medium 完整性）用 DSHOW 即可，不需要管理员权限。
        self.backend_order: List[str] = _normalize_backend_order(
            self.cfg.get("backend_fallbacks", "dshow,msmf,any"), primary=self.backend)
        self.open_with_params: bool = bool(self.cfg.get("open_with_params", True))
        self.open_retry: int = max(1, int(self.cfg.get("open_retry", 2)))
        self.open_retry_delay: float = max(0.0, float(self.cfg.get("open_retry_delay_s", 0.35)))
        self.verify_size: bool = bool(self.cfg.get("verify_size", True))
        self.min_aspect_ratio: float = float(self.cfg.get("min_aspect_ratio", 2.0))
        self.auto_detect: bool = bool(self.cfg.get("auto_detect", False))
        self.scan_max_index: int = max(0, int(self.cfg.get("scan_max_index", 5)))
        self.open_report: List[str] = []
        self._resolved_single_index: Optional[int] = None
        self.fourcc: str = str(self.cfg.get("fourcc", "MJPG") or "")
        self.req_w: int = int(self.cfg.get("width", 1280))
        self.req_h: int = int(self.cfg.get("height", 480))
        self.req_fps: int = int(self.cfg.get("fps", 30))
        self.warmup_frames: int = int(self.cfg.get("warmup_frames", 15))
        self.read_retry: int = int(self.cfg.get("read_retry", 3))
        self.blank_mean_th: float = float(self.cfg.get("black_mean_threshold", 1.0))
        self.blank_std_th: float = float(self.cfg.get("black_std_threshold", 1.0))
        # "无有效信号"更宽的判定：灰度和标准差都很小 => 驱动返回的是空缓冲/噪声，
        # 而不是"环境太暗"。实测本机唯一能打开的相机就是这种状态（mean≈2, std≈4）。
        self.no_signal_mean_th: float = float(self.cfg.get("no_signal_mean_threshold", 5.0))
        self.no_signal_std_th: float = float(self.cfg.get("no_signal_std_threshold", 12.0))

        self.eye_width: int = int(self.cfg.get("eye_width", 0) or 0)
        self.split: str = str(self.cfg.get("split", "left_right")).lower()

        # 运行时状态
        self._caps: List[cv2.VideoCapture] = []
        self._opened: bool = False
        self._seq: int = 0
        self._size: Tuple[int, int] = (0, 0)  # 单眼 (W, H)
        self._full_size: Tuple[int, int] = (0, 0)  # 整幅 (W, H)，dual 模式下等于单眼
        self._t_start: float = 0.0
        self._last_read_dt: float = 0.0  # dual 模式左右 read 的时间差（秒）
        self._blank_warned: bool = False
        self.stats: Dict[str, Any] = {
            "frames": 0,
            "read_fail": 0,
            "blank_frames": 0,
            "no_signal_frames": 0,
            "measured_fps": 0.0,
        }

    # ---------------------------------------------------------------- 打开/关闭
    def open(self) -> None:
        """打开相机并做预热。失败抛 CameraError（带真实尝试记录，不猜原因）。"""
        if self._opened:
            return
        self.close()  # 清掉上一次失败/未关闭留下的句柄
        self.open_report = []
        try:
            if self.mode == "side_by_side":
                idx = self._resolve_single_index()
                self._caps = [self._open_one(idx, name="single")]
            else:
                li = int(self.cfg.get("left_index", 0))
                ri = int(self.cfg.get("right_index", 1))
                if li == ri:
                    raise CameraError(f"dual 模式下左右索引相同（{li}），请在 config.yaml 中区分")
                self._caps = [self._open_one(li, name="left")]
                self._caps.append(self._open_one(ri, name="right"))

            self._warmup()
            self._measure_and_validate()
            self._opened = True
            self._t_start = time.perf_counter()
            for line in self.open_report:
                print(f"[相机] {line}")
        except Exception:
            # 关键：任何一步失败都释放已经打开的设备（dual 模式只开成一路时尤其重要）
            self.close()
            raise

    def _resolve_single_index(self) -> int:
        """返回 side_by_side 使用的 OpenCV 索引；支持 single_index: auto。"""
        raw = self.cfg.get("single_index", 0)
        if self.auto_detect or (isinstance(raw, str) and raw.strip().lower() in ("auto", "detect")):
            self._resolved_single_index = self._auto_detect_single_index()
            return self._resolved_single_index
        self._resolved_single_index = int(raw)
        return self._resolved_single_index

    def _auto_detect_single_index(self) -> int:
        """扫描索引，选第一个能给出"宽 >= 2*高"且尺寸符合请求的设备。

        为什么用这个判据（本机实测）：
            0 = 笔记本内置 640x480（4:3）
            1 = DECXIN 双目 1280x480 / 2560x960（8:3，左右拼接）
            2 = HP 虚拟摄像头 640x480（4:3，画面全黑）
        因此"能按请求给出 8:3 画面"足以把双目设备与另外两者分开，且不依赖设备顺序。
        """
        if self.mode != "side_by_side":
            raise CameraError("auto 检测目前只支持 side_by_side 模式")
        print(f"[相机] auto 检测：扫描索引 0..{self.scan_max_index}，"
              f"寻找 {self.req_w}x{self.req_h} 的左右拼接设备 ...")
        errors: List[str] = []
        for idx in range(self.scan_max_index + 1):
            cap, msg = self._try_open(idx, self.backend, use_params=True)
            if cap is None:
                errors.append(f"索引 {idx}: {msg}")
                continue
            try:
                try:
                    ok, frame = cap.read()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"索引 {idx}: read() 异常 {type(exc).__name__}: {exc}")
                    continue
                if not ok or frame is None or frame.size == 0:
                    errors.append(f"索引 {idx}: 能打开但读不到帧")
                    continue
                h, w = frame.shape[:2]
                if self.min_aspect_ratio > 0 and w < self.min_aspect_ratio * h:
                    errors.append(f"索引 {idx}: {w}x{h}（不是左右拼接比例）")
                    continue
                if self.req_w > 0 and self.req_h > 0 and (w, h) != (self.req_w, self.req_h):
                    errors.append(f"索引 {idx}: {w}x{h}（不是请求的 {self.req_w}x{self.req_h}）")
                    continue
                errors.append(f"索引 {idx}: {w}x{h} -> 选中")
                print("[相机] " + "\n[相机] ".join(errors))
                return idx
            finally:
                cap.release()
        raise CameraError(
            "auto 检测失败：没有找到符合尺寸的左右拼接相机。\n  尝试记录：\n    "
            + "\n    ".join(errors)
            + "\n  可把 config.yaml 的 single_index 改回具体索引（本机 1 = DECXIN）后重试。"
        )

    def _warmup(self) -> None:
        """丢弃前若干帧，等自动曝光/白平衡收敛。"""
        for _ in range(max(0, self.warmup_frames)):
            for cap in self._caps:
                try:
                    cap.read()
                except Exception:
                    pass

    def _measure_and_validate(self) -> None:
        """读取一帧确认真实尺寸，并拒绝"驱动静默降级"和明显的错误设备。"""
        if self.mode == "side_by_side":
            ok, frame = self._caps[0].read()
            if not ok or frame is None or frame.size == 0:
                raise CameraError("预热后仍无法读取拼接画面")
            h, w = frame.shape[:2]
            self._full_size = (w, h)
            self._check_requested_size(w, h)
            self._check_stereo_layout(w, h)
            ew = self.eye_width if self.eye_width > 0 else w // 2
            if ew <= 0 or ew * 2 > w + 1:
                raise CameraError(f"拼接画面宽度 {w} 无法按 eye_width={ew} 切分")
            self._size = (ew, h)
            left, right = self._split_frame(frame)
            self._warn_if_same(left, right, "左右半幅")
            m, s = self._image_quality(left)
            self.open_report.append(f"实际输出 {w}x{h}，单眼 {ew}x{h}，左眼 mean={m:.1f} std={s:.1f}")
            return

        # dual 模式
        sizes: List[Tuple[int, int]] = []
        frames: List[np.ndarray] = []
        for cap in self._caps:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                raise CameraError("预热后仍无法读取某一侧画面")
            sizes.append((frame.shape[1], frame.shape[0]))
            frames.append(frame)
        self._size = sizes[0]
        self._full_size = sizes[0]
        if sizes[0] != sizes[1]:
            # 不静默容忍：分辨率不一致会导致后续视差/点云错位
            raise CameraError(f"左右分辨率不一致：{sizes[0]} vs {sizes[1]}，请检查相机配置")
        self._check_requested_size(sizes[0][0], sizes[0][1])
        self._warn_if_same(frames[0], frames[1], "左右相机")
        self.open_report.append(
            f"dual 左={sizes[0][0]}x{sizes[0][1]} 右={sizes[1][0]}x{sizes[1][1]}")

    def _check_stereo_layout(self, w: int, h: int) -> None:
        """side_by_side 模式下拒绝明显不是"左右拼接"的画面。

        本机 DECXIN 的双目格式全是 8:3（1280x480、2560x960 ...），
        如果把请求尺寸误改成 640x480，某些后端会返回 4:3 单幅图；
        继续对半切会得到两幅 320x480 的窄图，视差/点云全部错位。
        """
        if self.min_aspect_ratio <= 0 or h <= 0:
            return
        if w < self.min_aspect_ratio * h:
            raise CameraError(
                f"实际输出 {w}x{h} 不像是左右拼接的双目画面"
                f"（宽/高={w / h:.2f}，要求 >= {self.min_aspect_ratio}）。\n"
                "        本机 DECXIN 的双目格式都是 8:3（如 1280x480、2560x960）。\n"
                "        如果确实使用其它布局的拼接相机，把 camera.min_aspect_ratio 调小或设为 0。"
            )

    def _check_requested_size(self, w: int, h: int) -> None:
        """请求了分辨率却拿到别的尺寸时明确报错，避免静默用错尺寸跑标定。"""
        if not self.verify_size or self.req_w <= 0 or self.req_h <= 0:
            return
        if (w, h) != (self.req_w, self.req_h):
            raise CameraError(
                f"相机实际输出 {w}x{h}，与 config.yaml 请求的 {self.req_w}x{self.req_h} 不一致。\n"
                "        这会让后续视差/点云与标定内参错位，所以这里直接失败而不是继续。\n"
                "        排查：1) python tools\\probe_cameras.py 看该相机支持哪些分辨率；\n"
                "              2) 把 width/height 改成实际支持的值；\n"
                "              3) 确认索引没指向虚拟摄像头/笔记本内置相机；\n"
                "              4) 如确实允许降级，把 camera.verify_size 设为 false。"
            )

    @staticmethod
    def _image_quality(img: np.ndarray) -> Tuple[float, float]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return float(gray.mean()), float(gray.std())

    def _warn_if_same(self, left: np.ndarray, right: np.ndarray, what: str) -> None:
        """左右完全相同时给出明确警告（单目相机/驱动复制是真实故障）。"""
        if left.shape == right.shape and np.array_equal(left, right):
            msg = f"{what}完全相同：可能打开了单目设备，或驱动把同一路画面复制成了两路。"
            self.open_report.append(msg)
            print(f"[警告] {msg}")

    def _try_open(self, index: int, backend_name: str,
                  use_params: bool) -> Tuple[Optional[cv2.VideoCapture], str]:
        """尝试一种 (后端, 打开方式) 组合，不读帧；成功返回 (cap, 说明)。"""
        api = BACKENDS.get(backend_name, cv2.CAP_ANY)
        params: List[int] = []
        if use_params:
            if self.req_w > 0:
                params += [cv2.CAP_PROP_FRAME_WIDTH, self.req_w]
            if self.req_h > 0:
                params += [cv2.CAP_PROP_FRAME_HEIGHT, self.req_h]
            if self.fourcc:
                params += [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc)]
            if self.req_fps > 0:
                params += [cv2.CAP_PROP_FPS, self.req_fps]
        mode = "打开时传参" if params else "打开后设置"
        try:
            cap = cv2.VideoCapture(index, api, params) if params else cv2.VideoCapture(index, api)
        except Exception as exc:  # noqa: BLE001 —— 驱动层可能直接抛 cv2.error
            return None, f"{backend_name} / {mode}: 构造异常 {type(exc).__name__}: {exc}"
        if not cap.isOpened():
            cap.release()
            return None, f"{backend_name} / {mode}: isOpened=False"

        if not params:
            try:
                if self.fourcc:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
                if self.req_w > 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_w)
                if self.req_h > 0:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_h)
                if self.req_fps > 0:
                    cap.set(cv2.CAP_PROP_FPS, self.req_fps)
            except Exception as exc:  # noqa: BLE001
                cap.release()
                return None, f"{backend_name} / {mode}: 设置参数异常 {type(exc).__name__}: {exc}"
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        timeout_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if timeout_prop is not None:
            try:
                cap.set(timeout_prop, 2000)  # 设备异常时避免 read() 永久阻塞
            except Exception:
                pass
        return cap, f"{backend_name} / {mode}: 已打开"

    def _open_one(self, index: int, name: str) -> cv2.VideoCapture:
        """按"主后端优先 + 回退 + 重试"打开单个索引；失败抛带实际记录的 CameraError。"""
        if self.open_with_params:
            plans: List[Tuple[str, bool]] = [(self.backend, True), (self.backend, False)]
        else:
            plans = [(self.backend, False), (self.backend, True)]
        for backend_name in self.backend_order:
            if backend_name != self.backend:
                plans.append((backend_name, True))

        attempts: List[str] = []
        for round_i in range(1, self.open_retry + 1):
            for backend_name, use_params in plans:
                cap, msg = self._try_open(index, backend_name, use_params)
                if cap is None:
                    attempts.append(f"第{round_i}轮 {msg}")
                    continue
                try:
                    ok, frame = cap.read()
                except Exception as exc:  # noqa: BLE001 —— 某些后端会抛 cv2.error
                    cap.release()
                    attempts.append(f"第{round_i}轮 {msg}，read() 异常 {type(exc).__name__}: {exc}")
                    continue
                if not ok or frame is None or frame.size == 0:
                    cap.release()
                    attempts.append(f"第{round_i}轮 {msg}，但读不到首帧")
                    continue
                h, w = frame.shape[:2]
                if self.verify_size and self.req_w > 0 and self.req_h > 0 \
                        and (w, h) != (self.req_w, self.req_h):
                    cap.release()
                    attempts.append(
                        f"第{round_i}轮 {msg}，实际 {w}x{h} != 请求 {self.req_w}x{self.req_h}")
                    continue
                self.open_report.append(f"索引 {index}（{name}）: {msg}，首帧 {w}x{h}")
                return cap
            if round_i < self.open_retry:
                time.sleep(self.open_retry_delay)
        raise CameraError(self._format_open_error(index, name, attempts))

    def _format_open_error(self, index: int, name: str, attempts: List[str]) -> str:
        """把失败尝试 + 当前设备清单整理成可读、可复现的错误（不猜 HRESULT）。"""
        lines: List[str] = [
            f"无法打开相机索引 {index}（{name}）。",
            f"请求：{self.req_w}x{self.req_h} {self.fourcc or '(不设置格式)'} "
            f"@{self.req_fps}fps，模式={self.mode}，主后端={self.backend}",
            "实际尝试：",
        ]
        lines += [f"  - {a}" for a in attempts[-12:]]
        try:
            devs = list_video_devices_from_registry()
        except Exception:
            devs = []
        # 只列"当前在线"的设备，并按 (名字, VID:PID) 去重：
        # 注册表里同一台 UVC 相机会有多个 MI_xx 接口和历史残留，全部列出来只会干扰判断。
        present: List[Dict[str, str]] = []
        seen = set()
        for d in devs:
            if d.get("present") != "是":
                continue
            key = (d.get("friendly_name", ""), d.get("vid_pid", ""))
            if key in seen:
                continue
            seen.add(key)
            present.append(d)
        if present:
            lines.append("当前在线的 UVC 视频设备（注册表，顺序 != OpenCV 索引）：")
            for d in present:
                lines.append(f"  - {d.get('friendly_name', '?')} ({d.get('vid_pid', '?')})")
        lines += [
            "常见原因（按可能性）：",
            "  1) 设备被其他程序占用：相机 App / 会议软件 / 另一个 Python 进程；",
            "  2) 索引因拔插或设备顺序变化指向了别的相机：运行 python tools\\probe_cameras.py 复核；",
            "  3) 进程处于低完整性/沙箱/受限会话：Windows 会拒绝物理相机（E_ACCESSDENIED）；"
            "请改用普通 PowerShell 或桌面终端；",
            "  4) 驱动不接受该分辨率/格式：改 config.yaml 的 width/height/fourcc，"
            "或把 camera.open_with_params 设为 false。",
            "注意：普通用户（Medium 完整性）即可打开相机，不需要管理员权限。",
        ]
        return "\n        ".join(lines)

    def close(self) -> None:
        """释放所有相机句柄。可重复调用（幂等）。"""
        for cap in self._caps:
            try:
                cap.release()
            except Exception:
                pass
        self._caps = []
        self._opened = False

    def __enter__(self) -> "StereoCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------------------------------------------------------------------- 读取
    def read(self) -> FramePair:
        """读取一对左右帧。

        输出：FramePair（见其文档）

        同步性说明（重要，不要误解）：
            - side_by_side 模式：左右眼来自**同一幅**传感器输出，time 上天然同步；
              但"左右是否真的硬件同曝光"仍需按厂家资料确认，不能仅凭此断言。
            - dual 模式：左右各调用一次 read()，两者之间存在毫秒级时间差
              （self._last_read_dt 记录了实测差值）。这**不是严格同步**，
              运动场景会出现左右不匹配，需用硬件触发/同步线或时间戳对齐解决。
        """
        if not self._opened:
            raise CameraError("相机尚未打开，请先调用 open()")

        if self.mode == "side_by_side":
            frame = self._read_with_retry(self._caps[0], who="single")
            left, right = self._split_frame(frame)
            now = time.perf_counter()
            left, right = self._check_blank(left, right)
            self._seq += 1
            self._update_fps()
            self._last_read_dt = 0.0
            return FramePair(left, right, now, now, self._seq, self.mode, synced=True)

        # dual 模式
        lf = self._read_with_retry(self._caps[0], who="left")
        t_l = time.perf_counter()
        rf = self._read_with_retry(self._caps[1], who="right")
        t_r = time.perf_counter()
        self._last_read_dt = t_r - t_l
        # 尺寸兜底：某些驱动在 set() 后返回不同尺寸，这里统一到左眼尺寸
        if rf.shape[:2] != lf.shape[:2]:
            rf = cv2.resize(rf, (lf.shape[1], lf.shape[0]))
        lf, rf = self._check_blank(lf, rf)
        self._seq += 1
        self._update_fps()
        return FramePair(lf, rf, t_l, t_r, self._seq, self.mode, synced=False)

    def _read_with_retry(self, cap: cv2.VideoCapture, who: str) -> np.ndarray:
        """读取一帧，失败时重试；仍失败抛 CameraReadError。"""
        last_err = ""
        for attempt in range(max(1, self.read_retry)):
            try:
                ok, frame = cap.read()
            except Exception as exc:  # noqa: BLE001 —— 部分后端（如 MSMF）会直接抛 cv2.error
                last_err = f"第 {attempt + 1} 次读取异常 {type(exc).__name__}: {exc}"
                continue
            if ok and frame is not None and frame.size > 0:
                return frame
            last_err = f"第 {attempt + 1} 次读取失败"
        self.stats["read_fail"] += 1
        raise CameraReadError(f"{who} 相机连续读取失败（{last_err}）。可能掉线或被其他程序抢占。")

    def _split_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """把"左右拼接"画面切成左右两幅。"""
        h, w = frame.shape[:2]
        ew = self.eye_width if self.eye_width > 0 else w // 2
        if ew <= 0 or ew * 2 > w + 1:
            raise CameraError(f"拼接画面宽度 {w} 无法按 eye_width={ew} 切分")
        a = frame[:, :ew].copy()
        b = frame[:, ew : ew * 2].copy()
        if self.split == "left_right":
            return a, b
        if self.split == "right_left":
            return b, a
        raise ValueError(f"camera.split 只能是 left_right 或 right_left，当前为 {self.split!r}")

    def _check_blank(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测"全黑"与"无有效信号"两类异常帧并计数；只警告一次，避免刷屏。

        判定分两级（阈值都来自 config.yaml，可调）：
            1) 全黑   ：mean < black_mean_threshold 且 std < black_std_threshold
            2) 无信号 ：mean < no_signal_mean_threshold 且 std < no_signal_std_threshold
        为什么不用"单看 mean 小"来判断：真实但很暗的场景（关灯的房间）也会有
        一定噪声（std 明显大于 0），而驱动返回空缓冲时是"接近 0 的均值 + 很小的噪声"。
        本函数不修改像素数据，只计数 + 首次警告。
        """
        def quality(img: np.ndarray) -> Tuple[float, float]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            return float(gray.mean()), float(gray.std())

        stats = [quality(left), quality(right)]
        blank = any(m < self.blank_mean_th and s < self.blank_std_th for m, s in stats)
        no_signal = any(m < self.no_signal_mean_th and s < self.no_signal_std_th for m, s in stats)

        if blank:
            self.stats["blank_frames"] += 1
        if no_signal or blank:
            self.stats["no_signal_frames"] += 1
            if not self._blank_warned:
                self._blank_warned = True
                (ml, sl), (mr, sr) = stats
                print(
                    f"[警告] 画面无有效图像数据（左眼 mean={ml:.2f} std={sl:.2f}，"
                    f"右眼 mean={mr:.2f} std={sr:.2f}）。常见原因：\n"
                    "       1) 相机未真正出流（USB 端口/线材/供电，或用了 USB2 跑不动）；\n"
                    "       2) 分辨率/FOURCC 未被驱动接受（见 tools/probe_cameras.py 结果）；\n"
                    "       3) 镜头被遮挡 / 镜头盖未取下 / 环境全黑；\n"
                    "       4) 曝光未收敛（可增大 camera.warmup_frames）；\n"
                    "       5) 设备被其他程序占用。\n"
                    "       请用『Windows 相机 App』交叉验证：若 App 里也没画面，则不是本项目代码的问题。"
                )
        return left, right

    def _update_fps(self) -> None:
        """用移动窗口统计实测帧率（每 30 帧更新一次，避免抖动）。"""
        self.stats["frames"] += 1
        n = self.stats["frames"]
        if n % 30 == 0:
            dt = time.perf_counter() - self._t_start
            if dt > 0:
                self.stats["measured_fps"] = round(n / dt, 2)

    # ------------------------------------------------------------------ 状态
    @property
    def eye_size(self) -> Tuple[int, int]:
        """单眼分辨率 (W, H)。"""
        return self._size

    @property
    def full_size(self) -> Tuple[int, int]:
        """相机原始输出分辨率 (W, H)；dual 模式下等于单眼。"""
        return self._full_size

    def describe(self) -> str:
        """返回一行可显示在画面上的设备/模式摘要。"""
        if self.mode == "side_by_side":
            idx = (self._resolved_single_index
                   if self._resolved_single_index is not None else self.cfg.get("single_index"))
            return (f"mode=side_by_side idx={idx} backend={self.backend} "
                    f"full={self._full_size[0]}x{self._full_size[1]}")
        return (
            f"mode=dual L={self.cfg.get('left_index')} R={self.cfg.get('right_index')} "
            f"backend={self.backend} eye={self._size[0]}x{self._size[1]}"
        )
