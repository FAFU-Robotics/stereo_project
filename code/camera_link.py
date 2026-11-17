"""camera_link.py —— 左右相机连接、USB 热插拔判定、串流启动步骤规划

=============================================================================
这一层只做「该不该断开、断开哪一侧、串流还要做哪些步骤」。
它不打开设备、不碰 UI。真正关句柄在 StereoCamera，停流水线在 Session。

【触发条件】
    1. 软件启动（viewer 调用 Session.prime）
       默认打开左、右两路。side_by_side 只有一个 USB 设备，左右眼是同一路
       采集切出来的，两路逻辑连接一起开、一起关。dual 才是两个独立设备。
    2. 热插拔 / 读帧失败
       某一侧 read() 连续失败，或该侧绑定的 USB 实例从系统里消失：
       只关闭这一侧的句柄。side_by_side 共享一个句柄，所以左右一起关。
    3. USB 设备断开
       确认拔掉的是「我们正在用的相机」之后，走软件断开：
       停采集/推理线程、started=False、按钮和锁与剩余句柄一致。
       还剩一路时保留这一路和进程锁，方便把掉线的那一路重新打开。
       一路都不剩时释放锁，避免下次启动被自己的锁挡住。

【怎样确认“是我们的相机”】
    打开成功时给左右各记一份当时在线的实例 ID。
    · config 里写了 usb_match / left_usb / right_usb：只认 ID 里带这段文字的设备。
      这样拔掉笔记本自带摄像头不会把双目相机拆掉。
    · 没写匹配串、打开时系统里只有一台相机：这台消失就认定是我们的。
    · 好几台相机、又对不上号：先不关。没在串流就试读一帧；正在串流就等采集
      线程的读失败（读超时约 2 秒）。枚举本身失败（返回 None）不当成拔出。

【异常】
    枚举、探测、release 都吞掉底层异常，变成「这一侧已断开」或空操作。
    同一次拔出可能先被 USB 轮询看到、随后又被读失败看到，断开必须幂等：
    句柄已经是空的，第二次调用什么都不做，也不要再次释放锁。

【串流为什么能更快】
    打开相机和加载模型互不依赖，可以并行。预热必须等两边都好。
    启动时如果 prime() 已经把相机打开、模型加载并预热过，开始串流只拉起
    采集线程和推理线程。用户点「停止」时默认留着相机和模型，下一次开始
    不再付这两段时间。标定和退出才会把相机交出去。
=============================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

# Camera class GUID（cfgmgr32 按类枚举时用）
_CAMERA_CLASS = "{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}"
_CM_GETIDLIST_FILTER_ENUMERATOR = 0x00000001
_CM_GETIDLIST_FILTER_PRESENT = 0x00000100
_CM_GETIDLIST_FILTER_CLASS = 0x00000200


@dataclass(frozen=True)
class DisconnectPlan:
    """一次热插拔要做什么。close_sides 为空表示这次不动连接。"""

    close_sides: Tuple[str, ...]
    software_disconnect: bool
    needs_probe: bool
    reason: str


@dataclass(frozen=True)
class StartPlan:
    """开始串流时哪些步骤还可以跳过。"""

    open_cameras: bool
    load_model: bool
    warmup: bool
    parallel: bool
    note: str


def _cm_id_list(filt: Optional[str], flags: int) -> Optional[Set[str]]:
    """cfgmgr32 枚举当前设备实例。失败返回 None，成功可以是空集。"""
    if os.name != "nt":
        return set()
    try:
        import ctypes
        from ctypes import wintypes

        cfg = ctypes.WinDLL("cfgmgr32", use_last_error=True)
        size_of = cfg.CM_Get_Device_ID_List_SizeW
        size_of.argtypes = [ctypes.POINTER(wintypes.ULONG), wintypes.LPCWSTR, wintypes.ULONG]
        size_of.restype = wintypes.ULONG
        get_list = cfg.CM_Get_Device_ID_ListW
        get_list.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.ULONG, wintypes.ULONG]
        get_list.restype = wintypes.ULONG

        needed = wintypes.ULONG(0)
        if size_of(ctypes.byref(needed), filt, flags) != 0 or needed.value <= 0:
            return None
        buf = ctypes.create_unicode_buffer(needed.value)
        if get_list(filt, buf, needed, flags) != 0:
            return None
        raw = ctypes.wstring_at(ctypes.addressof(buf), needed.value)
        return {part for part in raw.split("\x00") if part}
    except Exception:  # noqa: BLE001 —— 枚举失败交给调用方当「这一轮没有结果」
        return None


def list_present_camera_ids() -> Optional[Set[str]]:
    """当前仍连接的相机类设备实例 ID。

    返回：
        set  —— 枚举成功（可以是空集：系统里确实没有相机）
        None —— 枚举失败。调用方必须忽略这一轮，不能把它当成「全部拔出」。
    """
    flags = _CM_GETIDLIST_FILTER_PRESENT | _CM_GETIDLIST_FILTER_CLASS
    return _cm_id_list(_CAMERA_CLASS, flags)


def list_watched_ids(usb_match: str = "") -> Optional[Set[str]]:
    """热插拔要盯住的实例。

    写了 usb_match 时改走 USB 枚举，只留下 ID 里带这段文字的设备。
    双目相机会从列表里消失，笔记本自带摄像头不会混进来。
    没写匹配串时仍用相机类列表。
    """
    token = (usb_match or "").strip().lower()
    if not token:
        return list_present_camera_ids()
    flags = _CM_GETIDLIST_FILTER_PRESENT | _CM_GETIDLIST_FILTER_ENUMERATOR
    present = _cm_id_list("USB", flags)
    if present is None:
        return None
    # 一只相机会拆成 MI_00 / MI_02 好几个接口。打开时这些接口会短暂消失再出现。
    # 只要还有任意一个接口在，就当成同一台设备还插着，避免反复重开。
    if any(token in i.lower() for i in present):
        return {token}
    return set()


def bind_snapshot(
    mode: str,
    present_ids: Sequence[str],
    usb_match: str = "",
    left_usb: str = "",
    right_usb: str = "",
) -> Dict[str, Tuple[str, ...]]:
    """打开相机那一刻，左右各自绑定哪些实例 ID。"""
    present = tuple(str(i) for i in present_ids if str(i))
    match = (usb_match or "").strip().lower()

    def select(token: str) -> Tuple[str, ...]:
        token = (token or "").strip().lower()
        if token:
            hit = tuple(i for i in present if token in i.lower())
            if hit:
                return hit
        if match:
            hit = tuple(i for i in present if match in i.lower())
            if hit:
                return hit
        return present

    if (mode or "").lower() == "dual" and ((left_usb or "").strip() or (right_usb or "").strip()):
        return {"left": select(left_usb), "right": select(right_usb)}
    chosen = select(usb_match)
    return {"left": chosen, "right": chosen}


def sides_from_read_error(message: str, mode: str) -> Tuple[str, ...]:
    """从 CameraReadError 的原文判断掉的是哪一侧。"""
    text = message or ""
    low = text.lower()
    if (mode or "").lower() == "side_by_side" or "single" in low:
        return ("left", "right")
    sides = []
    if "left" in low or "左" in text:
        sides.append("left")
    if "right" in low or "右" in text:
        sides.append("right")
    if not sides:
        return ("left", "right")
    return tuple(sides)


def _hits(ids: Sequence[str], removed: Set[str]) -> bool:
    return any(i in removed for i in ids)


def plan_usb_disconnect(
    *,
    mode: str,
    open_sides: Sequence[str],
    bound: Dict[str, Sequence[str]],
    removed_ids: Sequence[str],
    usb_match: str = "",
    streaming: bool = False,
    read_error: str = "",
    sole_device: bool = False,
) -> DisconnectPlan:
    """根据拔出的实例 ID（或读帧失败）决定关哪一侧、要不要走软件断开。"""
    del streaming  # 是否在串流不改变「关哪一侧」；调用方看到 software_disconnect 再停线程
    mode = (mode or "side_by_side").lower()
    open_set = {s for s in open_sides if s in ("left", "right")}
    removed = {str(i) for i in removed_ids if str(i)}
    match = (usb_match or "").strip().lower()

    if read_error:
        if not open_set:
            return DisconnectPlan((), False, False, "")
        if mode == "side_by_side":
            close = tuple(s for s in ("left", "right") if s in open_set)
        else:
            wanted = sides_from_read_error(read_error, mode)
            close = tuple(s for s in wanted if s in open_set)
        if not close:
            return DisconnectPlan((), False, False, "")
        return DisconnectPlan(close, True, False, read_error)

    if not removed or not open_set:
        return DisconnectPlan((), False, False, "")
    if match and not any(match in r.lower() for r in removed):
        return DisconnectPlan((), False, False, "")

    left_ids = tuple(bound.get("left") or ())
    right_ids = tuple(bound.get("right") or ())

    if mode == "side_by_side":
        bound_ids = left_ids or right_ids
        # 打开时系统里还有别的摄像头，又没写 usb_match：左右眼被绑到了「当时所有相机」。
        # 这时拔掉任意一台都不能认定就是这只双目，等读帧失败再关，避免误伤。
        if len(bound_ids) > 1 and not match:
            if _hits(bound_ids, removed):
                return DisconnectPlan((), False, True, "usb-ambiguous")
            return DisconnectPlan((), False, False, "")
        if bound_ids and not _hits(bound_ids, removed):
            return DisconnectPlan((), False, False, "")
        if not bound_ids and not sole_device:
            return DisconnectPlan((), False, True, "usb-ambiguous")
        close = tuple(s for s in ("left", "right") if s in open_set)
        return DisconnectPlan(close, True, False, "左右相机 USB 已断开")

    shared = set(left_ids) == set(right_ids) and len(set(left_ids)) > 1
    left_hit = _hits(left_ids, removed) if left_ids else False
    right_hit = _hits(right_ids, removed) if right_ids else False
    if shared and (left_hit or right_hit):
        return DisconnectPlan((), False, True, "usb-ambiguous")
    if not left_ids and not right_ids:
        if sole_device:
            close = tuple(s for s in ("left", "right") if s in open_set)
            return DisconnectPlan(close, True, False, "左右相机 USB 已断开")
        return DisconnectPlan((), False, True, "usb-ambiguous")

    close_list = []
    if left_hit and "left" in open_set:
        close_list.append("left")
    if right_hit and "right" in open_set:
        close_list.append("right")
    if not close_list:
        return DisconnectPlan((), False, False, "")
    if close_list == ["left"]:
        reason = "左相机 USB 已断开"
    elif close_list == ["right"]:
        reason = "右相机 USB 已断开"
    else:
        reason = "左右相机 USB 已断开"
    return DisconnectPlan(tuple(close_list), True, False, reason)


def plan_stream_start(*, cameras_ready: bool, model_ready: bool, warmed: bool) -> StartPlan:
    """相机、模型、预热各自是否还要做。两边都要做时并行。"""
    open_cameras = not cameras_ready
    load_model = not model_ready
    warmup = not warmed
    parallel = open_cameras and load_model
    if not open_cameras and not load_model and not warmup:
        note = "相机与模型已就绪，只启动采集和推理线程"
    elif parallel:
        note = "打开相机与加载模型并行，预热等两边都完成"
    elif open_cameras:
        note = "模型已在内存里，这次只打开相机并预热"
    elif load_model or warmup:
        note = "相机已经打开，这次只加载或预热模型"
    else:
        note = "只补未完成的步骤"
    return StartPlan(open_cameras, load_model, warmup, parallel, note)


class UsbCameraWatch:
    """轮询在线相机实例，只报告「上一轮还在、这一轮没了」的 ID。"""

    def __init__(self, lister: Optional[Callable[[], Optional[Set[str]]]] = None) -> None:
        self._lister = lister or list_present_camera_ids
        self._prev: Optional[Set[str]] = None
        self._stable: Optional[Set[str]] = None
        self._pending: Optional[Set[str]] = None
        self._pending_hits = 0

    def poll(self) -> Tuple[str, ...]:
        removed, _arrived = self.poll_changes()
        return removed

    def poll_changes(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """返回 (拔出的 ID, 新出现的 ID)。

        第一轮只建立基线。同一变化要连续两轮都一样才上报，
        打开相机时接口闪一下不会被当成拔插。
        """
        try:
            now = self._lister()
        except Exception:  # noqa: BLE001
            return (), ()
        if now is None:
            # 枚举失败：保持上一轮快照，避免误报全部拔出
            return (), ()
        current = set(now)
        if self._stable is None:
            self._stable = set(current)
            self._pending = set(current)
            self._pending_hits = 1
            self._prev = set(current)
            return (), ()
        if self._pending == current:
            self._pending_hits += 1
        else:
            self._pending = set(current)
            self._pending_hits = 1
        self._prev = set(current)
        if self._pending_hits < 2 or self._pending == self._stable:
            return (), ()
        removed = tuple(sorted(self._stable - self._pending))
        arrived = tuple(sorted(self._pending - self._stable))
        self._stable = set(self._pending)
        return removed, arrived
