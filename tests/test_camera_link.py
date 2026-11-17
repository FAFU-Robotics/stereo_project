"""相机连接、USB 断开和串流启动步骤。不打开真实相机。

覆盖：
    · 拔掉的是别的 USB 设备时不动连接
    · side_by_side 拔掉绑定的那台：左右一起关，并要求走软件断开
    · dual 能分清左右时只关掉线的那一路
    · 对不上号时先探测，不立刻拆掉
    · 枚举失败（None）不能被当成「全部拔出」
    · 同一路断开两次是空操作
    · 串流已开始后再收到同一次拔出，started / 锁 / 剩余句柄保持一致
    · 相机和模型都准备好时，开始串流不再打开相机、不再加载模型
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from camera_link import (UsbCameraWatch, bind_snapshot, plan_stream_start,  # noqa: E402
                         plan_usb_disconnect)
from stereo_sdk.session import Session  # noqa: E402


LEFT = "USB\\VID_0BDA&PID_5883\\left"
RIGHT = "USB\\VID_046D&PID_0825\\right"
OTHER = "USB\\VID_AAAA&PID_BBBB\\other"


class _Cap:
    def __init__(self) -> None:
        self.released = False
        self.open = True

    def isOpened(self) -> bool:  # noqa: N802 —— 跟 cv2.VideoCapture 同名
        return self.open and not self.released

    def release(self) -> None:
        self.released = True
        self.open = False


class _Pipe:
    def __init__(self) -> None:
        self.error = None
        self.stopped = False

    def stop(self, timeout: float = 3.0) -> None:
        self.stopped = True


class _Cam:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.sides = ["left", "right"]
        self.closed = False

    def open_sides(self):
        return list(self.sides)

    def disconnect_side(self, side: str):
        if self.mode == "side_by_side":
            had = list(self.sides)
            self.sides = []
            self.closed = True
            return had
        if side in self.sides:
            self.sides.remove(side)
            return [side]
        return []

    def close(self) -> None:
        self.sides = []
        self.closed = True

    def dead_sides(self):
        return []

    def probe_sides(self):
        return []

    def describe(self) -> str:
        return self.mode


def _session(mode: str, bound, sole: bool, streaming: bool) -> Session:
    session = Session("config.yaml")
    pipe = _Pipe()
    session._cam = _Cam(mode)
    session._pipe = pipe
    session._started = streaming
    session._lock_held = True
    session._cameras_ready = True
    session._retain_cameras = True
    session._usb_bound = bound
    session._sole_device = sole
    session._cfg = {"camera": {}}
    return session


def test_unrelated_usb_removal_is_ignored() -> None:
    plan = plan_usb_disconnect(
        mode="side_by_side",
        open_sides=("left", "right"),
        bound={"left": (LEFT,), "right": (LEFT,)},
        removed_ids=[OTHER],
        streaming=True,
        sole_device=False,
    )
    assert plan.close_sides == ()
    assert plan.software_disconnect is False
    assert plan.needs_probe is False


def test_side_by_side_removal_closes_both_and_requests_software_stop() -> None:
    plan = plan_usb_disconnect(
        mode="side_by_side",
        open_sides=("left", "right"),
        bound={"left": (LEFT,), "right": (LEFT,)},
        removed_ids=[LEFT],
        streaming=True,
        sole_device=True,
    )
    assert plan.close_sides == ("left", "right")
    assert plan.software_disconnect is True
    assert "左右" in plan.reason


def test_dual_removal_closes_only_that_side() -> None:
    plan = plan_usb_disconnect(
        mode="dual",
        open_sides=("left", "right"),
        bound={"left": (LEFT,), "right": (RIGHT,)},
        removed_ids=[RIGHT],
        streaming=True,
    )
    assert plan.close_sides == ("right",)
    assert plan.software_disconnect is True
    assert plan.reason == "右相机 USB 已断开"


def test_ambiguous_removal_asks_for_a_probe() -> None:
    both = (LEFT, RIGHT)
    plan = plan_usb_disconnect(
        mode="dual",
        open_sides=("left", "right"),
        bound={"left": both, "right": both},
        removed_ids=[RIGHT],
        streaming=True,
    )
    assert plan.close_sides == ()
    assert plan.needs_probe is True
    assert plan.software_disconnect is False


def test_extra_camera_unplug_does_not_drop_the_stereo_device() -> None:
    """没写 usb_match、打开时有两台相机：拔掉其中一台先探测，不直接拆双目。"""
    plan = plan_usb_disconnect(
        mode="side_by_side",
        open_sides=("left", "right"),
        bound={"left": (LEFT, OTHER), "right": (LEFT, OTHER)},
        removed_ids=[OTHER],
        streaming=True,
        sole_device=False,
    )
    assert plan.close_sides == ()
    assert plan.needs_probe is True
    assert plan.software_disconnect is False


def test_usb_match_ignores_other_vendors() -> None:
    plan = plan_usb_disconnect(
        mode="side_by_side",
        open_sides=("left", "right"),
        bound={"left": (LEFT,), "right": (LEFT,)},
        removed_ids=[OTHER],
        usb_match="VID_0BDA&PID_5883",
        streaming=True,
        sole_device=False,
    )
    assert plan.close_sides == ()


def test_read_error_maps_to_the_failed_side() -> None:
    plan = plan_usb_disconnect(
        mode="dual",
        open_sides=("left", "right"),
        bound={"left": (LEFT,), "right": (RIGHT,)},
        removed_ids=[],
        read_error="left 相机连续读取失败（第 3 次读取失败）。",
        streaming=True,
    )
    assert plan.close_sides == ("left",)
    assert plan.software_disconnect is True


def test_matched_usb_collapses_to_one_presence_token() -> None:
    from camera_link import list_watched_ids
    import camera_link

    seen = [
        {"USB\\VID_0BDA&PID_5883&MI_00\\a", "USB\\VID_0BDA&PID_5883&MI_02\\b", "USB\\VID_04F2&PID_B828\\c"},
        set(),
    ]

    def fake(_filt, _flags):
        return set(seen.pop(0))

    original = camera_link._cm_id_list
    camera_link._cm_id_list = fake
    try:
        assert list_watched_ids("VID_0BDA&PID_5883") == {"vid_0bda&pid_5883"}
        assert list_watched_ids("VID_0BDA&PID_5883") == set()
    finally:
        camera_link._cm_id_list = original


def test_watch_reports_plug_in_separately_from_unplug() -> None:
    steps = [{"A"}, {"A", "B"}, {"A", "B"}, {"B"}, {"B"}]

    def lister():
        return set(steps.pop(0))

    watch = UsbCameraWatch(lister)
    assert watch.poll_changes() == ((), ())
    assert watch.poll_changes() == ((), ())          # 只出现一轮，先不报
    removed, arrived = watch.poll_changes()
    assert removed == () and arrived == ("B",)       # 连续两轮才算插上
    assert watch.poll_changes() == ((), ())
    removed, arrived = watch.poll_changes()
    assert removed == ("A",) and arrived == ()


def test_force_disconnect_stops_stream_and_releases_the_handle() -> None:
    session = _session(
        "side_by_side",
        {"left": (LEFT,), "right": (LEFT,)},
        sole=True,
        streaming=True,
    )
    info = session.force_disconnect("相机无画面，已停止串流")
    assert info["changed"] is True
    assert info["streaming_stopped"] is True
    assert info["remaining"] == []
    assert session.started is False
    assert session._cam is None
    assert session._lock_held is False
    again = session.force_disconnect("相机无画面，已停止串流")
    assert again["changed"] is False
    assert session.started is False


def test_watch_ignores_enumeration_failure_but_sees_real_unplug() -> None:
    steps = [{"A"}, None, {"A"}, set(), set()]

    def lister():
        return steps.pop(0)

    watch = UsbCameraWatch(lister)
    assert watch.poll() == ()          # 第一轮只建立基线
    assert watch.poll() == ()          # None = 枚举失败，不能当成全拔出
    assert watch.poll() == ()          # 设备还在
    assert watch.poll() == ()          # 只消失一轮，先不报
    assert watch.poll() == ("A",)      # 连续两轮都没了，才是真拔出


def test_bind_snapshot_splits_dual_by_token() -> None:
    bound = bind_snapshot("dual", [LEFT, RIGHT, OTHER], left_usb="0BDA", right_usb="046D")
    assert bound["left"] == (LEFT,)
    assert bound["right"] == (RIGHT,)


def test_start_plan_skips_work_already_done() -> None:
    cold = plan_stream_start(cameras_ready=False, model_ready=False, warmed=False)
    assert cold.parallel is True
    assert cold.open_cameras and cold.load_model and cold.warmup
    hot = plan_stream_start(cameras_ready=True, model_ready=True, warmed=True)
    assert hot.parallel is False
    assert not hot.open_cameras and not hot.load_model and not hot.warmup


def test_side_disconnect_is_idempotent_and_read_names_the_side() -> None:
    from camera import CameraReadError, StereoCamera

    dual = StereoCamera({
        "mode": "dual", "left_index": 0, "right_index": 1,
        "backend": "dshow", "width": 640, "height": 480, "warmup_frames": 0,
    })
    left, right = _Cap(), _Cap()
    dual._caps = [left, right]
    dual._opened = True
    assert dual.open_sides() == ["left", "right"]
    same_left, same_right = dual._caps
    dual.open()  # 两路都活着，不能去碰真实设备
    assert dual._caps[0] is same_left and dual._caps[1] is same_right

    assert dual.disconnect_side("right") == ["right"]
    assert right.released and not left.released
    assert dual.open_sides() == ["left"]
    assert dual.disconnect_side("right") == []

    dual._caps = [None, _Cap()]
    dual._opened = True
    try:
        dual.read()
    except CameraReadError as exc:
        assert "left" in str(exc)
    else:
        raise AssertionError("左路已空时 read() 应该立刻失败")

    sbs = StereoCamera({
        "mode": "side_by_side", "single_index": 1, "backend": "dshow",
        "width": 1280, "height": 480, "warmup_frames": 0,
    })
    cap = _Cap()
    sbs._caps = [cap]
    sbs._opened = True
    assert sbs.disconnect_side("left") == ["left", "right"]
    assert cap.released
    assert sbs.disconnect_side("left") == []
    assert sbs.open_sides() == []


def test_usb_removal_stops_stream_and_second_event_is_a_noop() -> None:
    session = _session(
        "side_by_side",
        {"left": (LEFT,), "right": (LEFT,)},
        sole=True,
        streaming=True,
    )
    pipe = session._pipe
    info = session.apply_usb_removal([LEFT])
    assert info["changed"] is True
    assert info["streaming_stopped"] is True
    assert info["remaining"] == []
    assert session.started is False
    assert session._cam is None
    assert session._lock_held is False
    assert pipe.stopped is True

    again = session.apply_usb_removal([LEFT])
    assert again["changed"] is False
    assert session.started is False


def test_dual_usb_removal_keeps_the_other_camera_and_the_lock() -> None:
    session = _session(
        "dual",
        {"left": (LEFT,), "right": (RIGHT,)},
        sole=False,
        streaming=True,
    )
    info = session.apply_usb_removal([LEFT])
    assert info["closed"] == ["left"]
    assert info["remaining"] == ["right"]
    assert info["streaming_stopped"] is True
    assert session.started is False
    assert session.cameras_ready is False
    assert session._cam is not None
    assert session._lock_held is True
    assert session._cam.open_sides() == ["right"]

    # 右路已经不在线，再报一次左路拔出不能把右路也关掉，也不能把锁放掉。
    again = session.apply_usb_removal([LEFT])
    assert again["changed"] is False
    assert session._lock_held is True
    assert session._cam.open_sides() == ["right"]


def test_runtime_read_loss_uses_the_same_disconnect_path() -> None:
    session = _session(
        "dual",
        {"left": (LEFT,), "right": (RIGHT,)},
        sole=False,
        streaming=True,
    )
    session._pipe.error = RuntimeError("right 相机连续读取失败")
    assert session.poll_runtime_loss() is True
    info = session.consume_runtime_loss()
    assert info["streaming_stopped"] is True
    assert info["remaining"] == ["left"]
    assert session.poll_runtime_loss() is False
    second = session.consume_runtime_loss()
    assert second["changed"] is False


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 项通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
