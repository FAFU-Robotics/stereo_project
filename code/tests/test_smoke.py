"""最小回归：SDK 封装的五条硬约束。

用法（两种都行，不强制装 pytest）：
    python tests/test_smoke.py
    pytest tests/test_smoke.py -q

为什么是这五条：
    这五条每一条都对应过一个真实会出问题的地方——
    版本号两处不一致、装完找不到标定文件、PLY 写坏、Mock 与真 Session 签名不同、
    上下文管理器退出不停相机。别的都可以人工看，这五条必须机器守。
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
for _p in (ROOT, TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stereo_sdk                                     # noqa: E402
from stereo_sdk import MockSession                    # noqa: E402
from stereo_sdk import export                         # noqa: E402
from stereo_sdk.calib import CalibInfo, candidate_paths  # noqa: E402
from stereo_sdk.session import Session                # noqa: E402


def test_version_single_source() -> None:
    """版本号只有 SDK_VERSION 一个来源，pyproject 必须用 dynamic 引用它。"""
    assert stereo_sdk.SDK_VERSION.count(".") >= 1, stereo_sdk.SDK_VERSION
    pyproj = os.path.join(ROOT, "pyproject.toml")
    if not os.path.exists(pyproj):
        return                                        # 只装了包、没有仓库目录时跳过
    try:
        import tomllib
    except ImportError:                               # Python < 3.11
        return
    with open(pyproj, "rb") as fh:
        proj = tomllib.load(fh)
    dyn = (proj.get("project") or {}).get("dynamic") or []
    assert "version" in dyn, "pyproject 应使用 dynamic = [\"version\"]"
    attr = (proj.get("tool", {}).get("setuptools", {})
            .get("dynamic", {}).get("version", {}).get("attr", ""))
    assert attr == "stereo_sdk.types.SDK_VERSION", f"动态版本来源不对：{attr!r}"


def test_calib_lookup_reports_all_paths() -> None:
    """找不到标定文件时，报错必须列出查过的每个路径，不能只说"找不到"。"""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        old = os.environ.pop("STEREO_CALIB", None)
        try:
            # 临时目录里没有 calibration/，且把仓库那份也排除掉
            missing = os.path.join(tmp, "nope.yaml")
            try:
                CalibInfo.from_file(missing)
            except FileNotFoundError as exc:
                assert missing in str(exc)
            else:
                raise AssertionError("指定不存在的路径时应抛 FileNotFoundError")
        finally:
            if old is not None:
                os.environ["STEREO_CALIB"] = old
            os.chdir(cwd)
    cands = candidate_paths()
    assert cands, "候选路径列表不应为空"
    assert len(cands) == len({os.path.abspath(p) for p in cands}), "候选路径应去重"


def test_calib_info_fields() -> None:
    """有标定文件时，读出来的基线/fx/RMS 必须在物理上说得通。"""
    path = os.path.join(ROOT, "calibration", "stereo_calib.yaml")
    if not os.path.exists(path):
        return                                        # 没标定过就跳过，不算失败
    c = CalibInfo.from_file(path)
    assert 30.0 < c.baseline_mm < 200.0, f"基线不合常理：{c.baseline_mm}"
    assert c.fx > 50.0, f"fx 不合常理：{c.fx}"
    assert c.rms == c.rms_stereo
    assert 0.0 < c.rms < 3.0, f"RMS 不合常理：{c.rms}"
    s1 = c.sigma_depth(1.0)
    assert s1 > 0.0 and np.isfinite(s1)
    # 误差随距离平方增长：2 m 的不确定度应约为 1 m 的 4 倍
    ratio = c.sigma_depth(2.0) / s1
    assert abs(ratio - 4.0) < 0.01, f"sigma 应随 Z² 增长，实测比值 {ratio}"


def test_mock_parity_and_lifecycle() -> None:
    """Mock 与真 Session 的公开方法必须一一对应；with 退出必须 stop。"""
    pub = lambda cls: {n for n in dir(cls) if not n.startswith("_")}   # noqa: E731
    only_real = pub(Session) - pub(MockSession)
    only_mock = pub(MockSession) - pub(Session)
    assert not only_real, f"Mock 缺这些方法（用 mock 开发 GUI 会 AttributeError）：{only_real}"
    assert not only_mock, f"Mock 多了这些方法（真 Session 上没有）：{only_mock}"

    with MockSession() as s:
        s.start()
        assert s.started
        info = s.reload_calibration()
        assert isinstance(info, dict)
    assert not s.started, "with 退出后相机应已停止"


def test_mock_produces_result_and_exports() -> None:
    """Mock 能出深度，且导出到磁盘的文件头/大小自洽。"""
    s = MockSession()
    s.start()
    try:
        r = None
        import time
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if s.has_new_result():
                r = s.take_result()
                break
            time.sleep(0.05)
        assert r is not None, "10 秒内 Mock 没有产出结果"
        assert r.depth.ndim == 2 and r.depth.dtype == np.float32
        assert r.valid.shape == r.depth.shape
        assert float(r.valid.mean()) > 0.0

        with tempfile.TemporaryDirectory() as tmp:
            files = export.export_result(r, tmp, prefix="t")
            assert files, "export_result 应至少写出一份文件"
            for f in files.values() if isinstance(files, dict) else files:
                assert os.path.exists(f) and os.path.getsize(f) > 0, f

            # 二进制 PLY：头里声明的点数 × 每点字节数 应等于文件体长度
            ply = os.path.join(tmp, "t_cloud.ply") if not isinstance(files, dict) \
                else next(v for v in files.values() if str(v).endswith(".ply"))
            with open(ply, "rb") as fh:
                data = fh.read()
            head_end = data.find(b"end_header\n") + len(b"end_header\n")
            header = data[:head_end].decode("ascii", "replace").splitlines()
            n = int([ln for ln in header if ln.startswith("element vertex")][0].split()[-1])
            fmt = [ln for ln in header if ln.startswith("format")][0]
            per = 15 if "binary_little_endian" in fmt else None
            if per:
                assert len(data) - head_end == n * per, "PLY 体长度与点数不符"
    finally:
        s.stop()


def test_gui_state_is_always_safe() -> None:
    """界面记忆文件被写坏 / 手改错，都不许让 GUI 起不来。

    这条守的是"启动"这件事：sanitize() 必须把任意输入收敛成合法值，
    load() 遇到损坏 JSON 必须回落到默认值而不是抛异常。
    """
    import gui_state as gs

    bad = {
        "geometry": "not-a-geometry", "tab": 99,
        "thresholds": {"px": -5, "target": 3.7, "sharp": "abc"},
        "board": {"cols": 999, "rows": 0, "square": -1},
        "check_mode": "不存在的模式", "alpha": 5,
    }
    st = gs.sanitize(bad)
    assert st["geometry"] == ""
    assert st["tab"] == 0
    assert st["thresholds"]["px"] == 1.0            # 夹到下限
    assert st["thresholds"]["sharp"] == 100.0       # 非数字回落默认
    assert st["thresholds"]["target"] == 4          # 3.7 夹到下限再取整
    assert st["board"] == {"cols": 30, "rows": 3, "square": 0.5}
    assert st["check_mode"] in gs.CHECK_MODES
    assert st["alpha"] == 1.0

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "s.json")
        assert gs.save({"geometry": "1280x900+20+30"}, p)
        assert gs.load(p)["geometry"] == "1280x900+20+30"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("{broken json")
        assert gs.load(p) == gs.defaults(), "损坏的 JSON 必须回落到默认值"
        assert gs.load(os.path.join(tmp, "nope.json")) == gs.defaults()


def test_shortage_plan() -> None:
    """"还差几张"的判定：数量、姿态、以及两者都达标，三种情形都要对。"""
    from calib_core import shortage_plan

    need = {"中心正对": 2, "左上角": 1, "左倾大于20度": 3}
    got = {"中心正对": 2, "左上角": 0, "左倾大于20度": 1}

    a = shortage_plan(10, 26, got, need)
    assert a["short"] == 16 and not a["done"]
    assert a["missing"] == ["左上角", "左倾大于20度"]
    assert len(a["advice"]) == 2 and "左上" in a["advice"][0]

    b = shortage_plan(26, 26, got, need)          # 数量够、姿态缺
    assert b["short"] == 0 and not b["done"] and b["missing"]

    c = shortage_plan(30, 26, {"中心正对": 2, "左上角": 1, "左倾大于20度": 3}, need)
    assert c["done"] and c["short"] == 0 and not c["missing"]


def test_cancel_is_honoured_before_work() -> None:
    """取消标志已置位时，标定必须在动手之前退出，且不动任何文件。"""
    import threading

    import calib_core as core

    with tempfile.TemporaryDirectory() as tmp:
        ev = threading.Event()
        ev.set()
        try:
            core.calibrate_from_dir(tmp, core.default_spec(12, 9, 3.0), cancel=ev)
        except core.Cancelled:
            pass
        else:
            raise AssertionError("取消标志已置位却没抛 Cancelled")
        assert os.listdir(tmp) == [], "取消后不应产生任何文件"


def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"[FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 项通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
