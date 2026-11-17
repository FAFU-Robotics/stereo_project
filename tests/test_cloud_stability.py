"""点云稳定性：视差尖峰和孤立散斑要被去掉，成片平面要留下。

用法：
    python tests/test_cloud_stability.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from depth_utils import (filter_disp_spikes, filter_speckles,  # noqa: E402
                         scaled_speckle_size)
from stereo_sdk.session import PRESETS  # noqa: E402


def test_precise_preset_is_gone() -> None:
    assert "precise" not in PRESETS
    assert set(PRESETS) == {"balanced", "fast"}
    assert PRESETS["balanced"] == (0.75, 8)
    assert PRESETS["fast"] == (0.5, 4)


def test_spike_is_removed_and_plane_stays() -> None:
    disp = np.full((16, 16), 40.0, np.float32)
    disp[8, 8] = 120.0          # 孤立近处飞点
    valid = np.ones(disp.shape, bool)
    kept = filter_disp_spikes(disp, valid, tol_px=4.0)
    assert not kept[8, 8]
    assert kept.sum() == disp.size - 1


def test_speckle_threshold_follows_resolution() -> None:
    full = scaled_speckle_size((480, 640), 200)
    half = scaled_speckle_size((240, 320), 200)
    mid = scaled_speckle_size((360, 480), 200)
    assert full == 200
    assert half == 50
    assert 100 <= mid <= 120
    # 半分辨率上 40 像素的碎块应被删掉，整幅平面留下
    small = np.zeros((240, 320), bool)
    small[10:14, 10:20] = True          # 40 px
    small[40:140, 40:140] = True        # 大平面
    out = filter_speckles(small, half)
    assert not out[10:14, 10:20].any()
    assert out[40:140, 40:140].all()


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
