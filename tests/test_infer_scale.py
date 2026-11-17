"""推理分辨率：不放大超过标定图，视差换回标定像素后深度不变。

用法：
    python tests/test_infer_scale.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from depth_utils import CameraParams, disparity_to_depth  # noqa: E402
from stereo_depth import disparity_to_source, resolve_infer_scale  # noqa: E402


def test_scale_stays_on_the_calibrated_grid() -> None:
    assert resolve_infer_scale(640, 1.0) == 1.0
    assert resolve_infer_scale(640, 2.0) == 1.0
    assert abs(resolve_infer_scale(640, 0.5) - 0.5) < 1e-9
    capped = resolve_infer_scale(1600, 1.0)
    assert capped < 1.0
    assert 1600 * capped < 1000
    assert resolve_infer_scale(640, 0) == 1.0


def test_source_disparity_keeps_the_same_depth() -> None:
    """半分辨率视差配缩小后的内参，应和拉回标定图后的视差配原内参得到同一深度。"""
    disp = np.full((4, 4), 10.0, np.float32)
    full = disparity_to_source(disp, 8, 8)
    assert full.shape == (8, 8)
    assert abs(float(full.mean()) - 20.0) < 0.05

    params = CameraParams(fx=500.0, fy=500.0, cx=4.0, cy=4.0, baseline_m=0.06,
                          width=8, height=8)
    depth_full, valid_full = disparity_to_depth(full, params, max_valid_disp=40.0)
    depth_half, valid_half = disparity_to_depth(disp, params.scaled(0.5), max_valid_disp=20.0)
    assert valid_full.all() and valid_half.all()
    assert abs(float(depth_full.mean()) - float(depth_half.mean())) < 1e-4


def test_native_scale_is_the_config_default() -> None:
    text = open(os.path.join(ROOT, "config.yaml"), encoding="utf-8").read()
    assert "scale: 0.75" in text
    assert "preset: balanced" in text
    assert "max_disp: 192" in text


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
