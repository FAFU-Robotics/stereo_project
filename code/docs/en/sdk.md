# Python SDK

`stereo_sdk` is a thin, stable layer over capture, rectification, FFS inference and export.
The complete API reference (Chinese, table per function) is
[docs/SDK_API参考.md](../SDK_API参考.md); this page is the English quick reference.

## Install

From the repository root:

```bat
python -m pip install -e .
```

Or simply run examples inside the repository — `examples/_common.py` adds the paths.

## Minimal example

```python
from stereo_sdk import Session

with Session("config.yaml") as cam:        # __enter__ calls start(); 3–10 s (model load)
    while True:
        if not cam.has_new_result():
            continue
        r = cam.take_result()               # never blocks; may repeat the latest result
        print(r.depth[240, 320])            # meters, 0 = invalid
        pts, cols = r.points, r.colors      # (N,3) meters / (N,3) BGR
```

Run the bundled version: `python examples\01_read_depth.py --mock`.

## Conventions

| Item | Convention |
|---|---|
| Units | Meters for both `depth` and `points` |
| Frame | Camera optical: x right, **y down**, z forward |
| Colors | **BGR** (same as OpenCV) |
| `take_result()` | Never blocks, always returns the latest result, **may be the same frame**. Use `has_new_result()` to iterate |
| `depth` size | Same as `r.frame.left` (upsampled to full resolution) |
| Invalid pixels | `depth = 0`, `valid = False`, absent from `points` |

## Export

```python
from stereo_sdk import export
paths = export.export_result(r, "out", prefix="scan01")
# left/right png, disparity npy + 16-bit png, depth npy + 16-bit png (mm), valid png, cloud.ply
```

Binary PLY is about 60% smaller than ASCII. Colors in PLY are RGB.

## Calibration info

```python
from stereo_sdk import CalibInfo
info = CalibInfo.load("calibration/stereo_calib.yaml")
print(info.baseline_mm, info.rms_stereo, info.rect_fx)
print(info.sigma_z(1.0))     # estimated depth sigma at 1 m
```

## Runtime parameters

```python
cam.set_params(zmin=0.2, zmax=2.0)        # applied to the next frame
cam.set_params(depth_range=(0.2, 2.0))
cam.reload_calibration()                  # after saving a new yaml, no restart needed
```

`scale` and `valid_iters` require a model restart; `set_params` reports which keys were
accepted and which need a restart.

## Examples

| File | What it shows |
|---|---|
| `01_read_depth.py` | One frame, one pixel, export |
| `02_save_pointcloud.py` | N frames of point clouds + end-to-end FPS |
| `03_live_view.py` | Live depth with mouse readout |
| `04_calib_report.py` | Calibration report: usable range, expected error |
| `05_depth_statistics.py` | Drift/quality over time |
| `06_save_dataset.py` | Reproducible mini dataset with index |

All examples accept `--mock` (no camera, no GPU) except `04`, which only reads the YAML.

## Threading

`Session` owns the camera and runs capture/inference in background threads; `take_result()` is
safe to call from the UI thread. Do not open two sessions in one process — the camera is
exclusive.
