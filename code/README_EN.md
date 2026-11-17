# Open Stereo Camera

A stereo depth system for Windows: dual-camera capture, stereo rectification, FFS-based
disparity/depth inference, colored point clouds, stereo calibration, and a Python SDK.

> This is the English overview. The Chinese documentation is the primary reference and is
> more detailed; see [README.md](README.md) and [docs/](docs/).

---

## Quick start

### Option A — use the prebuilt EXE (no Python required)

1. Download the split archive from Releases (`StereoCamera-v0.3.0.7z.001` … `.003`) and put
   all parts in one folder.
2. Extract the first part with 7-Zip. **Do not copy only the .exe** — the `_internal` folder
   is part of the application.
3. An NVIDIA GPU with a recent driver (CUDA 12.8 capable) and the stereo camera are required.
4. Double-click `StereoCamera.exe` → the **Stereo Viewer** opens → click **Start streaming**
   at the top right → use **2D** (Color/Depth/Disparity) or **3D** (point cloud).

The UI is available in Chinese and English. Use the **Language** dropdown at the top right of
the viewer to switch instantly, or set `ui.language: en` in `config.yaml`, or launch with
`--lang en`.

### Option B — run from source

```bat
git clone https://github.com/FAFU-Robotics/stereo_project.git
cd stereo_project\code

REM 1) PyTorch first (the default PyPI wheel is CPU-only)
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

REM 2) Remaining dependencies
python -m pip install -r requirements.txt

REM 3) Model weights (67.8 MiB, standard-library downloader)
python tools\fetch_ffs_weights.py

REM 4) Environment check (no camera needed)
python app_entry.py --selftest

REM 5) Run
python app_entry.py                 REM main window (device / depth / cloud / capture / calibration)
python app_entry.py --gui           REM calibration console
python main.py --mode depth         REM CLI depth window
```

Detailed steps and troubleshooting: [docs/en/installation.md](docs/en/installation.md).

---

## Entry points

| Entry | Command | Purpose |
|---|---|---|
| **Stereo Viewer** (default) | `python app_entry.py` / double-click EXE | Device status, color/depth/cloud views, snapshots, calibration entry |
| Calibration console | `python app_entry.py --gui` | Capture → calibrate → inspect rectification → save |
| CLI preview | `python main.py` | Side-by-side stereo preview |
| CLI depth | `python main.py --mode depth` | Disparity/depth window with FPS |
| CLI point cloud | `python main.py --mode pointcloud` | Depth window + Open3D colored cloud |

## Python SDK

```python
from stereo_sdk import Session

with Session("config.yaml") as cam:
    while True:
        if not cam.has_new_result():
            continue
        r = cam.take_result()
        print(r.depth[240, 320])       # meters, 0 = invalid
        pts, cols = r.points, r.colors  # (N,3) meters / (N,3) BGR
```

Units and conventions: depth and points are in **meters**; camera frame is **x right, y down,
z forward**; colors are **BGR**. `take_result()` never blocks and may return the same frame;
use `has_new_result()` before consuming. See [docs/en/sdk.md](docs/en/sdk.md).

## Calibration

The repository ships a verified stereo calibration (`calibration/stereo_calib.yaml`):

- stereo RMS **0.1246 px**, baseline **64.891 mm**, rectified fx **547.02 px**
- board: 12×9 squares, 3 mm, inner corners 11×8

To redo calibration, use the calibration console: capture 15–30 poses → calibrate →
inspect row alignment → save. Old files are archived under `calibration/archive/`.

## Performance (reference)

Measured on RTX 5060 Laptop with a 1280×480 MJPEG stream. The detailed report
([docs/性能测试报告.md](docs/性能测试报告.md)) was recorded on Python 3.12 + torch 2.8;
the current environment is Python 3.14.6 + torch 2.9.1+cu128 and has not been fully re-benched.
Treat the numbers as relative comparisons, not guarantees.

| Preset | scale / valid_iters | Inference | Approx. end-to-end |
|---|---|---|---|
| Fast | 0.5 / 4 | ~50 ms | ~20 FPS |
| Balanced | 0.75 / 8 | ~90 ms | ~10 FPS |
| Precise | 1.0 / 12 | ~150 ms | ~6 FPS |

## Repository layout

```
stereo_sdk/     public Python API (session, types, calib, export, recorder, mock)
tools/          calibration console, Stereo Viewer, packaging helpers, .bat launchers
examples/       six runnable SDK examples (--mock supported)
tests/          offline pipeline test and SDK smoke test
docs/           documentation (Chinese primary, English under docs/en/)
packaging/      PyInstaller spec, build script, packaging test report
third_party/    Fast-FoundationStereo source (do not modify)
```

Model weights are not tracked in git; fetch them with `python tools\fetch_ffs_weights.py`.

## License

Original code and documentation are MIT (see [LICENSE](LICENSE)).
`third_party/Fast-FoundationStereo/` is under the NVIDIA Source Code License, and the
`23-36-37` weights are research/evaluation only. See [NOTICE.md](NOTICE.md) for the full
boundary — the MIT license alone does not cover commercial use of those parts.

## Known limitations

- One process at a time can own the camera (browser video pages, meeting apps, or a second
  instance will block it).
- The EXE has only been validated on the development machine.
- Single-file EXE is not provided: the CUDA libraries are ~4.3 GB and would be unpacked on
  every launch.
- 3D interaction (orbit/pan/zoom) is manual-tested only.

---

English docs index: [docs/en/README.md](docs/en/README.md).
