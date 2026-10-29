# Installation

Two routes: use the prebuilt EXE (no Python), or build the environment from source.

---

## Route A — prebuilt EXE

### A1. Get the archive

Download every part of the release (`StereoCamera-v0.3.0.7z.001`, `.002`, `.003`) into the
same folder. The EXE alone is not runnable: the `_internal` directory beside it contains the
CUDA libraries and the Python runtime.

### A2. Extract

Right-click the `.001` part → **Extract here** with 7-Zip. You should get a `StereoCamera`
folder containing `StereoCamera.exe`, `_internal\`, `config.yaml`, `calibration\`, `weights\`.

### A3. Check prerequisites

- Windows 10/11 x64
- NVIDIA GPU with a driver supporting CUDA 12.8 (RTX 50 series requires it)
- The stereo camera connected directly to a USB port (avoid unpowered hubs)

### A4. Start and verify

1. Double-click `StereoCamera.exe` — the **Stereo Viewer** opens directly (no UAC prompt,
   administrator rights are not required).
2. Click **Environment check** in the left panel. Every item should be `[OK]` and CUDA must be
   available.
3. Click **Start streaming** (top right). The status changes to streaming and shows the
   resolution and FPS.
4. Switch to **2D** and enable **Color**, then **Depth**; or switch to **3D** for the point
   cloud. The first depth frame loads the model and takes a few seconds.
5. To calibrate, click **Calibration console**. The viewer stops streaming first, because the
   camera can only be owned by one window.

Full operating notes: [usage.md](usage.md).

---

## Route B — from source

All commands run in the repository root (the folder containing `config.yaml`).

### B1. Choose the interpreter

- Python **3.14.6** is what this project is tested on.
- Do not use a Python that lacks `tkinter` (the GUIs will not start).
- Avoid mixing this project with conda environments that pin an older OpenCV.

### B2. Install PyTorch first

Order matters: installing `torch` from the default PyPI index gives a CPU-only wheel.

```bat
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

Notes:

- RTX 50 series is `sm_120`; CUDA 12.4 wheels do not contain those kernels.
- On Python 3.14 only the cu128 line provides `cp314` wheels (2.9.0 / 2.9.1).

### B3. Install the rest

```bat
python -m pip install -r requirements.txt
```

If `import open3d` fails with `No module named 'plotly'` / `'dash'`, install them:

```bat
python -m pip install plotly dash
```

### B4. Fetch the weights

```bat
python tools\fetch_ffs_weights.py
```

The file lands in `weights\23-36-37\model_best_bp2_serialize.pth` (67.8 MiB). Weights are not
tracked in git.

### B5. Self-check

```bat
python app_entry.py --selftest
```

This checks paths, dependencies and CUDA without touching the camera.

### B6. Run

| Purpose | Command |
|---|---|
| Main window (same as double-clicking the EXE) | `python app_entry.py` |
| UI self-check (no camera) | `python app_entry.py --gui-selftest` |
| Calibration console | `python app_entry.py --gui` |
| CLI preview / depth / cloud | `python main.py` / `--mode depth` / `--mode pointcloud` |
| English UI | add `--lang en` to any of the above |

### B7. Optional: install as a Python package

```bat
python -m pip install -e .
```

Then `from stereo_sdk import Session` works outside the repository. See [sdk.md](sdk.md).

---

## Troubleshooting

| Symptom | Check |
|---|---|
| `CameraError: cannot open index 1` | Another app owns the camera; close it. Then confirm the index in `config.yaml` |
| `torch.compile is not supported on Python 3.14+` | You are running code that bypassed the compatibility shim; use the repository entry points |
| `ModuleNotFoundError: omegaconf` | Install it: the weights embed a `DictConfig` |
| Depth view stays black | Model still loading, or streaming not started |
| `open3d` import error | Install `plotly` and `dash`, and use open3d 0.20.0 on Python 3.14 |

More: [faq.md](faq.md).
