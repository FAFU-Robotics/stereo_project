# Usage

## 1. Three entry points

| Entry | How to open | Owns the camera |
|---|---|---|
| **Stereo Viewer** (default) | Double-click the EXE, or `python app_entry.py` | After you click Start streaming |
| **Calibration console** | Button in the viewer, or `python app_entry.py --gui` | After the camera is opened |
| **CLI window** | `python main.py`, or the `.bat` launcher menu | While running |

Only one entry can stream at a time. Stop streaming before handing the camera over.

## 2. Stereo Viewer

### Layout

| Area | Contents |
|---|---|
| Left top | Device status, resolution and FPS |
| Left middle | Inference preset, depth-range slider |
| Left buttons | Calibration console / Environment check / Model check |
| Center | `2D` and `3D` tabs; `Color (L)`, `Color (R)`, `Depth`, `Disparity`, `Point cloud` |
| Right top | Current calibration summary (RMS, baseline, intrinsics) |
| Right middle | Captures & snapshots list, Refresh, Open folder |
| Right bottom | Notifications |
| Top right | **Start streaming** / **Stop streaming** |
| Top right | `Language` selector (中文 / English) |

### Streaming

1. Click **Start streaming**. Status: Connecting → Streaming, with resolution and FPS.
2. The inference model loads the first time you open Depth or Point cloud (a few seconds).
3. **Stop streaming** releases the camera; closing the window also releases it.

### Language

The **`Language`** dropdown at the top right switches the whole window between
Chinese and English:

- the change is applied immediately — no restart, and an active stream keeps running;
- the choice is written back to `ui.language` in `config.yaml` and used on the next launch;
- `StereoCamera.exe --lang en` overrides the config for one session.

> The calibration console (the four-page workflow) is Chinese-only for now.
> Low-level libraries may still return error text in Chinese; it is shown verbatim
> in the Notifications panel.

### Views and mouse

- `2D`: enable any combination of Color (L/R), Depth, Disparity.
- `3D`: colored point cloud — **left drag to orbit, right drag to pan, wheel to zoom**.
- Hovering over the depth view shows the distance at the cursor.
- The depth color scale follows the depth-range slider (near warm, far cool).

### Shortcuts

| Key | Action |
|---|---|
| `Space` | Pause / resume display updates |
| `s` | Save a snapshot to `output\captures\` |
| `1` | Color view |
| `2` | Depth view |
| `3` | 3D point-cloud view |

### Calibration and snapshots

- **Snapshot** (`s`): creates a timestamped folder under `output\captures\`.
- **Review**: the right-hand list merges `output\calib` (calibration captures) and
  `output\captures` (snapshots). Select an item to review it; **Open folder** locates it in
  Explorer.
- **Calibration**: the button stops streaming, opens the four-page console, and the summary on
  the right refreshes when you return.
- **Environment check**: paths, dependencies, CUDA — no camera needed.
- **Model check**: loads the weights and runs one inference on a synthetic image.

## 3. Calibration console

Four pages: **Capture → Calibrate → Inspect rectification → Save**.

- Capture needs the camera; the other pages are CPU only.
- The **Data source** dropdown (page 2) lists capture sessions found under `output\calib`
  (and nearby project folders). If the list is empty, capture a few pairs first, or use
  **Browse…** to select any folder containing `left_*.png`.
- Saving writes `calibration\stereo_calib.yaml` and archives the previous file under
  `calibration\archive\`.
- To preview rectification without re-calibrating, use **Load existing calibration** on page 3.
  A check image needs one real left/right pair; if none is selected, the dialog lets you pick
  a folder immediately.

## 4. CLI windows

Keys (new runtime path, added by the launcher and menus automatically):

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `Space` | Pause / resume |
| `i` | Print stats |
| `p` or `s` | Save result to `output\captures\` |
| `[` / `]` | Narrow / widen the depth range |
| `o` | Reopen the 3D window (point-cloud mode) |
| `m` | Cycle inference preset |

## 5. What gets saved

`output\captures\<timestamp>_<ms>_cloud\` contains:

| File | Content |
|---|---|
| `left_raw.png`, `right_raw.png` | Rectified left/right images (uint8 BGR) |
| `disparity.npy` | Disparity, float32 pixels (model resolution) |
| `depth_meter.npy` | Depth, **float32 meters**, 0 = invalid |
| `valid.npy` | Validity mask (bool) |
| `depth_meter.txt` | Valid ratio and median depth |
| `cloud.ply` | Colored point cloud, ASCII, meters, camera frame (x right, y down, z forward), RGB |
| `meta.txt` | Frame index, preset, timings, shapes, timestamp |

> The PLY keeps the optical frame (**y down**). Open3D/MeshLab treat +y as up, so the cloud may
> look vertically flipped in those viewers. The data is not flipped.

## 6. Configuration

`config.yaml` sections: `project`, `camera`, `depth`, `calibration`, `ffs`, `pointcloud`,
`runtime`, `gui`, `ui`.

| Change | Where | Note |
|---|---|---|
| Camera index | `camera.single_index` | The built-in laptop camera is usually index 0 — never use it as the stereo camera |
| Depth range | `pointcloud.zmin_m` / `zmax_m` | Independent from `depth` |
| Inference preset | `ffs.scale` / `ffs.valid_iters` | Changing `scale` does not require recalibration; changing resolution does |
| Calibration on/off | `depth.calibrated` | `false` falls back to nominal values and **must not be used for measurement** |
| UI language | `ui.language` | `zh` / `en` |

## 7. Troubleshooting order

1. `logs\app-<date>.log` — the last lines usually name the cause.
2. **Environment check**, or `python app_entry.py --selftest`.
3. [faq.md](faq.md).
