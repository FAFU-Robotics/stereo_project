# FAQ

## The camera cannot be opened

Symptom: `CameraError: cannot open index 1`, or the viewer shows **Connection failed**.

Causes, in order of likelihood:

1. **Another process owns the camera.** Close browser video tabs, meeting apps, a second
   instance of this program, or the calibration console. Check with the viewer's device panel.
2. **The index changed** after re-plugging or using a different USB port. Run
   `python main.py --list` and compare with `camera.single_index` in `config.yaml`.
3. **Low-integrity / sandboxed session.** Windows denies physical cameras to restricted
   processes. Use a normal desktop terminal.
4. **The driver rejects the format.** Adjust `camera.width/height/fourcc`, or set
   `camera.open_with_params: false`.

Note: normal user rights are enough. Elevation is **not** required.

## The data source list is empty in the calibration console

The console lists capture sessions from `output\calib` under its data root plus nearby project
folders. When running a packaged EXE, the data root is the EXE folder — sessions captured from
the source tree are elsewhere.

Options:

- Click **Browse…** and pick any folder containing `left_*.png` / `right_*.png`.
- Or capture a few pairs in page 1 first.
- Or copy the session folder into `<exe folder>\output\calib`.

## "No usable capture image" when generating the inspection image

The rectification preview needs one real left/right pair. Without a selected session, the
console offers a folder picker — choose a capture folder and it retries automatically.

## Calibration exists but the preview stays empty

The calibration file was found (the header shows baseline/RMS), but no capture pair is
selected. Use **Browse…** to point at a session. The calibration itself is fine.

## Depth view is black or stays on "loading"

- The model loads on first use and takes a few seconds.
- Streaming must be started first.
- Check VRAM: an 8 GB laptop GPU can run the model, but not alongside another GPU-heavy app.

## `torch.compile is not supported on Python 3.14+`

`Fast-FoundationStereo` uses `@torch.compile`, which is a hard error on Python 3.14. This
repository replaces it with an identity decorator in `stereo_depth.py`. If you import the
upstream module directly, you will hit the error — use the repository entry points.

## `ModuleNotFoundError: open3d` / `plotly` / `dash`

`import open3d` loads `open3d.visualization.draw_plotly` at the top level, which requires
`plotly` and `dash`. Install both. On Python 3.14 only open3d 0.20.0 provides a `cp314` wheel.

## `ModuleNotFoundError: omegaconf`

The weights embed a `DictConfig`; `torch.load` needs `omegaconf` even though the code never
imports it directly. `python -m pip install omegaconf`.

## The picture looks flipped in MeshLab / Open3D

The saved PLY uses the optical frame (**y down**). Those viewers assume +y is up. The data is
correct; the 3D window flips y for display only.

## How do I switch the UI to English?

Set `ui.language: en` in `config.yaml`, or pass `--lang en`:

```bat
python app_entry.py --lang en
StereoCamera.exe --lang en
```

## Where are the logs?

`logs\app-<date>.log` next to the executable (or in the repository root when running from
source). GUI exceptions are also appended to `logs\viewer-errors.log`.
