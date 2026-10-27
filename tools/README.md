# tools 目录说明

> 22 个文件混在一起确实乱，但**不要搬动它们** —— 它们被 6 个 `.bat` 启动器、
> README 和 5 份 docs 引用了几十处，搬一次要改一大圈，收益远小于风险。
> 用这张表按用途找就行。

## 一、双击就能用的启动器（.bat）

| 文件 | 用途 | 需要管理员 |
|---|---|---|
| `启动_普通权限.bat` | 主启动器（普通权限，推荐）| ❌ |
| `启动_管理员.bat` | 主启动器（管理员；含标定采集等入口）| ✅ |
| `相机OpenCV自检.bat` | 用 OpenCV 单独验证能否开相机 | ❌ |
| `相机状态检查.bat` | DShow 直连测试（排查 `0x80070005`）| 视情况 |
| `标定采集_新板7x9_26mm.bat` | 用 A4 ChArUco 板采集标定图（板参数在文件顶部 `SQ=26.0`）| ✅ |

## 二、正式工具（会被反复用）

| 文件 | 用途 |
|---|---|
| `test_pipeline_offline.py` | **离线自检 10 场景**（不开相机、不用真权重、不需管理权）。改代码后必跑 |
| `capture_calib.py` | 标定图采集 + 质量门（清晰度/对比度/每格像素/姿态清单）|
| `run_calibration.py` | 计算双目标定 + 极线验证 + 写 `calibration/stereo_calib.yaml` |
| `bench_pipeline.py` | **分阶段**耗时（校/推理/点云/显示）+ 三线程估算 + 显存 |
| `bench_presets.py` | 三档预设的**真实深度误差**（用已知距离数据，离线）|
| `bench_ffs.py` | 只测推理（scale × valid_iters 矩阵）|
| `aruco_align.py` | ChArUco 位姿 → 把点云变换到板坐标系（跨相机对齐）|
| `compare_with_realsense.py` | 自制相机 vs RealSense 点云一致性（导出板坐标系 PLY）|
| `fetch_ffs_weights.py` | 下载/校验 FFS 权重 |

## 三、一次性探测脚本（排查环境用，留档备查）

| 文件 | 用途 |
|---|---|
| `probe_camera_backends.py` | **后端/分辨率探测**：MSMF vs DSHOW、是否命中 1280×480、真实帧率、左右亮度差 |
| `opencv_camera_check.py` | OpenCV 能否开相机的独立小测（含版本对比）|
| `dshow_capture_test.py` | DirectShow 直连（低层测试，用于区分"权限"与"占用"）|
| `preview_any.py` | 不依赖项目模块的极简预览（验证相机是否活着）|

> 这几个脚本的价值在**排错**：相机打不开时按
> `preview_any` → `opencv_camera_check` → `dshow_capture_test` → `probe_camera_backends`
> 的顺序试，能定位到"没接 / 被占用 / 权限 / 后端不支持该分辨率"中的哪一类。

## 四、改动注意

| 事项 | 说明 |
|---|---|
| 新增工具放哪 | 直接放 `tools/` 顶层（不要建子目录——会被 `.bat` 引用路径拖累）|
| 新增 .bat | 用纯 ASCII + CRLF（中文注释会导致 cmd 乱码/解析失败）|
| 所有脚本的相对路径 | 都以**项目根目录**为基准（脚本内部用 `PROJECT_ROOT` 拼接）|
