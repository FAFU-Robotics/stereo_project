# 文档索引

> 按"我想干什么"找文档，而不是按文件名猜。

## 先看这两份

| 文档 | 什么时候看 |
|---|---|
| [架构.md](架构.md) | **第一次进这个项目**：模块地图、数据流、要加东西加在哪 |
| [接口约定_SDK.md](接口约定_SDK.md) | 要写 GUI 或外部程序，只通过 `stereo_sdk` 交互时 |

## 从零搭环境 / 跑起来

| 文档 | 内容 |
|---|---|
| [../README.md](../README.md) | 安装、权重下载、启动器用法、常见问题 |

## GUI 与 SDK

| 文档 | 内容 |
|---|---|
| [标定GUI使用说明.md](标定GUI使用说明.md) | 四页工作台怎么用：抓图、标定、看校正、保存；快捷键与判读标准 |
| [../README_SDK.md](../README_SDK.md) | SDK 安装、`Session` 生命周期、字段单位约定 |
| [../examples/README.md](../examples/README.md) | 四个可直接运行的示例（读深度 / 存点云 / 实时预览 / 标定报告）|

## 打包发布

| 文档 | 内容 |
|---|---|
| [打包成EXE_准备清单.md](打包成EXE_准备清单.md) | PyInstaller 的 datas/hiddenimports/排除项、两档体积账、验收清单 |
| [量块精度对比_CloudCompare操作清单.html](量块精度对比_CloudCompare操作清单.html) | 用精密量块 + CloudCompare 复核绝对精度（可打印 A4）|

## 各阶段的实测记录（按时间顺序，含踩坑与结论）

| 阶段 | 文档 | 一句话结论 |
|---|---|---|
| 5 | [阶段5_标定实测记录.md](阶段5_标定实测记录.md) | 双目 RMS 0.1246 px，基线 64.891 mm |
| 6 | （见 README 的"精度验证"节）| 已知距离验证 ±0.3%；视差校正 1.0221 的由来 |
| 7 | [性能测试报告.md](性能测试报告.md) | 推理占 82%；三档预设 6.2 / 10.3 / 20.6 FPS |

## 常用工具速查

| 想做什么 | 命令 |
|---|---|
| 跑离线自检（不开相机、不需要管理权）| `python tests\test_pipeline_offline.py` |
| 分阶段性能基准 | `python tools\bench_pipeline.py` |
| 各预设的深度精度 | `python tools\bench_presets.py` |
| 只测推理耗时 | `python tools\bench_ffs.py` |
| 探测相机后端/分辨率 | `python tools\probe_camera_backends.py` |
| 采集标定图 | 双击 `tools\标定采集_新板7x9_26mm.bat`（板子参数在 .bat 顶部）|
| 换标定（图形界面）| 双击 `tools\calib_gui.bat` |
| 计算标定 | `python tools\run_calibration.py --dir output\calib\<时间戳>` |
| 取深度/点云（SDK）| `python examples\01_read_depth.py` / `02_save_pointcloud.py` |
| 两台相机点云对比 | `python tools\compare_with_realsense.py --help` |
