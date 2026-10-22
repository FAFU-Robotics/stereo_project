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

## 各阶段的实测记录（按时间顺序，含踩坑与结论）

| 阶段 | 文档 | 一句话结论 |
|---|---|---|
| 5 | [阶段5_标定实测记录.md](阶段5_标定实测记录.md) | 双目 RMS 0.1246 px，基线 64.891 mm |
| 6 | （见 README 的"精度验证"节）| 已知距离验证 ±0.3%；视差校正 1.0221 的由来 |
| 7 | （待补 性能测试报告.md）| 推理占 82%；三档预设 6.2 / 10.3 / 20.6 FPS |

## 常用工具速查

| 想做什么 | 命令 |
|---|---|
| 跑离线自检（不开相机、不需要管理权）| `python tools\offline_selftest.py` |
| 分阶段性能基准 | `python tools\bench_pipeline.py` |
| 各预设的深度精度 | `python tools\bench_presets.py` |
| 只测推理耗时 | `python tools\bench_ffs.py` |
| 探测相机后端/分辨率 | `python tools\probe_camera_backends.py` |
| 采集标定图 | 双击 `tools\标定采集_新板7x9_26mm.bat`（板子参数在 .bat 顶部）|
| 计算标定 | `python tools\run_calibration.py --dir output\calib\<时间戳>` |
| 两台相机点云对比 | `python tools\compare_with_realsense.py --help` |
