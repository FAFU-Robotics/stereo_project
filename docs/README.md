# 文档索引

按“我想干什么”找文档，而不是按文件名猜。下面按用途分组：**使用者文档**、**开发与维护文档**、**实测记录**、**历史文档**。

## 一、使用者文档（第一次接触看这组）

| 文档 | 什么时候看 | 读完得到什么 |
|---|---|---|
| [../README.md](../README.md) | 刚拿到这个仓库 | 项目能干什么、两条安装路线、性能与精度摘要 |
| [安装指南.md](安装指南.md) | 要把它装起来 | EXE 解压步骤 / 源码环境搭建，含 torch 版本选择矩阵 |
| [软件使用说明.md](软件使用说明.md) | 要操作界面 | 主界面（Stereo Viewer）、标定工作台、命令行窗口的分工，快捷键、保存物、命令行参数 |
| [标定GUI使用说明.md](标定GUI使用说明.md) | 要换标定 | 四页工作流的操作与判读标准 |
| [SDK_API参考.md](SDK_API参考.md) | 要写代码调 SDK | 逐条对应源码的 API、返回值与线程约定 |
| [../README_SDK.md](../README_SDK.md) | 想先跑通再深入 | pip 安装顺序、三行代码、三条必错点 |
| [常见问题.md](常见问题.md) | 出错了 | 按现象索引的排查表，每条都给了可验证的原因 |

## 二、开发与维护文档

| 文档 | 内容 | 什么时候看 |
|---|---|---|
| [架构.md](架构.md) | 模块地图、一帧数据的生命周期、标定链路、要加东西加在哪 | 要改代码 |
| [仓库结构说明.md](仓库结构说明.md) | 每个目录的职责、哪些不入库及原因、新增文件放哪 | 要加文件 |
| [发布与分发.md](发布与分发.md) | 5 GB EXE 的分卷打包、校验、GitHub Release 流程 | 要发版 |
| [发布说明模板.md](发布说明模板.md) | Release notes 必写的四条 | 发版时照抄 |
| [打包成EXE_准备清单.md](打包成EXE_准备清单.md) | PyInstaller 的 datas / hiddenimports / 排除项、体积账 | 要重新打包 |
| [../packaging/打包测试报告.md](../packaging/打包测试报告.md) | EXE 实测：通过了什么、**没验证什么** | 判断能不能交付 |
| [../tools/README.md](../tools/README.md) | 每个工具的用途与适用场景 | 找工具 |
| [../examples/README.md](../examples/README.md) | 六个示例各自解决什么问题 | 学 SDK 用法 |

## 三、实测记录（按时间/阶段，含结论与踩坑）

| 阶段 | 文档 | 一句话结论 |
|---|---|---|
| 5 | [阶段5_标定实测记录.md](阶段5_标定实测记录.md) | 双目 RMS 0.1246 px、基线 64.891 mm、极线失准 0.07 px |
| 6 | 见 [性能测试报告.md](性能测试报告.md) 第 5 节 | 三档预设的逐帧误差；视差系统性偏小 2%，校正系数 1.0221 |
| 7 | [性能测试报告.md](性能测试报告.md) | 推理占 82%；端到端 3.92 / 6.38 / 14.33 FPS |
| — | [量块精度对比_CloudCompare操作清单.html](量块精度对比_CloudCompare操作清单.html) | 用精密量块 + CloudCompare 复核绝对精度的操作步骤（可打印 A4） |

## 四、历史文档（保留作变更依据，内容已被上面取代）

| 文档 | 说明 |
|---|---|
| [接口约定_SDK.md](接口约定_SDK.md) | v0.1.0 时期的 GUI 契约。当前 API 以 [SDK_API参考.md](SDK_API参考.md) 为准；本文件保留，用于追溯 0.1.0 → 0.3.0 到底加了什么 |

## 五、常用命令速查

| 想做什么 | 命令 |
|---|---|
| 环境自检（不需要相机） | `python app_entry.py --selftest` |
| 只验推理链路（不需要相机） | `python app_entry.py --model-selftest` |
| 离线自检（不需要相机与显卡） | `python tests\test_pipeline_offline.py` |
| 分阶段性能基准 | `python tools\bench_pipeline.py` |
| 三档预设的深度误差 | `python tools\bench_presets.py` |
| 只测推理耗时矩阵 | `python tools\bench_ffs.py` |
| 探测相机后端/分辨率 | `python tools\probe_camera_backends.py` |
| 采集标定图 | 双击 `tools\标定采集_新板7x9_26mm.bat` |
| 标定工作台 | `python tools\calib_gui.py` |
| 只算标定（无界面） | `python tools\run_calibration.py --dir output\calib\<时间戳>` |
| 读深度 / 存点云 | `python examples\01_read_depth.py --mock` / `02_save_pointcloud.py --mock` |
| 质量统计 / 采数据集 | `python examples\05_depth_statistics.py --mock --seconds 3` / `06_save_dataset.py --mock` |
| 两台相机点云对比 | `python tools\compare_with_realsense.py --help` |

## 六、本索引的维护规则

1. 新增文档必须在上面某张表里登记一行，否则视为“不存在”——没人会在目录里翻找。
2. 实测记录不改写历史结论。结论被推翻时，**新增一节**说明“何时、凭什么证据推翻”，并在原文标注“已过时”。相机权限那条就是这么处理的两轮。
3. 文档里出现的路径、数字、命令必须能在仓库里找到出处。给不出出处的数字不写进文档。
