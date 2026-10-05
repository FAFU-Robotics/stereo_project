# 阶段 1 补充测试 —— OpenCV 版本对比（5.0.0 vs 4.12.0）

> 测试时间：2026-09-30
> 测试目的：验证"相机打不开/画面全黑"是否由 **OpenCV 5.0.0（较新的大版本）** 引起。
> **结论：不是。两个版本行为完全一致。**
> 依据用户要求第 7 条，测试到此停止，**未修改任何驱动、注册表或相机配置**。

## 1. 隔离安装方式（零污染）

| 项目 | 内容 |
|---|---|
| 目标目录 | `stereo_project/_work/cv412`（仅 cv2，43.2 MB，128 个文件） |
| 安装方式 | `python tools/install_opencv4_isolated.py --version 4.12.0.88 --target ...` |
| 为什么不用 pip | 本机 pip 的 `tempfile.mkdtemp()` 临时目录被 DSH 沙箱拒绝写入（`PermissionError: [Errno 13]`），故改为"wheel 即 zip，直接下载并解压"（纯标准库） |
| 生效方式 | 只对单条命令临时设置 `PYTHONPATH`，**不写入任何配置** |
| 现有环境影响 | **无**。python312 的 `cv2 5.0.0`、`numpy 2.5.3` 一个字节都没改（已验证导入路径仍指向 site-packages） |
| 卸载方式 | `python tools/install_opencv4_isolated.py --target <目录> --remove`，或直接删目录 |

导入验证：
```
不带 PYTHONPATH : cv2 5.0.0  -> D:\31931\Documents\python312\Lib\site-packages\cv2\__init__.py
带 PYTHONPATH   : cv2 4.12.0 -> D:\31931\Documents\stereo_project\_work\cv412\cv2\__init__.py
                  numpy 2.5.3（仍来自 python312，无 ABI 冲突）
```

## 2. 对比结果（同一脚本、同一参数）

命令（两个版本逐字相同，只有 `--out-dir` 与 `PYTHONPATH` 不同）：
```bat
python tools\probe_cameras.py --max-index 6 --backends dshow,msmf ^
       --timeout 20 --warmup 5 --measure 10 --out-dir output\probe\<版本>
```

| 测试项 | OpenCV 5.0.0（基线） | OpenCV 4.12.0（隔离） | 是否改善 |
|---|---|---|---|
| 能打开的相机索引 | **仅索引 2** | **仅索引 2** | ❌ 无 |
| 索引 0/1/3/4/5 | 全部打不开 | 全部打不开 | ❌ 无 |
| 索引 2 默认分辨率 | 640x480 | 640x480 | — 相同 |
| 索引 2 实测帧率 | **约 1.0 fps** | **约 1.0 fps** | ❌ 无 |
| 索引 2 画面 mean / std | 0.01 / 0.70 | 0.01 / 0.89 | ❌ 仍无有效信号 |
| MJPG 各分辨率（1280x480 / 2560x720 / 640x480 / 1280x720 / 1920x1080） | 全部"能打开但读不到帧" | 全部"能打开但读不到帧" | ❌ 无 |
| YUY2 各分辨率（1280x480 / 640x480） | 全部"能打开但读不到帧" | 全部"能打开但读不到帧" | ❌ 无 |
| 不设格式 + 1280x480 | 读到帧但全 0 | 读到帧但全 0 | ❌ 无 |
| MSMF 后端（所有索引） | 全部打不开 | 全部打不开 | ❌ 无 |
| **左右两个相机能否都采集** | **不能**：只有 1 个索引可打开，另一个相机完全打不开 | **不能**（同左） | ❌ 无 |

原始证据：
```
output/probe/cv5/report.json       + output/probe/cv5_console.txt
output/probe/cv412/report.json     + output/probe/cv412_console.txt
```

## 3. 已排除的原因（都有实测依据）

| 假设 | 实测结论 | 依据 |
|---|---|---|
| 相机硬件/驱动坏了 | ❌ 不是 | `pnputil`：`USB Camera`、`HP True Vision FHD Camera` 均为 **Started** |
| 相机没插好 | ❌ 不是 | 同上，且用户确认 Windows 相机 App **能出画面** |
| 桌面应用相机权限被禁 | ❌ 不是 | `HKCU\...\ConsentStore\webcam\NonPackaged = Allow` |
| Windows 帧服务器未启动 | ❌ 不是 | `FrameServer` / `FrameServerMonitor` 均为 Running |
| 我的 shell 与桌面不在同一会话 | ❌ 不是 | Chrome 与我同在 SessionId 2 |
| 是 OpenCV 5.0.0 太新 | ❌ **不是** | **本次测试：4.12.0 行为逐项一致** |
| 注册表里的 RealSense 干扰 | ❌ 不是 | 它们是历史残留（Disconnected），已从在线清单剔除 |

## 4. 剩余假设（按可能性排序，需要人机配合验证）

1. **本 DSH 沙箱进程环境访问不到真实 UVC 相机**。
   线索：DirectShow 的"视频输入设备"分类注册表里**只有 1 条记录，且是 HP 的虚拟摄像头
   `OMEN Cam ＆ Voice`（DevicePath 为空）**。这正好对应"OpenCV 只能枚举到 1 个设备、
   而且它不出画面"。→ 需要在**你自己的普通终端**里跑同一脚本验证。
2. 索引 2 实际就是那个 HP 虚拟摄像头（而非 DECXIN），这也解释"能打开但没有真实图像"。
   → 需要"拔掉 DECXIN 再测一次"的 A/B 对比来确认设备身份。
3. 相机虽枚举成功但不真正出流（USB 端口/线材/供电），而 App 走的是 MF 帧服务器路径所以能出画面。
   → 需要换机身 USB3 口直插后复测。

## 5. 环境恢复状态（已确认）

- `PYTHONPATH`：测试后已清除（只在单条命令内临时设置）。
- `python312`：**未改动**。`cv2 5.0.0` 与 `numpy 2.5.3` 路径、版本与测试前完全一致。
- 隔离目录 `_work/cv412`：保留（**不在任何 PATH 上，对现有程序零影响**），
  如需彻底清理：`python tools\install_opencv4_isolated.py --target D:\31931\Documents\stereo_project\_work\cv412 --remove`

> **【2026-10-05 项目精简时的更新】** 本次实验的结论已经确定（OpenCV 4.12 与 5.0 行为一致，
> 相机访问被拒与本问题无关），因此为腾出空间已删除：
> `_work/cv412`（118.7 MB）、`_work/wheels`（43.2 MB）、`tools/install_opencv4_isolated.py`、
> 以及本实验在 `output/probe/cv412`、`output/probe/cv5` 下的 JSON 报告。
> 本文档的**结论与数据仍然有效**，只是"复现方法"需要按第 2 节的步骤重新实现脚本。
- 现有双目程序文件：**未修改**（本阶段只新增了 tools 脚本与文档）。
- 驱动 / 注册表 / 相机配置：**未做任何修改**。

## 6. 下一步（需要用户操作）

请**在你自己的终端里**（不要通过我）执行一次同样的探测，用来判断假设 1：

```
双击运行：D:\31931\Documents\stereo_project\tools\相机探测.bat
```

- 若在你自己终端里**能出画面/能打开更多索引** → 说明限制来自 DSH 沙箱进程环境，
  相机相关的实时程序需要在普通终端运行（或把本会话切到完全权限）。
- 若在你自己终端里**结果相同** → 说明问题在系统层（DirectShow 枚举/驱动占用），
  下一步应做"拔插 A/B 测试"确认设备身份，以及换 USB 端口复测；
  在拿到这两项结果前，不再改动任何系统设置。
