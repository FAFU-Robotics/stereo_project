# 双目深度相机使用说明

双击 `StereoCamera.exe` 即可启动，不需要安装 Python。

## 启动

1. 解压整个 `StereoCamera` 文件夹。不要只拷贝 exe。
2. 用 USB 接上双目相机（本项目按 DECXIN，设备号 `0BDA:5883` 标定）。
3. 双击 `StereoCamera.exe`。
4. 在菜单里选择：
   - **标定工作台**：采集标定图、计算、检查、保存标定。
   - **实时预览**：只看左右相机画面。
   - **实时深度**：左右图 + 视差伪彩，画面上有采集帧率、推理耗时、端到端帧率。
   - **实时点云**：左右图 + 深度伪彩，并打开 3D 点云窗口。
   - **环境自检 / 模型自检**：不连相机，用来确认文件和显卡是否正常。

实时画面窗口里的按键：

| 键 | 作用 |
|---|---|
| Q 或 Esc | 退出当前功能，回到菜单 |
| P 或空格 | 暂停 / 继续 |
| S | 保存当前帧 |
| I | 把状态写进日志 |
| H | 显示或隐藏按键帮助 |
| O | 重新打开被关掉的 3D 点云窗口 |

关掉 3D 窗口不会退出程序。按 O 可以再打开。

## 相机

- 程序用 Windows 自带的 DirectShow 打开相机，没有单独的厂商 SDK。
- 相机是免驱 UVC。第一次插入若系统正在装驱动，等它完成再开程序。
- 默认使用配置里的索引 1、分辨率 1280×480（左右各 640×480）。这是这台 DECXIN 的标定分辨率，不要随便改，改了深度就不准。
- 笔记本自带摄像头通常是索引 0，不要把它当成双目相机。
- 不需要管理员权限。同一时间只能有一个程序占用相机。先关掉相机软件、浏览器视频页、以及另一个本程序窗口。

## 文件放在哪里

所有路径都相对于 `StereoCamera.exe` 所在文件夹。

| 内容 | 位置 | 能不能改 |
|---|---|---|
| 主配置 | `config.yaml` | 可以。改相机索引、分辨率、深度范围、推理档位 |
| 标定结果 | `calibration\stereo_calib.yaml` | 可以。标定工作台保存时会写这里，旧文件会备份 |
| 模型权重 | `weights\23-36-37\` | 可以整目录替换。里面要同时有 `model_best_bp2_serialize.pth` 和 `cfg.yaml` |
| 保存的图像、深度、点云 | `output\captures\` 或 `output\snapshots\` | 程序自动创建 |
| 运行日志 | `logs\app-日期.log` | 出错时把这个文件发给开发者 |
| 运行库、推理代码 | `_internal\` | 不要改，不要删 |

程序不读取你解压时所在的“当前文件夹”。从快捷方式或别的目录启动也可以。

`_internal` 里还有一份配置、标定和权重，作为后备。EXE 旁边的同名文件优先。替换权重时改 EXE 旁边的 `weights`，不用动 `_internal`。

## 保存出来的是什么

在实时窗口按 S 之后，`output\captures\<时间戳>_cloud\`（或 `snapshots`）里会有：

- `left_raw.png` / `right_raw.png`：左右图
- `disparity.npy`：视差
- `depth_meter.npy`：深度，单位米
- `depth_vis.png`：深度伪彩
- 点云 `.ply`：相机坐标系，x 向右、y 向下、z 向前

3D 窗口里为了正着看，把 y 翻成了向上。保存的 ply 没有翻转。

## 电脑需要具备的条件

程序自己带了 Python 运行库、OpenCV、PyTorch CUDA 运行库和模型。下面这些**不能**打进安装包，需要电脑上已经有：

1. **64 位 Windows 10 或 Windows 11**。
2. **NVIDIA 显卡驱动**。本程序用的 PyTorch 是 CUDA 12.8 版，驱动需要支持它（一般是较新的 Game Ready / Studio 驱动）。没有 NVIDIA 显卡，或驱动太旧时，模型自检会提示 CUDA 不可用，深度和点云不能用。不需要单独安装 CUDA Toolkit。
3. **Visual C++ 2015–2022 运行库（x64）**。多数 Windows 已经装过。若双击后提示缺少 `VCRUNTIME140.dll` 或 `MSVCP140.dll`，安装微软的 “Microsoft Visual C++ 2015-2022 Redistributable (x64)”。
4. **双目相机和它的 UVC 驱动**（系统自动安装）。

显存建议不少于 2 GB。本模型在 8 GB 笔记本显卡上实测推理约占 1 GB 以内。显存不够时，程序会尝试把推理档位调低。

## 常见问题

**双击没有窗口**  
看 `logs` 里当天的 `app-*.log`。若杀毒软件把 exe 隔离了，把整个文件夹加入信任后重新解压一份干净的。

**提示找不到权重或 config**  
确认 exe 旁边有 `config.yaml`、`weights\23-36-37\model_best_bp2_serialize.pth` 和同目录的 `cfg.yaml`。不要只复制 exe。

**打不开相机**  
确认 USB 已插好；关掉其它占用相机的程序；在菜单里先跑“环境自检”。索引不对时，编辑 `config.yaml` 的 `camera.single_index`。改分辨率后必须重新标定。

**深度或点云窗口一闪就回到菜单**  
打开 `logs\app-日期.log` 看最后几十行。最常见的是 CUDA 不可用，或相机打不开。

**画面有了但深度明显不对**  
确认 `calibration\stereo_calib.yaml` 还在，且 `config.yaml` 里 `depth.calibrated` 为 `true`。标定是按 1280×480 做的。

**程序关不掉相机**  
正常退出用 Q 或 Esc，或直接关掉菜单窗口。若进程残留，在任务管理器结束 `StereoCamera.exe` 后再开。锁文件在 `output\.camera.lock`，进程已经不在时，下次启动会自动接管。

## 命令行（可选）

exe 也可以不走菜单：

```bat
StereoCamera.exe --where
StereoCamera.exe --selftest
StereoCamera.exe --model-selftest
StereoCamera.exe --gui
StereoCamera.exe --mode preview --use-runtime
StereoCamera.exe --mode depth --use-runtime
StereoCamera.exe --mode pointcloud --use-runtime
```

这些窗口没有黑色控制台，文字写在 `logs` 里。
