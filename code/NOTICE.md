# NOTICE —— 许可边界与第三方声明

本文件说明本仓库各部分分别适用什么许可。**只读 MIT 那一段不足以判断能否商用**，请把本文件读完。

## 1. 各部分适用什么许可

| 范围 | 许可 | 能否商用 |
|---|---|---|
| 本仓库作者原创代码：`camera.py`、`calibration.py`、`depth_utils.py`、`pointcloud.py`、`pipeline_threaded.py`、`runtime.py`、`app_entry.py`、`app_paths.py`、`viewer.py`、`view3d.py`、`monitor.py`、`logger.py`、`profiler.py`、`main.py`、`stereo_sdk/`、`tools/`、`tests/`、`examples/` | MIT（见 [LICENSE](LICENSE)） | 可以 |
| 文档与图片：`docs/`、各 `README*.md` | MIT | 可以 |
| **`third_party/Fast-FoundationStereo/`（代码）** | **NVIDIA Source Code License（NSCL）**，见 [third_party/Fast-FoundationStereo/LICENSE.txt](third_party/Fast-FoundationStereo/LICENSE.txt) | **不可以** |
| **模型权重 `weights/23-36-37/model_best_bp2_serialize.pth`** | 官方 model card 声明 “for research and evaluation purposes only” | **不可以** |
| 打包产物 `StereoCamera.exe`（内含上述权重与 FFS 代码） | 同时受 MIT 与 NSCL 约束，取更严格者 | **不可以** |

## 2. 为什么 FFS 部分不能商用

NSCL 第 3.3 条（Use Limitation）原文：

> 3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use non-commercially. Notwithstanding the foregoing, NVIDIA and its affiliates may use the Work and any derivative works commercially. As used herein, "non-commercially" means for research purposes only.

即：FFS 的代码与其衍生作品**只能用于研究目的**。商用授权需直接向 NVIDIA 申请，本仓库无权授予，也无法代为转授。

同一份许可的 model card 亦写明：

> This model is for research and evaluation purposes only.

## 3. 为什么本仓库仍可以整体采用 MIT

NSCL 第 1 条对 “derivative works” 的定义作了排除：

> ...derivative works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of the Work.

本仓库与 FFS 的关系属于该排除项描述的情形：本项目代码在运行时 `import` FFS 的 Python 接口（`core/` 下的模型与推理函数），两者在文件系统与进程内都是可分离的，本仓库不分发 FFS 的修改版本，也不把 FFS 代码内联进自有源码。因此：

- 自有代码可单独以 MIT 授权；
- 一旦运行推理，FFS 部分即受 NSCL 的第 3.3 条约束，使用目的被限定为研究/评估；
- 分发整包（源码包或 `StereoCamera.exe`）时，NSCL 的声明义务随之生效，所以本文件必须随包分发。

> 上述分析是本仓库作者的工程判断，不构成法律意见。商用前请咨询法务，或直接向 NVIDIA 取得授权。

## 4. 第三方组件清单

| 组件 | 位置 | 许可 | 用途 |
|---|---|---|---|
| Fast-FoundationStereo | `third_party/Fast-FoundationStereo/` | NVIDIA Source Code License | 双目视差推理 |
| PyTorch | 运行环境依赖 | BSD-3-Clause | 深度学习运行时 |
| OpenCV | 运行环境依赖 | Apache-2.0 | 采集、校正、图像处理 |
| Open3D | 运行环境依赖 | MIT | 3D 点云窗口 |
| NumPy | 运行环境依赖 | BSD-3-Clause | 数值计算 |
| SciPy | 运行环境依赖 | BSD-3-Clause | 数值计算 |
| PyYAML | 运行环境依赖 | MIT | 读取 `config.yaml` |
| tkinter | Python 标准库 | PSF | 标定 GUI 与菜单 |
| Noto Sans SC | `assets/fonts/` | SIL Open Font License 1.1 | 系统没有可被 Tk 使用的中文字体时，界面与画面标注的回退字体 |
| PyInstaller | 构建期依赖 | GPL-2.0 with exception | 打包 EXE（其例外条款允许分发打包产物） |

## 5. 引用

若本项目或其数据用于论文或对外报告，请引用上游工作。以下条目照录自 `third_party/Fast-FoundationStereo/readme.md` 的 Citation 段：

```bibtex
@article{wen2026fastfoundationstereo,
  title={{Fast-FoundationStereo}: Real-Time Zero-Shot Stereo Matching},
  author={Bowen Wen and Shaurya Dewan and Stan Birchfield},
  journal={CVPR},
  year={2026}
}
```

论文预印本：arXiv:2512.11130（见 `third_party/Fast-FoundationStereo/model_card.md` 的 Reference 段）。
