# Open Stereo Camera

开源双目深度相机：USB 双目采集、标定、神经网络立体匹配、深度图与彩色点云。

| 目录 | 内容 |
|---|---|
| [CAD设计](CAD设计/) | 外壳图纸：主体、前盖，以及 3D 打印用的 `相机外壳.3mf` |
| [code](code/) | 源码、Python SDK、文档和打包脚本 |

Windows 软件不放在仓库目录里。到本页右侧 **Releases** 下载 `StereoCamera`，解压后运行。

自有代码为 MIT。立体匹配模型来自 NVIDIA Fast-FoundationStereo，只允许研究 / 评估，不可商用。边界见 [NOTICE.md](code/NOTICE.md)。

从源码运行时进入 `code/`：

```bat
git clone https://github.com/FAFU-Robotics/stereo_project.git
cd stereo_project\code
```

完整步骤见 [code/README.md](code/README.md)。
