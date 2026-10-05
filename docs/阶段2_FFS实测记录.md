# 阶段 2 实测记录 —— FFS 官方示例图像验证

> 记录时间：2026-09-30　执行人：DSH 助手（全部在本机实跑，命令与数字均为原始输出）
> 结论：**阶段 2 验收 5/5 全部通过。**

## 1. 权重（已在本机下载并验证）

| 项目 | 值 |
|---|---|
| 来源 | 官方 Google Drive 文件夹 `1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap` 下的 `23-36-37` |
| 文件 | `model_best_bp2_serialize.pth`（**67.8 MB**）+ `cfg.yaml` |
| 本地路径 | `weights/23-36-37/` |
| 许可 | 研究版，适用 NVIDIA Source Code License（研究/评估用途）；商用需换 C-Fast-FoundationStereo |
| 下载方式 | `python tools/fetch_ffs_weights.py`（纯标准库；本机 PowerShell/git 的 HTTPS 被系统凭据服务限制，只有 Python 的 OpenSSL 通道可用） |

`cfg.yaml` 实测内容（官方研究版 23-36-37）：
```
corr_levels=2  corr_radius=4  hidden_dims=[128]  max_disp=416  mixed_precision=true
n_downsample=2  n_gru_layers=1  slow_fast_gru=false  valid_iters=8  vit_size=vitl
```
`torch.load` 验证：类型 = `FastFoundationStereo`，参数 **17.7 M**，`normalize=True`。
本项目的 `ffs.max_disp` 用 **192** 覆盖官方 cfg 里的 416（与官方 readme 的 run_demo 默认一致）。

## 2. 环境与关键设置（实测必要）

| 设置 | 值 | 为什么 |
|---|---|---|
| PyTorch | 2.8.0+cu128 | RTX 5060 = sm_120；官方推荐的 cu124 不含该内核 |
| `TORCHDYNAMO_DISABLE` | `1` | Windows 无 Triton，`@torch.compile` 无法编译 CUDA 内核 |
| `ffs.build_volume` | `pytorch1` | triton 代价体内核在 Windows 不可用；`pytorch1` 是官方纯 PyTorch 路径 |
| 模型加载次数 | 1 次（3.2~3.6 s） | 常驻显存，绝不每帧重载 |

## 3. 官方示例结果（输入 960x540，scale=1.0，iters=8）

| 指标 | 实测值 |
|---|---|
| 有效视差比例 | **100.0 %** |
| 视差范围 | min 33.46 px，max 137.51 px，中位 95.57 px |
| 深度范围（fx=754.67，B=0.063 m） | min 0.346 m，中位 **0.497 m**，max 1.421 m |
| 点云 | 518 400 点，已导出 `cloud.ply` |
| 可视化 | `output/ffs_demo/<时间戳>/disp_vis.png`（左图 \| 右图 \| 视差伪彩） |

**目视核对**：键盘与纸巾盒（近）呈红/橙色，显示器与墙面（远）呈蓝/绿色，水杯轮廓清晰、
遮挡边界干净 —— 视差正确、方向正确（左图为左眼，未反）。

## 4. 性能基准（本机 RTX 5060 Laptop，纯 PyTorch，已关 torch.compile）

复现命令：
```bat
python tools\bench_ffs.py --size 640x480 --scales 1.0,0.5 --iters 8,4 --repeat 6
python tools\bench_ffs.py --scales 1.0 --iters 8,4 --repeat 6
```

### A) 640x480（≈真实相机单片分辨率，单眼）

| scale | valid_iters | 模型输入 | 冷启动 | **稳态** | 等效 FPS | 峰值显存 |
|---|---|---|---|---|---|---|
| 1.0 | 8 | 640x480 | 170.0 ms | **186.8 ms** | **5.35** | 2537 MB |
| 1.0 | 4 | 640x480 | 186.1 ms | **156.3 ms** | **6.40** | 2538 MB |
| 0.5 | 8 | 320x240 | 108.6 ms | **92.5 ms** | **10.81** | 748 MB |
| 0.5 | 4 | 320x240 | 64.0 ms | **71.0 ms** | **14.09** | 748 MB |

### B) 960x540（官方示例原图）

| scale | valid_iters | 模型输入 | 冷启动 | **稳态** | 等效 FPS | 峰值显存 |
|---|---|---|---|---|---|---|
| 1.0 | 8 | 960x540 | 304.2 ms | **309.8 ms** | **3.23** | 4243 MB |
| 1.0 | 4 | 960x540 | 260.1 ms | **266.6 ms** | **3.75** | 4244 MB |

### 与官方数据的对比（必须诚实说明）

官方在 **RTX 3090 + 640x480 + iters 8** 上给的 PyTorch 耗时是 **49.4 ms**；
本机是 **186.8 ms**，约 **3.8 倍慢**。原因（按影响排序）：
1. Windows 上**不能启用 `torch.compile`**（无 Triton），官方那 49.4 ms 是开着编译的；
2. 笔记本 GPU（RTX 5060 Laptop，功耗上限 65 W）算力低于台式 3090；
3. 官方那版 Triton GWC 内核在本机用不了，只能走 `pytorch1` 路径。

**对"实时"目标的直接结论**：

| 目标 | 可行配置 | 预期 FPS |
|---|---|---|
| 30 FPS | 纯 PyTorch 路径**做不到** | — |
| 10~15 FPS | `scale 0.5 + iters 4`（深度按 320x240 分辨率算，再上采样） | 10.8 ~ 14.1 |
| 5~6 FPS | `scale 1.0 + iters 4/8`（全分辨率 640x480） | 5.4 ~ 6.4 |

要真正上到 20~30 FPS，需要另做加速（阶段 7 再评估，都要先征求同意）：
- **TensorRT**（官方有现成 ONNX/TRT 脚本；官方 3090 数据是 49.4 → 23.4 ms，约 2.1 倍）
- **triton-windows**（社区版，用于恢复 `torch.compile`/triton 内核，需先小范围验证稳定性）

## 5. 本次踩到并修掉的真实问题（都已在代码里留注释）

| 问题 | 现象 | 解决 |
|---|---|---|
| Google Drive 下载报 HTTP 400 | 确认页表单字段漏了 `id/export` | 改为**按表单原样重建请求**（`tools/fetch_ffs_weights.py`） |
| `torch.load` 报 `No module named 'core'` | 官方模型是整对象序列化，反序列化需要官方仓库在 `sys.path` | 加载前把官方仓库根目录加入 `sys.path` |
| 冷启动污染性能统计 | 用 640x480 假图预热后跑 960x540，第一次仍要 **2088 ms**（对比稳态 270 ms），把均值从 270 拉到 897 | `warmup()` 改为**用真实尺寸图预热**，并单独列出冷启动 |
| 显存峰值看不出差异 | 未重置峰值统计，所有组合都显示历史最大值 | 每个测试组合前 `reset_peak_memory_stats()` |
| 官方 `torch.cuda.amp.autocast` 弃用警告 | 官方 `core/geometry.py` 里的写法在新 PyTorch 上告警 | **不改官方代码**（保持官方结构原样），仅记录；功能不受影响 |

## 6. 阶段 2 验收清单

| 检查项 | 结果 |
|---|---|
| 权重文件完整（.pth + cfg.yaml） | ✅ |
| 模型在 GPU 上加载成功 | ✅ |
| 推理成功并输出视差图 | ✅ |
| 视差图非空（有效像素 > 5%，实测 100%） | ✅ |
| 保存了视差/深度/伪彩图/点云/参数说明 | ✅ |
| 记录了推理耗时与 FPS | ✅ |
| 复现命令 | `python tools\run_ffs_demo.py --no-display --repeat 3`（退出码 0） |

## 7. 证据文件

> **【2026-10-05 项目精简时的更新】** 下列 `output/ffs_demo/<时间戳>/` 产物（73 MB，11 次基准运行）
> 已删除以腾出空间。**结论与数据仍然有效**，随时可用上面第 110 行的复现命令重新生成。

```
output/ffs_demo/<时间戳>/disp_vis.png     目视证据（左|右|视差）
output/ffs_demo/<时间戳>/disp.npy         视差（像素，float32，模型输入分辨率）
output/ffs_demo/<时间戳>/depth_meter.npy  深度（米）
output/ffs_demo/<时间戳>/cloud.ply        点云（左图着色）
output/ffs_demo/<时间戳>/meta.txt         本次全部参数与来源
output/bench/bench_*.json                 性能基准原始数据
```
（`output/` 不入库，需要长期保留请另行备份。）
