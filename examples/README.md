# examples —— 六个可直接运行的示例

全部支持 `--mock`：没有相机、没装 torch 也能跑通，先确认接口好用再接真机。

| 文件 | 回答什么问题 | 需要相机 | 需要 torch |
|---|---|---|---|
| `01_read_depth.py` | 深度长什么样、单位是什么、怎么存 | 否（`--mock`） | 否（mock） |
| `02_save_pointcloud.py` | 连续存 N 帧点云，顺便看端到端帧率 | 否 | 否（mock） |
| `03_live_view.py` | 实时看深度、鼠标点选测距 | 否 | 否（mock） |
| `04_calib_report.py` | 这份标定准不准、能测多远 | **不需要相机** | **不需要** |
| `05_depth_statistics.py` | 跑一段时间后，质量有没有退化 | 否 | 否（mock） |
| `06_save_dataset.py` | 采一小段可复现的数据集（整套文件 + 索引） | 否 | 否（mock） |

## 怎么跑

```bash
# 在仓库根目录
python examples/01_read_depth.py --mock
python examples/02_save_pointcloud.py --mock --frames 5
python examples/03_live_view.py --mock          # q 退出
python examples/04_calib_report.py
python examples/05_depth_statistics.py --mock --seconds 3 --csv output/stats.csv
python examples/06_save_dataset.py --mock --frames 5 --out output/dataset
```

接真机时去掉 `--mock`，并把 `--config` 指到你的 `config.yaml`（默认就在根目录）。

## 六个示例各自解决什么

**01 / 02 / 03** 是“最小可用”：读一帧、存几帧、实时看。它们回答“接口能不能用”。

**04 是唯一不需要相机的**：它只读 `calibration/stereo_calib.yaml`，推算深度不确定度与可用量程。换标定前先用它判断这份标定值不值。

**05 与 02 的区别**：02 只回答“能不能连续存”，05 回答“连续跑的时候质量有没有退化”。05 会逐帧记录有效像素比例、中位深度、推理耗时，并给出 p50/p95/min 与首末帧漂移。真实项目里显存增长、曝光漂移、队列堆积都是靠它发现的。

**06 与 02 的区别**：两者都落盘，但目的不同。

| | 02 | 06 |
|---|---|---|
| 内容 | 只有点云 | 左右图 / 视差 / 深度 / 掩码 / 点云 / 元数据 |
| 实现 | `save_ply` | `export_result` |
| PLY 格式 | 二进制 | 二进制（`--tiff` 可额外存 TIFF） |
| 文件名 | 固定 `cloud_0001.ply` | `seq0001.cloud.ply` 等带前缀 |
| 索引 | 无 | `manifest.csv` + `dataset.json`（含标定摘要） |
| 适合 | 快速看几帧 | 交给别人复现、批量后处理 |

## 跑出来是什么样（本机 mock 实测输出）

```
seq=1 preset=balanced scale=0.75 infer=95ms valid=291839 median=0.653m points=72904

视差 disparity : (360, 480)  float32 像素（模型分辨率）
深度 depth     : (480, 640)  float32 米（已上采样到全分辨率）
有效掩码 valid : (480, 640)  bool，95.0% 有效
点云 points    : (72904, 3)  float32 米，y 向下

中心像素 (320,240) 深度 = 0.5725 m
```

注意 `disparity` 是 360×480 而 `depth` 是 480×640——两者形状不同，**不要用同一套像素坐标去索引**。这是这个项目里最容易写错的一处。

`05` 的输出示例（3 秒 mock）：

```
帧数 18（5.99 结果 FPS，统计窗口 3.0 s）
指标                   min       p50       p95       max
有效像素比例            0.9495    0.9500    0.9503    0.9508
中位深度 (m)          0.6520    0.7314    0.7497    0.7498
推理耗时 (ms)        95.0000   95.0000   95.0000   95.0000
```

`06` 的输出示例（2 帧 mock）：

```
[1/2] seq0001 有效  95.0% 中位 0.652 m  点  72904
[2/2] seq0002 有效  95.0% 中位 0.666 m  点  72949
完成：2 帧，跳过 0 帧（有效点不足 30%）
```

## 03 的按键

| 操作 | 作用 |
|---|---|
| 鼠标左键 | 读取该像素深度（窗口左右两幅并排，点右半边会自动换算坐标） |
| `s` | 导出一整套到 `output/example03/` |
| `+` / `-` | 调深度显示上限（只影响伪彩映射，不改数据） |
| `q` / `ESC` | 退出 |

## 用 Mock 时要知道的两件事

1. `MockSession` 生成的是**合成场景**（移动的方块），深度值不是真实测量结果，只用来验证接口与显示链路。上面 `05` 里那 68 mm 的“中位深度漂移”正是合成方块在移动，不是真实漂移。
2. Mock 的 `infer` 耗时是**写死的固定值**（95.0 ms），不代表你机器上的真实性能。真实帧率必须用真机跑 `02` 或 `05` 才有意义。

## 写自己的程序时

完整 API 与字段说明见 [../docs/SDK_API参考.md](../docs/SDK_API参考.md)。三条必须遵守的约定：

1. `take_result()` **不是消费式的**，要逐帧处理请用 `has_new_result()` 判断或按 `r.seq` 去重。
2. 单位一律**米**，点云 **y 向下**。
3. `depth` 与 `disparity` 形状不同。
