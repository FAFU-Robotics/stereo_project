# examples —— 四个可直接运行的示例

全部支持 `--mock`：没有相机、没装 torch 也能跑通，先确认接口好用再接真机。

| 文件 | 回答什么问题 | 需要相机 |
|---|---|---|
| `01_read_depth.py` | 深度长什么样、单位是什么、怎么存 | 否（`--mock`） |
| `02_save_pointcloud.py` | 连续存 N 帧点云，顺便看端到端帧率 | 否 |
| `03_live_view.py` | 实时看深度、鼠标点选测距 | 否 |
| `04_calib_report.py` | 这份标定准不准、能测多远 | **不需要相机** |

## 怎么跑

```bash
# 在仓库根目录
python examples/01_read_depth.py --mock
python examples/02_save_pointcloud.py --mock --frames 5
python examples/03_live_view.py --mock          # q 退出
python examples/04_calib_report.py
```

接真机时去掉 `--mock`，并把 `--config` 指到你的 `config.yaml`（默认就在根目录）。

## 跑出来是什么样（本机 mock 实测输出）

```
seq=1 preset=balanced scale=0.75 infer=95ms valid=291839 median=0.653m points=72904

视差 disparity : (360, 480)  float32 像素（模型分辨率）
深度 depth     : (480, 640)  float32 米（已上采样到全分辨率）
有效掩码 valid : (480, 640)  bool，95.0% 有效
点云 points    : (72904, 3)  float32 米，y 向下

中心像素 (320,240) 深度 = 0.5725 m
```

注意 `disparity` 是 360×480 而 `depth` 是 480×640——两者形状不同，**不要用同一套像素坐标去索引**。
这是这个项目里最容易写错的一处。

## 03 的按键

| 操作 | 作用 |
|---|---|
| 鼠标左键 | 读取该像素深度（窗口左右两幅并排，点右半边会自动换算坐标） |
| `s` | 导出一整套到 `output/example03/` |
| `+` / `-` | 调深度显示上限（只影响伪彩映射，不改数据） |
| `q` / `ESC` | 退出 |

## 用 Mock 时要知道的两件事

1. `MockSession` 生成的是**合成场景**（移动的方块），深度值不是真实测量结果，
   只用来验证接口与显示链路。
2. Mock 的 `infer` 耗时是**写死的固定值**，不代表你机器上的真实性能。
   真实帧率必须用真机跑 `02_save_pointcloud.py` 才有意义。
