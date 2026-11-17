"""stereo_depth.py —— Fast-FoundationStereo（FFS）模型封装
=============================================================================
职责（单一）：把官方 FFS 模型"包装"成本项目可以反复调用的推理器。
    - 模型**只加载一次**，常驻显存（绝不每帧重新 torch.load）
    - 负责官方要求的全部预处理：BGR→RGB、缩放、32 像素对齐 pad、转 tensor
    - 负责官方要求的后处理：unpad、clip、可选上采样回原图尺寸
    - 记录推理耗时、显存占用，供状态栏显示

**不负责**：视差转深度（阶段 4 的 depth_utils.py）、点云（阶段 4 的 pointcloud.py）。

-----------------------------------------------------------------------
【官方接口事实（已逐行读过 third_party 源码，不是猜测）】
    core/foundation_stereo.py : class FastFoundationStereo(nn.Module)
        forward(self, image1, image2, iters=12, test_mode=False,
                low_memory=False, init_disp=None, profile=False,
                optimize_build_volume='pytorch1')
    core/utils/utils.py       : class InputPadder(dims, mode='sintel',
                                                   divis_by=8, force_square=False)
        .pad(*tensors) / .unpad(tensor)
    Utils.py                  : AMP_DTYPE = torch.float16
                                 normalize_image() 在 forward 内部自己做
    权重形式：官方 run_demo.py 用 torch.load(model_dir, weights_only=False)
              直接反序列化整个模型对象；同目录必须有 cfg.yaml。

【Windows 上的两个关键约束（已在阶段 0/1 实测确认）】
    1) 官方要求 torch==2.6.0+cu124，但 RTX 5060 是 sm_120，cu124 不含该内核，
       本机必须用 torch 2.8.0+cu128（已实测可用）。
    2) Windows 版 PyTorch 不带 Triton，而官方代码里有多处 @torch.compile，
       且 GWC 代价体的 triton 内核也用不了。因此：
         - 必须在 import torch **之前** 设置 TORCHDYNAMO_DISABLE=1（本模块第一件事）
         - 代价体后端固定用 'pytorch1'（纯 PyTorch 实现）
    代价是推理变慢；后续若要提速，再单独评估 TensorRT / triton-windows。

【数据格式约定】
    infer() 输入：left_bgr / right_bgr，np.uint8，(H, W, 3)，**BGR**（OpenCV 原生顺序）
    infer() 输出：dict，键见 infer() 文档
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ==== 必须在 import torch 之前执行：禁用 torch.compile（Windows 无 Triton）====
# 放在模块顶层而不是靠 bat 设环境变量，是为了"无论谁怎么调用都不会漏设"。
if str(os.environ.get("TORCHDYNAMO_DISABLE", "")).strip() not in ("1", "true", "True"):
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


# =============================================================================
# Python 3.14+ 兼容：torch.compile 是**硬错误**，而 FFS 在模块顶层就用它做装饰器
# -----------------------------------------------------------------------------
# 实测（2026-10-27，torch 2.9.1+cu128 / Python 3.14.6）：
#     torch/__init__.py:2590  raise RuntimeError("torch.compile is not supported on Python 3.14+")
# 而 third_party/.../core/submodule.py:376 是 `@torch.compile` ——
# **装饰器在 import 阶段就求值**，所以不处理的话 `import core.foundation_stereo` 直接抛
# RuntimeError，整个推理链路在 3.14 上完全不可用。
#
# TORCHDYNAMO_DISABLE=1 管不住它：那个开关决定"是否真的去编译"，而这里是版本检查，
# 在真正编译之前就把函数拒了。
#
# 处理方式：把 torch.compile 换成恒等装饰器。
# 为什么这不改变数值结果：在 3.12/3.13 + TORCHDYNAMO_DISABLE=1 的组合下，
# 装饰器本来也只是把函数包一层、按原样逐条执行（FFS 里没有任何地方依赖它的图语义）。
# 也就是说"恒等替换"与"已经验证过可用"的那套环境行为等价。
# =============================================================================
if sys.version_info >= (3, 14) and not getattr(torch.compile, "_stereo_identity", False):

    def _identity_compile(model=None, **kwargs):  # type: ignore[misc]
        """替代 torch.compile：不编译，原样返回传入的可调用对象。"""
        if model is None:
            return lambda fn: fn
        return model

    _identity_compile._stereo_identity = True  # type: ignore[attr-defined]
    torch.compile = _identity_compile  # type: ignore[assignment]


class FFSError(RuntimeError):
    """FFS 相关的可预期错误（权重缺失、接口不符、显存不足等）。"""


def resolve_infer_scale(src_w: int, requested: float, limit: int = 1000) -> float:
    """决定这一帧送给网络的缩放。不放大超过标定图，也不让模型宽度达到 limit。

    标定内参对应的是相机实际输出的那幅图（本机单眼 640×480）。
    scale = 模型宽度 / 标定图宽度。把标定图放大（scale>1）不会增加真实像素，
    视差单位也会和 fx 对不齐，所以这里直接夹到 1。
    标定图宽度本身小于 limit 时，scale=1 就是在标定分辨率上推理，不用重标定。
    """
    try:
        req = float(requested)
    except (TypeError, ValueError):
        req = 1.0
    if not (req > 0.0):
        req = 1.0
    req = min(req, 1.0)
    width = int(src_w)
    if width > 0 and width * req >= float(limit):
        req = (float(limit) - 1.0) / float(width)
    return req


def disparity_to_source(disp: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    """把模型网格上的视差变成标定图像素单位，并铺到标定图尺寸。

    模型视差的 1 像素等于标定图上的 (src_w / model_w) 像素。
    深度仍用原来的 fx、基线：full_disp = model_disp * src_w / model_w，
    再配未缩放的标定内参，和「模型视差 + fx*scale」是同一个深度。
    """
    src = np.asarray(disp, np.float32)
    if src.ndim != 2:
        raise ValueError(f"视差必须是 2D，当前形状 {src.shape}")
    mh, mw = int(src.shape[0]), int(src.shape[1])
    if mw <= 0 or mh <= 0 or int(src_w) <= 0 or int(src_h) <= 0:
        return np.ascontiguousarray(src)
    factor = float(src_w) / float(mw)
    if (mh, mw) == (int(src_h), int(src_w)):
        if factor == 1.0:
            return np.ascontiguousarray(src)
        return np.ascontiguousarray(src * factor)
    up = cv2.resize(src, (int(src_w), int(src_h)), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(up * factor)


def _project_root_hint(repo_root: str) -> str:
    """项目根的"提示值"：只用于报错信息里显示，不参与真实查找。

    打包后 repo_root/../.. 会指向 _MEIPASS（只读临时目录），而用户真正关心的
    "权重该放哪"是可写根（exe 旁边）。所以优先问 app_paths。
    """
    try:
        from app_paths import data_root
        return data_root()
    except Exception:                        # noqa: BLE001 —— app_paths 缺失时退回旧算法
        return os.path.normpath(os.path.join(repo_root, "..", ".."))


def _find_resource(rel: str, fallback_root: str) -> str:
    """按"可写根 → 只读根 → cwd"找只读资源（权重、FFS 仓库）。

    这样三种部署方式都能跑：源码运行、权重打进包里（_MEIPASS）、
    权重放在 exe 旁边（用户可自行替换/升级权重）。
    """
    try:
        # find() 先看可写根（exe 旁边），再看只读包内。
        # 用户把新权重放在 exe 旁即可替换，不必改 _internal。
        from app_paths import find
        return find(rel)
    except Exception:                        # noqa: BLE001
        return os.path.normpath(os.path.join(fallback_root, rel))


def _ensure_ffs_on_path(repo_root: str) -> None:
    """把官方仓库根目录加入 sys.path。

    为什么必须做：
        官方 core/*.py 里写的是 `sys.path.append(f'{code_dir}/../')` 再 `import Utils`，
        也就是"官方仓库根目录"必须能被 import 到。我们不修改官方代码，而是把它加到
        sys.path 里（保持官方目录结构原样，这是需求里明确要求的）。
    """
    if not os.path.isdir(repo_root):
        raise FFSError(f"找不到官方 FFS 仓库目录：{repo_root}")
    if not os.path.isfile(os.path.join(repo_root, "core", "foundation_stereo.py")):
        raise FFSError(
            f"{repo_root} 里没有 core/foundation_stereo.py，"
            "请确认已把官方源码解压到该目录（third_party/Fast-FoundationStereo）"
        )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class FFSDepthEstimator:
    """FFS 视差推理器（一个进程内只应创建一个实例）。

    典型用法：
        est = FFSDepthEstimator(cfg["project"], cfg["ffs"])
        ok, msg = est.weights_ready()      # 先检查权重是否就位
        est.load()                          # 只加载一次
        est.warmup(3)                       # 预热（第一次最慢）
        out = est.infer(left_bgr, right_bgr)

    输入（构造参数）：
        project_cfg : config.yaml 的 project 段（提供 ffs_repo 路径）
        ffs_cfg     : config.yaml 的 ffs 段（模型目录、迭代次数、缩放等）
        device      : 'cuda' 或 'cpu'；本项目固定 'cuda'
    """

    def __init__(self, project_cfg: Dict[str, Any], ffs_cfg: Dict[str, Any], device: str = "cuda") -> None:
        self.project_cfg = dict(project_cfg or {})
        self.cfg = dict(ffs_cfg or {})
        self.device = device

        self.repo_root: str = os.path.normpath(str(self.project_cfg.get("ffs_repo", "")))
        weights_rel: str = str(self.cfg.get("weights_dir", "weights/23-36-37"))
        # 权重目录是**只读资源**。原来靠 repo_root/../.. 反推项目根，这在源码布局下成立
        # （third_party/Fast-FoundationStereo -> 项目根），但打包后 exe 目录层级变了，
        # 而且用户很可能想把权重放在 exe 旁边方便替换。
        # 所以改成"按相对路径去找"：可写根 → _MEIPASS → cwd，见 app_paths.resource()。
        self.project_root = _project_root_hint(self.repo_root)
        if os.path.isabs(weights_rel):
            self.weights_dir = os.path.normpath(weights_rel)
        else:
            self.weights_dir = _find_resource(weights_rel, self.project_root)
        self.model_file: str = str(self.cfg.get("model_file", "model_best_bp2_serialize.pth"))
        self.model_path = os.path.join(self.weights_dir, self.model_file)
        self.cfg_path = os.path.join(self.weights_dir, "cfg.yaml")

        self.valid_iters: int = int(self.cfg.get("valid_iters", 8))
        self.max_disp: int = int(self.cfg.get("max_disp", 192))
        self.scale: float = float(self.cfg.get("scale", 1.0) or 1.0)
        self.build_volume: str = str(self.cfg.get("build_volume", "pytorch1"))
        self.hiera: int = int(self.cfg.get("hiera", 0) or 0)
        self.warmup_iters: int = int(self.cfg.get("warmup_iters", 3))

        self.model: Optional[torch.nn.Module] = None
        self.loaded: bool = False
        self.load_seconds: float = 0.0
        self.stats: Dict[str, Any] = {
            "runs": 0,
            "last_infer_ms": 0.0,
            "last_total_ms": 0.0,
            "avg_infer_ms": 0.0,
            "peak_mem_mb": 0.0,
            "model_input_size": (0, 0),
        }
        self._warned_large = False

    # ------------------------------------------------------------ 权重可用性
    def weights_ready(self) -> Tuple[bool, str]:
        """检查权重目录是否完整（不加载模型，秒回）。

        输出：(ok, message)
            ok=False 时 message 说明缺什么、该放到哪里。
        """
        missing: List[str] = []
        if not os.path.isfile(self.model_path):
            missing.append(self.model_file)
        if not os.path.isfile(self.cfg_path):
            missing.append("cfg.yaml")
        if missing:
            return False, (
                f"权重不完整，缺少：{', '.join(missing)}\n"
                f"        应放在：{self.weights_dir}\n"
                f"        官方要求该目录同时包含 {self.model_file} 和 cfg.yaml（cfg.yaml 会被官方代码读取）"
            )
        size_mb = os.path.getsize(self.model_path) / 1024 ** 2
        if size_mb < 10:
            return False, f"权重文件只有 {size_mb:.1f} MB，明显不完整（正常应为数百 MB）"
        return True, f"权重就位：{self.model_path}（{size_mb:.1f} MB）"

    # ---------------------------------------------------------------- 加载
    def load(self) -> None:
        """加载模型（只做一次）。失败抛 FFSError，并给出可操作的原因。"""
        if self.loaded:
            return
        ok, msg = self.weights_ready()
        if not ok:
            raise FFSError(msg)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise FFSError("CUDA 不可用，无法在 GPU 上推理（请检查显卡驱动/PyTorch 安装）")

        _ensure_ffs_on_path(self.repo_root)

        t0 = time.perf_counter()
        try:
            # 官方 run_demo.py 的做法：直接反序列化整个模型对象
            # weights_only=False 是官方要求（模型对象里不止有 state_dict）
            # 注意：这是一个"完全信任的本地文件"，不要对来源不明的 pth 这样做
            model = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            raise FFSError(
                f"加载权重失败：{type(exc).__name__}: {exc}\n"
                f"        文件：{self.model_path}\n"
                "        常见原因：文件下载不完整 / 不是这个模型的权重 / PyTorch 版本不匹配"
            ) from exc

        if not hasattr(model, "forward"):
            raise FFSError(f"torch.load 得到的对象不是模型（类型 {type(model)}），权重文件可能不对")

        # 官方参数覆盖：迭代次数与最大视差是可调的速度/精度折中
        try:
            model.args.valid_iters = self.valid_iters
            model.args.max_disp = self.max_disp
        except Exception:  # noqa: BLE001
            raise FFSError("模型缺少 args.valid_iters / args.max_disp，接口与预期不符（官方版本可能已变更）")

        model = model.to(self.device).eval()
        torch.autograd.set_grad_enabled(False)
        self.model = model
        self.loaded = True
        self.load_seconds = time.perf_counter() - t0

        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        print(
            f"[FFS] 模型已加载（{self.load_seconds:.2f}s，{n_param:.1f}M 参数）\n"
            f"      设备={self.device}({torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'}) "
            f"valid_iters={self.valid_iters} max_disp={self.max_disp} "
            f"代价体后端={self.build_volume} "
            f"torch.compile={'恒等替换(Py3.14+)' if getattr(torch.compile, '_stereo_identity', False) else ('关闭' if os.environ.get('TORCHDYNAMO_DISABLE') == '1' else '开启')}"
        )

    # ---------------------------------------------------------------- 推理
    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> Dict[str, Any]:
        """对一对左右图做视差推理。

        输入：
            left_bgr / right_bgr : np.uint8, (H, W, 3), **BGR**（OpenCV 摄像头的原生顺序）
                                   两图尺寸必须相同
        输出：dict
            {
              "disp"            : np.float32 (Hm, Wm)  模型输入分辨率下的视差（像素），已 clip(0,None)
              "disp_full"       : np.float32 (H, W)    标定图尺寸上的视差，单位是标定图像素
              "model_size"      : (Wm, Hm)             模型实际吃进去的尺寸（32 的整数倍）
              "src_size"        : (W, H)              送进来的原始尺寸（标定分辨率）
              "scale"           : float               实际用的缩放（模型宽度 / 标定图宽度）
              "left_model_rgb"  : np.uint8 (Hm, Wm, 3) 模型输入尺寸的左图（RGB）。
                                  点云着色前要用 cloud_color_bgr() 转回 BGR，不能直接送进 build()
              "timings_ms"      : {"preprocess","infer","total"}
              "valid_ratio"     : float               视差 > 0 的像素比例（有效深度比例的雏形）
            }

        注意：
            1) 官方模型吃的是 **RGB**；OpenCV 给的是 BGR，本函数内部已做转换。
            2) 模型建议输入宽度 < 1000；超过会打印一次警告（可用 ffs.scale 缩小）。
            3) 视差是"模型输入分辨率"下的像素值，换算深度时内参要同步乘 scale
               （官方 run_demo.py 里就是 K[:2] *= scale）。
        """
        if not self.loaded or self.model is None:
            raise FFSError("模型尚未加载，请先调用 load()")
        if left_bgr is None or right_bgr is None:
            raise FFSError("输入图像为空")
        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            raise FFSError(f"左右图尺寸不一致：{left_bgr.shape[:2]} vs {right_bgr.shape[:2]}")

        t_start = time.perf_counter()

        # ---- 1) 预处理：BGR->RGB。scale=1 时不再缩小，网络看到整幅标定图 ----
        img0 = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img0.shape[:2]
        scale = resolve_infer_scale(W0, self.scale)
        if abs(scale - 1.0) > 1e-6:
            dst_w = max(1, int(round(W0 * scale)))
            dst_h = max(1, int(round(H0 * scale)))
            # 缩小用面积插值，避免默认线性插值把细结构抹掉后再送给网络。
            img0 = cv2.resize(img0, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
            img1 = cv2.resize(img1, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
        H, W = img0.shape[:2]
        if W >= 1000 and not self._warned_large:
            self._warned_large = True
            print(f"[FFS][警告] 模型输入宽度 {W} >= 1000，官方建议 < 1000；可用 ffs.scale 缩小以提速。")

        left_model_rgb = img0.copy()

        t_pre_end = time.perf_counter()

        # ---- 2) 转 tensor 并 pad 到 32 的整数倍（官方 InputPadder）----
        from core.utils.utils import InputPadder  # 官方实现，直接复用

        t0 = torch.as_tensor(img0).to(self.device).float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(img1).to(self.device).float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)

        # ---- 3) 推理 ----
        from Utils import AMP_DTYPE  # 官方定义：torch.float16

        if self.device == "cuda":
            torch.cuda.synchronize()
        t_infer_start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(self.device == "cuda"), dtype=AMP_DTYPE):
            if not self.hiera:
                disp = self.model.forward(
                    t0, t1, iters=self.valid_iters, test_mode=True, optimize_build_volume=self.build_volume
                )
            else:
                disp = self.model.run_hierachical(
                    t0, t1, iters=self.valid_iters, test_mode=True, small_ratio=0.5
                )
        if self.device == "cuda":
            torch.cuda.synchronize()
        t_infer_end = time.perf_counter()

        # ---- 4) 后处理：unpad -> numpy -> clip ----
        disp = padder.unpad(disp.float())
        disp_np = disp.data.cpu().numpy().reshape(H, W).clip(0, None).astype(np.float32)
        # 换回标定图像素。深度可以继续用未改过的 fx，只要视差和 fx 用同一套像素。
        disp_full = disparity_to_source(disp_np, W0, H0)

        t_end = time.perf_counter()

        # ---- 5) 统计 ----
        infer_ms = (t_infer_end - t_infer_start) * 1000.0
        total_ms = (t_end - t_start) * 1000.0
        pre_ms = (t_pre_end - t_start) * 1000.0
        n = self.stats["runs"]
        self.stats["avg_infer_ms"] = (self.stats["avg_infer_ms"] * n + infer_ms) / (n + 1)
        self.stats["runs"] = n + 1
        self.stats["last_infer_ms"] = infer_ms
        self.stats["last_total_ms"] = total_ms
        self.stats["model_input_size"] = (W, H)
        if self.device == "cuda":
            self.stats["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2

        return {
            "disp": disp_np,
            "disp_full": disp_full,
            "model_size": (W, H),
            "src_size": (W0, H0),
            "scale": scale,
            "left_model_rgb": left_model_rgb,
            "timings_ms": {"preprocess": pre_ms, "infer": infer_ms, "total": total_ms},
            "valid_ratio": float((disp_np > 0).mean()),
        }

    # ---------------------------------------------------------------- 预热
    def warmup(self, n: Optional[int] = None, left: Optional[np.ndarray] = None, right: Optional[np.ndarray] = None) -> float:
        """预热 n 次，返回最后一次的 total 耗时(ms)。

        为什么必须"用真实尺寸的图"预热：
            CUDA / cuDNN 的算子选择与显存池是**按张量形状**缓存的。
            用 640x480 的假图预热，再跑 960x540 的真实图，第一次仍然会"冷启动"
            （实测 2088 ms 对 270 ms），从而把耗时统计彻底带偏。
            因此：能拿到真实图像时，就用真实图像预热（left/right 参数）。

        输入：
            n            : 预热次数，默认取 ffs.warmup_iters
            left / right : 可选，真实左右图（BGR uint8）。给了就用它们预热。
        输出：最后一次 total(ms)。预热**不计入**性能统计（会先重置 runs）。
        """
        if not self.loaded:
            raise FFSError("模型尚未加载，请先调用 load()")
        n = self.warmup_iters if n is None else int(n)
        if n <= 0:
            return 0.0
        if left is None or right is None:
            left = np.zeros((480, 640, 3), dtype=np.uint8)
            right = np.zeros((480, 640, 3), dtype=np.uint8)
            shape_note = "占位尺寸 640x480（未提供真实图像，冷启动可能仍会出现在第一次正式推理里）"
        else:
            shape_note = f"真实尺寸 {left.shape[1]}x{left.shape[0]}"
        last_ms = 0.0
        for i in range(n):
            out = self.infer(left, right)
            last_ms = out["timings_ms"]["total"]
            print(f"[FFS] 预热 {i + 1}/{n}（{shape_note}）：{last_ms:.0f} ms")
        # 预热数据不作为性能统计
        self.stats["runs"] = 0
        self.stats["avg_infer_ms"] = 0.0
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        return last_ms

    # ---------------------------------------------------------------- 信息
    def info(self) -> Dict[str, Any]:
        """返回一行式状态信息（供 viewer / main 显示）。"""
        args = getattr(self.model, "args", None) if self.model is not None else None
        return {
            "loaded": self.loaded,
            "load_seconds": round(self.load_seconds, 2),
            "model_path": self.model_path,
            "valid_iters": self.valid_iters,
            "max_disp": self.max_disp,
            "scale": self.scale,
            "build_volume": self.build_volume,
            "avg_infer_ms": round(self.stats["avg_infer_ms"], 1),
            "fps_from_infer": round(1000.0 / self.stats["avg_infer_ms"], 2) if self.stats["avg_infer_ms"] > 0 else 0.0,
            "peak_mem_mb": round(self.stats["peak_mem_mb"], 1),
            "model_args_keys": sorted(list(args.keys())) if hasattr(args, "keys") else [],
        }

    def close(self) -> None:
        """释放模型与显存。可重复调用。"""
        self.model = None
        self.loaded = False
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
