"""aruco_align.py —— 用 ChArUco 板把两台相机的点云放进【同一个坐标系】
=============================================================================
【这是什么】
    一个**独立的新工具**。它做一件事：从一张拍到了 ChArUco 板的照片里，
    解出"板子在该相机坐标系里的位姿"，并给出 4x4 变换矩阵。
    把点云乘上这个矩阵，点云就从**相机坐标系**进入**板子坐标系**。
    两台相机各自做一次，两片点云就落在同一个坐标系里，可以直接叠加比对。

【为什么可行】
    两台相机不需要固定在一起 —— 只要它们**拍的是同一块静止不动的板子**，
    板子本身就定义了公共世界坐标系。

【关键约定（搞错就全错，务必看清）】
    1) 坐标系单位是**米**；点云也必须是米（若点云是毫米，用 --ply-units mm）
    2) 输出两个矩阵，别用混：
         camera_from_board : p_cam = M · p_board   （solvePnP 原值）
         board_from_cam    : p_board = M · p_cam   ← **点云要乘的是这个**
    3) 粘进 CloudCompare（Edit > Apply transformation）的是 board_from_cam，
       矩阵按**行主序**写，选中点云后应用。
    4) ⚠️ 自制双目相机启用标定校正后，保存的左图是**校正后**的图像，
       点云也在**校正后左目坐标系**里 —— 所以内参必须用 P1 且畸变为 0：
           默认 --intrinsics-from calibration/stereo_calib.yaml --which rectified
       如果哪天关掉了 rectify_before_infer，则要改 --which raw（用 K1+D1）。
       用错会让位姿整体偏移（cx 差 36 px ≈ 几毫米级偏差），而且不容易发现。

【用法】
    REM 1) 先自检：把检测结果画出来，人工确认没检错
    python tools\\aruco_align.py --intrinsics-from calibration\\stereo_calib.yaml --which rectified ^
        --image output\\captures\\xxx\\left_raw.png --check check_self.png

    REM 2) 算矩阵（可一次给多张图，自动取平均，更稳）
    python tools\\aruco_align.py --which rectified --image shot1.png shot2.png shot3.png

    REM 3) 算矩阵 + 顺手把点云变换好（原文件不动）
    python tools\\aruco_align.py --which rectified --image left_raw.png ^
        --ply cloud.ply --out cloud_in_board_frame.ply

    REM RealSense：内参从官方软件读出来直接给（分辨率必须与导出的 PNG 一致）
    python tools\\aruco_align.py --intrinsics 600.5,600.1,320.1,240.2 ^
        --dist 0.01,-0.02,0,0,0 --image color.png --ply cloud.ply --out rs_board.ply

    REM 板子尺寸如有偏差（打印后实量），一定要按实测给
    ... --square-mm 35.6 --marker-mm 26.7

【参数速查】
    --image            一张或多张拍到板子的图（多张会取平均，并给出一致性）
    --intrinsics-from  从标定文件读内参（我们相机的 stereo_calib.yaml）
    --which            rectified（默认，用 P1、畸变=0）| raw（用 K1+D1）
    --intrinsics       "fx,fy,cx,cy"（RealSense 等外部来源）
    --dist             "k1,k2,p1,p2,k3"，默认全 0；--dist none 同义
    --square-mm/--marker-mm/--cols/--rows/--dict  板子规格（默认 36/27/5/7/DICT_4X4_1000）
    --ply/--out        可选：读点云 -> 乘 board_from_cam -> 写新文件（不覆盖原文件）
    --ply-units        m（默认）| mm
    --check FILE       把检测结果画到图上存盘（人工确认用）
    --min-corners      少于这么多角点就报错（默认 6）
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------- 默认板子规格
# 与桌面上那张打印板一致：ChArUco 5x7，方格 36 mm，标记 27 mm
DEF_COLS, DEF_ROWS = 5, 7
DEF_SQUARE_MM, DEF_MARKER_MM = 36.0, 27.0
DEF_DICT = "DICT_4X4_1000"


# =============================================================================
# 板子与检测
# =============================================================================
def get_dictionary(name: str):
    d = getattr(cv2.aruco, name, None)
    if d is None:
        cands = [n for n in dir(cv2.aruco) if n.startswith("DICT_")]
        raise SystemExit(f"[错误] 找不到字典 {name}；可用示例：{cands[:6]} …")
    return cv2.aruco.getPredefinedDictionary(d)


def build_board(cols: int, rows: int, square_mm: float, marker_mm: float, dict_name: str):
    """构造 ChArUco 板对象。square_mm/marker_mm 用**实测值**最准。"""
    if marker_mm >= square_mm:
        raise SystemExit(f"[错误] marker 边长 ({marker_mm}) 必须小于方格边长 ({square_mm})")
    # 内部单位取米：这样 solvePnP 求出的平移也是米，与点云单位一致
    return cv2.aruco.CharucoBoard((int(cols), int(rows)), square_mm / 1000.0,
                                  marker_mm / 1000.0, get_dictionary(dict_name))


def make_detector(board):
    """带亚像素角点精化的 ChArUco 检测器（精化能明显提高位姿精度）。"""
    params = cv2.aruco.DetectorParameters()
    for attr, val in (("cornerRefinementMethod", "CORNER_REFINE_SUBPIX"),
                      ("cornerRefinementWinSize", 5),
                      ("cornerRefinementMaxIterations", 50),
                      ("cornerRefinementMinAccuracy", 0.01)):
        if not hasattr(params, attr):
            continue
        if isinstance(val, str):                     # 枚举常量：先确认存在再取
            if hasattr(cv2.aruco, val):
                setattr(params, attr, getattr(cv2.aruco, val))
        else:                                        # 数值参数直接用
            setattr(params, attr, val)
    det = cv2.aruco.CharucoDetector(board)
    try:
        det.setDetectorParameters(params)
    except Exception:
        pass
    return det


# =============================================================================
# 内参
# =============================================================================
def load_intrinsics(args) -> Tuple[np.ndarray, np.ndarray, str]:
    """返回 (K, D, 说明文字)。"""
    if args.intrinsics:
        v = [float(x) for x in str(args.intrinsics).replace(" ", "").split(",")]
        if len(v) != 4:
            raise SystemExit("[错误] --intrinsics 需要 4 个数：fx,fy,cx,cy")
        K = np.array([[v[0], 0, v[2]], [0, v[1], v[3]], [0, 0, 1]], np.float64)
        D = np.zeros(5, np.float64)
        note = f"命令行给出 fx={v[0]:.2f} fy={v[1]:.2f} cx={v[2]:.2f} cy={v[3]:.2f}"
    elif args.intrinsics_from:
        path = args.intrinsics_from
        if not os.path.isabs(path):
            cand = os.path.join(PROJECT_ROOT, path)
            path = cand if os.path.exists(cand) else path
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
        if args.which == "rectified":
            P = np.asarray(d["P1"], np.float64)
            K = np.array([[P[0, 0], 0, P[0, 2]], [0, P[1, 1], P[1, 2]], [0, 0, 1]], np.float64)
            D = np.zeros(5, np.float64)
            note = (f"标定文件校正后内参 P1（fx={K[0,0]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}），"
                    f"畸变=0（图像已校正）")
        else:
            K = np.asarray(d["K1"], np.float64).copy()
            D = np.asarray(d["D1"], np.float64).reshape(-1).copy()
            note = f"标定文件原始内参 K1（fx={K[0,0]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}）+ D1"
    else:
        raise SystemExit("[错误] 必须给内参：--intrinsics-from 或 --intrinsics")

    if args.dist and args.dist.strip().lower() not in ("none", ""):
        v = [float(x) for x in args.dist.replace(" ", "").split(",")]
        D = np.asarray(v, np.float64).reshape(-1)
        note += f"；畸变由命令行给出 {v}"
    return K, D, note


# =============================================================================
# 位姿
# =============================================================================
def detect_one(det, img: np.ndarray) -> Dict[str, Any]:
    ch, ids, mk, mids = det.detectBoard(img)
    n_mk = 0 if mids is None else len(mids)
    n_ch = 0 if ids is None else len(ids)
    return {"corners": ch, "ids": ids, "marker_corners": mk, "marker_ids": mids,
            "n_markers": n_mk, "n_corners": n_ch}


def undistort_points_inv_brown(pts: np.ndarray, K: np.ndarray, coeffs: Sequence[float]) -> np.ndarray:
    """把**像素坐标**按 RealSense 的 Inverse Brown Conrady 模型反畸变。

    为什么需要单独处理：RealSense 彩色流的畸变模型是 RS2_DISTORTION_INVERSE_BROWN_CONRADY，
    它的系数含义是"**畸变坐标 -> 无畸变坐标**"（与 OpenCV 的标准 Brown-Conrady 正好相反）。
    直接把系数丢给 OpenCV 的 solvePnP 会**把校正做反**，位姿会有系统性偏差。

    做法：不重采样图像（那需要数值求逆），而是**只把检测到的角点坐标**按公式换算到无畸变坐标，
    然后用"畸变为 0"的针孔模型求解位姿 —— 数学上等价且没有插值损失。

    公式照搬 librealsense rsutil.h 的 rs2_deproject_pixel_to_point（INVERSE_BROWN_CONRADY 分支）：
        x = (u - cx)/fx ; y = (v - cy)/fy
        r2 = x^2 + y^2
        f  = 1 + k1*r2 + k2*r2^2 + k3*r2^3
        xu = x*f + 2*p1*x*y + p2*(r2 + 2*x^2)
        yu = y*f + 2*p2*x*y + p1*(r2 + 2*y^2)
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    k1, k2, p1, p2, k3 = (list(coeffs) + [0.0] * 5)[:5]
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    x = (p[:, 0] - cx) / fx
    y = (p[:, 1] - cy) / fy
    r2 = x * x + y * y
    f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xu = x * f + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yu = y * f + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)
    return np.column_stack([xu * fx + cx, yu * fy + cy]).astype(np.float64)


def pose_from_corners(board, det_res: Dict[str, Any], K: np.ndarray, D: np.ndarray,
                      dist_model: str = "opencv"
                      ) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """返回 (rvec, tvec, 重投影RMS px, 参与角点数)。

    dist_model:
        "opencv"   : D 直接交给 OpenCV（标准 Brown-Conrady）
        "inv_brown": D 是 RealSense 的 Inverse Brown Conrady —— 先把角点反畸变，再用畸变=0 求解
        "none"     : 图像已无畸变（例如我们自制相机用的是校正后内参 P1）
    """
    objp, imgp = board.matchImagePoints(det_res["corners"], det_res["ids"])
    if objp is None or len(objp) < 6:
        raise ValueError("有效角点不足")
    objp = np.asarray(objp, np.float64)
    imgp = np.asarray(imgp, np.float64).reshape(-1, 2)

    if dist_model == "inv_brown":
        imgp_use = undistort_points_inv_brown(imgp, K, D)
        D_use = None
    elif dist_model == "none":
        imgp_use = imgp
        D_use = None
    else:
        imgp_use = imgp
        D_use = D if len(D) else None

    flags = getattr(cv2, "SOLVEPNP_ITERATIVE", 0)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp_use.reshape(-1, 1, 2), K, D_use, flags=flags)
    if not ok:
        raise ValueError("solvePnP 失败")
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D_use)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - imgp_use) ** 2).sum(1)).mean())
    return rvec, tvec, rms, len(objp)


def average_poses(poses: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """多帧位姿取平均：平移取均值；旋转用 SVD 投影回 SO(3)（比欧拉角平均稳）。"""
    ts = np.array([t.reshape(3) for _, t in poses])
    Rs = np.array([cv2.Rodrigues(r)[0] for r, _ in poses])
    t_mean = ts.mean(axis=0)
    R_mean = Rs.mean(axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return cv2.Rodrigues(R_ortho)[0], t_mean.reshape(3, 1)


def mat4_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, np.float64).reshape(3)
    return M


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def fmt_matrix(T: np.ndarray, indent: str = "    ") -> str:
    return "\n".join(indent + " ".join(f"{v: .8f}" for v in row) for row in T)


def pose_report(rvec: np.ndarray, tvec: np.ndarray) -> Dict[str, float]:
    t = tvec.reshape(3)
    R = cv2.Rodrigues(rvec)[0]
    n = R[:, 2]                                   # 板面法向（相机系）
    tilt = float(np.degrees(np.arccos(np.clip(abs(float(n[2])), -1.0, 1.0))))
    return {"z_mm": float(t[2]) * 1000.0, "dist_mm": float(np.linalg.norm(t)) * 1000.0,
            "tilt_deg": tilt, "tx_mm": float(t[0]) * 1000.0, "ty_mm": float(t[1]) * 1000.0}


# =============================================================================
# 点云读写（读用 Open3D，写用自带的 PLY 写出器以便加注释）
# =============================================================================
def read_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        import open3d as o3d
    except ImportError:
        raise SystemExit("[错误] 读 PLY 需要 open3d（项目已装）；或用 --no-ply 只算矩阵")
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, np.float64)
    cols = np.asarray(pcd.colors, np.float64)
    if pts.size == 0:
        raise SystemExit(f"[错误] 点云为空或读不出：{path}")
    rgb = None
    if cols.size:
        rgb = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
    return pts, rgb


def _ascii(text: str) -> str:
    """PLY 头部只能是 ASCII（RPly/Open3D 按 ASCII 解析），非 ASCII 一律替换掉。"""
    return str(text).encode("ascii", "replace").decode("ascii")


def write_ply(path: str, pts: np.ndarray, rgb: Optional[np.ndarray],
              comments: Optional[Sequence[str]] = None) -> str:
    """写二进制 PLY。

    注意（项目里踩过的坑）：
      1) PLY 的 comment 行必须放在 **format 行之后**，放到 format 之前会让
         RPly / Open3D 报 "unable to parse header"。
      2) 头部只能是 **ASCII** —— 中文注释会导致写文件直接抛 UnicodeEncodeError
         （自测抓到过），所以这里统一做 ASCII 替换。
    """
    pts = np.asarray(pts, np.float32)
    n = len(pts)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    head = ["ply", "format binary_little_endian 1.0"]
    for c in (comments or []):
        head.append(f"comment {_ascii(c)}")
    head.append(f"element vertex {n}")
    head += ["property float x", "property float y", "property float z"]
    if rgb is not None and len(rgb) == n:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    head.append("end_header")

    if rgb is not None and len(rgb) == n:
        rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    else:
        rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]

    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(rec.tobytes())
    return path


def apply_transform(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """p' = T · p（p 为列向量，齐次坐标）。"""
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    out = (T @ ph.T).T
    return out[:, :3]


# =============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="用 ChArUco 板求点云对齐到统一坐标系的 4x4 变换")
    ap.add_argument("--image", nargs="+", required=True, help="一张或多张拍到板子的图像")
    ap.add_argument("--intrinsics-from", default="", help="标定文件（我们相机用 calibration/stereo_calib.yaml）")
    ap.add_argument("--which", choices=("rectified", "raw"), default="rectified",
                    help="从标定文件取哪套内参：rectified=校正后 P1（默认）| raw=原始 K1+D1")
    ap.add_argument("--intrinsics", default="", help='外部内参 "fx,fy,cx,cy"（如 RealSense）')
    ap.add_argument("--dist", default="", help='畸变系数 "k1,k2,p1,p2,k3"，默认全 0')
    ap.add_argument("--dist-model", choices=("opencv", "inv_brown", "none"), default="opencv",
                    help="畸变模型：opencv=标准Brown-Conrady(默认) | inv_brown=RealSense的"
                         "Inverse Brown Conrady（先把角点反畸变再求解）| none=图像已无畸变")
    ap.add_argument("--cols", type=int, default=DEF_COLS)
    ap.add_argument("--rows", type=int, default=DEF_ROWS)
    ap.add_argument("--square-mm", type=float, default=DEF_SQUARE_MM, help="方格实测边长 mm")
    ap.add_argument("--marker-mm", type=float, default=DEF_MARKER_MM, help="标记实测边长 mm")
    ap.add_argument("--dict", default=DEF_DICT, help="ArUco 字典名")
    ap.add_argument("--ply", default="", help="可选：要变换的点云 PLY")
    ap.add_argument("--out", default="", help="变换后的 PLY 输出路径（不覆盖原文件）")
    ap.add_argument("--ply-units", choices=("m", "mm"), default="m", help="输入点云单位（默认米）")
    ap.add_argument("--ply-convention", choices=("camera", "realsense_viewer"), default="camera",
                    help="输入点云的坐标约定：camera=相机系(x右,y下,z前，默认) | "
                         "realsense_viewer=RealSense 官方软件导出的约定(x右,y上,-z前，"
                         "实测 viewer 与 rs-convert 都是这个)")
    ap.add_argument("--check", default="", help="把检测结果画到这张图里存盘（人工确认）")
    ap.add_argument("--min-corners", type=int, default=6, help="少于这么多角点就报错（默认 6）")
    ap.add_argument("--no-ply", action="store_true", help="只算矩阵，不读点云")
    args = ap.parse_args(argv)

    board = build_board(args.cols, args.rows, args.square_mm, args.marker_mm, args.dict)
    det = make_detector(board)
    K, D, knote = load_intrinsics(args)

    print("=" * 88)
    print("  ChArUco 对齐工具 —— 把点云放进板子坐标系")
    print("=" * 88)
    print(f"  板子    : {args.cols}x{args.rows} 方格 {args.square_mm:g} mm / 标记 {args.marker_mm:g} mm"
          f" / {args.dict}")
    print(f"  内参    : {knote}")
    if args.which == "rectified" and args.intrinsics_from:
        print("  [提醒] 用的是**校正后**内参：传进来的图必须是校正后的图、点云也必须是校正后坐标系。")
        print("         我们程序保存的 left_raw.png 在校正开启时正是校正后的图（文件名有误导）。")

    # ---------------------------------------------------------------- 逐图检测
    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    rms_list: List[float] = []
    for path in args.image:
        if not os.path.exists(path):
            print(f"  [跳过] 文件不存在: {path}")
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if False else cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [跳过] 读不出图像: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = detect_one(det, gray)
        name = os.path.basename(path)
        if res["n_corners"] < args.min_corners:
            print(f"  [失败] {name}: 标记 {res['n_markers']} 个 / 角点 {res['n_corners']} 个"
                  f"（需要 ≥{args.min_corners} 个角点）")
            if args.check:
                vis = img.copy()
                if res["marker_corners"] is not None and len(res["marker_corners"]):
                    cv2.aruco.drawDetectedMarkers(vis, res["marker_corners"], res["marker_ids"])
                cv2.imwrite(args.check, vis)
                print(f"         已把检测情况画到: {args.check}（看一眼为什么没检到）")
            continue
        try:
            rvec, tvec, rms, n_used = pose_from_corners(board, res, K, D, dist_model=args.dist_model)
        except Exception as exc:  # noqa: BLE001
            print(f"  [失败] {name}: {exc}")
            continue
        pr = pose_report(rvec, tvec)
        poses.append((rvec, tvec))
        rms_list.append(rms)
        print(f"  [OK]   {name}: 标记 {res['n_markers']} / 角点 {res['n_corners']}（用 {n_used}）"
              f"  沿光轴 {pr['z_mm']:.1f} mm  倾角 {pr['tilt_deg']:.1f}°  重投影 RMS {rms:.3f} px")
        if args.check:
            vis = img.copy()
            cv2.aruco.drawDetectedMarkers(vis, res["marker_corners"], res["marker_ids"])
            try:
                cv2.drawFrameAxes(vis, K, D if len(D) else None, rvec, tvec,
                                  args.square_mm / 1000.0 * 1.5)
            except Exception:
                pass
            cv2.imwrite(args.check, vis)

    if not poses:
        print("\n  ❌ 没有任何一张图成功解出位姿，无法生成矩阵。")
        print("     排查：板子是否完整/清晰可见？内参与图像分辨率是否匹配？")
        print("           （用 --check 看检测结果；RealSense 的内参分辨率必须与图一致）")
        return 2

    # ---------------------------------------------------------------- 汇总位姿
    rvec, tvec = average_poses(poses) if len(poses) > 1 else poses[0]
    pr = pose_report(rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    T_cam_from_board = mat4_from_Rt(R, tvec)          # p_cam = M · p_board
    T_board_from_cam = invert_T(T_cam_from_board)     # p_board = M · p_cam  ← 点云用这个

    print("-" * 88)
    if len(poses) > 1:
        ts = np.array([t.reshape(3) for _, t in poses]) * 1000.0
        Rs = np.array([cv2.Rodrigues(r)[0] for r, _ in poses])
        ang = [np.degrees(np.arccos(np.clip((np.trace(Rs[0].T @ Ri) - 1) / 2, -1, 1)))
               for Ri in Rs]
        print(f"  多帧平均: {len(poses)} 张    平移标准差 {ts.std(axis=0).max():.2f} mm"
              f"    姿态一致性 ±{max(ang):.2f}°")
        print(f"            重投影 RMS 中位 {np.median(rms_list):.3f} px"
              f"（越小越好，>1 px 建议检查图像质量）")
    print(f"  板面位置: 沿光轴 {pr['z_mm']:.1f} mm   横向 ({pr['tx_mm']:.1f}, {pr['ty_mm']:.1f}) mm"
          f"   倾角 {pr['tilt_deg']:.1f}°")
    # 粗略误差量级：角点像素误差 / 焦距 * 距离 / sqrt(N)
    fx = float(K[0, 0])
    sig = np.median(rms_list) * pr["z_mm"] / fx / max(np.sqrt(len(poses) * 8), 1)
    print(f"  位置误差量级估计: 约 {sig:.2f} mm（由重投影误差与几何粗估，仅供参考）")

    print("-" * 88)
    print("  【点云要乘的矩阵】相机 -> 板子  (board_from_cam)")
    print(fmt_matrix(T_board_from_cam))
    print()
    print("  【solvePnP 原值】板子 -> 相机  (camera_from_board)")
    print(fmt_matrix(T_cam_from_board))
    print("-" * 88)
    print("  怎么用（二选一）：")
    print("   A) CloudCompare：选中点云 -> Edit > Apply transformation -> 粘贴上面【相机 -> 板子】那个矩阵")
    print("      （行主序；粘贴后先看板子是否落到原点附近来核对方向）")
    print("   B) 让本脚本直接变换点云：加 --ply <输入.ply> --out <输出.ply>")

    # ---------------------------------------------------------------- 变换点云
    if args.ply and not args.no_ply:
        if not args.out:
            print("\n  [提示] 给了 --ply 但没给 --out，只在上面打印矩阵，不改点云。")
        else:
            pts, rgb = read_ply(args.ply)
            if args.ply_units == "mm":
                pts = pts / 1000.0            # 统一到米，矩阵才是对的
            if args.ply_convention == "realsense_viewer":
                # 实测：RealSense 官方软件（viewer 的保存 与 rs-convert -l）导出的 PLY
                # 用的是**查看器约定**：x 右、y 上、-z 前。必须先转成相机约定（x 右、y 下、z 前），
                # 也就是把 y 和 z 各取反（绕 x 轴转 180°），否则变换结果会整体错掉。
                pts[:, 1] = -pts[:, 1]
                pts[:, 2] = -pts[:, 2]
                print("  [约定] 已按 RealSense 查看器约定 (x右,y上,-z前) 转换到相机约定 (x右,y下,z前)")
            ext = pts.max(axis=0) - pts.min(axis=0)
            if args.ply_units == "m" and float(np.max(ext)) > 20.0:
                print(f"\n  ⚠️ 输入点云的尺寸跨度 {np.max(ext):.1f} 单位 —— 若是毫米请加 --ply-units mm，"
                      f"否则变换结果会错。")
            new_pts = apply_transform(pts, T_board_from_cam)
            newext = new_pts.max(axis=0) - new_pts.min(axis=0)
            # PLY 注释只用 ASCII（中文会让写头部失败），内参说明在这里换成英文摘要
            ksum = (f"fx={K[0,0]:.3f} fy={K[1,1]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f} "
                    f"dist={'yes' if np.any(np.abs(D) > 0) else 'none'} "
                    f"source={'P1-rectified' if args.which == 'rectified' and args.intrinsics_from else 'given'}")
            write_ply(args.out, new_pts, rgb, comments=[
                "coordinate_system: ChArUco board frame (x right, y down, z into the board)",
                "units: meter",
                f"source_ply: {os.path.basename(args.ply)}",
                f"board: {args.cols}x{args.rows} square {args.square_mm:g}mm "
                f"marker {args.marker_mm:g}mm {args.dict}",
                f"intrinsics: {ksum}",
                f"reprojection_rms_px: {np.median(rms_list):.3f}",
                f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')} by tools/aruco_align.py",
            ])
            print(f"\n  已写出变换后的点云: {args.out}")
            print(f"    点数 {len(new_pts)}   变换后包围盒跨度 "
                  f"({newext[0]*1000:.1f}, {newext[1]*1000:.1f}, {newext[2]*1000:.1f}) mm")
            print(f"    原文件未改动: {args.ply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
