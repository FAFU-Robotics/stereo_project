@echo off
chcp 65001 >nul 2>&1
title 阶段2 - FFS 官方示例验证
REM ============================================================
REM  用官方 demo_data 的左右图跑一遍 Fast-FoundationStereo。
REM  这是阶段 2 的验收命令，不需要相机。
REM
REM  前提：weights\23-36-37\ 下同时有
REM          model_best_bp2_serialize.pth  和  cfg.yaml
REM        若没有，本窗口会提示下载方式。
REM ============================================================

set "PY=D:\31931\Documents\python312\python.exe"
set "TEMP=D:\31931\Documents\_tmp"
set "TMP=D:\31931\Documents\_tmp"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM Windows 版 PyTorch 不带 Triton，必须禁用 torch.compile（否则 @torch.compile 会失败）
set "TORCHDYNAMO_DISABLE=1"
REM 防止多个 OpenMP 运行时冲突（PyTorch + Open3D 常见）
set "KMP_DUPLICATE_LIB_OK=TRUE"

cd /d "%~dp0.."

echo ============================================================
echo   1) 先检查权重是否就位
echo ============================================================
"%PY%" tools\run_ffs_demo.py --check-only
if errorlevel 3 (
  echo.
  echo   [需要先下载权重] 可以试运行：
  echo       "%PY%" tools\fetch_ffs_weights.py
  echo   或在浏览器里手动下载后放到 weights\23-36-37\ 下。
  pause
  exit /b 1
)

echo.
echo ============================================================
echo   2) 正式推理（关掉点云窗口后程序结束）
echo ============================================================
"%PY%" tools\run_ffs_demo.py --repeat 3

echo.
echo ============================================================
echo   结果在 output\ffs_demo\ 下（disp.npy / depth_meter.npy /
echo   disp_vis.png / cloud.ply / meta.txt / timings.txt）
echo ============================================================
pause
