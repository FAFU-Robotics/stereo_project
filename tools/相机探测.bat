@echo off
chcp 65001 >nul 2>&1
title 双目相机探测 (probe_cameras)
REM ============================================================
REM  双击本文件即可运行相机探测。
REM  它只读：只打开相机读帧 + 保存快照，不改动任何系统设置。
REM ============================================================

set "PY=D:\31931\Documents\python312\python.exe"
set "TEMP=D:\31931\Documents\_tmp"
set "TMP=D:\31931\Documents\_tmp"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM 切到项目根目录（本文件在 tools\ 下，所以是上一级）
cd /d "%~dp0.."

if not exist "%PY%" (
  echo [错误] 找不到 Python: %PY%
  echo        请确认 D:\31931\Documents\python312\python.exe 存在。
  pause
  exit /b 1
)

echo ============================================================
echo   开始探测所有摄像头（大约 1~3 分钟，请不要关窗口）
echo   索引范围 0~5，后端 dshow + msmf
echo ============================================================
echo.

"%PY%" tools\probe_cameras.py --max-index 6 --backends dshow,msmf --timeout 20 --warmup 8 --measure 15

echo.
echo ============================================================
echo   探测结束
echo   完整结果 : output\probe\report.json
echo   快照图片 : output\probe\snapshots\
echo.
echo   请把上面窗口里的内容截图发给我（或复制文字）。
echo ============================================================
pause
