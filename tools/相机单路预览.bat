@echo off
chcp 65001 >nul 2>&1
title 单路相机预览 (preview_any)
REM ============================================================
REM  双击本文件，输入一个索引，就能看到这一路相机的画面。
REM  用途：逐个索引试，找出哪一个是 DECXIN 双目相机。
REM
REM  窗口内按键：  q/ESC = 退出    s = 保存当前帧    i = 打印状态
REM  注意看窗口里显示的 mean / std：
REM      mean 约等于 0 且 std 约等于 0  => 驱动没给真实图像（全黑）
REM      正常室内画面 mean 一般 20~200，std 大于 10
REM ============================================================

set "PY=D:\31931\Documents\python312\python.exe"
set "TEMP=D:\31931\Documents\_tmp"
set "TMP=D:\31931\Documents\_tmp"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0.."

set "IDX="
set /p IDX=请输入相机索引(0-5)然后回车: 
if "%IDX%"=="" set "IDX=2"

echo.
echo 打开索引 %IDX% ...（窗口内按 q 退出）
echo.
"%PY%" tools\preview_any.py --index %IDX%

echo.
pause
