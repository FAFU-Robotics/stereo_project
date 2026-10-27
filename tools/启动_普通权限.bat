@echo off
REM ============================================================================
REM  stereo_project launcher  --  NORMAL user permissions (NO UAC, NO admin)
REM ----------------------------------------------------------------------------
REM  Verified 2026-10-21 on this machine:
REM    A normal Medium-integrity process can open the DECXIN stereo camera
REM    (DirectShow index 1, 1280x480, both eyes 640x480) without elevation.
REM    The old "must run as administrator" note was caused by testing inside a
REM    low-integrity / sandboxed session, not by normal Windows permissions.
REM ----------------------------------------------------------------------------
REM  Keys:
REM    [1] live stereo preview      [2] live disparity
REM    [3] live point cloud         [4] camera self-check (all indexes)
REM    [5] list camera devices      [0] exit
REM ----------------------------------------------------------------------------
REM  ASCII-only on purpose: cmd.exe parses .bat files byte by byte in the active
REM  OEM code page, so non-ASCII text here would break parsing.
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1

set "PROJECT_ROOT=%~dp0.."
set "PY=D:\31931\Documents\python312\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "TORCHDYNAMO_DISABLE=1"
cd /d "%PROJECT_ROOT%"

if not exist "%PY%" (
  echo [ERROR] project python not found: %PY%
  pause
  exit /b 1
)

REM Warn if this window happens to be elevated: it works, but it is not needed
REM and would make the test non-representative.
whoami /groups | findstr /C:"S-1-16-12288" >nul 2>&1
if not errorlevel 1 (
  echo.
  echo [WARN] This window is ELEVATED ^(administrator^).
  echo        The camera works as a NORMAL user; elevation is not required.
  echo        For a clean test, close this window and double-click this file again.
  echo.
)

:menu
echo ============================================================
echo   stereo_project   [NORMAL permissions - no admin required]
echo ------------------------------------------------------------
echo   Project : %CD%
echo   Python  : %PY%
echo ------------------------------------------------------------
echo   [1] Live stereo preview        (main.py)
echo   [2] Live disparity             (main.py --mode depth)
echo   [3] Live point cloud           (main.py --mode pointcloud)
echo   [4] Camera self-check          (tools\opencv_camera_check.py --quick)
echo   [5] List camera devices        (main.py --list)
echo   [0] Exit
echo ============================================================
set "CH="
set /p "CH=Type 1..5 or 0 then press Enter: "
if "%CH%"=="1" "%PY%" main.py --use-runtime
if "%CH%"=="2" "%PY%" main.py --mode depth --use-runtime
if "%CH%"=="3" "%PY%" main.py --mode pointcloud --use-runtime
if "%CH%"=="4" "%PY%" tools\opencv_camera_check.py --quick
if "%CH%"=="5" "%PY%" main.py --list
if "%CH%"=="0" goto done
echo.
pause
goto menu

:done
endlocal
exit /b 0
