@echo off
REM ============================================================================
REM  stereo_project launcher  --  runs the project WITH administrator rights
REM ----------------------------------------------------------------------------
REM  WHY ELEVATED:
REM    On this machine the physical cameras can only be opened by a process
REM    running with administrator rights. A normal (non-elevated) process gets
REM    HRESULT 0x80070005 (E_ACCESSDENIED) from the Windows camera frame server.
REM    Verified: an elevated window received 39 real frames in 5 seconds from
REM    the DECXIN stereo camera (2560x960, side-by-side left/right).
REM    This is a WORKAROUND, not a fix of the underlying permission problem.
REM ----------------------------------------------------------------------------
REM  What it does:
REM    [1] OpenCV index scan          (checks which camera indexes OpenCV can open)
REM    [2] OpenCV capture test        (grabs frames and saves a PNG snapshot)
REM    then a menu:
REM      [1] live stereo preview      (runs main.py, opens a window)
REM      [2] repeat the OpenCV test
REM      [0] exit
REM ----------------------------------------------------------------------------
REM  ASCII-only on purpose: cmd.exe parses .bat files byte by byte in the active
REM  OEM code page, so non-ASCII text here would break parsing.
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1

REM ---- elevation check: High integrity SID = S-1-16-12288 ----
whoami /groups | findstr /C:"S-1-16-12288" >nul 2>&1
if errorlevel 1 (
  echo.
  echo  This launcher needs administrator rights ^(see the note inside the file^).
  echo  Asking Windows for them now - please click "Yes" on the prompt.
  echo.
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

title stereo_project (ADMIN)
set "PROJECT_ROOT=%~dp0.."
set "PY=D:\31931\Documents\python312\python.exe"
set "PYTHONPATH=%PROJECT_ROOT%\_work\dsgraph"
set "TEMP=%PROJECT_ROOT%\_work\pip_tmp"
set "TMP=%TEMP%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "TORCHDYNAMO_DISABLE=1"
cd /d "%PROJECT_ROOT%"
if not exist "%TEMP%" mkdir "%TEMP%" >nul 2>&1

if not exist "%PY%" (
  echo [ERROR] project python not found: %PY%
  pause
  exit /b 1
)

echo ============================================================
echo   stereo_project launcher  [running as ADMINISTRATOR]
echo ------------------------------------------------------------
echo   Project : %CD%
echo   Python  : %PY%
echo ------------------------------------------------------------
echo   Camera devices:
pnputil /enum-devices /class Camera | findstr /C:"Device Description" /C:"Status"
echo ============================================================
echo.

echo ==== STEP 1/2 : OpenCV index scan ====
"%PY%" main.py --list
echo.
echo ==== STEP 2/2 : OpenCV capture test (index 1 = DECXIN, 1280x480 MJPG) ====
"%PY%" tools\preview_any.py --index 1 --no-display --max-frames 15 --res 1280x480 --fourcc MJPG --save "output\probe\opencv_idx1.png"
echo.
echo   Snapshot (if any): output\probe\opencv_idx1.png
echo.

:menu
echo ============================================================
echo   [1] Live stereo preview            (main.py --mode preview)
echo   [2] Live disparity  [stage 3]      (main.py --mode depth)
echo   [3] Live disparity, faster preset  (--scale 0.5 --valid-iters 4)
echo   [4] DirectShow capture test        (pygrabber, no OpenCV)
echo   [5] OpenCV capture test            (saves a snapshot)
echo   [0] Exit
echo ============================================================
set "DO="
set /p "CH=Type 1, 2, 3, 4, 5 or 0 then press Enter: "
if "%CH%"=="1" set "DO=PREVIEW"
if "%CH%"=="2" set "DO=DEPTH"
if "%CH%"=="3" set "DO=DEPTHFAST"
if "%CH%"=="4" set "DO=DSTEST"
if "%CH%"=="5" set "DO=CVTEST"
if "%CH%"=="0" set "DO=EXIT"
if not defined DO goto menu
if /i "%DO%"=="EXIT" goto done
if /i "%DO%"=="PREVIEW"   call :do_preview
if /i "%DO%"=="DEPTH"     call :do_depth
if /i "%DO%"=="DEPTHFAST" call :do_depth_fast
if /i "%DO%"=="CVTEST"    call :do_cvtest
if /i "%DO%"=="DSTEST"    call :do_dstest
echo.
pause
goto menu

:do_preview
echo.
echo --- live stereo preview (stage 1) ---
echo     Keys: q or ESC = quit, p = pause, s = save, i = info, h = help
"%PY%" main.py
exit /b

:do_depth
echo.
echo --- live disparity, FFS (stage 3) ---
echo     Keys: q or ESC = quit, p = pause, s = save, i = info, h = help
echo     NOTE: uncalibrated - for process validation only, NOT for measurement
"%PY%" main.py --mode depth
exit /b

:do_depth_fast
echo.
echo --- live disparity, FASTER preset: scale 0.5 + valid_iters 4 ---
echo     (lower resolution disparity, roughly 2x faster than the default)
"%PY%" main.py --mode depth --scale 0.5 --valid-iters 4
exit /b

:do_cvtest
echo.
echo --- OpenCV capture test ---
"%PY%" tools\preview_any.py --index 1 --no-display --max-frames 15 --res 1280x480 --fourcc MJPG --save "output\probe\opencv_idx1.png"
exit /b

:do_dstest
echo.
echo --- DirectShow capture test (pygrabber) ---
"%PY%" tools\dshow_capture_test.py --index 1 --seconds 5
exit /b

:done
endlocal
exit /b 0
