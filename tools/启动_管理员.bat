@echo off
REM ============================================================================
REM  stereo_project launcher  --  LEGACY / DIAGNOSTIC: runs WITH administrator rights
REM ----------------------------------------------------------------------------
REM  2026-10-21 UPDATE: normal use does NOT need administrator rights.
REM    A normal Medium-integrity process opens the DECXIN camera (index 1,
REM    DSHOW, 1280x480) correctly. Use  启动_普通权限.bat  for normal runs.
REM    This file is kept only to diagnose a low-integrity / sandboxed session,
REM    where Windows may return E_ACCESSDENIED for physical cameras.
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

echo ============================================================
echo   NOTICE: normal use does NOT need administrator rights.
echo   Please use  tools\启动_普通权限.bat  for normal runs.
echo   This launcher is only for a restricted / sandboxed session.
echo ============================================================
choice /C YN /N /M "Continue with administrator rights? [Y/N] "
if errorlevel 2 exit /b 0

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
call "%~dp0find_python.bat"
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
echo   [4] Live point cloud [stage 4]     (main.py --mode pointcloud)
echo   [5] Live point cloud, faster       (--scale 0.5 --valid-iters 4)
echo   [6] Point cloud + AUTO-SAVE        (runs until q; saves one frame at #20)
echo   [7] Point cloud, max_disp=320      (for NEAR objects; slower, ~4.1 GB VRAM)
echo   [8] Disparity + AUTO-SAVE          (runs until q; saves one frame at #20)
echo   [9] DirectShow capture test        (pygrabber, no OpenCV)
echo  [10] OpenCV camera check            (standalone; normal-vs-admin compare)
echo  [11] Calibration board probe        (stage 5 step 0: usable at which distance?)
echo  [12] Capture calibration images     (stage 5 step 1: checklist + quality gate)
echo   [0] Exit
echo ============================================================
set "DO="
set /p "CH=Type 1..12 or 0 then press Enter: "
if "%CH%"=="1" set "DO=PREVIEW"
if "%CH%"=="2" set "DO=DEPTH"
if "%CH%"=="3" set "DO=DEPTHFAST"
if "%CH%"=="4" set "DO=CLOUD"
if "%CH%"=="5" set "DO=CLOUDFAST"
if "%CH%"=="6" set "DO=CLOUDSAVE"
if "%CH%"=="7" set "DO=CLOUDNEAR"
if "%CH%"=="8" set "DO=DEPTHSAVE"
if "%CH%"=="9" set "DO=DSTEST"
if "%CH%"=="10" set "DO=CVTEST"
if "%CH%"=="11" set "DO=CALPROBE"
if "%CH%"=="12" set "DO=CALCAP"
if "%CH%"=="0" set "DO=EXIT"
if not defined DO goto menu
if /i "%DO%"=="EXIT" goto done
if /i "%DO%"=="PREVIEW"    call :do_preview
if /i "%DO%"=="DEPTH"      call :do_depth
if /i "%DO%"=="DEPTHFAST"  call :do_depth_fast
if /i "%DO%"=="CLOUD"      call :do_cloud
if /i "%DO%"=="CLOUDFAST"  call :do_cloud_fast
if /i "%DO%"=="CLOUDSAVE"  call :do_cloud_save
if /i "%DO%"=="CLOUDNEAR"  call :do_cloud_near
if /i "%DO%"=="CALPROBE"   call :do_calprobe
if /i "%DO%"=="CALCAP"     call :do_calcap
if /i "%DO%"=="DEPTHSAVE"  call :do_depth_save
if /i "%DO%"=="CVTEST"     call :do_cvtest
if /i "%DO%"=="DSTEST"     call :do_dstest
echo.
pause
goto menu

:do_cloud_near
echo.
echo --- live point cloud with max_disp = 320 (for NEAR objects) ---
echo     Use this when a large part of the picture sits closer than ~20 cm:
echo     with max_disp=192 those pixels are extrapolated by the network; 320 gives
echo     the model a real search range for them. Cost: slower (~4 FPS) and ~4.1 GB VRAM.
"%PY%" main.py --mode pointcloud --max-disp 320
exit /b

:do_calprobe
echo.
echo --- calibration board feasibility probe   [stage 5 / step 0] ---
echo     Board: DFOPTIX Q12-50-3 = 12x9 squares, 3.0 mm each (11x8 inner corners).
echo     Place the board facing the camera at 10/12/15/18/20/25/30 cm.
echo     Press SPACE at each distance to record one frame; press q to finish.
echo     Output: corners found / pixels per square / sharpness / contrast / verdict,
echo     plus evidence images and a report under output\calib_probe\.
"%PY%" tools\capture_calib.py --probe
exit /b

:do_calcap
echo.
echo --- capture calibration images   [stage 5 / step 1] ---
echo     Board DFOPTIX Q12-50-3 (12x9 squares, 3.0 mm). Working distance 9 - 17 cm.
echo     SPACE = record one pair (rejected with a reason if it fails the quality gate)
echo     u = undo last     f = force record     q = finish and write the report
echo     Keep the board in the MIDDLE-RIGHT of the frame so BOTH eyes can see it.
echo     Checklist: centre, 4 corners, and 4 tilt directions are tracked live.
"%PY%" tools\capture_calib.py --capture
exit /b

:do_cloud_save
echo.
echo --- live point cloud with AUTOMATIC SAVE (runs until you quit) ---
echo     Aim at a TEXTURED scene 0.5 - 2 m away, normal lighting, hold still.
echo     One frame is saved automatically at frame 20 (PLY + depth + images).
echo     NO key press needed, and it does NOT stop by itself: press q to quit.
"%PY%" main.py --mode pointcloud --save-snapshot
echo.
echo     Output folder: output\captures\<timestamp>_cloud\
exit /b

:do_cloud
echo.
echo --- live point cloud, STAGE 4 ---
echo     OpenCV window : left / right / depth (metres)
echo     Open3D window : coloured point cloud (left image colours)
echo     Keys: q or ESC = quit, p = pause, s = save (PLY + depth + images), i = info
echo     NOTE: while uncalibrated the cloud is NOT metric (placeholder intrinsics).
"%PY%" main.py --mode pointcloud
exit /b

:do_cloud_fast
echo.
echo --- live point cloud, FASTER preset (scale 0.5 + valid_iters 4) ---
"%PY%" main.py --mode pointcloud --scale 0.5 --valid-iters 4
exit /b

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

:do_depth_save
echo.
echo --- live disparity with AUTOMATIC SAVE (runs until you quit) ---
echo     Aim the camera at a TEXTURED scene (desk, keyboard, mug, box...),
echo     NOT at a blank wall. One frame is saved automatically at frame 20.
echo     It does NOT stop by itself: press q to quit.
"%PY%" main.py --mode depth --save-snapshot
echo.
echo     Output folder: output\captures\<timestamp>_autosave\
exit /b

:do_cvtest
echo.
echo --- OpenCV camera check (standalone) ---
echo     Scans indexes 0..2 on both backends (dshow / msmf) and prints whether the
echo     process is elevated. Compare with the NORMAL run: the non-elevated bat
echo     in this tools folder whose name starts with the camera word.
"%PY%" tools\opencv_camera_check.py
exit /b

:do_dstest
echo.
echo --- DirectShow capture test (pygrabber) ---
"%PY%" tools\dshow_capture_test.py --index 1 --seconds 5
exit /b

:done
endlocal
exit /b 0
