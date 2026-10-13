@echo off
REM ============================================================================
REM  Intel RealSense live viewer  --  depth map + coloured point cloud
REM ----------------------------------------------------------------------------
REM  Runs tools\realsense_view.py WITH administrator rights.
REM
REM  WHY ELEVATED:
REM    1) On this machine the physical cameras can only be opened by an elevated
REM       process (a normal process gets HRESULT 0x80070005 from the frame server).
REM    2) The RealSense SDK creates a named mutex for cross-process device locking;
REM       in a restricted environment that fails with
REM       "CreateNamedMutex returned Mutex_TotalFailure".
REM    Running from this window (normal user session, elevated) avoids both.
REM ----------------------------------------------------------------------------
REM  This script only ADDS a tool: it does not modify any existing project file,
REM  and it never touches third_party\Fast-FoundationStereo.
REM
REM  ASCII-only on purpose: cmd.exe parses .bat files byte by byte in the active
REM  OEM code page, so non-ASCII text here would break parsing.
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1

REM ---- elevation check: High integrity SID = S-1-16-12288 ----
whoami /groups | findstr /C:"S-1-16-12288" >nul 2>&1
if errorlevel 1 (
  echo.
  echo  This viewer needs administrator rights ^(see the note inside this file^).
  echo  Asking Windows for them now - please click "Yes" on the prompt.
  echo.
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

title RealSense viewer (ADMIN)
set "PROJECT_ROOT=%~dp0.."
set "PY=D:\31931\Documents\python312\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "TEMP=%PROJECT_ROOT%\_work\pip_tmp"
set "TMP=%TEMP%"
cd /d "%PROJECT_ROOT%"
if not exist "%TEMP%" mkdir "%TEMP%" >nul 2>&1

echo ============================================================
echo   Intel RealSense viewer   [running as ADMINISTRATOR]
echo ------------------------------------------------------------
echo   Script : tools\realsense_view.py
echo   Shows  : colour image + depth colour map  (2D window)
echo            coloured point cloud             (3D window)
echo   Keys   : q=quit  s=save PLY+depth+colour  p=pause
echo            h=help  o=reopen the 3D window
echo ------------------------------------------------------------
echo   Step 1: device + factory parameters only (fast, no streaming)
echo ============================================================
echo.

if not exist "%PY%" (
  echo [ERROR] project python not found: %PY%
  pause
  exit /b 1
)

"%PY%" "tools\realsense_view.py" --check
echo.
echo ============================================================
echo   Step 2: live preview (colour + depth + point cloud)
echo ============================================================
"%PY%" "tools\realsense_view.py" %*
echo.
echo ------------------------------------------------------------
echo   Exit code: %ERRORLEVEL%
echo   0 = ok, 2 = device/open problem, 5 = missing python package
echo ------------------------------------------------------------
echo   Full output is also saved to:
echo     output\realsense\last_run.log
echo   If a window flashed and closed, send that log file.
echo ------------------------------------------------------------
pause
endlocal
