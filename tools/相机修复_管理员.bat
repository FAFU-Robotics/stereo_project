@echo off
REM ============================================================================
REM  Camera permission repair  --  ADMIN helper (double-click me)
REM ----------------------------------------------------------------------------
REM  Why this file exists:
REM    Running "net start FrameServer" in a NON-administrator window fails with
REM    "System error 5 - Access denied", and in Windows PowerShell "sc" is an
REM    alias for Set-Content, so "sc query ..." does NOT query a service
REM    (it even creates a junk file). This helper avoids both traps:
REM      - it asks Windows for administrator rights (UAC) by itself
REM      - it always calls sc.exe (with .exe) instead of sc
REM ----------------------------------------------------------------------------
REM  What it can do (ONLY when you pick a menu item):
REM    [1] restart the Windows Camera Frame Server service
REM    [2] disable and re-enable the two physical cameras
REM        (= software unplug/replug, clears a stuck camera state)
REM    [4] run the camera capture test inside this administrator window
REM  It does NOT touch the registry, drivers, or any HP software.
REM  Nothing is deleted and no setting is changed permanently.
REM ----------------------------------------------------------------------------
REM  WHY THIS FILE IS PLAIN ASCII ONLY: cmd.exe parses a .bat file byte by byte
REM  in the active OEM code page; non-ASCII text here breaks parsing.
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1

REM ---- status-only mode: camera_repair_admin.bat /status (no UAC needed) ----
if /i "%~1"=="/status" goto status_only

REM ---- elevation check: High integrity SID = S-1-16-12288 ----
whoami /groups | findstr /C:"S-1-16-12288" >nul 2>&1
if errorlevel 1 (
  echo.
  echo  Administrator rights are required for these repairs.
  echo  Asking Windows for them now - please click "Yes" on the prompt.
  echo.
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

title Camera permission repair (ADMIN)
set "PROJECT_ROOT=%~dp0.."
set "PY=D:\31931\Documents\python312\python.exe"
set "PYTHONPATH=%PROJECT_ROOT%\_work\dsgraph"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%PROJECT_ROOT%"

:menu
cls
echo ============================================================
echo   Camera permission repair   [running as ADMINISTRATOR]
echo ============================================================
echo   Frame Server service state:
sc.exe query FrameServer | findstr /C:"STATE"
echo.
echo   Camera devices:
pnputil /enum-devices /class Camera | findstr /C:"Instance ID" /C:"Device Description" /C:"Status"
echo ============================================================
echo   [1] Restart the Camera Frame Server service
echo   [2] Disable + re-enable both physical cameras (software replug)
echo   [3] Do both 1 and 2
echo   [4] Run the camera capture test HERE (administrator token)
echo   [0] Exit
echo ============================================================
set "DO="
set /p "CH=Type 1, 2, 3, 4 or 0 then press Enter: "
if "%CH%"=="1" set "DO=SVC"
if "%CH%"=="2" set "DO=DEV"
if "%CH%"=="3" set "DO=BOTH"
if "%CH%"=="4" set "DO=TEST"
if "%CH%"=="0" set "DO=EXIT"
if not defined DO goto menu
if /i "%DO%"=="EXIT" goto done
if /i "%DO%"=="SVC"  call :do_svc
if /i "%DO%"=="DEV"  call :do_dev
if /i "%DO%"=="BOTH" call :do_svc
if /i "%DO%"=="BOTH" call :do_dev
if /i "%DO%"=="TEST" call :do_test
echo.
echo ------------------------------------------------------------
echo  Next: close this window, then double-click the camera status
echo        check .bat file inside the tools\ folder
echo  (that test must run WITHOUT administrator rights - the camera
echo   permission is evaluated differently for elevated processes)
echo ------------------------------------------------------------
pause
goto menu

:do_svc
echo.
echo --- [1] Restarting Windows Camera Frame Server ---
sc.exe stop FrameServer
timeout /t 2 >nul
sc.exe start FrameServer
echo   Result:
sc.exe query FrameServer | findstr /C:"STATE"
exit /b

:do_dev
echo.
echo --- [2] Disabling and re-enabling the cameras ---
echo   (this is a software unplug/replug; the screen may flicker briefly)
pnputil /disable-device "USB\VID_0BDA&PID_5883&MI_00\6&d574a2&0&0000"
timeout /t 2 >nul
pnputil /enable-device  "USB\VID_0BDA&PID_5883&MI_00\6&d574a2&0&0000"
echo.
pnputil /disable-device "USB\VID_0408&PID_548D&MI_00\6&37a6916b&0&0000"
timeout /t 2 >nul
pnputil /enable-device  "USB\VID_0408&PID_548D&MI_00\6&37a6916b&0&0000"
echo.
echo   Re-enabled. Current device list:
pnputil /enum-devices /class Camera | findstr /C:"Device Description" /C:"Status"
exit /b

:do_test
echo.
echo --- [4] Camera capture test (administrator token) ---
if not exist "%PY%" (
  echo   [ERROR] project python not found: %PY%
  exit /b
)
if not exist "tools\dshow_capture_test.py" (
  echo   [ERROR] tools\dshow_capture_test.py not found. Current folder: %CD%
  exit /b
)
"%PY%" tools\dshow_capture_test.py --index 1 --seconds 5
echo   (exit code %ERRORLEVEL%: 0 = frames received, 3 = E_ACCESSDENIED, 5 = no frames)
exit /b

:status_only
echo ============================================================
echo   Camera / Frame Server status (read only, no admin needed)
echo ============================================================
echo   Frame Server:
sc.exe query FrameServer | findstr /C:"STATE"
echo.
echo   Camera devices:
pnputil /enum-devices /class Camera | findstr /C:"Device Description" /C:"Status"
echo.
echo   HP OMEN camera related services:
sc.exe query HPOmenCap | findstr /C:"STATE"
sc.exe query HPOmenCustomCapDriver | findstr /C:"STATE"
echo ============================================================
exit /b 0

:done
endlocal
exit /b 0
