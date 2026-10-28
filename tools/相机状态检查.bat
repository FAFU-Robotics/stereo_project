@echo off
REM ============================================================================
REM  Camera status check + DirectShow capture test (one click)
REM ----------------------------------------------------------------------------
REM  What it does (all READ-ONLY, it changes nothing):
REM    1) shows whether the Windows Camera Frame Server service is running
REM    2) shows whether HP OMEN camera helper processes are running
REM    3) runs a DirectShow capture test on camera index 1 (= the DECXIN stereo
REM       camera on this machine) and prints the real COM HRESULT if it fails
REM ----------------------------------------------------------------------------
REM  WHY THIS FILE IS PLAIN ASCII ONLY:
REM    cmd.exe parses a .bat file byte by byte in the active OEM code page;
REM    non-ASCII text here breaks parsing (see the camera probe .bat for details).
REM    Keep it ASCII-only with CRLF line endings.
REM ============================================================================

setlocal
chcp 65001 >nul 2>&1

call "%~dp0find_python.bat"
set "PROJECT_ROOT=%~dp0.."
set "PYTHONPATH=%PROJECT_ROOT%\_work\dsgraph"
set "TEMP=%PROJECT_ROOT%\_work\pip_tmp"
set "TMP=%TEMP%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT_ROOT%"

echo ============================================================
echo  [1/3] Windows Camera Frame Server service
echo ------------------------------------------------------------
sc query FrameServer | findstr /i "SERVICE_NAME STATE"
sc query FrameServerMonitor | findstr /i "STATE"
echo    (STATE should be RUNNING. If it is STOPPED, physical cameras are
echo     denied: run an ADMIN command prompt and use:  net start FrameServer
echo     or simply reboot the laptop.)

echo.
echo ============================================================
echo  [2/3] HP OMEN camera related processes
echo ------------------------------------------------------------
tasklist /fi "imagename eq HP.OMEN.VirtualCamHelper.exe" 2>nul | findstr /i "HP.OMEN"
tasklist /fi "imagename eq HP.OMEN.VadRedirectHelper.exe" 2>nul | findstr /i "HP.OMEN"
tasklist /fi "imagename eq OmenCap.exe" 2>nul | findstr /i "OmenCap"
echo    (If HP.OMEN.* are listed, close OMEN Gaming Hub and retry.)

echo.
echo ============================================================
echo  [3/3] DirectShow capture test on camera index 1 (DECXIN)
echo ============================================================
if not exist "%PY%" (
  echo [ERROR] project python not found: %PY%
  pause
  exit /b 1
)
if not exist "tools\dshow_capture_test.py" (
  echo [ERROR] tools\dshow_capture_test.py not found. Current folder: %CD%
  pause
  exit /b 1
)
"%PY%" tools\dshow_capture_test.py --index 1 --seconds 5
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Exit code = %RC%
echo    0 = frames were received (check the printed statistics)
echo    3 = E_ACCESSDENIED: physical camera access is denied
echo    5 = graph built but no frames arrived
echo ------------------------------------------------------------
echo  If you just changed something (reboot / service / unplug),
echo  run this file again and compare.
echo ============================================================

pause
endlocal
