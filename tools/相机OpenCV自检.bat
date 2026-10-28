@echo off
REM ============================================================================
REM  OpenCV camera check  --  run with NORMAL (non-elevated) permissions
REM ----------------------------------------------------------------------------
REM  Double-click this file to test the camera WITHOUT administrator rights.
REM  Then run the same check WITH administrator rights (launcher menu [10])
REM  and compare the two outputs:
REM      the "admin rights" line differs, everything else is comparable.
REM ----------------------------------------------------------------------------
REM  ASCII-only on purpose: cmd.exe parses .bat files byte by byte in the active
REM  OEM code page, so non-ASCII text here would break parsing.
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1

set "PROJECT_ROOT=%~dp0.."
call "%~dp0find_python.bat"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "TEMP=%PROJECT_ROOT%\_work\pip_tmp"
set "TMP=%TEMP%"
cd /d "%PROJECT_ROOT%"
if not exist "%TEMP%" mkdir "%TEMP%" >nul 2>&1

echo ============================================================
echo   OpenCV camera check  [NORMAL permissions, not elevated]
echo ------------------------------------------------------------
echo   Python : %PY%
echo   Script : tools\opencv_camera_check.py
echo ------------------------------------------------------------
echo   After this, run the ELEVATED version for comparison:
echo     double-click the ADMIN launcher bat in this tools folder, choose [10]
echo ============================================================
echo.

if not exist "%PY%" (
  echo [ERROR] project python not found: %PY%
  pause
  exit /b 1
)

"%PY%" "tools\opencv_camera_check.py" %*
echo.
echo ------------------------------------------------------------
echo   Exit code: %ERRORLEVEL%
echo   (0 = real frames captured, 3 = opens but no valid image, 5 = cannot open)
echo ------------------------------------------------------------
pause
endlocal
