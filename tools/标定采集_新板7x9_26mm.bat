@echo off
REM ============================================================================
REM  Calibration capture with the NEW A4 ChArUco board
REM ----------------------------------------------------------------------------
REM  Board : 7 x 9 squares, 26 mm each  (pattern 182 x 234 mm)
REM  Inner : 6 x 8 = 48 chessboard corners  <- what the calibrator uses
REM  Dist  : 30 - 45 cm  (closer than ~24 cm the board leaves the frame)
REM
REM  WHY THIS SEPARATE .bat:
REM    tools\capture_calib.py takes the board geometry from COMMAND LINE
REM    (defaults 12x9 / 3.0 mm = the old DFOPTIX board), NOT from config.yaml.
REM    The old board's 3.0 mm was a nominal value and is the suspected cause of
REM    a ~1.6-2% depth scale error, so we re-capture with this bigger board.
REM    This launcher only PASSES the new numbers - no existing code was changed.
REM
REM  >>> IMPORTANT: MEASURE THE PRINTED BOARD FIRST <<<
REM    Print at 100% (no "fit to page"), then measure 5 squares across
REM    (outer edge to outer edge) with a caliper. Nominal = 5 x 26 = 130.0 mm.
REM    If your print measures e.g. 128.5 mm, set SQ=25.7 below (128.5 / 5).
REM    Using the nominal value when the print is off re-introduces the very
REM    scale error we are trying to remove.
REM
REM  ASCII-only on purpose: cmd.exe parses .bat files byte by byte in the
REM  active OEM code page, so non-ASCII text here would break parsing.
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1

REM ---- printed square size in mm, as MEASURED (edit this single number) ----
set "SQ=26.0"

REM ---- elevation check: High integrity SID = S-1-16-12288 ----
whoami /groups | findstr /C:"S-1-16-12288" >nul 2>&1
if errorlevel 1 (
  echo.
  echo  This capture needs administrator rights ^(the physical camera needs elevation^).
  echo  Asking Windows for them now - please click "Yes" on the prompt.
  echo.
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

title calibration capture - new board 7x9 / 26 mm (ADMIN)
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
echo   Calibration capture   [running as ADMINISTRATOR]
echo ------------------------------------------------------------
echo   Board   : 7 x 9 squares, %SQ% mm each
echo   Inner   : 6 x 8 = 48 corners
echo   Distance: 30 - 45 cm
echo   Keys    : SPACE = record   u = undo   f = force   q = quit
echo ------------------------------------------------------------
echo   Checklist is shown live in the window:
echo     centre x2, four corners x2 each, 4 tilts (^>=20 deg) x3 each
echo   Keep the board VERTICAL and fully inside the frame.
echo ============================================================
echo.

if not exist "%PY%" (
  echo [ERROR] project python not found: %PY%
  pause
  exit /b 1
)

"%PY%" "tools\capture_calib.py" --capture --board-cols 7 --board-rows 9 --square-mm %SQ%

echo.
echo ------------------------------------------------------------
echo   Exit code: %ERRORLEVEL%
echo   Results were written to: output\calib\^<timestamp^>\
echo   Next step: run tools\run_calibration.py  (it reads the board from config.yaml)
echo ------------------------------------------------------------
pause
endlocal
