@echo off
REM ============================================================================
REM  Single camera preview launcher  (READ-ONLY: opens one camera index and
REM  shows a live window. It does not change any camera or system setting.)
REM ----------------------------------------------------------------------------
REM  WHY THIS FILE IS PLAIN ASCII ONLY:
REM    cmd.exe parses a .bat file byte by byte using the active OEM code page.
REM    A UTF-8 file containing Chinese text, with LF-only line endings, is
REM    mis-parsed by cmd and fails with errors such as:
REM        'REM' is not recognized as an internal or external command
REM        'on:' is not recognized ...
REM    Keep this file ASCII-only with CRLF line endings.
REM    (Python still prints Chinese; python.exe produces that text and cmd never
REM     parses it, so it is safe.)
REM ============================================================================

setlocal
chcp 65001 >nul 2>&1

REM --- Python used by THIS project (the PATH default is Python 3.14 and cannot
REM     run this project) ---
set "PY=D:\31931\Documents\python312\python.exe"

REM --- Project root = parent folder of this .bat (this file lives in tools\) ---
set "PROJECT_ROOT=%~dp0.."

REM --- Writable temp dir, kept inside the project ---
set "TEMP=%PROJECT_ROOT%\_work\pip_tmp"
set "TMP=%TEMP%"

REM --- Let Python print UTF-8 so its Chinese messages render after chcp 65001 ---
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "KMP_DUPLICATE_LIB_OK=TRUE"

cd /d "%PROJECT_ROOT%"
if errorlevel 1 (
  echo [ERROR] cannot enter project root: %PROJECT_ROOT%
  pause
  exit /b 1
)

if not exist "%TEMP%" mkdir "%TEMP%" >nul 2>&1

if not exist "%PY%" (
  echo [ERROR] project python not found:
  echo         %PY%
  pause
  exit /b 1
)

if not exist "tools\preview_any.py" (
  echo [ERROR] tools\preview_any.py not found. Current folder:
  echo         %CD%
  pause
  exit /b 1
)

REM --- Camera index: first argument, otherwise ask. Default is 2 because that
REM     is the only index that opens on this machine so far. ---
set "IDX=%~1"
if "%IDX%"=="" set /p "IDX=Enter camera index 0-5, then press Enter [default 2]: "
if "%IDX%"=="" set "IDX=2"

echo ============================================================
echo  Single camera preview
echo ------------------------------------------------------------
echo  Project root : %CD%
echo  Python       : %PY%
echo  Camera index : %IDX%
echo ------------------------------------------------------------
echo  Window keys  : q or ESC = quit, s = save one frame, i = status
echo.
echo  How to read the numbers printed below:
echo    mean near 0 and std near 0  -^> driver gives no real image data
echo    normal indoor picture       -^> mean about 20..200, std above 10
echo ============================================================
echo.

"%PY%" tools\preview_any.py --index %IDX%
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Preview closed. Exit code = %RC%
echo ============================================================

pause
endlocal
