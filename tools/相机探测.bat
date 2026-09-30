@echo off
REM ============================================================================
REM  Camera probe launcher  (READ-ONLY: it only opens cameras, reads frames and
REM  saves snapshots. It does not change drivers, registry or camera settings.)
REM ----------------------------------------------------------------------------
REM  WHY THIS FILE IS PLAIN ASCII ONLY:
REM    cmd.exe parses a .bat file byte by byte using the active OEM code page.
REM    The previous version of this file contained Chinese text (29% of bytes)
REM    and LF-only line endings. cmd mis-parsed those bytes and reported errors
REM    such as:
REM        'REM' is not recognized as an internal or external command
REM        'HONIOENCODING' is not recognized ...
REM        'backends' is not recognized ...
REM        'ots\' is not recognized ...
REM    Fix: ASCII-only content + CRLF line endings.
REM    DO NOT put non-ASCII characters into this file again.
REM    (The Python programs still print Chinese; that text is produced by
REM     python.exe and is never parsed by cmd, so it is safe.)
REM ============================================================================

setlocal
chcp 65001 >nul 2>&1

REM --- Python interpreter that THIS project uses (do not use the PATH default,
REM     which is Python 3.14 and cannot run this project) ---
set "PY=D:\31931\Documents\python312\python.exe"

REM --- Project root = parent folder of this .bat (this file lives in tools\) ---
set "PROJECT_ROOT=%~dp0.."

REM --- Writable temp dir (keeps pip/python temp files inside the project) ---
set "TEMP=%PROJECT_ROOT%\_work\pip_tmp"
set "TMP=%TEMP%"

REM --- Make Python speak UTF-8 so its Chinese messages render on chcp 65001 ---
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

if not exist "tools\probe_cameras.py" (
  echo [ERROR] tools\probe_cameras.py not found. Current folder:
  echo         %CD%
  pause
  exit /b 1
)

REM --- Default probe arguments. Any arguments you pass to this .bat replace them,
REM     e.g.  this_file.bat --index 2 --backends dshow --no-stage-b ---
set "PROBE_ARGS=--max-index 6 --backends dshow,msmf --timeout 20 --warmup 8 --measure 15"
if not "%~1"=="" set "PROBE_ARGS=%*"

echo ============================================================
echo  Camera probe (read-only)
echo ------------------------------------------------------------
echo  Project   : %CD%
echo  Python    : %PY%
echo  Arguments : %PROBE_ARGS%
echo ------------------------------------------------------------
echo  Preflight check (which python / which opencv):
"%PY%" -c "import sys,cv2,numpy;print('   python  :',sys.executable);print('   opencv  :',cv2.__version__);print('   numpy   :',numpy.__version__)"
if errorlevel 1 (
  echo [ERROR] python preflight check failed. See the message above.
  pause
  exit /b 2
)
echo ------------------------------------------------------------
echo  Scanning camera indexes. This can take 1-3 minutes.
echo  Do not close this window.
echo ============================================================
echo.

"%PY%" tools\probe_cameras.py %PROBE_ARGS%
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Probe finished.
echo  Exit code : %RC%
echo  Report    : output\probe\report.json
echo  Snapshots : output\probe\snapshots\
echo ------------------------------------------------------------
echo  Copy the text above (or take a screenshot) and send it back.
echo ============================================================

pause
endlocal
