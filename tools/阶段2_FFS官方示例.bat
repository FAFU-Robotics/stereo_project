@echo off
REM ============================================================================
REM  Stage 2 - Fast-FoundationStereo official demo verification
REM ----------------------------------------------------------------------------
REM  Runs the official demo images through the FFS model. No camera is needed.
REM
REM  Prerequisite:  weights\23-36-37\  must contain BOTH files
REM      model_best_bp2_serialize.pth
REM      cfg.yaml
REM
REM  WHY THIS FILE IS PLAIN ASCII ONLY:
REM    cmd.exe parses a .bat file byte by byte using the active OEM code page.
REM    Non-ASCII text in a .bat file makes cmd mis-parse following commands
REM    (for example "'REM' is not recognized ..."). See the camera probe .bat
REM    in this same tools\ folder for the full explanation.
REM    Keep this file ASCII-only with CRLF line endings.
REM ============================================================================

setlocal
chcp 65001 >nul 2>&1

REM --- Python used by THIS project (the PATH default is Python 3.14) ---
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

REM --- Windows PyTorch ships without Triton, so torch.compile must be disabled,
REM     otherwise the @torch.compile decorators inside FFS fail. ---
set "TORCHDYNAMO_DISABLE=1"

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

if not exist "tools\run_ffs_demo.py" (
  echo [ERROR] tools\run_ffs_demo.py not found. Current folder:
  echo         %CD%
  pause
  exit /b 1
)

echo ============================================================
echo  Step 1/2 : check model weights and environment
echo ============================================================
"%PY%" tools\run_ffs_demo.py --check-only
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo ============================================================
  echo  [ERROR] weight/environment check failed. Exit code = %RC%
  echo ------------------------------------------------------------
  echo  The weights are not in place yet. Two ways to get them:
  echo    1^) run:  "%PY%" tools\fetch_ffs_weights.py
  echo    2^) download the official research checkpoint in a browser and put
  echo       BOTH files into:  %CD%\weights\23-36-37\
  echo ============================================================
  pause
  exit /b %RC%
)

echo.
echo ============================================================
echo  Step 2/2 : run inference on the official demo images
echo ------------------------------------------------------------
echo  Default arguments: --repeat 3
echo  You can pass your own arguments to this .bat, for example:
echo      this_file.bat --repeat 5 --scale 0.5 --no-display
echo ============================================================
set "DEMO_ARGS=--repeat 3"
if not "%~1"=="" set "DEMO_ARGS=%*"

"%PY%" tools\run_ffs_demo.py %DEMO_ARGS%
set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Finished. Exit code = %RC%
echo  Results in : %CD%\output\ffs_demo\
echo ============================================================

pause
endlocal
