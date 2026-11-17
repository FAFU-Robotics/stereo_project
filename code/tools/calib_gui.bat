@echo off
REM ============================================================================
REM  Calibration workbench launcher  ->  tools\calib_gui.py
REM ----------------------------------------------------------------------------
REM  Four pages: capture / calibrate / check rectification / save parameters.
REM
REM  This launcher only STARTS the GUI. It installs nothing.
REM
REM  ASCII-only on purpose: cmd.exe parses .bat files byte by byte in the
REM  active OEM code page, so non-ASCII text here would break parsing.
REM
REM  Interpreter lookup order:
REM    1) the PY= line below, if that file exists
REM    2) `where python`
REM    3) common per-user installs
REM  If none of them is right, edit the PY= line at the top.
REM ============================================================================
setlocal EnableExtensions
chcp 65001 >nul 2>&1

set "PY=D:\31931\Documents\python312\python.exe"
if exist "%PY%" goto check

REM ---- 2) known-good per-user installs (checked before `where python`, which
REM         may hit a stripped interpreter earlier on PATH) -----------------
for %%P in (
    "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
) do (
    if exist %%P (
        set "PY=%%~P"
        goto check
    )
)

REM ---- 3) whatever `python` is on PATH --------------------------------
for /f "delims=" %%I in ('where python 2^>nul') do (
    set "PY=%%I"
    goto check
)

echo [ERROR] Python not found. Edit the PY= line at the top of this file.
pause
exit /b 1

:check
echo Python : %PY%
echo.

REM ---- hard requirements: without these the GUI cannot even start ----------
set "MISS="
for %%M in (tkinter cv2 numpy yaml) do (
    "%PY%" -c "import %%M" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] missing python module: %%M
        set "MISS=1"
    )
)
if defined MISS (
    echo.
    echo         Install the missing packages with:
    echo         "%PY%" -m pip install opencv-python numpy pyyaml
    echo.
    echo         tkinter cannot be pip-installed. If it is the missing one,
    echo         use the official CPython from python.org with tcl/tk checked.
    pause
    exit /b 1
)

REM ---- soft requirement: pillow makes image drawing faster, not mandatory --
"%PY%" -c "import PIL" >nul 2>&1
if errorlevel 1 (
    echo [NOTE] pillow not found - using the built-in Tk PNG decoder instead.
    echo        The GUI works; only the live preview repaint is a bit slower.
    echo        To remove this note: "%PY%" -m pip install pillow
    echo.
)

echo Starting calibration GUI ...
"%PY%" "%~dp0calib_gui.py" %*
if errorlevel 1 pause
