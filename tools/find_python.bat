@echo off
REM ============================================================================
REM  find_python.bat -- locate a usable Python and set PY **in the caller**.
REM
REM  Usage (from another .bat in this folder):
REM      call "%~dp0find_python.bat"
REM      if not exist "%PY%" ( ... your own error handling ... )
REM
REM  Deliberately NO `setlocal` here: the caller must be able to see PY.
REM  ASCII-only on purpose -- cmd.exe parses .bat bytes in the OEM code page.
REM ============================================================================
set "PY="

REM 1) per-user installs. "pythoncore-*" is what the python.org
REM    "install for me only" layout produces on recent builds.
for %%P in (
    "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
    "%LOCALAPPDATA%\Python\pythoncore-3.13-64\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
) do (
    if not defined PY if exist %%P set "PY=%%~P"
)

REM 2) all-users install
if not defined PY for %%P in (
    "%ProgramFiles%\Python314\python.exe"
    "%ProgramFiles%\Python313\python.exe"
    "%ProgramFiles%\Python312\python.exe"
) do (
    if not defined PY if exist %%P set "PY=%%~P"
)

REM 3) last resort: `python` on PATH (may be a stripped interpreter
REM    without torch -- the caller is expected to check for what it needs)
if not defined PY for /f "delims=" %%I in ('where python 2^>nul') do (
    if not defined PY set "PY=%%I"
)

if not defined PY (
    echo [ERROR] No Python interpreter found.
    echo         Install Python 3.14 from python.org, or edit this file
    echo         to point at the interpreter you use.
    exit /b 1
)
exit /b 0
