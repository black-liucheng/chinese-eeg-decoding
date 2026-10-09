@echo off
REM ===================================================================
REM  Locate a Python interpreter that can ACTUALLY run this project.
REM  Result is returned in %PY%.  errorlevel=1 means "not found".
REM
REM  Why not just "where python":
REM  this machine has both anaconda 3.13 (verified working) and a newer
REM  interpreter elsewhere.  The newer one has no PyTorch wheels yet, so
REM  picking it makes the one-click launcher fail with an import error
REM  that looks like a bug in our code.  So every candidate is really
REM  imported before it is accepted.
REM
REM  NOTE: this file is deliberately ASCII-only.  cmd.exe parses .bat in
REM  the OEM codepage (936 on Chinese Windows); UTF-8 Chinese comments
REM  get mis-decoded into stray '&' / '|' characters and break parsing.
REM  All Chinese user-facing text lives in the Python side instead.
REM
REM  Callers: run.bat, run_cli.bat
REM ===================================================================
setlocal enabledelayedexpansion
set "PY="

REM -- candidate 1: the anaconda base env verified on this machine -----
if exist "D:\anaconda\python.exe" (
    "D:\anaconda\python.exe" -c "import numpy, sklearn" >nul 2>&1
    if !errorlevel! equ 0 set "PY=D:\anaconda\python.exe"
)

REM -- candidate 2: whatever is first on PATH --------------------------
if not defined PY (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PY (
            "%%i" -c "import numpy, sklearn" >nul 2>&1
            if !errorlevel! equ 0 set "PY=%%i"
        )
    )
)

REM -- candidate 3: the active conda environment -----------------------
if not defined PY if defined CONDA_PREFIX (
    if exist "%CONDA_PREFIX%\python.exe" (
        "%CONDA_PREFIX%\python.exe" -c "import numpy, sklearn" >nul 2>&1
        if !errorlevel! equ 0 set "PY=%CONDA_PREFIX%\python.exe"
    )
)

endlocal & set "PY=%PY%"
if not defined PY exit /b 1
exit /b 0
