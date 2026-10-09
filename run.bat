@echo off
REM ===================================================================
REM  Chinese EEG Classification System -- one-click launcher (GUI)
REM
REM  Just double-click this file.  If Python or a dependency is missing
REM  you get a readable message instead of a window that flashes away.
REM
REM  ASCII-only on purpose: cmd.exe parses .bat in the OEM codepage, and
REM  UTF-8 Chinese text here would be mis-decoded into stray control
REM  characters.  The Chinese messages come from Python, which talks to
REM  the Windows console directly and has no such problem.
REM ===================================================================
setlocal
cd /d "%~dp0"

REM the package uses a src/ layout and is not pip-installed, so src/ has
REM to be on the module search path
set "PYTHONPATH=%~dp0src"
REM anaconda's MKL and pip's PyTorch ship conflicting OpenMP runtimes;
REM without this, "import torch" aborts with "OMP: Error #15"
set "KMP_DUPLICATE_LIB_OK=TRUE"

call "%~dp0scripts\_find_python.bat"
if errorlevel 1 (
    echo.
    echo   [ERROR] No usable Python found.
    echo   Install Python 3.13 ^(64-bit^), then run:
    echo       pip install -r requirements.txt
    echo       pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo   See README.md for details.
    echo.
    pause
    exit /b 1
)

echo   Interpreter: %PY%
"%PY%" -m cneeg gui
if errorlevel 1 (
    echo.
    echo   [ERROR] Could not start the GUI.  Usual causes:
    echo     1. PyQt5 missing - pip install PyQt5
    echo     2. torch missing - pip install torch --index-url https://download.pytorch.org/whl/cpu
    echo     3. mne missing   - pip install mne
    echo   Full install commands are in README.md.
    echo.
    pause
    exit /b 1
)

endlocal
