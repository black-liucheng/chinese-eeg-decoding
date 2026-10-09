@echo off
REM ===================================================================
REM  Chinese EEG Classification System -- full pipeline, no GUI
REM
REM  gen-data -> SVM (+ permutation test) -> CNN -> compare -> HTML report
REM
REM  Everything lands under data\ :
REM    data\processed\   preprocessed datasets (.npz)
REM    data\figures\     figures (.png)
REM    data\reports\     single-file HTML report
REM
REM  Takes roughly 20-30 minutes on a laptop CPU.  Be patient.
REM  ASCII-only on purpose -- see the note in run.bat.
REM ===================================================================
setlocal
cd /d "%~dp0"

set "PYTHONPATH=%~dp0src"
set "KMP_DUPLICATE_LIB_OK=TRUE"
REM without -u, stdout is block-buffered when redirected and progress
REM looks frozen for minutes at a time
set "PYTHONUNBUFFERED=1"

call "%~dp0scripts\_find_python.bat"
if errorlevel 1 (
    echo   [ERROR] No usable Python found.  See README.md.
    pause
    exit /b 1
)

echo   Interpreter: %PY%
echo.
echo ====================================================================
echo   Running the full pipeline.  Results also land in data\reports\.
echo ====================================================================
echo.

"%PY%" -u -m cneeg gen-data
if errorlevel 1 goto :failed

"%PY%" -u -m cneeg train --model svm --permutation
if errorlevel 1 goto :failed

"%PY%" -u -m cneeg train --model cnn
if errorlevel 1 goto :failed

"%PY%" -u -m cneeg compare
if errorlevel 1 goto :failed

echo.
echo ====================================================================
echo   Done.  Open the HTML report under data\reports\ in a browser.
echo ====================================================================
echo.
pause
endlocal
exit /b 0

:failed
echo.
echo   [ERROR] The pipeline failed at the step above.
echo   Please send the messages printed above to the project team.
echo.
pause
endlocal
exit /b 1
