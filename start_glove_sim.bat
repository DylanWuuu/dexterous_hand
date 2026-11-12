@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if exist "%~dp0tools\Library\bin\micromamba.exe" if exist "%~dp0conda_env\python.exe" goto bundled
if defined CONDA_PREFIX if /i not "%CONDA_DEFAULT_ENV%"=="base" goto activated
where conda >nul 2>nul
if errorlevel 1 (
    echo Open Miniforge Prompt and activate the environment described in README.md.
    goto failed
)
call conda run --no-capture-output -n dexterous-hand python "%~dp0glove_sim.py" %*
if errorlevel 1 goto failed
exit /b 0

:activated
python "%~dp0glove_sim.py" %*
if errorlevel 1 goto failed
exit /b 0

:bundled
"%~dp0tools\Library\bin\micromamba.exe" --root-prefix "%~dp0tools\mamba" run -p "%~dp0conda_env" python "%~dp0glove_sim.py" %*
if errorlevel 1 goto failed
exit /b 0

:failed
pause
exit /b 1
