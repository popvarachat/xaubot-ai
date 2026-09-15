@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo GOLDmicro 24 x 5 Chronological Research Launcher
echo - isolated Python 3.11 environment
echo - 24 challenger configurations x 5 time samples = 120 jobs
echo - one command / one research batch
echo - no live promotion / no active model overwrite
echo ============================================================

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_goldmicro_research_env.ps1" -Limit 24 -Samples 5 -SampleStrideBars 1000 -TopK 6
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" (
  echo GOLDmicro research stopped with exit code %EXITCODE%.
  echo Review the error above; live model was not changed.
) else (
  echo GOLDmicro 24 x 5 research matrix completed.
)

echo.
pause
exit /b %EXITCODE%
