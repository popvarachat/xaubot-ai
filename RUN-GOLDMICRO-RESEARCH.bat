@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo GOLDmicro Multi-Challenger Research Launcher
echo - isolated Python 3.11 environment
echo - 24 challenger models by default
echo - no live promotion / no active model overwrite
echo ============================================================

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_goldmicro_research_env.ps1" -Limit 24 -TopK 6
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" (
  echo GOLDmicro research stopped with exit code %EXITCODE%.
  echo Review the error above; live model was not changed.
) else (
  echo GOLDmicro research batch completed.
)

echo.
pause
exit /b %EXITCODE%
