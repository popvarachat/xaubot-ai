@echo off
setlocal
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\xaubot-ai\goldmicro-research-py311\Scripts\python.exe"

echo ============================================================
echo GOLDmicro Target Alignment Study
echo - 24 predictive configs x 5 chronological samples
echo - target horizons: 1 / 4 / 8 / 16 M15 bars
echo - 480 causal training/evaluation jobs in ONE run
echo - hard AUC gate remains 0.55; no auto-relaxation
echo - strategy PF/DD/cost runs only if a horizon clears the gate
echo - no live promotion / no active model overwrite
echo ============================================================

if not exist "%PY%" (
  echo ERROR: isolated Python 3.11 research environment not found:
  echo   %PY%
  echo Run RUN-GOLDMICRO-RESEARCH.bat once to create the environment.
  pause
  exit /b 2
)

git pull --ff-only
if errorlevel 1 (
  echo ERROR: git pull --ff-only failed.
  pause
  exit /b 3
)

"%PY%" scripts\goldmicro_research_preflight.py
if errorlevel 1 (
  echo ERROR: dependency preflight failed.
  pause
  exit /b 4
)

"%PY%" scripts\run_goldmicro_target_alignment_study.py --limit 24 --samples 5 --sample-stride-bars 1000 --horizons 1,4,8,16 --top-k 6
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" (
  echo Target alignment study stopped with exit code %EXITCODE%.
  echo Review the error above; live model was not changed.
) else (
  echo Target alignment study completed.
  echo Review target_alignment_report.json and strategy_oos_report.json.
)

echo.
pause
exit /b %EXITCODE%
