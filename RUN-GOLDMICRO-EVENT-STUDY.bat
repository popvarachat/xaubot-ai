@echo off
setlocal
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\xaubot-ai\goldmicro-research-py311\Scripts\python.exe"

echo ============================================================
echo GOLDmicro Event Target V2 Study
echo - 24 predictive configs x 5 chronological probes = 120 jobs
echo - target: causal SMC setup reaches TP before SL within 32 M15 bars
echo - same-bar TP+SL ambiguity is adverse / SL first
echo - raw-bar train/OOS embargo; no filtered-row gap shortcut
echo - hard AUC gate remains 0.55 on at least 4/5 probes
echo - no live promotion / no active model overwrite / no order_send
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

"%PY%" scripts\run_goldmicro_event_target_study.py --limit 24 --samples 5 --sample-stride-bars 1000 --top-k 6 --min-test-auc 0.55 --max-gap 0.12 --min-test-events 100 --min-pass-rate 0.80
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" (
  echo Event target study stopped with exit code %EXITCODE%.
  echo Review the error above; live model was not changed.
) else (
  echo Event target study completed.
  echo Review event_target_report.json and event_strategy_oos_queue.json.
)

echo.
pause
exit /b %EXITCODE%
