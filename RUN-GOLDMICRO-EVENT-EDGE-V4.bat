@echo off
setlocal
cd /d %~dp0

set "PY=%LOCALAPPDATA%\xaubot-ai\goldmicro-research-py311\Scripts\python.exe"
if not exist "%PY%" (
  echo Python venv not found: %PY%
  pause
  exit /b 1
)

echo Running GOLDmicro Event Economic-Edge V4 research study...
"%PY%" scripts\run_goldmicro_event_edge_v4_study.py
set RC=%ERRORLEVEL%

if not "%RC%"=="0" (
  echo.
  echo Event Edge V4 study FAILED with exit code %RC%.
  pause
  exit /b %RC%
)

echo.
echo Event Edge V4 study completed.
echo Review event_edge_report.json and event_edge_strategy_oos_queue.json.
echo.
pause
exit /b 0
