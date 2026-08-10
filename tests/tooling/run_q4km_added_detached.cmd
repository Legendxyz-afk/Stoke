@echo off
rem RUN ONE ADDED-SAMPLE LEG for a band that cannot freeze from the fixed grid.
rem
rem ▶ DETACHED for the same reason every other multi-hour leg in this tree is:
rem a leg is ~70 min against a 10 min agent-tool ceiling, and Task Scheduler is
rem the only mechanism here measured to survive one.
rem
rem ▶ THESE ARE NOT GRID LEGS. Ruling section 5 fixes the grid; its tripwire was
rem calibrated against that grid's covered positions. Added samples feed a TAU
rem FREEZE only -- see q4km_added.py's header for the full contract. They bank
rem as {pid}-added.json, which no grid consumer matches.
rem
rem ▶ PASS `EXTRA` AS THE 2nd ARGUMENT for every instance after the first, and
rem only after verifying BY COMMAND LINE that a ref_forward.py process is really
rem running. The guard reads free RAM at launch, so staggering is mandatory:
rem two launches racing each other both see the memory neither has taken yet.
rem
rem ▶ EXACTLY TWO QUOTES ON THE TASK SCHEDULER COMMAND LINE. `cmd /c` preserves
rem quotes only when the whole remainder holds exactly two; with four it strips
rem the first and the last and silently runs a DIFFERENT command line -- exit 1,
rem no log, no process. That defect cost a launch on 2026-08-10. The prompt id
rem and EXTRA need no quoting, so keep it that way.
rem
rem USAGE:  run_q4km_added_detached.cmd <pid> [EXTRA]
cd /d "%~dp0..\.."
set PID=%~1
set EXTRA=%~2
if "%PID%"=="" (
  echo usage: run_q4km_added_detached.cmd ^<pid^> [EXTRA]
  exit /b 2
)

set OUT=tests\tooling\.gate_cache\q4km_grid
if not exist "%OUT%" mkdir "%OUT%"

set EXTRAARG=
if /I "%EXTRA%"=="EXTRA" set EXTRAARG=--additional-instance

rem One log per prompt: the grid's shared-log lesson applies here too, since
rem added legs are expected to run concurrently with each other.
set LOG=%OUT%\added-%PID%.log

echo. >> "%LOG%"
echo ===== Q4KM ADDED %PID% %DATE% %TIME% %EXTRAARG% ===== >> "%LOG%"
python -u tests\tooling\q4km_added.py --prompt %PID% --out "%OUT%" %EXTRAARG% >> "%LOG%" 2>&1
echo ===== Q4KM ADDED %PID% EXIT %ERRORLEVEL% %DATE% %TIME% ===== >> "%LOG%"
exit /b 0
