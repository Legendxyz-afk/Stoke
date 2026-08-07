@echo off
rem RUN [2v2-iii]'s Q4_K_M REFERENCE GRID -- ruling section 5, the last evidence
rem the M3 stamp is waiting on.
rem
rem ▶ DETACHED because a window is ~3 h against a 10 min agent-tool ceiling.
rem Task Scheduler is this tree's established pattern for that and is the only
rem one measured to survive; every prior multi-hour gate leg used it.
rem
rem ▶ THE TASK MUST BE ONE-TIME AND MUST BE DELETED WHEN THE GRID CLOSES.
rem 2026-08-02 found `stoke-m2-ref-gate` still carrying an ENABLED AtLogOn
rem trigger, relaunching a finished gate one minute after every sign-in, and a
rem watchdog popping a console every 15 minutes long after its job was done.
rem A spent helper outliving its job is a defect, not a leftover.
rem
rem ▶ DO NOT RUN THIS WHILE ANYTHING HOLDS THE Q8_0 FILE, and do not run two
rem windows at once. ref_forward.py on Q4_K_M was measured 2026-08-07 to take
rem ~4.5 GB of ullAvailPhys (2.33 GB model page cache + ~2.1 GB numpy working
rem set), which is why section 5's launch condition is model + 2 GB. Two
rem concurrent windows do not fit this box and would reproduce the recorded
rem page-cache collapse (cache faults ~60 -> ~7,000, ~3x slower) -- which is
rem exactly what takes a run past the <=4 h it is authorized under.
rem
rem ▶ RESUME IS THE OUTPUT DIRECTORY'S. A leg is done iff its probe file exists
rem AND revalidates. Re-running this launcher after a stop continues from the
rem first leg that does not, and spends nothing on the ones that do.
rem
rem USAGE:  run_q4km_grid_detached.cmd A|B|CONT  <rate_s_per_pos>
cd /d "%~dp0..\.."
set WHICH=%~1
set RATE=%~2
if "%WHICH%"=="" set WHICH=A
if "%RATE%"=="" set RATE=18.42

set OUT=tests\tooling\.gate_cache\q4km_grid
set LOG=%OUT%\window.log
if not exist "%OUT%" mkdir "%OUT%"

echo. >> "%LOG%"
echo ===== Q4KM GRID %WHICH% %DATE% %TIME% (rate %RATE% s/pos) ===== >> "%LOG%"

if /I "%WHICH%"=="CONT" (
  python -u tests\tooling\q4km_grid.py --continuations --out "%OUT%" --rate-s-per-pos %RATE% >> "%LOG%" 2>&1
) else (
  python -u tests\tooling\q4km_grid.py --run %WHICH% --out "%OUT%" --rate-s-per-pos %RATE% >> "%LOG%" 2>&1
)

echo ===== Q4KM GRID %WHICH% EXIT %ERRORLEVEL% %DATE% %TIME% ===== >> "%LOG%"
exit /b 0
