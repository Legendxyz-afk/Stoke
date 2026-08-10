@echo off
rem FREEZE tau_Q4KM, or BUILD the divergence records -- detached.
rem
rem ▶ DETACHED BECAUSE A BACKGROUND SHELL WAS NOT ENOUGH. The freeze is ~60
rem `stoke --probe-pos` invocations at ~13 s each, about 13 minutes. Run under
rem the agent's background-shell mechanism it was killed after ~5 minutes, part
rem way through the second band, having written nothing -- the envelope and the
rem tau table are both written at the end. Task Scheduler is the one mechanism
rem in this tree measured to survive a long job, and every multi-hour gate leg
rem has used it.
rem
rem ▶ SEPARATE THE ADDED PROMPTS WITH `+`, NEVER A COMMA. cmd treats , and ; as
rem ARGUMENT SEPARATORS, so `freeze p32,p22` arrives as %1=freeze %2=p32 %3=p22
rem and the second prompt is silently dropped from the envelope -- a quiet
rem wrong answer, not a crash. `+` is converted to `,` below, which is the same
rem convention run_q4km_grid_detached.cmd already uses for --only and for the
rem same family of reason.
rem
rem ▶ FREEZE AND RECORDS ARE DIFFERENT MODES because q4km_tau.py refuses to do
rem both in one invocation: a tool that can set the threshold AND assemble the
rem records it will be applied to can always be run in the order that passes.
rem Run `freeze` first -- section 2.3 requires tau frozen before any grading.
rem
rem USAGE:  run_q4km_tau_detached.cmd freeze|records <p32+p22>
cd /d "%~dp0..\.."
set MODE=%~1
set ADDED=%~2
if "%MODE%"=="" set MODE=freeze
set ADDED=%ADDED:+=,%

set OUT=tests\tooling\.gate_cache\q4km_grid
set LOG=%OUT%\tau-%MODE%.log

set ADDEDARG=
if not "%ADDED%"=="" set ADDEDARG=--added "%ADDED%"

echo. >> "%LOG%"
echo ===== Q4KM TAU %MODE% %DATE% %TIME% (added '%ADDED%') ===== >> "%LOG%"

if /I "%MODE%"=="records" (
  python -u tests\tooling\q4km_tau.py --grid-dir "%OUT%" %ADDEDARG% --records-out "%OUT%\records-q4km.json" >> "%LOG%" 2>&1
) else (
  python -u tests\tooling\q4km_tau.py --grid-dir "%OUT%" %ADDEDARG% --envelope-out "%OUT%\tau-envelope.json" --tau-out "%OUT%\tau-q4km.json" >> "%LOG%" 2>&1
)

echo ===== Q4KM TAU %MODE% EXIT %ERRORLEVEL% %DATE% %TIME% ===== >> "%LOG%"
exit /b 0
