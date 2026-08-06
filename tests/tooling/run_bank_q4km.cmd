@echo off
rem BANK HEAD's Q4_K_M STREAMS -- ruling §10 item 1, required evidence for [2v2-iii].
rem
rem 50 prompts x 64 generated ids on the Q4_K_M file, stoke only, no oracle. The
rem streams are what a teacher-forced probe is later forced AGAINST, so they have
rem to exist before any probe run can be costed, let alone launched.
rem
rem ▶ DETACHED because it outlives an agent tool call: ~17 min against a 10 min
rem ceiling. Task Scheduler is this tree's established pattern for that and is the
rem only one measured to survive.
rem
rem ▶ DO NOT RUN THIS WHILE ANYTHING HOLDS THE Q8_0 FILE. Alternating two large
rem mmapped models is the recorded 2026-08-04 page-cache churn: the shared model
rem starts being trimmed and re-faulted, Cache Faults/sec goes ~60 -> ~7,000 with
rem Pages Input/sec near zero, and prompts run ~3x slower. Q4_K_M is 2.33 GiB and
rem Q8_0 is 3.99 GiB; both resident at once does not fit this box's headroom.
rem
rem ▶ RESUME IS THE CACHE'S, not this file's -- an interrupted run costs the
rem in-flight prompt only, and a re-run of a finished bank prints 50 'cached'
rem lines and spends nothing.
cd /d "%~dp0..\.."
set CACHE=tests\tooling\.gate_cache
set LOG=%CACHE%\bank-q4km.log
if not exist "%CACHE%" mkdir "%CACHE%"

echo. >> "%LOG%"
echo ===== BANK Q4_K_M %DATE% %TIME% ===== >> "%LOG%"
python -u tests\tooling\bank_streams.py --model q4_k_m --progress "%CACHE%\bank-q4km.jsonl" >> "%LOG%" 2>&1
echo ===== BANK Q4_K_M EXIT %ERRORLEVEL% %DATE% %TIME% ===== >> "%LOG%"
exit /b 0
