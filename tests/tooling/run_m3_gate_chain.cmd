@echo off
rem Waits for the q8_0 leg to finish, then runs the q4_k_m leg.
rem
rem ▶ WHY SEQUENTIAL, MEASURED RATHER THAN ASSUMED. Running both legs at once
rem was tried on 2026-08-04 to use the ~6.7 GB of idle RAM, and it does not
rem fit. With both oracles up, free RAM fell to 939 MB and HARD PAGE-INS hit
rem 9,086/sec -- the two model page caches alone are 4.28 + 2.50 GB, and each
rem oracle plus each stoke wants its own private GB on top, against 15.6 GB
rem total. That is the documented failure mode where Windows trims the
rem worker's working set and per-position cost rises ~3x, so concurrency made
rem BOTH legs slower than running them one after the other. Killing the second
rem leg took free RAM back to 4.3 GB and page-ins to 1.6k and falling.
rem
rem The idle RAM was real; it was just not spare. A 4.3 GB model that must
rem stay resident does not leave 6.7 GB of headroom for a second 2.5 GB model
rem plus two engines -- "free" memory that the page cache needs is already
rem spoken for.
rem
rem Polls a status FILE rather than a process handle so a leg that dies is
rem still noticed: no "done" record ever appears, and this waiter times out
rem instead of hanging forever behind a PID that no longer exists.

setlocal
cd /d "%~dp0..\.."
set CACHE=tests\tooling\.gate_cache
set S1=%CACHE%\m3-gate-status-q8_0.json
set LOG=%CACHE%\m3-gate-chain.log

echo. >> "%LOG%"
echo ===== CHAIN WAITING for q8_0 %DATE% %TIME% ===== >> "%LOG%"

rem 8 hours of 60 s polls. The leg is expected to take well under one.
set /a TRIES=480
:wait
if %TRIES% LEQ 0 (
  echo [chain] TIMED OUT waiting for q8_0 -- leg 2 NOT started >> "%LOG%"
  exit /b 1
)
if exist "%S1%" (
  findstr /c:"\"phase\":\"done\"" "%S1%" >nul 2>&1
  if not errorlevel 1 goto ready
)
set /a TRIES-=1
timeout /t 60 /nobreak >nul 2>&1
goto wait

:ready
echo [chain] q8_0 finished, starting q4_k_m %DATE% %TIME% >> "%LOG%"
rem Port 8090 is free again once leg 1's server has exited.
call "%~dp0run_m3_gate_leg.cmd" q4_k_m 8090
echo [chain] q4_k_m exit %ERRORLEVEL% %DATE% %TIME% >> "%LOG%"
exit /b 0
