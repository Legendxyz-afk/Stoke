@echo off
rem One leg of the M3 recorded gate.   usage: run_m3_gate_leg.cmd <model> <port>
rem
rem WHY LEGS ARE SEPARATE PROCESSES RATHER THAN A SEQUENTIAL SCRIPT. The two
rem legs are independent -- different model, different oracle server, and
rem since 2026-08-04 different cache keys -- so nothing forces them to be
rem serial. Running them concurrently costs almost no CPU, because the oracle
rem is pinned to `-t 1` and is fetched in a SERIAL loop, so each leg's oracle
rem occupies exactly one core. It costs RAM, which is the resource that was
rem sitting idle: Q8_0's oracle is ~4.5 GB and Q4_K_M's ~2.6 GB, plus ~1.2 GB
rem per stoke, against 15.6 GB total.
rem
rem ▶ PORTS MUST DIFFER. Both legs default to 8090; a collision would have the
rem second server fail to bind, or worse, have one leg quietly answered by the
rem other leg's model. The caller passes the port explicitly for that reason.
rem
rem ▶ WHAT --jobs CANNOT FIX, so nobody raises it hoping for speed. --jobs
rem parallelises the STOKE side only. Since K4/K5 the stoke side is ~19 min of
rem the leg while the oracle side is the rest, and the oracle is single-
rem threaded AND serialised by fetch_oracle_result's plain for-loop. Raising
rem --jobs past 1 therefore buys nothing on the bottleneck and oversubscribes
rem the CPU, since each stoke now runs 8 threads of its own.
rem
rem ▶ WHY NOT GIVE THE ORACLE MORE THREADS. `-t 1` is part of the frozen
rem oracle invocation. llama.cpp's reductions are thread-count dependent, so
rem changing it can change the ORACLE'S OWN token stream -- the baseline this
rem gate compares against. That is not a performance knob, it is the
rem experiment's control.

setlocal
set MODEL=%~1
set PORT=%~2
if "%MODEL%"=="" echo usage: run_m3_gate_leg.cmd ^<model^> ^<port^> & exit /b 2
if "%PORT%"=="" echo usage: run_m3_gate_leg.cmd ^<model^> ^<port^> & exit /b 2

cd /d "%~dp0..\.."
set CACHE=tests\tooling\.gate_cache
set LOG=%CACHE%\m3-gate-%MODEL%.log
set STATUS=%CACHE%\m3-gate-status-%MODEL%.json

if not exist "%CACHE%" mkdir "%CACHE%"
echo. >> "%LOG%"
echo ===== M3 GATE LEG %MODEL% port %PORT% %DATE% %TIME% ===== >> "%LOG%"
echo {"leg":"%MODEL%","phase":"running","port":%PORT%,"started":"%DATE% %TIME%"} > "%STATUS%"

python -u tests\tooling\parity_gate.py --model %MODEL% --port %PORT% --jobs 1 --clean >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo {"leg":"%MODEL%","phase":"done","rc":%RC%,"finished":"%DATE% %TIME%"} > "%STATUS%"
echo ===== LEG %MODEL% COMPLETE rc=%RC% ===== >> "%LOG%"
exit /b %RC%
