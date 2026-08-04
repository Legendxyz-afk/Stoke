@echo off
rem Detached launcher for THE M3 RECORDED GATE RUN -- forward-note gate [2],
rem the 50-prompt greedy oracle parity re-pass on BOTH files.
rem
rem THIS IS THE RECORDED RUN, not a survey. It enforces the waiver budget and
rem emits a PASS/FAIL verdict per leg. Two legs, sequentially:
rem     leg 1  --model q8_0     (the M2-era arithmetic class, now integer-core)
rem     leg 2  --model q4_k_m   (K-quants -- the file the bar is quoted on)
rem
rem --clean on both: THE recorded run never reads a dev cache. The per-prompt
rem cache is still WRITTEN atomically as each prompt completes, which is what
rem makes the run resumable and what the status bar counts.
rem
rem ▶ WHY --jobs 1 AND NOT THE OLD 4. When DEFAULT_JOBS=4 was chosen stoke was
rem single-threaded. Since K5 it defaults to 8 threads, so --jobs 4 would put
rem 32 threads on 16 logical processors and measure contention instead of the
rem engine. One stoke at 8T is ~5x a single thread where 4 jobs x 1T is ~4x,
rem and it holds a quarter of the RAM, which leaves room for the oracle.
rem
rem ▶ SIZING, re-probed 2026-08-04 against the CURRENT binary (the old header
rem quoted 6.07 s/forward, fitted on the M2 binary, and K2-K5 have all changed
rem forward-pass cost). Measured now: 0.140-0.179 s/forward on Q8_0 and
rem 0.127-0.151 on Q4_K_M, so the stoke side of all 50 prompts is ~19 min and
rem ~17 min respectively -- against ~12 h at the old rate. THE GATE IS NOW
rem DOMINATED BY THE ORACLE, not by stoke, which inverts the sizing assumption
rem every older comment in this tree is written against. Oracle side is
rem ESTIMATED at ~20-30 min per leg at -t 1 and has NOT been measured.
rem
rem `python -u` is REQUIRED, not cosmetic: with stdout redirected Python block-
rem buffers it, so a long run shows a 0-byte log the whole time and a crash
rem flushes nothing -- you lose both progress and the evidence.
rem
rem Console output APPENDS. One writer only -- never run two copies.

cd /d "%~dp0..\.."
set CACHE=tests\tooling\.gate_cache
set LOG=%CACHE%\m3-gate-console.log
set STATUS=%CACHE%\m3-gate-status.json

if not exist "%CACHE%" mkdir "%CACHE%"

echo. >> "%LOG%"
echo ===== M3 GATE RUN %DATE% %TIME% ===== >> "%LOG%"

rem ---- leg 1: Q8_0 -------------------------------------------------------
call :count_cache
echo {"leg":"q8_0","phase":"running","baseline":%CACHECOUNT%,"legs_done":0,"started":"%DATE% %TIME%"} > "%STATUS%"
echo [leg 1/2] q8_0 starting, cache baseline %CACHECOUNT% >> "%LOG%"
python -u tests\tooling\parity_gate.py --model q8_0 --jobs 1 --clean >> "%LOG%" 2>&1
set RC1=%ERRORLEVEL%
echo [leg 1/2] q8_0 exit %RC1% >> "%LOG%"

rem ---- leg 2: Q4_K_M ----------------------------------------------------
call :count_cache
echo {"leg":"q4_k_m","phase":"running","baseline":%CACHECOUNT%,"legs_done":1,"rc1":%RC1%,"started":"%DATE% %TIME%"} > "%STATUS%"
echo [leg 2/2] q4_k_m starting, cache baseline %CACHECOUNT% >> "%LOG%"
python -u tests\tooling\parity_gate.py --model q4_k_m --jobs 1 --clean >> "%LOG%" 2>&1
set RC2=%ERRORLEVEL%
echo [leg 2/2] q4_k_m exit %RC2% >> "%LOG%"

echo {"leg":"q4_k_m","phase":"done","legs_done":2,"rc1":%RC1%,"rc2":%RC2%,"finished":"%DATE% %TIME%"} > "%STATUS%"
echo ===== M3 GATE RUN COMPLETE rc1=%RC1% rc2=%RC2% ===== >> "%LOG%"
exit /b 0

:count_cache
set CACHECOUNT=0
for /f %%N in ('dir /b "%CACHE%\*.json" 2^>nul ^| find /c /v ""') do set CACHECOUNT=%%N
exit /b 0
