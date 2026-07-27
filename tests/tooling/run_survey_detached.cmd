@echo off
rem Detached launcher for the M2 parity DIVERGENCE-CLASS SURVEY.
rem
rem THIS IS NOT A GATE RUN. --survey is diagnostic only: it never early-stops,
rem never enforces the waiver budget, never consumes a reserve prompt, and can
rem not emit a PASS verdict. THE recorded M2 gate run is a separate, later
rem invocation (--clean WITHOUT --survey) and is driven by hand.
rem
rem Run via Task Scheduler (or a double-click) so the multi-hour run survives
rem terminal/session restarts -- a session-tracked background task gets reaped
rem at the session boundary, which is exactly how the first W4 worker was lost.
rem
rem RESUMABLE, and deliberately so: no --clean here. parity_gate.py stores each
rem prompt's oracle+stoke result in tests\tooling\.gate_cache\ atomically as it
rem completes, so relaunching this same file after a kill/crash/reboot picks up
rem every finished prompt from cache and only re-runs what is actually missing.
rem
rem Cost at the manager's 2026-07-27 measurements: ~7,038 forward passes,
rem ~9-11.5h serial, ~3-4h wall at --jobs 4. --jobs 4 is a RAM ceiling, not a
rem core count (each stoke.exe holds ~1,181 MB private); do not raise it on a
rem 16 GB box. Console output APPENDS -- one writer only, never run two copies.
rem `python -u` is REQUIRED, not cosmetic: with stdout redirected to a file
rem Python block-buffers it, so a multi-hour run shows a 0-byte log the whole
rem time and a crash flushes nothing at all -- you lose both progress and the
rem evidence. Verified: the first launch sat at 0 bytes while genuinely
rem running. -u makes progress observable and makes a failure diagnosable.
cd /d "%~dp0..\.."
python -u tests\tooling\parity_gate.py --survey --jobs 4 >> "tests\tooling\.gate_cache\survey-console.log" 2>&1
