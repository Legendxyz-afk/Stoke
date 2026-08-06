@echo off
rem Detached launcher for [2v2-ii]: the M2-vs-HEAD 50-prompt token sweep.
rem
rem WHAT THIS IS. Required evidence under the Fable ruling of 2026-08-05
rem (docs\superpowers\memos\2026-08-04-m3-gate-item-2-RULING-FABLE.md), which
rem retired item [2]'s live-oracle bar and replaced it with the classified-
rem divergence criterion. This is the criterion's first leg: run the archived
rem M2 binary over all 50 parity prompts and diff its token stream against
rem HEAD's, which the failed oracle gate already banked. It answers "did
rem K1-K6 change ANY token this corpus produces?" with NO llama-server at
rem all -- there is no oracle in this run, nothing binds a port, and nothing
rem here can emit a PASS. The verdict comes later, from the separate section-6
rem classification pass over whatever divergences this finds.
rem
rem ▶▶ THE ENGINE IS FROZEN WHILE THIS IS PENDING. m2_sweep.py:31 refuses any
rem banked stream not produced under stoke_hash 876ce53f..., and parity_gate.py
rem keys every cache entry on that same hash. REBUILDING build\win\stoke.exe --
rem including a pure-instrumentation rebuild that changes no arithmetic --
rem makes all 50 banked HEAD streams uncreditable and this script exits rather
rem than running. Recoverable (~19 min to re-bank) but discovered only after
rem the fact. Do not rebuild until this sweep is banked AND classified.
rem
rem ▶ RESUMABLE, AND DELIBERATELY WITHOUT A --clean EQUIVALENT. This exact job
rem died once at 6/50 to a planned shutdown. m2_sweep.py rewrites its state
rem file after every completed prompt and reloads on relaunch, so an
rem interruption costs one prompt. A row is reloaded ONLY if it carries the
rem ids it decided on, and ERROR rows are never reloaded -- so a resumed
rem verdict can always show its work, and a failure is retried rather than
rem inherited. Verified in BOTH directions before this launcher existed.
rem
rem ▶ ONE WRITER ONLY -- NEVER RUN TWO COPIES. flush() rewrites the whole state
rem file, so a second copy does not merely duplicate work, it clobbers rows.
rem The console log APPENDS for the same reason it does in the gate launchers.
rem
rem ▶ SIZING, measured not recalled: the M2 binary runs ~1.7 s/position, and
rem recorded per-prompt times ran 147-940 s ⇒ ~3.5 h serial for all 50. That
rem is inside the ruling's section 8.1 authorization for evidence-only runs
rem (~4 h), which is why this one is launched rather than asked.
rem
rem ▶ RAM IS THE PRECONDITION, NOT A QUIET BOX -- and the distinction is the
rem point. Box state governs VALIDITY for perf numbers but only DURATION for
rem correctness ones: these ids are deterministic (proven bit-identical at
rem 1/2/8/16 threads), so ambient load cannot change the answer, only how long
rem it takes. What it DOES need is headroom: the M2 process holds ~1.2 GB
rem private (KV allocated at max_ctx up front) against a 4.0 GB shared model
rem page cache, and if that cache is evicted the recorded ~3x rate collapse
rem applies. Launch condition is free RAM > model + 2 GB (~6 GB). The 08-05
rem attempt was refused at 0.31 GB free.
rem
rem `python -u` is REQUIRED, not cosmetic: with stdout redirected Python block-
rem buffers it, so a multi-hour run shows a 0-byte log the whole time and a
rem crash flushes nothing -- you lose both progress and the evidence.
cd /d "%~dp0..\.."
set CACHE=tests\tooling\.gate_cache
set LOG=%CACHE%\m2-sweep-console.log
set STATE=%CACHE%\m2_sweep.json

if not exist "%CACHE%" mkdir "%CACHE%"
echo. >> "%LOG%"
echo ===== M2-vs-HEAD SWEEP [2v2-ii] %DATE% %TIME% ===== >> "%LOG%"

python -u tests\tooling\m2_sweep.py "%STATE%" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo ===== SWEEP COMPLETE rc=%RC% %DATE% %TIME% ===== >> "%LOG%"
exit /b %RC%
