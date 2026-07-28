@echo off
rem Detached launcher for M2's SUBSTITUTE GATE: stoke's C forward pass vs
rem ref_forward.py, both pure float32.
rem
rem THIS IS THE RECORDED M2 GATE RUN, not a diagnostic. Kabir took memo
rem option 3 on 2026-07-28 (docs\superpowers\memos\2026-07-20-m2-oracle-
rem near-tie-policy.md): the survey measured 19 of 50 prompts diverging from
rem llama-server, 18 of 18 classifiable hits being a clean top-2 swap at
rem 0.001-0.204 nat gaps -- the oracle quantizes activations to Q8 for
rem integer dot products while M2's stoke is pure f32, so the two rank
rem near-ties differently. Oracle token-parity is DEFERRED TO M3, where the
rem integer core puts stoke in the oracle's own numeric class.
rem
rem SUBSET AND PASS CRITERION ARE FABLE'S DECISION, fixed BEFORE this run so
rem that no bar can be written after seeing a result. 24 prompts = a full
rem census of all 19 oracle-diverging prompts + one oracle-exact control per
rem tier (p11 prose, p16 code, p26 multilingual, p37 math, p45 adversarial;
rem long-context deliberately gets no control -- 3 long jobs fill exactly 3
rem slots in one round, so a 4th ~120-min job would force a second round and
rem roughly double that tier's wall clock for one data point).
rem
rem THIS RUN ONLY MEASURES. It generates ids and records comparisons; it does
rem NOT adjudicate waivers. Adjudication is a separate, later step under
rem Fable's four conditions.
rem
rem Run via Task Scheduler so a ~4.5h run survives session/terminal
rem boundaries -- a session-tracked background task gets reaped, which is how
rem the first W4 worker was lost.
rem
rem RESUMABLE: ref_gate.py writes each prompt's result to
rem tests\tooling\.ref_gate_cache\ atomically as it completes, so relaunching
rem after a kill/crash/reboot re-uses every finished prompt. The cache key
rem includes BOTH stoke.exe's and ref_forward.py's sha256, so touching either
rem correctly invalidates prior results instead of silently reusing them.
rem
rem Cost (ref_gate.py --dry-run, verified against W1's 13.8 s/position
rem benchmark; p48 measured 12.32 s/position live, so the model is slightly
rem conservative): 3,503 positions, ~13.4h serial, ~4.5h wall at --jobs 3,
rem makespan floor = p41 at ~124 min.
rem
rem --jobs 3 IS A RAM CEILING, NOT A CORE COUNT. One ref process was measured
rem at 2,292 MB private (a ~1.6 GB f32 token_embd cache plus transient
rem per-layer dequants; ref re-dequantizes every weight per matvec by design,
rem since caching all 36 layers would need 14-16 GB). 3 x 2.3 + ~4.3 shared
rem GGUF page cache is already most of a 15.6 GB box. Do not raise it.
rem Stoke is NOT re-run here -- its 64 ids for every corpus prompt are banked
rem in .gate_cache under a stoke.exe hash guard.
rem
rem Console output APPENDS -- one writer only, never run two copies.
rem `python -u` is REQUIRED, not cosmetic: with stdout redirected Python
rem block-buffers it, so a multi-hour run shows a 0-byte log throughout and a
rem crash flushes nothing -- you lose progress AND the evidence.
cd /d "%~dp0..\.."
echo. >> "tests\tooling\.ref_gate_cache\ref-gate-console.log"
echo ===== REF GATE RUN %DATE% %TIME% ===== >> "tests\tooling\.ref_gate_cache\ref-gate-console.log"
python -u tests\tooling\ref_gate.py --subset p01,p02,p05,p08,p09,p11,p12,p16,p20,p24,p25,p26,p28,p29,p37,p39,p41,p43,p44,p45,p46,p47,p48,p49 --jobs 3 >> "tests\tooling\.ref_gate_cache\ref-gate-console.log" 2>&1
