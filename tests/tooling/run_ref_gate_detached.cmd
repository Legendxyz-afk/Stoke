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
rem Run via Task Scheduler so an ~11h run survives session/terminal
rem boundaries -- a session-tracked background task gets reaped, which is how
rem the first W4 worker was lost.
rem
rem RESUMABLE: ref_gate.py writes each prompt's result to
rem tests\tooling\.ref_gate_cache\ atomically as it completes, so relaunching
rem after a kill/crash/reboot re-uses every finished prompt. The cache key
rem includes BOTH stoke.exe's and ref_forward.py's sha256, so touching either
rem correctly invalidates prior results instead of silently reusing them.
rem
rem Cost, re-probed 2026-07-30 after the 07-29 run died. THE RATE DEPENDS ON
rem WHAT ELSE IS RUNNING: p48 (66 positions) took 812.9s on a quiet box
rem (12.32 s/position) and 1335s with Chrome+Discord+VSCode open (20.22
rem s/position) -- same binaries, same --jobs 1, 1.64x apart, because ref
rem re-streams the 4.3 GB GGUF every position and needs it in page cache.
rem So over the 24 (3,503 positions): ~12-13h if the box is left alone,
rem ~20.5h if it is in use. Ceilings are sized off the LOADED rate, which is
rem why the run cannot spuriously error just because someone is gaming.
rem The old quote (~4.5h at --jobs 3) was arithmetic on a rate that had never
rem been measured under concurrency; see ref_gate.py's cost-model comment for
rem why "27.6 s/position" in the 07-29 log was a tautology, not a datum.
rem
rem --jobs 1, AND IT IS NOT A CORE COUNT -- ref is ~1-2 cores of 16. It is a
rem RAM/page-cache decision. One ref process is ~2,292 MB private (a ~1.6 GB
rem f32 token_embd cache plus transient per-layer dequants; ref
rem re-dequantizes every weight per matvec by design, since caching all 36
rem layers would need 14-16 GB) and it re-streams the whole 4.3 GB GGUF every
rem position, so it wants that file in page cache. At --jobs 3 that is
rem 3 x 2.3 GB private + 4.3 GB cache on a 15.6 GB box: it paged, per-worker
rem cost rose >=2.6x, and aggregate throughput was ~1.15x serial for 3x the
rem RAM -- all six prompts that ran hit their ceiling and banked NOTHING.
rem At --jobs 1 the hard cost is 2.3 GB, the page cache is give-and-take with
rem whatever else is running, and the box stays usable (games included) for
rem the whole 11h. Raising this requires PROBING the per-worker rate at the
rem new job count first: ref_gate.py refuses a job count it has no measured
rem rate for, by design.
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
python -u tests\tooling\ref_gate.py --subset p01,p02,p05,p08,p09,p11,p12,p16,p20,p24,p25,p26,p28,p29,p37,p39,p41,p43,p44,p45,p46,p47,p48,p49 --jobs 1 >> "tests\tooling\.ref_gate_cache\ref-gate-console.log" 2>&1
