@echo off
rem Detached launcher for the [2v2-ii] PROBE PASS -- section 6's classification
rem input, plus the class-noise envelope tau is frozen from.
rem
rem WHAT IT DOES, in one pass over the completed sweep. For every prompt it
rem probes BOTH binaries at the one position where the two streams are
rem comparable, and routes the result by what the sweep found:
rem   divergent prompt -> GRADING RECORD at first_diff (the position section 6 judges)
rem   agreeing prompt  -> ENVELOPE SAMPLE at index 63 (class noise, for the tau freeze)
rem Those sets are disjoint, which is what keeps the threshold independent of
rem the divergences it will grade.
rem
rem ▶ IT CANNOT GRADE, AND THAT IS DELIBERATE. Ruling 2.3 requires tau frozen
rem BEFORE [2v2-ii]'s divergences are graded. The freeze is a separate
rem invocation (--freeze-tau) and classify.py fatals on a tau nobody froze, so
rem the ordering is structural rather than remembered.
rem
rem ▶▶ THE ENGINE FREEZE STILL APPLIES. m2_classify_probes.py re-verifies
rem build\win\stoke.exe against 876ce53f... and refuses to run otherwise:
rem probes from a rebuilt engine cannot explain streams it never generated.
rem
rem ▶ SIZING, measured 2026-08-06 at the SUSTAINED pinned rate (~3.9 s/position,
rem NOT the ~1.7 that short post-restart boost windows show): 6,378 M2 prefill
rem positions total = ~6.9 h serial, ~2.3 h at --jobs 3. Inside section 8.1's
rem ~4 h evidence-only authorization. The agreeing prompts dominate (4.8 h of
rem it) because they probe at index 63 while divergent ones stop at first_diff;
rem the five long prompts alone are ~2.8 h of that.
rem
rem ▶ RESUMABLE per prompt: records, envelope and the unprobeable sidecar are
rem each rewritten as every prompt completes and reloaded on relaunch, so an
rem interruption costs one prompt.
rem
rem ▶ PLACEMENT AND PRIORITY ARE SET BY THE TOOL, not here -- a probe IS a full
rem re-generation, so it is the same workload under the same physics that cost
rem this project 1.6x when left unplaced (work bouncing onto E-cores and second
rem hyperthreads). m2_classify_probes.py pins itself to one thread per P-core
rem and spawns children at HIGH_PRIORITY_CLASS explicitly, because priority is
rem NOT reliably inherited.
rem
rem `python -u` is REQUIRED: with stdout redirected Python block-buffers it, so
rem a multi-hour run shows a 0-byte log and a crash flushes nothing.
cd /d "%~dp0..\.."
set CACHE=tests\tooling\.gate_cache
set LOG=%CACHE%\m2-probes-console.log

if not exist "%CACHE%" mkdir "%CACHE%"
echo. >> "%LOG%"
echo ===== [2v2-ii] PROBE PASS %DATE% %TIME% ===== >> "%LOG%"

python -u tests\tooling\m2_classify_probes.py ^
  --sweep "%CACHE%\m2_sweep.json" ^
  --out "%CACHE%\m3_records.json" ^
  --envelope "%CACHE%\m3_envelope.json" ^
  --jobs 3 >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo ===== PROBE PASS COMPLETE rc=%RC% %DATE% %TIME% ===== >> "%LOG%"
exit /b %RC%
