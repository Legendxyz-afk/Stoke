@echo off
rem L2 CONTROLS: what does a NON-divergent M2-vs-HEAD layer curve look like?
rem
rem ▶▶ WHY THIS RUN EXISTS -- the p41 ladder answered with a borrowed bar.
rem l2_diff.py's 1e-3 threshold was calibrated for C-f32 vs numpy-f32, where the
rem only difference is float32 non-associativity (W3: worst layer 6.1e-4). p41's
rem ladder ran M2 f32 vs HEAD INTEGER-CORE, which differ BY CONSTRUCTION -- K2
rem quantizes activations to int8 per block -- so 36/36 layers over 1e-3 is the
rem expected result for that pair and carries no information. The STEP test, the
rem part that actually discriminates a structural bug from drift, was NEGATIVE.
rem
rem So the absolute bar has to be MEASURED for this pair rather than inherited,
rem exactly as tau was: calibrate on positions where the two engines AGREE, then
rem judge the divergent position against that. Disjoint sets, same logic.
rem
rem ▶ THE CONTROL POSITION IS THE LAST PREFILL POSITION, and it is free of the
rem teacher-forcing risk the p41 dump had to take. `--dump` fires at
rem `t == n_prompt-1`, and during prefill BOTH binaries consume the identical
rem given tokens -- no generation has happened yet, so no divergence can have
rem occurred. Running each prompt unmodified therefore dumps a guaranteed
rem agreeing position, with no appended tokens and no re-tokenization to verify.
rem
rem ▶ THE CONTROLS, chosen to bracket p41's divergence at abs 478:
rem   p41 itself at abs 476 -- SAME prompt, two positions earlier, no divergence
rem                            yet. The tightest possible control: if the curve
rem                            here matches the curve at 478, then nothing about
rem                            the layer stack changed at the divergence.
rem   p43 at abs 470, p42 at abs 461 -- different prompts, same band, for breadth
rem                            so the verdict does not rest on one curve.
rem
rem ▶ COST: 477 + 471 + 462 = 1,410 M2 prefill positions. THE LOOP IS SEQUENTIAL
rem -- an earlier version of this header claimed "run concurrently ~31 min wall",
rem which this file never did. At the measured SUSTAINED pinned rate (~3.9
rem s/position) a from-scratch run is ~92 min; the ~1.6 s/position boost window a
rem cool box gives for a few minutes after a restart would say ~38 min, and
rem quoting either without naming the window is how the FOR-FABLE memo ended up
rem with a rate it could not reproduce. HEAD's side is ~0.15 s/position,
rem negligible. One prompt alone is ~30 min sustained.
rem
rem ▶ RESUMABLE PER PROMPT. A control whose two dumps are already complete on
rem disk is not re-run -- the 08-06 box shutdown landed mid-p42 with p41 and p43
rem already banked, and without this the only way to finish p42 was to spend 60
rem min re-deriving the two curves that were already in the log.
cd /d "%~dp0..\.."
set CACHE=tests\tooling\.gate_cache
set LOG=%CACHE%\l2-controls.log
set MODEL=models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf
rem COUPLED TO %MODEL%: layers are dumped in ascending order (model.c:645), so
rem the last layer's file is the completeness marker for a dump directory. Qwen3-4B
rem has 36 layers => L35. Change this line whenever %MODEL% changes; a model with
rem MORE layers would make L35 a false "complete" and skip a truncated dump.
set LASTLAYER=L35

echo. >> "%LOG%"
echo ===== L2 CONTROLS %DATE% %TIME% ===== >> "%LOG%"

for %%P in (p41 p43 p42) do call :control %%P

echo ===== L2 CONTROLS COMPLETE %DATE% %TIME% ===== >> "%LOG%"
exit /b 0

rem ---------------------------------------------------------------------------
rem :control <prompt> -- dump both sides at the last prefill position, then diff.
rem
rem The skip test is a FLAG, not `if exist A if exist B (...) else (...)`: in that
rem form the `else` binds to the INNER `if`, so a missing A runs NEITHER branch --
rem it would skip the dumps and then hand l2_diff two directories that do not
rem exist. Failure would be loud here, but the same shape silently does nothing in
rem cases where the diff is optional.
:control
set P=%~1
set SKIP=
if exist "%CACHE%\ctl_head_%P%\%LASTLAYER%.f32" if exist "%CACHE%\ctl_m2_%P%\%LASTLAYER%.f32" set SKIP=1
if defined SKIP (
  echo [%P%] dumps already complete on disk -- prefills skipped >> "%LOG%"
) else (
  if exist "%CACHE%\ctl_head_%P%" rmdir /s /q "%CACHE%\ctl_head_%P%"
  if exist "%CACHE%\ctl_m2_%P%"   rmdir /s /q "%CACHE%\ctl_m2_%P%"
  echo [%P%] HEAD dump at last prefill >> "%LOG%"
  build\win\stoke.exe run "%MODEL%" --raw -f tests\data\parity\%P%.txt -n 1 --dump "%CACHE%\ctl_head_%P%" >> "%LOG%" 2>&1
  echo [%P%] M2 dump at last prefill >> "%LOG%"
  build\stoke-m2-gate-d2b17c53.exe run "%MODEL%" --raw -f tests\data\parity\%P%.txt -n 1 --dump "%CACHE%\ctl_m2_%P%" >> "%LOG%" 2>&1
)
rem l2_diff is seconds and is re-run even on a skip, so a resumed run leaves a log
rem holding all three curves rather than only the ones this invocation computed.
echo [%P%] l2_diff control curve >> "%LOG%"
python -u tests\tooling\l2_diff.py "%CACHE%\ctl_m2_%P%" "%CACHE%\ctl_head_%P%" >> "%LOG%" 2>&1
goto :eof
