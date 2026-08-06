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
rem ▶ COST: 477 + 471 + 462 = 1,410 M2 prefill positions at the measured
rem sustained pinned rate (~3.9 s/position). Run concurrently they finish in
rem ~31 min wall. HEAD's side is ~0.15 s/position, negligible.
cd /d "%~dp0..\.."
set CACHE=tests\tooling\.gate_cache
set LOG=%CACHE%\l2-controls.log
set MODEL=models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf

echo. >> "%LOG%"
echo ===== L2 CONTROLS %DATE% %TIME% ===== >> "%LOG%"

for %%P in (p41 p43 p42) do (
  if exist "%CACHE%\ctl_head_%%P" rmdir /s /q "%CACHE%\ctl_head_%%P"
  if exist "%CACHE%\ctl_m2_%%P"   rmdir /s /q "%CACHE%\ctl_m2_%%P"
  echo [%%P] HEAD dump at last prefill >> "%LOG%"
  build\win\stoke.exe run "%MODEL%" --raw -f tests\data\parity\%%P.txt -n 1 --dump "%CACHE%\ctl_head_%%P" >> "%LOG%" 2>&1
  echo [%%P] M2 dump at last prefill >> "%LOG%"
  build\stoke-m2-gate-d2b17c53.exe run "%MODEL%" --raw -f tests\data\parity\%%P.txt -n 1 --dump "%CACHE%\ctl_m2_%%P" >> "%LOG%" 2>&1
  echo [%%P] l2_diff control curve >> "%LOG%"
  python -u tests\tooling\l2_diff.py "%CACHE%\ctl_m2_%%P" "%CACHE%\ctl_head_%%P" >> "%LOG%" 2>&1
)

echo ===== L2 CONTROLS COMPLETE %DATE% %TIME% ===== >> "%LOG%"
exit /b 0
