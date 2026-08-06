@echo off
rem The L2 rung of the ladder for p41, the one ESCALATE from [2v2-ii].
rem
rem WHY THIS RUN EXISTS. classify.py graded p41's divergence ESCALATE: worst
rem margin 0.460324 at abs 479 sits in [tau, 2*tau) = [0.394898, 0.789796) for
rem band [300,600). Ruling 9.4-1 says the ladder, not the constant, settles the
rem borderline band: max-abs-diff <= 1e-3 with NO STEP => ARITHMETIC-CLASS, a
rem STEP => BUG. l2_diff.py implements exactly that and is pre-authorized.
rem
rem ▶▶ WHY A CONSTRUCTED PROMPT, AND WHY IT IS SAFE. `--dump` writes L00..L35
rem only at the LAST PREFILL position (model.c: `last_prefill ? dump_dir :
rem NULL`), so to dump the state that PRODUCED generated index 2 the prefill
rem must end at absolute position 478. p41's own prompt is 477 tokens, and both
rem binaries agree on generated ids 0 and 1 (first_diff = 2), so appending those
rem two tokens' emitted bytes gives a 479-token prefill over a prefix BOTH sides
rem actually produced. The construction is VERIFIED, not assumed: tokenizing the
rem constructed file returns exactly p41's 477 ids followed by [18611, 334].
rem Without that check a BPE re-segmentation across the join would silently dump
rem a different position and the ladder would answer a question nobody asked.
rem
rem ▶ THE OUTPUT IDS ARE A SECOND CROSS-CHECK. With -n 1 each binary must emit
rem its OWN divergent token -- M2 3966, HEAD 16141 -- because that is what the
rem banked streams say it chose at this position. If either prints something
rem else, the dump does not correspond to the graded divergence and the L2
rem numbers must not be read.
rem
rem ▶ COST: the M2 binary prefills 479 positions at the measured sustained pinned
rem rate (~3.9 s/position) = ~31 min; HEAD is ~0.15 s/position = ~72 s. Well
rem inside section 8.1's evidence-only authorization.
rem
rem The engine freeze still applies: build\win\stoke.exe must remain 876ce53f...
cd /d "%~dp0..\.."
set CACHE=tests\tooling\.gate_cache
set LOG=%CACHE%\p41-ladder.log
set TF=%CACHE%\p41_tf479.txt
set MODEL=models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf

echo. >> "%LOG%"
echo ===== p41 L2 LADDER %DATE% %TIME% ===== >> "%LOG%"

if exist "%CACHE%\dump_head" rmdir /s /q "%CACHE%\dump_head"
if exist "%CACHE%\dump_m2"   rmdir /s /q "%CACHE%\dump_m2"

echo [1/3] HEAD dump (expect it to emit id 16141) >> "%LOG%"
build\win\stoke.exe run "%MODEL%" --raw -f "%TF%" -n 1 --dump "%CACHE%\dump_head" >> "%LOG%" 2>&1

echo [2/3] M2 dump (expect it to emit id 3966) >> "%LOG%"
build\stoke-m2-gate-d2b17c53.exe run "%MODEL%" --raw -f "%TF%" -n 1 --dump "%CACHE%\dump_m2" >> "%LOG%" 2>&1

echo [3/3] l2_diff >> "%LOG%"
python -u tests\tooling\l2_diff.py "%CACHE%\dump_m2" "%CACHE%\dump_head" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

echo ===== p41 LADDER COMPLETE rc=%RC% %DATE% %TIME% ===== >> "%LOG%"
exit /b %RC%
