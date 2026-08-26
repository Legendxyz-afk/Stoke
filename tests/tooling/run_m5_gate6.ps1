# =====================================================================
# M5 GATE ITEM [6] -- PREEMPT-LOSSLESS ON A FORCED-PREEMPTION REAL-MODEL TRACE.
#
#   [6] preempt-lossless demonstrated on a forced-preemption real-model trace
#
# Item [1] asserts losslessness in the SIM (S6: pressure run vs infinite-pool
# reference, streams byte-identical). The sim's cost model is an INPUT, so it
# cannot settle this: [6] asks for the real model.
#
# ▶▶ WHY THIS HAD NEVER BEEN RUN, WHICH IS THE FINDING BEHIND THE RUNNER.
# Every serve measurement this project has banked reports `preemptions 0`, and
# not by luck -- the serve driver COULD NOT PRODUCE the trace:
#
#   1. The pool is `n_seq * blocks(ctx)`, so every sequence can always reach its
#      own context. `--pool-blocks` was added for this gate.
#   2. Admission checks `blocks(prompt) + W`, so a request whose whole GROWTH
#      fits inside W's 8-block reserve can never outgrow what admission already
#      approved. At n 32 a sequence grows 2 blocks against a reserve of 8 --
#      measured: a squeezed pool made the clients SERIALISE (rows 1,349 ->
#      4,805, preemptions still 0). Growth must exceed W*16 = 128 tokens.
#   3. Admission also refuses to overcommit, so a merely SMALL pool just admits
#      fewer sequences. Measured at pool 200: 2 of 4 admitted, both fit at full
#      length, preemptions 0.
#
# ⇒ Preemption fires only when the pool sits in the window
#      [ k * (blocks(prompt) + W) , k * blocks(prompt + n_gen) )
# -- enough for k requests to be ADMITTED, not enough for them to FINISH.
# Here blocks(1174)+8 = 82 and blocks(1474) = 93, so k=2 gives [164, 186).
# POOL 170 IS INSIDE IT AND IS MEASURED TO PREEMPT 3 TIMES.
#
# This is the real-model form of the lesson the sim already recorded: a pressure
# trace must be DELIBERATELY ADVERSARIAL, or it banks as a pass proving nothing.
#
# ▶ THE FIXTURE IS `fanout-hetero`, NOT S3, and that is not a detail.
# S3's eight clients differ by ONE DIGIT and its continuation does not depend on
# its prefix, so a byte-compare over it cannot see a replay defect. The hetero
# set has pairwise-distinct streams AND a load-bearing shared prefix, and both
# dedup mutants DIFFER on it (2026-08-25).
#
# ▶ AND THE GATE CARRIES ITS OWN PROOF IT COULD FAIL: arm M is a binary built
# with -DSTOKE_MUT_PREEMPT, whose preempt replay is off by one row. It must
# DIFFER. Without it, two SAMEs would be equally consistent with a preemption
# path that never executed.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_gate6.ps1
#        [-Seqs 4] [-NGen 300] [-Pool 170] [-Threads 8] [-Ctx 1600] [-Quick]
# =====================================================================
param(
    [int]$Seqs    = 4,
    [int]$NGen    = 300,
    [int]$Pool    = 170,
    [int]$Threads = 8,
    [int]$Ctx     = 1600,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$root  = "M:\New Drive\Desktop\Programming\stoke"
$exe   = "$root\build\win\stoke.exe"
$mut   = "$root\build\win-mut-preempt\stoke.exe"
$model = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$pdir  = "$root\tests\data\bench_prompts"
$gr    = "$root\docs\superpowers\gate-reports"

# ⚠ -Quick MUST STILL LAND INSIDE THE PREEMPTION WINDOW, or it "fails" for a
# reason that has nothing to do with the code. At 2 seqs over c00/c01 the
# longest prompt is 1169 tokens: blocks = 74, +W = 82, so k=2 admits at 164;
# blocks(1169+200) = 86, so k=2 finishes at 172. The window is [164, 172).
# A smaller pool (88 was the first guess) admits ONE sequence at a time and the
# clients serialise -- preemptions 0, and claim 2 fails on the fixture.
if ($Quick) { $Seqs = 2; $NGen = 200; $Pool = 168 }
$SHAPE = "s${Seqs}n${NGen}p${Pool}"
$Stamp = "{0}-m5-gate6{1}" -f (Get-Date -Format "yyyy-MM-dd"),
                              $(if ($Quick) { "-quick" } else { "" })
$log    = "$gr\$Stamp-raw.txt"
$report = "$gr\$Stamp-report.md"
$tmp    = "$env:TEMP\stoke-gate6"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

foreach ($p in @($exe, $mut, $model, $pdir)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}
$newestSrc = (Get-ChildItem "$root\src" -Include *.c, *.h -Recurse |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1)
foreach ($b in @($exe, $mut)) {
    if ((Get-Item -LiteralPath $b).LastWriteTime -lt $newestSrc.LastWriteTime) {
        throw "STOP: $b is OLDER than $($newestSrc.Name) -- rebuild before gating"
    }
}

# Write-Host, not Write-Output: Say is called inside a function whose return
# value is parsed, and output-stream status lines become part of it.
function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Host $line
}
function Grp($text, $re, $i) {
    $m = [regex]::Match($text, $re)
    if ($m.Success) { return [double]$m.Groups[$i].Value }
    return $null
}
function PromptArgs($n) {
    $a = @()
    for ($i = 0; $i -lt $n; $i++) {
        $a += "-f"; $a += ('"' + ("$pdir\fanout-hetero-c{0:d2}.txt" -f $i) + '"')
    }
    return $a
}

function Arm($id, $binary, $pool) {
    $ids = "$tmp\$id-$SHAPE.txt"
    $out = "$tmp\$id-$SHAPE-stdout.txt"
    if ((Test-Path $ids) -and (Test-Path $out)) {
        $t = Get-Content $out -Raw
        if ($t -and $null -ne (Grp $t 'preemptions (\d+)' 1)) {
            Say "  $id already banked, skipping"
            return @{ ids = $ids; text = $t }
        }
    }
    # --no-dedup IS DELIBERATE AND IS STATED IN THE REPORT. With dedup on,
    # admission still reserves the whole prompt while adoption makes the actual
    # residency SHARED, so `reserved` exceeds what the sequences occupy and the
    # window above closes. That path's re-adoption-after-preemption is asserted
    # in the sim; on the real model it is OWED, and the report says so.
    $a = @("bench", "--serve", ('"' + $model + '"')) + (PromptArgs $Seqs) +
         @("--raw", "--seqs", "$Seqs", "--ctx", "$Ctx", "-n", "$NGen",
           "--threads", "$Threads", "--kv", "f32", "--no-dedup",
           "--ids", ('"' + $ids + '"'))
    if ($pool -gt 0) { $a += @("--pool-blocks", "$pool") }
    $t0 = Get-Date
    $p = Start-Process -FilePath $binary -ArgumentList $a -RedirectStandardOutput $out `
                       -RedirectStandardError "$tmp\e-$id.txt" -NoNewWindow -PassThru -Wait
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if ($p.ExitCode -ne 0) {
        Say ("  !! $id exited {0}: {1}" -f $p.ExitCode, (Get-Content "$tmp\e-$id.txt" -Raw))
        throw "STOP: $id exited $($p.ExitCode) -- a broken arm must stop, never DIFFER"
    }
    $t = Get-Content $out -Raw
    if ($t -match '! only \d+ of \d+ expected ids') { throw "STOP: $id produced fewer ids than asked" }
    Say ("  {0,-3} preemptions {1,-4} steps {2,-5} rows {3,-6} pool {4,-5} ({5}s)" -f
         $id, (Grp $t 'preemptions (\d+)' 1), (Grp $t 'steps\s+(\d+)' 1),
         (Grp $t 'rows (\d+)' 1), $(if ($pool -gt 0) { $pool } else { "default" }), $secs)
    return @{ ids = $ids; text = $t }
}

Say "M5 gate [6] -- preempt-lossless on a forced-preemption REAL-MODEL trace. stamp $Stamp"
Say "seqs $Seqs x n $NGen, ctx $Ctx, threads $Threads, squeezed pool $Pool, fixture fanout-hetero, dedup OFF"
Say "exe commit $(git -C $root rev-parse --short HEAD)"
if ($Quick) { Say "*** -Quick: NOT a gate ***" }

$R = Arm "R" $exe 0        # reference: derived pool, must NOT preempt
$P = Arm "P" $exe $Pool    # pressure:  squeezed pool, MUST preempt
$M = Arm "M" $mut $Pool    # mutant:    same pressure, MUST differ

$preR = Grp $R.text 'preemptions (\d+)' 1
$preP = Grp $P.text 'preemptions (\d+)' 1
$preM = Grp $M.text 'preemptions (\d+)' 1
$hR = (Get-FileHash -LiteralPath $R.ids).Hash
$hP = (Get-FileHash -LiteralPath $P.ids).Hash
$hM = (Get-FileHash -LiteralPath $M.ids).Hash

# THE THREE CLAIMS, EACH ABLE TO FAIL ON ITS OWN.
$c1 = ($preR -eq 0)          # the reference really is unpressured
$c2 = ($preP -gt 0)          # the pressure arm really preempted
$c3 = ($hP -eq $hR)          # and it lost nothing
$c4 = ($hM -ne $hR)          # and this comparison can tell
$green = ($c1 -and $c2 -and $c3 -and $c4)

$lines = @()
$lines += "# M5 gate item [6] -- preempt-lossless on a forced-preemption real-model trace"
$lines += ""
$lines += "Run ``$Stamp``. Fixture **fanout-hetero** ($Seqs clients, pairwise-distinct streams over a"
$lines += "load-bearing shared prefix), $NGen generated, ctx $Ctx, threads $Threads, kv f32, **dedup OFF**."
$lines += "Raw ``$Stamp-raw.txt``."
$lines += ""
$lines += "| arm | binary | pool blocks | preemptions | steps | rows | ids sha256 |"
$lines += "|---|---|---|---|---|---|---|"
foreach ($x in @(@{n = "R (reference)"; r = $R; p = "derived"; h = $hR },
                 @{n = "P (pressure)";  r = $P; p = "$Pool";   h = $hP },
                 @{n = "M (mutant)";    r = $M; p = "$Pool";   h = $hM })) {
    $lines += ("| {0} | {1} | {2} | {3:n0} | {4:n0} | {5:n0} | ``{6}`` |" -f
        $x.n, $(if ($x.n -like "M*") { "win-mut-preempt" } else { "win" }), $x.p,
        (Grp $x.r.text 'preemptions (\d+)' 1), (Grp $x.r.text 'steps\s+(\d+)' 1),
        (Grp $x.r.text 'rows (\d+)' 1), $x.h.Substring(0, 16))
}
$lines += ""
$lines += "## The four claims, each able to fail on its own"
$lines += ""
$lines += "| # | claim | result |"
$lines += "|---|---|---|"
$lines += ("| 1 | the reference is genuinely unpressured (preemptions == 0) | {0} |" -f $(if ($c1) { "**PASS** (0)" } else { "**FAIL** ($preR)" }))
$lines += ("| 2 | the pressure arm genuinely preempted (preemptions > 0) | {0} |" -f $(if ($c2) { "**PASS** ($preP)" } else { "**FAIL** (0 -- the trace never exercised the path)" }))
$lines += ("| 3 | **preemption lost nothing**: P's stream == R's, byte for byte | {0} |" -f $(if ($c3) { "**PASS**" } else { "**FAIL**" }))
$lines += ("| 4 | the comparison can tell: the off-by-one mutant DIFFERS | {0} |" -f $(if ($c4) { "**PASS** (mutant preempted $preM times and diverged)" } else { "**FAIL** -- claim 3 proves nothing" }))
$lines += ""
$lines += ("## VERDICT (gate [6]): {0}" -f $(if ($green) { "**MET**" } else { "**NOT MET**" }))
$lines += ""
$lines += "Claim 4 is why claims 1-3 mean anything. The mutant is the shipping source built"
$lines += 'with `-DSTOKE_MUT_PREEMPT`, which replays one row too FEW on preemption, so the'
$lines += "final replayed row reproduces the logits of the wrong position: the run stays alive"
$lines += "and the output is merely wrong -- the failure class ``kv.h`` names as the dangerous"
$lines += "one. Two SAMEs without it would be equally consistent with a preemption path that"
$lines += "never executed, which is exactly what every previous serve measurement recorded."
$lines += ""
$lines += "## Why a forced-preemption trace needed a new flag"
$lines += ""
$lines += "The serve driver could not produce one. The pool is ``n_seq * blocks(ctx)``, so every"
$lines += "sequence can always reach its own context; ``--pool-blocks`` was added for this gate."
$lines += "Two further guards had to be defeated, both measured on the way here:"
$lines += ""
$lines += "- **admission's watermark.** It approves ``blocks(prompt) + W``, so a request whose"
$lines += "  whole growth fits in W's 8-block reserve can never outgrow it. At n 32 a sequence"
$lines += "  grows 2 blocks: a squeezed pool made the clients SERIALISE instead (rows 1,349 ->"
$lines += "  4,805, preemptions still 0). Growth must exceed W*16 = 128 tokens."
$lines += "- **admission's refusal to overcommit.** A merely small pool admits fewer sequences."
$lines += "  At pool 200, 2 of 4 admitted and both fit at full length: preemptions 0."
$lines += ""
$lines += "Preemption fires only for a pool inside **[k*(blocks(prompt)+W), k*blocks(prompt+n_gen))**"
$lines += "-- enough for k requests to be ADMITTED, not enough for them to FINISH. Here that is"
$lines += "[164, 186) at k=2, and $Pool sits inside it."
$lines += ""
$lines += "## What this gate does NOT cover"
$lines += ""
$lines += "**The dedup path.** With dedup on, admission still reserves the whole prompt while"
$lines += "adoption makes the residency SHARED, so ``reserved`` exceeds what the sequences occupy"
$lines += "and the window above closes -- measured: at pools 84 and 88 the clients serialised"
$lines += "and re-prefilled (rows 1,349 -> 4,805) without one preemption. Re-adoption after"
$lines += "preemption is asserted in the sim (``test_sched``: 1024 adopted over 9 admissions with"
$lines += "3 preemptions). **On the real model it is OWED**, and since dedup is now the serve"
$lines += "DEFAULT that is the more important of the two paths."
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
Say ("VERDICT: gate [6] {0} -- R {1} preemptions, P {2}, streams {3}, mutant {4}" -f
     $(if ($green) { "MET" } else { "NOT MET" }), $preR, $preP,
     $(if ($c3) { "IDENTICAL" } else { "DIFFER" }), $(if ($c4) { "DIFFERS (good)" } else { "SAME (bad)" }))
