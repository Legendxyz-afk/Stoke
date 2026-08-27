# =====================================================================
# WHAT IS THE vpmaddubsw Q6_K KERNEL WORTH ON A REAL PREFILL TOKEN?
#
# The isolated kernel reads 7.68x at N=16 (83.68 vs 10.90 GF/s, 2026-08-27).
# ▶ 7.68x ON ONE KERNEL IS NOT 7.68x ON A TOKEN, and this harness exists
# because that sentence is the whole point of the rung. `ffn_down` is Q6_K in 18
# of 36 layers; the other matvecs are Q4_K and untouched, and prefill also spends
# time in attention (which grows with position and is not touched here),
# quantization to Q8_K, RoPE, norms and softmax.
#
# ============================ THE PREDICTION ============================
# Pre-registered so the run can refute it. Bands are NOT re-fitted afterwards:
# a falsified prediction stays on the page with the right measurement beside it,
# because re-fitting is the one move that costs a pre-registration its value.
#
#   [P1] SHAPE -- speedup RISES with prompt length, then saturates.
#        Chunked prefill (STOKE_PREFILL_CHUNK 128) tiles at KERNELS_BATCH_TILE
#        16, and the kernel's advantage is strongly N-dependent: from the banked
#        table, maddubs/ref is 2.45x at N=2, 2.97x at N=4, 4.04x at N=8 and
#        7.68x at N=16. A 23-token prompt runs ONE full 16-tile plus a 7
#        remainder; a 1436-token prompt runs overwhelmingly full 16-tiles. So
#        p00 must show the least and `long` the most.
#        FALSIFIED IF: speedup(long) <= speedup(p00), outside the null.
#
#   [P2] MAGNITUDE -- a CEILING, not an estimate, AND RE-DERIVED FROM THE
#        MEASURED KERNEL RATE BEFORE ANY DATA EXISTED.
#        ADDENDUM 19 put the matvec-portion gain at ~1.67x, but that assumed
#        Q6_K reaching ~70 GF/s. The kernel actually measures 83.68 GF/s at
#        N=16, so 1.67x is the wrong ceiling and a true value between 1.67 and
#        1.71 would read as FALSIFIED when it is not. Re-derived from ADDENDUM
#        19's own terms (0.897 GFLOP/token of Q6_K; the rest of the matvec work
#        is a FIXED 0.0907 units):
#            Q6_K at 10.9 GF/s -> 0.0823 + 0.0907 = 0.173 units  (baseline)
#            Q6_K at 83.68     -> 0.0107 + 0.0907 = 0.1014       -> 1.71x
#            Q6_K free (inf)   -> 0.0000 + 0.0907 = 0.0907       -> 1.91x ASYMPTOTE
#        So 1.71x bounds the MATVEC PORTION. Prefill is strictly MORE than its
#        matvecs -- attention, Q8_K quantization, RoPE, norms, softmax are all
#        untouched -- so END-TO-END must sit strictly below 1.71x.
#        FALSIFIED IF: any cell reaches 1.71x. A result above the ceiling means
#        THE MEASUREMENT IS WRONG, not that the kernel is brilliant -- the same
#        reading run_m5_e3_prefill_ab.ps1 applied to its own 128x bound.
#        ⚠ This band was fixed BEFORE the first run. Re-deriving a prediction
#        from better inputs before seeing data is getting it right; changing it
#        afterwards is re-fitting, and that is what costs a pre-registration its
#        value. If the run falsifies this, it stays falsified.
#
#   ⚠ NOTE ON N=1: the batched kernel is SLOWER solo (0.86x at N=1). It cannot
#   be reached there -- vec.c:371 guards the batched path with `n_seq > 1` and
#   routes N=1 to the solo kernel. Recorded so a future reader does not "fix" it.
#
# ▶▶ THE NULL ARM IS NOT OPTIONAL AND IT IS NOT SHARED ACROSS CELLS.
# `ref` is measured TWICE per cell, as two separate processes. ref2/ref1 is what
# this harness CANNOT RESOLVE, and it is the only honest denominator for a
# maddubs/ref ratio that is also computed between processes. It is reported PER
# CELL and never as one pooled number: on 2026-08-27 a between-process null read
# 30% at the smallest cell (the first thing measured, absorbing load, warm-up and
# paging) and 1.0% at the largest, and a single pooled floor would have buried a
# real 7.68x under a floor belonging to a different cell.
#
# ⚠ AND THE PROPERTY THE COMPARISON ASSUMES IS EQUAL IS PRINTED, NOT ASSUMED:
# prefill TOKEN COUNT. A faster run over fewer tokens is not a faster prefill,
# and a name is not a specification -- the 2026-08-27 Q4_K/Q6_K mix-up cost a
# withdrawn 6.5x for exactly this reason. Cells whose arms disagree on token
# count are REFUSED, not averaged.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_q6k_reprice.ps1
#        [-Cycles 3] [-NGen 4] [-Threads 6] [-Quick]
# =====================================================================
param(
    [int]$Cycles  = 3,
    [int]$NGen    = 4,     # decode is not the subject; keep it small
    [int]$Threads = 6,
    [switch]$Quick,
    [string]$Stamp = ""
)

$ErrorActionPreference = "Stop"
$root  = "M:\New Drive\Desktop\Programming\stoke"
$exe   = "$root\build\win\stoke.exe"
$mdl   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$gr    = "$root\docs\superpowers\gate-reports"
if (-not $Stamp) { $Stamp = "{0}-m5-q6k-reprice{1}" -f (Get-Date -Format "yyyy-MM-dd"), $(if ($Quick) { "-quick" } else { "" }) }
$log    = "$gr\$Stamp-raw.txt"
$report = "$gr\$Stamp-report.md"

foreach ($p in @($exe, $mdl, $gr)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}
$newestSrc = (Get-ChildItem "$root\src" -Include *.c, *.h -Recurse |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1)
if ((Get-Item -LiteralPath $exe).LastWriteTime -lt $newestSrc.LastWriteTime) {
    throw "STOP: $exe is OLDER than $($newestSrc.Name) -- rebuild before measuring"
}

# The bank carries the BINARY'S CONTENT and the run's SHAPE. A rebuild must start
# a new bank rather than resume units measured against code that no longer
# exists -- the stale-artifact failure, which in a bank is invisible because the
# report names the new commit while grading the old binary.
$binKey = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 8)
$tmp = "$env:TEMP\stoke-q6k-reprice-n$NGen-t$Threads-$binKey$(if ($Quick) { '-quick' } else { '' })"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# P-cores only. Unpinned, E-core migration has been worth 5.8x on this box --
# larger than anything this rung is trying to measure.
$affMask = [IntPtr]0xFFF
(Get-Process -Id $PID).ProcessorAffinity = $affMask

$POINTS = @(
    @{ name = "p00";  file = "$root\tests\data\parity\p00.txt"       },
    @{ name = "p40";  file = "$root\tests\data\parity\p40.txt"       },
    @{ name = "long"; file = "$root\tests\data\ctx-profile-long.txt" }
)
if ($Quick) { $POINTS = @($POINTS[0], $POINTS[1]) }
foreach ($pt in $POINTS) { if (-not (Test-Path $pt.file)) { throw "missing prompt: $($pt.file)" } }

# ARMS. `ref2` is the NULL: the same kernel, a second process. Its ratio to ref1
# is this harness's resolution floor for that cell.
$ARMS = @(
    @{ tag = "ref";  env = "ref"     },
    @{ tag = "mad";  env = "maddubs" },
    @{ tag = "ref2"; env = "ref"     }
)

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Host $line     # Write-Host, never Write-Output: inside a function the
}                        # output stream IS the return value.

function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
}

# Returns @(prefill_tok, prefill_ms). BOTH, because the token count is what makes
# the ms comparable at all.
function PrefillOf($file, $kernel, $tag) {
    $o = "$tmp\$tag.txt"; $e = "$tmp\$tag-err.txt"; $d = "$tmp\$tag.done"
    if ((Test-Path $d) -and (Test-Path $e)) {
        $cached = [regex]::Match((Get-Content $e -Raw), 'prefill (\d+) tok in ([\d.]+) ms')
        if ($cached.Success) { return @([int]$cached.Groups[1].Value, [double]$cached.Groups[2].Value) }
    }
    Remove-Item -Force -ErrorAction SilentlyContinue $d
    $env:STOKE_Q6K_KERNEL = $kernel
    $p = Start-Process -FilePath $exe -ArgumentList @("batch", ('"' + $mdl + '"'), "--raw",
             "-n", "$NGen", "-f", ('"' + $file + '"'), "--seqs", "1",
             "--threads", "$Threads", "--kv", "f32", "--timing") `
         -RedirectStandardOutput $o -RedirectStandardError $e -NoNewWindow -PassThru -Wait
    $env:STOKE_Q6K_KERNEL = $null
    if ($p.ExitCode -ne 0) { throw "STOP: $tag exited $($p.ExitCode) -- a broken arm must stop, never report a slowdown" }
    $m = [regex]::Match((Get-Content $e -Raw), 'prefill (\d+) tok in ([\d.]+) ms')
    if (-not $m.Success) { throw "STOP: $tag -- no prefill line to parse; the instrument, not the engine" }
    New-Item -ItemType File -Force -Path $d | Out-Null
    return @([int]$m.Groups[1].Value, [double]$m.Groups[2].Value)
}

Say "Q6_K re-pricing. stamp $Stamp"
Say "exe $exe  commit $(git -C $root rev-parse --short HEAD)"
Say "bank $tmp"
Say "cycles $Cycles  n_gen $NGen  threads $Threads  affinity 0xFFF (P-cores)"
if ($Quick) { Say "*** -Quick: $($POINTS.Count) points, NOT the full rung ***" }

$rows = @()
for ($c = 1; $c -le $Cycles; $c++) {
    Say "CYCLE $c"
    # Point order AND arm order both rotate, so a thermal ramp cannot be read as
    # a gain for whichever arm consistently ran second.
    $pOrder = if ($c % 2 -eq 1) { $POINTS } else { $POINTS[($POINTS.Count - 1)..0] }
    foreach ($pt in $pOrder) {
        $aOrder = if ($c % 2 -eq 1) { $ARMS } else { $ARMS[($ARMS.Count - 1)..0] }
        foreach ($arm in $aOrder) {
            $r = PrefillOf $pt.file $arm.env "$($pt.name)-$($arm.tag)-c$c"
            $rows += [pscustomobject]@{ cycle = $c; point = $pt.name; arm = $arm.tag
                                        toks = $r[0]; ms = $r[1] }
            Say ("  c{0} {1,-5} {2,-5} {3} tok in {4} ms" -f $c, $pt.name, $arm.tag, $r[0], $r[1])
        }
    }
}
$env:STOKE_Q6K_KERNEL = $null

# ---- grading: EVERY flag the verdict reads is assigned here, above the block --
$cells = @()
foreach ($pt in $POINTS) {
    $mine = @($rows | Where-Object { $_.point -eq $pt.name })
    $tokSet = @($mine | ForEach-Object { $_.toks } | Sort-Object -Unique)
    $refMs  = Median @($mine | Where-Object { $_.arm -eq "ref"  } | ForEach-Object { $_.ms })
    $madMs  = Median @($mine | Where-Object { $_.arm -eq "mad"  } | ForEach-Object { $_.ms })
    $ref2Ms = Median @($mine | Where-Object { $_.arm -eq "ref2" } | ForEach-Object { $_.ms })
    $speed = if ($refMs -and $madMs)  { [math]::Round($refMs / $madMs, 4) }  else { $null }
    $null1 = if ($refMs -and $ref2Ms) { [math]::Round($refMs / $ref2Ms, 4) } else { $null }
    $nullPct = if ($null1) { [math]::Round([math]::Abs($null1 - 1.0) * 100.0, 2) } else { $null }
    $gainPct = if ($speed) { [math]::Round([math]::Abs($speed - 1.0) * 100.0, 2) } else { $null }
    $cells += [pscustomobject]@{
        point = $pt.name; toks = ($tokSet -join "/"); tokOk = ($tokSet.Count -eq 1)
        refMs = $refMs; madMs = $madMs; ref2Ms = $ref2Ms
        speed = $speed; nullR = $null1; nullPct = $nullPct
        resolved = ($null -ne $gainPct -and $null -ne $nullPct -and $gainPct -gt $nullPct)
    }
}
$allTokOk   = (@($cells | Where-Object { -not $_.tokOk }).Count -eq 0)
$overCeil   = @($cells | Where-Object { $null -ne $_.speed -and $_.speed -ge 1.71 })
$p2Held     = ($overCeil.Count -eq 0)
$firstCell  = $cells[0]
$lastCell   = $cells[$cells.Count - 1]
$p1Held     = ($null -ne $firstCell.speed -and $null -ne $lastCell.speed -and
               $lastCell.speed -gt $firstCell.speed)

$lines = @()
$lines += "# The Q6_K ``vpmaddubsw`` kernel, re-priced on a real prefill"
$lines += ""
if ($Quick) { $lines += "> !! **-Quick, NOT the full rung** -- $($POINTS.Count) points."; $lines += "" }
$lines += "7.68x on the isolated kernel at N=16. This is what it is worth END TO END."
$lines += ""
$lines += ("$Cycles cycles, point-order and arm-order both rotating, $Threads threads, " +
           "affinity **0xFFF (P-cores)**. One binary, ``STOKE_Q6K_KERNEL`` selecting the arm, " +
           "so no code-layout confound. Commit ``$(git -C $root rev-parse --short HEAD)``, " +
           "bank key ``$binKey``. Medians over cycles.")
$lines += ""
$lines += "| prompt | prefill tok | ref ms | maddubs ms | **speedup** | null (ref2/ref1) | resolved? |"
$lines += "|---|---|---|---|---|---|---|"
foreach ($cl in $cells) {
    $lines += ("| ``{0}`` | {1} | {2} | {3} | **{4}x** | {5}% | {6} |" -f
        $cl.point,
        $(if ($cl.tokOk) { $cl.toks } else { "**" + $cl.toks + " -- ARMS DISAGREE**" }),
        $cl.refMs, $cl.madMs, $cl.speed, $cl.nullPct,
        $(if ($cl.resolved) { "yes" } else { "**NO -- inside the null**" }))
}
$lines += ""
$lines += "## The pre-registered predictions"
$lines += ""
$lines += "| # | prediction | result |"
$lines += "|---|---|---|"
$lines += ("| P1 | speedup RISES with prompt length (16-tile share grows) | {0} |" -f
           $(if ($p1Held) { "**HELD** ($($lastCell.point) $($lastCell.speed)x > $($firstCell.point) $($firstCell.speed)x)" }
             else { "**FALSIFIED** ($($lastCell.point) $($lastCell.speed)x vs $($firstCell.point) $($firstCell.speed)x)" }))
$lines += ("| P2 | end-to-end prefill speedup is BELOW the 1.71x matvec-portion ceiling | {0} |" -f
           $(if ($p2Held) { "**HELD** -- no cell reaches 1.71x" }
             else { "**FALSIFIED** -- $($overCeil.Count) cell(s) at/above the ceiling; suspect the MEASUREMENT" }))
$lines += ("| -- | every arm of a cell prefilled the SAME token count | {0} |" -f
           $(if ($allTokOk) { "**yes**" } else { "**NO -- those cells are refused, not averaged**" }))
$lines += ""
$lines += "**The null column is the point.** ``ref2/ref1`` is the same kernel measured twice"
$lines += "as two processes, so it is what this harness cannot resolve for THAT cell. It is"
$lines += "reported per cell and never pooled: a between-process null on this box read 30% at"
$lines += "the smallest cell and 1.0% at the largest, and one pooled floor would bury a real"
$lines += "result under a floor belonging to a different cell. A speedup smaller than its own"
$lines += "cell's null reports NOTHING, however tidy it looks."
$lines += ""
$lines += "A cell above the 1.71x ceiling would mean the measurement is wrong, not the kernel"
$lines += "brilliant: prefill is strictly more than its matvecs, so an end-to-end number cannot"
$lines += "exceed the matvec-portion bound. That bound is ADDENDUM 19's 1.67x RE-DERIVED at the"
$lines += "kernel's MEASURED 83.68 GF/s rather than the 70 GF/s it assumed (0.0107 + 0.0907 ="
$lines += "0.1014 units against a 0.173 baseline); the asymptote, with Q6_K free, is 1.91x."
$lines += "The band was fixed BEFORE the first run -- re-deriving from better inputs beforehand"
$lines += "is getting it right, changing it afterwards is re-fitting. Predictions are recorded"
$lines += "as written; a falsified one stays on the page with the right measurement beside it."
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
Say ("VERDICT: P1 {0}, P2 {1}, token-counts {2}" -f
     $(if ($p1Held) { "HELD" } else { "FALSIFIED" }),
     $(if ($p2Held) { "HELD" } else { "FALSIFIED" }),
     $(if ($allTokOk) { "agree" } else { "DISAGREE" }))
foreach ($cl in $cells) {
    Say ("  {0,-5} {1,7} x   null {2}%   {3}" -f $cl.point, $cl.speed, $cl.nullPct,
         $(if ($cl.resolved) { "resolved" } else { "INSIDE THE NULL" }))
}
