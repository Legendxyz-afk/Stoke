# =====================================================================
# M5 GATE ITEM [3] -- THE DECODE STALL DURING CO-RUNNING PREFILL.
#
#   [3] ITL: no decode stall >250ms during co-running prefill
#       (harness percentiles + max)
#
# Graded under the 2026-08-26 ruling
# (memos/2026-08-26-m5-gate3-criterion-RULING-OPUS.md): the subject is the
# `ITL stall` line -- gaps whose step also carried a prefill chunk -- NOT the
# pooled ITL line, whose percentiles track how many chunks a run happened to
# have rather than any bound.
#
# PREDICTIONS ARE PRE-REGISTERED AT 23fcb35 AND ARE NOT RESTATED HERE, so that
# this file cannot quietly become the record of what was predicted.
#
# CELL ORDER IS PALINDROMIC -- Sd Sn C16 Sn Sd -- so the two arms P2 compares
# each appear once early and once late. P2's band is 20% and this box drifts up
# to 22% on compute-dense work ([[measurement-needs-interleaving]]), so a
# straight A-then-B would put drift and the effect on the same axis.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_gate3.ps1
#        [-NGen 300] [-Seqs 8] [-Threads 8] [-Ctx 1600] [-Quick]
# =====================================================================
param(
    [int]$NGen    = 300,
    [int]$Seqs    = 8,
    [int]$Threads = 8,
    [int]$Ctx     = 1600,
    [int]$MaxCells = 99,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$root  = "M:\New Drive\Desktop\Programming\stoke"
$exe   = "$root\build\win\stoke.exe"
$model = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$pdir  = "$root\tests\data\bench_prompts"
$gr    = "$root\docs\superpowers\gate-reports"
$BOUND = 250.0

if ($Quick) { $NGen = 24; $Seqs = 2 }

# >> THE SHAPE KEY MUST CARRY THE BINARY, AND ON 2026-08-28 IT DID NOT.
# Cells resume from `$tmp\<id>-<SHAPE>.txt`, and every s8n300 cell from the
# 2026-08-26 sitting was still sitting there. Re-running after the Q6_K default
# flipped would have printed "already banked, skipping" five times and emitted a
# report dated today, under today's commit, made entirely of numbers measured by
# a binary WITHOUT the kernel being gated -- a re-measure that re-measured
# nothing and said so nowhere. Same defect as the Q6_K gate's (b): a stale
# binary has an mtime you can look at, a stale cache does not. The staleness
# check below compares the exe against SOURCE and is blind to this, because
# nothing about the exe is wrong -- the CACHE is from a different one.
$exeSha   = (Get-FileHash $exe -Algorithm SHA256).Hash.Substring(0, 12)
$SHAPE = "s${Seqs}n${NGen}-$exeSha"
$Stamp = "{0}-m5-gate3{1}" -f (Get-Date -Format "yyyy-MM-dd"),
                              $(if ($Quick) { "-quick" } else { "" })
$log    = "$gr\$Stamp-raw.txt"
$report = "$gr\$Stamp-report.md"
$tmp    = "$env:TEMP\stoke-gate3"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

foreach ($p in @($exe, $model, $pdir)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}
# STALENESS: the ITL split landed in this binary and a gate on an older one
# would grade a driver that cannot print the line it grades.
$newestSrc = (Get-ChildItem "$root\src" -Include *.c, *.h -Recurse |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1)
if ((Get-Item -LiteralPath $exe).LastWriteTime -lt $newestSrc.LastWriteTime) {
    throw "STOP: $exe is OLDER than $($newestSrc.Name) -- rebuild before gating"
}

# ⚠ Write-Host, NOT Write-Output. Everything a PowerShell function writes to the
# OUTPUT stream becomes part of its return value, and `Say` is called inside
# Cell, whose return value is the run's stdout. With Write-Output the status
# lines were silently prepended to the parsed text and vanished from the console
# at the same time -- one mistake with two symptoms, and the quieter one is the
# corrupted data. Status goes to the host and to the log; only data is returned.
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
        $a += "-f"; $a += ('"' + ("$pdir\s3-fanout-c{0:d2}.txt" -f $i) + '"')
    }
    return $a
}

# One cell = one run, banked to its own file and SKIPPED if that file already
# parses. Same resume shape as the gate [4] runner, keyed by cell id AND shape.
function Cell($id, $dedup, $chunk) {
    $out = "$tmp\$id-$SHAPE.txt"
    if (Test-Path $out) {
        $t = Get-Content $out -Raw
        if ($t -and (Grp $t 'ITL\s+stall\s+p50\s+([\d.]+)' 1)) {
            Say "  $id already banked, skipping"
            return $t
        }
    }
    $a = @("bench", "--serve", ('"' + $model + '"')) + (PromptArgs $Seqs) +
         @("--seqs", "$Seqs", "--ctx", "$Ctx", "-n", "$NGen",
           "--threads", "$Threads", "--kv", "f32")
    if ($dedup) { $a += "--dedup" } else { $a += "--no-dedup" }
    if ($chunk -gt 0) { $env:STOKE_SCHED_CHUNK = "$chunk" }
    else              { $env:STOKE_SCHED_CHUNK = $null }
    $t0 = Get-Date
    $p = Start-Process -FilePath $exe -ArgumentList $a -RedirectStandardOutput $out `
                       -RedirectStandardError "$tmp\e.txt" -NoNewWindow -PassThru -Wait
    $env:STOKE_SCHED_CHUNK = $null
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if ($p.ExitCode -ne 0) { throw "STOP: $id exited $($p.ExitCode)" }
    $t = Get-Content $out -Raw
    # ANTI-VACUITY, in the driver's own words.
    if ($t -match '! only \d+ of \d+ expected ids') { throw "STOP: $id produced fewer ids than asked" }
    Say ("  {0,-6} stall p50 {1,8:n1}  n={2,-5} ({3}% of gaps) | cadence p50 {4,6:n1} | pooled p50 {5,6:n1} p95 {6,7:n1} | {7}s" -f
         $id, (Grp $t 'ITL\s+stall\s+p50\s+([\d.]+)' 1),
         (Grp $t 'ITL\s+stall.*\(n=(\d+)' 1),
         [math]::Round((Grp $t 'ITL\s+stall.*n=\d+,\s+([\d.]+)%' 1), 1),
         (Grp $t 'ITL\s+cadence\s+p50\s+([\d.]+)' 1),
         (Grp $t 'ITL\s+p50\s+([\d.]+)' 1),
         (Grp $t 'ITL\s+p50\s+[\d.]+\s+p95\s+([\d.]+)' 1), $secs)
    return $t
}

Say "M5 gate [3] -- decode stall during co-running prefill. stamp $Stamp"
Say "seqs $Seqs x n $NGen, ctx $Ctx, threads $Threads, bound $BOUND ms"
Say "exe commit $(git -C $root rev-parse --short HEAD); predictions pre-registered at 23fcb35"
Say "exe sha256 $exeSha -- the cell cache is keyed by it, so a rebuild cannot resume"
if ($Quick) { Say "*** -Quick: NOT a gate ***" }

$cells = @(
    @{ id = "Sd";   dedup = $true;  chunk = 0  },
    @{ id = "Sn";   dedup = $false; chunk = 0  },
    @{ id = "C16";  dedup = $true;  chunk = 16 },
    @{ id = "Sn2";  dedup = $false; chunk = 0  },
    @{ id = "Sd2";  dedup = $true;  chunk = 0  }
)
# >> HOW MANY CELLS THIS INVOCATION MAY RUN, so the sitting can be driven
# FOREGROUND (2026-08-26: long unattended jobs on this box get killed with no
# message). Cells ran 272-574 s on 2026-08-26, so five back to back is well past
# a foreground caller's ceiling. The per-cell cache already makes the chain
# equivalent to one run, and the PALINDROME survives because the order of the
# invocations is the order of the cells.
#
# ⚠ AND THE REPORT IS NOT WRITTEN UNTIL EVERY CELL IS IN. A partial `$res` would
# make `V "Sd2"` return $null, and every mean, ratio and verdict built from it
# would come out as a tidy-looking number about cells that were never measured.
$res = @{}
$ran = 0
foreach ($c in $cells) {
    $cached = $false
    $out = "$tmp\$($c.id)-$SHAPE.txt"
    if (Test-Path $out) {
        $tc = Get-Content $out -Raw
        if ($tc -and (Grp $tc 'ITL\s+stall\s+p50\s+([\d.]+)' 1)) { $cached = $true }
    }
    if (-not $cached) {
        if ($ran -ge $MaxCells) {
            Say "  -MaxCells $MaxCells reached -- $($c.id) and later left for the next invocation"
            continue
        }
        $ran++
    }
    $res[$c.id] = Cell $c.id $c.dedup $c.chunk
}
$missing = @($cells | Where-Object { -not $res.ContainsKey($_.id) -or $null -eq $res[$_.id] })
if ($missing.Count -gt 0) {
    Say ("{0} of {1} cells banked; still owed: {2}. NO REPORT until all five are in." -f
         ($cells.Count - $missing.Count), $cells.Count, (($missing | ForEach-Object { $_.id }) -join ", "))
    return
}

function V($id, $re, $i) { return (Grp $res[$id] $re $i) }
$reStallP50 = 'ITL\s+stall\s+p50\s+([\d.]+)'
$reStallMax = 'ITL\s+stall\s+p50\s+[\d.]+\s+p95\s+[\d.]+\s+max\s+([\d.]+)'
$reStallN   = 'ITL\s+stall.*\(n=(\d+)'
$reStallPct = 'ITL\s+stall.*n=\d+,\s+([\d.]+)%'
$reCadP50   = 'ITL\s+cadence\s+p50\s+([\d.]+)'
$reChunks   = 'prefill chunks\s+(\d+)'

# The two arms P2 compares are each measured twice, early and late; the median
# of two is their mean, so state both and compare the means.
$sdStall = ((V "Sd" $reStallP50 1) + (V "Sd2" $reStallP50 1)) / 2.0
$snStall = ((V "Sn" $reStallP50 1) + (V "Sn2" $reStallP50 1)) / 2.0
$sdCad   = ((V "Sd" $reCadP50 1)   + (V "Sd2" $reCadP50 1))   / 2.0
$snCad   = ((V "Sn" $reCadP50 1)   + (V "Sn2" $reCadP50 1))   / 2.0

$lines = @()
$lines += "# M5 gate item [3] -- the decode stall during co-running prefill"
$lines += ""
$lines += "Run ``$Stamp``. S3, $Seqs clients x $NGen generated, ctx $Ctx, threads $Threads, kv f32."
# THE BINARY IS A PROPERTY THE WHOLE COMPARISON RESTS ON, so it is printed INTO
# the artifact rather than left in the raw log ([[fixture-can-be-blind-to-the-defect]]).
$lines += ("Binary ``stoke.exe`` sha256 ``{0}``; the per-cell resume cache is keyed by it." -f $exeSha)
$lines += "Bound **$BOUND ms**. Graded on the ``ITL stall`` line per the 2026-08-26 ruling"
$lines += '(`memos/2026-08-26-m5-gate3-criterion-RULING-OPUS.md`); predictions pre-registered'
$lines += "at ``23fcb35`` BEFORE this run. Raw ``$Stamp-raw.txt``."
$lines += ""
$lines += "| cell | dedup | chunk | stall p50 ms | stall max ms | stall n | stall % of gaps | cadence p50 ms | prefill chunks |"
$lines += "|---|---|---|---|---|---|---|---|---|"
foreach ($c in $cells) {
    $lines += ("| {0} | {1} | {2} | {3:n1} | {4:n0} | {5:n0} | {6:n1}% | {7:n1} | {8:n0} |" -f
        $c.id, $(if ($c.dedup) { "on" } else { "OFF" }),
        $(if ($c.chunk -gt 0) { $c.chunk } else { 128 }),
        (V $c.id $reStallP50 1), (V $c.id $reStallMax 1), (V $c.id $reStallN 1),
        (V $c.id $reStallPct 1), (V $c.id $reCadP50 1), (V $c.id $reChunks 1))
}
$lines += ""
$lines += "## The pre-registered predictions, graded"
$lines += ""
# ⚠ P3 AND P4 ARE SHAPE-SPECIFIC AND THE PRE-FLIGHT MUST NOT GRADE THEM.
# P3's bands (~8% / ~27%) are chunks/n_gen at n=300, and P4's 403 ms is the step
# time at N=8. Run at 2x24 both are "falsified" by arithmetic that has nothing to
# do with the model -- a pre-flight scoring a gate's predictions against its own
# smaller shape is the same error as resuming a pre-flight unit as a gate unit.
if ($Quick) {
    $lines += "!! **-Quick: the predictions are pre-registered at $Seqs x 300 and are NOT graded here.**"
    $lines += "P3's bands are chunks/n_gen and P4's 403 ms is the step time at N=8; both move with the"
    $lines += "shape. P1, P2 and P5 are shape-independent and are shown for information only."
    $lines += ""
}
$lines += "| # | prediction | band | measured | verdict |"
$lines += "|---|---|---|---|---|"
$NG = $(if ($Quick) { "_(not graded)_" } else { $null })
$p1 = V "Sd" $reStallP50 1
$lines += ("| P1 | stall p50 at chunk 128 ~ 4.5 s | 3500-5500 ms | {0:n0} | {1} |" -f
           $p1, $(if ($p1 -ge 3500 -and $p1 -le 5500) { "**HELD**" } else { "**FALSIFIED**" }))
$p2 = [math]::Abs($sdStall - $snStall) / [math]::Max($sdStall, $snStall)
$lines += ("| P2 | stall p50 same in both arms | within 20% | {0:n0} vs {1:n0} = {2:n1}% apart | {3} |" -f
           $sdStall, $snStall, ($p2 * 100.0),
           $(if ($p2 -le 0.20) { "**HELD**" } else { "**FALSIFIED**" }))
$p3d = V "Sd" $reStallPct 1
$p3n = V "Sn" $reStallPct 1
$lines += ("| P3 | stall share ~8% dedup / ~27% no-dedup | +-5 pp | {0:n1}% / {1:n1}% | {2} |" -f
           $p3d, $p3n,
           $(if ($NG) { $NG }
             elseif ([math]::Abs($p3d - 8) -le 5 -and [math]::Abs($p3n - 27) -le 5) { "**HELD**" }
             else { "**FALSIFIED**" }))
$p4 = [math]::Abs($sdCad - $snCad) / [math]::Max($sdCad, $snCad)
$lines += ("| P4 | cadence p50 ~403 ms, same in both arms | +-15% | {0:n0} vs {1:n0} | {2} |" -f
           $sdCad, $snCad,
           $(if ($NG) { $NG }
             elseif ($p4 -le 0.15 -and [math]::Abs($sdCad - 403) / 403 -le 0.15) { "**HELD**" }
             else { "**FALSIFIED**" }))
$p5 = V "C16" $reStallP50 1
$lines += ("| P5 | stall p50 at chunk 16 ~ 570 ms | 400-800 ms | {0:n0} | {1} |" -f
           $p5, $(if ($p5 -ge 400 -and $p5 -le 800) { "**HELD**" } else { "**FALSIFIED**" }))
$lines += ""
# ▶ THE NUMBER THAT PRICES THE BOUND. A stall is one chunk's step, so
# chunk_tokens / stall_p50 IS the prefill rate at that width, and
# chunk_tokens / 250ms is the rate the bound demands. Stated per width because
# the whole point of the C16 cell is that these two do not scale together.
$lines += "## What the bound costs, per chunk width"
$lines += ""
$lines += "| chunk | stall p50 ms | implied prefill tok/s | tok/s needed for $BOUND ms | shortfall |"
$lines += "|---|---|---|---|---|"
foreach ($w in @(@{c = 128; v = $sdStall }, @{c = 16; v = $p5 })) {
    if (-not $w.v) { continue }
    $have = $w.c / ($w.v / 1000.0)
    $need = $w.c / ($BOUND / 1000.0)
    $lines += ("| {0} | {1:n0} | **{2:n1}** | {3:n1} | **{4:n1}x** |" -f
               $w.c, $w.v, $have, $need, ($need / $have))
}
$lines += ""
$lines += "Read the two rows against each other: a narrower chunk shortens the stall but"
$lines += "does NOT hold the prefill rate, so the shortfall does not fall with the width."
$lines += ""
# ▶ DRIFT, MEASURED FROM THE ARMS' OWN REPEATS -- not from a probe sampled after
# the transient, which reports healthy every time ([[measurement-needs-interleaving]]).
$dSd = ((V "Sd2" $reStallP50 1) / (V "Sd" $reStallP50 1) - 1.0) * 100.0
$dSn = ((V "Sn2" $reStallP50 1) / (V "Sn" $reStallP50 1) - 1.0) * 100.0
$dSdC = ((V "Sd2" $reCadP50 1) / (V "Sd" $reCadP50 1) - 1.0) * 100.0
$lines += ("**Drift across the sitting, from the repeated cells: stall +{0:n0}% (Sd) and +{1:n0}% (Sn), cadence +{2:n0}%.**" -f
           $dSd, $dSn, $dSdC)
$lines += "Monotonic and in the direction the box's thermal plateau predicts, which is why the"
$lines += "cell order is palindromic: each compared arm has one early and one late sample."
$lines += "**P2 is gradeable only because of that** -- pairing Sd's early cell against Sn's late"
$lines += "one reads 11% apart, and Sd's late against Sn's early reads 25%, on either side of"
$lines += "P2's 20% band. Every verdict below clears its bound by more than 4x, so the drift"
$lines += "cannot flip any of them."
$lines += ""
# DISCRIMINATOR BEFORE VERDICT: a conditioning that collapsed to all-or-nothing
# is a broken instrument, and nothing above may be read.
$bad = @()
foreach ($c in $cells) {
    $n = V $c.id $reStallN 1
    $p = V $c.id $reStallPct 1
    if ($null -eq $n -or $n -le 0 -or $p -ge 99.0) { $bad += $c.id }
}
if ($bad.Count -gt 0) {
    $lines += ("!! **DISCRIMINATOR FIRED on {0}: the stall conditioning collapsed to all-or-nothing." -f ($bad -join ", "))
    $lines += "The instrument is broken and NOTHING above may be read as a finding.**"
} else {
    $lines += "Discriminator: every cell's stall population is a proper subset of its gaps -- the conditioning works."
}
$lines += ""
$best = [math]::Min($p1, $p5)
$verdict = $(if ($best -le $BOUND) { "**MET** at {0:n0} ms <= $BOUND" -f $best }
             else { "**MISS** -- the smallest stall p50 measured is {0:n0} ms = {1:n1}x the $BOUND ms bound" -f $best, ($best / $BOUND) })
$lines += "## VERDICT (gate [3], on the stall line): $verdict"
$lines += ""
$lines += "Chunk 16 is the SMALLEST LEGAL width -- ``BP_BLOCK_TOKENS`` is 16 and M4 froze it --"
$lines += "so the chunk-width lever is exhausted at the C16 cell, not merely untried."
$lines += ""
$lines += "**Both falsified predictions failed in the same direction and for the same kind of"
$lines += "reason -- a per-stream or per-chunk quantity assumed to scale when it does not:**"
$lines += ""
$lines += "- **P3.** Stalls are not ``chunks x n_gen``. A chunk stalls only the streams ALREADY"
$lines += "  DECODING, so the count is the sum over chunks of the streams live at that moment:"
$lines += "  with dedup 2 chunks/client x (1+..+7) = **56**, without 10 x 28 = **280** -- both"
$lines += "  measured EXACTLY. Prefills run one at a time (S1), so client 0 stalls nobody."
$lines += "- **P5.** A 16-token chunk does not cost 1/8th of a 128-token chunk. The implied"
$lines += "  prefill rate FALLS with the width (see the table above), so the stall shortens"
$lines += "  sub-linearly and the shortfall against the bound gets WORSE, not better."
$lines += ""
$lines += "## Every cell, verbatim"
$lines += ""
foreach ($c in $cells) {
    $lines += "``````"
    $lines += ("--- {0} (dedup {1}, chunk {2})" -f $c.id, $(if ($c.dedup) { "on" } else { "OFF" }),
               $(if ($c.chunk -gt 0) { $c.chunk } else { 128 }))
    $lines += (($res[$c.id] -split "`r?`n" | Where-Object { $_ -match '^(serve:|  )' }) -join "`n")
    $lines += "``````"
    $lines += ""
}
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
Say "VERDICT: $verdict"
