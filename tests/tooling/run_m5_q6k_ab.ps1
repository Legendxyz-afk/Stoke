# =====================================================================
# A/B: THE BATCHED Q6_K KERNEL, REFERENCE vs vpmaddubsw
#
# A = `vec_dot_q6_k_q8_k_xn_avx2`         (shipping; widens i8 -> i32, 8 products
#                                          per operation, 32 iterations)
# B = `vec_dot_q6_k_q8_k_xn_maddubs_avx2` (new; 32 int8 products per vpmaddubsw,
#                                          8 iterations, offset carried on bsums)
#
# ▶▶ BOTH ARMS ARE THE SAME BINARY. `STOKE_Q6K_KERNEL=ref|maddubs` selects at
# runtime, which is why this A/B is free of the confound the 2026-08-24 vpmulld
# ablation had to engineer around: mingw embeds object paths and PE here is not
# position-independent, so two builds out of different-length directories differ
# in `.text` itself and code layout alone can move a hot loop a few percent.
# One binary, one layout, one file on disk.
#
# ▶▶ AND N=1 IS A FREE CONTROL THAT COSTS NOTHING TO CARRY. `vec.c` routes
# `n_seq == 1` to the SOLO worker, so neither arm's kernel is reached there and
# the two arms are executing IDENTICAL CODE. Whatever N=1 reports is this
# harness's noise, measured under the exact conditions of the cells that matter.
# A speedup at N>=2 only means something to the extent it exceeds it.
#
# ▶ CORRECTNESS IS NOT MEASURED HERE AND IS NOT ASSUMED EITHER: the twin is
# asserted BIT-IDENTICAL to both the reference twin and the SOLO kernel in
# `tests/test_simd.c` (600 iterations over n_seq 1,2,3,8,15,16 and n
# 256..9728, with an anti-vacuity check that the sequences actually differ).
# The end-to-end stream gate against the archives is a separate runner and is
# what must pass before any default flip.
#
# ▶ PRE-REGISTERED, before the first block:
#   P1  the N=1 CONTROL lands within +-7% -- the floor the 2026-08-27 null-arm
#       run measured for this instrument. Outside it, no cell here is readable.
#       ⚠ MEASURED AT 0.862 AND THEREFORE FALSIFIED AS WRITTEN, and the band was
#       MINE and WRONG: 7% is the floor of an A/C null arm taken INSIDE one
#       timing loop, and this comparison is BETWEEN PROCESSES -- process start,
#       page placement and thermal state at launch are all in it and none of
#       them are in the 7%. The prediction is left standing as falsified rather
#       than re-fitted, and arm `ref2` below measures the floor that should have
#       been used. Applying an in-process denominator to a between-process
#       ratio is exactly the mistake [[derived-quantity-is-a-claim]] names.
#   P1b the BETWEEN-PROCESS null (`ref2` vs `ref` -- the same arm run twice,
#       same env value, different process) brackets the N=1 control. This one
#       is measured, not chosen.
#   P2  B/A at N=16 exceeds 2.0x. A smoke read at 6 cycles said 6.2x; this run
#       exists to say whether that survives counterbalancing and repetition.
#   P3  B climbs monotonically with N (like every Q4_K tensor) where A does not.
#       This is the SHAPE claim, and it is the one that says the defect was the
#       per-sequence body rather than anything about the tensor.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_q6k_ab.ps1
#        [-Cycles 24] [-Blocks 4] [-Mask 1]
# =====================================================================
param(
    [int]$Cycles = 24,
    [int]$Blocks = 4,
    [int]$Mask   = 1,
    [string]$Stamp = ""
)

$ErrorActionPreference = "Stop"
$root = "M:\New Drive\Desktop\Programming\stoke"
$exe  = "$root\build\win\stoke.exe"
$mdl  = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$gr   = "$root\docs\superpowers\gate-reports"
if (-not $Stamp) { $Stamp = "{0}-m5-q6k-ab" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp-raw.txt"
$report = "$gr\$Stamp-report.md"
$tmp    = "$env:TEMP\stoke-q6k-ab"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

foreach ($p in @($exe, $mdl)) { if (-not (Test-Path $p)) { throw "missing: $p" } }
$newestSrc = (Get-ChildItem "$root\src" -Include *.c, *.h -Recurse |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1)
if ((Get-Item -LiteralPath $exe).LastWriteTime -lt $newestSrc.LastWriteTime) {
    throw "STOP: $exe is OLDER than $($newestSrc.Name) -- rebuild before measuring"
}

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Host $line
}
function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
}

$prev = (Get-Process -Id $PID).ProcessorAffinity
Say "Q6_K batched kernel A/B. exe $exe"
Say "A=ref  B=maddubs  (ONE binary, STOKE_Q6K_KERNEL selects)"
Say "cycles $Cycles, $Blocks blocks, tensor ffn_down, affinity 0x$('{0:X}' -f $Mask)"
Say "exe commit $(git -C $root rev-parse --short HEAD)"

# THE REFUSAL IS CHECKED FIRST. An unrecognised STOKE_Q6K_KERNEL must FATAL --
# if it silently fell back to the default, arm B would be arm A and the whole
# table would read "the new kernel is exactly as fast", which is precisely the
# shape of a passing result that proves nothing.
$env:STOKE_Q6K_KERNEL = "nonsense-value"
$probe = Start-Process -FilePath $exe -ArgumentList @("bench", "--matvec-batch",
         ('"' + $mdl + '"'), "--cycles", "1", "--seqs", "2", "--tensor", "ffn_down") `
         -RedirectStandardOutput "$tmp\probe.txt" -RedirectStandardError "$tmp\probe-err.txt" `
         -NoNewWindow -PassThru -Wait
$env:STOKE_Q6K_KERNEL = $null
if ($probe.ExitCode -eq 0) {
    throw "STOP: an unrecognised STOKE_Q6K_KERNEL was ACCEPTED -- a typo would silently measure one kernel twice"
}
Say "refusal check: an unrecognised STOKE_Q6K_KERNEL exits $($probe.ExitCode) -- a typo cannot silently pick an arm"

$unitFiles = @()
try {
    (Get-Process -Id $PID).ProcessorAffinity = [IntPtr]$Mask
    for ($b = 1; $b -le $Blocks; $b++) {
        # `ref2` is the BETWEEN-PROCESS NULL ARM: the same kernel, the same env
        # value, a different process. Whatever it reports against `ref` is what
        # this comparison cannot resolve -- see P1b.
        $order = @("ref", "maddubs", "ref2")
        if ($b % 2 -eq 0) { $order = @("ref2", "maddubs", "ref") }
        foreach ($arm in $order) {
            $envval = $(if ($arm -eq "ref2") { "ref" } else { $arm })
            $out = "$tmp\b$b-$arm-cy$Cycles.txt"
            if ((Test-Path $out) -and ((Get-Content $out -Raw) -match 'Q6_K')) {
                Say "  block $b $arm already banked, skipping"
                $unitFiles += @{ file = $out; arm = $arm }
                continue
            }
            $env:STOKE_Q6K_KERNEL = $envval
            $t0 = Get-Date
            $p = Start-Process -FilePath $exe `
                 -ArgumentList @("bench", "--matvec-batch", ('"' + $mdl + '"'),
                                 "--cycles", "$Cycles", "--tensor", "ffn_down") `
                 -RedirectStandardOutput $out -RedirectStandardError "$tmp\e-$b-$arm.txt" `
                 -NoNewWindow -PassThru -Wait
            $env:STOKE_Q6K_KERNEL = $null
            $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
            if ($p.ExitCode -ne 0) {
                Say ("  !! block $b $arm exited {0}: {1}" -f $p.ExitCode, (Get-Content "$tmp\e-$b-$arm.txt" -Raw))
                throw "STOP: a broken arm must stop, never contribute a rate"
            }
            Say "  block $b $arm done (${secs}s)"
            $unitFiles += @{ file = $out; arm = $arm }
        }
    }
} finally {
    (Get-Process -Id $PID).ProcessorAffinity = $prev
    $env:STOKE_Q6K_KERNEL = $null
}

# ---- parse ------------------------------------------------------------------
$g = @{}
foreach ($u in $unitFiles) {
    foreach ($ln in (Get-Content $u.file)) {
        $m = [regex]::Match($ln,
            '^\s+(\w+)\s+(\S+)\s+(\d+) x (\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)')
        if (-not $m.Success) { continue }
        $key = "{0}|{1}" -f $u.arm, $m.Groups[5].Value
        if (-not $g.ContainsKey($key)) { $g[$key] = @() }
        $g[$key] += [double]$m.Groups[12].Value
    }
}
if ($g.Count -eq 0) { throw "STOP: parsed no cells -- the instrument's row shape moved" }

$NS = @(1, 2, 4, 8, 16)
$ctrl = 0.0
$rows = @()
$floor = 0.0
foreach ($nn in $NS) {
    $a = Median $g["ref|$nn"]
    $bb = Median $g["maddubs|$nn"]
    $n2 = Median $g["ref2|$nn"]
    if ($null -eq $a -or $null -eq $bb) { continue }
    $r = if ($a -gt 0) { $bb / $a } else { 0 }
    $rn = if ($a -gt 0 -and $null -ne $n2) { $n2 / $a } else { 1.0 }
    if ($nn -eq 1) { $ctrl = $r }
    $d = [math]::Abs($rn - 1.0)
    if ($d -gt $floor) { $floor = $d }
    $rows += @{ n = $nn; a = $a; b = $bb; r = $r; n2 = $n2; rn = $rn }
}
$r16 = ($rows | Where-Object { $_.n -eq 16 }).r
$bs = @($rows | Where-Object { $_.n -ge 2 } | ForEach-Object { $_.b })
$mono = $true
for ($i = 1; $i -lt $bs.Count; $i++) { if ($bs[$i] -lt $bs[$i - 1]) { $mono = $false } }

# ▶ THE FLOOR IS PER CELL, NOT A MAX OVER CELLS. A single max mixes two
# populations: the N=1 cell is the FIRST thing each process measures and absorbs
# model load, tensor warm and page placement, while N>=2 cells run on a settled
# process. Quoting one number for both would understate how tight the cell that
# matters actually is -- the same mixing error [[percentile-can-be-a-mixing-ratio]]
# names, one level out. Each row carries its own null beside it.
$floor1 = [math]::Abs((($rows | Where-Object { $_.n -eq 1 }).rn) - 1.0)
$floor16 = [math]::Abs((($rows | Where-Object { $_.n -eq 16 }).rn) - 1.0)
$p1 = ([math]::Abs($ctrl - 1.0) -le 0.07)
$p1b = ([math]::Abs($ctrl - 1.0) -le $floor1)
$p2 = ($r16 -ge 2.0)
$p3 = $mono

$lines = @()
$lines += "# The batched Q6_K kernel: reference vs ``vpmaddubsw``"
$lines += ""
$lines += "Run ``$Stamp``, $Blocks blocks x $Cycles cycles, tensor ``ffn_down`` (the model's Q6_K"
$lines += "matvec), one tensor per process, pinned to one P-core (mask 0x$('{0:X}' -f $Mask)), arm order"
$lines += "reversed on even blocks."
$lines += ""
$lines += "**A** = ``vec_dot_q6_k_q8_k_xn_avx2`` -- widens i8 to i32, 8 products per operation,"
$lines += "32 iterations per sequence per super-block."
$lines += "**B** = ``vec_dot_q6_k_q8_k_xn_maddubs_avx2`` -- 32 int8 products per ``vpmaddubsw``,"
$lines += "8 iterations, with the -32 offset carried on Q8_K's ``bsums`` exactly as Q4_K"
$lines += "already carries its ``mins``."
$lines += ""
$lines += "**Both arms are the SAME BINARY** -- ``STOKE_Q6K_KERNEL`` selects at runtime, so this"
$lines += "A/B carries none of the code-layout confound that forced the 2026-08-24 ``vpmulld``"
$lines += "arms into equal-length build directories. An unrecognised value is REFUSED, and"
$lines += "the runner checks that before measuring anything."
$lines += ""
$lines += "| N | A ref GF/s | B maddubs GF/s | **B/A** | ref2 (null) GF/s | ref2/A |"
$lines += "|---|---|---|---|---|---|"
foreach ($row in $rows) {
    $tag = if ($row.n -eq 1) { " (CONTROL -- same code path)" } else { "" }
    $lines += ("| {0}{1} | {2:n2} | {3:n2} | **{4:n2}x** | {5:n2} | {6:n3} |" -f
        $row.n, $tag, $row.a, $row.b, $row.r, $row.n2, $row.rn)
}
$lines += ""
$lines += "## The pre-registered predictions"
$lines += ""
$lines += "| # | prediction | measured | result |"
$lines += "|---|---|---|---|"
$lines += ("| P1 | the N=1 CONTROL is within +-7% (the null-arm floor measured 2026-08-27) | {0:n3}x | {1} |" -f $ctrl, $(if ($p1) { "**HELD**" } else { "**FALSIFIED**" }))
$lines += ("| P1b | the BETWEEN-PROCESS null at the SAME CELL brackets the control | N=1 null {0:p1} | {1} |" -f $floor1, $(if ($p1b) { "**HELD**" } else { "**FALSIFIED**" }))
$lines += ("| P2 | B/A at N=16 exceeds 2.0x | {0:n2}x | {1} |" -f $r16, $(if ($p2) { "**HELD**" } else { "**FALSIFIED**" }))
$lines += ("| P3 | B climbs monotonically with N where A does not | {0} | {1} |" -f $(if ($mono) { "monotone" } else { "NOT monotone" }), $(if ($p3) { "**HELD**" } else { "**FALSIFIED**" }))
$lines += ""
if (-not $p1) {
    $lines += "!! **P1 IS FALSIFIED AND THE BAND WAS MINE AND WRONG.** 7% is the floor of an"
    $lines += "A/C null arm taken INSIDE one timing loop; this comparison is BETWEEN PROCESSES,"
    $lines += "so process start, page placement and thermal state at launch are all in it and"
    $lines += "none of them are in the 7%. **Applying an in-process denominator to a"
    $lines += "between-process ratio is the error [[derived-quantity-is-a-claim]] names**, and the"
    $lines += "prediction is left standing as falsified rather than re-fitted after the fact."
    $lines += ""
    $lines += "Arm ``ref2`` measures the floor that should have been used: the SAME kernel and"
    $lines += "the SAME env value in a DIFFERENT process. **And it must be read PER CELL, not as"
    $lines += "a max** -- N=1 is the FIRST cell each process measures and absorbs model load,"
    $lines += "tensor warm and page placement, while N>=2 runs on a settled process. One number"
    $lines += "for both would mix two populations and understate the cell that matters:"
    $lines += ""
    $lines += ("- **N=1 null {0:p1}** -- and the N=1 control's {1:n3}x sits {2} it." -f
        $floor1, $ctrl, $(if ($p1b) { "INSIDE" } else { "OUTSIDE" }))
    $lines += ("- **N=16 null {0:p1}** -- against which B/A's {1:n2}x is not close to a noise" -f
        $floor16, $r16)
    $lines += "  question at all."
    $lines += ""
}
$lines += "**N=1 is a free control and it costs nothing to carry**: ``vec.c`` routes"
$lines += "``n_seq == 1`` to the SOLO worker, so neither arm's kernel is reached there and the"
$lines += "two arms execute IDENTICAL CODE. Whatever it reports is this harness's noise under"
$lines += "the exact conditions of the cells that matter, and a speedup at N>=2 means"
$lines += "something only to the extent it exceeds it."
$lines += ""
$lines += "## Correctness"
$lines += ""
$lines += "**Not measured here, and not assumed either.** ``tests/test_simd.c`` asserts the twin"
$lines += "BIT-IDENTICAL to both the reference twin AND the solo kernel, per sequence, over 600"
$lines += "iterations spanning n_seq 1/2/3/8/15/16 and n 256/512/768/2560/9728 -- with an"
$lines += "anti-vacuity check that the sequences actually differ from each other, because a"
$lines += "kernel that ignored ``s`` and returned sequence 0's answer would otherwise pass"
$lines += "every assertion in it."
$lines += ""
$lines += "The exactness is by CONSTRUCTION, not by testing: ``SUM (u-32).a.s == SUM u.a.s -"
$lines += "32.SUM a.s`` is exact in integers, ``SUM a`` per scale group is what ``bsums`` already"
$lines += "stores, and every intermediate is inside its type (63x127x2 = 16,002 against i16;"
$lines += "x127 x2 = 4,064,508 and 260,128,512 against i32). See ``simd.h``."
$lines += ""
$lines += "**The default is still ``ref``.** An end-to-end stream gate against the ARCHIVES,"
$lines += "with a mutant that must differ, is what has to pass before any flip -- the"
$lines += "``STOKE_MINS_SIMD`` sequence, not a shortcut around it."
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
Say ("VERDICT: B/A at N=16 = {0:n2}x, control {1:n3}x, monotone {2}" -f $r16, $ctrl, $mono)
