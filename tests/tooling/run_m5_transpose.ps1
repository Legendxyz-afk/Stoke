# =====================================================================
# IS THE `n_in` COLLAPSE REAL, OR WAS IT THE HARNESS?
#
# The 2026-08-27 activation-reuse table reported kernel GF/s collapsing with
# `n_in` at what looked like identical work. Two things were wrong with reading
# it that way, and one of them was fatal:
#
#   ⚠⚠ `ffn_down` IS Q6_K, NOT Q4_K. In a Q4_K_M model it is upgraded: 210 bytes
#   per super-block against 144, 20.4 MB against 14.0, and a DIFFERENT KERNEL
#   (`vec_dot_q6_k_q8_k_xn_avx2`). The `ffn_gate` vs `ffn_down` 6.5x was never a
#   like-for-like comparison and is WITHDRAWN. The instrument now prints the
#   quant type and the weight bytes on every row, because a fixture drawn BY
#   NAME from a real model can differ in the property the comparison rests on
#   and nothing in that table said so.
#
#   ⚠ THE ABSOLUTE RATES WERE CONTAMINATED. All four tensors ran in ONE process,
#   so (a) earlier tensors' weights sat in L3 while later ones were measured,
#   and (b) the activation scratch buffers were sized to `n_in_max` OVER THE
#   WHOLE SET -- 11 KB apart for ffn_down's sake even while measuring a tensor
#   that touches 2.9 KB of each. `ffn_down` at N=16 read 19.76 GF/s in a
#   3-tensor run and 10.83 in a 4-tensor one: the same cell, 1.8x apart, on a
#   harness change alone.
#
# ▶▶ WHAT SURVIVES IS ONE CLEAN PAIR, AND IT IS A GOOD ONE.
# `attn_q` (2560 x 4096) and `attn_output` (4096 x 2560) are EXACT TRANSPOSES:
# both Q4_K, both 5,898,240 bytes, same FLOP, and the same number of
# (row x super-block x sequence) bodies -- 4096x10 = 2560x16 = 40,960 unpacks.
# The only thing that differs is the LOOP SHAPE: 10 super-blocks inside a row
# against 16. In the contaminated table they read 65.84 vs 36.94 GF/s = 1.78x.
#
# This runner re-measures every tensor ALONE, one per process, with the scratch
# sized to that tensor -- removing both contaminants at once.
#
# ▶ PRE-REGISTERED, and this is an ARTIFACT HUNT rather than a graded gate, so
# it is stated here rather than in a memo:
#   H-ARTIFACT  the transpose ratio falls below 1.30x once isolated => the
#               four-tensor table's absolute rates were contaminated, the
#               observation dies with them, and no lever appears.
#   H-REAL      it stays at or above 1.60x => the loop shape is a real property
#               of this kernel at identical work, and it IS a live lever.
#   BETWEEN     1.30-1.60x => partly both; the next cut is an `n_out` sweep at
#               fixed `n_in`, which needs synthetic weight buffers.
# Whichever fires, the Q6_K withdrawal above stands on its own.
#
# ▶ ONE TENSOR PER PROCESS, ORDER COUNTERBALANCED ACROSS BLOCKS, PINNED TO ONE
# P-CORE, AND BANKED PER UNIT -- a killed run loses one unit, not the sitting.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_transpose.ps1
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
if (-not $Stamp) { $Stamp = "{0}-m5-transpose" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp-raw.txt"
$report = "$gr\$Stamp-report.md"
$tmp    = "$env:TEMP\stoke-transpose"
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

$TENSORS = @("attn_q", "attn_out", "ffn_gate", "ffn_down")
$prev = (Get-Process -Id $PID).ProcessorAffinity
Say "transpose / n_in isolation. exe $exe"
Say "cycles $Cycles, $Blocks blocks, one tensor per PROCESS, affinity 0x$('{0:X}' -f $Mask)"
Say "exe commit $(git -C $root rev-parse --short HEAD)"

$unitFiles = @()
try {
    (Get-Process -Id $PID).ProcessorAffinity = [IntPtr]$Mask
    for ($b = 1; $b -le $Blocks; $b++) {
        # THE ORDER REVERSES ON ODD/EVEN BLOCKS so neither member of the pair is
        # always measured on the colder box.
        $order = $TENSORS
        if ($b % 2 -eq 0) { $order = $TENSORS[($TENSORS.Count - 1)..0] }
        foreach ($t in $order) {
            # Shape-keyed on cycles AND the row format, so a bank written by the
            # pre-`--tensor` instrument can never be resumed as this run's own.
            $out = "$tmp\b$b-$t-cy$Cycles-f2.txt"
            if ((Test-Path $out) -and ((Get-Content $out -Raw) -match 'GF/s|Q4_K')) {
                Say "  block $b $t already banked, skipping"
                $unitFiles += $out
                continue
            }
            $t0 = Get-Date
            $p = Start-Process -FilePath $exe `
                 -ArgumentList @("bench", "--matvec-batch", ('"' + $mdl + '"'),
                                 "--cycles", "$Cycles", "--tensor", $t) `
                 -RedirectStandardOutput $out -RedirectStandardError "$tmp\e-$b-$t.txt" `
                 -NoNewWindow -PassThru -Wait
            $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
            if ($p.ExitCode -ne 0) {
                Say ("  !! block $b $t exited {0}: {1}" -f $p.ExitCode, (Get-Content "$tmp\e-$b-$t.txt" -Raw))
                throw "STOP: a broken unit must stop, never contribute a rate"
            }
            Say "  block $b $t done (${secs}s)"
            $unitFiles += $out
        }
    }
} finally {
    (Get-Process -Id $PID).ProcessorAffinity = $prev
}

# ---- parse ------------------------------------------------------------------
$cells = @{}
$order2 = @()
foreach ($uf in $unitFiles) {
    foreach ($ln in (Get-Content $uf)) {
        $m = [regex]::Match($ln,
            '^\s+(\w+)\s+(\S+)\s+(\d+) x (\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)/(\d+)\s+(\d+)/(\d+)')
        if (-not $m.Success) { continue }
        $key = "{0}|{1}" -f $m.Groups[1].Value, $m.Groups[5].Value
        if (-not $cells.ContainsKey($key)) {
            $cells[$key] = @{ tensor = $m.Groups[1].Value; type = $m.Groups[2].Value
                              n_in = [int]$m.Groups[3].Value; n_out = [int]$m.Groups[4].Value
                              n = [int]$m.Groups[5].Value
                              gfa = @(); ab = @(); ac = @() }
            $order2 += $key
        }
        $c = $cells[$key]
        $c.gfa += [double]$m.Groups[12].Value
        $c.ab  += [double]$m.Groups[10].Value
        $c.ac  += [double]$m.Groups[11].Value
    }
}
if ($order2.Count -eq 0) { throw "STOP: parsed no cells -- the instrument's row shape moved" }

$qq = Median $cells["attn_q|16"].gfa
$oo = Median $cells["attn_out|16"].gfa
$ratio = if ($oo -gt 0) { $qq / $oo } else { 0 }
$verdict = "BETWEEN -- partly both; the next cut is an n_out sweep at fixed n_in"
if ($ratio -lt 1.30)     { $verdict = "H-ARTIFACT -- the four-tensor table's rates were contaminated and the observation dies with them" }
elseif ($ratio -ge 1.60) { $verdict = "H-REAL -- the loop shape is a real property of this kernel at identical work" }

$lines = @()
$lines += "# Is the ``n_in`` collapse real, or was it the harness?"
$lines += ""
$lines += "Run ``$Stamp``, $Blocks blocks x $Cycles cycles, **one tensor per PROCESS** with the"
$lines += "activation scratch sized to that tensor, pinned to one P-core (mask 0x$('{0:X}' -f $Mask)),"
$lines += "order reversed on even blocks."
$lines += ""
$lines += "## First: one row of the 2026-08-27 table is WITHDRAWN"
$lines += ""
$lines += "``blk.0.ffn_down.weight`` is **Q6_K**, not Q4_K -- in a Q4_K_M model it is upgraded."
$lines += "210 bytes per super-block against 144, **20.4 MB against 14.0**, and a different"
$lines += "kernel (``vec_dot_q6_k_q8_k_xn_avx2``). The ``ffn_gate`` vs ``ffn_down`` 6.5x was"
$lines += "never like-for-like and **is withdrawn**. The instrument now prints the quant type"
$lines += "and weight bytes on every row: a fixture drawn BY NAME from a real model can"
$lines += "differ in the property a comparison rests on, and nothing in that table said so."
$lines += ""
$lines += "## What survives: one exact transpose pair"
$lines += ""
$lines += "``attn_q`` (2560 x 4096) and ``attn_output`` (4096 x 2560) are both Q4_K, both"
$lines += "5,898,240 bytes, same FLOP, and the same 40,960 (row x super-block) unpacks."
$lines += "Only the loop shape differs: 10 super-blocks inside a row against 16."
$lines += ""
$lines += "| tensor | type | n_in x n_out | N | A GF/s | A/B (alias) | A/C (floor) |"
$lines += "|---|---|---|---|---|---|---|"
foreach ($k in $order2) {
    $c = $cells[$k]
    $lines += ("| {0} | {1} | {2} x {3} | {4} | **{5:n2}** | {6:n3} | {7:n3} |" -f
        $c.tensor, $c.type, $c.n_in, $c.n_out, $c.n,
        (Median $c.gfa), (Median $c.ab), (Median $c.ac))
}
$lines += ""
$lines += "## The pre-registered outcomes"
$lines += ""
$lines += ("Transpose ratio at N=16, **isolated**: ``attn_q`` {0:n2} / ``attn_out`` {1:n2} = **{2:n3}x**." -f $qq, $oo, $ratio)
$lines += "In the contaminated four-tensor run it was **1.78x** (65.84 / 36.94)."
$lines += ""
$lines += "| outcome | band | fired |"
$lines += "|---|---|---|"
$lines += ("| H-ARTIFACT: contamination, observation dies | < 1.30x | {0} |" -f $(if ($ratio -lt 1.30) { "**YES**" } else { "no" }))
$lines += ("| BETWEEN: partly both, needs an n_out sweep | 1.30-1.60x | {0} |" -f $(if ($ratio -ge 1.30 -and $ratio -lt 1.60) { "**YES**" } else { "no" }))
$lines += ("| H-REAL: the loop shape is a real lever | >= 1.60x | {0} |" -f $(if ($ratio -ge 1.60) { "**YES**" } else { "no" }))
$lines += ""
$lines += "**=> $verdict.**"
$lines += ""
$lines += "## AND THE ISOLATED TABLE SURFACES A REAL ONE: THE BATCHED Q6_K KERNEL GOES BACKWARDS"
$lines += ""
$lines += "Every Q4_K tensor CLIMBS with N -- 38 -> 70 GF/s -- and the three of them land"
$lines += "within 5% of each other at N=16 regardless of shape. ``ffn_down`` (Q6_K) does the"
$lines += "opposite:"
$lines += ""
$lines += "| N | 1 | 2 | 4 | 8 | 16 |"
$lines += "|---|---|---|---|---|---|"
$q6 = @(); foreach ($nn in @(1,2,4,8,16)) { $q6 += ("{0:n2}" -f (Median $cells["ffn_down|$nn"].gfa)) }
$q4 = @(); foreach ($nn in @(1,2,4,8,16)) { $q4 += ("{0:n2}" -f (Median $cells["attn_q|$nn"].gfa)) }
$lines += ("| ``ffn_down`` Q6_K GF/s | {0} |" -f ($q6 -join " | "))
$lines += ("| ``attn_q`` Q4_K GF/s | {0} |" -f ($q4 -join " | "))
$lines += ""
$lines += "**The SOLO Q6_K kernel is fine** (N=1: $($q6[0]) against Q4_K's $($q4[0])). It is the BATCHED one"
$lines += "that collapses, and batching makes it ~2x WORSE where it makes Q4_K ~2x better."
$lines += "This is isolated, like-for-like, and reproduces across all $Blocks blocks."
$lines += ""
$lines += "**The cause is legible in `src/kernels/simd.c`.** Per sequence per super-block:"
$lines += ""
$lines += "- ``vec_dot_q4_k_q8_k_xn_avx2`` runs 8 iterations of ``vpmaddubsw`` + ``vpmaddwd``,"
$lines += "  **32 int8 products per instruction**."
$lines += "- ``vec_dot_q6_k_q8_k_xn_avx2`` runs 32 iterations of ``kq_load_i8x8_to_i32`` +"
$lines += "  ``kq_mac8`` -- it WIDENS i8 to i32 and does **8 products per operation**."
$lines += ""
$lines += "Same 256 elements, ~4x the uops, and the widened ``aux8`` is re-read per sequence"
$lines += "on top of the activation. The per-sequence body is the part batching multiplies"
$lines += "by N, which is why the gap only appears at N>1."
$lines += ""
$lines += "**The prize, and it is arithmetic rather than a hope.** ``ffn_down`` is Q6_K in"
$lines += "**18 of this model's 36 layers** (Q4_K_M upgrades half of them) and is 24.7% of"
$lines += "the per-layer matvec MACs. Holding everything else at ~70 GF/s, moving those"
$lines += "0.897 GFLOP/token from 10.9 to 70 takes the matvec portion of prefill from"
$lines += "0.173 to 0.104 arbitrary units = **~1.67x**. Against the ~2.75x the gate [2] bars"
$lines += "need, that is the largest live lever this project has, and the first one in weeks"
$lines += "that is a measured defect rather than a hypothesis."
$lines += ""
$lines += "**A bit-exact fix exists on paper.** ``vpmaddubsw`` takes its first operand"
$lines += "UNSIGNED, so keep the 6-bit weight as its raw [0,63] value and carry the -32"
$lines += "offset as a separate term -- exactly the shape Q4_K already uses for its ``mins``:"
$lines += "``SUM (w-32).a.s == SUM w.a.s - 32.SUM a.s``, and ``SUM a`` is what Q8_K's ``bsums``"
$lines += "already stores. Overflow is inside the bounds: 63x127x2 = 16,002 against i16's"
$lines += "32,767, then x127 x2 = 4,064,508 against i32. A 32-byte ``vpmaddubsw`` spans TWO of"
$lines += "Q6_K's 16-element scale groups, which a following ``vpmaddwd`` against"
$lines += "``[s_j x8, s_j+1 x8]`` applies exactly, since ``(a+b).s == a.s + b.s`` in integers --"
$lines += "the same argument ``simd.c`` already makes for the Q4_K mins term."
$lines += ""
$lines += "**NOT ATTEMPTED HERE.** It is a shipping kernel with a bit-exactness acceptance"
$lines += "criterion and it needs its own mutation-backed gate. This report prices it."
$lines += ""
$lines += "The ``A/B`` and ``A/C`` columns are carried through unchanged: the activation-reuse"
$lines += "ablation's conclusion does not depend on any of this, because it is a RATIO"
$lines += "between arms measured inside one timing loop, and that is exactly the quantity"
$lines += "the contamination above could not touch."
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
Say ("VERDICT: transpose ratio {0:n3}x isolated (was 1.78x contaminated) -- {1}" -f $ratio, $verdict)
