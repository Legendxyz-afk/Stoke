# =====================================================================
# ABLATION: what are the ACTIVATION RE-READS costing the batched Q4_K kernel?
#
# A = matvec_batch with n_seq DISTINCT activation-scratch buffers (the real
#     case). B = the same call with every activation pointer ALIASED to ONE
#     buffer: same loads, same arithmetic, one cache-resident working set.
#
# Prediction and discriminators are pre-registered in
# docs/superpowers/memos/2026-08-27-m5-activation-reuse-PREREGISTERED.md and
# were committed before this script was run. Headline predictions: A/B <= 1.15
# on attn_q at N=16, <= 1.25 on ffn_down, and 1.00 +- 0.03 on the N=1 control.
#
# ▶▶ BOTH ARMS LIVE IN ONE PROCESS AND ALTERNATE REP BY REP, which is a
# STRONGER protocol than the mullo/mins ablations could use. Those had to build
# two binaries and interleave whole processes -- so they also had to control for
# code layout (mingw embeds object paths; PE here is not position-independent,
# and two builds out of different-length directories differ in .text itself).
# Here there is ONE binary and one code path: the arms differ in the value of a
# pointer array and in nothing else. The alternation happens inside the timing
# loop with the order flipping each cycle, so a warming box cannot charge one
# arm the warmer half.
#
# ▶ PINNED TO ONE P-CORE. [[measurement-needs-interleaving]] prices unpinned
# E-core migration at up to 5.8x on this box and thermal drift at 22% on
# compute-dense work. `bench` never starts the pool, so this is a single-thread
# measurement and one logical CPU is the whole of the affinity question. The
# mask is set on THIS PowerShell process and inherited by the child.
#
# ⚠ THE ARMS ARE NOT INTERCHANGEABLE: B COMPUTES THE WRONG ANSWER by
# construction (every sequence reads the last-quantized activation). The
# instrument asserts that, and asserts the fixture's own distinctness, and
# REFUSES TO RUN a cell where either check passes vacuously. Nothing here needs
# a mutant binary or a deleted build directory.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_actreuse.ps1
#        [-Cycles 12] [-Blocks 3] [-Mask 1]
# =====================================================================
param(
    [int]$Cycles = 12,
    [int]$Blocks = 3,
    [int]$Mask   = 1,
    [string]$Stamp = ""
)

$ErrorActionPreference = "Stop"
$root  = "M:\New Drive\Desktop\Programming\stoke"
$exe   = "$root\build\win\stoke.exe"
$mdl   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$gr    = "$root\docs\superpowers\gate-reports"
$memo  = "$root\docs\superpowers\memos\2026-08-27-m5-activation-reuse-PREREGISTERED.md"
if (-not $Stamp) { $Stamp = "{0}-m5-actreuse" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp-raw.txt"
$report = "$gr\$Stamp-report.md"
$tmp    = "$env:TEMP\stoke-actreuse"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

foreach ($p in @($exe, $mdl, $memo)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}
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

# ▶ BLOCKS, NOT ONE LONG RUN. The box warms and reaches an equilibrium; a single
# block is a warming transient plus a plateau, and cross-cell comparison belongs
# on the plateau. Each block is a complete independent sweep, so block-to-block
# agreement is the drift detector -- taken from the ARMS' own repeats rather
# than from a probe sampled after the transient, which reports healthy every
# time ([[measurement-needs-interleaving]]).
$prev = (Get-Process -Id $PID).ProcessorAffinity
Say "activation-reuse ablation. exe $exe"
Say "cycles $Cycles per arm per cell, $Blocks blocks, affinity mask 0x$('{0:X}' -f $Mask) (was 0x$('{0:X}' -f [int]$prev))"
Say "exe commit $(git -C $root rev-parse --short HEAD)"
Say "pre-registration: $(Split-Path -Leaf $memo)"

$blocks = @()
try {
    (Get-Process -Id $PID).ProcessorAffinity = [IntPtr]$Mask
    for ($b = 1; $b -le $Blocks; $b++) {
        $out = "$tmp\block$b.txt"
        if (Test-Path $out) {
            $t = Get-Content $out -Raw
            if ($t -match 'attn_q') { Say "  block $b already banked, skipping"; $blocks += $out; continue }
        }
        $t0 = Get-Date
        $p = Start-Process -FilePath $exe `
             -ArgumentList @("bench", "--matvec-batch", ('"' + $mdl + '"'),
                             "--cycles", "$Cycles") `
             -RedirectStandardOutput $out -RedirectStandardError "$tmp\e$b.txt" `
             -NoNewWindow -PassThru -Wait
        $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        if ($p.ExitCode -ne 0) {
            Say ("  !! block $b exited {0}: {1}" -f $p.ExitCode, (Get-Content "$tmp\e$b.txt" -Raw))
            throw "STOP: a broken arm must stop, never report a ratio"
        }
        Say "  block $b done (${secs}s)"
        $blocks += $out
    }
} finally {
    (Get-Process -Id $PID).ProcessorAffinity = $prev
}

# ---- parse every block's table into cells -----------------------------------
# The instrument's row shape:
#   attn_q      2560 x 4096     16     45.6      3.142     2.980   1.054   ...   7/12
$cells = @{}
$order = @()
for ($i = 0; $i -lt $blocks.Count; $i++) {
    foreach ($ln in (Get-Content $blocks[$i])) {
        $m = [regex]::Match($ln,
            '^\s+(\w+)\s+(\d+) x (\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)/(\d+)')
        if (-not $m.Success) { continue }
        $key = "{0}|{1}" -f $m.Groups[1].Value, $m.Groups[4].Value
        if (-not $cells.ContainsKey($key)) {
            $cells[$key] = @{ tensor = $m.Groups[1].Value; n_in = [int]$m.Groups[2].Value
                              n_out = [int]$m.Groups[3].Value; n = [int]$m.Groups[4].Value
                              act = [double]$m.Groups[5].Value
                              a = @(); b = @(); ratio = @(); gfa = @(); won = 0; pairs = 0 }
            $order += $key
        }
        $c = $cells[$key]
        $c.a     += [double]$m.Groups[6].Value
        $c.b     += [double]$m.Groups[7].Value
        $c.ratio += [double]$m.Groups[8].Value
        $c.gfa   += [double]$m.Groups[9].Value
        $c.won   += [int]$m.Groups[11].Value
        $c.pairs += [int]$m.Groups[12].Value
    }
}
if ($order.Count -eq 0) { throw "STOP: parsed no cells -- the instrument's table shape moved" }

function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return ($s[$n / 2 - 1] + $s[$n / 2]) / 2.0
}

# ---- the pre-registered checks ---------------------------------------------
$worst = 0.0; $worstCell = ""
$ctrlBad = @()
foreach ($k in $order) {
    $c = $cells[$k]
    $r = Median $c.ratio
    if ($c.n -eq 1) {
        if ([math]::Abs($r - 1.0) -gt 0.03) { $ctrlBad += ("{0} N=1 A/B {1:n3}" -f $c.tensor, $r) }
    } elseif ($r -gt $worst) { $worst = $r; $worstCell = "$($c.tensor) N=$($c.n)" }
}
$p1 = (Median $cells["attn_q|16"].ratio)
$p2 = (Median $cells["ffn_down|16"].ratio)

$verdict = "the memory hypothesis is DEAD -- no named lever remains"
if ($worst -ge 2.0)      { $verdict = "the memory hypothesis IS the gap -- cache-block the row loop" }
elseif ($worst -ge 1.25) { $verdict = "real but PARTIAL -- it joins the ~18% of named compute levers" }

$lines = @()
$lines += "# What are the activation RE-READS costing the batched Q4_K kernel?"
$lines += ""
$lines += "Run ``$Stamp``, $Blocks blocks x $Cycles interleaved A/B cycles per cell, single-threaded,"
$lines += "pinned to one P-core (mask 0x$('{0:X}' -f $Mask)). Instrument ``stoke bench --matvec-batch``."
$lines += "Prediction and discriminators pre-registered BEFORE the first cell:"
$lines += "``memos/$(Split-Path -Leaf $memo)``."
$lines += ""
$lines += "**A** = ``matvec_batch`` with N distinct activation buffers -- the real case."
$lines += "**B** = the same call with every activation pointer ALIASED to one buffer:"
$lines += "same loads, same arithmetic, one cache-resident working set."
$lines += "**B computes the wrong answer by construction**, and each cell refuses to run"
$lines += "unless it does AND unless the N activations differ from each other."
$lines += ""
$lines += "``act KiB`` is the activation tile arm A RE-READS FOR EVERY OUTPUT ROW."
$lines += "**A/B is an upper bound**: a real redesign still has to read each activation once."
$lines += ""
$lines += "| tensor | n_in x n_out | N | act KiB | A med ms | B med ms | **A/B** | A GF/s | B won pairings |"
$lines += "|---|---|---|---|---|---|---|---|---|"
foreach ($k in $order) {
    $c = $cells[$k]
    $lines += ("| {0} | {1} x {2} | {3} | {4:n1} | {5:n3} | {6:n3} | **{7:n3}** | {8:n2} | {9}/{10} |" -f
        $c.tensor, $c.n_in, $c.n_out, $c.n, $c.act,
        (Median $c.a), (Median $c.b), (Median $c.ratio), (Median $c.gfa), $c.won, $c.pairs)
}
$lines += ""
$lines += "## The pre-registered predictions"
$lines += ""
$lines += "| # | cell | predicted | measured | result |"
$lines += "|---|---|---|---|---|"
$lines += ("| P1 | ``attn_q`` N=16 | A/B <= 1.15 | {0:n3} | {1} |" -f $p1, $(if ($p1 -le 1.15) { "**HELD**" } else { "**FALSIFIED**" }))
$lines += ("| P2 | ``ffn_down`` N=16 | A/B <= 1.25 | {0:n3} | {1} |" -f $p2, $(if ($p2 -le 1.25) { "**HELD**" } else { "**FALSIFIED**" }))
$lines += ("| P3 | N=1 control | 1.00 +- 0.03 | see table | {0} |" -f $(if ($ctrlBad.Count -eq 0) { "**HELD**" } else { "**FALSIFIED** (" + ($ctrlBad -join "; ") + ")" }))
$lines += ""
$lines += "## Verdict against the discriminators fixed before the run"
$lines += ""
$lines += ("Largest A/B over every batched cell: **{0:n3}** ({1})." -f $worst, $worstCell)
$lines += ""
$lines += "| if | then | fired |"
$lines += "|---|---|---|"
$lines += ("| A/B >= 2.0 anywhere | the memory hypothesis IS the gap | {0} |" -f $(if ($worst -ge 2.0) { "**YES**" } else { "no" }))
$lines += ("| A/B in [1.25, 2.0) | real but partial | {0} |" -f $(if ($worst -ge 1.25 -and $worst -lt 2.0) { "**YES**" } else { "no" }))
$lines += ("| A/B <= 1.15 everywhere | the hypothesis is DEAD | {0} |" -f $(if ($worst -le 1.15) { "**YES**" } else { "no" }))
$lines += ""
$lines += "**⇒ $verdict.**"
$lines += ""
if ($ctrlBad.Count -gt 0) {
    $lines += "> ⚠ **WITHDRAWAL CONDITION 3 FIRED**: the N=1 control is not 1.00 +- 0.03"
    $lines += "> ($($ctrlBad -join '; ')). ``vec.c`` routes ``n_seq == 1`` to the solo worker, so"
    $lines += "> there is nothing to alias and this deviation is HARNESS BIAS, not physics."
    $lines += "> No cell in the table above is readable until it is explained."
    $lines += ""
}
$lines += "## What this measures, and what it does not"
$lines += ""
$lines += "- It is an **upper bound**, not a projected speedup. Perfect reuse is unbuildable;"
$lines += "  a real redesign blocks the row loop over a tile that still has to be read once."
$lines += "- **Single-threaded.** t=1 is the headline because the 3.22x gap was measured at"
$lines += "  every thread count and t=1 has no pool, no barrier and no dispatch. At t=8 the"
$lines += "  same tile is contended by 8 cores against a shared L3, which can only make the"
$lines += "  re-reads matter MORE -- so a small ratio here is the conservative reading."
$lines += "- **A/B is measured, the CAUSE is not.** A ratio near 1 says the re-reads are not"
$lines += "  costing time; it does not say which cache level served them."
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
Say ("VERDICT: worst A/B {0:n3} ({1}) -- {2}" -f $worst, $worstCell, $verdict)
