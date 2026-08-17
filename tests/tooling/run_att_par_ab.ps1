# M4.5 L4 -- what are the two attention changes worth? Four arms, one binary.
#
# THE CHANGE UNDER TEST. Attention was the last single-threaded work in the
# forward pass: matvec has run on the worker pool since K5, but the per-head
# attention loop ran on the calling thread alone. At ctx 1500 attention is
# 84.6% of decode (2026-08-17 profile), so ~85% of a long-context token was
# using one core out of eight. The heads are now dispatched to the same pool,
# one unit per q-head. Separately, the QK loop scores ATT_LANES positions per
# pass with independent accumulators, which fills a pipeline that a single
# dependent add chain leaves stalled.
#
# Both are BIT-EXACT -- verified against the archived binaries by
# run_att_par_verify.ps1 at f32 and q8 KV, at 2 and 6 threads -- so this script
# answers a pure speed question and nothing else.
#
# THE FOUR ARMS, chosen so the two changes can be told apart:
#   par0/lanes1  the pre-L4 engine: serial heads, one position at a time
#   par0/lanes4  lanes only        -- what the interleaved QK loop is worth
#   par1/lanes1  pool only         -- what the parallel heads are worth
#   par1/lanes4  both              -- the candidate
# Arms are env vars on ONE binary, so all four interleave inside one sitting.
# A before/after separated by a rebuild measures this box, not the code.
#
# TWO POINTS, because the prediction is about CONTEXT, not about speed in
# general: attention is 24.2% of decode at ctx 87 and 76.5% at ctx 554, so a
# fixed speedup of the attention term MUST show up larger at p40 than at p00.
# A flat gain across the two would mean the win is not coming from where this
# says it is, and that is the finding that would falsify the attribution.

param(
    [int]$Cycles = 3,
    [int]$NGen = 128,
    [int]$Threads = 8,
    [switch]$IncludeLong,
    # Q-heads per pool claim. 1 is the shipped default -- smallest claim, best
    # balance on a 6P+4E box. Larger values trade balance for kv-cache reuse
    # (a GQA group of 4 q-heads shares one kv-head), which is worth measuring
    # rather than assuming; this exists so the sweep costs a flag, not an edit.
    [int]$Chunk = 0,
    [string]$Stamp = "2026-08-17-m45-att-par-ab"
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$tmpE = "$env:TEMP\stoke-attab-e.txt"
$tmpO = "$env:TEMP\stoke-attab-o.txt"
$Q4    = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE = "$root\build\win\stoke.exe"
$LB    = "$root\bench\vendor\llama-b10037\llama-bench.exe"

$POINTS = @(
    @{ name = "p00"; file = "$root\tests\data\parity\p00.txt"; ptok = 23;  ctx = 87 },
    @{ name = "p40"; file = "$root\tests\data\parity\p40.txt"; ptok = 490; ctx = 554 }
)
if ($IncludeLong) {
    $POINTS += @{ name = "long"; file = "$root\tests\data\ctx-profile-long.txt"; ptok = 1436; ctx = 1500 }
}

$ARMS = @(
    @{ id = "par0lanes1"; par = "0"; lanes = "1" },
    @{ id = "par0lanes4"; par = "0"; lanes = "4" },
    @{ id = "par1lanes1"; par = "1"; lanes = "1" },
    @{ id = "par1lanes4"; par = "1"; lanes = "4" }
)

$script:LastCpuS = 0.0

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding ascii $log
    Write-Output $line
}

function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    # Explicit halves. PowerShell rounds .5 to EVEN, so $s[$n/2] on an even
    # count silently returns the second-highest of four -- a real defect found
    # on 2026-08-17 in two runners, where every even-count median either had
    # ever published read high by one position.
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    $lo = $s[[int]($n / 2) - 1]; $hi = $s[[int]($n / 2)]
    return [math]::Round(($lo + $hi) / 2.0, 4)
}

# Runs a child to completion and records its CPU-SECONDS as well as its rate.
# The second signal is independent of the first: the same work costs ~4x more
# processor-time on an efficiency core, so a unit parked on the wrong cores
# reads HIGH here while also finishing later, and can be discarded without
# anyone having to argue about whether its tok/s "looks low".
function RunCapture($exe, $argList) {
    $p = Start-Process -FilePath $exe -ArgumentList $argList `
                       -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO `
                       -NoNewWindow -PassThru
    $cpu = 0.0
    while (-not $p.HasExited) {
        try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
        Start-Sleep -Milliseconds 250
    }
    try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
    $script:LastCpuS = [math]::Round($cpu, 1)
}

function StokeRate($file, $par, $lanes) {
    [Environment]::SetEnvironmentVariable('STOKE_ATT_PAR', $par)
    [Environment]::SetEnvironmentVariable('STOKE_ATT_LANES', $lanes)
    if ($Chunk -gt 0) { [Environment]::SetEnvironmentVariable('STOKE_ATT_CHUNK', "$Chunk") }
    RunCapture $STOKE @("batch", ('"' + $Q4 + '"'), "--raw", "-n", "$NGen",
        "-f", ('"' + $file + '"'), "--seqs", "1", "--threads", "$Threads",
        "--kv", "f32", "--timing")
    $t = (Get-Content $tmpE -Raw)
    if (-not $t) { return $null }
    $m = [regex]::Match($t, '\(([\d.]+)\s+tok/s\s+aggregate\)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

# -d puts llama at the SAME context depth stoke reaches through its prompt, so
# the two engines are compared doing the same amount of attention work.
function LlamaRate($depth) {
    RunCapture $LB @("-m", ('"' + $Q4 + '"'), "-p", "0", "-n", "$NGen",
        "-d", "$depth", "-r", "3", "-t", "$Threads")
    $m = [regex]::Match((Get-Content $tmpO -Raw), 'tg\d+[^|]*\|\s*([\d.]+)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

Say "START cycles=$Cycles n=$NGen threads=$Threads points=$($POINTS.Count) arms=$($ARMS.Count)"
$rows = @()
for ($c = 1; $c -le $Cycles; $c++) {
    $sb = [math]::Round((Get-Counter '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue / 1GB, 2)
    Say "CYCLE $c (standby $sb GB)"
    $order = if ($c % 2 -eq 1) { $POINTS } else { $POINTS[($POINTS.Count - 1)..0] }
    foreach ($pt in $order) {
        # Rotate the arm order every cycle so no arm keeps the same position in
        # the sitting -- position in a sitting is itself a source of drift on
        # this box, and a fixed order lets that drift masquerade as a gain.
        $shift = ($c - 1) % $ARMS.Count
        $armOrder = @()
        for ($i = 0; $i -lt $ARMS.Count; $i++) { $armOrder += $ARMS[($i + $shift) % $ARMS.Count] }
        foreach ($arm in $armOrder) {
            $v = StokeRate $pt.file $arm.par $arm.lanes
            $rows += [pscustomobject]@{ cycle = $c; name = $pt.name; ctx = $pt.ctx;
                                        arm = $arm.id; rate = $v; cpu = $script:LastCpuS }
            Say ("  c{0} {1,-5} {2,-11} {3,8}  cpu={4}s" -f $c, $pt.name, $arm.id,
                 $(if ($null -eq $v) { "PARSE-FAIL" } else { $v }), $script:LastCpuS)
        }
        $lv = LlamaRate $pt.ptok
        $rows += [pscustomobject]@{ cycle = $c; name = $pt.name; ctx = $pt.ctx;
                                    arm = "llama"; rate = $lv; cpu = $script:LastCpuS }
        Say ("  c{0} {1,-5} {2,-11} {3,8}  cpu={4}s" -f $c, $pt.name, "llama",
             $(if ($null -eq $lv) { "PARSE-FAIL" } else { $lv }), $script:LastCpuS)
    }
}
[Environment]::SetEnvironmentVariable('STOKE_ATT_PAR', $null)
[Environment]::SetEnvironmentVariable('STOKE_ATT_LANES', $null)
[Environment]::SetEnvironmentVariable('STOKE_ATT_CHUNK', $null)

function Cell($name, $arm) {
    return Median @($rows | Where-Object { $_.name -eq $name -and $_.arm -eq $arm -and $null -ne $_.rate } |
                    ForEach-Object { $_.rate })
}

$lines = @()
$lines += "# M4.5 L4 -- attention on the pool, and the interleaved QK loop"
$lines += ""
$lines += "$Cycles cycles, point-order and arm-order both rotating, $Threads threads, unpinned, n=$NGen, ``--kv f32``. All four stoke arms come from ONE binary via ``STOKE_ATT_PAR`` / ``STOKE_ATT_LANES``; all four are byte-identical in output, so this is speed only. llama at matched depth (``-d``)."
$lines += ""
$lines += "| point | avg ctx | par0/lanes1 | par0/lanes4 | par1/lanes1 | par1/lanes4 | best/base | llama | best/llama |"
$lines += "|---|---|---|---|---|---|---|---|---|"
foreach ($pt in $POINTS) {
    $a = Cell $pt.name "par0lanes1"
    $b = Cell $pt.name "par0lanes4"
    $c2 = Cell $pt.name "par1lanes1"
    $d = Cell $pt.name "par1lanes4"
    $l = Cell $pt.name "llama"
    $gain = if ($a -and $d) { "{0:n3}x" -f ($d / $a) } else { "n/a" }
    $vsl  = if ($d -and $l) { "{0:n3}x" -f ($d / $l) } else { "n/a" }
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | **{6}** | {7} | **{8}** |" -f
               $pt.name, $pt.ctx, $a, $b, $c2, $d, $gain, $l, $vsl)
}
$lines += ""
$lines += "## Per-token cost, which is where the attention term is visible"
$lines += ""
$lines += "| point | avg ctx | base ms/tok | best ms/tok | saved ms/tok | llama ms/tok |"
$lines += "|---|---|---|---|---|---|"
foreach ($pt in $POINTS) {
    $a = Cell $pt.name "par0lanes1"
    $d = Cell $pt.name "par1lanes4"
    $l = Cell $pt.name "llama"
    if ($a -and $d) {
        $am = [math]::Round(1000.0 / $a, 2)
        $dm = [math]::Round(1000.0 / $d, 2)
        $lm = if ($l) { [math]::Round(1000.0 / $l, 2) } else { "n/a" }
        $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} |" -f $pt.name, $pt.ctx, $am, $dm,
                   [math]::Round($am - $dm, 2), $lm)
    }
}
$lines += ""
$lines += "## The slope, which is the M5-critical number"
$lines += ""
$lines += "ms/token per CONTEXT-token, fitted between the extreme points. Before L4 stoke's slope was 0.1791 against llama's 0.0150 (2026-08-17, `db85e0f`) -- 11.9x worse -- and extrapolating it to ctx 1024 was what made M5's aggregate bars look unreachable. That extrapolation was measured against a SINGLE-THREADED attention loop, so this row is what replaces it."
$lines += ""
if ($POINTS.Count -ge 2) {
    $p1 = $POINTS[0]; $p2 = $POINTS[$POINTS.Count - 1]
    $dctx = $p2.ctx - $p1.ctx
    $lines += "| arm | ms/tok @ ctx $($p1.ctx) | ms/tok @ ctx $($p2.ctx) | slope ms/context-token |"
    $lines += "|---|---|---|---|"
    foreach ($armId in (@($ARMS | ForEach-Object { $_.id }) + @("llama"))) {
        $a = Cell $p1.name $armId
        $b = Cell $p2.name $armId
        if ($a -and $b -and $dctx -gt 0) {
            $am = 1000.0 / $a; $bm = 1000.0 / $b
            $lines += ("| {0} | {1} | {2} | **{3}** |" -f $armId, [math]::Round($am, 2),
                       [math]::Round($bm, 2), [math]::Round(($bm - $am) / $dctx, 4))
        }
    }
    $lines += ""
    $lines += "Two points fit a line and nothing more. The 2026-08-17 profile showed this cost is CONVEX -- the middle point sat 14% below the chord, consistent with the per-layer KV crossing out of L3 -- so a slope read off the extremes OVERSTATES the middle and UNDERSTATES the far end. Use it to compare arms in the same sitting, not as a law."
}
$lines += ""
$lines += "## Wrong-cores check (CPU-seconds per unit)"
$lines += ""
$lines += "| point | arm | min | median | max |"
$lines += "|---|---|---|---|---|"
foreach ($pt in $POINTS) {
    foreach ($arm in ($ARMS | ForEach-Object { $_.id })) {
        $cs = @($rows | Where-Object { $_.name -eq $pt.name -and $_.arm -eq $arm } | ForEach-Object { $_.cpu })
        if ($cs.Count -gt 0) {
            $lines += ("| {0} | {1} | {2} | {3} | {4} |" -f $pt.name, $arm,
                       ($cs | Measure-Object -Minimum).Minimum, (Median $cs),
                       ($cs | Measure-Object -Maximum).Maximum)
        }
    }
}
$lines += ""
$lines += "A unit whose CPU-seconds sit far above its arm's median was doing the same work somewhere slower -- efficiency cores, thermal throttling, contention -- and should be discarded rather than averaged in. EcoQoS is fixed in the engine, but this check exists to catch the NEXT thing that quietly moves work."
$lines += ""
$lines += "Raw log: ``$Stamp-raw.txt``."
($lines -join "`r`n") | Out-File -Encoding ascii $rep
Say "REPORT -> $rep"
Say "DONE"
