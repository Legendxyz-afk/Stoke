# M4.5 -- the thread-scaling curve, stoke vs llama at t = 1,2,4,6,8,10.
#
# WHY THIS RUN EXISTS. `bench --ops` attributed 225.9 ms/token of
# SINGLE-THREADED matvec work (36 layers 202.9 + logits 23.0) against an engine
# that takes ~73 ms/token at 6 threads. Perfect 6x scaling of the matvec alone
# predicts ~51 ms, so ~22 ms/token is parallel inefficiency + thread sync +
# attention over the KV cache -- none of which `bench --ops` measures at all.
# 22 ms is larger than the 4.5% still owed to P-BAR, so this outranks rung D.
#
# ▶ WHAT THE CURVE DECIDES, and it is not a matter of taste. If stoke's
# single-thread rate already sits at the same fraction of llama's as its
# 6-thread rate does, the gap is PER-THREAD WORK and the parallel story is a
# red herring. If the fraction DEGRADES with t, the gap is scaling. One
# measurement separates two very different next moves.
#
# ▶ WHY UNPINNED. Every prior perf run pinned to 0xFFF = the 6 P-cores' twelve
# hyperthread siblings. That mask is defensible at t<=6 and MEANINGLESS for a
# thread-scaling question: at t=8 and t=10 it stacks threads onto three
# physical cores and measures hyperthread contention rather than scaling, and
# it hides the 4 E-cores that a real run would use. The 2026-08-11 llama-bench
# curve this run is anchored against was itself taken unpinned, so unpinned is
# also the only setting under which that anchor means anything. The affinity
# mask is set to ALL logical processors explicitly rather than left to
# inheritance, and logged, because a child process inherits its parent's mask
# and that is how a pinned number gets published as an unpinned one.
#
# ▶ THE ANCHOR IS SIX POINTS, NOT ONE. 2026-08-11, quiesced, llama-bench
# b10037, `-p 0 -n 64 -r 3` on this same Q4_K_M: 4.98 / 9.84 / 16.07 / 17.45 /
# 17.85 / 17.93 at t = 1/2/4/6/8/10. This run uses THE SAME COMMAND, so every
# llama cell is independently anchored. A box that has drifted cannot fake six
# points at once, and if it has drifted the stoke column is read as ratios
# against the llama column measured minutes earlier rather than as absolutes.
#
# ▶ ONE BRIDGE CELL. stoke t=6 at n=128 is the exact configuration of the
# P-BAR ladder (12.523 pre-Q6_K, 14.48 with it), so it ties this curve to the
# ladder without re-running the ladder.

param(
    [int]$Cycles     = 4,
    [int]$NGen       = 64,
    [int]$BridgeN    = 128,
    [int]$QuietTries = 20,
    [int]$QuietCpu   = 12,
    [string]$Stamp   = "2026-08-17-m45-thread-curve",
    # ▶ A STRING, DELIBERATELY, NOT [int[]]. Under `powershell -File`, every
    # argument arrives as a string and `-Threads 1,10` binds to int[] by way of
    # PowerShell's culture-aware numeric parse, which reads the comma as a
    # THOUSANDS SEPARATOR: the array becomes @(110) and the run measures a
    # single 110-thread cell while logging and banking exactly like a healthy
    # run. The pre-flight caught it; nothing downstream would have. Splitting
    # here cannot be fooled that way.
    [string]$Threads = "1,2,4,6,8,10"
)

$ErrorActionPreference = 'Continue'
$TVALS = @($Threads -split '\s*,\s*' | Where-Object { $_ } | ForEach-Object { [int]$_ })
if ($TVALS.Count -lt 1) { throw "no thread values parsed from '$Threads'" }

$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$bank = "$gr\$Stamp-bank.jsonl"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$tmpE = "$env:TEMP\stoke-curve-err.txt"
$tmpO = "$env:TEMP\stoke-curve-out.txt"

$Q4     = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE  = "$root\build\win\stoke.exe"
$LB     = "$root\bench\vendor\llama-b10037\llama-bench.exe"
$PROMPT = "$root\tests\data\parity\p00.txt"
$ISLC   = "M:\New Drive\Downloads\ISLC v1.0.4.5\Intelligent standby list cleaner ISLC.exe"

# 2026-08-11 quiesced llama-bench curve, same binary, same flags, same model.
$ANCHOR_L = @{ 1 = 4.98; 2 = 9.84; 4 = 16.07; 6 = 17.45; 8 = 17.85; 10 = 17.93 }
$ANCHOR_BRIDGE = 14.48   # stoke t=6, n=128, unpinned, post-Q6_K

# Full mask over every logical processor -- see the header note on inheritance.
$nlog = [int](Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$mask = [IntPtr]([int64][math]::Pow(2, $nlog) - 1)
(Get-Process -Id $PID).ProcessorAffinity = $mask

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Bank($obj) { ($obj | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 $bank }
function Banked() {
    if (-not (Test-Path $bank)) { return @() }
    $out = @()
    foreach ($l in (Get-Content $bank)) { if ($l.Trim()) { try { $out += ($l | ConvertFrom-Json) } catch {} } }
    return $out
}
# A unit that failed to PARSE is not done. A transient capture failure must
# cost a re-run, never a permanent hole in the curve.
function HasUnit($id) {
    foreach ($b in (Banked)) { if ($b.id -eq $id -and $null -ne $b.tok_s) { return $true } }
    return $false
}
function CpuBusy() {
    $c = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object { $_.Name -eq "_Total" }
    return [int]$c.PercentProcessorTime
}
function StandbyGB() {
    $s = (Get-Counter '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue
    return [math]::Round($s / 1GB, 2)
}
# ▶ NOT the inherited `$s[[int](($n-1)/2)]`. For an EVEN count that index is
# (n-1)/2 = x.5 and PowerShell's [int] cast rounds half to EVEN, so n=4 gives
# index 2 -- the second-HIGHEST of four, not a median. It reads high on a clean
# cell and, on a cell with one contaminated run, silently lands on the middle
# of the survivors and looks correct. Found by recomputing this report's
# medians by hand rather than by anything failing.
function Median($xs) {
    $s = @($xs | Sort-Object)
    $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return [math]::Round(($s[$n / 2 - 1] + $s[$n / 2]) / 2.0, 3)
}
# Start-Process, never `2> file`: PowerShell decorates a native command's
# stderr and WRAPS IT AT CONSOLE WIDTH, which on 2026-08-14 split the token
# this regex matches and cost an entire arm's numbers.
function RunCapture($exe, $argList) {
    Start-Process -FilePath $exe -ArgumentList $argList `
                  -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO `
                  -NoNewWindow -Wait
}
# "decode 64 tok in 4219.6 ms (15.166 tok/s aggregate)" -- on stderr.
function StokeRate($threads, $n) {
    RunCapture $STOKE @("batch", ('"' + $Q4 + '"'), "--raw", "-n", "$n",
                        "-f", ('"' + $PROMPT + '"'), "--seqs", "1",
                        "--threads", "$threads", "--kv", "f32", "--timing")
    $t = (Get-Content $tmpE -Raw)
    if (-not $t) { return $null }
    $m = [regex]::Match($t, '\(([\d.]+)\s+tok/s\s+aggregate\)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}
# llama-bench prints its markdown table on STDOUT: "| ... | tg64 | 17.45 ± 0.10 |"
function LlamaRate($threads, $n) {
    RunCapture $LB @("-m", ('"' + $Q4 + '"'), "-p", "0", "-n", "$n", "-r", "3", "-t", "$threads")
    $t = (Get-Content $tmpO -Raw)
    if (-not $t) { return $null }
    $m = [regex]::Match($t, 'tg\d+\s*\|\s*([\d.]+)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

$islcWasRunning = [bool](Get-Process -Name "*ISLC*" -ErrorAction SilentlyContinue)
Say "START  cycles=$Cycles n=$NGen bridge_n=$BridgeN threads=$($TVALS -join '|') (from '$Threads')"
Say "  affinity mask 0x$('{0:X}' -f [int64]$mask) over $nlog logical processors (UNPINNED by design)"
Say "  islc_was_running=$islcWasRunning"

try {
    Get-Process -Name "*ISLC*" -ErrorAction SilentlyContinue | Stop-Process -Force

    # THREE CONSECUTIVE SAMPLES. A single sample on a bouncing box dips under
    # any threshold eventually; on 2026-08-14 a 12% gate passed at 23:25:58 and
    # the block started at 22% seconds later.
    $tries = 0; $consec = 0
    while ($tries -lt $QuietTries) {
        $busy = CpuBusy
        if ($busy -le $QuietCpu) {
            $consec++
            if ($consec -ge 3) { Say "  box quiet at $busy% ($consec consecutive) after $tries min"; break }
            Say "  quiet sample $consec/3 at $busy%"
        } else {
            if ($consec -gt 0) { Say "  quiet streak broken at $busy%" }
            $consec = 0
            Say "  waiting: cpu $busy% > $QuietCpu%"
        }
        Start-Sleep -Seconds 60
        $tries++
    }
    if ($tries -ge $QuietTries) { Say "  NEVER WENT QUIET -- measuring anyway and flagging it" }
    Bank @{ id = "quiesce"; kind = "quiesce"; cpu_busy = (CpuBusy); standby_gb = (StandbyGB)
            waited_min = $tries; nlog = $nlog; utc = (Get-Date -Format s) }

    for ($cyc = 1; $cyc -le $Cycles; $cyc++) {
        # Warm PER CYCLE and CHECK it. Free RAM reads healthy while the model
        # is entirely absent from the standby cache, which once produced a
        # 1.4 tok/s "result" that was really the disk.
        for ($w = 1; $w -le 3; $w++) {
            $null = StokeRate 6 8
            if ((StandbyGB) -ge 2.0) { break }
            Say "  warm $w/3 left standby at $(StandbyGB) GB -- warming again"
        }
        $sb = StandbyGB
        if ($sb -lt 2.0) { Say "  ⚠ standby only $sb GB -- cycle $cyc MAY BE DISK-BOUND" }
        Say "CYCLE $cyc starting (cpu $(CpuBusy)%, standby $sb GB)"
        Bank @{ id = "c$cyc-state"; kind = "cyclestate"; cycle = $cyc
                cpu_busy = (CpuBusy); standby_gb = $sb; utc = (Get-Date -Format s) }

        # Rotate BOTH the thread direction and which engine leads, so neither
        # position-in-cycle nor engine order can masquerade as a result.
        $order = if ($cyc % 2 -eq 1) { $TVALS } else { $TVALS[($TVALS.Count - 1)..0] }
        $stokeFirst = ($cyc -le 2)

        foreach ($t in $order) {
            $pair = if ($stokeFirst) { @("S", "L") } else { @("L", "S") }
            foreach ($eng in $pair) {
                $id = "c$cyc-t$t-$eng"
                if (HasUnit $id) { Say "  $id [banked]"; continue }
                $v = if ($eng -eq "S") { StokeRate $t $NGen } else { LlamaRate $t $NGen }
                Bank @{ id = $id; kind = "rate"; cycle = $cyc; threads = $t; engine = $eng
                        n = $NGen; tok_s = $v; cpu_busy = (CpuBusy); utc = (Get-Date -Format s) }
                Say ("  {0}  {1}" -f $id, $(if ($null -eq $v) { "PARSE-FAIL" } else { $v }))
            }
        }

        # The bridge to the P-BAR ladder: stoke t=6 at n=128.
        $bid = "c$cyc-bridge"
        if (-not (HasUnit $bid)) {
            $bv = StokeRate 6 $BridgeN
            Bank @{ id = $bid; kind = "bridge"; cycle = $cyc; threads = 6; engine = "S"
                    n = $BridgeN; tok_s = $bv; cpu_busy = (CpuBusy); utc = (Get-Date -Format s) }
            Say ("  {0}  {1}" -f $bid, $(if ($null -eq $bv) { "PARSE-FAIL" } else { $bv }))
        }
    }
}
finally {
    if ($islcWasRunning) {
        if (-not (Get-Process -Name "*ISLC*" -ErrorAction SilentlyContinue)) {
            Start-Process -FilePath $ISLC; Start-Sleep -Seconds 3
        }
        Say ("ISLC restore attempted: running={0}" -f [bool](Get-Process -Name "*ISLC*" -ErrorAction SilentlyContinue))
    } else { Say "ISLC was not running at start -- not started" }

    $rows   = @(Banked)
    $rate   = @($rows | Where-Object { $_.kind -eq "rate"   -and $null -ne $_.tok_s })
    $bridge = @($rows | Where-Object { $_.kind -eq "bridge" -and $null -ne $_.tok_s })

    $lines = @()
    $lines += "# M4.5 -- thread-scaling curve, stoke vs llama"
    $lines += ""
    $lines += "Run ``$Stamp``. UNPINNED over $nlog logical processors (6P+4E). stoke = ``batch --seqs 1 --kv f32 -n $NGen`` on p00; llama = ``llama-bench -p 0 -n $NGen -r 3``, b10037. $Cycles cycles, thread direction and engine order both rotate. Median per cell."
    $lines += ""
    $lines += "## The curve"
    $lines += ""
    $lines += "| t | stoke tok/s | llama tok/s | stoke/llama | stoke speedup vs t=1 | llama speedup vs t=1 | llama anchor 08-11 | anchor dev |"
    $lines += "|---|---|---|---|---|---|---|---|"
    $sBase = Median @($rate | Where-Object { $_.threads -eq $TVALS[0] -and $_.engine -eq "S" } | ForEach-Object { $_.tok_s })
    $lBase = Median @($rate | Where-Object { $_.threads -eq $TVALS[0] -and $_.engine -eq "L" } | ForEach-Object { $_.tok_s })
    foreach ($t in $TVALS) {
        $s = Median @($rate | Where-Object { $_.threads -eq $t -and $_.engine -eq "S" } | ForEach-Object { $_.tok_s })
        $l = Median @($rate | Where-Object { $_.threads -eq $t -and $_.engine -eq "L" } | ForEach-Object { $_.tok_s })
        $sl = if ($s -and $l) { "{0:n3}x" -f ($s / $l) } else { "n/a" }
        $ss = if ($s -and $sBase) { "{0:n2}x" -f ($s / $sBase) } else { "n/a" }
        $ls = if ($l -and $lBase) { "{0:n2}x" -f ($l / $lBase) } else { "n/a" }
        $an = $ANCHOR_L[$t]
        $ad = if ($l -and $an) { "{0:n1}%" -f (100.0 * ($l - $an) / $an) } else { "n/a" }
        $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f $t, $s, $l, $sl, $ss, $ls, $an, $ad)
    }
    $lines += ""
    $lines += "## The anchor guard"
    $lines += ""
    $lines += "Six independent llama points, same binary and flags as the 2026-08-11 quiesced curve. A drifted box cannot reproduce six at once; if several deviate in the same direction, read the stoke/llama column (measured minutes apart on the same box) and not the absolutes."
    $lines += ""
    $bm = Median @($bridge | ForEach-Object { $_.tok_s })
    if ($bm) {
        $bd = 100.0 * ($bm - $ANCHOR_BRIDGE) / $ANCHOR_BRIDGE
        $lines += ("**Bridge cell** -- stoke t=6, n=${BridgeN}: **{0}** against the post-Q6_K ladder value of {1}, a deviation of **{2:n1}%**. This ties the curve to the P-BAR ladder without re-running it." -f $bm, $ANCHOR_BRIDGE, $bd)
        $lines += ""
    }
    $lines += "## Per-cell spread"
    $lines += ""
    $lines += "| cell | runs | min | max | spread |"
    $lines += "|---|---|---|---|---|"
    foreach ($t in $TVALS) {
        foreach ($e in @("S", "L")) {
            $v = @($rate | Where-Object { $_.threads -eq $t -and $_.engine -eq $e } | ForEach-Object { $_.tok_s })
            if ($v.Count -gt 0) {
                $mn = ($v | Measure-Object -Minimum).Minimum
                $mx = ($v | Measure-Object -Maximum).Maximum
                $sp = if ($mn) { "{0:n2}%" -f (100.0 * ($mx - $mn) / $mn) } else { "n/a" }
                $lines += ("| t=$t $e | $($v.Count) | $mn | $mx | $sp |")
            }
        }
    }
    $lines += ""
    $lines += "A cell whose spread is wide relative to the differences being read is not evidence, however good its median looks."
    $lines += ""
    $lines += "Bank: ``$Stamp-bank.jsonl``. Raw log: ``$Stamp-raw.txt``."
    ($lines -join "`r`n") | Out-File -Encoding utf8 $rep
    Say "REPORT written -> $rep"
    Say "DONE"
}
