# M4.5 -- does spinning before blocking stop the pool drifting onto E-cores?
#
# > THE FINDING THIS TESTS (2026-08-17). A forward pass issues ~250 matvec
# dispatches per token, so with a pure condvar wait each helper slept and woke
# thousands of times a second. On this 6P+4E box Windows reads a thread that
# sleeps that often as low-utilisation work and prefers an E-CORE for it, so the
# whole pool drifts onto the slow cores. Measured on one binary minutes apart:
# unpinned 2.39/2.45 tok/s vs 15.79/15.76 pinned to P-cores -- a 6.6x swing,
# with the sick runs burning 74 CPU-seconds to the healthy runs' 26 at ~45% duty
# cycle. Raising process priority recovered part of it (2.39 -> 5.60), which is
# the tell that this is a PLACEMENT decision, not arithmetic.
#
# llama.cpp does not exhibit it because ggml's workers spin, so they read as
# CPU-hungry and keep the P-cores. This probe asks whether giving stoke the same
# wait shape removes stoke's exposure.
#
# > WHAT WOULD FALSIFY IT: spin=0 and spin=2000 measuring the same unpinned.
# Then the wait shape is not what places the threads, and the 6.6x has some
# other cause -- which is the position three earlier hypotheses ended in
# (hyperthread stacking, disk residency, working-set trimming), each killed by
# its own control. The pinned arm is the ceiling: whatever spinning achieves
# unpinned should be read against it, not against the sick number.

param(
    [int]$Passes = 3,
    [int]$NGen   = 64,
    [int]$Threads = 6,
    [string]$Stamp = "2026-08-17-m45-spin-probe"
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$tmpE = "$env:TEMP\stoke-spin-e.txt"
$tmpO = "$env:TEMP\stoke-spin-o.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE = "$root\build\win\stoke.exe"
$LB    = "$root\bench\vendor\llama-b10037\llama-bench.exe"
$PROMPT = "$root\tests\data\parity\p00.txt"

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}

# mask 0 means "leave affinity alone" (all logical processors).
$CONFIGS = @(
    @{ name = "unpinned spin=0";     mask = 0xFFFF; spin = "0" },
    @{ name = "unpinned spin=2000";  mask = 0xFFFF; spin = "2000" },
    @{ name = "unpinned spin=20000"; mask = 0xFFFF; spin = "20000" },
    @{ name = "0xFFF spin=2000";     mask = 0xFFF;  spin = "2000" }
)

$self = Get-Process -Id $PID
$rows = @()
Say "START passes=$Passes n=$NGen threads=$Threads"

for ($pass = 1; $pass -le $Passes; $pass++) {
    # Rotate config order so position cannot be read as a config effect.
    $order = @()
    for ($i = 0; $i -lt $CONFIGS.Count; $i++) {
        $order += $CONFIGS[($i + $pass - 1) % $CONFIGS.Count]
    }
    foreach ($cfg in $order) {
        $self.ProcessorAffinity = [IntPtr]$cfg.mask
        [Environment]::SetEnvironmentVariable('STOKE_POOL_SPIN', $cfg.spin)
        $p = Start-Process -FilePath $STOKE -ArgumentList @(
                "batch", ('"' + $Q4 + '"'), "--raw", "-n", "$NGen",
                "-f", ('"' + $PROMPT + '"'), "--seqs", "1",
                "--threads", "$Threads", "--kv", "f32", "--timing") `
             -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO `
             -NoNewWindow -PassThru
        $cpu = 0
        while (-not $p.HasExited) {
            try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
            Start-Sleep -Milliseconds 300
        }
        $t = (Get-Content $tmpE -Raw)
        $m = [regex]::Match($t, '\(([\d.]+)\s+tok/s\s+aggregate\)')
        $rate = if ($m.Success) { [double]$m.Groups[1].Value } else { $null }
        $rows += [pscustomobject]@{ pass = $pass; cfg = $cfg.name; rate = $rate
                                    cpu = [math]::Round($cpu, 1) }
        Say ("  p{0} {1,-22} {2,7} tok/s   cpu {3,5}s" -f $pass, $cfg.name,
             $(if ($null -eq $rate) { "FAIL" } else { $rate }), [math]::Round($cpu, 1))
    }
    # llama in the same interleave: the arm that never exhibited the collapse.
    $self.ProcessorAffinity = [IntPtr]0xFFFF
    Start-Process -FilePath $LB -ArgumentList @("-m", ('"' + $Q4 + '"'), "-p", "0",
        "-n", "$NGen", "-r", "3", "-t", "$Threads") `
        -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO -NoNewWindow -Wait
    $lm = [regex]::Match((Get-Content $tmpO -Raw), 'tg\d+\s*\|\s*([\d.]+)')
    $lr = if ($lm.Success) { [double]$lm.Groups[1].Value } else { $null }
    $rows += [pscustomobject]@{ pass = $pass; cfg = "llama unpinned"; rate = $lr; cpu = 0 }
    Say ("  p{0} {1,-22} {2,7} tok/s" -f $pass, "llama unpinned", $lr)
}
[Environment]::SetEnvironmentVariable('STOKE_POOL_SPIN', $null)
$self.ProcessorAffinity = [IntPtr]0xFFFF

function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return [math]::Round(($s[$n / 2 - 1] + $s[$n / 2]) / 2.0, 3)
}

$lines = @()
$lines += "# M4.5 -- spin-before-block probe"
$lines += ""
$lines += "$Passes passes, config order rotating, $Threads threads, n=$NGen, N=1. ``STOKE_POOL_SPIN`` sets the pool's spin budget; 0 is the pre-2026-08-17 pure-condvar wait."
$lines += ""
$lines += "| config | median tok/s | min | max | mean CPU-s |"
$lines += "|---|---|---|---|---|"
foreach ($n in (@($CONFIGS | ForEach-Object { $_.name }) + @("llama unpinned"))) {
    $v = @($rows | Where-Object { $_.cfg -eq $n -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    if ($v.Count -gt 0) {
        $c = @($rows | Where-Object { $_.cfg -eq $n } | ForEach-Object { $_.cpu })
        $lines += ("| {0} | {1} | {2} | {3} | {4} |" -f $n, (Median $v),
                   ($v | Measure-Object -Minimum).Minimum,
                   ($v | Measure-Object -Maximum).Maximum,
                   [math]::Round(($c | Measure-Object -Average).Average, 1))
    }
}
$lines += ""
$lines += "**Reading it.** The ``0xFFF spin=2000`` row is the ceiling -- affinity already excludes the E-cores, so it shows what the machine can do. If ``unpinned spin=2000`` approaches that row while ``unpinned spin=0`` stays low, the wait shape is what was placing the pool on E-cores. If spin=0 and spin=2000 agree, the hypothesis is dead and the 6.6x has some other cause."
$lines += ""
$lines += "CPU-seconds is the second signal, and an independent one: the same work costs MORE processor-time on a slower core, so a sick run reads high there (74s vs 26s when this was first measured) while also finishing later."
$lines += ""
$lines += "Raw log: ``$Stamp-raw.txt``."
($lines -join "`r`n") | Out-File -Encoding utf8 $rep
Say "REPORT -> $rep"
Say "DONE"
