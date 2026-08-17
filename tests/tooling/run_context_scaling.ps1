# M4.5 -- how does each engine's per-token cost grow with GENERATION LENGTH?
#
# > WHY THIS IS THE QUESTION NOW. P-BAR was graded three ways on 2026-08-17 and
# missed all three at essentially the same value: 0.827 at t=6 pinned n=128,
# 0.828 at t=8 unpinned n=128. Thread count is therefore NOT where the remaining
# gap lives. But at n=64 the same t=8 configuration measured 0.892, comfortably
# over the bar -- so the gap is a function of n:
#
#     stoke t=8   n=64  16.716 (59.8 ms/tok)   n=128  15.492 (64.5 ms/tok)  +4.7
#     llama t=8   n=64  18.73  (53.4 ms/tok)   n=128  18.70  (53.5 ms/tok)  +0.1
#
# llama's per-token cost is FLAT as generation doubles; stoke's grows 4.7 ms.
# The whole deficit at n=128 is 11 ms/token, so this differential is most of it,
# and closing it would put stoke at ~0.89. That makes attention over the KV
# cache the suspect -- which is step (2) of the ruling's own 7.5 attribution
# ladder ("attention share profile; if >=15%, execute L4 then re-measure"), not
# rung D.
#
# ! BUT THOSE TWO ROWS COME FROM DIFFERENT SITTINGS, which is precisely the
# comparison this project's measurement discipline forbids -- a before/after
# separated by real work measures the machine, not the code. So the differential
# above is a LEAD, not a number. This script settles it by measuring every n in
# ONE interleaved sitting, which is the only form in which it counts.
#
# > WHAT WOULD FALSIFY THE LEAD: stoke's ms/token growing no faster than
# llama's across n. Then the n=64 0.892 was a sitting artefact, the gap is flat
# in n after all, and the attribution goes back to per-thread matvec work.
#
# n=256 is included deliberately: two points can only ever draw a line, and the
# claim is about a SLOPE. If the effect is real it should keep going.

param(
    [int]$Cycles = 3,
    [string]$Ns  = "64,128,256",
    [int]$Threads = 8,
    [string]$Stamp = "2026-08-17-m45-context-scaling"
)

$ErrorActionPreference = 'Continue'
$NVALS = @($Ns -split '\s*,\s*' | Where-Object { $_ } | ForEach-Object { [int]$_ })

$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$tmpE = "$env:TEMP\stoke-ctx-e.txt"
$tmpO = "$env:TEMP\stoke-ctx-o.txt"
$Q4     = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE  = "$root\build\win\stoke.exe"
$LB     = "$root\bench\vendor\llama-b10037\llama-bench.exe"
$PROMPT = "$root\tests\data\parity\p00.txt"

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return [math]::Round(($s[$n / 2 - 1] + $s[$n / 2]) / 2.0, 3)
}
function StokeRate($n) {
    Start-Process -FilePath $STOKE -ArgumentList @("batch", ('"' + $Q4 + '"'), "--raw",
        "-n", "$n", "-f", ('"' + $PROMPT + '"'), "--seqs", "1", "--threads", "$Threads",
        "--kv", "f32", "--timing") -RedirectStandardError $tmpE `
        -RedirectStandardOutput $tmpO -NoNewWindow -Wait
    $m = [regex]::Match((Get-Content $tmpE -Raw), '\(([\d.]+)\s+tok/s\s+aggregate\)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}
function LlamaRate($n) {
    Start-Process -FilePath $LB -ArgumentList @("-m", ('"' + $Q4 + '"'), "-p", "0",
        "-n", "$n", "-r", "3", "-t", "$Threads") -RedirectStandardError $tmpE `
        -RedirectStandardOutput $tmpO -NoNewWindow -Wait
    $m = [regex]::Match((Get-Content $tmpO -Raw), 'tg\d+\s*\|\s*([\d.]+)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

Say "START cycles=$Cycles ns=$($NVALS -join '|') threads=$Threads"
$rows = @()
for ($c = 1; $c -le $Cycles; $c++) {
    # Warm and check, then rotate n-order and engine-order so neither position
    # in the cycle nor engine order can be read as an n effect.
    $null = StokeRate 8
    $sb = [math]::Round((Get-Counter '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue / 1GB, 2)
    Say "CYCLE $c (standby $sb GB)"
    $order = if ($c % 2 -eq 1) { $NVALS } else { $NVALS[($NVALS.Count - 1)..0] }
    foreach ($n in $order) {
        $pair = if ($c -le 2) { @("S", "L") } else { @("L", "S") }
        foreach ($eng in $pair) {
            $v = if ($eng -eq "S") { StokeRate $n } else { LlamaRate $n }
            $rows += [pscustomobject]@{ cycle = $c; n = $n; engine = $eng; rate = $v }
            Say ("  c{0} n={1,-4} {2}  {3}" -f $c, $n, $eng, $(if ($null -eq $v) { "PARSE-FAIL" } else { $v }))
        }
    }
}

$lines = @()
$lines += "# M4.5 -- per-token cost vs generation length, one interleaved sitting"
$lines += ""
$lines += "$Cycles cycles, n-order and engine-order both rotating, $Threads threads, unpinned. stoke = ``batch --seqs 1 --kv f32``; llama = ``llama-bench -p 0 -r 3``. Median per cell."
$lines += ""
$lines += "| n | stoke tok/s | stoke ms/tok | llama tok/s | llama ms/tok | stoke/llama |"
$lines += "|---|---|---|---|---|---|"
$base = @{}
foreach ($n in $NVALS) {
    $s = Median @($rows | Where-Object { $_.n -eq $n -and $_.engine -eq "S" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $l = Median @($rows | Where-Object { $_.n -eq $n -and $_.engine -eq "L" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $sm = if ($s) { [math]::Round(1000.0 / $s, 2) } else { $null }
    $lm = if ($l) { [math]::Round(1000.0 / $l, 2) } else { $null }
    $base[$n] = @{ s = $sm; l = $lm }
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} |" -f $n, $s, $sm, $l, $lm,
               $(if ($s -and $l) { "{0:n3}x" -f ($s / $l) } else { "n/a" }))
}
$lines += ""
$lines += "## The slope, which is the whole question"
$lines += ""
# ASCII only in this file. Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI,
# so a UTF-8 "delta" arrives as two bytes -- and the second renders as U+201D,
# which PowerShell accepts as a STRING DELIMITER. That terminated this very line
# and turned the rest into a bare token. Non-ASCII in a .ps1 is a parse hazard,
# not a style question.
$lines += "| from -> to | stoke d(ms/tok) | llama d(ms/tok) | stoke excess |"
$lines += "|---|---|---|---|"
for ($i = 1; $i -lt $NVALS.Count; $i++) {
    $a = $NVALS[$i - 1]; $b = $NVALS[$i]
    if ($base[$a].s -and $base[$b].s -and $base[$a].l -and $base[$b].l) {
        $ds = [math]::Round($base[$b].s - $base[$a].s, 2)
        $dl = [math]::Round($base[$b].l - $base[$a].l, 2)
        $lines += ("| $a -> $b | {0} | {1} | {2} |" -f $ds, $dl, [math]::Round($ds - $dl, 2))
    }
}
$lines += ""
$lines += "**Reading it.** If stoke's ms/token climbs while llama's stays flat, the remaining P-BAR gap is stoke's cost per unit of CONTEXT -- attention over the KV cache -- and the next lever is sec 7.5 step (2)/L4, not rung D. If both climb together, the lead measured across sittings was an artefact and attribution returns to per-thread matvec work."
$lines += ""
$lines += "Every n here was measured in ONE sitting, interleaved. The lead that prompted this run came from two different sittings, which is the comparison this box has repeatedly punished."
$lines += ""
$lines += "Raw log: ``$Stamp-raw.txt``."
($lines -join "`r`n") | Out-File -Encoding utf8 $rep
Say "REPORT -> $rep"
Say "DONE"
