# M4.5 -- what is the interleaved QK loop worth? A/B on ONE binary.
#
# STOKE_ATT_LANES=1 is the pre-2026-08-17 one-position-at-a-time loop;
# STOKE_ATT_LANES=4 scores four independent positions per pass with four
# accumulators, each still summed left-to-right in vec_dot_f32's exact order.
# Verified byte-identical to the pre-change archive at both settings, so this is
# purely a speed question.
#
# Both arms come from one binary and interleave inside one sitting, because a
# before/after separated by real work measures this box and not the code.
#
# p40 (490 prompt tokens, avg ctx 554) carries most of the signal at a fraction
# of the cost: attention is already 76.5% of decode there, while the 1436-token
# prompt costs MINUTES in prefill alone -- prefill attention is O(n^2) and runs
# on the same loop, which is its own finding.

param(
    [int]$Cycles = 3,
    [int]$NGen = 128,
    [int]$Threads = 8,
    [switch]$IncludeLong,
    [string]$Stamp = "2026-08-17-m45-att-lanes-ab"
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$tmpE = "$env:TEMP\stoke-lanes-e.txt"
$tmpO = "$env:TEMP\stoke-lanes-o.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE = "$root\build\win\stoke.exe"
$LB    = "$root\bench\vendor\llama-b10037\llama-bench.exe"

$POINTS = @(
    @{ name = "p00"; file = "$root\tests\data\parity\p00.txt"; ptok = 23;  ctx = 87 },
    @{ name = "p40"; file = "$root\tests\data\parity\p40.txt"; ptok = 490; ctx = 554 }
)
if ($IncludeLong) {
    $POINTS += @{ name = "long"; file = "$root\tests\data\ctx-profile-long.txt"; ptok = 1436; ctx = 1500 }
}

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return [math]::Round(($s[$n / 2 - 1] + $s[$n / 2]) / 2.0, 4)
}
function StokeRate($file, $lanes) {
    [Environment]::SetEnvironmentVariable('STOKE_ATT_LANES', $lanes)
    Start-Process -FilePath $STOKE -ArgumentList @("batch", ('"' + $Q4 + '"'), "--raw",
        "-n", "$NGen", "-f", ('"' + $file + '"'), "--seqs", "1", "--threads", "$Threads",
        "--kv", "f32", "--timing") -RedirectStandardError $tmpE `
        -RedirectStandardOutput $tmpO -NoNewWindow -Wait
    $m = [regex]::Match((Get-Content $tmpE -Raw), '\(([\d.]+)\s+tok/s\s+aggregate\)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}
function LlamaRate($depth) {
    Start-Process -FilePath $LB -ArgumentList @("-m", ('"' + $Q4 + '"'), "-p", "0",
        "-n", "$NGen", "-d", "$depth", "-r", "3", "-t", "$Threads") `
        -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO -NoNewWindow -Wait
    $m = [regex]::Match((Get-Content $tmpO -Raw), 'tg\d+[^|]*\|\s*([\d.]+)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

Say "START cycles=$Cycles n=$NGen threads=$Threads points=$($POINTS.Count)"
$rows = @()
for ($c = 1; $c -le $Cycles; $c++) {
    Say "CYCLE $c"
    $order = if ($c % 2 -eq 1) { $POINTS } else { $POINTS[($POINTS.Count - 1)..0] }
    foreach ($pt in $order) {
        # Rotate which lane setting goes first so position cannot be read as gain.
        $arms = if ($c % 2 -eq 1) { @('1', '4') } else { @('4', '1') }
        foreach ($lanes in $arms) {
            $v = StokeRate $pt.file $lanes
            $rows += [pscustomobject]@{ cycle = $c; name = $pt.name; ctx = $pt.ctx; arm = "lanes$lanes"; rate = $v }
            Say ("  c{0} {1,-5} lanes={2}  {3}" -f $c, $pt.name, $lanes, $(if ($null -eq $v) { "PARSE-FAIL" } else { $v }))
        }
        $lv = LlamaRate $pt.ptok
        $rows += [pscustomobject]@{ cycle = $c; name = $pt.name; ctx = $pt.ctx; arm = "llama"; rate = $lv }
        Say ("  c{0} {1,-5} llama    {2}" -f $c, $pt.name, $lv)
    }
}
[Environment]::SetEnvironmentVariable('STOKE_ATT_LANES', $null)

$lines = @()
$lines += "# M4.5 -- interleaved QK loop (ATT_LANES) A/B"
$lines += ""
$lines += "$Cycles cycles, point-order and arm-order both rotating, $Threads threads, n=$NGen. One binary; ``STOKE_ATT_LANES=1`` is the old loop. Both settings verified byte-identical to the pre-change archive, so this is speed only."
$lines += ""
$lines += "| point | avg ctx | lanes=1 | lanes=4 | speedup | llama | lanes=4 / llama |"
$lines += "|---|---|---|---|---|---|---|"
foreach ($pt in $POINTS) {
    $a = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.arm -eq "lanes1" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $b = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.arm -eq "lanes4" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $l = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.arm -eq "llama" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $sp = if ($a -and $b) { "{0:n3}x" -f ($b / $a) } else { "n/a" }
    $bl = if ($b -and $l) { "{0:n3}x" -f ($b / $l) } else { "n/a" }
    $lines += ("| {0} | {1} | {2} | {3} | **{4}** | {5} | **{6}** |" -f $pt.name, $pt.ctx, $a, $b, $sp, $l, $bl)
}
$lines += ""
$lines += "The gain should GROW with context, because attention is 24.2% of decode at ctx 87 and 84.6% at ctx 1500 -- a fixed speedup on the attention term shows up as a larger total speedup where attention dominates. A flat gain across contexts would mean the win is not coming from where it was predicted to."
$lines += ""
$lines += "Raw log: ``$Stamp-raw.txt``."
($lines -join "`r`n") | Out-File -Encoding utf8 $rep
Say "REPORT -> $rep"
Say "DONE"
