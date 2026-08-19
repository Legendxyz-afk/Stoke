# M5 rung E3 -- what is chunked prefill worth? A/B on ONE binary.
#
# WHY THIS IS SEPARATE FROM run_m5_e2_ab.ps1. That harness reads the decode
# aggregate, and E3 changes NOTHING about decode -- its entire effect is in
# prefill, which that line reports separately and which the A/B ignores. A
# measurement that cannot see the thing it is about is worse than none.
#
# ============================ THE PREDICTION ============================
# Pre-registered so the run can refute it.
#
#   Prefill was position-by-position, and a position streams the whole
#   2,497,280,736 B of weights: 57.3 ms of pure traffic each at W=43.6 GB/s.
#   Chunking at 128 amortises that term by the chunk width, leaving attention
#   (which grows with position and is NOT amortised) plus compute.
#
#   For a prompt of P tokens the weight term goes P*57.3 ms -> ceil(P/128)*57.3.
#   At p40 (490 tokens): 28.1 s -> 0.23 s of weight traffic. The floor is then
#   set by attention and by arithmetic, neither of which this rung touches, so
#   the SPEEDUP IS BOUNDED BY WHAT FRACTION OF PREFILL WAS WEIGHT STREAMING --
#   it is emphatically not 128x, and a result anywhere near 128x would mean the
#   measurement is wrong, not that the rung is brilliant.
#
#   No halved figure is quoted here because the argument gives a BOUND, not an
#   estimate: the standing "price it at half" rule exists for arguments that
#   predict a specific gain, and this one predicts a ceiling.
#
#   ▶ THE FALSIFIER: prefill tokens/second must rise with PROMPT LENGTH under
#   chunking and be roughly flat without it. A short prompt (p00, 23 tokens)
#   spans one chunk and must show almost nothing.
# =======================================================================

param(
    [int]$Cycles = 3,
    [int]$NGen = 8,          # small: decode is not the subject and costs time
    [int]$Threads = 6,
    [switch]$NoPin,
    [string]$Stamp = "2026-08-19-m5-e3-prefill-ab"
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$tmpE = "$env:TEMP\stoke-e3ab-e.txt"
$tmpO = "$env:TEMP\stoke-e3ab-o.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE = "$root\build\win\stoke.exe"

$nlog = [int]$env:NUMBER_OF_PROCESSORS
$affMask = if ($NoPin) { [IntPtr]([int64][math]::Pow(2, $nlog) - 1) } else { [IntPtr]0xFFF }
(Get-Process -Id $PID).ProcessorAffinity = $affMask

$POINTS = @(
    @{ name = "p00";  file = "$root\tests\data\parity\p00.txt";        ptok = 23 },
    @{ name = "p40";  file = "$root\tests\data\parity\p40.txt";        ptok = 490 },
    @{ name = "long"; file = "$root\tests\data\ctx-profile-long.txt";  ptok = 1436 }
)

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
# Returns @(prefill_tokens, prefill_ms) -- BOTH, because the token count is what
# makes the ms comparable: prefix dedup can legitimately shorten a prefill, and
# a faster run over fewer tokens is not a faster prefill.
function PrefillMs($file, $chunk) {
    [Environment]::SetEnvironmentVariable('STOKE_PREFILL_CHUNK', $chunk)
    Start-Process -FilePath $STOKE -ArgumentList @("batch", ('"' + $Q4 + '"'), "--raw",
        "-n", "$NGen", "-f", ('"' + $file + '"'), "--seqs", "1", "--threads", "$Threads",
        "--kv", "f32", "--timing") -RedirectStandardError $tmpE `
        -RedirectStandardOutput $tmpO -NoNewWindow -Wait
    $m = [regex]::Match((Get-Content $tmpE -Raw),
                        'prefill (\d+) tok in ([\d.]+) ms')
    if ($m.Success) { return @([int]$m.Groups[1].Value, [double]$m.Groups[2].Value) }
    return @($null, $null)
}

Say ("START cycles=$Cycles n=$NGen threads=$Threads affinity=0x{0:X} ({1})" -f
     [int64]$affMask, $(if ($NoPin) { "unpinned" } else { "PINNED to P-cores" }))

$rows = @()
for ($c = 1; $c -le $Cycles; $c++) {
    Say "CYCLE $c"
    $order = if ($c % 2 -eq 1) { $POINTS } else { $POINTS[($POINTS.Count - 1)..0] }
    foreach ($pt in $order) {
        # Arm order rotates with the cycle so a thermal ramp cannot be read as
        # a gain for whichever arm consistently ran second.
        $arms = if ($c % 2 -eq 1) { @('1', '128') } else { @('128', '1') }
        foreach ($chunk in $arms) {
            $r = PrefillMs $pt.file $chunk
            $rows += [pscustomobject]@{ cycle = $c; name = $pt.name; arm = "chunk$chunk"
                                        toks = $r[0]; ms = $r[1] }
            Say ("  c{0} {1,-5} chunk={2,-3}  {3}" -f $c, $pt.name, $chunk,
                 $(if ($null -eq $r[1]) { "PARSE-FAIL" } else { "$($r[0]) tok in $($r[1]) ms" }))
        }
    }
}
[Environment]::SetEnvironmentVariable('STOKE_PREFILL_CHUNK', $null)

$lines = @()
$lines += "# M5 E3 -- chunked prefill A/B"
$lines += ""
$lines += ("$Cycles cycles, point-order and arm-order both rotating, $Threads threads. One binary; " +
           "``STOKE_PREFILL_CHUNK=1`` is the pre-M5 position-by-position path. Both widths verified " +
           "to write a bit-identical KV cache (``m5_e3_chunked_prefill_...`` and the entry byte-compare), " +
           "so this is speed only. Affinity {0}." -f
           $(if ($NoPin) { "unpinned" } else { "**PINNED 0xFFF**" }))
$lines += ""
$lines += "| prompt | prefill tok | chunk 1 | chunk 128 | speedup | tok/s at 128 |"
$lines += "|---|---|---|---|---|---|"
foreach ($pt in $POINTS) {
    $a = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.arm -eq "chunk1" -and $null -ne $_.ms } | ForEach-Object { $_.ms })
    $b = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.arm -eq "chunk128" -and $null -ne $_.ms } | ForEach-Object { $_.ms })
    $tk = @($rows | Where-Object { $_.name -eq $pt.name -and $null -ne $_.toks } | ForEach-Object { $_.toks }) | Select-Object -First 1
    # Token counts must AGREE across arms, or the ms are not comparable.
    $tset = @($rows | Where-Object { $_.name -eq $pt.name -and $null -ne $_.toks } | ForEach-Object { $_.toks } | Sort-Object -Unique)
    $warn = if ($tset.Count -gt 1) { " !! token counts differ: $($tset -join '/')" } else { "" }
    $sp = if ($a -and $b) { "{0:n2}x" -f ($a / $b) } else { "n/a" }
    $rate = if ($b -and $tk) { "{0:n1}" -f ($tk / ($b / 1000.0)) } else { "n/a" }
    $lines += ("| {0} | {1} | {2} ms | {3} ms | **{4}** | {5}{6} |" -f $pt.name, $tk, $a, $b, $sp, $rate, $warn)
}
$lines += ""
$lines += "## Against the pre-registered prediction"
$lines += ""
$lines += "The weight term goes ``P x 57.3 ms`` to ``ceil(P/128) x 57.3 ms``; attention and arithmetic are untouched, so the speedup is BOUNDED by what fraction of prefill was weight streaming. A figure anywhere near 128x means the measurement is wrong, not that the rung is brilliant."
$lines += ""
$lines += "**The falsifier:** prefill tok/s must RISE with prompt length under chunking and be roughly flat without it. p00 (23 tokens) spans a single chunk and must show almost nothing -- if it shows a large gain, the win is not coming from weight amortisation."
$lines += ""
$lines += "Raw log: ``$Stamp-raw.txt``."
($lines -join "`r`n") | Out-File -Encoding utf8 $rep
Say "REPORT -> $rep"
Say "DONE"
