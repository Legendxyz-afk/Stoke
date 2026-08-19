# M5 rung E2 -- what is kv-head-outer attention worth? A/B on ONE binary.
#
# STOKE_ATT_GQA=0 is M4.5's q-head-outer loop, which streams each kv-head's K
# and V once per Q-HEAD (four times over, at GQA 4:1). STOKE_ATT_GQA=1 walks
# kv-head-outer and fetches them once per GROUP. Both are verified byte-
# identical to the archives by run_m5_e2_verify.ps1, so this is purely a speed
# question.
#
# Both arms come from ONE binary and interleave inside ONE sitting: a
# before/after separated by real work measures this box and not the code, and
# the pre-`76dee7f` archive is on record swinging 10.9x between sittings.
#
# ============================ THE PREDICTION ============================
# Pre-registered so the run can refute it rather than be read to agree with it.
#
#   (1) L4 measured decode as a LINE: 106.75 ms/token at ctx 1500 with slope
#       0.0289 ms/context-token, so intercept 63.40 ms. Both halves are
#       independently bandwidth-bound (intercept 90.4% of ceiling on 2.4973 GB
#       of weights; slope 93.6% of ceiling on 36 x 32 x 1024 B per context
#       token). E2 divides the SLOPE's byte count by gqa_group and leaves the
#       intercept alone: slope 0.0289 -> 0.00723 at the same 93.6% efficiency.
#
#         ctx  87:  65.9 -> 64.0 ms   argued 1.03x   priced at half 1.015x
#         ctx 554:  79.4 -> 67.4 ms   argued 1.18x   priced at half 1.09x
#         ctx 1500: 106.75 -> 74.2 ms argued 1.44x   priced at half 1.22x
#
#       The halving is the standing rule: this project's kernel arguments run
#       1.5-2x hot (L4 predicted 317->63 and measured 106.75; A-prime 0.907x;
#       Q6_K 1.108x against 1.20x).
#
#   (2) THE CURVE IS A BETTER TEST THAN THE HEADLINE RATIO. E2 removes a term
#       proportional to context and touches nothing constant, so the gain MUST
#       be ~1.0x at ctx 87 and grow from there. A large, flat gain across
#       contexts REFUTES the mechanism even while looking like a win -- it
#       would mean the speedup came from somewhere the argument does not name,
#       and the argument is what the next rung gets built on.
#
#   (3) N=8 IS THE RISK POINT, NOT THE REWARD POINT. E2 drops the unit count
#       from n_head (32) to n_head_kv (8) per sequence, so each claim is 4x
#       larger and an E-core straggler holds four heads at the barrier instead
#       of one. At N=1 that is 8 units over 6-8 workers -- the tightest it ever
#       gets. If E2 regresses anywhere, expect it here first.
# =======================================================================

param(
    [int]$Cycles = 3,
    [int]$NGen = 128,
    [int]$Threads = 6,
    [int]$SeqsPoint = 8,
    # PINNED BY DEFAULT here, unlike the P-BAR harness: this is a KERNEL A/B,
    # and E-core migration produced a 5.8x spread on identical runs on an idle
    # box. The P-BAR run wants the shipping configuration; this one wants the
    # quietest instrument that can see a 1.2x effect.
    [switch]$NoPin,
    [switch]$SkipLong,
    [string]$Stamp = "2026-08-19-m5-e2-ab"
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$tmpE = "$env:TEMP\stoke-e2ab-e.txt"
$tmpO = "$env:TEMP\stoke-e2ab-o.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE = "$root\build\win\stoke.exe"
$LB    = "$root\bench\vendor\llama-b10037\llama-bench.exe"

$nlog = [int]$env:NUMBER_OF_PROCESSORS
$affMask = if ($NoPin) { [IntPtr]([int64][math]::Pow(2, $nlog) - 1) } else { [IntPtr]0xFFF }
(Get-Process -Id $PID).ProcessorAffinity = $affMask

# ctx values are prompt tokens + NGen/2, i.e. the AVERAGE context a decode step
# sees, which is what the traffic term is proportional to -- not the final one.
$POINTS = @(
    @{ name = "p00";  file = "$root\tests\data\parity\p00.txt";        ptok = 23;   ctx = 87;   seqs = 1 },
    @{ name = "p40";  file = "$root\tests\data\parity\p40.txt";        ptok = 490;  ctx = 554;  seqs = 1 },
    @{ name = "p40x$SeqsPoint"; file = "$root\tests\data\parity\p40.txt"; ptok = 490; ctx = 554; seqs = $SeqsPoint }
)
if (-not $SkipLong) {
    $POINTS += @{ name = "long"; file = "$root\tests\data\ctx-profile-long.txt"; ptok = 1436; ctx = 1500; seqs = 1 }
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
function StokeRate($file, $gqa, $seqs) {
    # Start-Process with -RedirectStandardError, NOT `2> file`: PowerShell wraps
    # a native exe's stderr at console width, so a regex can fail purely because
    # the binary's name is long. Recorded trap, not a precaution.
    [Environment]::SetEnvironmentVariable('STOKE_ATT_GQA', $gqa)
    Start-Process -FilePath $STOKE -ArgumentList @("batch", ('"' + $Q4 + '"'), "--raw",
        "-n", "$NGen", "-f", ('"' + $file + '"'), "--seqs", "$seqs", "--threads", "$Threads",
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

Say ("START cycles=$Cycles n=$NGen threads=$Threads points=$($POINTS.Count) " +
     "affinity=0x{0:X} over $nlog logical ({1})" -f [int64]$affMask,
     $(if ($NoPin) { "unpinned" } else { "PINNED to P-cores" }))

$rows = @()
for ($c = 1; $c -le $Cycles; $c++) {
    Say "CYCLE $c"
    # Point order AND arm order both rotate: either one alone leaves a thermal
    # ramp readable as a gain for whichever arm ran second.
    $order = if ($c % 2 -eq 1) { $POINTS } else { $POINTS[($POINTS.Count - 1)..0] }
    foreach ($pt in $order) {
        $arms = if ($c % 2 -eq 1) { @('0', '1') } else { @('1', '0') }
        foreach ($gqa in $arms) {
            $v = StokeRate $pt.file $gqa $pt.seqs
            $rows += [pscustomobject]@{ cycle = $c; name = $pt.name; ctx = $pt.ctx; arm = "gqa$gqa"; rate = $v }
            Say ("  c{0} {1,-8} gqa={2}  {3}" -f $c, $pt.name, $gqa, $(if ($null -eq $v) { "PARSE-FAIL" } else { $v }))
        }
        # llama only at the N=1 points: llama-bench has no N-way batch here, and
        # a reference that does not measure the same thing is worse than none.
        if ($pt.seqs -eq 1) {
            $lv = LlamaRate $pt.ptok
            $rows += [pscustomobject]@{ cycle = $c; name = $pt.name; ctx = $pt.ctx; arm = "llama"; rate = $lv }
            Say ("  c{0} {1,-8} llama    {2}" -f $c, $pt.name, $lv)
        }
    }
}
[Environment]::SetEnvironmentVariable('STOKE_ATT_GQA', $null)

$lines = @()
$lines += "# M5 E2 -- kv-head-outer attention A/B"
$lines += ""
$lines += ("$Cycles cycles, point-order and arm-order both rotating, $Threads threads, n=$NGen. " +
           "One binary; ``STOKE_ATT_GQA=0`` is M4.5's q-head-outer loop. Both arms verified " +
           "byte-identical to the archives (``run_m5_e2_verify.ps1``), so this is speed only. " +
           "Affinity {0} over $nlog logical processors." -f
           $(if ($NoPin) { "unpinned" } else { "**PINNED 0xFFF (P-core siblings)**" }))
$lines += ""
$lines += "| point | N | avg ctx | gqa=0 | gqa=1 | speedup | llama | gqa=1 / llama |"
$lines += "|---|---|---|---|---|---|---|---|"
foreach ($pt in $POINTS) {
    $a = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.arm -eq "gqa0" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $b = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.arm -eq "gqa1" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $l = Median @($rows | Where-Object { $_.name -eq $pt.name -and $_.arm -eq "llama" -and $null -ne $_.rate } | ForEach-Object { $_.rate })
    $sp = if ($a -and $b) { "{0:n3}x" -f ($b / $a) } else { "n/a" }
    $bl = if ($b -and $l) { "{0:n3}x" -f ($b / $l) } else { "n/a" }
    $lines += ("| {0} | {1} | {2} | {3} | {4} | **{5}** | {6} | **{7}** |" -f
               $pt.name, $pt.seqs, $pt.ctx, $a, $b, $sp, $(if ($l) { $l } else { "--" }), $bl)
}
$lines += ""
$lines += "## Against the pre-registered prediction"
$lines += ""
$lines += "1. **Argued 1.03x / 1.18x / 1.44x at ctx 87 / 554 / 1500; priced at half, 1.015x / 1.09x / 1.22x.** Derived by dividing the measured slope's byte count by gqa_group and leaving the intercept alone. A number near 1.44x would be the FIRST kernel argument here to land at full value and should be treated as suspicious until re-run, not as vindication."
$lines += "2. **The SHAPE matters more than the headline.** E2 removes a term proportional to context; the weight term it leaves alone is constant. A flat gain across p00/p40/long refutes the mechanism even if the totals look good."
$lines += "3. **N=$SeqsPoint is the risk point.** The unit count per sequence falls 32 -> 8, so a straggler holds 4x more work at the barrier. A regression is expected here before anywhere else, and `STOKE_ATT_CHUNK` is the knob if one appears."
$lines += ""
$lines += "Raw log: ``$Stamp-raw.txt``."
($lines -join "`r`n") | Out-File -Encoding utf8 $rep
Say "REPORT -> $rep"
Say "DONE"
