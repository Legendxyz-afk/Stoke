# =====================================================================
# ABLATION: what is `vpmulld` costing this kernel?
#
# A = the shipping SIMD kernel.
# B = the SAME kernel with `_mm256_mullo_epi32(sub, scale)` replaced by
#     `_mm256_add_epi32(sub, scale)`.
#
# ⚠⚠ ARM B COMPUTES THE WRONG ANSWER ON PURPOSE. It is an INSTRUMENT, not a
# candidate. It is never installed as an arm, never shipped, and the build
# directory is deleted after the run. Its output is garbage by construction and
# the script CHECKS that it is garbage (see the power check below) -- an ablation
# whose output matched A would mean the substitution never took effect and every
# number here would be measuring the same binary twice.
#
# WHY `add` AND NOT DELETION: deleting the multiply lets GCC dead-code the
# `scales8` unpack and the `vpbroadcastd`, crediting the ablation with savings no
# real change could collect. `vpaddd` keeps the scale live, the operand count the
# same and the dependency edge into `acc` the same, so the delta prices the
# multiply and nothing else.
#
# WHAT THE NUMBER IS: an UPPER BOUND on what taking the integer scale off the
# critical path could buy. A real redesign still has to apply the scale
# somewhere. Prediction and discriminators are pre-registered in
# docs/superpowers/memos/2026-08-24-m5-mullo-ablation-PREREGISTERED.md and were
# committed before this script was run.
#
# t=1 IS THE HEADLINE. At one thread there is no pool, no barrier and no
# dispatch, so the ratio is the kernel's own. t=8 is carried because it is the
# operating point.
#
# THE PROTOCOL IS COPIED FROM run_m5_mins_ab.ps1 RATHER THAN REFACTORED OUT OF
# IT, deliberately: that script produced committed evidence
# (2026-08-20-m5-mins-ab-v2-report.md) and changing it would put that report's
# provenance in question to save a few lines. The two runs being the same shape
# is the point -- their ratios are meant to be read side by side.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_mullo_ablation.ps1
# =====================================================================
param(
    [int]$CyclesT8 = 4,
    [int]$CyclesT1 = 4,
    [int]$Ctx      = 704,
    [string]$Kv    = "f32",
    [string]$Stamp = ""
)

$ErrorActionPreference = "Stop"
$root   = "M:\New Drive\Desktop\Programming\stoke"
$armA   = "$root\build\win-mins-simd\stoke.exe"
$armB   = "$root\build\win-mullo-ablate\stoke.exe"
$mdl    = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$prompt = "$root\tests\data\bench_prompts\s2-512-c00.txt"
$gr     = "$root\docs\superpowers\gate-reports"
if (-not $Stamp) { $Stamp = "{0}-m5-mullo-ablation" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp-raw.txt"
$bank   = "$gr\$Stamp-bank.jsonl"
$report = "$gr\$Stamp-report.md"
$tmpE   = "$env:TEMP\stoke-mullo-e.txt"
$tmpO   = "$env:TEMP\stoke-mullo-o.txt"
foreach ($pth in @($armA, $armB, $mdl, $prompt)) {
    if (-not (Test-Path $pth)) { throw "missing: $pth" }
}
$FLOPS_PER_TOK = 8.460

function Say($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $msg
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Bank($obj) { ($obj | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 $bank }
function Banked() {
    if (-not (Test-Path $bank)) { return @() }
    $acc = @()
    foreach ($ln in (Get-Content $bank)) {
        if ($ln.Trim()) { try { $acc += ($ln | ConvertFrom-Json) } catch {} }
    }
    return $acc
}
function Median($xs) {
    $srt = @($xs | Sort-Object); $cnt = $srt.Count
    if ($cnt -eq 0) { return $null }
    if ($cnt % 2 -eq 1) { return $srt[[int](($cnt - 1) / 2)] }
    return [math]::Round(($srt[$cnt / 2 - 1] + $srt[$cnt / 2]) / 2.0, 4)
}

# Start-Process for RAW stderr: PowerShell re-wraps a native command's stderr at
# console width, and the two arms' file names differ in length -- which is
# exactly how the 2026-08-14 overnight lost a whole arm to a regex.
function PrefillRate($exePath, $nThreads) {
    $argList = @("batch", ('"' + $mdl + '"'), "-f", ('"' + $prompt + '"'),
                 "--seqs", "1", "--ctx", "$Ctx", "-n", "8",
                 "--threads", "$nThreads", "--kv", $Kv, "--timing")
    $proc = Start-Process -FilePath $exePath -ArgumentList $argList `
                          -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO `
                          -NoNewWindow -PassThru -Wait
    if ($proc.ExitCode -ne 0) { return $null }
    $txt = (Get-Content $tmpE -Raw)
    if (-not $txt) { return $null }
    $mt = [regex]::Match($txt, 'prefill\s+(\d+)\s+tok in\s+([\d.]+)\s+ms')
    if (-not $mt.Success) { return $null }
    $tok = [double]$mt.Groups[1].Value
    $ms  = [double]$mt.Groups[2].Value
    if ($ms -le 0) { return $null }
    return [math]::Round($tok / ($ms / 1000.0), 4)
}

Say "vpmulld ablation. A=$armA"
Say "                  B=$armB   *** B COMPUTES THE WRONG ANSWER BY DESIGN ***"
Say "ctx $Ctx  kv $Kv  t8 cycles $CyclesT8  t1 cycles $CyclesT1"

# ▶ THE POWER CHECK, AND IT RUNS FIRST. If B's stream matched A's, the
# substitution never took effect and every ratio below would be one binary
# measured twice -- a result that would look like "vpmulld is free" and be
# indistinguishable from it. This is withdrawal condition 1 of the
# pre-registration, asserted rather than assumed.
$p00 = "$root\tests\data\parity\p00.txt"
$sA = (& $armA run $mdl --raw -n 8 -f $p00 --threads 6 --kv f32 | Out-String)
$sB = (& $armB run $mdl --raw -n 8 -f $p00 --threads 6 --kv f32 | Out-String)
if ($sA -eq $sB) {
    Say "STOP: the ablation arm produced A's stream. The substitution did not take"
    Say "effect, and a 1.00x from this run would be an artifact. Nothing measured."
    exit 1
}
Say "power check: B's stream DIFFERS from A's -- the ablation is in the binary."

@{ total = (($CyclesT8 + $CyclesT1) * 2); stamp = $Stamp } |
    ConvertTo-Json -Compress | Out-File -Encoding utf8 "$gr\$Stamp.status.json"

foreach ($cfg in @(@{t = 8; n = $CyclesT8}, @{t = 1; n = $CyclesT1})) {
    for ($cyc = 1; $cyc -le $cfg.n; $cyc++) {
        # ARM ORDER ALTERNATES. The box warms across a cycle, so a fixed A-then-B
        # order would charge B the warmer half every time.
        $order = @("A", "B")
        if ($cyc % 2 -eq 0) { $order = @("B", "A") }
        foreach ($arm in $order) {
            $exePath = $armA
            if ($arm -eq "B") { $exePath = $armB }
            $t0 = Get-Date
            $rate = PrefillRate $exePath $cfg.t
            $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
            $id = "t$($cfg.t)-c$cyc-$arm"
            if ($null -eq $rate) {
                Say "  $id PARSE-FAIL after ${secs}s"
                Bank @{ id = $id; kind = "fail"; arm = $arm; threads = $cfg.t; utc = (Get-Date -Format s) }
                continue
            }
            Bank @{ id = $id; kind = "cell"; arm = $arm; threads = $cfg.t; cycle = $cyc
                    rate = $rate; gflops = [math]::Round($rate * $FLOPS_PER_TOK, 1)
                    secs = $secs; utc = (Get-Date -Format s) }
            Say ("  {0,-10} {1,8:n4} tok/s = {2,6:n1} GFLOP/s  ({3}s)" -f
                 $id, $rate, ($rate * $FLOPS_PER_TOK), $secs)
        }
    }
}

$all = @(Banked | Where-Object { $_.kind -eq "cell" })
$lines = @()
$lines += "# What is ``vpmulld`` costing the Q4_K batched kernel?"
$lines += ""
$lines += "Run ``$Stamp``. **A = the shipping SIMD kernel. B = the same kernel with the"
$lines += "integer scale multiply replaced by an add -- B COMPUTES THE WRONG ANSWER and"
$lines += "exists only to be timed.** Prediction pre-registered before the first cell:"
$lines += "``memos/2026-08-24-m5-mullo-ablation-PREREGISTERED.md`` (1.15x, accept 1.10-1.25x;"
$lines += ">=1.25x => p01-throughput-bound, <=1.08x => the scale is a red herring)."
$lines += ""
$lines += "PREFILL of a 525-token prompt, ctx $Ctx, kv $Kv, arms interleaved with the order"
$lines += "alternating per cycle. B/A is an UPPER BOUND on what removing the integer scale"
$lines += "from the critical path could buy: a real redesign still has to apply the scale."
$lines += ""
$lines += "| threads | A tok/s | B tok/s | B/A | A GFLOP/s | B GFLOP/s | pairings B won |"
$lines += "|---|---|---|---|---|---|---|"
foreach ($tt in @(1, 8)) {
    $ca = @($all | Where-Object { $_.arm -eq "A" -and $_.threads -eq $tt })
    $cb = @($all | Where-Object { $_.arm -eq "B" -and $_.threads -eq $tt })
    $ra = Median (@($ca | ForEach-Object { [double]$_.rate }))
    $rb = Median (@($cb | ForEach-Object { [double]$_.rate }))
    if ($null -eq $ra -or $null -eq $rb) { continue }
    # Pairings, not just medians: a median hides whether the win is consistent or
    # is one lucky cell, and 6-of-6 was what made the 08-20 result trustworthy
    # after its own pre-flight had read the other way.
    $won = 0; $tot = 0
    foreach ($c in $ca) {
        $m = @($cb | Where-Object { $_.cycle -eq $c.cycle })
        if ($m.Count -eq 1) { $tot++; if ([double]$m[0].rate -gt [double]$c.rate) { $won++ } }
    }
    $lines += ("| {0} | {1:n4} | {2:n4} | **{3:n3}x** | {4:n1} | {5:n1} | {6}/{7} |" -f $tt, $ra, $rb,
               ($rb / $ra), ($ra * $FLOPS_PER_TOK), ($rb * $FLOPS_PER_TOK), $won, $tot)
}
$lines += ""
$lines += "| unit | arm | t | tok/s | GFLOP/s | wall s |"
$lines += "|---|---|---|---|---|---|"
foreach ($u in $all) {
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} |" -f $u.id, $u.arm, $u.threads, $u.rate, $u.gflops, $u.secs)
}
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
