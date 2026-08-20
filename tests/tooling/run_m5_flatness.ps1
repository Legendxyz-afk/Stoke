# =====================================================================
# IS DECODE THROUGHPUT FLAT IN N?  -- the figure gate [2]'s finding rests on.
#
# THE CLAIM. A decode step at N rows streams the weights ONCE but performs
# N x 8.46 GFLOP, and arithmetic does not amortise. So past the point where
# compute overtakes traffic (N = 1.48 at ctx 704, `m5_compute_budget.py`)
# aggregate tok/s should be FLAT in N, where the traffic budget predicts it
# keeps climbing (1.23x from N=8 to N=16).
#
# WHY THIS EXISTS SEPARATELY FROM run_m5_bars.ps1, AND IT IS A METHOD FIX.
# That runner interleaves the arms it does NOT need to compare (serve vs batch)
# and runs its CELLS sequentially -- so the load-bearing comparison, N=8 against
# N=16, spanned twelve minutes of a heating box. On 2026-08-20 that box was
# measured throttling from 143% to 126% of nominal DURING the run, and llama's
# own arms split as clock throttling predicts (compute-dense pp512 -22%,
# memory-bound tg128 -6%). A monotonic drift between two cells biases their
# RATIO, which is exactly the quantity this question is about, and a post-hoc
# correction is an extrapolation where a design would have been an measurement.
#
# TWO FIXES, both about the ratio rather than the level:
#
#   1. COUNTERBALANCED ORDER. Pass 1 runs N ascending, pass 2 descending, and
#      so on. Under any monotonic drift the forward pass biases large N down
#      and the reverse pass biases it up by the same amount, so the pair's mean
#      is unbiased to first order. Averaging repeats in a FIXED order does not
#      do this -- it just measures the same bias three times.
#
#   2. THE CLOCK IS BANKED WITH EVERY UNIT. `% Processor Performance` before
#      and after each cell, so drift is a MEASURED per-unit covariate instead
#      of a curve extrapolated from two llama points taken an hour apart.
#
# And the cells are made cheap -- a 23-token prompt instead of S2's 512 -- so a
# whole counterbalanced pair costs ~6 minutes and the drift WITHIN a pair is
# small to begin with. Prefill is not what is being measured here.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_flatness.ps1
#              [-Seqs "2,4,8,16"] [-Passes 4] [-Threads 8] [-Kv f32]
# =====================================================================
param(
    [string[]]$Seqs   = @("2", "4", "8", "16"),
    [int]$Passes      = 4,          # even, so forward and reverse are balanced
    [int]$Threads     = 8,
    [int]$Ctx         = 192,
    [int]$NGen        = 128,
    [string]$Kv       = "f32",
    [string]$Stamp    = ""
)

$ErrorActionPreference = "Stop"
$Seqs = @($Seqs | ForEach-Object { $_ -split ',' } |
          ForEach-Object { $_.Trim() } | Where-Object { $_ })

$root   = "M:\New Drive\Desktop\Programming\stoke"
$exe    = "$root\build\win\stoke.exe"
$model  = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$prompt = "$root\tests\data\parity\p00.txt"
$gr     = "$root\docs\superpowers\gate-reports"
if (-not $Stamp) { $Stamp = "{0}-m5-flatness" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp-raw.txt"
$bank   = "$gr\$Stamp-bank.jsonl"
$report = "$gr\$Stamp-report.md"
$tmp    = "$env:TEMP\stoke-flat-e.txt"
$tmpOut = "$env:TEMP\stoke-flat-o.txt"
foreach ($p in @($exe, $model, $prompt)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}
if ($Passes % 2 -ne 0) {
    throw "-Passes must be EVEN, or the counterbalancing is not balanced"
}

$script:LastCpuS = 0.0

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Bank($o) { ($o | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 $bank }
function Banked() {
    if (-not (Test-Path $bank)) { return @() }
    $out = @()
    foreach ($l in (Get-Content $bank)) {
        if ($l.Trim()) { try { $out += ($l | ConvertFrom-Json) } catch {} }
    }
    return $out
}
function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return [math]::Round(($s[$n / 2 - 1] + $s[$n / 2]) / 2.0, 3)
}
# Percent of NOMINAL clock. >100 is turbo; this box boosts to ~175% cool and
# was measured at 126% after 45 minutes of load. One sample costs ~1 s.
function ClockPct() {
    try {
        return [math]::Round((Get-Counter '\Processor Information(_Total)\% Processor Performance').CounterSamples[0].CookedValue, 1)
    } catch { return $null }
}
function RunCapture($e, $argList, $errPath, $outPath) {
    $p = Start-Process -FilePath $e -ArgumentList $argList `
                       -RedirectStandardError $errPath -RedirectStandardOutput $outPath `
                       -NoNewWindow -PassThru
    try { $null = $p.Handle } catch {}   # or TotalProcessorTime is silently 0
    $cpu = 0.0
    while (-not $p.HasExited) {
        try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
        Start-Sleep -Milliseconds 250
    }
    try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
    $script:LastCpuS = [math]::Round($cpu, 1)
}

# `stoke batch --timing` prints "decode 1016 tok in 44031.2 ms (23.071 tok/s
# aggregate)" on STDERR. That aggregate is decode-only, which is also this
# path's M0-comparable number: a lockstep batch finishes every prefill before
# any first token, so its span carries no prefill.
function BatchDecode($n) {
    $a = @("batch", ('"' + $model + '"'), "-f", ('"' + $prompt + '"'),
           "--seqs", "$n", "--ctx", "$Ctx", "-n", "$NGen",
           "--threads", "$Threads", "--kv", $Kv, "--timing")
    RunCapture $exe $a $tmp $tmpOut
    $t = (Get-Content $tmp -Raw)
    if (-not $t) { return $null }
    $m = [regex]::Match($t, '\(([\d.]+)\s+tok/s\s+aggregate\)')
    if (-not $m.Success) { return $null }
    $dec = [regex]::Match($t, 'decode\s+(\d+)\s+tok')
    return @{ rate = [double]$m.Groups[1].Value
              dec_tok = $(if ($dec.Success) { [int]$dec.Groups[1].Value } else { $null }) }
}

Say "M5 decode flatness -- is aggregate tok/s flat in N? stamp $Stamp"
Say "seqs: $($Seqs -join ',')  passes: $Passes (counterbalanced)  threads: $Threads  ctx: $Ctx  n: $NGen  kv: $Kv"
@{ total = ($Seqs.Count * $Passes); stamp = $Stamp } |
    ConvertTo-Json -Compress | Out-File -Encoding utf8 "$gr\$Stamp.status.json"

for ($pass = 1; $pass -le $Passes; $pass++) {
    # ODD passes ascend, EVEN passes descend. The pair cancels linear drift.
    $order = @($Seqs)
    $dir = "up"
    if ($pass % 2 -eq 0) { [array]::Reverse($order); $dir = "down" }
    Say "PASS $pass ($dir): $($order -join ' -> ')"

    foreach ($n in $order) {
        $id = "p$pass-n$n"
        $c0 = ClockPct
        $t0 = Get-Date
        $r = BatchDecode ([int]$n)
        $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        $c1 = ClockPct
        if ($null -eq $r) {
            Say "  $id PARSE-FAIL after ${secs}s"
            Bank @{ id = $id; kind = "fail"; pass = $pass; seqs = [int]$n; utc = (Get-Date -Format s) }
            continue
        }
        $clock = $null
        if ($null -ne $c0 -and $null -ne $c1) { $clock = [math]::Round(($c0 + $c1) / 2.0, 1) }
        Bank @{ id = $id; kind = "cell"; pass = $pass; dir = $dir; seqs = [int]$n
                rate = $r.rate; dec_tok = $r.dec_tok; secs = $secs
                cpu_s = $script:LastCpuS; clock_pct = $clock
                clock_before = $c0; clock_after = $c1; utc = (Get-Date -Format s) }
        Say ("  {0,-9} N={1,-3} {2,7:n3} tok/s   ({3}s wall, {4}s cpu, clock {5}%)" -f
             $id, $n, $r.rate, $secs, $script:LastCpuS, $clock)
    }
}

# ---------------------------------------------------------------- report
$all = @(Banked | Where-Object { $_.kind -eq "cell" })
$lines = @()
$lines += "# Is decode throughput flat in N?"
$lines += ""
$lines += "Run ``$Stamp``. ``stoke batch --timing`` decode-only aggregate, 23-token prompt,"
$lines += "ctx $Ctx, -n $NGen, threads $Threads, kv $Kv, $Passes counterbalanced passes"
$lines += "(odd ascending, even descending) so monotonic thermal drift cancels between a"
$lines += "forward and a reverse pass instead of accumulating into the ratio."
$lines += ""
$lines += "| N | median tok/s | up-pass median | down-pass median | GFLOP/s | vs N=$($Seqs[0]) |"
$lines += "|---|---|---|---|---|---|"
# 8.460 GFLOP/token at ctx 704 from m5_compute_budget.py; the matvec term
# (8.045) dominates and the attention term shrinks at this ctx, so the rate is
# computed from the matvec term alone plus this ctx's attention.
$FLOPS_PER_TOK = 8.045 + (4.0 * 36 * 32 * 128 * $Ctx / 1e9)
$base = $null
foreach ($n in $Seqs) {
    $cells = @($all | Where-Object { $_.seqs -eq [int]$n })
    if ($cells.Count -eq 0) { continue }
    $med = Median (@($cells | ForEach-Object { [double]$_.rate }))
    $up = Median (@($cells | Where-Object { $_.dir -eq "up" } | ForEach-Object { [double]$_.rate }))
    $dn = Median (@($cells | Where-Object { $_.dir -eq "down" } | ForEach-Object { [double]$_.rate }))
    if ($null -eq $base) { $base = $med }
    $lines += ("| {0} | {1:n3} | {2} | {3} | {4:n0} | {5:n3}x |" -f $n, $med,
               $(if ($null -ne $up) { "{0:n3}" -f $up } else { "-" }),
               $(if ($null -ne $dn) { "{0:n3}" -f $dn } else { "-" }),
               ($med * $FLOPS_PER_TOK), ($med / $base))
}
$lines += ""
$lines += "**Read the up and down columns against each other.** If they disagree by more"
$lines += "than the pass-to-pass spread, drift was still large inside a pass and the"
$lines += "counterbalancing is carrying more weight than it should -- shorten the cells"
$lines += "or let the box cool between passes rather than trusting the mean."
$lines += ""
$lines += "| unit | N | tok/s | wall s | cpu s | cores | clock % of nominal |"
$lines += "|---|---|---|---|---|---|---|"
foreach ($u in $all) {
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5:n2} | {6} |" -f $u.id, $u.seqs, $u.rate,
               $u.secs, $u.cpu_s, $(if ($u.secs -gt 0) { $u.cpu_s / $u.secs } else { 0 }),
               $u.clock_pct)
}
$lines += ""
$lines += "Clock is ``\Processor Information(_Total)\% Processor Performance``, sampled"
$lines += "before and after each cell and averaged. This box boosts to ~175% of nominal"
$lines += "cool and was measured at 126% after 45 minutes of sustained load, so a unit's"
$lines += "clock is a covariate, not a constant."
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
