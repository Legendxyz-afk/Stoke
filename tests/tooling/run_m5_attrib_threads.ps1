# =====================================================================
# ATTRIBUTING THE 2.83x -- step 1: is it the KERNEL or the POOL?
#
# stoke prefills at 273 GFLOP/s against llama.cpp's 772, same sitting, same
# file, t=8. That end-to-end number contains the Q4_K dot kernel's instruction
# efficiency, the thread pool's parallelisation loss, and everything in the
# forward that is not matvec_batch. At t=1 there is no pool, no barrier and no
# dispatch -- so the SINGLE-THREAD rate isolates the first, and the t=1-to-t=8
# ratio prices the second. Both engines take a thread flag, so both go through
# the identical split with the shipping binaries and no instrumentation.
#
# Prediction and falsifiers: memos/2026-08-20-m5-attribution-PREREGISTERED.md,
# committed before this ran.
#
# WHY PREFILL AND NOT DECODE. Prefill runs matvec_batch at 128-row chunks --
# a saturated kernel tile -- so it measures the kernel near its asymptote
# rather than at an occupancy that varies with the thing being swept. It is
# also the number the 2.83x was computed from, so the two are comparable.
#
# WHY `prefill N tok in M ms` AND NOT WALL. `--timing` reports load separately.
# ISLC is running and purges the standby list, so every invocation re-reads the
# model; parsing the prefill term puts that outside the measurement. A wall-clock
# rate would charge a ~2 s load to a 16 s t=8 run and a 130 s t=1 run unequally,
# which biases the very ratio this script exists to measure.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_attrib_threads.ps1
#              [-Threads "1,2,4,8"] [-Cycles 3]
# =====================================================================
param(
    [string[]]$Threads = @("1", "2", "4", "8"),
    [int]$Cycles       = 3,
    [int]$Ctx          = 704,
    [string]$Kv        = "f32",
    [string]$Stamp     = ""
)

$ErrorActionPreference = "Stop"
$Threads = @($Threads | ForEach-Object { $_ -split ',' } |
             ForEach-Object { $_.Trim() } | Where-Object { $_ })

$root   = "M:\New Drive\Desktop\Programming\stoke"
$exe    = "$root\build\win\stoke.exe"
$LB     = "$root\bench\vendor\llama-b10037\llama-bench.exe"
$model  = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$prompt = "$root\tests\data\bench_prompts\s2-512-c00.txt"
$gr     = "$root\docs\superpowers\gate-reports"
if (-not $Stamp) { $Stamp = "{0}-m5-attrib-threads" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp-raw.txt"
$bank   = "$gr\$Stamp-bank.jsonl"
$report = "$gr\$Stamp-report.md"
$tmp    = "$env:TEMP\stoke-attrib-e.txt"
$tmpOut = "$env:TEMP\stoke-attrib-o.txt"
foreach ($p in @($exe, $LB, $model, $prompt)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}

# 8.460 GFLOP/token at ctx 704, from m5_compute_budget.py reading the GGUF's
# own tensor table. Carried here as ONE constant with its provenance rather
# than recomputed by hand per row.
$FLOPS_PER_TOK = 8.460
$script:LastCpuS = 0.0
$script:MidClock = $null

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
    $script:MidClock = $null
    $t0 = Get-Date
    while (-not $p.HasExited) {
        try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
        if ($null -eq $script:MidClock -and ((Get-Date) - $t0).TotalSeconds -ge 3.0) {
            $script:MidClock = ClockPct     # under load, not between cells
        }
        Start-Sleep -Milliseconds 250
    }
    try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
    $script:LastCpuS = [math]::Round($cpu, 1)
}

# PREFILL rate from `stoke batch --timing`:
#   "timing: load 1234.5 ms | 1 seqs x ctx 704 (44 blocks, kv f32) |
#    prefill 525 tok in 16300.0 ms | decode 8 tok in ... "
# -n 8 keeps the decode tail short; only the prefill term is parsed.
function StokePrefill($t) {
    $a = @("batch", ('"' + $model + '"'), "-f", ('"' + $prompt + '"'),
           "--seqs", "1", "--ctx", "$Ctx", "-n", "8",
           "--threads", "$t", "--kv", $Kv, "--timing")
    RunCapture $exe $a $tmp $tmpOut
    $txt = (Get-Content $tmp -Raw)
    if (-not $txt) { return $null }
    $m = [regex]::Match($txt, 'prefill\s+(\d+)\s+tok in\s+([\d.]+)\s+ms')
    if (-not $m.Success) { return $null }
    $tok = [double]$m.Groups[1].Value
    $ms  = [double]$m.Groups[2].Value
    if ($ms -le 0) { return $null }
    return @{ tok = $tok; ms = $ms; rate = [math]::Round($tok / ($ms / 1000.0), 3) }
}

# llama-bench prints a markdown row per config; `-n 0` suppresses tg so only
# the pp row exists and the regex cannot pick up the wrong one.
function LlamaPrefill($t) {
    $a = @("-m", ('"' + $model + '"'), "-p", "512", "-n", "0", "-r", "2", "-t", "$t")
    RunCapture $LB $a $tmp $tmpOut
    $txt = (Get-Content $tmpOut -Raw)
    if (-not $txt) { return $null }
    $m = [regex]::Match($txt, 'pp\d+\s*\|\s*([\d.]+)')
    if (-not $m.Success) { return $null }
    return @{ rate = [double]$m.Groups[1].Value }
}

Say "M5 attribution -- kernel or pool? stamp $Stamp"
Say "threads: $($Threads -join ',')  cycles: $Cycles  ctx: $Ctx  kv: $Kv"
@{ total = ($Threads.Count * $Cycles * 2); stamp = $Stamp } |
    ConvertTo-Json -Compress | Out-File -Encoding utf8 "$gr\$Stamp.status.json"
if (@(Get-Process | Where-Object { $_.Name -match 'ISLC' }).Count -gt 0) {
    Say "  note: ISLC is running -- prefill_ms excludes load, so this is tolerated"
}

for ($c = 1; $c -le $Cycles; $c++) {
    # Thread order REVERSED on even cycles: the box warms over a cycle, and a
    # fixed order would put t=1 always-cool and t=8 always-hot, biasing the very
    # scaling ratio being measured. Same reasoning as run_m5_flatness.ps1.
    $order = @($Threads)
    $dir = "up"
    if ($c % 2 -eq 0) { [array]::Reverse($order); $dir = "down" }
    Say "CYCLE $c ($dir): $($order -join ' -> ')"

    foreach ($t in $order) {
        foreach ($arm in @("S", "L")) {
            $id = "c$c-t$t-$arm"
            $t0 = Get-Date
            if ($arm -eq "S") { $r = StokePrefill ([int]$t) } else { $r = LlamaPrefill ([int]$t) }
            $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
            if ($null -eq $r) {
                Say "  $id PARSE-FAIL after ${secs}s"
                Bank @{ id = $id; kind = "fail"; arm = $arm; threads = [int]$t
                        cycle = $c; secs = $secs; utc = (Get-Date -Format s) }
                continue
            }
            $g = [math]::Round($r.rate * $FLOPS_PER_TOK, 1)
            Bank @{ id = $id; kind = "cell"; arm = $arm; threads = [int]$t; cycle = $c
                    dir = $dir; rate = $r.rate; gflops = $g; secs = $secs
                    cpu_s = $script:LastCpuS; clock_pct = $script:MidClock
                    utc = (Get-Date -Format s) }
            $cores = 0.0
            if ($secs -gt 0) { $cores = [math]::Round($script:LastCpuS / $secs, 2) }
            Say ("  {0,-10} {1,7:n3} tok/s = {2,6:n1} GFLOP/s  ({3}s, {4} cores, clock {5}%)" -f
                 $id, $r.rate, $g, $secs, $cores, $script:MidClock)
        }
    }
}

# ---------------------------------------------------------------- report
$all = @(Banked | Where-Object { $_.kind -eq "cell" })
$lines = @()
$lines += "# Attributing the 2.83x: is it the kernel or the pool?"
$lines += ""
$lines += "Run ``$Stamp``. PREFILL of a 525-token prompt, ctx $Ctx, kv $Kv, $Cycles cycles with"
$lines += "the thread order reversed on even cycles. stoke's number is ``--timing``'s ``prefill``"
$lines += "term (load excluded); llama's is ``llama-bench -p 512 -n 0`` (warmed). GFLOP/s uses"
$lines += "$FLOPS_PER_TOK GFLOP/token from ``m5_compute_budget.py``."
$lines += ""
$lines += "| threads | stoke tok/s | stoke GFLOP/s | llama tok/s | llama GFLOP/s | gap | stoke scaling | llama scaling |"
$lines += "|---|---|---|---|---|---|---|---|"
$sBase = $null; $lBase = $null
foreach ($t in ($Threads | Sort-Object { [int]$_ })) {
    $S = @($all | Where-Object { $_.arm -eq "S" -and $_.threads -eq [int]$t } | ForEach-Object { [double]$_.rate })
    $L = @($all | Where-Object { $_.arm -eq "L" -and $_.threads -eq [int]$t } | ForEach-Object { [double]$_.rate })
    $sm = Median $S; $lm = Median $L
    if ($null -eq $sBase -and $null -ne $sm) { $sBase = $sm }
    if ($null -eq $lBase -and $null -ne $lm) { $lBase = $lm }
    $gap = "-"
    if ($null -ne $sm -and $null -ne $lm -and $sm -gt 0) { $gap = "{0:n2}x" -f ($lm / $sm) }
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f $t,
        $(if ($null -ne $sm) { "{0:n3}" -f $sm } else { "-" }),
        $(if ($null -ne $sm) { "{0:n1}" -f ($sm * $FLOPS_PER_TOK) } else { "-" }),
        $(if ($null -ne $lm) { "{0:n3}" -f $lm } else { "-" }),
        $(if ($null -ne $lm) { "{0:n1}" -f ($lm * $FLOPS_PER_TOK) } else { "-" }),
        $gap,
        $(if ($null -ne $sm -and $sBase) { "{0:n2}x" -f ($sm / $sBase) } else { "-" }),
        $(if ($null -ne $lm -and $lBase) { "{0:n2}x" -f ($lm / $lBase) } else { "-" }))
}
$lines += ""
$lines += "**The discriminator, pre-registered:** a single-thread gap near 2.8x means the gap is"
$lines += "INSTRUCTION EFFICIENCY and the kernel is the target; a single-thread gap under 1.8x"
$lines += "means the pool is, and the bit-exactness question is largely moot."
$lines += ""
$lines += "**The SCALING columns are the instrument for parallel efficiency, not ``cores``.**"
$lines += "``cpu s`` is whole-process -- it includes the model load and the short decode tail --"
$lines += "while the rate is prefill-only, so ``cores`` is a health tell (is anything running at"
$lines += "all?) and NOT prefill occupancy. Reading it as occupancy would price the pool from a"
$lines += "number that contains two phases the rate does not."
$lines += ""
$lines += "| unit | arm | t | tok/s | GFLOP/s | wall s | cpu s | cores (whole proc) | clock % |"
$lines += "|---|---|---|---|---|---|---|---|---|"
foreach ($u in $all) {
    $cores = 0.0
    if ($u.secs -gt 0) { $cores = [math]::Round($u.cpu_s / $u.secs, 2) }
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |" -f
               $u.id, $u.arm, $u.threads, $u.rate, $u.gflops, $u.secs, $u.cpu_s, $cores, $u.clock_pct)
}
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
