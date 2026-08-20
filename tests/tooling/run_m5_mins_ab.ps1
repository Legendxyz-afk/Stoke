# =====================================================================
# A/B: does vectorising the Q4_K mins term buy anything?
#
# A = build/ab/mins-scalar.exe   (STOKE_MINS_SIMD=0, the shipping default)
# B = build/ab/mins-simd.exe     (STOKE_MINS_SIMD=1)
#
# WHAT IS ALREADY ESTABLISHED, so this run only has to answer "how much":
#   - the two are BIT-EXACT on the real model through the batch path
#     (p00x8 f32, p00x8 q8, p41x2 f32 all SAME), and that comparison has
#     POWER: one wrong lane in the duplication mask makes it DIFFER.
#   - the suite (14 binaries) and all four goldens pass against B.
#   - objdump puts the scalar spelling at ~40 uops beside the vector dot's
#     ~57, i.e. ~40-45% of the emitted per-sequence work.
#
# WHY THE uop RATIO IS NOT THE PREDICTION. stoke's kernel runs at roughly 2.5
# uops/cycle against a 6-wide front end, so it is NOT front-end bound; and
# scalar `imul` and vector integer multiply issue to different ports, so the
# two may already overlap. Removing 40% of the uops can therefore buy anywhere
# from ~1.0x to ~1.5x. That is precisely why this is measured and not argued.
#
# t=1 IS THE HEADLINE, NOT t=8. At one thread there is no pool, no barrier and
# no dispatch, so the ratio is the kernel's own. t=8 is carried too because it
# is the operating point, but a kernel change should be read where the kernel
# is the only thing moving.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_mins_ab.ps1
# =====================================================================
param(
    [int]$CyclesT8 = 4,
    [int]$CyclesT1 = 2,
    [int]$Ctx      = 704,
    [string]$Kv    = "f32",
    [string]$Stamp = ""
)

$ErrorActionPreference = "Stop"
$root   = "M:\New Drive\Desktop\Programming\stoke"
$armA   = "$root\build\ab\mins-scalar.exe"
$armB   = "$root\build\ab\mins-simd.exe"
$mdl    = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$prompt = "$root\tests\data\bench_prompts\s2-512-c00.txt"
$gr     = "$root\docs\superpowers\gate-reports"
if (-not $Stamp) { $Stamp = "{0}-m5-mins-ab" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp-raw.txt"
$bank   = "$gr\$Stamp-bank.jsonl"
$report = "$gr\$Stamp-report.md"
$tmpE   = "$env:TEMP\stoke-minsab-e.txt"
$tmpO   = "$env:TEMP\stoke-minsab-o.txt"
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

# Start-Process for RAW stderr: PowerShell re-wraps a native command's stderr
# at console width, and arm A's and arm B's file names differ in length --
# which is exactly how the 2026-08-14 overnight lost a whole arm to a regex.
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

Say "mins-term A/B. A=$armA  B=$armB"
Say "ctx $Ctx  kv $Kv  t8 cycles $CyclesT8  t1 cycles $CyclesT1"
@{ total = (($CyclesT8 + $CyclesT1) * 2); stamp = $Stamp } |
    ConvertTo-Json -Compress | Out-File -Encoding utf8 "$gr\$Stamp.status.json"

foreach ($cfg in @(@{t = 8; n = $CyclesT8}, @{t = 1; n = $CyclesT1})) {
    for ($cyc = 1; $cyc -le $cfg.n; $cyc++) {
        # ARM ORDER ALTERNATES. The box warms across a cycle, so a fixed A-then-B
        # order would charge B the warmer half every time -- and B is the arm
        # under test, which is the worst possible place to put a systematic bias.
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
$lines += "# Does vectorising the Q4_K mins term buy anything?"
$lines += ""
$lines += "Run ``$Stamp``. A = ``STOKE_MINS_SIMD=0`` (shipping default), B = ``=1``."
$lines += "PREFILL of a 525-token prompt, ctx $Ctx, kv $Kv, arms interleaved with the"
$lines += "order alternating per cycle. The two are already BIT-EXACT on the real model"
$lines += "through the batch path, and that comparison is mutation-verified."
$lines += ""
$lines += "| threads | A tok/s | B tok/s | B/A | A GFLOP/s | B GFLOP/s |"
$lines += "|---|---|---|---|---|---|"
foreach ($tt in @(1, 8)) {
    $ra = Median (@($all | Where-Object { $_.arm -eq "A" -and $_.threads -eq $tt } | ForEach-Object { [double]$_.rate }))
    $rb = Median (@($all | Where-Object { $_.arm -eq "B" -and $_.threads -eq $tt } | ForEach-Object { [double]$_.rate }))
    if ($null -eq $ra -or $null -eq $rb) { continue }
    $lines += ("| {0} | {1:n4} | {2:n4} | **{3:n3}x** | {4:n1} | {5:n1} |" -f $tt, $ra, $rb,
               ($rb / $ra), ($ra * $FLOPS_PER_TOK), ($rb * $FLOPS_PER_TOK))
}
$lines += ""
$lines += "**t=1 is the headline** -- no pool, no barrier, no dispatch, so the ratio is the"
$lines += "kernel's own. objdump put the scalar spelling at ~40 uops against the vector"
$lines += "dot's ~57; the kernel runs at ~2.5 uops/cycle against a 6-wide front end, so the"
$lines += "uop ratio was an upper bound and never a prediction."
$lines += ""
$lines += "| unit | arm | t | tok/s | GFLOP/s | wall s |"
$lines += "|---|---|---|---|---|---|"
foreach ($u in $all) {
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} |" -f $u.id, $u.arm, $u.threads, $u.rate, $u.gflops, $u.secs)
}
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
