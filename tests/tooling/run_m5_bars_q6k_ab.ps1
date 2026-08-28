# =====================================================================
# WHAT THE Q6_K KERNEL BOUGHT ON THE GATE [2] CELL -- SAME SITTING, ONE BINARY.
#
# The 2026-08-28 bars re-measure came in at 8.960 AGGREGATE(M0) on 8f32-S
# against 4.938 banked on 2026-08-20, and that 1.81x is NOT attributable as it
# stands. Two reasons, and the second is the one that would have gone unsaid:
#
#   1. Eight days of M5 work separate the two binaries, not one kernel.
#   2. THE BOX WAS AT A DIFFERENT CLOCK. llama-bench is a FIXED third-party
#      binary and its own numbers moved between the sittings -- pp512
#      72.24 -> 92.29 (+27.8%) while tg128 moved 17.48 -> 18.77 (+7.4%).
#      That split is the signature [[measurement-needs-interleaving]] records
#      for this box: compute-dense work loses ~22% to thermal drift and
#      memory-bound work ~6%. The 08-20 sitting was the degraded one.
#
# So this measures the kernel the only way that answers the question: BOTH ARMS
# FROM ONE BINARY IN ONE SITTING, selected by STOKE_Q6K_KERNEL, on the exact
# gate [2] cell (8 clients, S2's own prompts, ctx 704, n 128, kv f32,
# --no-dedup -- the same load-bearing flag the bars runner carries).
#
# ORDER IS PALINDROMIC -- ref, mad, mad, ref -- so each arm appears once early
# and once late and drift lands on a different axis from the arm. A straight
# A-then-B on a box that drifts 22% cannot tell them apart.
#
# THE FLOOR IS MEASURED, NOT CHOSEN, AND IT IS EACH ARM'S OWN SPREAD.
# [[derived-quantity-is-a-claim]]: a null arm bounds THE ARM IT IS BUILT FROM
# and nothing else, so the floor here is max(spread(ref), spread(mad)) and the
# ratio must clear 2x it to be called resolved.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_bars_q6k_ab.ps1
#        [-Seqs 8] [-Reps 2] [-Stamp <name>]
# =====================================================================
param(
    [int]$Seqs     = 8,
    [int]$Threads  = 8,
    [int]$Ctx      = 704,
    [int]$NGen     = 128,
    # HOW MANY UNITS THIS INVOCATION MAY RUN. The sitting is driven foreground
    # (2026-08-26: long unattended jobs on this box get killed with no message)
    # and a `ref` unit is ~2x a `maddubs` one, so all four back to back exceed a
    # foreground caller's ceiling. Banking per unit makes the chain equivalent to
    # one run, and the palindromic order survives because the ORDER of the
    # invocations is the order of the units.
    [int]$Units    = 99,
    [string]$Stamp = ""
)

$ErrorActionPreference = "Stop"

$root  = "M:\New Drive\Desktop\Programming\stoke"
$exe   = "$root\build\win\stoke.exe"
$model = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$pdir  = "$root\tests\data\bench_prompts"
$gr    = "$root\docs\superpowers\gate-reports"

foreach ($p in @($exe, $model, $pdir)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}

if (-not $Stamp) { $Stamp = "{0}-m5-bars-q6k-ab" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp-raw.txt"
$bank   = "$gr\$Stamp-bank.jsonl"
$report = "$gr\$Stamp-report.md"
$tmpO   = "$env:TEMP\stoke-q6kab-o.txt"
$tmpE   = "$env:TEMP\stoke-q6kab-e.txt"
$exeSha = (Get-FileHash $exe -Algorithm SHA256).Hash

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Host $line
}
function Bank($obj) { ($obj | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 $bank }
function Banked() {
    if (-not (Test-Path $bank)) { return @() }
    $out = @()
    foreach ($l in (Get-Content $bank)) {
        if ($l.Trim()) { try { $out += ($l | ConvertFrom-Json) } catch {} }
    }
    return $out
}
function Grp($text, $re, $i) {
    $m = [regex]::Match($text, $re)
    if ($m.Success) { return [double]$m.Groups[$i].Value }
    return $null
}
function CpuBusy() {
    $c = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor |
         Where-Object { $_.Name -eq "_Total" }
    return [int]$c.PercentProcessorTime
}
function ServeArgs() {
    $a = @("bench", "--serve", ('"' + $model + '"'))
    for ($i = 0; $i -lt $Seqs; $i++) {
        $a += "-f"; $a += ('"' + ("$pdir\s2-512-c{0:d2}.txt" -f $i) + '"')
    }
    return $a + @("--seqs", "$Seqs", "--ctx", "$Ctx", "-n", "$NGen", "--no-dedup",
                  "--threads", "$Threads", "--kv", "f32")
}

# THE BINARY MUST BE PROVEN TO READ THE VARIABLE, and nothing downstream can
# prove it: the two kernels are BIT-IDENTICAL, so output cannot tell them apart,
# and using the timing difference as evidence that the switch worked is circular
# -- the timing difference is the thing being measured. `vec.c` refuses an
# unrecognised value on purpose, so an unrecognised value is the non-circular
# probe: if the run does NOT die, the variable is not being read and every
# number below would be one kernel measured twice.
Say "M5 gate [2] cell -- Q6_K kernel A/B. stamp $Stamp"
Say "exe sha256 $($exeSha.Substring(0,12))  commit $(git -C $root rev-parse --short HEAD)"
Say "cell: seqs $Seqs, ctx $Ctx, n $NGen, kv f32, --no-dedup, threads $Threads"
if (@(Banked | Where-Object { $_.kind -eq "probe" -and $_.exe_sha -eq $exeSha }).Count -gt 0) {
    Say "SWITCH PROBE already banked for this binary, skipping"
} else {
    $env:STOKE_Q6K_KERNEL = "not-a-kernel"
    $p = Start-Process -FilePath $exe -ArgumentList (ServeArgs) -RedirectStandardOutput $tmpO `
                       -RedirectStandardError $tmpE -NoNewWindow -PassThru -Wait
    $env:STOKE_Q6K_KERNEL = $null
    $refusal = (Get-Content $tmpE -Raw)
    if ($p.ExitCode -eq 0) {
        throw ("STOP: STOKE_Q6K_KERNEL=not-a-kernel exited 0 -- the binary is NOT reading " +
               "the variable, so both arms would be the same kernel measured twice.")
    }
    if ($refusal -notmatch 'STOKE_Q6K_KERNEL') {
        throw "STOP: the run failed but not with the kernel refusal -- got: $refusal"
    }
    Say "SWITCH PROVEN LIVE: an unrecognised value is refused (exit $($p.ExitCode))"
    # KEYED BY THE BINARY, like everything else banked here: a probe that proved
    # a DIFFERENT exe reads the variable proves nothing about this one.
    Bank @{ id = "switch-probe"; kind = "probe"; exit = $p.ExitCode; refused = $true;
            exe_sha = $exeSha; utc = (Get-Date -Format s) }
}

# QUIESCE, two consecutive quiet reads, same gate as the bars runner.
$quiet = 0; $seen = @()
for ($try = 0; $try -lt 6; $try++) {
    $busy = CpuBusy; $seen += $busy
    if ($busy -le 12) { $quiet++; if ($quiet -ge 2) { break } }
    else { $quiet = 0; Say "  waiting: cpu $busy% > 12%" }
    Start-Sleep -Seconds 20
}
$quiesced = ($quiet -ge 2)
$last2 = ($seen | Select-Object -Last 2) -join "%, "
if ($quiesced) { Say "QUIESCE GATE PASSED on cpu $last2%" }
else { Say "!! QUIESCE GATE NOT MET -- last reads $last2%. Absolutes are indicative; the RATIO is what this run is for." }

$order = @("ref", "maddubs", "maddubs", "ref")
$i = 0
$ran = 0
foreach ($arm in $order) {
    $i++
    $id = "u$i-$arm"
    if (@(Banked | Where-Object { $_.id -eq $id -and $null -ne $_.gate }).Count -gt 0) {
        Say "  $id already banked, skipping"; continue
    }
    if ($ran -ge $Units) { Say "  -Units $Units reached -- $id and later left for the next invocation"; break }
    $ran++
    $env:STOKE_Q6K_KERNEL = $arm
    $t0 = Get-Date
    $p = Start-Process -FilePath $exe -ArgumentList (ServeArgs) -RedirectStandardOutput $tmpO `
                       -RedirectStandardError $tmpE -NoNewWindow -PassThru
    try { $null = $p.Handle } catch {}
    $cpu = 0.0
    while (-not $p.HasExited) {
        try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
        Start-Sleep -Milliseconds 250
    }
    try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
    $env:STOKE_Q6K_KERNEL = $null
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $t = (Get-Content $tmpO -Raw)
    if ($t -match '! only \d+ of \d+ expected ids') { throw "STOP: $id produced fewer ids than asked" }
    $gate = Grp $t 'AGGREGATE\(M0\)\s+([\d.]+)' 1
    if ($null -eq $gate) { throw "STOP: $id did not print AGGREGATE(M0)" }
    Bank @{ id = $id; kind = "unit"; arm = $arm; gate = $gate; secs = $secs;
            cpu_s = [math]::Round($cpu, 1); itl50 = (Grp $t 'ITL\s+p50\s+([\d.]+)' 1);
            ttft50 = (Grp $t 'TTFT p50\s+([\d.]+)' 1); quiesced = $quiesced;
            exe_sha = $exeSha; cpu_busy = (CpuBusy); utc = (Get-Date -Format s) }
    Say ("  {0,-12} {1,8:n3} tok/s   ({2}s wall, {3}s cpu)" -f $id, $gate, $secs, [math]::Round($cpu, 1))
}

# ------------------------------------------------------------------ report
$all = @(Banked | Where-Object { $_.kind -eq "unit" })
function ArmVals($a) { return @($all | Where-Object { $_.arm -eq $a } | ForEach-Object { [double]$_.gate }) }
function Spread($xs) {
    if ($xs.Count -lt 2) { return $null }
    $lo = ($xs | Measure-Object -Minimum).Minimum
    $hi = ($xs | Measure-Object -Maximum).Maximum
    if ($lo -le 0) { return $null }
    return (($hi - $lo) / $lo)
}
function Mean($xs) { if ($xs.Count -eq 0) { return $null } return (($xs | Measure-Object -Average).Average) }

$rv = ArmVals "ref"; $mv = ArmVals "maddubs"
$rm = Mean $rv; $mm = Mean $mv
$sr = Spread $rv; $sm = Spread $mv
$floor = $null
if ($null -ne $sr -and $null -ne $sm) { $floor = [math]::Max($sr, $sm) }
$ratio = $null
if ($null -ne $rm -and $rm -gt 0 -and $null -ne $mm) { $ratio = $mm / $rm }

$lines = @()
$lines += "# What the Q6_K maddubs kernel bought on the gate [2] cell"
$lines += ""
$lines += "Run ``$Stamp``. ONE binary, ``stoke.exe`` sha256 ``$($exeSha.Substring(0,12))``, both arms"
$lines += "selected at runtime by ``STOKE_Q6K_KERNEL``. Cell is gate [2]'s own: $Seqs clients on S2's"
$lines += "prompts, ctx $Ctx, n $NGen, kv f32, ``--no-dedup``, threads $Threads. Order ref/mad/mad/ref."
$lines += ""
$lines += "**The switch was proven live before any arm ran**: ``STOKE_Q6K_KERNEL=not-a-kernel``"
$lines += "is refused by ``vec.c`` and the probe run exited nonzero. The two kernels are"
$lines += "bit-identical, so output cannot distinguish them, and the timing difference is the"
$lines += "thing being measured -- it cannot also be the evidence that the switch worked."
$lines += ""
$lines += "## Units"
$lines += ""
$lines += "| id | kernel | AGGREGATE(M0) | wall s | cpu s | quiesced |"
$lines += "|---|---|---|---|---|---|"
foreach ($u in $all) {
    $lines += ("| {0} | {1} | {2:n3} | {3} | {4} | {5} |" -f $u.id, $u.arm, [double]$u.gate, $u.secs, $u.cpu_s, $u.quiesced)
}
$lines += ""
$lines += "## The ratio, against a floor made of the arms' own spreads"
$lines += ""
$lines += ("- ``ref`` mean **{0:n3}** tok/s over {1} runs, spread {2}" -f $rm, $rv.Count,
           $(if ($null -ne $sr) { "{0:n2}%" -f ($sr * 100) } else { "-" }))
$lines += ("- ``maddubs`` mean **{0:n3}** tok/s over {1} runs, spread {2}" -f $mm, $mv.Count,
           $(if ($null -ne $sm) { "{0:n2}%" -f ($sm * 100) } else { "-" }))
$lines += ("- floor = max(spread) = {0}" -f
           $(if ($null -ne $floor) { "{0:n2}%" -f ($floor * 100) } else { "-" }))
$lines += ""
if ($null -ne $ratio -and $null -ne $floor) {
    $gain = [math]::Abs($ratio - 1.0)
    $resolved = ($gain -ge (2.0 * $floor))
    $verdict = "UNRESOLVED"
    if ($resolved) { $verdict = "RESOLVED" }
    $lines += ("**maddubs / ref = {0:n4}x** -- gain {1:n2}% against a floor of {2:n2}%, so **{3}**" -f
               $ratio, ($gain * 100), ($floor * 100), $verdict)
    $lines += ""
    $lines += "A cell resolves only when the gain clears TWICE its floor. That rule exists because"
    $lines += "``gain > floor`` passed a cell on 2026-08-27 at 33.91% over 32.76% -- which is exactly"
    $lines += "what a gain that IS the noise looks like."
} else {
    $lines += "**Not enough units banked to state a ratio.**"
}
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
