# =====================================================================
# M5 GATE ITEM [2] -- THE BARS, MEASURED.
#
#   [2] real: S2 aggregate >=17.7 @8 AND >=14.2 @16 AND >=10.8 @8 /
#       >=16.8 @16 (both families of bars) on Q4_K_M, M0 methodology
#       (medians, pinned, quiesced, same-file).
#
# S2 is M0's locked scenario: 512 prompt tokens, 128 generated, N
# closed-loop clients, no shared prefix (`bench/scenarios.py`). The
# prompts here are M0's own, exported verbatim by
# `tests/tooling/export_bench_prompts.py` -- 525 real tokens each once
# the `[client N] ` prefix and the ChatML template are counted.
#
# WHAT EACH ARM IS FOR, because three arms is two more than a bar needs:
#
#   S  `stoke bench --serve`  -- M5's continuous scheduler. THE gate number
#      is its AGGREGATE(M0): decode-only ids over the span from the first
#      token anywhere to the last token anywhere, which is exactly
#      `bench/bench.py`'s aggregate_decode_tps and therefore exactly what
#      the 17.7 / 14.2 / 10.8 / 16.8 cells were computed with.
#
#   B  `stoke batch --seqs N --timing` -- M4's STATIC path, same prompts,
#      same ctx, same pool. It is the attribution instrument the playbook's
#      ladder names first: "if static N=8 already missed the bar, the
#      scheduler was never the problem". It also splits prefill from decode,
#      which the serve driver can only do through TTFT.
#
#      >> AND A WARNING ABOUT COMPARING B TO S DIRECTLY. A lockstep batch
#      finishes every prefill BEFORE any first token, so its M0 span
#      contains no prefill at all, while a continuous scheduler is charged
#      with every later arrival's prefill inside its span. That is not a
#      defect in the metric -- it is what M0 measured llama-server with --
#      but B's aggregate and S's AGGREGATE(M0) are NOT an A/B of the
#      scheduler. Compare S's decode-only rate (N / ITL p50) against B's
#      decode aggregate; compare S's AGGREGATE(M0) against the BARS.
#
#   L  `llama-bench -p 512 -n 128` -- the non-stoke referent, and it is here
#      for two independent jobs. (1) Box health: a stoke-only collapse
#      (EcoQoS, 2026-08-17) hits every stoke arm at once, so an S-vs-B
#      comparison still "agrees with itself" while both are wrong; only a
#      non-stoke arm exposes it. (2) `pp512` is the PREFILL referent -- M0's
#      llamacpp-tuned reached its first token in 7.4 s on a 512-token prompt
#      (~69 tok/s), and on S2's shape prefill is 512 of the 640 tokens of
#      work per request.
#
# PROTOCOL, per [[measurement-needs-interleaving]] and the 08-19 P-BAR:
# threads 8 for every engine (each at its own fastest count, 08-17 curve),
# affinity UNPINNED over 16 logical (0xFFF is 12 hyperthread siblings, not
# 6 P-cores, and it costs ~1.7x once threads stack), arms interleaved
# INSIDE each block, medians across blocks, every unit banked as it lands.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_bars.ps1
#              [-Cells 8f32,8q8,16f32,16q8] [-Blocks 3] [-Threads 8]
#              [-QuietTries 10] [-Stamp <name>]
# =====================================================================
param(
    [string[]]$Cells      = @("8f32", "16f32", "8q8", "16q8"),
    [int]$Blocks          = 3,
    [int]$Threads         = 8,
    [int]$Ctx             = 704,
    [int]$NGen            = 128,
    [int]$QuietTries      = 10,
    [string]$Stamp        = ""
)

$ErrorActionPreference = "Stop"

# >> SPLIT ON COMMA OURSELVES. `powershell -File script.ps1 -Cells a,b,c` hands
# the whole thing over as ONE string -- only in-process invocation binds a
# comma list to [string[]]. Found by this script's own bad-cell guard four
# minutes into a two-hour run, which is the only reason the guard is a `throw`
# and not a `continue`.
$Cells = @($Cells | ForEach-Object { $_ -split ',' } |
           ForEach-Object { $_.Trim() } | Where-Object { $_ })

$root  = "M:\New Drive\Desktop\Programming\stoke"
$exe   = "$root\build\win\stoke.exe"
$LB    = "$root\bench\vendor\llama-b10037\llama-bench.exe"
$model = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$pdir  = "$root\tests\data\bench_prompts"
$gr    = "$root\docs\superpowers\gate-reports"

if (-not $Stamp) { $Stamp = "{0}-m5-bars" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp-raw.txt"
$bank   = "$gr\$Stamp-bank.jsonl"
$report = "$gr\$Stamp-report.md"
$status = "$gr\$Stamp.status.json"
$tmp    = "$env:TEMP\stoke-m5bars-e.txt"
$tmpOut = "$env:TEMP\stoke-m5bars-o.txt"

foreach ($p in @($exe, $LB, $model, $pdir)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}

$QUIET_CPU = 12
$script:LastCpuS = 0.0

# ---------------------------------------------------------------- helpers
# Lifted from run_overnight_pbar.ps1 with its scars intact: Start-Process
# for raw stderr (PowerShell wraps a native command's stderr at console
# width), all three standby buckets (Normal alone cried wolf on 3/3 good
# blocks), and a Median that does not round half-to-even on an even count.
function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Bank($obj) {
    ($obj | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 $bank
}
function Banked() {
    if (-not (Test-Path $bank)) { return @() }
    $out = @()
    foreach ($l in (Get-Content $bank)) {
        if ($l.Trim()) { try { $out += ($l | ConvertFrom-Json) } catch {} }
    }
    return $out
}
function HasUnit($id) {
    foreach ($b in (Banked)) {
        if ($b.id -eq $id) {
            if ($b.kind -eq "cell") { if ($null -ne $b.gate) { return $true } }
            else { return $true }
        }
    }
    return $false
}
function RunCapture($e, $argList, $errPath, $outPath) {
    $p = Start-Process -FilePath $e -ArgumentList $argList `
                       -RedirectStandardError $errPath `
                       -RedirectStandardOutput $outPath `
                       -NoNewWindow -PassThru
    # >> TOUCH .Handle ONCE, OR CPU-SECONDS ARE SILENTLY ZERO. Found in this
    # file's own pre-flight: every unit banked cpu_s = 0. A Process object from
    # -PassThru has no cached handle, so TotalProcessorTime throws, the
    # try/catch swallows it, and the EcoQoS tell -- the one check that is
    # INDEPENDENT of whether a rate "looks low" -- reads zero for every arm at
    # once, which is exactly how it would fail to fire when it mattered.
    try { $null = $p.Handle } catch {}
    $cpu = 0.0
    while (-not $p.HasExited) {
        try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
        Start-Sleep -Milliseconds 250
    }
    try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
    $script:LastCpuS = [math]::Round($cpu, 1)
}
function CpuBusy() {
    $c = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor |
         Where-Object { $_.Name -eq "_Total" }
    return [int]$c.PercentProcessorTime
}
function StandbyGB() {
    $s = 0.0
    foreach ($ctr in @('\Memory\Standby Cache Normal Priority Bytes',
                       '\Memory\Standby Cache Reserve Bytes',
                       '\Memory\Standby Cache Core Bytes')) {
        try { $s += (Get-Counter $ctr).CounterSamples[0].CookedValue } catch {}
    }
    return [math]::Round($s / 1GB, 2)
}
function Median($xs) {
    $s = @($xs | Sort-Object)
    $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return [math]::Round(($s[$n / 2 - 1] + $s[$n / 2]) / 2.0, 3)
}
function Grp($text, $re, $i) {
    $m = [regex]::Match($text, $re)
    if ($m.Success) { return [double]$m.Groups[$i].Value }
    return $null
}
function PromptArgs($n) {
    $a = @()
    for ($i = 0; $i -lt $n; $i++) {
        $a += "-f"
        $a += ('"' + ("$pdir\s2-512-c{0:d2}.txt" -f $i) + '"')
    }
    return $a
}

# ------------------------------------------------------------------ arms
function ServeCell($seqs, $kv) {
    $a = @("bench", "--serve", ('"' + $model + '"')) + (PromptArgs $seqs) +
         @("--seqs", "$seqs", "--ctx", "$Ctx", "-n", "$NGen",
           "--threads", "$Threads", "--kv", $kv)
    RunCapture $exe $a $tmp $tmpOut
    $t = (Get-Content $tmpOut -Raw)
    if (-not $t) { return $null }
    # ANTI-VACUITY FIRST: the driver prints this line when it produced fewer
    # ids than asked, and every rate above it would still look tidy.
    if ($t -match '! only \d+ of \d+ expected ids') {
        Say "  !! serve produced fewer ids than asked -- unit REJECTED"
        return $null
    }
    return @{
        gate    = (Grp $t 'AGGREGATE\(M0\)\s+([\d.]+)' 1)
        e2e     = (Grp $t 'end-to-end\s+([\d.]+)' 1)
        span_ms = (Grp $t 'decode ids\s*/\s*([\d.]+)\s*ms span' 1)
        ttft50  = (Grp $t 'TTFT p50\s+([\d.]+)' 1)
        ttft95  = (Grp $t 'TTFT p50\s+[\d.]+\s+p95\s+([\d.]+)' 1)
        itl50   = (Grp $t 'ITL\s+p50\s+([\d.]+)' 1)
        itl95   = (Grp $t 'ITL\s+p50\s+[\d.]+\s+p95\s+([\d.]+)' 1)
        itlmax  = (Grp $t 'ITL\s+p50\s+[\d.]+\s+p95\s+[\d.]+\s+max\s+([\d.]+)' 1)
        steps   = (Grp $t 'steps\s+(\d+)' 1)
        chunks  = (Grp $t 'prefill chunks\s+(\d+)' 1)
        preempt = (Grp $t 'preemptions\s+(\d+)' 1)
    }
}

function BatchCell($seqs, $kv) {
    $a = @("batch", ('"' + $model + '"')) + (PromptArgs $seqs) +
         @("--seqs", "$seqs", "--ctx", "$Ctx", "-n", "$NGen",
           "--threads", "$Threads", "--kv", $kv, "--timing")
    RunCapture $exe $a $tmp $tmpOut
    $t = (Get-Content $tmp -Raw)
    if (-not $t) { return $null }
    $pf_tok = Grp $t 'prefill\s+(\d+)\s+tok' 1
    $pf_ms  = Grp $t 'prefill\s+\d+\s+tok in\s+([\d.]+)\s+ms' 1
    $pf_rate = $null
    if ($pf_ms -and $pf_ms -gt 0) { $pf_rate = [math]::Round($pf_tok / ($pf_ms / 1000.0), 3) }
    return @{
        gate     = (Grp $t '\(([\d.]+)\s+tok/s\s+aggregate\)' 1)   # decode-only
        pf_tok   = $pf_tok
        pf_ms    = $pf_ms
        pf_rate  = $pf_rate
        dec_tok  = (Grp $t 'decode\s+(\d+)\s+tok' 1)
        dec_ms   = (Grp $t 'decode\s+\d+\s+tok in\s+([\d.]+)\s+ms' 1)
    }
}

function LlamaCell() {
    $a = @("-m", ('"' + $model + '"'), "-p", "512", "-n", "$NGen",
           "-r", "3", "-t", "$Threads")
    RunCapture $LB $a $tmp $tmpOut
    $t = (Get-Content $tmpOut -Raw)
    if (-not $t) { return $null }
    return @{
        gate = (Grp $t 'tg\d+\s*\|\s*([\d.]+)' 1)     # decode, the box-health anchor
        pp   = (Grp $t 'pp\d+\s*\|\s*([\d.]+)' 1)     # prefill referent
    }
}

# ------------------------------------------------------------------- run
Say "M5 gate [2] -- the bars. stamp $Stamp"
Say "cells: $($Cells -join ',')  blocks: $Blocks  threads: $Threads  ctx: $Ctx  n: $NGen"
Say "exe: $exe"
@{ total = ($Cells.Count * $Blocks * 2 + $Blocks); stamp = $Stamp } |
    ConvertTo-Json -Compress | Out-File -Encoding utf8 $status

# ISLC. The Intelligent Standby List Cleaner purges the standby list on a
# timer, so between legs the model is evicted and every run re-reads it from
# disk -- and worse, the standby guard below then flags every block as
# disk-bound and stops meaning anything. Stop it ONCE for a chain and restore
# it after the last leg (the 08-19 lesson: a leg that restores it purges the
# cache the NEXT leg depends on). This only WARNS -- stopping someone's
# program is not a benchmark script's decision to take unattended.
if (@(Get-Process | Where-Object { $_.Name -match 'ISLC' }).Count -gt 0) {
    Say "  !! ISLC IS RUNNING -- standby readings and the disk-bound guard are meaningless"
}

# QUIESCE GATE. One sample is not a quiesce check -- the 08-14 harness bug
# was exactly that -- so require two consecutive quiet reads.
# >> AND IT SAYS WHICH HAPPENED. The pre-flight version printed "quiesced:
# cpu 17%" after running out of tries at 17% > 12%: a gate that reports its
# own name rather than its own result is worse than no gate.
$quiet = 0
$seen = @()
for ($try = 0; $try -lt $QuietTries; $try++) {
    $busy = CpuBusy
    $seen += $busy
    if ($busy -le $QUIET_CPU) {
        $quiet++
        if ($quiet -ge 2) { break }
    } else {
        $quiet = 0
        Say "  waiting: cpu $busy% > $QUIET_CPU%"
    }
    Start-Sleep -Seconds 20
}
$sb0 = StandbyGB
$quiesced = ($quiet -ge 2)
# Report the readings the GATE USED, not a fresh sample: the first version
# printed "QUIESCE GATE PASSED: cpu 39%", which is a sentence that argues
# against itself and would have been read as a broken gate.
$last2 = ($seen | Select-Object -Last 2) -join "%, "
if ($quiesced) {
    Say "QUIESCE GATE PASSED on cpu $last2% (<= $QUIET_CPU% twice), standby $sb0 GB"
} else {
    Say "!! QUIESCE GATE NOT MET after $QuietTries tries -- last reads $last2%, standby $sb0 GB."
    Say "!! Every rate below is INDICATIVE. Ratios inside a block survive; absolutes do not."
}
Bank @{ id = "quiesce"; kind = "quiesce"; passed = $quiesced; reads = $seen;
        standby_gb = $sb0; utc = (Get-Date -Format s) }

for ($blk = 1; $blk -le $Blocks; $blk++) {
    $sbNow = StandbyGB
    Say "BLOCK $blk starting (cpu $(CpuBusy)%, standby $sbNow GB)"

    foreach ($cell in $Cells) {
        $m = [regex]::Match($cell, '^(\d+)(f32|q8)$')
        if (-not $m.Success) { throw "bad cell '$cell' -- want e.g. 8f32 or 16q8" }
        $seqs = [int]$m.Groups[1].Value
        $kv   = $m.Groups[2].Value

        foreach ($arm in @("S", "B")) {
            $id = "b$blk-$cell-$arm"
            if (HasUnit $id) { Say "  $id already banked, skipping"; continue }
            $t0 = Get-Date
            if ($arm -eq "S") { $r = ServeCell $seqs $kv } else { $r = BatchCell $seqs $kv }
            $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
            if ($null -eq $r -or $null -eq $r.gate) {
                Say "  $id PARSE-FAIL after ${secs}s -- will re-run on resume"
                Bank @{ id = $id; kind = "fail"; arm = $arm; cell = $cell;
                        secs = $secs; utc = (Get-Date -Format s) }
                continue
            }
            $r.id = $id; $r.kind = "cell"; $r.arm = $arm; $r.cell = $cell
            $r.seqs = $seqs; $r.kv = $kv; $r.block = $blk; $r.secs = $secs
            $r.cpu_s = $script:LastCpuS; $r.cpu_busy = (CpuBusy)
            $r.standby_gb = (StandbyGB); $r.utc = (Get-Date -Format s)
            Bank $r
            Say ("  {0,-14} {1,8:n3} tok/s   ({2}s wall, {3}s cpu)" -f $id, $r.gate, $secs, $r.cpu_s)
        }
    }

    $id = "b$blk-L"
    if (-not (HasUnit $id)) {
        $t0 = Get-Date
        $r = LlamaCell
        $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        if ($null -eq $r -or $null -eq $r.gate) {
            Say "  $id PARSE-FAIL after ${secs}s"
            Bank @{ id = $id; kind = "fail"; arm = "L"; secs = $secs; utc = (Get-Date -Format s) }
        } else {
            $r.id = $id; $r.kind = "cell"; $r.arm = "L"; $r.cell = "llama"
            $r.block = $blk; $r.secs = $secs; $r.cpu_s = $script:LastCpuS
            $r.cpu_busy = (CpuBusy); $r.standby_gb = (StandbyGB); $r.utc = (Get-Date -Format s)
            Bank $r
            Say ("  {0,-14} tg {1,7:n3}  pp {2,7:n3} tok/s  ({3}s)" -f $id, $r.gate, $r.pp, $secs)
        }
    }
}

# ---------------------------------------------------------------- report
Say "all blocks done -- writing report"
$all = @(Banked | Where-Object { $_.kind -eq "cell" })

$lines = @()
$lines += "# M5 gate item [2] -- the bars, measured"
$lines += ""
$lines += "Run ``$Stamp``. S2 = 512 prompt tokens (525 real, ChatML) x $NGen generated,"
$lines += "M0's own prompts (``tests/data/bench_prompts/``), ctx $Ctx, threads $Threads, unpinned,"
$lines += "$Blocks interleaved blocks on a quiesced box. Raw: ``$Stamp-raw.txt``, bank: ``$Stamp-bank.jsonl``."
$lines += ""
$lines += "**The gate number is AGGREGATE(M0)**: decode-only ids over the span from the first"
$lines += "token anywhere to the last token anywhere -- `bench/bench.py`'s ``aggregate_decode_tps``,"
$lines += "which is how the 17.7 / 14.2 / 10.8 / 16.8 cells were computed."
$lines += ""
$lines += "## Medians by cell"
$lines += ""
$lines += "| cell | S serve AGG(M0) | S decode-only (N/ITL p50) | S TTFT p50 s | B batch decode | B prefill tok/s | bar | verdict |"
$lines += "|---|---|---|---|---|---|---|---|"
# `B batch decode` IS the static path's M0 number: a lockstep batch finishes
# every prefill before any first token, so its span carries no prefill. That is
# why it is NOT an A/B of the scheduler against S -- compare S's decode-only
# column to it, and S's AGG(M0) to the bar.
foreach ($cell in $Cells) {
    $m = [regex]::Match($cell, '^(\d+)(f32|q8)$')
    $seqs = [int]$m.Groups[1].Value
    $S = @($all | Where-Object { $_.cell -eq $cell -and $_.arm -eq "S" })
    $B = @($all | Where-Object { $_.cell -eq $cell -and $_.arm -eq "B" })
    $sAgg = Median (@($S | ForEach-Object { [double]$_.gate }))
    $sItl = Median (@($S | ForEach-Object { [double]$_.itl50 }))
    $sTtf = Median (@($S | ForEach-Object { [double]$_.ttft50 }))
    $bDec = Median (@($B | ForEach-Object { [double]$_.gate }))
    $bPf  = Median (@($B | ForEach-Object { [double]$_.pf_rate }))
    $sDec = $null
    if ($sItl -and $sItl -gt 0) { $sDec = [math]::Round($seqs / ($sItl / 1000.0), 2) }

    # GATE BARS exist only at N=8 and N=16 -- both families apply at both, and
    # the binding one is the higher (llamacpp-tuned 17.7 at 8; 2x-ollama 16.8
    # at 16, where the inversion makes it the harder bar).
    #
    # >> EVERY OTHER N IS GRADED AGAINST M0's llamacpp-tuned CELL AND LABELLED
    # `ref`. An N-sweep exists to show the SHAPE of the curve, and inventing a
    # gate bar for a width the gate never named would turn a diagnostic into a
    # verdict -- which is how a measurement stops being read as one.
    $GATE = @{ 8 = 17.7; 16 = 16.8 }
    $REF  = @{ 1 = 15.6; 2 = 23.5; 4 = 33.6; 8 = 17.7; 16 = 14.2 }   # M0 07-17
    $isGate = $GATE.ContainsKey($seqs)
    if ($isGate) { $bar = $GATE[$seqs] } else { $bar = $REF[$seqs] }
    $barTxt = "-"
    if ($null -ne $bar) { if ($isGate) { $barTxt = ">=$bar" } else { $barTxt = "$bar ref" } }
    $verdict = "-"
    if ($null -ne $sAgg -and $null -ne $bar) {
        if (-not $isGate) { $verdict = ("{0:n2}x ref" -f ($sAgg / $bar)) }
        elseif ($sAgg -ge $bar) { $verdict = "**MET**" }
        else { $verdict = ("MISS ({0:n2}x)" -f ($sAgg / $bar)) }
    }
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | >={6} | {7} |" -f $cell,
        $(if ($null -ne $sAgg) { "{0:n3}" -f $sAgg } else { "-" }),
        $(if ($null -ne $sDec) { "{0:n2}" -f $sDec } else { "-" }),
        $(if ($null -ne $sTtf) { "{0:n1}" -f ($sTtf / 1000.0) } else { "-" }),
        $(if ($null -ne $bDec) { "{0:n3}" -f $bDec } else { "-" }),
        $(if ($null -ne $bPf)  { "{0:n2}" -f $bPf }  else { "-" }),
        $barTxt, $verdict)
}
$lines += ""
$L = @($all | Where-Object { $_.arm -eq "L" })
$lTg = Median (@($L | ForEach-Object { [double]$_.gate }))
$lPp = Median (@($L | ForEach-Object { [double]$_.pp }))
$lines += "## The non-stoke referent (llama-bench b10037, t $Threads)"
$lines += ""
$lines += ("- tg$NGen (decode, N=1): **{0}** tok/s -- the box-health anchor" -f $lTg)
$lines += ("- pp512 (prefill): **{0}** tok/s -- the prefill referent" -f $lPp)
$lines += ""
$lines += "## Every unit"
$lines += ""
$lines += "| id | arm | cell | gate tok/s | wall s | cpu s | standby GB | cpu % |"
$lines += "|---|---|---|---|---|---|---|---|"
foreach ($u in $all) {
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f
        $u.id, $u.arm, $u.cell, $u.gate, $u.secs, $u.cpu_s, $u.standby_gb, $u.cpu_busy)
}
$lines += ""
$lines += "Any block whose standby cache summed under the 2.33 GB model was reading the"
$lines += "model from DISK and is not comparable; all three buckets are summed (the"
$lines += "Normal-only guard cried wolf on 3/3 good blocks on 2026-08-18). If the raw log"
$lines += "carries an ISLC warning, read the standby column as unavailable rather than low."
$q = @(Banked | Where-Object { $_.kind -eq "quiesce" })
if ($q.Count -gt 0 -and $q[0].passed -ne $true) {
    $lines += ""
    $lines += "**THE QUIESCE GATE DID NOT PASS.** Every absolute rate here is INDICATIVE."
}
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
