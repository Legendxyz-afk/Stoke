# =====================================================================
# M5 GATE ITEM [4] -- S3 FANOUT, MEASURED.
#
#   [4] real: S3 aggregate >= 13.3 (llamacpp-default's cell), M0
#       methodology -- medians, quiesced, same-file, AGGREGATE(M0).
#
# S3 is M0's locked fanout scenario: 8 closed-loop clients, ~200 user
# tokens each over a 1024-token SHARED system prefix, 300 generated. The
# prompts are M0's own, exported verbatim by export_bench_prompts.py.
#
# ▶▶ WHY THERE ARE TWO STOKE ARMS AND ONLY ONE OF THEM IS THE GATE.
#
#   Sd  `bench --serve --dedup` -- THE GATE ARM. The 13.3 bar was set by
#       llama-server b10037 IN ITS DEFAULTS, and its own --help says
#       --slot-prompt-similarity defaults to 0.10 (enabled), --cache-ram to
#       8192 MiB (enabled) and --cache-idle-slots to enabled. S3's clients
#       share ~83% of their tokens and, closed-loop, re-send the same
#       prompt to a slot they already occupied. So the referent was NOT
#       charged 8 fresh prefills of the shared prefix, and a stoke reading
#       that is charged for them is not measuring the same thing.
#       (memo 2026-08-24-m5-gate4-s3-status.md section 4.)
#
#   Sn  `bench --serve` with no dedup -- THE FLOOR, and it is banked as a
#       floor and NEVER as a gate result. It exists to price the dedup work
#       against the estimate in that memo's section 2 (~0.42x of the bar
#       without, ~0.89x with) and to be the denominator the change is
#       graded on.
#
#   L   `llama-bench -p 512 -n 128` -- the non-stoke box-health anchor,
#       once per block. A stoke-only collapse (EcoQoS, 2026-08-17) hits
#       every stoke arm at once, so Sd-vs-Sn still "agrees with itself"
#       while both are wrong; only a non-stoke arm exposes that.
#
#       ⚠ IT IS NOT A SAME-SITTING S3 REFERENT. That would need
#       llama-server up on this box, which is Kabir's call to make (see
#       [[ask-before-running-servers]]), so the bar is the RECORDED 13.3 --
#       the same arrangement gate item [2] was graded under.
#
# ▶ THE ARMS ALSO WRITE THEIR STREAMS (--ids) AND THE REPORT COMPARES
# THEM. It costs one file write per unit and it makes every perf unit a
# correctness unit at the FULL S3 shape (300 generated), where
# run_m5_dedup_verify.ps1 grades at 24. If dedup ever moved the stream at
# length, this notices without a second sitting.
#
# PROTOCOL, per [[measurement-needs-interleaving]]: threads 8, unpinned,
# arms interleaved INSIDE each block with the ORDER ALTERNATING per block,
# medians across blocks, every unit banked as it lands, quiesce gate first,
# and block 1 read against the later blocks for the warming transient.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_gate4.ps1
#              [-Blocks 3] [-NGen 300] [-Seqs 8] [-Threads 8] [-Stamp <name>]
#              [-Quick]
# =====================================================================
param(
    [int]$Blocks     = 3,
    [int]$NGen       = 300,
    [int]$Seqs       = 8,
    [int]$Threads    = 8,
    [int]$Ctx        = 1600,
    [int]$QuietTries = 10,
    [string]$Stamp   = "",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$root  = "M:\New Drive\Desktop\Programming\stoke"
$exe   = "$root\build\win\stoke.exe"
$LB    = "$root\bench\vendor\llama-b10037\llama-bench.exe"
$model = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$pdir  = "$root\tests\data\bench_prompts"
$gr    = "$root\docs\superpowers\gate-reports"

# THE BAR, and where it comes from: M0's baseline table, llamacpp-default on
# S3. Named once, here, so no line below can quietly disagree with it.
$BAR = 13.3

if ($Quick) { $NGen = 16; $Seqs = 2; $Blocks = 1 }
# ⚠ THE PRE-FLIGHT GETS ITS OWN STAMP, AND ITS UNITS CARRY THEIR SHAPE.
# Without this, `-Quick` writes b1-Sd/b1-Sn/b1-B/b1-L into the SAME bank the
# gate resumes from, and the gate then "resumes" 2-seq/16-token pre-flight
# numbers as its own 8-seq/300-token units -- a pre-flight that poisons the
# result it was run to protect. run_m5_dedup_verify.ps1 already keys its units
# `-s${seqs}n${NGen}` for exactly this reason; both belts are worn here.
if (-not $Stamp) {
    $Stamp = "{0}-m5-gate4{1}" -f (Get-Date -Format "yyyy-MM-dd"),
                                  $(if ($Quick) { "-quick" } else { "" })
}
$SHAPE = "s${Seqs}n${NGen}"
$log    = "$gr\$Stamp-raw.txt"
$bank   = "$gr\$Stamp-bank.jsonl"
$report = "$gr\$Stamp-report.md"
$status = "$gr\$Stamp.status.json"
$tmp    = "$env:TEMP\stoke-gate4"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

foreach ($p in @($exe, $LB, $model, $pdir)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}

$QUIET_CPU = 12
$script:LastCpuS = 0.0

# ---------------------------------------------------------------- helpers
# Lifted from run_m5_bars.ps1 with its scars intact: Start-Process for raw
# stderr, all three standby buckets, a Median that does not round
# half-to-even, and .Handle touched once so cpu-seconds are not silently 0.
function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Bank($obj) { ($obj | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 $bank }
$script:Done = 0
function Progress() {
    $script:Done++
    @{ total = ($Blocks * 4); done = $script:Done; stamp = $Stamp; shape = $SHAPE
       updated = (Get-Date -Format s) } | ConvertTo-Json -Compress |
        Out-File -Encoding utf8 $status
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
            # A BANKED FAILURE IS NOT A BANKED UNIT. The failure path says "will
            # re-run on resume" and an unqualified `else { return $true }` made
            # that a lie: a parse-fail or a rejected no-adoption unit would be
            # skipped forever, and the gate would quietly grade on 2 of 3 blocks
            # while the report's row count looked deliberate.
            if ($b.kind -eq "cell") { if ($null -ne $b.gate) { return $true } }
            elseif ($b.kind -ne "fail") { return $true }
        }
    }
    return $false
}
function RunCapture($e, $argList, $errPath, $outPath) {
    $p = Start-Process -FilePath $e -ArgumentList $argList `
                       -RedirectStandardError $errPath `
                       -RedirectStandardOutput $outPath `
                       -NoNewWindow -PassThru
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
        $a += "-f"; $a += ('"' + ("$pdir\s3-fanout-c{0:d2}.txt" -f $i) + '"')
    }
    return $a
}

# ------------------------------------------------------------------ arms
function ServeCell($dedup, $idsPath) {
    $a = @("bench", "--serve", ('"' + $model + '"')) + (PromptArgs $Seqs) +
         @("--seqs", "$Seqs", "--ctx", "$Ctx", "-n", "$NGen",
           "--threads", "$Threads", "--kv", "f32",
           "--ids", ('"' + $idsPath + '"'))
    if ($dedup) { $a += "--dedup" }
    RunCapture $exe $a "$tmp\e.txt" "$tmp\o.txt"
    $t = (Get-Content "$tmp\o.txt" -Raw)
    if (-not $t) { return $null }
    # ANTI-VACUITY FIRST: the driver prints this when it produced fewer ids
    # than asked, and every rate above it would still look tidy.
    if ($t -match '! only \d+ of \d+ expected ids') {
        Say "  !! produced fewer ids than asked -- unit REJECTED"
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
        rows    = (Grp $t 'rows (\d+)' 1)
        steps   = (Grp $t 'steps\s+(\d+)' 1)
        chunks  = (Grp $t 'prefill chunks\s+(\d+)' 1)
        preempt = (Grp $t 'preemptions\s+(\d+)' 1)
        adopted = (Grp $t 'dedup: (\d+) of' 1)
    }
}

# M4's STATIC path on the same prompts, with dedup, and it is the ATTRIBUTION
# instrument the playbook's ladder names first: if the static batch already
# misses the bar on decode, the scheduler was never the problem. It also splits
# prefill from decode, which the serve driver can only do through TTFT.
#
# ▶ IT IS COLLECTED IN THIS SITTING OR NOT AT ALL. Attribution taken later is
# taken at a different clock -- the box reaches equilibrium and cross-sitting
# absolutes do not compare ([[measurement-needs-interleaving]]).
#
# ⚠ ITS AGGREGATE IS NOT COMPARABLE TO Sd's. A lockstep batch finishes every
# prefill BEFORE any first token, so its M0 span contains no prefill at all,
# while the scheduler is charged with every later arrival's prefill inside its
# span. Compare Sd's decode-only rate (N / ITL p50) against B's decode
# aggregate; compare Sd's AGGREGATE(M0) against the bar.
function BatchCell() {
    $a = @("batch", ('"' + $model + '"')) + (PromptArgs $Seqs) +
         @("--seqs", "$Seqs", "--ctx", "$Ctx", "-n", "$NGen",
           "--threads", "$Threads", "--kv", "f32", "--timing", "--dedup")
    RunCapture $exe $a "$tmp\e.txt" "$tmp\o.txt"
    $t = (Get-Content "$tmp\e.txt" -Raw)     # timing + dedup go to stderr here
    if (-not $t) { return $null }
    $pf_tok = Grp $t 'prefill\s+(\d+)\s+tok' 1
    $pf_ms  = Grp $t 'prefill\s+\d+\s+tok in\s+([\d.]+)\s+ms' 1
    $pf_rate = $null
    if ($pf_ms -and $pf_ms -gt 0) { $pf_rate = [math]::Round($pf_tok / ($pf_ms / 1000.0), 3) }
    return @{
        gate    = (Grp $t '\(([\d.]+)\s+tok/s\s+aggregate\)' 1)   # decode-only
        pf_tok  = $pf_tok
        pf_ms   = $pf_ms
        pf_rate = $pf_rate
        dec_tok = (Grp $t 'decode\s+(\d+)\s+tok' 1)
        dec_ms  = (Grp $t 'decode\s+\d+\s+tok in\s+([\d.]+)\s+ms' 1)
        adopted = (Grp $t 'dedup: (\d+) of' 1)
    }
}

function LlamaCell() {
    $a = @("-m", ('"' + $model + '"'), "-p", "512", "-n", "128", "-r", "3",
           "-t", "$Threads")
    RunCapture $LB $a "$tmp\e.txt" "$tmp\o.txt"
    $t = (Get-Content "$tmp\o.txt" -Raw)
    if (-not $t) { return $null }
    return @{ gate = (Grp $t 'tg\d+\s*\|\s*([\d.]+)' 1)
              pp   = (Grp $t 'pp\d+\s*\|\s*([\d.]+)' 1) }
}

# ------------------------------------------------------------------- run
# STALENESS, BEFORE ANYTHING IS MEASURED. Every measurement bug this project
# has recorded was caught by a pre-flight, and "the binary predates the feature
# under test" is the cheapest of them to check: --dedup landed in `729ebfc` and
# an exe older than src/ is not the engine this gate names.
$newestSrc = (Get-ChildItem "$root\src" -Include *.c, *.h -Recurse |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1)
if ((Get-Item -LiteralPath $exe).LastWriteTime -lt $newestSrc.LastWriteTime) {
    throw ("STOP: $exe is OLDER than $($newestSrc.Name) -- rebuild before gating")
}

Say "M5 gate [4] -- S3 fanout. stamp $Stamp"
Say "seqs $Seqs x n $NGen, ctx $Ctx, threads $Threads, blocks $Blocks, bar $BAR"
Say "exe: $exe  commit $(git -C $root rev-parse --short HEAD)"
Say "exe built $((Get-Item -LiteralPath $exe).LastWriteTime.ToString('s')), newest src $($newestSrc.Name) $($newestSrc.LastWriteTime.ToString('s'))"
if ($Quick) { Say "*** -Quick: NOT a gate (2 seqs, n 16, 1 block) ***" }
# 4 arms per block, not 3 -- L is a unit too, and a total that undercounts is a
# progress figure that reads done while a block is still running.
@{ total = ($Blocks * 4); done = 0; stamp = $Stamp; shape = $SHAPE
   started = (Get-Date -Format s) } |
    ConvertTo-Json -Compress | Out-File -Encoding utf8 $status

if (@(Get-Process | Where-Object { $_.Name -match 'ISLC' }).Count -gt 0) {
    Say "  !! ISLC IS RUNNING -- standby readings and the disk-bound guard are meaningless"
}

# MEMORY HEADROOM, CHECKED BEFORE THE FIRST UNIT. The pool here is
# 8 x blocks(1600) = 800 blocks = ~3.8 GB of f32 KV beside a 2.33 GB model --
# far more than gate [2]'s ctx 704 cells. A box that has to page for it
# measures the page file. ullAvailPhys, not FreePhysicalMemory: the latter
# excludes exactly the standby pages an mmapped model lives in (2026-08-06).
$avail = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)
$standby0 = StandbyGB
Say "  free physical $avail GB + standby $standby0 GB (pool needs ~3.8 GB + 2.33 GB model)"

# QUIESCE GATE. One sample is not a quiesce check -- require two consecutive
# quiet reads, and report the readings the GATE USED, not a fresh sample.
$quiet = 0; $seen = @()
for ($try = 0; $try -lt $QuietTries; $try++) {
    $busy = CpuBusy
    $seen += $busy
    if ($busy -le $QUIET_CPU) { $quiet++; if ($quiet -ge 2) { break } }
    else { $quiet = 0; Say "  waiting: cpu $busy% > $QUIET_CPU%" }
    Start-Sleep -Seconds 20
}
$quiesced = ($quiet -ge 2)
$last2 = ($seen | Select-Object -Last 2) -join "%, "
if ($quiesced) { Say "QUIESCE GATE PASSED on cpu $last2% (<= $QUIET_CPU% twice)" }
else {
    Say "!! QUIESCE GATE NOT MET after $QuietTries tries -- last reads $last2%."
    Say "!! Every rate below is INDICATIVE. Ratios inside a block survive; absolutes do not."
}
Bank @{ id = "quiesce"; kind = "quiesce"; passed = $quiesced; reads = $seen
        standby_gb = $standby0; free_gb = $avail; utc = (Get-Date -Format s) }

for ($blk = 1; $blk -le $Blocks; $blk++) {
    Say "BLOCK $blk starting (cpu $(CpuBusy)%, standby $(StandbyGB) GB)"
    # COUNTERBALANCED: the arm order ROTATES by block, so no arm is always
    # first (warming) or always last (plateau) and a trend cannot be read as an
    # arm difference.
    $rot = @(@("Sd", "Sn", "B"), @("Sn", "B", "Sd"), @("B", "Sd", "Sn"))
    $order = $rot[($blk - 1) % 3]
    Say "  order: $($order -join ' then ')"

    foreach ($arm in $order) {
        $id = "b$blk-$arm-$SHAPE"
        if (HasUnit $id) { Say "  $id already banked, skipping"; continue }
        $ids = "$tmp\$id-ids.txt"
        $t0 = Get-Date
        if ($arm -eq "B") { $r = BatchCell } else { $r = ServeCell ($arm -eq "Sd") $ids }
        $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        if ($null -eq $r -or $null -eq $r.gate) {
            Say "  $id PARSE-FAIL after ${secs}s -- will re-run on resume"
            Bank @{ id = $id; kind = "fail"; arm = $arm; secs = $secs; utc = (Get-Date -Format s) }
            continue
        }
        # ANTI-VACUITY ON THE ARM ITSELF: a dedup arm that adopted nothing is
        # a no-dedup run wearing a flag, and it would read as "dedup does not
        # help" rather than as a broken unit.
        if (($arm -eq "Sd" -or $arm -eq "B") -and (-not $r.adopted -or $r.adopted -le 0)) {
            Say "  !! ${id}: --dedup was passed and NOTHING was adopted -- unit REJECTED"
            Bank @{ id = $id; kind = "fail"; arm = $arm; reason = "no adoption"
                    secs = $secs; utc = (Get-Date -Format s) }
            continue
        }
        $r.id = $id; $r.kind = "cell"; $r.arm = $arm; $r.block = $blk
        $r.secs = $secs; $r.cpu_s = $script:LastCpuS; $r.cpu_busy = (CpuBusy)
        $r.standby_gb = (StandbyGB); $r.utc = (Get-Date -Format s)
        # THE SHAPE TRAVELS WITH THE UNIT, so the report can refuse to median
        # across shapes even if a bank is ever hand-merged.
        $r.shape = $SHAPE; $r.seqs = $Seqs; $r.ngen = $NGen; $r.ctx = $Ctx
        if ($arm -ne "B") {
            $r.ids_file = $ids
            # NOT bare Get-FileHash: $ErrorActionPreference is Stop, so a
            # missing ids file would kill a sitting mid-block over a unit that
            # otherwise measured fine. Bank it hashless and let the stream
            # check below report a gap instead of the run dying at minute 40.
            if (Test-Path -LiteralPath $ids) {
                $r.ids_hash = (Get-FileHash -LiteralPath $ids).Hash
            } else {
                Say "  !! ${id}: no ids file at $ids -- banked without a stream hash"
            }
        }
        Bank $r
        Progress
        Say ("  {0,-10} AGG(M0) {1,7:n3} tok/s  TTFT p50 {2,6:n0} ms  adopted {3}  ({4}s wall, {5}s cpu)" -f
             $id, $r.gate, $r.ttft50, $(if ($r.adopted) { $r.adopted } else { 0 }), $secs, $r.cpu_s)
    }

    $id = "b$blk-L-$SHAPE"
    if (-not (HasUnit $id)) {
        $t0 = Get-Date
        $r = LlamaCell
        $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        if ($null -eq $r -or $null -eq $r.gate) {
            Say "  $id PARSE-FAIL after ${secs}s"
            Bank @{ id = $id; kind = "fail"; arm = "L"; secs = $secs; utc = (Get-Date -Format s) }
        } else {
            $r.id = $id; $r.kind = "cell"; $r.arm = "L"; $r.block = $blk
            $r.secs = $secs; $r.cpu_s = $script:LastCpuS; $r.cpu_busy = (CpuBusy)
            $r.standby_gb = (StandbyGB); $r.utc = (Get-Date -Format s)
            $r.shape = $SHAPE; $r.seqs = $Seqs; $r.ngen = $NGen; $r.ctx = $Ctx
            Bank $r
            Progress
            Say ("  {0,-10} tg128 {1,7:n3}  pp512 {2,7:n3} tok/s  ({3}s)" -f $id, $r.gate, $r.pp, $secs)
        }
    }
}

# ---------------------------------------------------------------- report
Say "all blocks done -- writing report"
# ⚠ THE REPORT GRADES ONE SHAPE. The id key stops a differently-shaped unit
# being RESUMED; this stops one being MEDIANED. A unit banked before shapes were
# recorded has no .shape and is excluded rather than assumed -- a gate that
# guesses what it is averaging is the thing this file exists to prevent.
$all = @(Banked | Where-Object { $_.kind -eq "cell" -and $_.shape -eq $SHAPE })
$skipped = @(Banked | Where-Object { $_.kind -eq "cell" -and $_.shape -ne $SHAPE }).Count
if ($skipped -gt 0) { Say "  !! $skipped banked cell(s) of another shape EXCLUDED from this report" }
$Sd = @($all | Where-Object { $_.arm -eq "Sd" })
$Sn = @($all | Where-Object { $_.arm -eq "Sn" })
$B  = @($all | Where-Object { $_.arm -eq "B" })
$L  = @($all | Where-Object { $_.arm -eq "L" })

$mSd = Median (@($Sd | ForEach-Object { [double]$_.gate }))
$mSn = Median (@($Sn | ForEach-Object { [double]$_.gate }))

# THE STREAMS, ACROSS ARMS. Every Sd unit's ids must equal every Sn unit's.
# Free here, and it is the same claim run_m5_dedup_verify.ps1 gates at n 24,
# checked at n $NGen.
$hashes = @($all | Where-Object { $_.ids_hash } | ForEach-Object { $_.ids_hash })
$streamsAgree = (@($hashes | Sort-Object -Unique).Count -le 1)

$lines = @()
$lines += "# M5 gate item [4] -- S3 fanout, measured"
$lines += ""
$lines += "Run ``$Stamp``. S3 = $Seqs clients x ~200 user tokens over a 1024-token SHARED system"
$lines += "prefix, $NGen generated, ctx $Ctx, threads $Threads, kv f32, unpinned. Bar **$BAR**"
$lines += "(M0's llamacpp-default cell). Raw ``$Stamp-raw.txt``, bank ``$Stamp-bank.jsonl``."
$lines += ""
$lines += "**The gate arm is Sd (``--dedup``).** The bar was set by llama-server in its"
$lines += "defaults, where ``--slot-prompt-similarity`` (0.10), ``--cache-ram`` (8192 MiB) and"
$lines += "``--cache-idle-slots`` are all ON, so the referent was never charged eight fresh"
$lines += "prefills of the shared prefix. **Sn is the FLOOR, not a gate result.**"
$lines += ""
$lines += "| arm | AGGREGATE(M0) tok/s | vs bar $BAR | TTFT p50 s | ITL p50 ms | ITL max ms | rows | adopted |"
$lines += "|---|---|---|---|---|---|---|---|"
# ⚠ EVERY MEDIAN IS COMPUTED INTO A VARIABLE FIRST, and that is not style.
# `Median (@(...)), (Median (@(...)))` inside a -f argument list is parsed as a
# COMMAND invocation: the first Median swallows the rest of the comma list as
# its own arguments, the format string is handed fewer args than it has
# placeholders, and the whole report dies AFTER every unit has been paid for.
# It killed the first pre-flight (2026-08-26) at exactly that point.
foreach ($pair in @(@{n = "Sd (dedup) -- GATE"; u = $Sd; m = $mSd },
                    @{n = "Sn (no dedup) -- floor"; u = $Sn; m = $mSn })) {
    if ($pair.u.Count -eq 0) { continue }
    $uTtft   = Median (@($pair.u | ForEach-Object { [double]$_.ttft50 }))
    $uItl    = Median (@($pair.u | ForEach-Object { [double]$_.itl50 }))
    $uItlMax = Median (@($pair.u | ForEach-Object { [double]$_.itlmax }))
    $uRows   = Median (@($pair.u | ForEach-Object { [double]$_.rows }))
    $uAdopt  = Median (@($pair.u | ForEach-Object { [double]$_.adopted }))
    $uRatio  = 0.0
    if ($pair.m) { $uRatio = $pair.m / $BAR }
    $uTtftS  = 0.0
    if ($null -ne $uTtft) { $uTtftS = $uTtft / 1000.0 }
    $lines += ("| {0} | {1:n3} | {2:n2}x | {3:n1} | {4:n1} | {5:n0} | {6} | {7} |" -f
        $pair.n, $pair.m, $uRatio, $uTtftS, $uItl, $uItlMax, $uRows, $uAdopt)
}
$lines += ""
if ($mSd -and $mSn -and $mSn -gt 0) {
    $lines += ("**What dedup bought: {0:n2}x** ({1:n3} -> {2:n3} tok/s aggregate)." -f
               ($mSd / $mSn), $mSn, $mSd)
    $lines += ""
}
# ANTI-VACUITY THAT COSTS NOTHING AND CHECKS THE ARMS AGAINST EACH OTHER: the
# rows the dedup arm did NOT run must equal the positions it reported adopting,
# and the steps it saved must be those positions divided by the prefill chunk.
# An arm that "adopted" without skipping work would pass every check above.
$rSd = Median (@($Sd | ForEach-Object { [double]$_.rows }))
$rSn = Median (@($Sn | ForEach-Object { [double]$_.rows }))
$aSd = Median (@($Sd | ForEach-Object { [double]$_.adopted }))
$cSd = Median (@($Sd | ForEach-Object { [double]$_.chunks }))
$cSn = Median (@($Sn | ForEach-Object { [double]$_.chunks }))
if ($rSd -and $rSn -and $aSd) {
    $delta = $rSn - $rSd
    $ok = $(if ($delta -eq $aSd) { "matches exactly" } else { "**DOES NOT MATCH -- $delta rows vs $aSd adopted**" })
    $lines += ("**Rows the gate arm did not run: {0:n0} ({1:n0} -> {2:n0}), against {3:n0} positions reported adopted -- {4}.**" -f
               $delta, $rSn, $rSd, $aSd, $ok)
    if ($cSd -and $cSn -and ($cSn - $cSd) -gt 0) {
        $lines += ("Prefill chunks {0:n0} -> {1:n0}: {2:n0} chunks removed at {3:n0} tokens each = {4:n0}." -f
                   $cSn, $cSd, ($cSn - $cSd), ($delta / ($cSn - $cSd)), $delta)
    }
    $lines += ""
}
$verdict = "-"
if ($null -ne $mSd) {
    if ($mSd -ge $BAR) { $verdict = "**MET** at {0:n3} >= $BAR" -f $mSd }
    else { $verdict = "**MISS** at {0:n3} = {1:n2}x of $BAR" -f $mSd, ($mSd / $BAR) }
}
if ($B.Count -gt 0) {
    $bDec = Median (@($B | ForEach-Object { [double]$_.gate }))
    $bPf  = Median (@($B | ForEach-Object { [double]$_.pf_rate }))
    $sItl = Median (@($Sd | ForEach-Object { [double]$_.itl50 }))
    $sDec = $null
    if ($sItl -and $sItl -gt 0) { $sDec = [math]::Round($Seqs / ($sItl / 1000.0), 2) }
    $lines += "## Attribution -- the static path on the same prompts (``batch --dedup``)"
    $lines += ""
    $lines += ("- B decode aggregate: **{0:n3}** tok/s; B prefill: **{1:n2}** tok/s" -f $bDec, $bPf)
    $lines += ("- Sd decode-only (N / ITL p50): **{0}** tok/s" -f $sDec)
    $lines += ""
    $lines += "Read B's decode against Sd's decode-only, NOT against Sd's AGGREGATE(M0):"
    $lines += "a lockstep batch's span carries no prefill at all, while the scheduler's carries"
    $lines += "every later arrival's. If B's decode already misses the bar, the scheduler was"
    $lines += "never the problem -- that is the playbook ladder's first rung."
    $lines += ""
}
$lines += "## VERDICT (gate arm Sd): $verdict"
$lines += ""
# THE VERDICT RESTS ON THE MEDIAN, but "every block cleared it" and "the median
# cleared it and one block did not" are different results and the reader is
# entitled to know which one this is.
$sdVals = @($Sd | ForEach-Object { [double]$_.gate } | Sort-Object)
if ($sdVals.Count -gt 1) {
    $lines += ("Gate-arm blocks, worst to best: {0}. **{1} of {2} clear the bar on their own.**" -f
               (($sdVals | ForEach-Object { "{0:n3}" -f $_ }) -join ", "),
               @($sdVals | Where-Object { $_ -ge $BAR }).Count, $sdVals.Count)
    $lines += ""
}
$lines += ("**Streams across every stoke unit: {0}.**" -f
           $(if ($streamsAgree) { "IDENTICAL" } else { "**THEY DIFFER -- read this before any rate above**" }))
$lines += "Both arms wrote their ids; a dedup arm that moved the stream would make every"
$lines += "rate here a rate for a different computation. Each file carries per-sequence"
$lines += "ids AND the decoded text, so the compare covers both."
$lines += ""
# ASCII ONLY IN EMITTED STRINGS: this file is UTF-8 with NO BOM, so PowerShell
# 5.1 decodes it as the ANSI codepage and any non-ASCII literal reaches the
# report as mojibake. Comments can carry whatever they like; $lines cannot.
$lines += "!! **THIS IS NOT AN INDEPENDENT CORRECTNESS GATE, AND MUST NOT BE QUOTED AS ONE.**"
$lines += "S3's eight clients differ by ONE DIGIT, so all eight streams inside a file are"
$lines += "the same stream, and the continuation is a reply to a filler word list that does"
$lines += "not depend on the shared prefix. Measured 2026-08-25: on S3 a mutant that zeroes"
$lines += "the whole adopted K/V is visible only from 8x24 up, and a mutant that skips 16"
$lines += "unwritten positions is NEVER visible here. The dedup correctness gate is the"
# SINGLE-QUOTED, and that is load-bearing: in a double-quoted PowerShell string
# a markdown backtick before r/n/t/b/a/f/v/0 IS AN ESCAPE. `run_... emitted a
# carriage return followed by "un_...", and the report said so.
$lines += 'HETERO fixture (`run_m5_dedup_verify.ps1`, green on all four legs at `4dd7e2f`),'
$lines += "where both mutants differ. What this check buys is that the two arms above priced"
$lines += "the same computation."
$lines += ""
# THE GATE IS A THROUGHPUT GATE, AND THE SAME UNITS PRICE THE LATENCY ONE.
# Reported here so nobody reads "[4] MET" as "S3 is served well".
$itlP50 = Median (@($Sd | ForEach-Object { [double]$_.itl50 }))
$itlMax = Median (@($Sd | ForEach-Object { [double]$_.itlmax }))
$ttftP50 = Median (@($Sd | ForEach-Object { [double]$_.ttft50 }))
if ($itlP50) {
    $lines += ("**And what the same units say about gate [3], which is a different gate:** the")
    $lines += ("gate arm's ITL p50 is **{0:n0} ms** and its max **{1:n0} ms** against a 250 ms bound," -f $itlP50, $itlMax)
    $lines += ("with TTFT p50 **{0:n0} s**. [4] is throughput; on this same data [3] MISSES on the" -f ($ttftP50 / 1000.0))
    $lines += "median as well as the max. Item [4] being met says the fanout is SERVED, not that"
    $lines += "it is served interactively."
    $lines += ""
}
$blks = @($all | ForEach-Object { [int]$_.block } | Sort-Object -Unique)
if ($blks.Count -gt 1) {
    $lines += "## By block (column 1 against the rest: block 1 is the warming transient)"
    $lines += ""
    $lines += ("| arm | " + (($blks | ForEach-Object { "block $_" }) -join " | ") + " |")
    $lines += ("|---|" + (($blks | ForEach-Object { "---" }) -join "|") + "|")
    foreach ($arm in @("Sd", "Sn", "L")) {
        $row = "| $arm |"
        foreach ($b in $blks) {
            $u = @($all | Where-Object { $_.arm -eq $arm -and [int]$_.block -eq $b })
            if ($u.Count -gt 0) { $row += (" {0:n3} |" -f [double]$u[0].gate) } else { $row += " - |" }
        }
        $lines += $row
    }
    $lines += ""
}
if ($L.Count -gt 0) {
    # Same command-form trap as the arm table above -- precomputed, not inlined.
    $lTg = Median (@($L | ForEach-Object { [double]$_.gate }))
    $lPp = Median (@($L | ForEach-Object { [double]$_.pp }))
    $lines += ("**Box health (llama-bench b10037, t $Threads):** tg128 median {0}, pp512 median {1} tok/s." -f
               $lTg, $lPp)
    $lines += "A stoke-only collapse would move both stoke arms and leave this one alone."
    $lines += ""
}
$lines += "## Every unit"
$lines += ""
$lines += "| id | arm | AGG(M0) | wall s | cpu s | standby GB | cpu % | adopted |"
$lines += "|---|---|---|---|---|---|---|---|"
foreach ($u in $all) {
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f
        $u.id, $u.arm, $u.gate, $u.secs, $u.cpu_s, $u.standby_gb, $u.cpu_busy, $u.adopted)
}
$q = @(Banked | Where-Object { $_.kind -eq "quiesce" })
if ($q.Count -gt 0 -and $q[0].passed -ne $true) {
    $lines += ""
    $lines += "**THE QUIESCE GATE DID NOT PASS.** Every absolute rate here is INDICATIVE."
}
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
Say ("VERDICT: Sd median {0} vs bar {1}" -f $mSd, $BAR)
