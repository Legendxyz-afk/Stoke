# M4.5 -- the overnight P-BAR evaluation, and the owed correctness re-runs.
#
# Runs unattended. Every decision it needs is made here in advance; it never
# waits for input and it never stops on a bad arm -- it records and continues,
# because a night that ends with no data is worse than a night that ends with
# data plus an honest caveat.
#
# WHAT IT DOES, in order:
#   PHASE 1  the two owed correctness re-runs (p41 Q4_K_M vs pre-maddubs, and
#            the Q8_0 p41 anchor vs the m3 gate archive) against the RELINKED
#            binary. Deterministic; contention costs only time.
#   PHASE 2  stop ISLC, warm the model into the standby cache, then WAIT for
#            the box to go quiet instead of assuming it is.
#   PHASE 3  four measurement BLOCKS spaced across the night. Each block is
#            three cycles of three arms, and the ARM ORDER ROTATES between
#            cycles.
#   PHASE 4  restore ISLC (in `finally`, so it happens even on a crash) and
#            write the report.
#
# ▶ WHY BLOCKS SPACED WITH IDLE GAPS, rather than one long run. This box drifts
# 15-20% over hours of sustained load, and every perf number this project has
# had to retract came from a single block. Four blocks separated by idle gaps
# start from different thermal states, so drift shows up as disagreement
# BETWEEN blocks instead of hiding inside one median.
#
# ▶ WHY THE ARM ORDER ROTATES. On 2026-08-14 the A arm ran first in every cycle
# and came back 4.971 / 7.445 / 8.387 -- a 1.69x spread, monotonically
# declining, well under its own 10.01 anchor. Fixed position and interference
# are confounded when the order never changes. Rotating A,B,L -> B,L,A -> L,A,B
# separates them.
#
# ▶ THE ANCHOR CHECK IS THE GUARD. Arm A is the archived pre-maddubs binary,
# whose N=1 rate was measured at 10.01 tok/s on a quiesced box. If A does not
# reproduce that, the box was not ready and every ratio computed against it is
# inflated. The report states A's deviation on its face; it does NOT silently
# publish a speedup taken from a depressed baseline.
#
# EVERY UNIT IS BANKED THE MOMENT IT LANDS, to a JSONL that is re-read on
# restart so a killed run resumes instead of repeating. Two background shells
# were killed mid-run on 2026-08-14 and one lost everything because it built
# its results in memory and wrote at the end; this does not.

# Parameterised ONLY so the pre-flight smoke run can exercise this exact file
# rather than a copy of it -- a copy is a second thing to get wrong, and the
# defaults below are the production values.
param(
    [int]$Blocks      = 4,
    [int]$Cycles      = 3,
    [int]$BlockGapM   = 75,
    [int]$QuietTries  = 40,
    [int]$NGen        = 128,
    [string]$Stamp    = "2026-08-15-m45-overnight-pbar",
    [string]$CorrPrompt = "p41",
    # The two engine arms, so the SAME fixed machinery can measure a different
    # pair (e.g. maddubs vs A') instead of a second script drifting from this
    # one. Empty means the defaults below.
    [string]$ArmAExe  = "",
    [string]$ArmBExe  = "",
    [string]$ArmALabel = "A pre-maddubs",
    [string]$ArmBLabel = "B maddubs",
    [double]$AnchorA  = 10.01,
    # Poll until this scheduled task is no longer Running before measuring, so
    # a follow-on measurement cannot overlap the one in front of it.
    [string]$WaitForTask = "",
    [switch]$SkipCorrectness,
    # A follow-on run starts while the run in front of it has ISLC stopped, so
    # it records islc_was_running=False and would leave Kabir's tool off for
    # good. This forces the restore regardless.
    [switch]$AlwaysRestoreISLC
)

$ErrorActionPreference = 'Continue'   # a failed unit must not kill the night
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFF   # P-cores

$root  = "M:\New Drive\Desktop\Programming\stoke"
$gr    = "$root\docs\superpowers\gate-reports"
$stamp = $Stamp
$bank  = "$gr\$stamp-bank.jsonl"
$log   = "$gr\$stamp-raw.txt"
$rep   = "$gr\$stamp-report.md"
$tmp   = "$env:TEMP\stoke-overnight-err.txt"
$tmpOut = "$env:TEMP\stoke-overnight-out.txt"

$Q4    = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$Q8    = "$root\models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
$PRE   = "$root\build\stoke-m45-c-head-bc9dd3d6.exe"   # pre-maddubs
$NEW   = "$root\build\win\stoke.exe"                    # maddubs
if ($ArmAExe) { $PRE = $ArmAExe }
if ($ArmBExe) { $NEW = $ArmBExe }
$M3    = "$root\build\stoke-m3-gate-876ce53f.exe"
$LB    = "$root\bench\vendor\llama-b10037\llama-bench.exe"
$ISLC  = "M:\New Drive\Downloads\ISLC v1.0.4.5\Intelligent standby list cleaner ISLC.exe"

$N_GEN        = $NGen
$BLOCKS       = $Blocks
$CYCLES       = $Cycles
$BLOCK_GAP_M  = $BlockGapM   # minutes between block STARTS
$ANCHOR_A     = $AnchorA     # arm A's quiesced N=1 rate, for the guard
$QUIET_CPU    = 12           # percent; box must be at or under this to start
$QUIET_TRIES  = $QuietTries  # x 60s -- how long to wait for the box to settle
$CORR         = $CorrPrompt

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

# A rate unit that failed to PARSE is not done -- it must re-run, or a
# transient capture bug becomes a permanent hole in the night's data.
function HasUnit($id) {
    foreach ($b in (Banked)) {
        if ($b.id -eq $id) {
            if ($b.kind -eq "rate") {
                if ($null -ne $b.tok_s) { return $true }
            } else {
                return $true
            }
        }
    }
    return $false
}

# ▶▶ WHY Start-Process AND NOT `& $exe ... 2> $tmp`, learned the hard way at
# 23:26 on 2026-08-14. PowerShell renders a native command's stderr as a
# NativeCommandError record and WRAPS IT AT CONSOLE WIDTH when it lands in a
# file. Arm A's binary name is 29 characters against arm B's 9, so the
# "<name> : " prefix shifted the text and the wrap fell INSIDE
# "(5.238 tok/s aggregate)" -- the regex missed, arm A logged PARSE-FAIL, and
# arm B on the identical code path parsed fine. Every arm-A measurement for
# the whole night would have been lost to a cosmetic difference in a file name.
#
# Start-Process writes the child's raw stderr bytes with no PowerShell
# decoration and no wrapping, so the capture no longer depends on how wide a
# console someone imagined.
function RunCapture($exe, $argList, $errPath, $outPath) {
    Start-Process -FilePath $exe -ArgumentList $argList `
                  -RedirectStandardError $errPath `
                  -RedirectStandardOutput $outPath `
                  -NoNewWindow -Wait
}

# tok/s from `stoke batch --timing`; the format is
#   "decode 128 tok in 11637.2 ms (11.000 tok/s aggregate)"
function StokeRate($exe, $model, $prompt, $kv) {
    $a = @("batch", ('"' + $model + '"'), "--raw", "-n", "$N_GEN",
           "-f", ('"' + $prompt + '"'), "--seqs", "1", "--threads", "6",
           "--timing")
    if ($kv) { $a += @("--kv", $kv) }
    RunCapture $exe $a $tmp $tmpOut
    $t = (Get-Content $tmp -Raw)
    if (-not $t) { return $null }
    $m = [regex]::Match($t, '\(([\d.]+)\s+tok/s\s+aggregate\)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

# t/s from llama-bench's markdown row: "| ... | tg128 | 18.05 ± 0.10 |"
# (llama-bench puts the table on STDOUT, so that is the file we parse.)
function LlamaRate($model) {
    $a = @("-m", ('"' + $model + '"'), "-p", "0", "-n", "$N_GEN",
           "-r", "3", "-t", "6")
    RunCapture $LB $a $tmp $tmpOut
    $t = (Get-Content $tmpOut -Raw)
    if (-not $t) { return $null }
    $m = [regex]::Match($t, 'tg\d+\s*\|\s*([\d.]+)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

function StreamsMatch($exeA, $exeB, $model, $prompt, $kv) {
    $a = @("run", $model, "--raw", "-n", "48", "-f", $prompt, "--threads", "6")
    if ($kv) { $a += @("--kv", $kv) }
    $x = (& $exeA @a | Out-String)
    $y = (& $exeB @a | Out-String)
    return ($x -eq $y)
}

function CpuBusy() {
    $c = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor |
         Where-Object { $_.Name -eq "_Total" }
    return [int]$c.PercentProcessorTime
}

function StandbyGB() {
    $s = (Get-Counter '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue
    return [math]::Round($s / 1GB, 2)
}

function Median($xs) {
    $s = @($xs | Sort-Object)
    if ($s.Count -eq 0) { return $null }
    return $s[[int](($s.Count - 1) / 2)]
}

# ---------------------------------------------------------------- run
$islcWasRunning = [bool](Get-Process -Name "*ISLC*" -ErrorAction SilentlyContinue)
Say "START  islc_was_running=$islcWasRunning  blocks=$BLOCKS cycles=$CYCLES n=$N_GEN"

try {
    # ------------------------------------------------------- PHASE 0
    # Never measure on top of another measurement. A follow-on run waits for
    # the one in front of it to finish rather than trusting a start time.
    if ($WaitForTask) {
        $waited = 0
        while ($waited -lt 480) {
            $st = (Get-ScheduledTask -TaskName $WaitForTask -ErrorAction SilentlyContinue)
            if (-not $st -or $st.State -ne "Running") {
                Say "PHASE 0: '$WaitForTask' is not running (waited $waited min) -- proceeding"
                break
            }
            if ($waited % 15 -eq 0) { Say "PHASE 0: waiting for '$WaitForTask' ($waited min)" }
            Start-Sleep -Seconds 60
            $waited++
        }
        Start-Sleep -Seconds 120   # let the box settle after the previous run
    }

    # ------------------------------------------------------- PHASE 1
    Say "PHASE 1: the owed correctness re-runs"
    if ($SkipCorrectness) { Say "  skipped by request (already discharged)" }
    $checks = @(
        @{ id = "corr-q4km-p41"; desc = "Q4_K_M p41 relinked vs pre-maddubs";
           a = $PRE; b = $NEW; model = $Q4; prompt = "$root\tests\data\parity\$CORR.txt"; kv = "f32" },
        @{ id = "corr-q8-p41";   desc = "Q8_0 p41 anchor vs m3 archive";
           a = $M3;  b = $NEW; model = $Q8; prompt = "$root\tests\data\parity\$CORR.txt"; kv = $null }
    )
    foreach ($c in $checks) {
        if ($SkipCorrectness) { continue }
        if (HasUnit $c.id) { Say "  $($c.id) [banked]"; continue }
        $ok = StreamsMatch $c.a $c.b $c.model $c.prompt $c.kv
        Bank @{ id = $c.id; kind = "correctness"; desc = $c.desc;
                identical = $ok; utc = (Get-Date -Format s) }
        Say ("  {0}: {1}" -f $c.id, $(if ($ok) { "IDENTICAL" } else { "DIFFER  <-- INVESTIGATE" }))
    }

    # ------------------------------------------------------- PHASE 2
    Say "PHASE 2: stopping ISLC, waiting for quiet"
    Get-Process -Name "*ISLC*" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
    Say ("  standby cache {0} GB before the wait" -f (StandbyGB))

    # ▶ THREE CONSECUTIVE SAMPLES, not one. The single-sample version passed at
    # 23:25:56 on an 11% reading and block 1 started seconds later at 22% --
    # a bouncing signal (16-33%, Windows' own idle-triggered maintenance) dips
    # under any threshold eventually if you keep asking. A gate on a noisy
    # signal has to see the quiet persist.
    $tries = 0
    $consec = 0
    while ($tries -lt $QUIET_TRIES) {
        $busy = CpuBusy
        if ($busy -le $QUIET_CPU) {
            $consec++
            if ($consec -ge 3) {
                Say "  box quiet at $busy% ($consec consecutive) after $tries min"
                break
            }
            Say "  quiet sample $consec/3 at $busy%"
        } else {
            if ($consec -gt 0) { Say "  quiet streak broken at $busy%" }
            $consec = 0
            Say "  waiting: cpu $busy% > $QUIET_CPU%"
        }
        Start-Sleep -Seconds 60
        $tries++
    }
    if ($tries -ge $QUIET_TRIES) {
        Say "  NEVER WENT QUIET -- measuring anyway and flagging it in the report"
    }
    Bank @{ id = "quiesce"; kind = "quiesce"; cpu_busy = (CpuBusy);
            standby_gb = (StandbyGB); waited_min = $tries; utc = (Get-Date -Format s) }

    # ------------------------------------------------------- PHASE 3
    $orders = @(@("A","B","L"), @("B","L","A"), @("L","A","B"))
    $t0 = Get-Date
    for ($blk = 1; $blk -le $BLOCKS; $blk++) {
        $due = $t0.AddMinutes(($blk - 1) * $BLOCK_GAP_M)
        while ((Get-Date) -lt $due) {
            $wait = [int]($due - (Get-Date)).TotalSeconds
            if ($wait -le 0) { break }
            Start-Sleep -Seconds ([math]::Min($wait, 300))
        }
        # ▶ WARM IMMEDIATELY BEFORE EACH BLOCK, NOT ONCE AT THE START. The
        # first version warmed the cache and THEN waited up to 40 minutes for
        # the box to go quiet, by which time Windows had evicted the model:
        # block 1 began with standby at 0.21 GB against a 2.33 GB model, i.e.
        # measuring the DISK. The 80-minute gaps between blocks have the same
        # effect, so every block re-warms and the result is CHECKED rather
        # than assumed -- free RAM reads healthy while the model is entirely
        # absent from cache, which is the trap that produced a 1.4 tok/s
        # "result" once before.
        for ($wtry = 1; $wtry -le 3; $wtry++) {
            $null = StokeRate $NEW $Q4 "$root\tests\data\parity\p00.txt" "f32"
            $sb = StandbyGB
            if ($sb -ge 2.0) { break }
            Say "  warm $wtry/3 left standby at $sb GB -- warming again"
        }
        $sbNow = StandbyGB
        if ($sbNow -lt 2.0) {
            Say "  ⚠ standby only $sbNow GB after 3 warms -- block $blk MAY BE DISK-BOUND"
        }
        Say "BLOCK $blk starting (cpu $(CpuBusy)%, standby $sbNow GB)"
        Bank @{ id = "b$blk-state"; kind = "blockstate"; block = $blk;
                cpu_busy = (CpuBusy); standby_gb = $sbNow;
                utc = (Get-Date -Format s) }
        for ($cyc = 1; $cyc -le $CYCLES; $cyc++) {
            foreach ($arm in $orders[($cyc - 1) % $orders.Count]) {
                $id = "b$blk-c$cyc-$arm"
                if (HasUnit $id) { Say "  $id [banked]"; continue }
                $v = $null
                if ($arm -eq "A") { $v = StokeRate $PRE $Q4 "$root\tests\data\parity\p00.txt" "f32" }
                if ($arm -eq "B") { $v = StokeRate $NEW $Q4 "$root\tests\data\parity\p00.txt" "f32" }
                if ($arm -eq "L") { $v = LlamaRate $Q4 }
                Bank @{ id = $id; kind = "rate"; block = $blk; cycle = $cyc;
                        arm = $arm; tok_s = $v; cpu_busy = (CpuBusy);
                        utc = (Get-Date -Format s) }
                Say ("  {0}  {1}" -f $id, $(if ($null -eq $v) { "PARSE-FAIL" } else { $v }))
            }
        }
    }
}
finally {
    # ------------------------------------------------------- PHASE 4
    if ($islcWasRunning -or $AlwaysRestoreISLC) {
        if (-not (Get-Process -Name "*ISLC*" -ErrorAction SilentlyContinue)) {
            Start-Process -FilePath $ISLC
            Start-Sleep -Seconds 3
        }
        $back = [bool](Get-Process -Name "*ISLC*" -ErrorAction SilentlyContinue)
        Say ("ISLC restore attempted (was_running={0}, forced={1}): running={2}" -f $islcWasRunning, [bool]$AlwaysRestoreISLC, $back)
    } else {
        Say "ISLC was not running at start -- not started"
    }

    $rows = @(Banked)
    $rate = @($rows | Where-Object { $_.kind -eq "rate" -and $null -ne $_.tok_s })
    $lines = @()
    $lines += "# M4.5 overnight P-BAR evaluation"
    $lines += ""
    $lines += "Run $stamp. Arms: A = $ArmALabel, B = $ArmBLabel, L = llama-bench b10037."
    $lines += ""
    $lines += "- A: ``$PRE``"
    $lines += "- B: ``$NEW``"
    $lines += "N=1 via ``batch --seqs 1``, p00, -n $N_GEN, 6 threads, P-cores. Arm order rotates per cycle."
    $lines += ""
    $lines += "## Correctness (owed re-runs)"
    $lines += ""
    foreach ($c in @($rows | Where-Object { $_.kind -eq "correctness" })) {
        $lines += ("- **{0}** -- {1}" -f $(if ($c.identical) { "IDENTICAL" } else { "DIFFER" }), $c.desc)
    }
    $lines += ""
    $lines += "## Rates by block"
    $lines += ""
    $lines += "| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |"
    $lines += "|---|---|---|---|---|---|---|---|"
    $states = @($rows | Where-Object { $_.kind -eq "blockstate" })
    $blockMed = @{}
    for ($b = 1; $b -le $BLOCKS; $b++) {
        $ma = Median @($rate | Where-Object { $_.block -eq $b -and $_.arm -eq "A" } | ForEach-Object { $_.tok_s })
        $mb = Median @($rate | Where-Object { $_.block -eq $b -and $_.arm -eq "B" } | ForEach-Object { $_.tok_s })
        $ml = Median @($rate | Where-Object { $_.block -eq $b -and $_.arm -eq "L" } | ForEach-Object { $_.tok_s })
        $blockMed[$b] = @{ a = $ma; b = $mb; l = $ml }
        $ba = "n/a"; $bl = "n/a"
        if ($ma -and $mb) { $ba = "{0:n3}x" -f ($mb / $ma) }
        if ($ml -and $mb) { $bl = "{0:n3}x" -f ($mb / $ml) }
        $st = @($states | Where-Object { $_.block -eq $b })
        $sbv = "?"; $cpv = "?"
        if ($st.Count -gt 0) {
            $sbv = $st[$st.Count - 1].standby_gb
            $cpv = $st[$st.Count - 1].cpu_busy
        }
        $flag = ""
        if ($sbv -ne "?" -and [double]$sbv -lt 2.0) { $flag = " ⚠ disk-bound" }
        $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} | {6}{7} | {8} |" -f $b, $ma, $mb, $ml, $ba, $bl, $sbv, $flag, $cpv)
    }
    $lines += ""
    $lines += "A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in."
    $lines += ""
    $allA = Median @($rate | Where-Object { $_.arm -eq "A" } | ForEach-Object { $_.tok_s })
    $allB = Median @($rate | Where-Object { $_.arm -eq "B" } | ForEach-Object { $_.tok_s })
    $allL = Median @($rate | Where-Object { $_.arm -eq "L" } | ForEach-Object { $_.tok_s })
    $lines += "## Overall medians"
    $lines += ""
    $lines += ("- A pre-maddubs : {0}" -f $allA)
    $lines += ("- B maddubs     : {0}" -f $allB)
    $lines += ("- L llama-bench : {0}" -f $allL)
    $lines += ""
    $lines += "## The anchor guard"
    $lines += ""
    if ($allA) {
        $dev = 100.0 * ($allA - $ANCHOR_A) / $ANCHOR_A
        $lines += ("Arm A measured **{0}** against its quiesced anchor of {1}, a deviation of **{2:n1}%**." -f $allA, $ANCHOR_A, $dev)
        if ([math]::Abs($dev) -gt 10) {
            $lines += ""
            $lines += "⚠ **A did not reproduce its anchor within 10%.** Every ratio computed against it is suspect: a depressed baseline inflates the speedup. Treat the B/A column as UNRELIABLE and read B/L instead, which does not depend on A."
        } else {
            $lines += ""
            $lines += "A reproduced its anchor within 10%, so the B/A ratios rest on a baseline that is behaving as it did when the anchor was frozen."
        }
    }
    $lines += ""
    $lines += "## P-BAR"
    $lines += ""
    if ($allB -and $allL) {
        $bar = 0.85 * $allL
        $lines += ("P-BAR is 0.85x llama measured in the same sitting: **{0:n2} tok/s**. B measured **{1}**." -f $bar, $allB)
        if ($allB -ge $bar) {
            $lines += ""
            $lines += "**P-BAR MET.**"
        } else {
            $lines += ""
            $lines += ("**P-BAR NOT MET** -- short by {0:n3}x. A' (row tiling) is the next lever and is not optional." -f ($bar / $allB))
        }
    }
    $lines += ""
    $lines += "## Stability across blocks"
    $lines += ""
    $bs = @($rate | Where-Object { $_.arm -eq "B" } | ForEach-Object { $_.tok_s })
    if ($bs.Count -gt 1) {
        $mn = ($bs | Measure-Object -Minimum).Minimum
        $mx = ($bs | Measure-Object -Maximum).Maximum
        $lines += ("Arm B across every run: min {0}, max {1}, spread {2:n2}x over {3} runs spanning the night." -f $mn, $mx, ($mx / $mn), $bs.Count)
        $lines += ""
        $lines += "A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure."
    }
    $lines += ""
    $lines += "Bank: ``$stamp-bank.jsonl``. Raw log: ``$stamp-raw.txt``."
    ($lines -join "`r`n") | Out-File -Encoding utf8 $rep
    Say "REPORT written -> $rep"
    Say "DONE"
}
