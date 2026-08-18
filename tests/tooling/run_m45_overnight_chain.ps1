# M4.5 -- the overnight chain: every remaining perf question, in one start.
#
# WHY THIS EXISTS. Correctness for L4 closed today (10/10 byte-compare, one
# sitting, a48de9f). Speed is the only open item in the milestone, and it needs
# four legs that together run ~8 h on a box that must be QUIET. Kabir starts it
# once and shuts the box down at 07:30.
#
# THE LEGS, in priority order -- if the night runs short, the tail is what gets
# dropped, so the most load-bearing measurement runs first:
#   1  preflight A/B   1 cycle, n=8       ~15 min   catches a broken instrument
#   2  the real A/B    3 cycles, n=128    ~2.0 h    L4's open question
#   3  P-BAR t=6 PINNED                   ~3.5 h    the config the bar was last graded at
#   4  P-BAR t=8 UNPINNED                 ~3.0 h    the second config it was graded at
#
# > THE DEADLINE IS HARD. Every leg is checked against 07:15 before it starts,
# and skipped with a reason if it cannot finish -- a leg cut off mid-run by a
# shutdown is not a shorter measurement, it is a wasted one, and the P-BAR's
# per-block bank would then hold a partial night that reads like a whole one.
#
# > THE BOX GATE. run_overnight_pbar.ps1 waits for quiet itself (PHASE 2); the
# A/B does not, so this waits for it. A game was running when this was queued,
# which is exactly the case the gate is for: measuring over Satisfactory would
# have produced numbers that look like a regression in the engine.
#
# > WHY THE PREFLIGHT GATES THE REAL RUN. This project's record is that
# instrument bugs are caught by preflight or by watching the first unit, never
# by a run finishing (four separate harness bugs in one night on 2026-08-14).
# If the preflight cannot parse a rate, leg 2 is skipped rather than run blind
# for two hours -- but the P-BAR legs still go, because they are a different
# runner with its own parser and its own bank.

param(
    [string]$DeadlineHHmm = "07:15",
    [int]$QuietPollMin    = 3,
    [double]$QuietCpuPct  = 12.0,
    [double]$MinFreeGB    = 3.5
)

$ErrorActionPreference = 'Continue'   # a failed leg must not kill the night

$root   = "M:\New Drive\Desktop\Programming\stoke"
$gr     = "$root\docs\superpowers\gate-reports"
$log    = "$gr\2026-08-18-m45-overnight-chain.txt"
$status = "$gr\2026-08-18-m45-overnight-chain-status.txt"
$ps     = "powershell.exe"

# Deadline: the next occurrence of DeadlineHHmm. Queued in the evening, so it
# is tomorrow morning; computed rather than assumed so a restart at 02:00 still
# lands on the right day.
$now = Get-Date
$deadline = [datetime]::ParseExact($DeadlineHHmm, "HH:mm", $null)
$deadline = $now.Date.AddHours($deadline.Hour).AddMinutes($deadline.Minute)
if ($deadline -le $now) { $deadline = $deadline.AddDays(1) }

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding ascii $log
}

function Stamp($stage, $note) {
    ("stage={0}`r`nsince={1}`r`ndeadline={2}`r`nnote={3}" -f
        $stage, (Get-Date -Format s), ($deadline.ToString("s")), $note) |
        Out-File -Encoding ascii $status
}

function MinutesLeft() { return [math]::Round(($deadline - (Get-Date)).TotalMinutes, 0) }

# Waits for a box that is actually idle. Two independent signals, because free
# RAM reads healthy while a game holds the CPU, and the CPU reads idle while a
# game holds the RAM the model needs.
function WaitForQuiet($budgetMin) {
    Stamp "waiting-for-quiet" "cpu<=$QuietCpuPct% and free>=$MinFreeGB GB"
    $waited = 0
    while ($true) {
        $cpu = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 5 -MaxSamples 6).CounterSamples |
               Measure-Object -Property CookedValue -Average
        $cpuAvg = [math]::Round($cpu.Average, 1)
        $os = Get-CimInstance Win32_OperatingSystem
        $freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        if ($cpuAvg -le $QuietCpuPct -and $freeGB -ge $MinFreeGB) {
            Say ("  box is quiet: cpu {0}% free {1} GB after {2} min" -f $cpuAvg, $freeGB, $waited)
            return $true
        }
        if ($waited -ge $budgetMin) {
            Say ("  GAVE UP waiting for quiet after {0} min (cpu {1}% free {2} GB)" -f $waited, $cpuAvg, $freeGB)
            return $false
        }
        if (($waited % 15) -eq 0) {
            Say ("  waiting: cpu {0}% free {1} GB, {2} min left before the deadline" -f $cpuAvg, $freeGB, (MinutesLeft))
        }
        Start-Sleep -Seconds ($QuietPollMin * 60)
        $waited += $QuietPollMin
    }
}

function Invoke-Leg($name, $budgetMin, $argList) {
    $left = MinutesLeft
    if ($left -lt $budgetMin) {
        Say ("=== {0} SKIPPED -- needs ~{1} min, only {2} min before the deadline" -f $name, $budgetMin, $left)
        return 99
    }
    Stamp $name ("budget ~{0} min, {1} min before the deadline" -f $budgetMin, $left)
    Say ("=== {0} START (budget ~{1} min, {2} min left)" -f $name, $budgetMin, $left)
    & $ps -NoProfile -ExecutionPolicy Bypass @argList
    $code = $LASTEXITCODE   # captured immediately; anything else here masks it
    Say ("=== {0} END exit={1}" -f $name, $code)
    return $code
}

Say "================ M4.5 OVERNIGHT CHAIN"
Say ("deadline {0} (box goes down 07:30), started {1}" -f $deadline.ToString("s"), $now.ToString("s"))
Say ("binary under test: build\win\stoke.exe, branch " + (& git -C $root branch --show-current))

# The box was busy when this was queued. Give it most of the night to go quiet
# rather than a token wait: an idle box at 23:00 is still worth four legs.
$quiet = WaitForQuiet ([math]::Max(30, (MinutesLeft) - 420))
if (-not $quiet) {
    Say "the box never went quiet -- running the P-BAR legs anyway, since that runner"
    Say "has its own quiesce gate and will refuse to measure a busy box on its own terms."
}

$pre = 99
$ab  = 99
if ($quiet) {
    $pre = Invoke-Leg "leg1-preflight-ab" 25 @(
        "-File", "$root\tests\tooling\run_att_par_ab.ps1",
        "-Cycles", "1", "-NGen", "8", "-Threads", "8",
        "-Stamp", "2026-08-18-m45-att-par-ab-preflight")

    # The preflight's whole job is to fail here instead of two hours from now.
    $preRaw = "$gr\2026-08-18-m45-att-par-ab-preflight-raw.txt"
    $bad = $false
    if (-not (Test-Path $preRaw)) { $bad = $true; Say "  preflight produced no raw log" }
    else {
        $hits = @(Select-String -Path $preRaw -Pattern "PARSE-FAIL" -SimpleMatch)
        if ($hits.Count -gt 0) { $bad = $true; Say ("  preflight has {0} PARSE-FAIL line(s)" -f $hits.Count) }
    }

    if ($bad) {
        Say "leg2 SKIPPED -- the instrument could not parse its own output. Fix the runner,"
        Say "do not spend two hours producing numbers it cannot read."
    } else {
        $ab = Invoke-Leg "leg2-ab-4arm" 140 @(
            "-File", "$root\tests\tooling\run_att_par_ab.ps1",
            "-Cycles", "3", "-NGen", "128", "-Threads", "8", "-IncludeLong",
            "-Stamp", "2026-08-18-m45-att-par-ab")
    }
}

# > ANCHORS ARE PER-CONFIGURATION, and that is a deliberate departure from the
# resume note's single "-AnchorA 9.989". The anchor guard asks whether arm A
# reproduced ITS OWN quiesced rate, and arm A's rate differs by configuration:
# 9.989 at t=6 pinned, 10.835 at t=8 unpinned (2026-08-17 reports). Passing
# 9.989 to the unpinned leg would print a ~8% deviation that means nothing.
$p1 = Invoke-Leg "leg3-pbar-t6-pinned" 210 @(
    "-File", "$root\tests\tooling\run_overnight_pbar.ps1",
    "-Blocks", "3", "-Cycles", "3", "-NGen", "128", "-Seqs", "1",
    "-ArmBExe", "$root\build\win\stoke.exe",
    "-ArmALabel", "A pre-maddubs", "-ArmBLabel", "B HEAD + L4",
    "-AnchorA", "9.989",
    "-ThreadsA", "6", "-ThreadsB", "6", "-ThreadsL", "6", "-Pin",
    "-AlwaysRestoreISLC",
    "-Stamp", "2026-08-18-m45-pbar-l4-t6-pinned")

$p2 = Invoke-Leg "leg4-pbar-t8-unpinned" 180 @(
    "-File", "$root\tests\tooling\run_overnight_pbar.ps1",
    "-Blocks", "3", "-Cycles", "3", "-NGen", "128", "-Seqs", "1",
    "-ArmBExe", "$root\build\win\stoke.exe",
    "-ArmALabel", "A pre-maddubs", "-ArmBLabel", "B HEAD + L4",
    "-AnchorA", "10.835",
    "-ThreadsA", "8", "-ThreadsB", "8", "-ThreadsL", "8",
    "-SkipCorrectness", "-AlwaysRestoreISLC",
    "-Stamp", "2026-08-19-m45-pbar-l4-t8-unpinned")

Stamp "done" ("preflight={0} ab={1} pbar_pinned={2} pbar_unpinned={3}" -f $pre, $ab, $p1, $p2)
Say ("================ CHAIN DONE  preflight={0} ab={1} pbar_t6={2} pbar_t8={3}" -f $pre, $ab, $p1, $p2)
Say "reports are in docs\superpowers\gate-reports\ under their stamps."
Say "REMEMBER: delete the scheduled task stoke-m45-overnight -- a spent helper"
Say "outliving its job is a defect this project has recorded five times."
exit 0
