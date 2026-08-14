# M4.5 -- the two queued measurements, in the order they should run, so a free
# box needs ONE start command and no supervision.
#
#   leg 1  the ten owed ESCALATE ladders + three controls   (run_c_ladder.ps1)
#   leg 2  the NULL-1 Q4_K_M base rate, ~2.5-3 h            (run_null1_baserate.ps1)
#
# Leg 2 runs even if leg 1 fails: they are independent measurements and the
# base rate is the one that unblocks the ladder. Each leg keeps its own raw log
# and its own artifacts; this script only sequences them and records the exit
# codes in one place.
#
# Both legs are deterministic and resumable. Contention changes how long they
# take and nothing else -- but nothing TIMED is valid unless the box is quiet,
# and neither leg produces a timing number, by design.

$ErrorActionPreference = 'Continue'   # a failed leg must not skip the next one
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFF   # P-cores, standing call

$root = "M:\New Drive\Desktop\Programming\stoke"
$log  = "$root\docs\superpowers\gate-reports\2026-08-14-m45-queue.txt"
$ps   = "powershell.exe"

function Invoke-Leg($name, $script) {
    "=== {0} {1} START" -f (Get-Date -Format s), $name |
        Out-File -Append -Encoding utf8 $log
    & $ps -NoProfile -ExecutionPolicy Bypass -File "$root\tests\tooling\$script"
    $code = $LASTEXITCODE   # captured immediately; an echo here would mask it
    "=== {0} {1} END exit={2}" -f (Get-Date -Format s), $name, $code |
        Out-File -Append -Encoding utf8 $log
    return $code
}

$a = Invoke-Leg "leg1-ladder"   "run_c_ladder.ps1"
$b = Invoke-Leg "leg2-baserate" "run_null1_baserate.ps1"

"=== {0} QUEUE DONE ladder={1} baserate={2}" -f (Get-Date -Format s), $a, $b |
    Out-File -Append -Encoding utf8 $log

if ($a -ne 0) { exit $a }
exit $b
