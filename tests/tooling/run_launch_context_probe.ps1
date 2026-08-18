# Does the LAUNCH CONTEXT change the measurement? A controlled answer.
#
# Overnight, the pre-EcoQoS-fix archive read 0.922 tok/s under Task Scheduler
# where its anchor is 10.835, while arm B and llama -- measured in the same
# minutes, same box -- were healthy. A depressed box cannot do that to one arm
# only, so the suspect is the LAUNCH: a Task Scheduler process has no
# interactive window and Windows may class it as BACKGROUND, which throttles
# frequency and permits parking its threads on the E-cores. The archive
# predates 76dee7f and never opts out; arm B opts out per worker; llama spins
# rather than sleeping and never looks like background work.
#
# This script is the scheduled half of the experiment. Run the SAME two configs
# interactively, then run this via a scheduled task within a few minutes, and
# compare. Same binary, same flags, same box, same hour -- only the launch
# differs.
#
# Appends one line per config so the two halves land in one file in order.

param(
    [string]$Tag = "scheduled"
)

$root = "M:\New Drive\Desktop\Programming\stoke"
$out  = "$root\docs\superpowers\gate-reports\2026-08-19-m45-launch-context-probe.txt"
$exe  = "$root\build\stoke-m45-c-head-bc9dd3d6.exe"   # pre-76dee7f, no QoS opt-out
$mdl  = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$prm  = "$root\tests\data\parity\p00.txt"
$tmpE = "$env:TEMP\lcprobe-e.txt"
$tmpO = "$env:TEMP\lcprobe-o.txt"

foreach ($cfg in @(@{ t = 8; pin = $false }, @{ t = 6; pin = $true })) {
    # > THE PATHS ARE QUOTED, AND THAT IS A BUG FIX. Start-Process joins an
    # ArgumentList array with spaces and quotes NOTHING, so "M:\New Drive\..."
    # arrives as two arguments and the binary exits on a usage error in under a
    # second -- which this script then logged four times as PARSE-FAIL, reading
    # like a broken engine rather than a broken caller. run_overnight_pbar.ps1
    # has quoted its paths all along; this did not.
    $p = Start-Process -FilePath $exe -ArgumentList @(
            "batch", ('"' + $mdl + '"'), "--raw", "-n", "128", "-f", ('"' + $prm + '"'),
            "--seqs", "1", "--threads", "$($cfg.t)", "--kv", "f32", "--timing") `
        -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO -NoNewWindow -PassThru
    if ($cfg.pin) {
        Start-Sleep -Milliseconds 200
        try { $p.ProcessorAffinity = [IntPtr]0xFFF } catch {}
    }
    $p.WaitForExit()
    # A unit that exits in under a second did not measure anything, and saying
    # PARSE-FAIL for it hides the reason. Report the two failures differently.
    $elapsed = [math]::Round(($p.ExitTime - $p.StartTime).TotalSeconds, 1)
    $m = [regex]::Match((Get-Content $tmpE -Raw), '\(([\d.]+)\s+tok/s\s+aggregate\)')
    $rate = if ($m.Success) { $m.Groups[1].Value }
            elseif ($elapsed -lt 2.0) { "DID-NOT-RUN (exit $($p.ExitCode) in ${elapsed}s)" }
            else { "PARSE-FAIL" }
    ("{0}  {1,-11} t={2} pin={3,-5}  {4} tok/s" -f
        (Get-Date -Format s), $Tag, $cfg.t, $cfg.pin, $rate) |
        Out-File -Append -Encoding ascii $out
}
