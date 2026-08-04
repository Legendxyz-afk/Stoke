# Writes the q8_0 leg's "done" record when its python driver exits.
#
# WHY THIS EXISTS. Leg 1 was launched by run_m3_gate_detached.cmd, which runs
# both legs sequentially. That launcher's cmd.exe was killed on 2026-08-04 to
# cancel its leg-2 invocation (leg 2 was going to be started another way), and
# killing it took with it the ONLY thing that would ever write leg 1's
# completion record -- the `echo {"phase":"done"...}` after the python call.
# The python driver itself survived, because killing a parent does not kill a
# child on Windows, and has been running fine ever since.
#
# So the leg finishes and NOTHING notices: run_m3_gate_chain.cmd polls
# m3-gate-status-q8_0.json for "phase":"done", that string never appears, and
# the chain silently times out at ~19:46 having never started leg 2.
#
# ▶ WHY A HANDLE AND NOT A POLL. The exit CODE is the leg's verdict (0 = GATE
# PASS, 1 = GATE FAIL) and it is destroyed the moment the process is reaped.
# A script that polls Get-Process until the pid disappears can only ever
# report "it ended" -- it cannot say whether the gate passed, and inventing a
# 0 there would fabricate a PASS. GetProcessById opens a handle NOW, while the
# process is alive; that handle keeps the exit code readable after exit, so
# WaitForExit/ExitCode reports the real verdict.
#
# ▶ WHY IT DOES NOT START LEG 2 ITSELF. run_m3_gate_chain.cmd is already
# running and already waits on exactly this file. Two things racing to launch
# leg 2 is how you get two servers on port 8090 and a leg quietly answered by
# the wrong model. This writes the record; the chain does the launching.

param(
    [int]$Pid1 = 10668,
    [string]$Cache = "M:\New Drive\Desktop\Programming\stoke\tests\tooling\.gate_cache"
)

$ErrorActionPreference = 'Stop'
$status = Join-Path $Cache 'm3-gate-status-q8_0.json'
$log    = Join-Path $Cache 'm3-gate-watch.log'

function Note($m) { Add-Content -Path $log -Value ("{0}  {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m) -Encoding utf8 }

try {
    $p = [System.Diagnostics.Process]::GetProcessById($Pid1)
} catch {
    # Already gone -- the exit code died with it. Record that honestly rather
    # than guessing, and leave the chain waiting: a human decides what a leg
    # with no verdict means.
    Note "pid $Pid1 already exited before the watcher attached -- EXIT CODE LOST, no done record written"
    exit 2
}

if ($p.ProcessName -ne 'python') {
    Note "pid $Pid1 is '$($p.ProcessName)', not python -- refusing to watch a recycled pid"
    exit 2
}

# ▶ TOUCHING .Handle IS THE WHOLE POINT, NOT A NO-OP. GetProcessById does not
# open the process handle eagerly; for a process this session did not start,
# .NET caches nothing, and by the time WaitForExit returns there is no handle
# left to read the exit status from -- .ExitCode comes back EMPTY. That is not
# hypothetical: the 2026-08-04 q8_0 run recorded `rc=` and wrote the malformed
# JSON `{"rc":,...}`, which json.load rejects, so the status line silently
# dropped the leg instead of showing its verdict. Dereferencing .Handle forces
# the handle open NOW, while the process still lives, and keeps it valid past
# exit. A watcher that cannot report the verdict is worse than no watcher: it
# looks like it worked.
$null = $p.Handle

Note "attached to python pid $Pid1 (started $($p.StartTime)), waiting for exit"
$p.WaitForExit()
$rc = $p.ExitCode
if ($null -eq $rc -or "$rc" -eq '') {
    Note "python pid $Pid1 exited but EXIT CODE UNREADABLE -- no done record written; read the GATE PASS/FAIL line from the console log instead"
    exit 3
}
Note "python pid $Pid1 exited rc=$rc"

$fin = Get-Date -Format 'dd-MM-yyyy HH:mm:ss'
$json = '{"leg":"q8_0","phase":"done","rc":' + $rc + ',"finished":"' + $fin + '","recorded_by":"watch_m3_leg1.ps1"}'
# NO BOM. Windows PowerShell 5.1's `-Encoding utf8` writes one, and the other
# writer of this file (run_m3_gate_leg.cmd's `echo >`) does not -- so the
# consumers were only ever tested BOM-free. A BOM read back under cp1252
# arrives as stray bytes before the '{' and json.load raises, which drops the
# leg from the status line entirely: the verdict this script exists to record
# becomes invisible. Write the bytes explicitly rather than trusting a default.
[System.IO.File]::WriteAllText($status, $json, (New-Object System.Text.UTF8Encoding($false)))
Note "wrote done record rc=$rc -- chain waiter may now start leg 2"
exit 0
