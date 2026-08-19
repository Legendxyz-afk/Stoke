# M5 entry -- the unattended chain: wait, rebuild, gate, then measure.
#
# The M4.5 close ran its whole endgame as one detached chain overnight and that
# is the pattern here, for the same reason: each stage costs tens of minutes,
# each depends on the one before, and a session boundary must not be able to
# eat the sequence (session-tracked background tasks get reaped -- the standing
# ops lesson).
#
# STAGES, and why the order is what it is:
#   1  WAIT for any in-flight verify driver to exit. The binary under gate must
#      not be swapped mid-run: a gate that grades a stale-or-swapped artifact is
#      worse than no gate, it manufactures confidence.
#   2  REBUILD, through bash. mingw32-make pins SHELL := sh and the Makefile's
#      recipes are POSIX; invoked from PowerShell, `mkdir -p` resolves to cmd's
#      mkdir and the build dies at the first recipe. Measured, not guessed.
#   3  ENTRY GATE (correctness) -- must be green before anything is measured.
#   4  A/B (speed) -- LAST, and only if the gate passed. A speed number for a
#      binary whose arithmetic has not been checked is not a result.
#
# The A/B is skipped on a red gate rather than run anyway: the whole point of
# the ordering is that a DIFFER makes the speed question moot.

param(
    [switch]$SkipAb,
    [int]$AbCycles = 3,
    [string]$Stamp = "2026-08-19-m5-chain"
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$log  = "$root\docs\superpowers\gate-reports\$Stamp.txt"
$bash = "M:\Git\usr\bin\bash.exe"

function Say($m) {
    $line = "{0}  [chain] {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding ascii $log
    Write-Output $line
}

Set-Location $root
Say "START pid=$PID"

# ---- 1. wait for any verify driver still holding build\win\stoke.exe --------
# Matched on the SCRIPT NAME in the command line, not on the process name:
# every stage here is powershell.exe, and killing or waiting on "powershell"
# would include this chain itself.
$waited = 0
while ($true) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
              Where-Object { $_.CommandLine -like '*run_m5_*_verify.ps1*' -and
                             $_.ProcessId -ne $PID })
    if ($busy.Count -eq 0) { break }
    if ($waited -eq 0) { Say "waiting for $($busy.Count) verify driver(s) to finish" }
    Start-Sleep -Seconds 30
    $waited += 30
    if ($waited -gt 4 * 3600) { Say "GAVE UP waiting after 4 h"; exit 1 }
}
if ($waited -gt 0) { Say "clear after $([int]($waited/60)) min" }

# ---- 2. rebuild ------------------------------------------------------------
Say "rebuilding"
# -lc, NOT -c. A non-login bash here has no /mingw64 or ucrt64 on PATH, so
# `mingw32-make` is simply not found and the chain dies at the rebuild with a
# 127 that reads like a broken script. Verified both ways before this ran.
& $bash -lc "cd '/m/New Drive/Desktop/Programming/stoke' && mingw32-make 2>&1 | tail -3"
if ($LASTEXITCODE -ne 0) { Say "BUILD FAILED rc=$LASTEXITCODE -- stopping"; exit 1 }
Say "build ok"

# ---- 3. correctness ---------------------------------------------------------
Say "entry verify starting"
& powershell -NoProfile -ExecutionPolicy Bypass -File "tests\tooling\run_m5_entry_verify.ps1" -NGen 24
$gate = $LASTEXITCODE
Say "entry verify rc=$gate"
if ($gate -ne 0) {
    Say "GATE RED -- A/B deliberately NOT run: a speed number for unchecked arithmetic is not a result"
    exit $gate
}

# ---- 4. speed ---------------------------------------------------------------
if ($SkipAb) { Say "A/B skipped by flag"; Say "DONE"; exit 0 }
Say "A/B starting ($AbCycles cycles)"
& powershell -NoProfile -ExecutionPolicy Bypass -File "tests\tooling\run_m5_e2_ab.ps1" -Cycles $AbCycles
Say "A/B rc=$LASTEXITCODE"
Say "DONE"
