# tests/tooling/run_ref_probe.ps1 -- the pinned launcher for ref_forward.py's
# teacher-forced probe (M7 entry probe, M3 ruling §2.3 / §8.9).
#
# ⚠⚠ WHY A LAUNCHER EXISTS AT ALL, AND WHY IT IS NOT OPTIONAL.
# numpy here is scipy-openblas 0.3.30, built DYNAMIC_ARCH / NO_AFFINITY. With
# no thread count in the environment it sizes itself to every logical CPU and
# splits each dot product across them, so THE LAST BITS OF EVERY LOGIT depend
# on how many threads it picked -- and on this box an unpinned run also spreads
# across two different core types (0-11 are the six P-cores' siblings, 12-15
# are E-cores). A reference whose output is not reproducible cannot freeze a
# tolerance, and a tolerance frozen from an irreproducible measurement grades
# nothing.
#
# OpenBLAS reads its thread count at LOAD time -- before ref_forward.py's first
# line runs -- so this cannot be done from inside the tool. Affinity is set on
# THIS process and inherited by the child (the M4.5 ops lesson: children
# inherit, so pin the launcher, not the payload).
#
# ref_forward.py RECORDS what it observes into the probe JSON's "env" block, so
# the banked file says what produced it rather than leaving a later reader to
# assume it was pinned.
#
# Usage:
#   powershell -File tests/tooling/run_ref_probe.ps1 `
#       -Gguf models/Llama-3.2-3B-Instruct-Q4_K_M.gguf `
#       -IdsFile stream.ids -Out probe.json [-Mode batched] [-ProbeAt all]
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Gguf,
    [Parameter(Mandatory = $true)][string]$IdsFile,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$ProbeAt = "all",
    [ValidateSet("serial", "batched")][string]$Mode = "serial",
    [ValidateSet("closed", "accum")][string]$RopeTheta = "closed",
    [int]$Threads = 6,
    # 0xFFF = CPUs 0-11 = the six P-cores and their SMT siblings. E-cores
    # (12-15) are excluded: a thread that migrates there runs the same
    # arithmetic at a different speed, which corrupts the cost model, and with
    # a different thread count it can corrupt the last bits too.
    [string]$Affinity = "0xFFF"
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

foreach ($f in @($Gguf, $IdsFile)) {
    if (-not (Test-Path -LiteralPath $f)) { throw "run_ref_probe: no such file: $f" }
}
if ($Threads -lt 1) { throw "run_ref_probe: -Threads must be >= 1" }

$mask = [int64]$Affinity
(Get-Process -Id $PID).ProcessorAffinity = [System.IntPtr]$mask
$got = [int64](Get-Process -Id $PID).ProcessorAffinity
if ($got -ne $mask) {
    # A silently-ignored affinity request is the failure mode this whole file
    # exists to prevent, so it is a THROW and not a warning.
    throw "run_ref_probe: affinity did not take -- asked $Affinity, got 0x$([Convert]::ToString($got,16))"
}

$env:OPENBLAS_NUM_THREADS = "$Threads"
$env:OMP_NUM_THREADS = "$Threads"
$env:MKL_NUM_THREADS = "$Threads"

Write-Host ("run_ref_probe: affinity=0x{0:x} threads={1} mode={2} rope-theta={3}" `
            -f $got, $Threads, $Mode, $RopeTheta)

& python -u (Join-Path $here "ref_forward.py") $Gguf `
    --force-ids $IdsFile --probe-at $ProbeAt --probe-out $Out `
    --mode $Mode --rope-theta $RopeTheta --progress
$rc = $LASTEXITCODE
if ($rc -ne 0) { throw "run_ref_probe: ref_forward.py exited $rc" }

# ⚠ The verdict is taken from a line the run PRINTS, never from the exit code.
if (-not (Test-Path -LiteralPath $Out)) {
    throw "run_ref_probe: exited 0 but wrote no probe file: $Out"
}
Write-Host "run_ref_probe: OK -> $Out"
