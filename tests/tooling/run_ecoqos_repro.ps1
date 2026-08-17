# M4.5 -- the DETERMINISTIC reproducer for the intermittent ~6-11x collapse,
# and the regression test for its fix.
#
# ▶ THE BUG. stoke would intermittently run at 1.4-2.5 tok/s instead of ~16,
# for stretches of tens of minutes, while llama-bench measured minutes apart in
# the same interleave was unaffected. It cost a P-BAR block (2026-08-17 block 2)
# and it is the likeliest explanation for the 2026-08-15 ladder's depressed
# arm A (8.262 where the same binary and config later gave 10.03).
#
# ▶ WHAT IT IS. Windows EcoQoS. The scheduler throttles work it classifies as
# background by placing it on the EFFICIENCY cores, and it applies that
# classification heuristically -- which is exactly why the state came and went
# with no correlate in memory pressure, disk, affinity mask or thermals. A
# forward pass issues ~250 matvec dispatches per token and each helper slept on
# a condvar between them, which is what background work looks like.
#
# FOUR HYPOTHESES DIED BEFORE THIS ONE, each to its own control: hyperthread
# stacking (0x555 forbids it and collapsed anyway), disk residency (hard faults
# ~0 during a collapse), working-set trimming and memory compression (the
# working set holds FLAT at 2430 MB through a 2.4 tok/s run; compression at 0),
# and static partitioning (pool.c has chunked dynamically via an atomic counter
# since K5). The survivor came from testing affinity masks WHILE the box was
# sick rather than waiting for a state that could not be summoned -- pinning to
# the P-cores recovered 2.4 -> 15.8 on the spot.
#
# ▶ WHY THIS SCRIPT EXISTS. The wild state cannot be summoned, so it could not
# be used to verify a fix. Process-level EcoQoS induces the SAME state on
# demand, and a PER-THREAD opt-out overrides a process-level setting -- so this
# both proves the diagnosis and tests the fix:
#     throttled + STOKE_NO_QOS=1  -> threads stay on E-cores  -> SICK   (~1.4)
#     throttled + opt-out active  -> threads return to P      -> ~16, baseline
# If both arms come back sick, the opt-out in pool.c does not work and must not
# be trusted.
#
# RESULT WHEN LANDED (2026-08-17, two passes): baseline 16.209/16.104 ·
# throttled+disabled 1.441/1.400 at 102 CPU-seconds · throttled+opt-out
# 16.100/15.849 at 27-28 CPU-seconds. An 11.5x recovery. The CPU-seconds are
# the independent signal: the same work costs ~4x more processor-time on the
# efficiency cores, so a sick run reads high there AND finishes later.

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class Qos {
    [StructLayout(LayoutKind.Sequential)]
    public struct PPT { public uint Version; public uint ControlMask; public uint StateMask; }
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetProcessInformation(IntPtr h, int cls, ref PPT info, uint size);
    public static bool Throttle(IntPtr h) {
        PPT p = new PPT();
        p.Version = 1;            // PROCESS_POWER_THROTTLING_CURRENT_VERSION
        p.ControlMask = 0x1;      // EXECUTION_SPEED
        p.StateMask = 0x1;        // ON == EcoQoS, prefer the efficiency cores
        // ProcessPowerThrottling == 4 in PROCESS_INFORMATION_CLASS
        return SetProcessInformation(h, 4, ref p, (uint)Marshal.SizeOf(p));
    }
}
"@ -ErrorAction Stop

$root = "M:\New Drive\Desktop\Programming\stoke"
Set-Location $root
$Q4  = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$err = "$env:TEMP\eq-e.txt"
$out = "$env:TEMP\eq-o.txt"

function Trial($label, $noqos, $throttle) {
    [Environment]::SetEnvironmentVariable('STOKE_NO_QOS', $noqos)
    $p = Start-Process "$root\build\win\stoke.exe" -ArgumentList @("batch", ('"' + $Q4 + '"'),
            "--raw", "-n", "64", "-f", ('"' + $root + '\tests\data\parity\p00.txt"'),
            "--seqs", "1", "--threads", "6", "--kv", "f32", "--timing") `
        -RedirectStandardError $err -RedirectStandardOutput $out -NoNewWindow -PassThru
    # Applied just after launch: model load gives ~0.5 s of headroom before the
    # first token, and the throttle takes effect for the whole decode.
    $applied = $false
    if ($throttle) { try { $applied = [Qos]::Throttle($p.Handle) } catch { $applied = $false } }
    $cpu = 0
    while (-not $p.HasExited) {
        try { $p.Refresh(); $cpu = $p.TotalProcessorTime.TotalSeconds } catch {}
        Start-Sleep -Milliseconds 300
    }
    $m = [regex]::Match((Get-Content $err -Raw), '\(([\d.]+)\s+tok/s\s+aggregate\)')
    $rate = if ($m.Success) { [double]$m.Groups[1].Value } else { $null }
    # Write-Host, not a bare string: in PowerShell EVERY uncaptured expression
    # joins the return pipeline, so printing the line normally made this
    # function return [line, rate] and the caller silently averaged strings.
    Write-Host ("{0,-34} {1,7} tok/s   cpu {2,6}s   throttle_applied={3}" -f $label,
        $(if ($null -eq $rate) { "FAIL" } else { $rate }), [math]::Round($cpu, 1), $applied)
    return $rate
}

$sick = @(); $fixed = @()
foreach ($pass in 1, 2) {
    "--- pass $pass ---"
    $null    = Trial "baseline, no throttle, opt-out ON"  $null $false
    $sick   += Trial "THROTTLED, opt-out DISABLED (old)" '1'   $true
    $fixed  += Trial "THROTTLED, opt-out ON (the fix)"   $null $true
}
[Environment]::SetEnvironmentVariable('STOKE_NO_QOS', $null)

$ms = ($sick  | Measure-Object -Average).Average
$mf = ($fixed | Measure-Object -Average).Average
""
# ▶ A GATE THAT CANNOT PASS BY COMPUTING NOTHING. The first version wrapped the
# comparison in `if ($ms -and $mf)` and printed OK when both were null -- which
# is exactly what happened, because the rates never reached these variables. A
# missing measurement is a FAILURE, not a pass.
if ($null -eq $ms -or $null -eq $mf -or $sick.Count -eq 0 -or $fixed.Count -eq 0) {
    Write-Host "FAILED: no rates parsed -- the reproducer measured nothing, which is not a pass."
    exit 1
}
Write-Host ("recovery: {0:n2} -> {1:n2} tok/s = {2:n1}x" -f $ms, $mf, ($mf / $ms))
if (($mf / $ms) -lt 2.0) {
    Write-Host "REGRESSION: the per-thread QoS opt-out is no longer rescuing the throttled case."
    exit 1
}
Write-Host "OK"
