# M4.5 -- is stoke's collapse RESIDENCY? Measure the mechanism, not the rate.
#
# ▶ WHERE THIS CAME FROM. The affinity probe killed its own hypothesis: 0x555
# forbids hyperthread stacking and collapsed anyway (9.058, 9.53), while rep 2
# was healthy on ALL THREE masks (15.35-15.45). So the variable is time, not
# placement. llama stayed inside +/-5% throughout.
#
# ▶ WHAT THE RATES IMPLY, since the model is streamed once per token and its
# size is known (2.33 GiB of weights):
#     15.4 tok/s -> ~35.8 GB/s   plausible RAM, near this box's 44.8 GB/s ceiling
#      9.5 tok/s -> ~22.1 GB/s   half of RAM speed
#      1.78 tok/s ->  ~4.2 GB/s  NOT memory. That is a disk.
# So the hypothesis is that the model is not resident and decode is reading it
# from storage, every token.
#
# ▶ THE INSTRUMENT IS THE RULING'S OWN. `Page Reads/sec` is what the ruling
# already specifies for D's residency gate ("< 5 sustained"). A HARD fault is a
# disk read; a soft fault is not. Counting them during decode separates
# "stoke is slow" from "the box is feeding stoke from a disk".
#
# ▶ WHY STOKE AND NOT LLAMA -- the asymmetry to test. `llama-bench -r 3` loads
# the model ONCE and runs three repetitions inside one process, so it pays
# first-touch once and is measured warm. stoke is launched fresh for every
# single measurement and pays process-fresh costs every time. If residency is
# the mechanism, that difference alone explains why one engine looks stable and
# the other does not -- and it means the two engines have never been measured on
# equal terms.

param(
    [int]$Runs   = 6,
    [int]$NGen   = 128,
    [int]$Threads = 6,
    [string]$Stamp = "2026-08-17-m45-residency-probe"
)

$ErrorActionPreference = 'Continue'

$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$tmpE = "$env:TEMP\stoke-res-err.txt"
$tmpO = "$env:TEMP\stoke-res-out.txt"

$Q4     = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE  = "$root\build\win\stoke.exe"
$PROMPT = "$root\tests\data\parity\p00.txt"

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Mem() {
    $c = Get-Counter '\Memory\Available MBytes', '\Memory\Standby Cache Normal Priority Bytes', '\Memory\Cache Bytes'
    [pscustomobject]@{
        AvailGB   = [math]::Round($c.CounterSamples[0].CookedValue / 1024, 2)
        StandbyGB = [math]::Round($c.CounterSamples[1].CookedValue / 1GB, 2)
        CacheGB   = [math]::Round($c.CounterSamples[2].CookedValue / 1GB, 2)
    }
}

Say "START  runs=$Runs n=$NGen threads=$Threads"
Say "  model: $([math]::Round((Get-Item $Q4).Length/1GB,2)) GB on drive $($Q4.Substring(0,1)):"
$d = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$($Q4.Substring(0,2))'"
Say "  drive type=$($d.DriveType) (3=fixed, 2=removable, 4=network)  free=$([math]::Round($d.FreeSpace/1GB,1)) GB"

$results = @()
for ($i = 1; $i -le $Runs; $i++) {
    $m0 = Mem
    # Sample Page Reads/sec THROUGHOUT the run, in a job, so a burst confined to
    # first-touch is distinguishable from sustained disk reading.
    $job = Start-Job -ScriptBlock {
        $s = @()
        for ($k = 0; $k -lt 400; $k++) {
            $s += (Get-Counter '\Memory\Page Reads/sec').CounterSamples[0].CookedValue
            Start-Sleep -Milliseconds 250
        }
        $s
    }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Start-Process -FilePath $STOKE -ArgumentList @("batch", ('"' + $Q4 + '"'), "--raw",
        "-n", "$NGen", "-f", ('"' + $PROMPT + '"'), "--seqs", "1",
        "--threads", "$Threads", "--kv", "f32", "--timing") `
        -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO -NoNewWindow -Wait
    $sw.Stop()
    Stop-Job $job -ErrorAction SilentlyContinue
    $samples = @(Receive-Job $job -ErrorAction SilentlyContinue)
    Remove-Job $job -Force -ErrorAction SilentlyContinue

    $t = (Get-Content $tmpE -Raw)
    $rate = $null
    $mm = [regex]::Match($t, '\(([\d.]+)\s+tok/s\s+aggregate\)')
    if ($mm.Success) { $rate = [double]$mm.Groups[1].Value }
    $m1 = Mem
    $pr = if ($samples.Count -gt 0) { [math]::Round(($samples | Measure-Object -Average).Average, 1) } else { $null }
    $prMax = if ($samples.Count -gt 0) { [math]::Round(($samples | Measure-Object -Maximum).Maximum, 0) } else { $null }

    $results += [pscustomobject]@{
        Run = $i; Rate = $rate; WallS = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        PageReadsAvg = $pr; PageReadsMax = $prMax
        AvailGB0 = $m0.AvailGB; StandbyGB0 = $m0.StandbyGB
        AvailGB1 = $m1.AvailGB; StandbyGB1 = $m1.StandbyGB
    }
    Say ("  run {0}: {1} tok/s  wall {2}s  pagereads avg {3} max {4}  standby {5} -> {6} GB  avail {7} -> {8} GB" -f `
         $i, $rate, [math]::Round($sw.Elapsed.TotalSeconds,1), $pr, $prMax, $m0.StandbyGB, $m1.StandbyGB, $m0.AvailGB, $m1.AvailGB)
}

$lines = @()
$lines += "# M4.5 -- residency probe: is the collapse a disk?"
$lines += ""
$lines += "$Runs identical stoke runs, n=$NGen, $Threads threads, back to back. ``Page Reads/sec`` sampled at 4 Hz across each run -- the same counter the ruling specifies for D's residency gate."
$lines += ""
$lines += "| run | tok/s | wall s | page reads/s avg | max | standby GB before -> after | avail GB before -> after |"
$lines += "|---|---|---|---|---|---|---|"
foreach ($r in $results) {
    $lines += ("| {0} | {1} | {2} | {3} | {4} | {5} -> {6} | {7} -> {8} |" -f `
        $r.Run, $r.Rate, $r.WallS, $r.PageReadsAvg, $r.PageReadsMax, $r.StandbyGB0, $r.StandbyGB1, $r.AvailGB0, $r.AvailGB1)
}
$lines += ""
$lines += "**Reading it.** A slow run with page reads near zero is NOT a residency problem and the disk hypothesis dies. A slow run with sustained page reads in the thousands is the model being streamed off storage every token, and the fix is residency, not kernels. The ruling's own threshold for 'resident' is < 5 sustained."
$lines += ""
$lines += "Raw log: ``$Stamp-raw.txt``."
($lines -join "`r`n") | Out-File -Encoding utf8 $rep
Say "REPORT -> $rep"
Say "DONE"
