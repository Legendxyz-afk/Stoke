# M4.5 -- does the affinity MASK SHAPE, not its size, decide stoke's rate?
#
# > WHY. In the 2026-08-17 P-BAR run, block 2 measured BOTH stoke arms at
# roughly 0.55-0.64x of blocks 1 and 3 -- A 10.03 -> 5.39, B 15.44 -> 9.98 --
# while llama, interleaved between them minutes apart, was UNAFFECTED and in
# fact at its best of the run (18.66 / 18.75 / 18.73). Within block 2 the
# stoke values agree to 0.2%, so it was a stable state and not jitter. A
# box-level cause (thermal, contention, page cache) cannot hit one engine and
# spare the other; the cause has to be something stoke does and llama does not.
#
# > THE HYPOTHESIS. `src/pool.c` creates workers with bare `pthread_create` and
# sets NO affinity, so placement is the scheduler's choice. The standing mask
# 0xFFF is not "the 6 P-cores" -- it is their TWELVE hyperthread siblings.
# Nothing stops Windows putting 6 threads on 3 physical cores, two to a core,
# leaving three cores idle. That costs very close to the factor observed.
#
# > THE TEST. Same engine, same thread count, three masks that differ only in
# SHAPE:
#     0xFFF   12 siblings of 6 P-cores   -- the standing mask, stacking possible
#     0x555   CPUs 0,2,4,6,8,10          -- one sibling per P-core, stacking IMPOSSIBLE
#     0xFFFF  all 16 logical             -- unpinned, E-cores available
# If the hypothesis holds, 0x555 is at least as fast as 0xFFF and far steadier,
# and llama moves much less than stoke across all three.
#
# > THE ASSUMPTION, STATED: that logical CPUs are numbered P-cores first with
# siblings adjacent, so the even CPUs 0..10 are one-per-physical-P-core. That is
# the standard Windows enumeration for Intel hybrid parts and it is exactly what
# this probe puts to the test -- if 0x555 behaved like three cores rather than
# six, it would come out at roughly half of 0xFFF, not level with it.

param(
    [int]$Reps    = 3,
    [int]$NGen    = 128,
    [int]$Threads = 6,
    [string]$Stamp = "2026-08-17-m45-affinity-probe"
)

$ErrorActionPreference = 'Continue'

$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp-raw.txt"
$rep  = "$gr\$Stamp-report.md"
$bank = "$gr\$Stamp-bank.jsonl"
$tmpE = "$env:TEMP\stoke-aff-err.txt"
$tmpO = "$env:TEMP\stoke-aff-out.txt"

$Q4     = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$STOKE  = "$root\build\win\stoke.exe"
$LB     = "$root\bench\vendor\llama-b10037\llama-bench.exe"
$PROMPT = "$root\tests\data\parity\p00.txt"

$MASKS = [ordered]@{
    "0xFFF"  = 0xFFF     # 12 siblings of 6 P-cores -- stacking possible
    "0x555"  = 0x555     # one sibling per P-core   -- stacking impossible
    "0xFFFF" = 0xFFFF    # all 16 logical           -- unpinned
}

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Output $line
}
function Bank($o) { ($o | ConvertTo-Json -Compress) | Out-File -Append -Encoding utf8 $bank }
function Median($xs) {
    $s = @($xs | Sort-Object); $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int](($n - 1) / 2)] }
    return [math]::Round(($s[$n / 2 - 1] + $s[$n / 2]) / 2.0, 3)
}
function RunCapture($exe, $argList) {
    Start-Process -FilePath $exe -ArgumentList $argList `
                  -RedirectStandardError $tmpE -RedirectStandardOutput $tmpO -NoNewWindow -Wait
}
function StokeRate() {
    RunCapture $STOKE @("batch", ('"' + $Q4 + '"'), "--raw", "-n", "$NGen",
                        "-f", ('"' + $PROMPT + '"'), "--seqs", "1",
                        "--threads", "$Threads", "--kv", "f32", "--timing")
    $t = (Get-Content $tmpE -Raw)
    if (-not $t) { return $null }
    $m = [regex]::Match($t, '\(([\d.]+)\s+tok/s\s+aggregate\)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}
function LlamaRate() {
    RunCapture $LB @("-m", ('"' + $Q4 + '"'), "-p", "0", "-n", "$NGen", "-r", "3", "-t", "$Threads")
    $t = (Get-Content $tmpO -Raw)
    if (-not $t) { return $null }
    $m = [regex]::Match($t, 'tg\d+\s*\|\s*([\d.]+)')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return $null
}

Say "START  reps=$Reps n=$NGen threads=$Threads masks=$($MASKS.Keys -join ',')"

# Warm, and check it -- a cold model reads the disk and would look exactly like
# the effect being hunted.
for ($w = 1; $w -le 3; $w++) {
    (Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF
    $null = StokeRate
    $sb = [math]::Round((Get-Counter '\Memory\Standby Cache Normal Priority Bytes').CounterSamples[0].CookedValue / 1GB, 2)
    if ($sb -ge 2.0) { break }
}
Say "  warm done, standby $sb GB"

for ($r = 1; $r -le $Reps; $r++) {
    # Rotate mask order per rep so position cannot masquerade as mask effect.
    $keys = @($MASKS.Keys)
    $order = @()
    for ($i = 0; $i -lt $keys.Count; $i++) { $order += $keys[($i + $r - 1) % $keys.Count] }
    foreach ($mk in $order) {
        (Get-Process -Id $PID).ProcessorAffinity = [IntPtr]$MASKS[$mk]
        foreach ($eng in @("S", "L")) {
            $v = if ($eng -eq "S") { StokeRate } else { LlamaRate }
            Bank @{ rep = $r; mask = $mk; engine = $eng; tok_s = $v; utc = (Get-Date -Format s) }
            Say ("  r{0} {1,-6} {2}  {3}" -f $r, $mk, $eng, $(if ($null -eq $v) { "PARSE-FAIL" } else { $v }))
        }
    }
}
(Get-Process -Id $PID).ProcessorAffinity = [IntPtr]0xFFFF

$rows = @()
foreach ($l in (Get-Content $bank)) { if ($l.Trim()) { try { $rows += ($l | ConvertFrom-Json) } catch {} } }
$rows = @($rows | Where-Object { $null -ne $_.tok_s })

$lines = @()
$lines += "# M4.5 -- affinity mask SHAPE probe"
$lines += ""
$lines += "Does the shape of the affinity mask, not its size, decide stoke's rate? $Reps reps, mask order rotating, $Threads threads, n=$NGen, N=1."
$lines += ""
$lines += "| mask | meaning | stoke | llama | stoke/llama | stoke spread |"
$lines += "|---|---|---|---|---|---|"
$meaning = @{ "0xFFF" = "12 siblings of 6 P-cores (standing mask)"; "0x555" = "one sibling per P-core"; "0xFFFF" = "all 16 logical (unpinned)" }
foreach ($mk in $MASKS.Keys) {
    $sv = @($rows | Where-Object { $_.mask -eq $mk -and $_.engine -eq "S" } | ForEach-Object { $_.tok_s })
    $lv = @($rows | Where-Object { $_.mask -eq $mk -and $_.engine -eq "L" } | ForEach-Object { $_.tok_s })
    $s = Median $sv; $l = Median $lv
    $sl = if ($s -and $l) { "{0:n3}x" -f ($s / $l) } else { "n/a" }
    $sp = if ($sv.Count -gt 1) { "{0:n1}%" -f (100.0 * (($sv | Measure-Object -Maximum).Maximum - ($sv | Measure-Object -Minimum).Minimum) / ($sv | Measure-Object -Minimum).Minimum) } else { "n/a" }
    $lines += ("| ``$mk`` | {0} | {1} | {2} | {3} | {4} |" -f $meaning[$mk], $s, $l, $sl, $sp)
}
$lines += ""
$lines += "If ``0x555`` matches or beats ``0xFFF`` while llama barely moves, then stoke's exposure is hyperthread STACKING under a mask that permits it, and the engine's missing thread affinity -- ``src/pool.c`` creates workers with bare ``pthread_create`` -- is a real defect rather than a measurement artefact."
$lines += ""
$lines += "Bank: ``$Stamp-bank.jsonl``. Raw log: ``$Stamp-raw.txt``."
($lines -join "`r`n") | Out-File -Encoding utf8 $rep
Say "REPORT -> $rep"
Say "DONE"
