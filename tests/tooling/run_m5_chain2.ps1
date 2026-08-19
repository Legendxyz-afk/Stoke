# M5 entry -- chain 2: the evidence the first chain does not produce.
#
# Chain 1 is wait -> rebuild -> entry gate -> decode A/B. Two gaps remain after
# it, and both are named in the ledger's NEXT list:
#
#   1  E1'S BATCH PATH HAS NO REAL-MODEL EVIDENCE. Every byte-compare gate here
#      drives `stoke run`, which is n_seq=1 -- so the flattened (sequence, unit)
#      dispatch that E1 exists for is exercised only by the C tests at fixture
#      scale. m4_batch_identity.py is the instrument that closes it: it compares
#      each sequence's block from `stoke batch --seqs N` against that prompt's
#      solo `stoke run` output, byte for byte, over N in {1,2,4,8,16}.
#
#   2  E3 HAS NO SPEED NUMBER AT ALL. The decode A/B reads the decode aggregate
#      and E3 changes nothing about decode; its whole effect is in prefill.
#
# CORRECTNESS BEFORE SPEED, again: if batch identity fails, the prefill A/B is
# skipped, because a number from a binary whose batching is wrong is not a
# result. Same rule chain 1 applies to the gate.

param(
    [int]$AbCycles = 3,
    [string]$Stamp = "2026-08-19-m5-chain2"
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$log  = "$root\docs\superpowers\gate-reports\$Stamp.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

function Say($m) {
    $line = "{0}  [chain2] {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding ascii $log
    Write-Output $line
}

Set-Location $root
Say "START pid=$PID"

# ---- wait for chain 1 (and any verify driver) to be completely done ---------
# Matched on script NAME, excluding this process: every stage is powershell.exe.
$waited = 0
while ($true) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
              Where-Object { ($_.CommandLine -like '*run_m5_chain.ps1*' -or
                              $_.CommandLine -like '*run_m5_*_verify.ps1*' -or
                              $_.CommandLine -like '*run_m5_e2_ab.ps1*') -and
                             $_.ProcessId -ne $PID })
    if ($busy.Count -eq 0) { break }
    if ($waited % 600 -eq 0) { Say "waiting for $($busy.Count) job(s)" }
    Start-Sleep -Seconds 30
    $waited += 30
    if ($waited -gt 6 * 3600) { Say "GAVE UP waiting after 6 h"; exit 1 }
}
Say "clear after $([int]($waited/60)) min"

# ---- 1. batch identity on the real model (E1's only real-model evidence) ----
Say "m4_batch_identity starting (N in 1,2,4,8,16)"
& python "tests\tooling\m4_batch_identity.py" --stoke "build\win\stoke.exe" `
    --model $Q4 -n 16 2>&1 |
    ForEach-Object { $_ | Out-File -Append -Encoding ascii $log; Write-Output $_ }
$bi = $LASTEXITCODE
Say "m4_batch_identity rc=$bi"
if ($bi -ne 0) {
    Say "BATCH IDENTITY RED -- prefill A/B deliberately NOT run"
    exit $bi
}

# ---- 2. E3's prefill A/B ----------------------------------------------------
Say "prefill A/B starting ($AbCycles cycles)"
& powershell -NoProfile -ExecutionPolicy Bypass -File "tests\tooling\run_m5_e3_prefill_ab.ps1" -Cycles $AbCycles
Say "prefill A/B rc=$LASTEXITCODE"
Say "DONE"
