# M5 ENTRY -- all three rungs together: the correctness gate, in one command.
#
# WHY THIS EXISTS BESIDE run_m5_e2_verify.ps1, WHICH IS NOT SUPERSEDED. That
# script gated E2 alone across par 0/1 and its log is E2's evidence. This one
# adds the axis E3 introduced and is the gate for the entry as it SHIPS.
#
# ▶▶ E3 IS THE RISKIEST OF THE THREE FOR BIT-EXACTNESS, AND NOT FOR AN OBVIOUS
# REASON. E1 and E2 rearrange loops. E3 changes WHICH KERNEL RUNS: prefill at
# n_seq=1 takes the solo K-quant path (rows_q4_k_avx2 -> vec_dot_q4_k_q8_k_avx2)
# and a batched chunk takes the fused one (rows_q4_k_batch_avx2 ->
# vec_dot_q4_k_q8_k_xn_avx2), because matvec_batch routes on `simd && n_seq > 1`.
# Those two kernels being byte-identical is M4 gate item [2]'s result -- but that
# gate exercised it on the BATCH path, and E3 puts it on the `run` path that
# every parity artifact in this project was made with. Same claim, new caller.
#
# THE ARMS are prefill chunk widths:
#   chunk1   -- prefill position by position, so `run` takes the SOLO K-quant
#               kernel throughout. This is also the arm that proves the build.
#   chunk128 -- the shipping default, which routes prefill through the FUSED
#               kernel. A DIFFER here and not in chunk1 points straight at that
#               kernel switch rather than at "the entry".
#
# ▶ IT HAD FOUR ARMS UNTIL 2026-08-19 (base/e2/e3/ship over a gqa x chunk grid).
# The gqa axis selected M4.5's q-head-outer attention, which was deleted once
# the A/B that needed it in-binary had reported -- so two of the four arms named
# a configuration that no longer exists. They were removed rather than left to
# run as silent duplicates of the other two: a gate whose arms are secretly the
# same arm still passes, and reads to a later auditor like twice the evidence.
#
# par 0 vs 1 is NOT an axis here: run_m5_e2_verify.ps1 ran all four gqa x par
# combinations green across these same cases, and neither E1 nor E3 changes how
# attention is dispatched.
#
# Axes and their reasons are the E2 script's, unchanged: --kv q8 is MANDATORY
# (at f32 kv_read is zero-copy and never touches the per-worker KV scratch, so
# an f32-only check cannot see a buffer that is too small or shared); p00 vs p41
# because a 19-word prompt exercises almost none of what these rungs touch; two
# thread counts because thread-invariance is a fresh claim under a new unit
# decomposition.

param(
    [int]$NGen = 24,
    [string]$Stamp = "2026-08-19-m5-entry-verify",
    [switch]$Quick
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$Q8   = "$root\models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
$archiveQ4 = "$root\build\stoke-m45-maddubs-f3b7181a.exe"  # pre-lanes, pre-pool
$archiveQ8 = "$root\build\stoke-m3-gate-876ce53f.exe"      # the Q8_0 anchor
$NEW  = "$root\build\win\stoke.exe"

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding ascii $log
    Write-Output $line
}

function Stream($exe, $model, $prompt, $kv, $threads) {
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "STOP: no such binary '$exe' -- the instrument is broken, not the engine"
    }
    $cmdArgs = @("run", $model, "--raw", "-n", "$NGen", "-f", $prompt, "--threads", "$threads")
    if ($kv -ne "") { $cmdArgs += @("--kv", $kv) }
    $out = (& $exe @cmdArgs | Out-String)
    if ($out.Trim().Length -eq 0) {
        throw "STOP: '$exe' produced no stream on $prompt"
    }
    return $out
}

Set-Location $root

$newest = (Get-ChildItem -Path "$root\src" -Recurse -Include *.c, *.h |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1)
$exeTime = (Get-Item -LiteralPath $NEW).LastWriteTime
if ($exeTime -lt $newest.LastWriteTime) {
    throw ("STOP: $NEW ($exeTime) is OLDER than $($newest.Name) " +
           "($($newest.LastWriteTime)) -- rebuild before gating")
}

Say "=== M5 ENTRY verification (E1+E2+E3), branch $(git branch --show-current)"
Say "    n=$NGen  new=$NEW  commit=$(git rev-parse --short HEAD)"
Say ("    binary $exeTime, newest source $($newest.Name) $($newest.LastWriteTime)")

$env:STOKE_ATT_MIN_POS = "0"
$env:STOKE_ATT_PAR = "1"

# ▶ THE gqa AXIS IS GONE (2026-08-19). It selected between E2's kv-head-outer
# loop and M4.5's q-head-outer one; the latter was deleted once the A/B that
# needed it in-binary had reported. Two arms remain, and they are the two
# configurations that still exist. The banked log from the four-arm run stands
# as its own artifact -- it records what it ran, which was true then.
$arms = @(
    @{ name = "chunk1";   chunk = "1" },
    @{ name = "chunk128"; chunk = "128" }
)

$fail = 0
$prompts = if ($Quick) { @("p00") } else { @("p00", "p41") }
if ($Quick) { Say "    *** -Quick: PRE-FLIGHT ONLY, p00 at t=6, no Q8_0 leg. NOT a gate. ***" }
$cases = @()
foreach ($kv in @("f32", "q8")) {
    foreach ($p in $prompts) {
        foreach ($t in $(if ($Quick) { @(6) } else { @(2, 6) })) {
            $cases += @{ kv = $kv; p = $p; t = $t }
        }
    }
}

$nQ8 = $(if ($Quick) { 0 } else { 2 })
@{ name = $Stamp; total = ($cases.Count + $nQ8); arms = $arms.Count
   plan = "$($cases.Count) Q4_K_M + $nQ8 Q8_0 cases x $($arms.Count) rung arms"
   started = (Get-Date -Format s) } | ConvertTo-Json -Compress |
    Out-File -Encoding ascii "$gr\$Stamp.status.json"

foreach ($c in $cases) {
    $prompt = "$root\tests\data\parity\$($c.p).txt"
    # The archive predates every switch and ignores them; set anyway so the
    # referent's environment is stated rather than assumed.
    $env:STOKE_PREFILL_CHUNK = "1"
    $want = Stream $archiveQ4 $Q4 $prompt $c.kv $c.t
    $res = @()
    foreach ($a in $arms) {
        $env:STOKE_PREFILL_CHUNK = $a.chunk
        $got = Stream $NEW $Q4 $prompt $c.kv $c.t
        $ok = ($got -eq $want)
        if (-not $ok) { $fail++ }
        $res += ("{0}={1}" -f $a.name, $(if ($ok) { "SAME" } else { "DIFFER" }))
    }
    Say ("  Q4_K_M {0,-4} kv={1,-3} t={2}  {3}" -f $c.p, $c.kv, $c.t, ($res -join "  "))
}

foreach ($t in $(if ($Quick) { @() } else { @(2, 6) })) {
    $prompt = "$root\tests\data\parity\p41.txt"
    $env:STOKE_PREFILL_CHUNK = "1"
    $want = Stream $archiveQ8 $Q8 $prompt "" $t
    $res = @()
    foreach ($a in $arms) {
        $env:STOKE_PREFILL_CHUNK = $a.chunk
        $got = Stream $NEW $Q8 $prompt "" $t
        $ok = ($got -eq $want)
        if (-not $ok) { $fail++ }
        $res += ("{0}={1}" -f $a.name, $(if ($ok) { "SAME" } else { "DIFFER" }))
    }
    Say ("  Q8_0   p41  vs m3   t={0}  {1}" -f $t, ($res -join "  "))
}

$env:STOKE_ATT_PAR = $null
$env:STOKE_ATT_MIN_POS = $null
$env:STOKE_PREFILL_CHUNK = $null

if ($fail -eq 0) {
    if ($Quick) {
        Say "PRE-FLIGHT GREEN on p00 -- run WITHOUT -Quick for the gate evidence."
        exit 0
    }
    Say "ALL RUNG ARMS BIT-IDENTICAL TO THE ARCHIVES -- E1+E2+E3 move no arithmetic."
    Say "Speed is a SEPARATE question and is NOT measured here."
    exit 0
}
Say "$fail ARM(S) MOVED THE STREAM -- read WHICH arm: chunk1 implicates the build"
Say "or the attention nest, chunk128 the fused K-quant kernel on the run path. STOP."
exit 1
