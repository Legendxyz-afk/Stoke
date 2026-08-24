# M5 -- the Q4_K mins term in SIMD (`STOKE_MINS_SIMD=1`): the correctness gate.
#
# WHAT THIS HAS TO PROVE. `vec_dot_q4_k_q8_k_xn_avx2` computed its per-sequence
# mins term with a scalar loop; the B arm replaces it with one `vpmaddwd`. The
# claim is BIT-EXACTNESS BY CONSTRUCTION -- every term integer, integer addition
# associative, and nothing overflows (|bs| <= 2048 from 16 quants clamped to
# +-128; a pair <= 4096 against i16's 32767; x a 6-bit min <= 258,048; vpmaddwd
# sums two to <= 516,096 against i32; bound derived from `quantize_row_q8_k`).
# "By construction" is an ARGUMENT. This project ships on measurement, and a
# change on the M3/M4 parity lineage is graded against the archives, not against
# three ad-hoc cases (memo 2026-08-20-m5-attribution-RESULTS.md section 3).
#
# ▶▶ WHERE THE KERNEL EXECUTES, AND WHY `stoke run` IS NOT VACUOUS HERE.
# The 08-20 memo warned that the byte-compare "had to go through `batch --seqs
# N`, not `stoke run`", because the kernel only runs at n_seq > 1. That warning
# was about an N=1 comparison and it does NOT make the run path vacuous: E3
# chunks prefill through `model_forward_batch` (model.c prefill_range, default
# STOKE_PREFILL_CHUNK=128), and vec.c tiles n_seq at KERNELS_BATCH_TILE=16 --
# so `run` on p00 (~30 prompt tokens) enters the kernel at n_seq=16 then 13, and
# on p41 (~525) at 16 eight times per chunk then the remainder. Decode still
# runs at n_seq=1 and does not reach it.
#
# THAT IS AN ARGUMENT TOO, SO THE GATE DOES NOT REST ON IT. The `mutant` arm
# below is built from the same source with ONE WRONG LANE in the duplication
# mask, and it runs inside this gate on one case per leg. A gate that has never
# been shown to fail is not evidence; this one carries its own proof of power in
# its own log. (Pre-flight 2026-08-24: mutant DIFFERs on both legs, scalar and
# simd both SAME.)
#
# THE ARMS, and why each one is here:
#   scalar -- HEAD's source at the shipping default, rebuilt today. It must
#             STILL match the archives. This is what separates "the define broke
#             it" from "the rebuild broke it", and it doubles as proof that THIS
#             BUILD is sound, which is why no separate referent-validation run
#             is needed. (Same role as the E2 script's gqa0/par1 arm.)
#   simd   -- the change: -DSTOKE_MINS_SIMD=1, identical CFLAGS otherwise, via
#             the Makefile's EXTRA_CFLAGS hook so no flag can silently differ.
#   mutant -- simd with dup[5] = 3 instead of 2. Runs on ONE case per leg. Its
#             DIFFER is the gate's power; a SAME from it means this gate proves
#             nothing and the run must be stopped.
#
# ▶ THERE IS DELIBERATELY NO chunk1 ARM. The define is textually scoped to two
# `#if STOKE_MINS_SIMD` blocks inside one function that only executes at
# n_seq > 1, so a chunk1 arm is byte-identical to the scalar arm BY
# CONSTRUCTION. run_m5_entry_verify.ps1's header names the failure this avoids:
# a gate whose arms are secretly the same arm still passes, and reads to a later
# auditor like twice the evidence.
#
# LEG 1 -- the 10-case lineage grid on the `run` path, against the archives.
# Axes and their reasons are inherited from run_m5_e2_verify.ps1 unchanged:
# --kv q8 is MANDATORY (at f32 kv_read is zero-copy and never touches the
# per-worker KV scratch); p00 vs p41 because a 19-word prompt exercises almost
# none of the loop; two thread counts because the pool distributes rows.
#   ⚠ THE TWO Q8_0 CASES ARE A CONTROL, NOT EVIDENCE FOR THIS CHANGE. A Q8_0
#   file has no K-quant tensors, so matvec_batch never takes the branch this
#   kernel lives on and the define cannot reach it. They are here because they
#   are part of the 10-case grid and they prove the rebuild disturbed nothing
#   else -- and they are LABELLED so a later reader does not count them twice.
#
# LEG 2 -- the `batch` path, where the 08-20 evidence was taken, so this gate is
# a superset of it rather than a substitute. It reaches n_seq in {2, 8, 16},
# which leg 1 never does (leg 1 only ever sees 16 and a chunk remainder). The
# arithmetic per (super-block, sequence) cannot depend on n_seq -- there is no
# cross-sequence reduction -- but the CODEGEN can, and that is what is being
# checked.
#   ⚠ p41 IS NOT IN LEG 2. The referent is pre-pool, so it prefills 1050
#   positions with single-threaded attention; measured cost is ~20 min for that
#   one referent against ~3 min for all of p00's. Leg 1 already grades p41
#   against the same archive at length.
#
# Every arm is compared against an ARCHIVED binary, never against another arm:
# agreement among new arms is also consistent with all of them being wrong
# together (the L4 lesson, and the reason the scalar arm exists at all).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_mins_verify.ps1
#   ... -Quick    pre-flight only (~5 min), NOT a gate.

param(
    [int]$NGen = 24,
    [string]$Stamp = "2026-08-24-m5-mins-verify",
    # PRE-FLIGHT, NOT A GATE. Every harness defect this project has recorded was
    # caught by pre-flight or by watching the FIRST unit, and none by a long run
    # finishing -- so the long run is worth starting only after the short one is
    # green. A -Quick pass is NOT evidence for a gate report and the log says so.
    [switch]$Quick
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$Q8   = "$root\models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
# The SAME two referents M4.5 and the M5 entry closed against. Reusing them
# rather than a fresh build keeps the anchor one that has already been graded.
$archiveQ4 = "$root\build\stoke-m45-maddubs-f3b7181a.exe"  # pre-lanes, pre-pool
$archiveQ8 = "$root\build\stoke-m3-gate-876ce53f.exe"      # the Q8_0 anchor

$armScalar = "$root\build\win-mins-scalar\stoke.exe"
$armSimd   = "$root\build\win-mins-simd\stoke.exe"
$armMutant = "$root\build\win-mins-mutant\stoke.exe"
$arms = @(
    @{ name = "scalar"; exe = $armScalar },
    @{ name = "simd";   exe = $armSimd }
)

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding ascii $log
    Write-Output $line
}

# Hash12 and not H: `h` is a PowerShell ALIAS for Get-History and aliases beat
# functions in name resolution, so a helper named H silently became a history
# lookup that took the whole model stream as an -Id. Cost one pre-flight run on
# 2026-08-24. Same family as run_m5_e2_verify.ps1's $args note.
function Hash12($s) {
    $md5 = [System.Security.Cryptography.MD5]::Create()
    $b = [System.Text.Encoding]::UTF8.GetBytes($s)
    (($md5.ComputeHash($b) | ForEach-Object { $_.ToString("x2") }) -join "").Substring(0, 12)
}

function Stream($exe, $cmdArgs) {
    # A comparison is only as good as its referent, so the referent is checked
    # before it is used -- a missing or damaged $exe otherwise degrades into a
    # DIFFER, which reads as "the code under test is wrong" and is the single
    # most expensive way for an instrument to fail.
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "STOP: no such binary '$exe' -- the instrument is broken, not the engine"
    }
    $out = (& $exe @cmdArgs | Out-String)
    if ($out.Trim().Length -eq 0) {
        throw "STOP: '$exe' produced no stream -- args: $($cmdArgs -join ' ')"
    }
    return $out
}

Set-Location $root

# THE BINARIES UNDER TEST MUST BE NEWER THAN THE SOURCE. A gate that grades a
# stale artifact is worse than no gate -- it manufactures confidence. Cost the
# L4 session four minutes of a green gate about the wrong binary.
#
# ⚠ THE MUTANT IS EXEMPT, AND NOT AS A CONVENIENCE. It is built from a working
# tree that was deliberately modified and then reverted, so the source it was
# built from does not exist on disk any more and a newest-source comparison
# CANNOT be satisfied by it -- it fired here on a ONE-SECOND gap left by the
# `git checkout` that restored simd.c. What actually protects the mutant is
# stronger than a timestamp: if it were stale, or accidentally the same build as
# an arm, it would MATCH the archive and the power check below would fail the
# whole gate. Its guard is its own job.
$newest = (Get-ChildItem -Path "$root\src" -Recurse -Include *.c, *.h |
           Sort-Object LastWriteTime -Descending)[0]
if (-not (Test-Path -LiteralPath $armMutant)) { throw "STOP: missing arm '$armMutant'" }
foreach ($a in @($armScalar, $armSimd)) {
    if (-not (Test-Path -LiteralPath $a)) { throw "STOP: missing arm '$a'" }
    $t = (Get-Item -LiteralPath $a).LastWriteTime
    if ($t -lt $newest.LastWriteTime) {
        throw ("STOP: $a ($t) is OLDER than $($newest.Name) " +
               "($($newest.LastWriteTime)) -- rebuild before gating")
    }
}
# The mutant must not BE one of the graded arms -- a copy-paste that pointed all
# three at the same file would turn the power check into a tautology it passes.
$hMut = (Get-FileHash -Algorithm MD5 $armMutant).Hash
foreach ($a in $arms) {
    if ((Get-FileHash -Algorithm MD5 $a.exe).Hash -eq $hMut) {
        throw "STOP: the mutant is byte-identical to the '$($a.name)' arm -- the power check would be a tautology"
    }
}

Say "=== M5 mins-term SIMD verification, branch $(git branch --show-current)"
Say "    n=$NGen  commit=$(git rev-parse --short HEAD)"
Say "    scalar=$armScalar"
Say "    simd  =$armSimd   (-DSTOKE_MINS_SIMD=1 via EXTRA_CFLAGS)"
Say "    mutant=$armMutant (dup[5]=3, power check only -- a SAME here voids the gate)"
Say ("    newest source $($newest.Name) $($newest.LastWriteTime)")

# Matches run_m5_entry_verify.ps1 so the two logs are directly comparable.
# MIN_POS=0 takes the pooled attention path from position 0 rather than only
# past the dispatch threshold; the threshold exists for speed, and this run is
# about correctness.
$env:STOKE_ATT_MIN_POS = "0"
$env:STOKE_ATT_PAR = "1"

$fail = 0
$powerSeen = 0   # DIFFERs from the mutant; must end at 2 (one per leg)

$prompts = if ($Quick) { @("p00") } else { @("p00", "p41") }
if ($Quick) { Say "    *** -Quick: PRE-FLIGHT ONLY. NOT a gate. ***" }
$cases = @()
foreach ($kv in @("f32", "q8")) {
    foreach ($p in $prompts) {
        foreach ($t in $(if ($Quick) { @(6) } else { @(2, 6) })) {
            $cases += @{ kv = $kv; p = $p; t = $t }
        }
    }
}
# Leg 2: n_seq shapes leg 1 never reaches. ctx 256 for every case so the block
# pool is sized identically across them (M4 reserves the WHOLE context per
# sequence at admission, so 16 x 512 would be a pool question, not a kernel one).
$batchCases = @()
if (-not $Quick) {
    # All at t=6. Thread count is leg 1's axis and does not interact with n_seq:
    # the pool distributes ROWS (n_out), and n_seq lives entirely inside one
    # row's kernel call. Putting seqs=16 at t=2 would have cost ~14 min against
    # a pre-pool referent to re-ask a question leg 1 already answers.
    $batchCases += @{ seqs = 2;  kv = "f32"; t = 6 }
    $batchCases += @{ seqs = 8;  kv = "f32"; t = 6 }
    $batchCases += @{ seqs = 16; kv = "f32"; t = 6 }
    $batchCases += @{ seqs = 8;  kv = "q8";  t = 6 }
} else {
    $batchCases += @{ seqs = 8;  kv = "f32"; t = 6 }
}

# THE DENOMINATOR, WRITTEN ONCE, FOR THE STATUS BAR -- and deliberately ONLY the
# denominator. Progress is counted from the log's own SAME/DIFFER lines, never
# from anything this file claims. That split is what stops a "running" claim
# outliving the run.
$nQ8 = $(if ($Quick) { 0 } else { 2 })
@{ name = $Stamp; total = ($cases.Count + $nQ8 + $batchCases.Count); arms = $arms.Count
   plan = "$($cases.Count) Q4_K_M run + $nQ8 Q8_0 run + $($batchCases.Count) batch cases x $($arms.Count) arms, + 2 mutant power checks"
   started = (Get-Date -Format s) } | ConvertTo-Json -Compress |
    Out-File -Encoding ascii "$gr\$Stamp.status.json"

Say ""
Say "--- LEG 1: run path, chunk 128, vs the archives (10 cases) ---"

$firstLeg1 = $true
foreach ($c in $cases) {
    $prompt = "$root\tests\data\parity\$($c.p).txt"
    # The archive predates chunked prefill and ignores the variable; it is set
    # anyway so the referent's environment is stated rather than assumed.
    $env:STOKE_PREFILL_CHUNK = "1"
    $refArgs = @("run", $Q4, "--raw", "-n", "$NGen", "-f", $prompt,
                 "--threads", "$($c.t)", "--kv", $c.kv)
    $want = Stream $archiveQ4 $refArgs
    $res = @()
    $armList = $arms
    # The mutant runs on the FIRST leg-1 case only: its job is to show this
    # comparison can fail, and one case does that as well as ten.
    if ($firstLeg1) { $armList = $arms + @(@{ name = "mutant"; exe = $armMutant }) }
    foreach ($a in $armList) {
        $env:STOKE_PREFILL_CHUNK = "128"
        $got = Stream $a.exe $refArgs
        $ok = ($got -eq $want)
        if ($a.name -eq "mutant") {
            if ($ok) {
                Say "  *** MUTANT MATCHED THE ARCHIVE -- this gate has NO POWER on the"
                Say "  *** run path. Every SAME below is vacuous. STOP."
                $fail++
            } else { $powerSeen++ }
            $res += ("mutant=" + $(if ($ok) { "SAME(!!)" } else { "DIFFER(good)" }))
            continue
        }
        if (-not $ok) { $fail++ }
        $res += ("{0}={1}" -f $a.name, $(if ($ok) { "SAME" } else { "DIFFER" }))
    }
    Say ("  Q4_K_M {0,-4} kv={1,-3} t={2}  ref={3}  {4}" -f
         $c.p, $c.kv, $c.t, (Hash12 $want), ($res -join "  "))
    $firstLeg1 = $false
}

# The Q8_0 model against the m3 archive. ⚠ CONTROL, NOT EVIDENCE: a Q8_0 file
# carries no K-quant tensors, so matvec_batch never reaches the modified kernel.
# What these two cases prove is that the rebuild disturbed nothing else.
foreach ($t in $(if ($Quick) { @() } else { @(2, 6) })) {
    $prompt = "$root\tests\data\parity\p41.txt"
    $env:STOKE_PREFILL_CHUNK = "1"
    # kv omitted entirely: the m3 archive predates --kv, and passing it a flag
    # it never had fails as a usage error that reads as a stream diff.
    $refArgs = @("run", $Q8, "--raw", "-n", "$NGen", "-f", $prompt, "--threads", "$t")
    $want = Stream $archiveQ8 $refArgs
    $res = @()
    foreach ($a in $arms) {
        $env:STOKE_PREFILL_CHUNK = "128"
        $got = Stream $a.exe $refArgs
        $ok = ($got -eq $want)
        if (-not $ok) { $fail++ }
        $res += ("{0}={1}" -f $a.name, $(if ($ok) { "SAME" } else { "DIFFER" }))
    }
    Say ("  Q8_0   p41  vs m3   t={0}  ref={1}  {2}   [CONTROL: no K-quant tensor, kernel unreachable]" -f
         $t, (Hash12 $want), ($res -join "  "))
}

Say ""
Say "--- LEG 2: batch path, p00, n_seq in {2,8,16}, vs the m45 archive ---"

$firstLeg2 = $true
foreach ($b in $batchCases) {
    $prompt = "$root\tests\data\parity\p00.txt"
    $env:STOKE_PREFILL_CHUNK = "1"
    $refArgs = @("batch", $Q4, "--raw", "-n", "$NGen", "-f", $prompt,
                 "--seqs", "$($b.seqs)", "--ctx", "256",
                 "--threads", "$($b.t)", "--kv", $b.kv)
    $want = Stream $archiveQ4 $refArgs
    $res = @()
    $armList = $arms
    if ($firstLeg2) { $armList = $arms + @(@{ name = "mutant"; exe = $armMutant }) }
    foreach ($a in $armList) {
        $env:STOKE_PREFILL_CHUNK = "128"
        $got = Stream $a.exe $refArgs
        $ok = ($got -eq $want)
        if ($a.name -eq "mutant") {
            if ($ok) {
                Say "  *** MUTANT MATCHED THE ARCHIVE -- no power on the batch path. STOP."
                $fail++
            } else { $powerSeen++ }
            $res += ("mutant=" + $(if ($ok) { "SAME(!!)" } else { "DIFFER(good)" }))
            continue
        }
        if (-not $ok) { $fail++ }
        $res += ("{0}={1}" -f $a.name, $(if ($ok) { "SAME" } else { "DIFFER" }))
    }
    Say ("  batch  seqs={0,-2} kv={1,-3} t={2}  ref={3}  {4}" -f
         $b.seqs, $b.kv, $b.t, (Hash12 $want), ($res -join "  "))
    $firstLeg2 = $false
}

$env:STOKE_PREFILL_CHUNK = $null
$env:STOKE_ATT_MIN_POS = $null
$env:STOKE_ATT_PAR = $null

Say ""
$wantPower = $(if ($Quick) { 2 } else { 2 })
if ($powerSeen -lt $wantPower) {
    Say "POWER CHECK INCOMPLETE: $powerSeen/$wantPower mutant DIFFERs seen. The gate"
    Say "did not demonstrate it can fail, so its SAMEs mean nothing. STOP."
    exit 1
}
if ($fail -eq 0) {
    if ($Quick) {
        Say "PRE-FLIGHT GREEN, power $powerSeen/2 -- run WITHOUT -Quick for the gate evidence."
        exit 0
    }
    Say "ALL ARMS BIT-IDENTICAL TO THE ARCHIVES, power $powerSeen/2 confirmed by the"
    Say "mutant on BOTH legs. STOKE_MINS_SIMD=1 moves no arithmetic."
    Say "Speed is a SEPARATE question and is NOT measured here: that is the"
    Say "interleaved A/B, already banked at 2026-08-20-m5-mins-ab-v2-report.md."
    exit 0
}
Say "$fail ARM(S) MOVED THE STREAM -- read WHICH: a scalar DIFFER implicates the"
Say "rebuild or the tree, a simd-only DIFFER implicates the vpmaddwd spelling. STOP."
exit 1
