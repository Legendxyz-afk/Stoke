# M5 rung E2 -- kv-head-outer attention: the correctness gate, in one command.
#
# WHAT THIS HAS TO PROVE. E2 rewrites the attention loop nest so a kv-head's K
# and V are fetched ONCE for its whole GQA group instead of once per q-head (at
# ctx 1500: 1.77 GB of KV reads per token becoming 0.44 GB -- design note
# docs/superpowers/design-notes/2026-08-19-m5-entry.md section 3). The claim is
# that this moves NO arithmetic: same dot products, same left-to-right order,
# same accumulation order over j; only the loop nest and the fetch count change.
# A claim of that shape is worth exactly the byte-compare that checks it.
#
# THE ARMS, and why each one is here:
#   gqa0/par1 -- M4.5's shipping path, unchanged code. It must STILL match the
#                archive: E2's commit also moved where the per-worker KV scratch
#                starts, and this arm is what proves that move was invisible. It
#                doubles as proof that THIS BUILD of the m45 source is sound,
#                which is why a separate referent-validation run is not needed.
#   gqa0/par0 -- the same, serial. Two agreeing arms can agree by sharing a
#                defect, and the serial path is where a scratch-layout mistake
#                would hide (one worker, one buffer, no contention to expose it).
#   gqa1/par1 -- E2 itself, pooled: the shipping default.
#   gqa1/par0 -- E2 serial. E2 changes the UNIT COUNT, so "the units are
#                independent" is a fresh claim; running it on one thread removes
#                the pool from the explanation of a pass.
#
# THE AXES:
#   kv f32 vs q8 -- MANDATORY, NOT OPTIONAL. At f32 kv_read is zero-copy and
#                   never touches the per-worker KV scratch, so an f32-only
#                   check cannot see a buffer that is too small or shared. That
#                   is exactly how the 4-lane loop's 3072-byte overrun survived
#                   its first review, and E2 CHANGES THAT BUFFER'S LAYOUT.
#   p00 vs p41   -- 19 words vs ~400. E2's whole subject is what happens as the
#                   context grows, and a short prompt exercises almost none of
#                   the loop it rewrites.
#   threads 2/6  -- E2 changes the UNIT COUNT (n_head_kv=8, not n_head=32), so
#                   thread-count invariance is not inherited from L4; it is a
#                   fresh claim about a different decomposition.
#
# Every arm is compared against an ARCHIVED binary, never against another arm:
# agreement among three new arms would also be consistent with all three being
# wrong together (the L4 lesson, and the reason arm A exists at all).

param(
    [int]$NGen = 24,
    [string]$Stamp = "2026-08-19-m5-e2-verify",
    # PRE-FLIGHT, NOT A GATE. -Quick drops the long prompt and the Q8_0 model,
    # turning ~3.5 h into ~5 min. It exists because every harness defect this
    # project has recorded was caught by pre-flight or by watching the FIRST
    # unit, and none by a long run finishing -- so the long run is worth
    # starting only after the short one is green. A -Quick pass is NOT evidence
    # for a gate report and the log says so.
    [switch]$Quick
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$gr   = "$root\docs\superpowers\gate-reports"
$log  = "$gr\$Stamp.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$Q8   = "$root\models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
# The SAME two referents M4.5 closed against (its byte-compare was 10/10 on
# these). Reusing them rather than the m45 build keeps the anchor one that has
# already been graded, and makes the gqa0 arms a check on this build too.
$archiveQ4 = "$root\build\stoke-m45-maddubs-f3b7181a.exe"  # pre-lanes, pre-pool
$archiveQ8 = "$root\build\stoke-m3-gate-876ce53f.exe"      # the Q8_0 anchor
$NEW  = "$root\build\win\stoke.exe"

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding ascii $log
    Write-Output $line
}

function Stream($exe, $model, $prompt, $kv, $threads) {
    # A comparison is only as good as its referent, so the referent is checked
    # before it is used -- a missing or damaged $exe otherwise degrades into a
    # DIFFER, which reads as "the code under test is wrong" and is the single
    # most expensive way for an instrument to fail.
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "STOP: no such binary '$exe' -- the instrument is broken, not the engine"
    }
    # kv="" omits the flag entirely: the m3 archive predates --kv, so passing it
    # a flag it never had fails as a usage error and reads as a stream diff.
    # $cmdArgs and not $args: $args is a PowerShell AUTOMATIC variable holding a
    # function's unbound arguments, and shadowing an automatic is a real trap
    # this harness has already been bitten by (see run_att_par_verify.ps1).
    $cmdArgs = @("run", $model, "--raw", "-n", "$NGen", "-f", $prompt, "--threads", "$threads")
    if ($kv -ne "") { $cmdArgs += @("--kv", $kv) }
    $out = (& $exe @cmdArgs | Out-String)
    if ($out.Trim().Length -eq 0) {
        throw "STOP: '$exe' produced no stream on $prompt"
    }
    return $out
}

Set-Location $root

# THE BINARY UNDER TEST MUST BE NEWER THAN THE SOURCE. A gate that grades a
# stale artifact is worse than no gate -- it manufactures confidence. Cost the
# L4 session four minutes of a green gate about the wrong binary.
$newest = (Get-ChildItem -Path "$root\src" -Recurse -Include *.c, *.h |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1)
$exeTime = (Get-Item -LiteralPath $NEW).LastWriteTime
if ($exeTime -lt $newest.LastWriteTime) {
    throw ("STOP: $NEW ($exeTime) is OLDER than $($newest.Name) " +
           "($($newest.LastWriteTime)) -- rebuild before gating")
}

Say "=== M5 E2 kv-head-outer attention verification, branch $(git branch --show-current)"
Say "    n=$NGen  new=$NEW"
Say ("    binary $exeTime, newest source $($newest.Name) $($newest.LastWriteTime)")

# MIN_POS=0 so the pooled path is taken from position 0 rather than only past
# the dispatch threshold -- this run is about correctness, and the threshold
# exists for speed.
$env:STOKE_ATT_MIN_POS = "0"

# The three arms, as (gqa, lanes) pairs. lanes is meaningless when gqa=1 and is
# pinned to the default there rather than left unset, so the log records the
# whole environment each arm ran under.
$arms = @(
    @{ name = "gqa0/par1"; gqa = "0"; par = "1"; lanes = "4" },
    @{ name = "gqa0/par0"; gqa = "0"; par = "0"; lanes = "4" },
    @{ name = "gqa1/par1"; gqa = "1"; par = "1"; lanes = "4" },
    @{ name = "gqa1/par0"; gqa = "1"; par = "0"; lanes = "4" }
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

# THE DENOMINATOR, WRITTEN ONCE, FOR THE STATUS BAR -- and deliberately ONLY
# the denominator. The case grid is fixed here and the log cannot reveal it, so
# the bar has to be told; but progress is counted from the log's own SAME/DIFFER
# lines, never from anything this file claims. That split is what stops a
# "running" claim outliving the run, which is how the bar once sat on
# "M2 24/24 BANKED" for two days.
$nQ8 = $(if ($Quick) { 0 } else { 2 })
@{ name = $Stamp; total = ($cases.Count + $nQ8); arms = $arms.Count
   plan = "$($cases.Count) Q4_K_M + $nQ8 Q8_0 cases x $($arms.Count) arms"
   started = (Get-Date -Format s) } | ConvertTo-Json -Compress |
    Out-File -Encoding ascii "$gr\$Stamp.status.json"

foreach ($c in $cases) {
    $prompt = "$root\tests\data\parity\$($c.p).txt"
    # The archive predates every one of these switches and ignores them; they
    # are cleared anyway so the referent's environment is stated, not assumed.
    $env:STOKE_ATT_LANES = "1"; $env:STOKE_ATT_PAR = "0"; $env:STOKE_ATT_GQA = "0"
    $want = Stream $archiveQ4 $Q4 $prompt $c.kv $c.t
    $res = @()
    foreach ($a in $arms) {
        $env:STOKE_ATT_GQA = $a.gqa
        $env:STOKE_ATT_LANES = $a.lanes
        $env:STOKE_ATT_PAR = $a.par
        $got = Stream $NEW $Q4 $prompt $c.kv $c.t
        $ok = ($got -eq $want)
        if (-not $ok) { $fail++ }
        $res += ("{0}={1}" -f $a.name, $(if ($ok) { "SAME" } else { "DIFFER" }))
    }
    Say ("  Q4_K_M {0,-4} kv={1,-3} t={2}  {3}" -f $c.p, $c.kv, $c.t, ($res -join "  "))
}

# The Q8_0 model against the m3 archive: a different WEIGHT type through the
# same attention loop, and the one anchor that predates every M4.5 change.
foreach ($t in $(if ($Quick) { @() } else { @(2, 6) })) {
    $prompt = "$root\tests\data\parity\p41.txt"
    $env:STOKE_ATT_LANES = "1"; $env:STOKE_ATT_PAR = "0"; $env:STOKE_ATT_GQA = "0"
    $want = Stream $archiveQ8 $Q8 $prompt "" $t
    $res = @()
    foreach ($a in $arms) {
        $env:STOKE_ATT_GQA = $a.gqa
        $env:STOKE_ATT_LANES = $a.lanes
        $env:STOKE_ATT_PAR = $a.par
        $got = Stream $NEW $Q8 $prompt "" $t
        $ok = ($got -eq $want)
        if (-not $ok) { $fail++ }
        $res += ("{0}={1}" -f $a.name, $(if ($ok) { "SAME" } else { "DIFFER" }))
    }
    Say ("  Q8_0   p41  vs m3   t={0}  {1}" -f $t, ($res -join "  "))
}

$env:STOKE_ATT_LANES = $null
$env:STOKE_ATT_PAR = $null
$env:STOKE_ATT_GQA = $null
$env:STOKE_ATT_MIN_POS = $null

if ($fail -eq 0) {
    if ($Quick) {
        Say "PRE-FLIGHT GREEN on p00 -- run WITHOUT -Quick for the gate evidence."
        exit 0
    }
    Say "ALL ARMS BIT-IDENTICAL TO THE ARCHIVES -- E2 moves no arithmetic."
    Say "Speed is a SEPARATE question and is NOT measured here: that needs the"
    Say "interleaved protocol on a quiet box, both arms from this one binary."
    exit 0
}
Say "$fail ARM(S) MOVED THE STREAM -- E2's bit-exactness claim is false as written. STOP."
exit 1
