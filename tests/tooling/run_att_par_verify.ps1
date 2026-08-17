# M4.5 L4 -- attention on the pool: the correctness gate, in one command.
#
# WHAT THIS HAS TO PROVE. The heads now run on the worker pool, one unit per
# q-head. The claim is that this is BIT-EXACT at any thread count, because a
# head's arithmetic is untouched and the parallelism is over independent
# outputs -- the same argument pool.h makes for matvec. A claim of that shape
# is worth exactly as much as the byte-compare that checks it.
#
# THE AXES, and why each one is here:
#   lanes 1 vs 4  -- the interleaved QK loop. Bit-exact by the same argument.
#   par 0 vs 1    -- serial heads vs pooled heads.
#   kv f32 vs q8  -- MANDATORY, NOT OPTIONAL. At f32 kv_read is zero-copy and
#                    never touches the per-worker KV scratch, so an f32-only
#                    check cannot see a buffer that is too small or shared.
#                    That is exactly how the 4-lane loop's 3072-byte overrun
#                    of the model-level s_kv survived its first review.
#   threads 2/6   -- thread-count invariance is the property being claimed, so
#                    it is checked at more than one count.
#
# Every arm is compared against the SAME archived binaries the standing recipe
# uses, not against each other: agreement among four new arms would also be
# consistent with all four being wrong together.

param(
    [int]$NGen = 48,
    [string]$Stamp = "2026-08-17-m45-att-par-verify"
)

$ErrorActionPreference = 'Continue'
$root = "M:\New Drive\Desktop\Programming\stoke"
$log  = "$root\docs\superpowers\gate-reports\$Stamp.txt"
$Q4   = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$Q8   = "$root\models\Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
# >> NAMED archiveQ4/archiveQ8, NOT $REF/$M3, AND THAT IS A BUG FIX.
# PowerShell variable names are CASE-INSENSITIVE, so the first version's
# `$ref = Stream $REF ...` assigned the token stream ON TOP OF the reference
# BINARY PATH. The first case passed for real, then every later case compared
# against a garbled referent and printed DIFFER -- four false failures of a
# change that was in fact correct, and the run was still cheerfully logging
# them. Distinct names, and the guard in Stream below, are the fix.
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
    # before it is used. Without this, anything that damages $exe -- the
    # case-insensitivity bug above, a moved archive, a typo -- degrades into a
    # DIFFER, which reads as "the code under test is wrong" and is the single
    # most expensive way for an instrument to fail.
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "STOP: no such binary '$exe' -- the instrument is broken, not the engine"
    }
    # kv="" omits the flag entirely: the m3 archive predates --kv (it is the
    # PRE-M4 binary, and Q8 KV is an M4 feature), so passing it a flag it never
    # had would fail as a usage error and be read as a stream difference.
    # $cmdArgs and not $args: $args is a PowerShell AUTOMATIC variable holding a
    # function's unbound arguments, and shadowing an automatic is the same class
    # of trap as the case-insensitive clobber above.
    $cmdArgs = @("run", $model, "--raw", "-n", "$NGen", "-f", $prompt, "--threads", "$threads")
    if ($kv -ne "") { $cmdArgs += @("--kv", $kv) }
    $out = (& $exe @cmdArgs | Out-String)
    if ($out.Trim().Length -eq 0) {
        throw "STOP: '$exe' produced no stream on $prompt"
    }
    return $out
}

Set-Location $root
Say "=== attention-on-the-pool verification, branch $(git branch --show-current)"
Say "    n=$NGen  new=$NEW"

# MIN_POS=0 so the pooled path is taken from position 0 rather than only after
# the context passes the dispatch threshold -- this run is about correctness,
# and the threshold exists for speed.
$env:STOKE_ATT_MIN_POS = "0"

$fail = 0
$cases = @()
foreach ($kv in @("f32", "q8")) {
    foreach ($p in @("p00", "p41")) {
        foreach ($t in @(2, 6)) {
            $cases += @{ kv = $kv; p = $p; t = $t }
        }
    }
}

foreach ($c in $cases) {
    $prompt = "$root\tests\data\parity\$($c.p).txt"
    $env:STOKE_ATT_LANES = "1"; $env:STOKE_ATT_PAR = "0"
    $want = Stream $archiveQ4 $Q4 $prompt $c.kv $c.t
    $res = @()
    foreach ($lanes in @("1", "4")) {
        foreach ($par in @("0", "1")) {
            $env:STOKE_ATT_LANES = $lanes; $env:STOKE_ATT_PAR = $par
            $got = Stream $NEW $Q4 $prompt $c.kv $c.t
            $ok = ($got -eq $want)
            if (-not $ok) { $fail++ }
            $res += ("lanes{0}/par{1}={2}" -f $lanes, $par, $(if ($ok) { "SAME" } else { "DIFFER" }))
        }
    }
    Say ("  Q4_K_M {0,-4} kv={1,-3} t={2}  {3}" -f $c.p, $c.kv, $c.t, ($res -join "  "))
}

# The Q8_0 model against the m3 archive: a different weight type through the
# same attention loop, and the one anchor that predates every M4.5 change.
foreach ($t in @(2, 6)) {
    $prompt = "$root\tests\data\parity\p41.txt"
    $want = Stream $archiveQ8 $Q8 $prompt "" $t
    $env:STOKE_ATT_LANES = "4"; $env:STOKE_ATT_PAR = "1"
    $got = Stream $NEW $Q8 $prompt "" $t
    $ok = ($got -eq $want)
    if (-not $ok) { $fail++ }
    Say ("  Q8_0   p41  vs m3   t={0}  lanes4/par1={1}" -f $t, $(if ($ok) { "SAME" } else { "DIFFER" }))
}

$env:STOKE_ATT_LANES = $null
$env:STOKE_ATT_PAR = $null
$env:STOKE_ATT_MIN_POS = $null

if ($fail -eq 0) {
    Say "ALL ARMS BIT-IDENTICAL TO THE ARCHIVES. Speed is a separate question and"
    Say "is NOT measured here -- that needs the interleaved protocol on a quiet box."
    exit 0
}
Say "$fail ARM(S) MOVED THE STREAM -- the parallel claim is false as written. STOP."
exit 1
