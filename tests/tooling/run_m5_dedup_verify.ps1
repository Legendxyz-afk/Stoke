# M5 -- PREFIX DEDUP ON THE SERVE PATH (`bench --serve --dedup`): the
# correctness gate.
#
# WHAT THIS HAS TO PROVE. `sched.c` now calls `kv_adopt_prefix` at admission and
# starts `n_done` at what it returns, so a request whose leading whole blocks are
# already resident SKIPS those positions entirely. The claim is that the token
# stream does not move. M4 gate item [3] proved that for `stoke batch --dedup`;
# the scheduler is a NEW CALLER of the same claim, and "same claim, new caller"
# is what run_m5_entry_verify.ps1 was written for.
#
# ▶▶ THE ARMS ARE ONE BINARY, WHICH IS STRONGER THAN TWO. Dedup is a runtime
# flag, not a build flag, so A and B differ in exactly one boolean and no
# compiler difference can hide inside the comparison. (The mins gate had to build
# two binaries because its change WAS a build flag; this one does not.)
#
# ▶▶ LEG 1 IS THE SCENARIO. LEG 2 IS THE ONE WITH POWER. This is the finding
# that shaped the script, and it is worth stating first: **M0's S3 clients differ
# from one another by ONE DIGIT.** Diff `s3-fanout-c00.txt` against `c01.txt` and
# the only difference in 5,957 bytes is `[client 0]` vs `[client 1]`; S2 is the
# same. All eight clients therefore generate the SAME ids, so a byte-compare over
# S3 cannot see a dedup bug that handed client 1 client 0's blocks -- the two are
# indistinguishable by construction. Leg 1 still tests position accounting (and
# the mutant below shows it can fail), but it is BLIND to the cross-client case
# and this script asserts that blindness out loud rather than leaving a later
# reader to assume coverage it does not have.
#
# Leg 2 runs the identical comparison on `fanout-hetero-c{00..07}.txt`: S3's REAL
# shared system prefix (same `build_messages`, same 64 adoptable blocks) with
# eight genuinely different bodies. Its reference streams are asserted PAIRWISE
# DISTINCT before anything is graded -- if they ever collapse, the leg reports
# itself blind instead of passing.
#
# ▶▶ LEG 3 IS THE MUTANTS, AND THERE ARE TWO BECAUSE THE FIRST ONE CAME BACK
# SAME. A gate that has never been shown to fail is not evidence -- and this one
# was shown to be PARTLY BLIND before it was ever run in anger, which is the
# whole reason the pre-flight exists.
#
#   skip16 -- `r->n_done = adopted` -> `adopted + 16u`. It skips ONE MORE BLOCK
#     than it adopted, so 16 positions are backed (kv_reserve grows the
#     sequence) but never written. This is the failure kv.h names as the one
#     that "leaves the run alive and the output merely wrong".
#     ⚠ MEASURED 2026-08-25: **it produces the SAME ids as the no-dedup arm.**
#     Not a surprise once looked at: 16 unwritten positions have K = 0, so
#     softmax weights them exp(0) = 1 against exp(q.k) for 1,222 real ones --
#     a perturbation far under what greedy argmax resolves. It is kept and run
#     because that NEGATIVE result is the finding: **the byte-compare cannot see
#     a 16-position skip, so the `kv_len(kv) != adopted` guard in sched.c and the
#     sim's position ledger are load-bearing, not decoration.** It is recorded,
#     never required to differ.
#     (This is M4's recorded lesson recurring: a fixture-level token comparison
#     could not see prefix-dedup damage there either, which is why the batch
#     driver got the same guard.)
#
#   kv -- zeroes the K/V under the WHOLE adopted prefix, every layer, every kv
#     head, at admission. It asks the question the gate actually rests on: "if
#     the adopted blocks held the wrong K/V, would this comparison notice?" It
#     MUST DIFFER. A SAME from it means the byte-compare is powerless over
#     adopted-KV correctness and the run must be stopped.
#
# Both are built from a deliberately edited working tree into their own
# BUILDDIRs and the edits are then reverted, so -- as in the mins gate -- they
# are EXEMPT from the newest-source check: the source they were built from does
# not exist on disk any more. What protects them is stronger than a timestamp:
# a stale or accidentally-identical build reports SAME, and for `kv` that fails
# the gate.
#
# ▶▶ LEG 4 PUTS THE SERVE PATH ON THE PARITY LINEAGE FOR THE FIRST TIME. Legs
# 1-3 compare two arms of one new binary; the project's standing rule is that an
# arm is graded against an ARCHIVE, because agreement among new arms is also
# consistent with all of them being wrong together. `bench --serve` has never
# been compared to anything outside itself. Leg 4 runs ONE sequence through the
# scheduler and byte-compares its ids against the m4.5 archive's `stoke run` on
# the same prompt. It grades the NO-DEDUP arm -- the referent legs 1-2 rest on.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_dedup_verify.ps1
#   ... -Quick   pre-flight only (2 seqs, n 4, leg 1 + mutant), NOT a gate.

param(
    [int]$NGen = 24,
    [int]$Seqs = 8,
    [string]$Stamp = "",
    [switch]$Quick
)

$ErrorActionPreference = 'Continue'
$root   = "M:\New Drive\Desktop\Programming\stoke"
$gr     = "$root\docs\superpowers\gate-reports"
if (-not $Stamp) { $Stamp = "{0}-m5-dedup-verify" -f (Get-Date -Format "yyyy-MM-dd") }
$log    = "$gr\$Stamp.txt"
$report = "$gr\$Stamp-report.md"
$status = "$gr\$Stamp.status.json"

$Q4     = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$NEW    = "$root\build\win\stoke.exe"
# ▶ WHICH MUTANT IS REQUIRED TO DIFFER IS A PROPERTY OF THE LEG, NOT OF THE
# MUTANT. On a leg whose answer does not depend on its prefix, NOTHING done to
# that prefix can show -- demanding a DIFFER there would be demanding that the
# model behave in a way this prompt cannot. `kv` is required only on a leg with
# power (see $legs.blind), and on the S3 scenario both mutants are recorded.
$MUTANTS = @(
    @{ name = "kv";     exe = "$root\build\win-dedup-mutkv\stoke.exe" },
    @{ name = "skip16"; exe = "$root\build\win-dedup-mutant\stoke.exe" }
)
$archiveQ4 = "$root\build\stoke-m45-maddubs-f3b7181a.exe"   # pre-lanes, pre-pool
$pdir   = "$root\tests\data\bench_prompts"
$tmp    = "$env:TEMP\stoke-dedup"

# PRE-FLIGHT RUNS THE LEG WITH POWER, not the cheapest one. The first version of
# this script pre-flighted on S3 and came back "gate has NO POWER" -- correctly,
# but for a reason that is a property of those prompts and could never be fixed
# by re-running them.
if ($Quick) { $Seqs = 2; $NGen = 8 }

New-Item -ItemType Directory -Force -Path $tmp | Out-Null
foreach ($p in @($Q4, $NEW, $pdir, $archiveQ4)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "STOP: missing $p" }
}

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding ascii $log
    Write-Output $line
}

# ---- BANK AND RESUME, and it is not optimism about crashes.
# The first full run of this gate was killed 17 minutes in, with leg s3 complete
# and hetero's two arms complete, and every one of those units had to be paid
# for again. Same shape the soak runner already carries ("an unattended job that
# banks nothing until it completes"). A unit is (id, hash-of-its-ids-file); on
# resume a unit whose ids file is still on disk is trusted and skipped.
$bank = "$gr\$Stamp-bank.jsonl"
function Bank($obj) { ($obj | ConvertTo-Json -Compress) | Out-File -Append -Encoding ascii $bank }
function BankedUnit($id) {
    if (-not (Test-Path $bank)) { return $null }
    $hit = $null
    foreach ($l in (Get-Content $bank)) {
        if (-not $l.Trim()) { continue }
        try { $o = $l | ConvertFrom-Json } catch { continue }
        # LAST wins: a re-run of a unit supersedes an earlier record of it.
        if ($o.id -eq $id -and $o.hash -and (Test-Path -LiteralPath $o.ids)) { $hit = $o }
    }
    return $hit
}

# Start-Process, not `&`: PowerShell wraps a native command's stderr at console
# width, and a long binary path has broken a regex here before. stdout goes to a
# file for the same reason.
function RunCapture($exe, $argList, $outPath) {
    if (-not (Test-Path -LiteralPath $exe)) {
        throw "STOP: no such binary '$exe' -- the instrument is broken, not the engine"
    }
    $errPath = "$tmp\err.txt"
    $p = Start-Process -FilePath $exe -ArgumentList $argList `
                       -RedirectStandardError $errPath `
                       -RedirectStandardOutput $outPath `
                       -NoNewWindow -PassThru -Wait
    if ($p.ExitCode -ne 0) {
        Say "  !! exit $($p.ExitCode): $(Get-Content $errPath -Raw)"
        throw "STOP: '$exe' exited $($p.ExitCode)"
    }
    return (Get-Content $outPath -Raw)
}

function PromptArgs($stem, $n) {
    $a = @()
    for ($i = 0; $i -lt $n; $i++) {
        $a += "-f"; $a += ('"' + ("$pdir\$stem{0:d2}.txt" -f $i) + '"')
    }
    return $a
}

# One serve run. `idsPath` is the artifact that gets compared; the returned text
# is the stats block, which carries the anti-vacuity lines.
# ▶ --raw IS NOT A DETAIL HERE. The expected adoption comes from the fixture's
# manifest, which the exporter computes with `stoke tokenize` -- and that command
# tokenizes RAW (tok_encode, no chat template). Without --raw the driver wraps
# every prompt in ChatML, which shifts the shared run by the template's tokens
# and makes the exact expectation wrong by a block: measured 1168 adopted
# against a manifest that said 1152, and the gate correctly refused it. Raw on
# both sides means one tokenization, not two that agree by luck. (The model
# still answers from the shared records in raw mode -- checked before this line
# was written, on c00 and c04.)
function ServeRun($exe, $stem, $seqs, $dedup, $idsPath) {
    $a = @("bench", "--serve", ('"' + $Q4 + '"')) + (PromptArgs $stem $seqs) +
         @("--raw", "--seqs", "$seqs", "--ctx", "1600", "-n", "$NGen",
           "--threads", "8", "--ids", ('"' + $idsPath + '"'))
    if ($dedup) { $a += "--dedup" }
    return RunCapture $exe $a "$tmp\stdout.txt"
}

function Grp($text, $re, $i) {
    $m = [regex]::Match($text, $re)
    if ($m.Success) { return $m.Groups[$i].Value }
    return $null
}

# Split an --ids file into per-client blocks, so "the clients differ" and "the
# arms agree" are separate questions answered from the same artifact.
function ClientBlocks($path) {
    $raw = Get-Content -LiteralPath $path -Raw
    $parts = [regex]::Split($raw, '(?m)^== seq \d+ ==\r?\n')
    return @($parts | Where-Object { $_.Trim() })
}

function Same($a, $b) {
    if (-not (Test-Path -LiteralPath $a) -or -not (Test-Path -LiteralPath $b)) { return $false }
    return ((Get-FileHash -LiteralPath $a).Hash -eq (Get-FileHash -LiteralPath $b).Hash)
}

# One serve unit, banked. Returns @{ ids; hash; out; secs; resumed }.
# `out` is the stats block and is EMPTY on a resumed unit -- the checks that read
# it (adoption count, row counts) run only on a fresh unit, and the banked record
# carries their results instead.
function Unit($id, $exe, $stem, $seqs, $dedup) {
    # ▶ THE SHAPE IS PART OF THE KEY. -Quick runs the same leg names at 2 seqs
    # x 8 generated; without this a full 8x24 gate would RESUME a pre-flight
    # unit and grade it as the gate. A resume that returns the wrong experiment
    # is worse than no resume.
    $id = "$id-s${seqs}n${NGen}"
    $ids = "$tmp\$id-ids.txt"
    $b = BankedUnit $id
    if ($b) {
        Say "   $id resumed from bank (hash $($b.hash.Substring(0,8)))"
        return @{ ids = $b.ids; hash = $b.hash; out = ""; secs = $b.secs
                  adopted = $b.adopted; denom = $b.denom; rows = $b.rows
                  resumed = $true }
    }
    $t0 = Get-Date
    $out = ServeRun $exe $stem $seqs $dedup $ids
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $h = (Get-FileHash -LiteralPath $ids).Hash
    $rec = @{ id = $id; ids = $ids; hash = $h; secs = $secs
              adopted = (Grp $out 'dedup: (\d+) of (\d+) admitted' 1)
              denom = (Grp $out 'dedup: (\d+) of (\d+) admitted' 2)
              rows = (Grp $out 'rows (\d+)' 1); utc = (Get-Date -Format s) }
    Bank $rec
    return @{ ids = $ids; hash = $h; out = $out; secs = $secs
              adopted = $rec.adopted; denom = $rec.denom; rows = $rec.rows
              resumed = $false }
}

Set-Location $root

# THE BINARY UNDER TEST MUST BE NEWER THAN THE SOURCE. A gate that grades a stale
# artifact manufactures confidence. (The mutant is exempt -- see the header.)
$newest = (Get-ChildItem -Path "$root\src" -Recurse -Include *.c, *.h |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1)
if ((Get-Item -LiteralPath $NEW).LastWriteTime -lt $newest.LastWriteTime) {
    throw ("STOP: $NEW is OLDER than $($newest.Name) -- rebuild before gating")
}

Say "=== M5 dedup-on-serve verification, branch $(git branch --show-current)"
Say "    commit $(git rev-parse --short HEAD)  seqs=$Seqs  n=$NGen"
Say "    new=$NEW  mutant=$MUT"
if ($Quick) { Say "    *** -Quick: PRE-FLIGHT ONLY (2 seqs, n 8, hetero leg). NOT a gate. ***" }
@{ name = $Stamp; started = (Get-Date -Format s); seqs = $Seqs; n = $NGen
   legs = $(if ($Quick) { 2 } else { 4 }) } | ConvertTo-Json -Compress |
    Out-File -Encoding ascii $status

$fail = 0
$rows = @()

# ---------------------------------------------------------------- legs 1 + 2
# How much each set SHOULD adopt is read from the manifest the exporter writes
# with the engine's own tokenizer -- not recomputed here. A runner that derives
# it in its own arithmetic is a second implementation of the thing under test,
# and the first version of this script proved it: `(seqs-1) x shared_prefix`
# called a green run red, because adoption is PAIRWISE and clients share more
# with each other than all of them share together.
$legs = @(
    @{ name = "s3";     stem = "s3-fanout-c";     blind = $true
       manifest = "$pdir\s3-fanout-manifest.json" },
    @{ name = "hetero"; stem = "fanout-hetero-c"; blind = $false
       manifest = "$pdir\fanout-hetero-manifest.json" }
)
if ($Quick) { $legs = @($legs[1]) }   # hetero -- see the -Quick note above

foreach ($leg in $legs) {
    $lg = $leg.name

    Say "-- leg $lg`: arm A (no dedup), $Seqs seqs x $NGen gen"
    $uA = Unit "$lg-A" $NEW $leg.stem $Seqs $false
    $A = $uA.ids
    $rowsA = $uA.rows
    $secA = $uA.secs
    Say "   A: $rowsA rows, ${secA}s"

    # THE REFERENCE ARM'S OWN CLIENTS, checked before anything is graded against
    # it. On S3 they are expected to be IDENTICAL (see the header) and that is
    # recorded as a limit of the leg; on hetero an identical pair means the
    # fixture stopped being a fixture and the leg must not be read as evidence.
    $blocks = ClientBlocks $A
    $distinct = @($blocks | Sort-Object -Unique).Count
    if ($leg.blind) {
        Say "   A: $distinct distinct streams over $($blocks.Count) clients -- EXPECTED 1"
        Say "      (M0's clients differ by one digit; this leg cannot see a cross-client error)"
    } else {
        if ($distinct -ne $blocks.Count) {
            Say "   !! A: only $distinct distinct streams over $($blocks.Count) clients"
            Say "   !! THIS LEG IS BLIND -- not evidence. STOP."
            $fail++
        } else {
            Say "   A: all $distinct client streams distinct -- the leg has power"
        }
    }

    Say "-- leg $lg`: arm B (--dedup)"
    $uB = Unit "$lg-B" $NEW $leg.stem $Seqs $true
    $B = $uB.ids
    $ded = $uB.adopted; $den = $uB.denom; $rowsB = $uB.rows; $secB = $uB.secs
    Say "   B: $rowsB rows, adopted $ded of $den, ${secB}s"

    # ANTI-VACUITY: "dedup was enabled" and "dedup fired" are different claims.
    # Expected exactly (seqs-1) x 1024 -- one prefill at a time (S1) means every
    # client but the first finds the whole 64-block prefix resident.
    if (-not (Test-Path -LiteralPath $leg.manifest)) {
        throw "STOP: no manifest at $($leg.manifest) -- run export_bench_prompts.py --verify"
    }
    $man = Get-Content -LiteralPath $leg.manifest -Raw | ConvertFrom-Json
    $want = [int]$man.expected_adopted[$Seqs]
    if ([int]$ded -ne $want) {
        Say "   !! adoption was $ded, expected $want -- the comparison below is not the one intended"
        $fail++
    }
    if ([int]$rowsB -ge [int]$rowsA) {
        Say "   !! B carried no fewer rows than A -- adoption did no work"
        $fail++
    }

    $ok = ($uA.hash -eq $uB.hash)
    if (-not $ok) { $fail++ }
    Say ("   RESULT leg {0}: dedup vs no-dedup = {1}" -f $lg,
         $(if ($ok) { "SAME (bit-identical)" } else { "DIFFER" }))

    # ---- the mutants, on this leg
    $mutRes = @{}
    foreach ($mu in $MUTANTS) {
        $tag = $mu.name
        $must = (-not $leg.blind) -and ($tag -eq "kv")
        if (-not (Test-Path -LiteralPath $mu.exe)) {
            Say "   !! NO '$tag' MUTANT at $($mu.exe)"
            if ($must) { Say "   !! the SAMEs above are unproven. Not a gate."; $fail++ }
            $mutRes[$tag] = "missing"
            continue
        }
        Say "-- leg $lg`: mutant '$tag'$(if ($must) { ' (MUST differ)' } else { ' (recorded)' })"
        # A mutant that DIES is also a mutant that did not reproduce the arm's
        # stream -- skip16 can leave a client with a zero-length prefill chunk,
        # which the serve driver correctly reports as livelock. Recorded as
        # `aborted`, which is a DIFFER for the purpose it serves here.
        $aborted = $false
        $uM = $null
        try { $uM = Unit "$lg-M-$tag" $mu.exe $leg.stem $Seqs $true }
        catch { $aborted = $true; Say "   '$tag' ABORTED: $($_.Exception.Message)" }
        $differs = $aborted -or ($uM.hash -ne $uA.hash)
        $mutRes[$tag] = $(if ($aborted) { "aborted" } elseif ($differs) { "DIFFERS" } else { "same" })
        if ($differs) {
            Say "   '$tag' does not reproduce A ($($mutRes[$tag])) -- the comparison can fail"
        } elseif ($must) {
            Say "   !! '$tag' PRODUCED THE SAME STREAM AS A -- this gate has NO POWER. STOP."
            $fail++
        } else {
            Say "   '$tag' is invisible on this leg -- recorded, not a failure"
            Say "      (the sched.c residency guard and the sim ledger are what cover it)"
        }
    }

    $rows += @{ leg = $lg; same = $ok; adopted = $ded; denom = $den
                rows_a = $rowsA; rows_b = $rowsB; secs_a = $secA; secs_b = $secB
                distinct = $distinct; clients = $blocks.Count; mut = $mutRes }
}

# ------------------------------------------------------------------- leg 4
# The serve path against an ARCHIVE. One sequence, no dedup, --raw (so no chat
# template stands between the two drivers), and the archive's own `stoke run` on
# the same file. The --ids file carries a `== seq 0 ==` header that `run` does
# not print, so the header line is stripped before comparing -- the ids and the
# decoded text below them are compared verbatim.
$lineage = $null
if (-not $Quick) {
    # SINGLE QUOTES. A backtick in a double-quoted PowerShell string is the
    # ESCAPE character, so "`stoke run`" would emit "s" and a CARRIAGE RETURN --
    # run_m5_bars.ps1 published a corrupted citation exactly this way.
    Say '-- leg 4: serve (1 seq, no dedup) vs the m4.5 archive''s `stoke run`'
    $prompt = "$root\tests\data\parity\p00.txt"
    $sIds = "$tmp\lineage-serve.txt"
    # ▶ --ctx IS NOT OPTIONAL HERE, and leaving it out cost this leg a run.
    # Without it the driver sizes the pool as longest_prompt + n_gen -- for p00
    # that is 47 positions = 3 blocks -- and then `sched_submit` correctly
    # refuses the request, because admission needs blocks(prompt) + W with W's
    # FLOOR of 8. A one-sequence run still has to clear the decode watermark.
    $a = @("bench", "--serve", ('"' + $Q4 + '"'), "-f", ('"' + $prompt + '"'),
           "--raw", "--seqs", "1", "--ctx", "256", "-n", "$NGen",
           "--threads", "6", "--ids", ('"' + $sIds + '"'))
    $null = RunCapture $NEW $a "$tmp\stdout.txt"
    $env:STOKE_PREFILL_CHUNK = "1"
    $runOut = RunCapture $archiveQ4 @("run", ('"' + $Q4 + '"'), "--raw", "-n", "$NGen",
                                      "-f", ('"' + $prompt + '"'), "--threads", "6") `
                         "$tmp\archive-run.txt"
    $env:STOKE_PREFILL_CHUNK = $null
    # ▶ COMPARED AS ID LISTS, NOT AS TEXT, AND THE REASON IS EOS. `stoke run`
    # stops at the end-of-turn token; `bench --serve` submits eos_id = -1 and
    # generates the full -n. So the archive's stream is a PREFIX of the serve
    # one whenever the model finishes early, and a whole-text compare would call
    # that a DIFFER. The archive's own count decides how many ids are compared,
    # and how many that was is printed rather than assumed.
    function IdsOf($text) {
        $out = @()
        foreach ($l in ($text -split "`r?`n")) {
            if ($l -match '^\d+$') { $out += $l } elseif ($l -match '^---') { break }
        }
        return $out
    }
    $serveIds = IdsOf ((Get-Content -LiteralPath $sIds -Raw) -replace '(?m)^== seq 0 ==\r?\n', '')
    $archIds  = IdsOf $runOut
    $n = $archIds.Count
    $lineage = ($n -gt 0 -and $serveIds.Count -ge $n -and
                (($serveIds[0..($n - 1)] -join ",") -eq ($archIds -join ",")))
    Say "   archive produced $n ids (of $NGen asked; fewer means it hit EOS), serve $($serveIds.Count)"
    if (-not $lineage) {
        $fail++
        Say "   !! serve and the archive DIFFER on p00 -- read this before legs 1-3:"
        Say "      the referent those legs rest on is not on the parity lineage."
        for ($i = 0; $i -lt [math]::Min($serveIds.Count, $n); $i++) {
            if ($serveIds[$i] -ne $archIds[$i]) {
                Say ("      first difference at id {0}: serve {1} vs archive {2}" -f
                     $i, $serveIds[$i], $archIds[$i])
                break
            }
        }
    } else {
        Say '   serve == the archive''s `stoke run` on p00: SAME'
    }
}

# ------------------------------------------------------------------ report
$lines = @()
$lines += "# M5 -- prefix dedup on the serve path: the correctness gate"
$lines += ""
$lines += "Run ``$Stamp``, $Seqs seqs x $NGen generated, ctx 1600, threads 8, Q4_K_M."
$lines += "Raw log ``$Stamp.txt``. Arms are ONE binary differing in one runtime flag."
$lines += ""
$lines += "| leg | prompts | adopted | rows A -> B | dedup vs no-dedup | mutant kv | mutant skip16 | client streams |"
$lines += "|---|---|---|---|---|---|---|---|"
foreach ($r in $rows) {
    $lines += ("| {0} | {1} clients | {2} of {3} | {4} -> {5} | {6} | {7} | {8} | {9} distinct of {1} |" -f
        $r.leg, $r.clients, $r.adopted, $r.denom, $r.rows_a, $r.rows_b,
        $(if ($r.same) { "**SAME**" } else { "**DIFFER**" }),
        $r.mut["kv"], $r.mut["skip16"], $r.distinct)
}
$lines += ""
if ($null -ne $lineage) {
    $lines += ("**Leg 4 (lineage).** serve, 1 seq, no dedup, vs the m4.5 archive's ``stoke run`` on p00: **{0}**." -f
               $(if ($lineage) { "SAME" } else { "DIFFER" }))
    $lines += ""
}
$lines += "**What leg ``s3`` cannot see.** M0's S3 clients differ by ONE DIGIT"
$lines += "(``[client 0]`` vs ``[client 1]``) inside an otherwise byte-identical word list,"
$lines += "so all eight generate the same ids and a cross-client adoption error is"
$lines += "indistinguishable from correct behaviour there. That is why leg ``hetero`` exists:"
$lines += "S3's real shared system prefix (same 64 adoptable blocks) with eight genuinely"
$lines += "different bodies, whose reference streams are asserted pairwise distinct."
$lines += ""
$lines += "**The mutants.** ``kv`` zeroes the K/V under the whole adopted prefix at admission"
$lines += "(every layer, every kv head) and asks the question the gate rests on: would this"
$lines += "comparison notice if the adopted blocks held the wrong K/V? It MUST differ."
$lines += "``skip16`` skips one block MORE than it adopted, over positions that are backed but"
$lines += "never written -- the failure ``kv.h`` names as the one that leaves the run alive and"
$lines += "the output merely wrong. **It is INVISIBLE here** (K = 0 gets softmax weight exp(0)"
$lines += "against exp(q.k) for 1,222 real positions), so it is recorded rather than required:"
$lines += "what covers that failure is the ``kv_len(kv) != adopted`` guard in ``sched.c`` and the"
$lines += "sim's position ledger, both of which DO catch it."
$lines += ""
if ($fail -eq 0) {
    $lines += "## VERDICT: GREEN -- adoption on the serve path does not move the stream."
} else {
    $lines += "## VERDICT: $fail CHECK(S) FAILED -- read the raw log."
}
$lines | Out-File -Encoding utf8 $report
Say "report: $report"

if ($fail -eq 0) {
    if ($Quick) { Say "PRE-FLIGHT GREEN -- run without -Quick for the gate evidence."; exit 0 }
    Say "GREEN: dedup on the serve path is bit-identical to no dedup, on a set that could tell."
    exit 0
}
Say "$fail CHECK(S) FAILED -- STOP."
exit 1
