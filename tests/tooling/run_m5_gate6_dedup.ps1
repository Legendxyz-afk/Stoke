# =====================================================================
# M5 GATE ITEM [6], ON THE SHIPPING DEFAULT -- PREEMPT-LOSSLESS WITH PREFIX
# DEDUP ON, ON A FORCED-PREEMPTION REAL-MODEL TRACE.
#
# Gate [6] was met on 2026-08-26 with `--no-dedup`, and its own report named
# what it did not cover:
#
#   "**The dedup path.** With dedup on, admission still reserves the whole
#    prompt while adoption makes the residency SHARED, so `reserved` exceeds
#    what the sequences occupy and the window above closes -- measured: at
#    pools 84 and 88 the clients serialised and re-prefilled (rows 1,349 ->
#    4,805) without one preemption. [...] **On the real model it is OWED**, and
#    since dedup is now the serve DEFAULT that is the more important path."
#
# This runner pays that. It is NOT a re-run of [6] and does not re-open it:
# [6] is met as worded. This asks the same question of the path that ships.
#
# ▶▶ WHY THE WINDOW CLOSED, AND WHAT ACTUALLY REOPENS IT.
# Preemption needs a pool big enough for k requests to be ADMITTED and too
# small for them to FINISH. With dedup those two are measured on DIFFERENT
# quantities, and that asymmetry is the whole difficulty:
#
#   ADMISSION is blind to adoption. admit_if_possible checks
#   `free >= blocks(target) + W` BEFORE the Kv exists, so what adoption will
#   find is not yet knowable (sched.c says so, deliberately). Each client is
#   therefore admitted as if it paid for its whole prompt: ~74 blocks here.
#
#   RESIDENCY is not. The 2nd..kth clients adopt 72 of those 74 blocks by
#   refcount, so each costs the pool ~2 blocks, not 74.
#
# So the 08-26 attempt squeezed the pool to what the sequences OCCUPY (84, 88)
# -- and admission, reading the un-adopted size, then admitted ONE at a time.
# Serialisation, preemptions 0. The pool was below the wrong number.
#
# ▶ THE LEVER IS PRIVATE GROWTH, AND IT IS GUARD #2 OF [6] GENERALISED.
# [6] found "growth must exceed W*16 = 128 tokens". With dedup the shared
# prefix is paid ONCE while every generated token is PRIVATE, so what has to
# clear the admission slack is the growth summed over clients. Eight clients
# generating 300 tokens each add ~150 blocks of private residency on top of an
# 86-block admitted footprint -- and THAT is a window a pool can sit inside.
# Widening k, not shrinking the pool, is what reopens it.
#
#   admission ceiling (all k admitted, none preempted):
#       max over j of [ occupancy before j + blocks(p_j) + W_j ]   ~= 175
#   residency floor  (all k finish):  measured by arm R's own peak  ~= 233
#   POOL 200 SITS INSIDE, and claim 2 below checks that BY MEASUREMENT.
#
# ▶▶ AND THAT IS THE OTHER REASON THIS RUNNER EXISTS: `peak blocks used` IS A
# NEW INSTRUMENT (sched.h SchedStats, printed by the serve driver). [6] had to
# ARGUE that pool 170 was below what the workload needed. This MEASURES it: run
# the shape once at the derived pool, read the peak, and every pool below it is
# forced to preempt by construction rather than by arithmetic. The peak is
# sampled INSIDE the step at maximum demand -- after every row is backed and
# before any retirement -- because a driver polling at step boundaries misses
# the peak by the whole residency of the first sequence to retire.
#
# ▶ THE FIXTURE IS `fanout-hetero` FOR THE REASON ADDENDUM 13 ESTABLISHED:
# S3's eight clients differ by ONE DIGIT and its continuation does not depend
# on its prefix, so a byte-compare over it cannot see a replay OR an adoption
# defect at all. Hetero has pairwise-distinct streams over a load-bearing
# shared prefix (common_prefix_all 1162 >= 1152 = 72 whole blocks).
#
# ▶▶ TWO MUTANTS, BECAUSE THE TWO FAILURES ARE DIFFERENT.
#   MP  -DSTOKE_MUT_PREEMPT   replays one row too FEW  -> the REPLAY is wrong.
#   MK  -DSTOKE_MUT_ADOPT_KV  zeroes the adopted prefix's K/V at admission
#                             -> the ADOPTED BLOCKS are wrong.
# MP is [6]'s mutant and shows the compare sees a replay defect on THIS shape.
# MK is this gate's own, and it asks the question the dedup path rests on: if
# the shared blocks held the wrong K/V, would a byte-compare notice? A SAME
# from MK means the comparison is powerless over adopted-KV correctness -- not
# hypothetical, since the `skip16` mutant was MEASURED SAME on 2026-08-25
# (16 unwritten positions score exp(0) against a thousand real keys and greedy
# argmax does not resolve it).
# Both are built from the SHIPPING SOURCE through EXTRA_CFLAGS, both print a
# banner on every run, and nothing defines either macro.
#
# ▶ PRE-REGISTERED, BEFORE THE FIRST ARM RAN (arithmetic in the run's own
# shape, so it cannot be re-fitted afterwards):
#   P1  R's peak occupancy lands in
#         [ SUM blocks(p_i + NGen - (k-1)), SUM blocks(p_i + NGen) ] - adopted/16
#       using R's OWN measured adoption. For this fixture at 8x300: [232, 233].
#   P2  R adopts at least (k-1)*1152 positions.
#   P3  P (pool 200) preempts at least once.
#   P4  P adopts MORE than R -- i.e. a re-admitted victim really re-adopted.
# P1/P2 are checks on the model of the system; P3/P4 are gate claims 4 and 5.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_gate6_dedup.ps1
#        [-Seqs 8] [-NGen 300] [-Pool 200] [-Threads 8] [-Ctx 1600] [-Quick]
#
# ⚠ -Quick IS A PLUMBING PRE-FLIGHT AND IT CANNOT GRADE THIS GATE. At 2 seqs
# and n 16 the private growth is ~2 blocks, so NO pool both admits both clients
# and starves them: the window is EMPTY BY ARITHMETIC, and claims 2, 4, 5, 6
# and 7 would report FAIL for a reason that has nothing to do with the code.
# (That is the 2026-08-26 gate [3] pre-flight's exact mistake -- it scored two
# predictions "FALSIFIED" whose bands were arithmetic in n_gen and N.) So
# -Quick grades ONLY what its shape can: the binaries run, the flags are
# accepted, adoption fires, and MK differs. It writes to its own tmp DIRECTORY
# under its own STAMP with SHAPE-KEYED ids -- three independent reasons the
# gate can never resume a pre-flight unit as its own.
# =====================================================================
param(
    [int]$Seqs    = 8,
    [int]$NGen    = 300,
    [int]$Pool    = 200,
    [int]$Threads = 8,
    [int]$Ctx     = 1600,
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$root  = "M:\New Drive\Desktop\Programming\stoke"
$exe   = "$root\build\win\stoke.exe"
$mutP  = "$root\build\win-mut-preempt\stoke.exe"
$mutK  = "$root\build\win-mut-adoptkv\stoke.exe"
$model = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$pdir  = "$root\tests\data\bench_prompts"
$gr    = "$root\docs\superpowers\gate-reports"

if ($Quick) { $Seqs = 2; $NGen = 16 }
$BASE  = "s${Seqs}n${NGen}c${Ctx}t${Threads}"
$Stamp = "{0}-m5-gate6-dedup{1}" -f (Get-Date -Format "yyyy-MM-dd"),
                                    $(if ($Quick) { "-quick" } else { "" })
$log    = "$gr\$Stamp-raw.txt"
$report = "$gr\$Stamp-report.md"
$tmp    = "$env:TEMP\stoke-gate6-dedup$(if ($Quick) { '-quick' } else { '' })"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

foreach ($p in @($exe, $mutP, $mutK, $model, $pdir)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}
# THE NEWEST-SOURCE CHECK COVERS ALL THREE BINARIES, mutants included: unlike
# the 2026-08-25 dedup mutants (built from an edited tree that no longer
# exists), these two are built from the shipping source through EXTRA_CFLAGS,
# so a timestamp comparison is meaningful for them and a stale mutant is a
# stoppable error rather than a silent one.
$newestSrc = (Get-ChildItem "$root\src" -Include *.c, *.h -Recurse |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1)
foreach ($b in @($exe, $mutP, $mutK)) {
    if ((Get-Item -LiteralPath $b).LastWriteTime -lt $newestSrc.LastWriteTime) {
        throw "STOP: $b is OLDER than $($newestSrc.Name) -- rebuild before gating"
    }
}

# ▶ THE FIXTURE'S OWN FACTS, READ RATHER THAN ASSUMED. Every band below is
# arithmetic in these numbers, so a regenerated fixture must move the
# predictions with it instead of silently falsifying them.
$mani = Get-Content "$pdir\fanout-hetero-manifest.json" -Raw | ConvertFrom-Json
$PTOK = @($mani.tokens)
$CPFX = [int]$mani.common_prefix_all
if ($PTOK.Count -lt $Seqs) { throw "STOP: fixture has $($PTOK.Count) clients, asked for $Seqs" }
$SHARED_BLOCKS = [math]::Floor($CPFX / 16)   # whole blocks every client shares
if ($SHARED_BLOCKS -lt 1) { throw "STOP: fixture shares $CPFX tokens -- nothing to adopt" }

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Host $line
}
function Grp($text, $re, $i) {
    $m = [regex]::Match($text, $re)
    if ($m.Success) { return [double]$m.Groups[$i].Value }
    return $null
}
function Blocks($tokens) { return [math]::Ceiling($tokens / 16.0) }
function PromptArgs($n) {
    $a = @()
    for ($i = 0; $i -lt $n; $i++) {
        $a += "-f"; $a += ('"' + ("$pdir\fanout-hetero-c{0:d2}.txt" -f $i) + '"')
    }
    return $a
}

# ▶ EVERY ARM STATES --dedup EXPLICITLY EVEN THOUGH IT IS THE DEFAULT. The
# 2026-08-26 flip's lesson was that an ABSENT flag changes meaning when a
# default moves, and three runners had encoded "off" by omission. A measuring
# site states its direction.
function Arm($id, $binary, $pool, $mustSucceed) {
    $key = "$id-$BASE-p$(if ($pool -gt 0) { $pool } else { 'd' })"
    $ids = "$tmp\$key.txt"
    $out = "$tmp\$key-stdout.txt"
    if ((Test-Path $ids) -and (Test-Path $out)) {
        $t = Get-Content $out -Raw
        if ($t -and $null -ne (Grp $t 'preemptions (\d+)' 1)) {
            Say "  $id already banked, skipping"
            return @{ ids = $ids; text = $t; ok = $true }
        }
    }
    $a = @("bench", "--serve", ('"' + $model + '"')) + (PromptArgs $Seqs) +
         @("--raw", "--seqs", "$Seqs", "--ctx", "$Ctx", "-n", "$NGen",
           "--threads", "$Threads", "--kv", "f32", "--dedup",
           "--ids", ('"' + $ids + '"'))
    if ($pool -gt 0) { $a += @("--pool-blocks", "$pool") }
    $t0 = Get-Date
    $p = Start-Process -FilePath $binary -ArgumentList $a -RedirectStandardOutput $out `
                       -RedirectStandardError "$tmp\e-$key.txt" -NoNewWindow -PassThru -Wait
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $t = ""
    if (Test-Path $out) { $t = Get-Content $out -Raw }
    $short = ($t -match '! only \d+ of \d+ expected ids')
    if ($p.ExitCode -ne 0 -or $short) {
        # A MUTANT THAT DIES OR TRUNCATES IS A MUTANT THAT DID NOT REPRODUCE THE
        # REFERENCE, which is all a mutant arm has to do -- the dedup_verify
        # runner has recorded that since M4. A REFERENCE arm that does either
        # must STOP: a broken measurement must never report a DIFFER.
        Say ("  !! $id exited {0}{1}" -f $p.ExitCode, $(if ($short) { " (short ids)" } else { "" }))
        if ($mustSucceed) {
            Say ("     stderr: {0}" -f (Get-Content "$tmp\e-$key.txt" -Raw))
            throw "STOP: $id failed and it is a reference arm"
        }
        return @{ ids = $ids; text = $t; ok = $false }
    }
    Say ("  {0,-3} preempt {1,-4} steps {2,-5} rows {3,-6} adopted {4,-6} peak {5,-4} pool {6,-8} ({7}s)" -f
         $id, (Grp $t 'preemptions (\d+)' 1), (Grp $t 'steps\s+(\d+)' 1),
         (Grp $t 'rows (\d+)' 1), (Grp $t 'dedup: (\d+) of' 1),
         (Grp $t 'peak (\d+) of' 1),
         $(if ($pool -gt 0) { $pool } else { "derived" }), $secs)
    return @{ ids = $ids; text = $t; ok = $true }
}

Say "M5 gate [6] on the SHIPPING DEFAULT -- preempt-lossless with DEDUP ON. stamp $Stamp"
Say "seqs $Seqs x n $NGen, ctx $Ctx, threads $Threads, squeezed pool $Pool, fixture fanout-hetero, dedup ON"
Say "fixture: prompts $($PTOK[0..($Seqs-1)] -join ',') tokens, common prefix $CPFX ($SHARED_BLOCKS blocks)"
Say "exe commit $(git -C $root rev-parse --short HEAD)"
if ($Quick) { Say "*** -Quick: PLUMBING PRE-FLIGHT, NOT A GATE -- claims 2,4,5,6,7 are ungradeable at this shape ***" }

# ---- the pre-registered peak band, computed BEFORE R runs -------------------
$sumHi = 0; $sumLo = 0; $sumPrompt = 0
for ($i = 0; $i -lt $Seqs; $i++) {
    $sumHi += Blocks ([int]$PTOK[$i] + $NGen)
    $sumLo += Blocks ([int]$PTOK[$i] + $NGen - ($Seqs - 1))
    $sumPrompt += Blocks ([int]$PTOK[$i])
}
Say ("P1 band before adoption: peak = [{0},{1}] - adopted/16" -f $sumLo, $sumHi)

$R = Arm "R"  $exe  0     $true    # reference: derived pool, must NOT preempt
$P = Arm "P"  $exe  $Pool $true    # pressure:  squeezed pool, MUST preempt
$MP = Arm "MP" $mutP $Pool $false  # mutant: replay off by one,  MUST differ
$MK = Arm "MK" $mutK $Pool $false  # mutant: adopted K/V zeroed, MUST differ

$preR = Grp $R.text 'preemptions (\d+)' 1
$preP = Grp $P.text 'preemptions (\d+)' 1
$adR  = Grp $R.text 'dedup: (\d+) of' 1
$adP  = Grp $P.text 'dedup: (\d+) of' 1
$denR = Grp $R.text 'dedup: \d+ of (\d+)' 1
$denP = Grp $P.text 'dedup: \d+ of (\d+)' 1
$pkR  = Grp $R.text 'peak (\d+) of' 1
$pkP  = Grp $P.text 'peak (\d+) of' 1
$hR = (Get-FileHash -LiteralPath $R.ids).Hash
$hP = (Get-FileHash -LiteralPath $P.ids).Hash
$hMP = $(if ($MP.ok -and (Test-Path $MP.ids)) { (Get-FileHash -LiteralPath $MP.ids).Hash } else { "ABORTED" })
$hMK = $(if ($MK.ok -and (Test-Path $MK.ids)) { (Get-FileHash -LiteralPath $MK.ids).Hash } else { "ABORTED" })

# P1: the band, now closed with R's OWN measured adoption.
$adBlocks = [math]::Floor($adR / 16)
$p1lo = $sumLo - $adBlocks
$p1hi = $sumHi - $adBlocks
$p1 = ($pkR -ge $p1lo -and $pkR -le $p1hi)
$p2 = ($adR -ge ($Seqs - 1) * $SHARED_BLOCKS * 16)

# ---- THE CLAIMS, EACH ABLE TO FAIL ON ITS OWN ------------------------------
$c1 = ($preR -eq 0)                 # the reference really is unpressured
$c2 = ($pkR -gt $Pool)              # ...and the squeeze is real, BY MEASUREMENT
$c3 = $p2                           # adoption really fired in the reference
$c4 = ($preP -gt 0)                 # the pressure arm really preempted
$c5 = ($adP -gt $adR)               # ...and a readmitted victim RE-ADOPTED
$c6 = ($hP -eq $hR)                 # and preemption lost nothing
$c7 = ($hMP -ne $hR)                # the compare sees a REPLAY defect
$c8 = ($hMK -ne $hR)                # the compare sees an ADOPTED-KV defect
$green = ($PRESSURED -and $c1 -and $c2 -and $c3 -and $c4 -and $c5 -and $c6 -and
          $c7 -and $c8)

function Mark($b, $yes, $no) { if ($b) { return $yes } else { return $no } }
# ▶ A PRE-FLIGHT MUST NOT GRADE WHAT ITS SHAPE CANNOT GRADE, AND A BANNER OVER A
# COLUMN OF **FAIL** IS NOT THAT. The 2026-08-26 gate [3] pre-flight printed
# "FALSIFIED" for two predictions whose bands were arithmetic in n_gen and N; the
# banner was there and the table still said FALSIFIED. Items that depend on a
# preemption window this shape cannot open render `n/a` instead.
$PRESSURED = -not $Quick
function MarkP($b, $yes, $no) {
    if (-not $PRESSURED) { return "n/a -- ungradeable at this shape" }
    if ($b) { return $yes } else { return $no }
}

$lines = @()
$lines += "# M5 gate [6] on the shipping default -- preempt-lossless with prefix dedup ON"
$lines += ""
if ($Quick) {
    $lines += "> !! **-Quick PLUMBING PRE-FLIGHT, NOT A GATE.** At $Seqs seqs x n $NGen the private"
    $lines += "> growth is ~2 blocks and NO pool both admits every client and starves it: the"
    $lines += "> preemption window is EMPTY BY ARITHMETIC. Claims 2, 4, 5, 6 and 7 are"
    $lines += "> **ungradeable at this shape**, and this report renders them ``n/a`` rather"
    $lines += "> than grading them. What it DOES check: the binaries run, the flags are"
    $lines += "> accepted, adoption fires (claim 3), the reference is reproducible (P == R"
    $lines += "> at an unsqueezed pool), and the adopted-K/V mutant has power (claim 8)."
    $lines += ""
}
$lines += "Run ``$Stamp``. Fixture **fanout-hetero** ($Seqs clients, pairwise-distinct streams"
$lines += "over a load-bearing shared prefix of $CPFX tokens = $SHARED_BLOCKS whole blocks),"
$lines += "$NGen generated, ctx $Ctx, threads $Threads, kv f32, **dedup ON (the serve default)**."
$lines += "Raw ``$Stamp-raw.txt``."
$lines += ""
$lines += "| arm | binary | pool | preempt | steps | rows | adopted | peak blocks | ids sha256 |"
$lines += "|---|---|---|---|---|---|---|---|---|"
foreach ($x in @(@{n = "R (reference)"; r = $R; b = "win";             p = "derived"; h = $hR },
                 @{n = "P (pressure)";  r = $P; b = "win";             p = "$Pool";   h = $hP },
                 @{n = "MP (mutant)";   r = $MP; b = "win-mut-preempt"; p = "$Pool";  h = $hMP },
                 @{n = "MK (mutant)";   r = $MK; b = "win-mut-adoptkv"; p = "$Pool";  h = $hMK })) {
    $lines += ("| {0} | ``{1}`` | {2} | {3} | {4} | {5} | {6} | {7} | ``{8}`` |" -f
        $x.n, $x.b, $x.p,
        (Grp $x.r.text 'preemptions (\d+)' 1), (Grp $x.r.text 'steps\s+(\d+)' 1),
        (Grp $x.r.text 'rows (\d+)' 1), (Grp $x.r.text 'dedup: (\d+) of' 1),
        (Grp $x.r.text 'peak (\d+) of' 1),
        $(if ($x.h -eq "ABORTED") { "ABORTED" } else { $x.h.Substring(0, 16) }))
}
$lines += ""
$lines += "## Pre-registered before the first arm ran"
$lines += ""
$lines += "| # | prediction | band | measured | result |"
$lines += "|---|---|---|---|---|"
$lines += ("| P1 | R's peak = SUM blocks(p_i + n) - adopted/16 | [$p1lo, $p1hi] | $pkR | {0} |" -f (Mark $p1 "**HELD**" "**FALSIFIED**"))
$lines += ("| P2 | R adopts >= (k-1) x $($SHARED_BLOCKS * 16) positions | >= $(($Seqs - 1) * $SHARED_BLOCKS * 16) | $adR | {0} |" -f (Mark $p2 "**HELD**" "**FALSIFIED**"))
$lines += ("| P3 | pool $Pool preempts at least once | > 0 | $preP | {0} |" -f (MarkP $c4 "**HELD**" "**FALSIFIED**"))
$lines += ("| P4 | a readmitted victim RE-ADOPTS (P adopts > R) | > $adR | $adP | {0} |" -f (MarkP $c5 "**HELD**" "**FALSIFIED**"))
$lines += ""
$lines += "## The eight claims, each able to fail on its own"
$lines += ""
$lines += "| # | claim | result |"
$lines += "|---|---|---|"
$lines += ("| 1 | the reference is genuinely unpressured (preemptions == 0) | {0} |" -f (Mark $c1 "**PASS** (0)" "**FAIL** ($preR)"))
$lines += ("| 2 | **the squeeze is real BY MEASUREMENT**: R's peak residency ($pkR) exceeds the squeezed pool ($Pool) | {0} |" -f (MarkP $c2 "**PASS**" "**FAIL** -- the pool was never below what the workload needs"))
$lines += ("| 3 | adoption fired in the reference (>= (k-1) x $($SHARED_BLOCKS * 16) positions) | {0} |" -f (Mark $c3 "**PASS** ($adR of $denR)" "**FAIL** ($adR)"))
$lines += ("| 4 | the pressure arm genuinely preempted (preemptions > 0) | {0} |" -f (MarkP $c4 "**PASS** ($preP)" "**FAIL** (0 -- the trace never exercised the path)"))
$lines += ("| 5 | **a readmitted victim RE-ADOPTED**: P adopted more than R | {0} |" -f (MarkP $c5 "**PASS** ($adP of $denP vs $adR of $denR)" "**FAIL** -- re-adoption after preemption was never exercised"))
$lines += ("| 6 | **preemption lost nothing**: P's stream == R's, byte for byte | {0} |" -f (MarkP $c6 "**PASS**" "**FAIL**"))
$lines += ("| 7 | the compare sees a REPLAY defect: ``MUT_PREEMPT`` differs | {0} |" -f (MarkP $c7 "**PASS**" "**FAIL** -- claim 6 proves nothing about replay"))
$lines += ("| 8 | the compare sees an ADOPTED-KV defect: ``MUT_ADOPT_KV`` differs | {0} |" -f (Mark $c8 "**PASS**" "**FAIL** -- claim 6 proves nothing about the shared blocks"))
$lines += ""
$lines += ("## VERDICT: preempt-losslessness on the DEDUP path is {0}" -f
           $(if (-not $PRESSURED) { "**NOT GRADED -- this was a pre-flight**" }
             else { Mark $green "**DEMONSTRATED**" "**NOT DEMONSTRATED**" }))
$lines += ""
$lines += "Claims 7 and 8 are why 1-6 mean anything, and there are two of them because the"
$lines += "two failures are different. ``MUT_PREEMPT`` replays one row too FEW, so the final"
$lines += "replayed row reproduces the logits of the wrong position. ``MUT_ADOPT_KV`` zeroes"
$lines += "the K/V under the whole adopted prefix at admission -- through the SHARED blocks,"
$lines += "so it damages the donor too, which is the shape a refcount bug would have. Both"
$lines += "leave the run alive and the output merely wrong: the class ``kv.h`` names as the"
$lines += "dangerous one, and the class a weaker mutant could not show -- ``skip16`` (16"
$lines += "backed-but-unwritten positions) was MEASURED SAME on 2026-08-25."
$lines += ""
$lines += "## Why the dedup path needed a different window, not a smaller pool"
$lines += ""
$lines += "Gate [6] squeezed to 170 with 4 clients and no dedup. The same reasoning at"
$lines += "8 clients WITH dedup put the pool at 84-88 and got serialisation and"
$lines += "**preemptions 0**, because admission and residency measure different things:"
$lines += ""
$lines += "- **admission is blind to adoption** -- ``admit_if_possible`` checks"
$lines += "  ``free >= blocks(target) + W`` before the ``Kv`` exists, so every client is"
$lines += "  admitted as if it paid for its whole ~74-block prompt."
$lines += "- **residency is not** -- the 2nd..kth clients adopt $SHARED_BLOCKS of those blocks by"
$lines += "  refcount and cost the pool ~2 blocks each."
$lines += ""
$lines += "A pool below the RESIDENCY is therefore also below the ADMISSION ceiling, and"
$lines += "admission just runs the clients one at a time. What reopens the window is"
$lines += "**private growth**: the shared prefix is paid once, every generated token is"
$lines += "private, so k clients x n tokens is the term that outgrows the admitted"
$lines += 'footprint. That is gate [6]''s own guard #2 ("growth must exceed W*16")'
$lines += "generalised -- widen k and n, do not shrink the pool."
$lines += ""
$lines += "| quantity | blocks |"
$lines += "|---|---|"
$lines += "| footprint once all $Seqs are admitted, before a token is generated (SUM blocks(p_i) - adopted/16) | $($sumPrompt - $adBlocks) |"
$lines += "| **residency at completion -- R's measured peak** | **$pkR** |"
$lines += "| squeezed pool | $Pool |"
$lines += "| P's own peak (it cannot exceed its pool) | $pkP |"
$lines += ""
$lines += "## What this does NOT cover"
$lines += ""
$lines += "- **Q8 KV.** Every arm is ``--kv f32``. Q8's re-adoption is arithmetically the same"
$lines += "  path, but that is an argument, not this measurement."
$lines += "- **Staggered arrivals.** All $Seqs requests are submitted at once (``--stagger 0``),"
$lines += "  which is the shape that makes the pool bind. A late arrival preempting an"
$lines += "  already-decoding cohort is a different trace."
$lines += "- **The admission asymmetry itself is UNFIXED and is a finding, not a defect"
$lines += "  fixed here.** With dedup on, admission reserves the un-adopted prompt, so under"
$lines += "  memory pressure the shipping default admits fewer sequences than the pool can"
$lines += "  hold, and readmits a victim only when ``free >= blocks(target) + W`` although it"
$lines += '  then occupies `blocks(target) - adopted`. It is deliberate (sched.c: "admitting'
$lines += '  on the un-adopted size is conservative in the only direction that matters") and'
$lines += "  it is why this trace exists at all. Whether it should hold is an M6 question."
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
Say ("VERDICT: dedup-path preempt-losslessness {0} -- R {1} preempt / peak {2}, P {3} preempt, adopted {4}->{5}, streams {6}, MP {7}, MK {8}" -f
     $(if (-not $PRESSURED) { "NOT GRADED (pre-flight)" }
       else { Mark $green "DEMONSTRATED" "NOT DEMONSTRATED" }),
     $preR, $pkR, $preP, $adR, $adP,
     (Mark $c6 "IDENTICAL" "DIFFER"), (Mark $c7 "DIFFERS (good)" "SAME (bad)"),
     (Mark $c8 "DIFFERS (good)" "SAME (bad)"))
