# =====================================================================
# THE STREAM GATE FOR THE vpmaddubsw Q6_K KERNEL — what must pass before the
# default flips.
#
# ▶▶ WHY A UNIT TEST IS NOT ENOUGH, AND THIS IS THE PROJECT'S STANDING RULE.
# `tests/test_simd.c` already asserts the new twin bit-identical to the
# reference twin AND to the solo kernel over 600 random tensors. That grades the
# KERNEL. It does not grade the ENGINE: the twin could be perfect and the
# dispatch in `vec.c` could route the wrong way, or route correctly and be
# reached from a path nobody checked. And an arm is graded against an ARCHIVE
# rather than against another arm of the same build, because agreement among new
# arms is also consistent with all of them being wrong together.
#
# So every case below is byte-compared to a binary built BEFORE this kernel
# existed: `build/stoke-m45-gate-fddd12cd.exe`, the m4.5 gate archive, whose
# streams the M3/M4/M4.5 gates already certified.
#
# ▶▶ THE MUTANT IS ALSO THE REACHABILITY PROOF, and that is why it is cheap.
# A case where the Q6_K batched kernel is never reached would compare EQUAL to
# the archive no matter what this kernel does -- a green cell that tests
# nothing, which is the exact shape of failure this project has banked before.
# `build/win-mut-q6k` swaps the two scale-group halves of `scv[c]`; if it
# DIFFERS on a case, the kernel was live in that case. One arm, two jobs.
#
# ⚠ WHY `stoke run` REACHES A *BATCHED* KERNEL AT ALL, since it decodes one
# sequence: E3 chunks PREFILL through `model_forward_batch` (STOKE_PREFILL_CHUNK
# 128) and `vec.c` tiles at KERNELS_BATCH_TILE 16, so `run` enters `matvec_batch`
# at n_seq=16 and then the chunk remainder. That correction was recorded
# 2026-08-24 and it is what makes the archives -- which only speak `run` --
# gradeable for a batched kernel at all.
#
# ▶ BOTH ARMS ARE GRADED, not just the new one. `ref` must match the archive too:
# if it does not, something OTHER than this kernel moved and no verdict here is
# readable.
#
# ⚠⚠ THIS RUNNER HAS NEVER BEEN RUN. It was written 2026-08-27 as the box was
# going down. Treat its first sitting as a pre-flight: on this project a runner
# banked as "written and desk-reviewed" has arrived with seven defects, two of
# them fatal.
#
# ▶ WHAT *WAS* CHECKED BEFORE IT WAS BANKED, cheaply and without a model load:
#   - it PARSES (`[Parser]::ParseFile`), which a desk review is not;
#   - every path it names exists -- both binaries, the archive, the prompts;
#   - **the ARCHIVE ACCEPTS `--threads` and `--kv` on BOTH `run` AND `batch`**,
#     which its one-line usage summary does NOT list and which every case here
#     depends on. Probed by pointing it at a nonexistent model: it got past arg
#     parsing and failed on the FILE, so the flags were accepted. That is the
#     difference between a first sitting that measures and one that dies on
#     case 1 -- and the usage string would have talked you out of it.
#
# Usage: powershell -ExecutionPolicy Bypass -File tests\tooling\run_m5_q6k_gate.ps1
#        [-NGen 24] [-Quick]
# =====================================================================
param(
    [int]$NGen = 24,
    [switch]$Quick,
    [string]$Stamp = ""
)

$ErrorActionPreference = "Stop"
$root = "M:\New Drive\Desktop\Programming\stoke"
$exe  = "$root\build\win\stoke.exe"
$mut  = "$root\build\win-mut-q6k\stoke.exe"
$arch = "$root\build\stoke-m45-gate-fddd12cd.exe"
$mdl  = "$root\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
$par  = "$root\tests\data\parity"
$gr   = "$root\docs\superpowers\gate-reports"
if (-not $Stamp) { $Stamp = "{0}-m5-q6k-gate{1}" -f (Get-Date -Format "yyyy-MM-dd"), $(if ($Quick) { "-quick" } else { "" }) }
$log    = "$gr\$Stamp-raw.txt"
$report = "$gr\$Stamp-report.md"
# ⚠ THE BANK IS KEYED BY THE RUN'S SHAPE, NOT JUST BY -Quick. RunCase resumes any
# case whose output file already exists, and `n_gen` is not in the case id -- so a
# bank shared across -NGen values would silently serve a 24-token capture as a
# 48-token result. That is the 2026-08-26 defect ("keyed units without their
# shape") with a different variable in it, and it is fixed the same way: the shape
# goes in the key.
$tmp    = "$env:TEMP\stoke-q6k-gate-n$NGen$(if ($Quick) { '-quick' } else { '' })"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

foreach ($p in @($exe, $mut, $arch, $mdl, $par)) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}
# The archive is EXEMPT from the newest-source check by construction -- it is
# older than the source on purpose, which is the whole point of an archive. The
# two live binaries are not.
$newestSrc = (Get-ChildItem "$root\src" -Include *.c, *.h -Recurse |
              Sort-Object LastWriteTime -Descending | Select-Object -First 1)
foreach ($b in @($exe, $mut)) {
    if ((Get-Item -LiteralPath $b).LastWriteTime -lt $newestSrc.LastWriteTime) {
        throw "STOP: $b is OLDER than $($newestSrc.Name) -- rebuild before gating"
    }
}

function Say($m) {
    $line = "{0}  {1}" -f (Get-Date -Format s), $m
    $line | Out-File -Append -Encoding utf8 $log
    Write-Host $line
}

# ▶ Start-Process for RAW stdout. PowerShell re-wraps a native command's output
# at console width and the three binaries' path lengths differ -- which is
# exactly how the 2026-08-14 overnight lost a whole arm to a regex.
function RunCase($binary, $kernelEnv, $caseArgs, $tag) {
    $out  = "$tmp\$tag.txt"
    $err  = "$tmp\$tag-err.txt"
    $done = "$tmp\$tag.done"
    # ⚠ RESUME ON A SENTINEL, NEVER ON "THE FILE LOOKS NON-EMPTY". This matrix is
    # ~15 min of model loads and will be run in sittings, so the bank WILL be
    # resumed. A case killed mid-generation leaves a PARTIAL capture, which is
    # non-empty and would be served from cache as a finished one -- a silently
    # truncated stream that compares DIFFER and reads as a failed kernel. The
    # sentinel is written only after exit-0 and a non-empty capture, so an
    # interrupted case simply re-runs.
    if ((Test-Path $done) -and (Test-Path $out)) {
        $t = Get-Content $out -Raw
        if ($t -and $t.Trim().Length -gt 0) { return $out }
    }
    Remove-Item -Force -ErrorAction SilentlyContinue $done
    if ($null -eq $kernelEnv) { $env:STOKE_Q6K_KERNEL = $null }
    else                      { $env:STOKE_Q6K_KERNEL = $kernelEnv }
    $p = Start-Process -FilePath $binary -ArgumentList $caseArgs `
         -RedirectStandardOutput $out -RedirectStandardError $err `
         -NoNewWindow -PassThru -Wait
    $env:STOKE_Q6K_KERNEL = $null
    if ($p.ExitCode -ne 0) {
        Say ("  !! $tag exited {0}: {1}" -f $p.ExitCode, (Get-Content $err -Raw))
        throw "STOP: a broken case must stop, never report a DIFFER"
    }
    $txt = Get-Content $out -Raw
    if (-not $txt -or $txt.Trim().Length -eq 0) {
        throw "STOP: $tag produced NO OUTPUT -- an empty capture compares unequal to everything and would read as a DIFFER"
    }
    New-Item -ItemType File -Force -Path $done | Out-Null
    return $out
}
function HashOf($f) { return (Get-FileHash -LiteralPath $f).Hash }

# ▶▶ THE MUTANT'S BANNER IS A *DIRECT* REACHABILITY PROOF, AND THE ORIGINAL DRAFT
# THREW IT AWAY. `q6k_select()` is called from ONE place -- vec.c:374, inside
# `simd && n_seq > 1` and `!is_q4` -- so this line on stderr means the batched
# Q6_K AVX2 kernel was DISPATCHED in this case. Nothing else emits it.
#
# Without it, "mutant SAME as archive" has two causes the report cannot tell
# apart, and it announces the wrong one:
#   - the kernel was never reached          -> the cell grades nothing;
#   - the kernel ran WRONG and the stream did not care -> the FIXTURE is blind.
# The second is not hypothetical on this project: zeroing 1024 positions of
# adopted K/V across 36 layers once produced byte-identical ids. Reading stderr
# separates them for free -- the file was already being written.
#
# Returns $null (not $false) when the capture is missing, because "no evidence"
# and "evidence of absence" must not be graded the same way.
function MutBannerSeen($tag) {
    $e = "$tmp\$tag-err.txt"
    if (-not (Test-Path $e)) { return $null }
    $t = Get-Content $e -Raw
    if (-not $t) { return $false }
    return [bool]($t -match "STOKE_MUT_Q6K BUILD")
}

# ---- the case matrix --------------------------------------------------------
# `run` covers the chunked-prefill path into matvec_batch; `batch --seqs N`
# covers the decode-side batched path. kv f32 and q8 are different arithmetic
# classes and a kernel change has to clear both.
$cases = @()
foreach ($pr in @("p00", "p41")) {
    foreach ($kv in @("f32", "q8")) {
        foreach ($th in @(2, 6)) {
            $cases += @{ id = "run-$pr-$kv-t$th"; kind = "run"
                         args = @("run", ('"' + $mdl + '"'), "--raw", "-n", "$NGen",
                                  "-f", ('"' + "$par\$pr.txt" + '"'),
                                  "--threads", "$th", "--kv", $kv) }
        }
    }
}
foreach ($sq in @(2, 8, 16)) {
    $cases += @{ id = "batch-s$sq"; kind = "batch"
                 args = @("batch", ('"' + $mdl + '"'), "--raw", "-n", "$NGen",
                          "-f", ('"' + "$par\p00.txt" + '"'),
                          "--seqs", "$sq", "--ctx", "1024", "--threads", "6",
                          "--kv", "f32") }
}
if ($Quick) { $cases = @($cases[0], $cases[8]) }

Say "Q6_K stream gate. stamp $Stamp"
Say "exe $exe"
Say "mut $mut   *** DELIBERATELY WRONG: scale lanes swapped ***"
Say "archive $arch  (predates this kernel -- the referent)"
Say "exe commit $(git -C $root rev-parse --short HEAD)"
if ($Quick) { Say "*** -Quick: $($cases.Count) cases, NOT a gate ***" }

$rows = @()
foreach ($c in $cases) {
    $fa = RunCase $arch $null      $c.args "arch-$($c.id)"
    $fr = RunCase $exe  "ref"      $c.args "ref-$($c.id)"
    $fm = RunCase $exe  "maddubs"  $c.args "mad-$($c.id)"
    $fx = RunCase $mut  "maddubs"  $c.args "mut-$($c.id)"
    $ha = HashOf $fa; $hr = HashOf $fr; $hm = HashOf $fm; $hx = HashOf $fx
    $live = MutBannerSeen "mut-$($c.id)"
    $rows += @{ id = $c.id; kind = $c.kind
                refOk = ($hr -eq $ha); madOk = ($hm -eq $ha)
                mutDiff = ($hx -ne $ha); mutLive = $live; h = $ha.Substring(0, 12) }
    Say ("  {0,-18} ref {1,-6} maddubs {2,-6} mutant {3}  dispatched {4}" -f $c.id,
         $(if ($hr -eq $ha) { "MATCH" } else { "DIFFER" }),
         $(if ($hm -eq $ha) { "MATCH" } else { "DIFFER" }),
         $(if ($hx -ne $ha) { "DIFFERS (good)" }
           elseif ($live -eq $true) { "SAME *BUT LIVE* -- FIXTURE IS BLIND" }
           else { "SAME (kernel not reached!)" }),
         $(if ($null -eq $live) { "unknown" } elseif ($live) { "YES" } else { "no" }))
}

$allRef = @($rows | Where-Object { -not $_.refOk }).Count -eq 0
$allMad = @($rows | Where-Object { -not $_.madOk }).Count -eq 0
$anyMut = @($rows | Where-Object { $_.mutDiff }).Count
$allMut = ($anyMut -eq $rows.Count)
$nLive   = @($rows | Where-Object { $_.mutLive -eq $true }).Count
$allLive = ($nLive -eq $rows.Count)
# ⚠ EVERY FLAG THE VERDICT READS IS ASSIGNED HERE, ABOVE THE CLAIM BLOCK THAT
# USES IT. On 2026-08-27 a report came out with eight PASSes and a verdict of
# "NOT DEMONSTRATED" because one flag was assigned below its use and read $null.
# $allLive joins the AND: mutDiff without a banner is a CONTRADICTION (output
# moved but the kernel was never dispatched), and a contradiction must not flip
# a default.
$green = ($allRef -and $allMad -and $allMut -and $allLive -and -not $Quick)

$lines = @()
$lines += "# Q6_K ``vpmaddubsw`` kernel -- the stream gate"
$lines += ""
if ($Quick) {
    $lines += "> !! **-Quick PRE-FLIGHT, NOT A GATE** -- $($cases.Count) of the full matrix's cases."
    $lines += ""
}
$lines += "Run ``$Stamp``, $($rows.Count) cases, every one byte-compared against"
$lines += "``stoke-m45-gate-fddd12cd.exe`` -- a binary built BEFORE this kernel existed, whose"
$lines += "streams the M3/M4/M4.5 gates already certified. **An arm is graded against an"
$lines += "ARCHIVE, never against another arm of the same build**, because agreement among"
$lines += "new arms is also consistent with all of them being wrong together."
$lines += ""
$lines += "| case | ref vs archive | maddubs vs archive | mutant | kernel dispatched | archive sha |"
$lines += "|---|---|---|---|---|---|"
foreach ($r in $rows) {
    $lines += ("| ``{0}`` | {1} | {2} | {3} | {4} | ``{5}`` |" -f $r.id,
        $(if ($r.refOk) { "**MATCH**" } else { "**DIFFER**" }),
        $(if ($r.madOk) { "**MATCH**" } else { "**DIFFER**" }),
        $(if ($r.mutDiff) { "DIFFERS (good)" }
          elseif ($r.mutLive -eq $true) { "**SAME -- but the kernel WAS live: the FIXTURE is blind**" }
          else { "**SAME -- kernel never reached**" }),
        $(if ($null -eq $r.mutLive) { "*unknown -- no stderr capture*" }
          elseif ($r.mutLive) { "yes (banner)" } else { "**no**" }),
        $r.h)
}
$lines += ""
$lines += "## The claims"
$lines += ""
# ⚠⚠ `-f $(if ...)`, NOT `-f (if ...)`. THE FOUR LINES BELOW KILLED THIS RUNNER ON
# ITS FIRST SITTING, 2026-08-27 -- AFTER every case had been measured and banked,
# at report assembly. Inside bare parentheses PowerShell parses `if` as a COMMAND
# NAME, so the file PARSES CLEAN and dies at runtime with "The term 'if' is not
# recognized". `[Parser]::ParseFile` passed on it twice.
#
# So the standing pre-flight has a hole with a known shape: a parse check proves
# the file is executable, NOT that its statements resolve. It cannot see a bad
# command name, a missing cmdlet, or a $null flag. Only running does that -- and
# the cheap way to run an expensive gate is exactly what saved this one: the
# measurement banks per case, so the fix cost a re-assembly and no compute.
$lines += "| # | claim | result |"
$lines += "|---|---|---|"
$lines += ("| 1 | the OLD kernel still reproduces the archive (nothing else moved) | {0} |" -f $(if ($allRef) { "**PASS**" } else { "**FAIL** -- no verdict below is readable" }))
$lines += ("| 2 | the NEW kernel reproduces the archive, byte for byte | {0} |" -f $(if ($allMad) { "**PASS**" } else { "**FAIL**" }))
$lines += ("| 3 | the mutant DIFFERS on every case -- the error reaches the STREAM | {0} |" -f $(if ($allMut) { "**PASS** ($anyMut/$($rows.Count))" } else { "**FAIL** ($anyMut/$($rows.Count)) -- see claim 4 for which kind of nothing that is" }))
$lines += ("| 4 | the batched Q6_K kernel was DISPATCHED in every case (mutant banner) | {0} |" -f $(if ($allLive) { "**PASS** ($nLive/$($rows.Count))" } else { "**FAIL** ($nLive/$($rows.Count))" }))
$lines += ""
$lines += "**Claims 3 and 4 are different questions and the draft of this runner asked only"
$lines += "the first.** Claim 4 reads the mutant's stderr banner, which ``q6k_select()`` emits"
$lines += "from exactly one call site -- ``vec.c:374``, inside ``simd && n_seq > 1`` and"
$lines += "``!is_q4`` -- so it means the batched Q6_K AVX2 kernel was *dispatched*, whatever"
$lines += "the output did afterwards. Without it, a ``SAME`` mutant cell has two causes that"
$lines += "the table cannot separate and the report would have announced the wrong one:"
$lines += "the kernel was never reached (the cell grades nothing), or the kernel ran WRONG"
$lines += "and the stream did not care (**the fixture is blind**). The second is not"
$lines += "hypothetical here -- zeroing 1024 positions of adopted K/V across 36 layers once"
$lines += "produced byte-identical ids on this project. 3 PASS + 4 PASS is the only"
$lines += "combination in which claims 1 and 2 carry their full weight."
$lines += ""
$lines += ("## VERDICT: {0}" -f $(if ($Quick) { "**NOT GRADED -- pre-flight**" } elseif ($green) { "**the default may flip**" } else { "**DO NOT FLIP**" }))
$lines += ""
$lines += "**Claim 3 is why 1 and 2 mean anything.** A case that never reaches the Q6_K"
$lines += "batched kernel compares EQUAL to the archive whatever the kernel does -- a green"
$lines += "cell that tests nothing. The mutant swaps the two scale-group halves of"
$lines += "``scv[c]``, which is the mistake this formulation invites: Q4_K's eight scales line"
$lines += "up with its eight 32-byte chunks exactly, while Q6_K has SIXTEEN, so a chunk"
$lines += "spans TWO groups and the halves must match the halves ``vpmaddubsw`` produced."
$lines += "It leaves the run alive and merely wrong -- the class ``kv.h`` names as dangerous."
$lines += ""
$lines += "``stoke run`` reaches a BATCHED kernel because E3 chunks PREFILL through"
$lines += "``model_forward_batch`` and ``vec.c`` tiles at 16, so ``run`` enters ``matvec_batch``"
$lines += "at n_seq=16 then the remainder -- the 2026-08-24 correction, and what makes the"
$lines += "archives (which only speak ``run``) gradeable for this at all."
$lines | Out-File -Encoding utf8 $report
Say "report: $report"
Say ("VERDICT: ref {0}, maddubs {1}, mutant differs {2}/{3}, dispatched {4}/{3} -- {5}" -f
     $(if ($allRef) { "MATCH" } else { "DIFFER" }),
     $(if ($allMad) { "MATCH" } else { "DIFFER" }), $anyMut, $rows.Count, $nLive,
     $(if ($Quick) { "NOT GRADED (pre-flight)" } elseif ($green) { "MAY FLIP" } else { "DO NOT FLIP" }))
