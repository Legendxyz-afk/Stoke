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
$tmp    = "$env:TEMP\stoke-q6k-gate$(if ($Quick) { '-quick' } else { '' })"
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
    $out = "$tmp\$tag.txt"
    $err = "$tmp\$tag-err.txt"
    if (Test-Path $out) {
        $t = Get-Content $out -Raw
        if ($t -and $t.Trim().Length -gt 0) { return $out }
    }
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
    return $out
}
function HashOf($f) { return (Get-FileHash -LiteralPath $f).Hash }

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
    $rows += @{ id = $c.id; kind = $c.kind
                refOk = ($hr -eq $ha); madOk = ($hm -eq $ha)
                mutDiff = ($hx -ne $ha); h = $ha.Substring(0, 12) }
    Say ("  {0,-18} ref {1,-6} maddubs {2,-6} mutant {3}" -f $c.id,
         $(if ($hr -eq $ha) { "MATCH" } else { "DIFFER" }),
         $(if ($hm -eq $ha) { "MATCH" } else { "DIFFER" }),
         $(if ($hx -ne $ha) { "DIFFERS (good)" } else { "SAME (kernel not reached!)" }))
}

$allRef = @($rows | Where-Object { -not $_.refOk }).Count -eq 0
$allMad = @($rows | Where-Object { -not $_.madOk }).Count -eq 0
$anyMut = @($rows | Where-Object { $_.mutDiff }).Count
$allMut = ($anyMut -eq $rows.Count)
$green = ($allRef -and $allMad -and $allMut -and -not $Quick)

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
$lines += "| case | ref vs archive | maddubs vs archive | mutant | archive sha |"
$lines += "|---|---|---|---|---|"
foreach ($r in $rows) {
    $lines += ("| ``{0}`` | {1} | {2} | {3} | ``{4}`` |" -f $r.id,
        $(if ($r.refOk) { "**MATCH**" } else { "**DIFFER**" }),
        $(if ($r.madOk) { "**MATCH**" } else { "**DIFFER**" }),
        $(if ($r.mutDiff) { "DIFFERS (good)" } else { "**SAME -- kernel never reached**" }),
        $r.h)
}
$lines += ""
$lines += "## The claims"
$lines += ""
$lines += "| # | claim | result |"
$lines += "|---|---|---|"
$lines += ("| 1 | the OLD kernel still reproduces the archive (nothing else moved) | {0} |" -f (if ($allRef) { "**PASS**" } else { "**FAIL** -- no verdict below is readable" }))
$lines += ("| 2 | the NEW kernel reproduces the archive, byte for byte | {0} |" -f (if ($allMad) { "**PASS**" } else { "**FAIL**" }))
$lines += ("| 3 | the mutant DIFFERS on every case -- so the kernel was LIVE in every one | {0} |" -f (if ($allMut) { "**PASS** ($anyMut/$($rows.Count))" } else { "**FAIL** ($anyMut/$($rows.Count)) -- a SAME case graded nothing" }))
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
Say ("VERDICT: ref {0}, maddubs {1}, mutant differs {2}/{3} -- {4}" -f
     $(if ($allRef) { "MATCH" } else { "DIFFER" }),
     $(if ($allMad) { "MATCH" } else { "DIFFER" }), $anyMut, $rows.Count,
     $(if ($Quick) { "NOT GRADED (pre-flight)" } elseif ($green) { "MAY FLIP" } else { "DO NOT FLIP" }))
