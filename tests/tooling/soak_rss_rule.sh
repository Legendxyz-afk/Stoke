#!/bin/sh
# M7 GATE ITEM [3] -- THE 24 h SOAK'S RSS RULE. Pre-registered 2026-09-23,
# BEFORE the run (Kabir agreed the rule that day); the reasoning, the numbers
# it came from and what it cannot see are in
# docs/superpowers/design-notes/2026-09-23-m7-soak-rss-rule.md.
#
# Usage: sh tests/tooling/soak_rss_rule.sh <series.csv> <run_secs> <block_bytes> [window_secs]
#   <series.csv>  the sidecar series run_m7_soak.sh banks (<stamp>.rss.csv)
#   <run_secs>    the soak's traffic length (TOTAL_SECS: 86400 for the gate)
#   <block_bytes> one KV block of the served model (run_m7_soak.sh derives it
#                 from the model header; 4718592 for the soak's Qwen3-4B)
#   [window_secs] the graded window, default 21600 (the 24 h run's final 6 h);
#                 10800 for each run of the 4 x 6 h scheme (Kabir 2026-10-12,
#                 docs/superpowers/design-notes/2026-10-12-m7-soak-4x6h-rule.md)
# Prints ONE line and exits 0 FLAT, 1 CLIMBING, 2 VOID. VOID is not a pass.
#
# THE RULE. Over the FINAL window of the series (6 h by default), the
# least-squares growth of
#   (a) VmData                 -- mapped memory: the half RssAnon cannot see (a
#                                 leak that allocates without touching, or
#                                 pages the kernel swapped out), and
#   (b) RssAnon - arena        -- resident anonymous memory outside the KV
#                                 arena, whose own residency the sidecar reads
#                                 from pagemap,
# must each be AT MOST ONE KV BLOCK PER 6 h OF WINDOW. Growth = fitted slope x
# the window. The threshold is a RATE so that every window grades the same
# leak: one block over the 24 h run's 6 h window, half a block over a 6 h
# run's 3 h window -- ~0.75 MiB/h either way.
#
# WHY THE FINAL WINDOW AND NOT THE WHOLE RUN. The 2026-09-18 1 h baseline
# (ASan build) grew ~18-20 MiB/h in BOTH quantities for the whole hour after a
# 5-min warm-up, plus one +21 MiB step; the 09-14 probe on the PLAIN build
# read flat. A bounded fill (ASan's quarantine, an allocator warming its size
# classes) and a leak look identical while the fill lasts; only a leak is
# still climbing at hour 18. So the rule asks the one question that separates
# them -- has it STOPPED? -- and a fill that has not finished by the window
# fails, which is the conservative side: "not shown bounded" is not "bounded".
# A leak slower than one block per 6 h (~0.75 MiB/h) passes; the rule is
# blind below that, and the CSV's integer MiB puts ~1 MiB of quantization on
# the estimate, so it is not an instrument for anything smaller.
#
# ROWS THAT ARE NOT MEASUREMENTS, dropped and counted rather than fitted:
#   - a malformed row (a field missing or not a number): the sidecar killed
#     mid-printf leaves a partial last line such as "864", and read as a row
#     it would end the series at t=864 and VOID the gate run.
#   - inside the window, arena 0 with 0 swapped. Two things write that row:
#     arena_rss's python read failing (it prints "0 0"), and the race at the
#     sidecar's last `kill -0` -- the server gone before /proc was read, so
#     mem_fields and arena_rss BOTH print zeros. By the window the arena has
#     been written for hours, so neither is a reading: fitted, the first moves
#     the whole arena into "non-arena" and a flat run CLIMBS, and the second
#     drags VmData's fit ~20,000 MiB negative and a leak reads FLAT. (A
#     separate "rss or VmData is 0" guard was written first and removed: the
#     mutation check showed no row reaches it that this one does not drop.)
# VOID when the reading does not exist: fewer than 6 usable rows; a series
# that stops more than 10 min before the run's end (the sidecar or the server
# died, and its last 6 h are not the run's last 6 h); a series shorter than
# the window; fewer rows in the window than half the 5 s cadence; a header
# that is not the sidecar's nine columns (a reader that guesses columns is
# grading something else).
set -u
csv=${1:-}; run=${2:-}; bb=${3:-}; win=${4:-21600}
case "${run:-x}${bb:-x}${win:-x}" in
    *[!0-9]*) echo "VOID (usage: soak_rss_rule.sh <series.csv> <run_secs> <block_bytes> [window_secs])"; exit 2 ;;
esac
[ -r "$csv" ] || { echo "VOID (no readable series at '$csv')"; exit 2; }

awk -F, -v run="$run" -v bb="$bb" -v W="$win" '
{ sub(/\r$/, "") }
NR == 1 {
    if ($0 != "t_s,rss_mb,anon_mb,file_mb,data_mb,hwm_mb,gens,arena_mb,arena_swapped_mb") hdr_bad = 1
    next
}
hdr_bad { next }
{
    if ($1 !~ /^[0-9]+$/ || $2 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/ || $5 !~ /^[0-9]+$/ ||
        $8 !~ /^[0-9]+$/ || $9 !~ /^[0-9]+$/) { bad++; next }
    n++; t[n] = $1 + 0; a[n] = $3 + 0; d[n] = $5 + 0; r[n] = $8 + 0; s[n] = $9 + 0
}
END {
    if (hdr_bad) { print "VOID (the series header is not the sidecar nine columns -- a reader that guesses columns grades something else)"; exit 2 }
    blk = bb / 1048576
    thr = blk * W / 21600
    if (n < 6) { printf "VOID (%d usable rows -- the sidecar did not run)\n", n; exit 2 }
    tl = t[n]
    if (tl < run - 600) { printf "VOID (the series ends at t=%d s, more than 10 min before the run ends at %d s -- the sidecar or the server died early, so its final window is not the run final window)\n", tl, run; exit 2 }
    if (tl - t[1] < W) { printf "VOID (the series spans %d s, shorter than the %d s window -- nothing to grade)\n", tl - t[1], W; exit 2 }
    w0 = tl - W; m = 0; failed = 0
    for (i = 1; i <= n; i++) {
        if (t[i] < w0) continue
        if (r[i] == 0 && s[i] == 0) { failed++; continue }
        m++; wt[m] = t[i]; wd[m] = d[i]; wx[m] = a[i] - r[i]
    }
    if (m < W / 10) { printf "VOID (too sparse: %d usable rows in the final %d s, under half the sidecar 5 s cadence)\n", m, W; exit 2 }
    # Centred sums: VmData sits near 1.5e7 MiB under ASan, and uncentred
    # products of that size lose the slope in the doubles.
    for (i = 1; i <= m; i++) { st += wt[i]; sd += wd[i]; sx += wx[i] }
    mt = st / m; md = sd / m; mx = sx / m
    for (i = 1; i <= m; i++) {
        dt = wt[i] - mt; ctt += dt * dt; ctd += dt * (wd[i] - md); ctx += dt * (wx[i] - mx)
    }
    gd = ctd / ctt * W; gx = ctx / ctt * W
    detail = sprintf("final %d s: VmData %+.1f MiB, non-arena anon %+.1f MiB, least squares over %d rows t=%d..%d s, %d malformed rows and %d arena-0 rows dropped; threshold %.2f MiB = one block (%.2f MiB) per 6 h of window", W, gd, gx, m, wt[1], wt[m], bad, failed, thr, blk)
    if (gd > thr && gx > thr) { printf "CLIMBING (VmData and non-arena anon both grew more than the threshold -- %s)\n", detail; exit 1 }
    if (gd > thr)             { printf "CLIMBING (VmData grew more than the threshold: mapped growth -- %s)\n", detail; exit 1 }
    if (gx > thr)             { printf "CLIMBING (non-arena anon grew more than the threshold: touched growth -- %s)\n", detail; exit 1 }
    printf "FLAT (%s)\n", detail
    exit 0
}' "$csv"
