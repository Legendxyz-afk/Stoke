#!/bin/sh
# The 24 h soak's RSS rule, graded before it grades anything.
#
# tests/tooling/soak_rss_rule.sh is gate [3]'s memory verdict (pre-registered
# 2026-09-23, docs/superpowers/design-notes/2026-09-23-m7-soak-rss-rule.md).
# A 24 h run is one reading a day, so a grader that is wrong costs a day and a
# grader that is SILENTLY wrong costs the gate. Each case below is a synthetic
# sidecar series (the sidecar's nine columns, integer MiB, one row per 5 s --
# exactly what run_m7_soak.sh writes) whose right verdict is known, and each
# names the grader change that would flip it:
#
#   flat, noise          no growth, with and without +-3 MiB jitter -> FLAT
#   nonarena_leak        RssAnon outside the arena +1 MiB/h all day  -> CLIMBING
#                        (drop the non-arena branch and this passes)
#   mapped_leak          VmData +1 MiB/h, nothing touched            -> CLIMBING
#                        (drop the VmData branch and this passes)
#   slow                 both +0.5 MiB/h = +3 MiB over 6 h, under one block
#   fill14 / fill20      +18 MiB/h (the 1 h baseline's slope) that stops at
#                        14 h / at 20 h: a bounded fill passes only once it
#                        has finished before the window opens
#   step_early/late      a +21 MiB step (the baseline had one) at 10 h / 21 h
#   mapped_zero_tail     mapped_leak plus the row the sidecar writes when the
#                        server is gone between `kill -0` and /proc (all
#                        zeros) -- unguarded, that row drags VmData's fit
#                        negative by ~20,000 MiB and a leak reads FLAT
#   arena_readfail       flat plus five rows where the pagemap read failed
#                        (arena_rss prints "0 0") -- unguarded, non-arena
#                        jumps by the whole arena and a flat run CLIMBS
#                        (both rows are dropped by the same arena-0 check)
#   truncated_tail       flat plus a last line cut to "864" (the sidecar
#                        killed mid-printf) -- read as a row, the series
#                        "ends" at t=864 and the gate run VOIDs
#   died10, short        the series stops at 10 h of 24 / a 1 h run -> VOID
#   sparse               one row a minute inside the window           -> VOID
#   badheader            a renamed column                             -> VOID
#   baseline_1h          the real 2026-09-18 1 h series               -> VOID
#
# VOID is not a pass: the soak treats anything but FLAT as a failed gate.
#
# Usage: sh tests/tooling/test_soak_rss_rule.sh
#   SOAK_RSS_RULE=<path> points it at another grader (the mutation check).
set -u

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
rule=${SOAK_RSS_RULE:-$here/soak_rss_rule.sh}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

DAY=86400
BLOCK=4718592    # one KV block of the soak's model: 2 x 36 x 16 x 8 x 128 x 4 B

# gen <profile> <series_secs>: a synthetic sidecar series on stdout.
gen() {
    awk -v prof="$1" -v T="$2" 'BEGIN {
        srand(7)
        if (prof == "badheader")
            print "t_s,rss_mb,anon_mb,file_mb,VmData,hwm_mb,gens,arena_mb,arena_swapped_mb"
        else
            print "t_s,rss_mb,anon_mb,file_mb,data_mb,hwm_mb,gens,arena_mb,arena_swapped_mb"
        end = T + 40
        for (t = 0; t <= end; t += 5) {
            h = t / 3600
            if (prof == "sparse" && t >= end - 21600 && t % 60 != 0) continue
            gn = 0; gd = 0
            if (prof == "nonarena_leak") gn = h
            else if (prof == "mapped_leak" || prof == "mapped_zero_tail") gd = h
            else if (prof == "slow") { gn = 0.5 * h; gd = gn }
            else if (prof == "fill14") { gn = 18 * (h < 14 ? h : 14); gd = gn }
            else if (prof == "fill20") { gn = 18 * (h < 20 ? h : 20); gd = gn }
            else if (prof == "step_early") { gn = (h >= 10) ? 21 : 0; gd = gn }
            else if (prof == "step_late") { gn = (h >= 21) ? 21 : 0; gd = gn }
            else if (prof == "noise") gn = 6 * rand() - 3
            ar = 1147 * h / 2; if (ar > 1147) ar = 1147; ar = int(ar)
            anon = ar + int(160 + gn)
            dm = int(14681400 + gd)
            col = ar
            if (prof == "arena_readfail" && t > end - 1800 && t % 360 == 0) col = 0
            printf "%d,%d,%d,2392,%d,%d,%d,%d,0\n", t, anon + 2392, anon, dm, anon + 2392, int(t / 5), col
        }
        if (prof == "mapped_zero_tail")
            printf "%d,0,0,0,0,0,%d,0,0\n", end + 5, int(T / 5)
        if (prof == "truncated_tail")
            printf "864"
    }'
}

pass=0; total=0
# check <case> <csv> <run_secs> <expected prefix> <expected rc>
check() {
    total=$((total + 1))
    out=$(sh "$rule" "$2" "$3" "$BLOCK" 2>&1)
    rc=$?
    case "$out" in
        "$4"*) if [ "$rc" -eq "$5" ]; then
                   pass=$((pass + 1)); echo "  ok   $1: $out"; return
               fi ;;
    esac
    echo "  FAIL $1: expected rc=$5 and a line starting '$4', got rc=$rc: $out"
}

for p in flat noise nonarena_leak mapped_leak slow fill14 fill20 step_early \
         step_late mapped_zero_tail arena_readfail truncated_tail sparse badheader; do
    gen "$p" "$DAY" > "$tmp/$p.csv"
done
gen flat 36000 > "$tmp/died10.csv"
gen flat 3600 > "$tmp/short.csv"

echo "test_soak_rss_rule: grader $rule"
check flat             "$tmp/flat.csv"             "$DAY" "FLAT ("                        0
check noise            "$tmp/noise.csv"            "$DAY" "FLAT ("                        0
check nonarena_leak    "$tmp/nonarena_leak.csv"    "$DAY" "CLIMBING (non-arena"           1
check mapped_leak      "$tmp/mapped_leak.csv"      "$DAY" "CLIMBING (VmData"              1
check slow             "$tmp/slow.csv"             "$DAY" "FLAT ("                        0
check fill14           "$tmp/fill14.csv"           "$DAY" "FLAT ("                        0
check fill20           "$tmp/fill20.csv"           "$DAY" "CLIMBING ("                    1
check step_early       "$tmp/step_early.csv"       "$DAY" "FLAT ("                        0
check step_late        "$tmp/step_late.csv"        "$DAY" "CLIMBING ("                    1
check mapped_zero_tail "$tmp/mapped_zero_tail.csv" "$DAY" "CLIMBING (VmData"              1
check arena_readfail   "$tmp/arena_readfail.csv"   "$DAY" "FLAT ("                        0
check truncated_tail   "$tmp/truncated_tail.csv"   "$DAY" "FLAT ("                        0
check died10           "$tmp/died10.csv"           "$DAY" "VOID (the series ends at"      2
check short            "$tmp/short.csv"            3600   "VOID (the series spans"        2
check sparse           "$tmp/sparse.csv"           "$DAY" "VOID (too sparse"              2
check badheader        "$tmp/badheader.csv"        "$DAY" "VOID (the series header"       2
check baseline_1h      "$root/docs/superpowers/gate-reports/2026-09-18-0844-m7-soak.rss.csv" \
                                                   3600   "VOID (the series spans"        2

if [ "$pass" -eq "$total" ]; then
    echo "test_soak_rss_rule: $pass/$total OK"
    exit 0
fi
echo "test_soak_rss_rule: FAILED ($((total - pass)) of $total)"
exit 1
