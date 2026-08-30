#!/bin/sh
# tests/tooling/run_a6_rate_ab.sh -- A6's RATE HALF, one unit per invocation.
#
# Usage (from the repo root, under bash/sh -- NOT PowerShell):
#     sh tests/tooling/run_a6_rate_ab.sh <block> <M5|A6|NULL> [cell]
#     sh tests/tooling/run_a6_rate_ab.sh report
#
# ▶▶▶ WHAT IS BEING ASKED, AND WHY IT IS NOT THE BARS GATE. M6's standing
# constraint is that a refactor touching the scheduler re-measures the M5 bars
# or does not land. The bars gate answers "is the ABSOLUTE rate still above the
# threshold", and its numbers are only meaningful on a box that meets its
# quiesce bar -- which this one does not today (median 14.5% _Total after
# stopping ISLC, against a bar of two consecutive reads <= 12%; the previous
# sitting VOIDED a bars attempt for the same reason and wrote down that a pass
# at exactly 12% was the tail of the distribution rather than a quiet box).
#
# The question A6 actually raises is narrower and is an ATTRIBUTION: did this
# milestone's code change the rate? That is answered by an A/B, and an A/B with
# a null arm is the one instrument that survives a noisy box -- the noise lands
# in the null and bounds the claim. So this runner does NOT re-run the bars and
# does NOT compare anything to the banked 8.960; doing either would be exactly
# the cross-sitting attribution the standing rules forbid.
#
# ▶▶ THE ARMS.
#   M5    the tag-`m5` binary, built from a worktree at bc969b9 with IDENTICAL
#         CFLAGS (checked: only CXXFLAGS and new link rules changed since).
#   A6    today's binary.
#   NULL  today's binary AGAIN -- a second process, same code. This is the
#         floor: whatever NULL-vs-A6 reports is what this harness cannot
#         resolve, at the level the comparison is made (process vs process),
#         and M5-vs-A6 means nothing except against it. Per-cell, never a max.
#
# ⚠ PRE-FLIGHT, RUN BEFORE ANY TIMING: both binaries were driven with --ids on
# this exact cell and their token streams are BYTE-IDENTICAL. Arms that produce
# different output are not doing the same work and their rates are not
# comparable -- and greedy decoding being unchanged is itself an A6 claim.
#
# ⚠ WHAT A6 ACTUALLY ADDED TO THIS PATH, from the diff rather than from memory:
# ONE perfectly-predicted branch (`if (h->sample)` in serve_step, OUTSIDE the
# token loop) plus two pointers on ServeHost. `sample_prologue` -- the mutex and
# the slot sweep -- lives in stoke.c, and `bench --serve` never enters stoke.c;
# it drives serve_host/sched directly. sched.c gained only `sched_is_live`,
# which nothing on this path calls. So the PRE-REGISTERED PREDICTION is a NULL
# RESULT: M5-vs-A6 within the null arm's own spread. It is recorded here, before
# the first unit ran, so that a surprise cannot be re-read as an expectation.
#
# ▶ SLICED ONE UNIT PER INVOCATION and banked as it lands, per the standing
# rule: long jobs on this box have been killed mid-run with no message. Each
# unit carries its OWN quiesce reading, because slicing moves the quiesce gate
# to per-invocation.
set -u

root="M:/New Drive/Desktop/Programming/stoke"
cd "$root" || exit 1

bin_a6="./build/win/stoke.exe"
# The REFERENCE arm and its bank are overridable, because this runner is now
# used for more than A6: every scheduler touch owes the same A/B against the
# tag it is trying not to move. ⚠ ONE BANK PER REFERENCE, never shared -- a
# bank keyed by nothing is how stale numbers get republished under a new commit.
bin_m5="${STOKE_AB_REF:-./build/win-m5/stoke.exe}"
model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bank="${STOKE_AB_BANK:-docs/superpowers/gate-reports/2026-08-30-a6-rate-ab-bank.jsonl}"
mkdir -p docs/superpowers/gate-reports

if [ "${1:-}" = "report" ]; then
    [ -f "$bank" ] || { echo "no bank at $bank"; exit 1; }
    awk -F'"' '{ for (i = 1; i <= NF; i++) if ($i == "arm:") ; }' /dev/null 2>/dev/null
    echo "=== A6 rate A/B -- banked units ==="
    sed 's/[{}"]//g; s/,/  /g' "$bank"
    echo
    echo "=== per-arm medians (AGGREGATE(M0), tok/s) ==="
    for a in M5 A6 NULL; do
        vals=$(grep "\"arm\":\"$a\"" "$bank" | sed 's/.*"aggm0":\([0-9.]*\).*/\1/' | sort -n)
        n=$(echo "$vals" | grep -c .)
        [ "$n" -gt 0 ] || continue
        med=$(echo "$vals" | awk -v n="$n" 'NR==int((n+1)/2){print}')
        printf '%-5s n=%d  median %s  all: %s\n' "$a" "$n" "$med" "$(echo "$vals" | tr '\n' ' ')"
    done
    exit 0
fi

block="${1:?usage: run_a6_rate_ab.sh <block> <M5|A6|NULL> [cell]}"
arm="${2:?usage: run_a6_rate_ab.sh <block> <M5|A6|NULL> [cell]}"
cell="${3:-8f32}"

case "$cell" in
    8f32)  seqs=8;  kv=f32 ;;
    16f32) seqs=16; kv=f32 ;;
    8q8)   seqs=8;  kv=q8  ;;
    16q8)  seqs=16; kv=q8  ;;
    *) echo "bad cell '$cell'"; exit 2 ;;
esac

case "$arm" in
    M5)        bin="$bin_m5" ;;
    A6|NULL)   bin="$bin_a6" ;;
    *) echo "bad arm '$arm' -- want M5, A6 or NULL"; exit 2 ;;
esac
[ -x "$bin" ] || { echo "missing binary: $bin"; exit 1; }

id="b$block-$cell-$arm"
if grep -q "\"id\":\"$id\"" "$bank" 2>/dev/null; then
    echo "$id: already banked, skipping"
    exit 0
fi

# ⚠ THE UNIT'S OWN QUIESCE READING. Two _Total samples, the gate's own
# instrument. Banked with the unit rather than checked once for the sitting:
# slicing moves the quiesce gate to per-invocation, and a run that publishes one
# flag for nine units is a run whose seventh unit can be noisy in silence.
q=$(powershell.exe -NoProfile -Command "\$a=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\").PercentProcessorTime; Start-Sleep -Seconds 2; \$b=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name='_Total'\").PercentProcessorTime; Write-Output \"\$a,\$b\"" 2>/dev/null | tr -d '\r')
q1=$(echo "$q" | cut -d, -f1)
q2=$(echo "$q" | cut -d, -f2)

sha=$(sha256sum "$bin" | cut -c1-16)

P=""
i=0
while [ "$i" -lt "$seqs" ]; do
    P="$P -f tests/data/bench_prompts/s2-512-c$(printf '%02d' $i).txt"
    i=$((i + 1))
done

out="build/win/a6ab-$id.out"
echo "=== $id  arm=$arm  bin=$sha  quiesce=$q1,$q2  $(date +%H:%M:%S) ==="
start=$(date +%s)
# --no-dedup is load-bearing: the banked bars were measured before dedup existed
# on the serve path, and S2's clients differ by one digit, so dedup would adopt
# nearly every prompt position and change the workload.
"$bin" bench --serve "$model" $P --seqs "$seqs" --ctx 704 -n 128 \
    --no-dedup --threads 8 --kv "$kv" > "$out" 2>&1
rc=$?
end=$(date +%s)

if [ "$rc" -ne 0 ]; then
    echo "$id: FAILED rc=$rc"
    tail -5 "$out"
    exit 1
fi
# ANTI-VACUITY, from the bars runner: the driver prints this when it produced
# fewer ids than asked, and every rate above it still looks tidy.
if grep -q '! only [0-9]* of [0-9]* expected ids' "$out"; then
    echo "$id: REJECTED -- serve produced fewer ids than asked"
    exit 1
fi

aggm0=$(sed -n 's/.*AGGREGATE(M0)[[:space:]]*\([0-9.]*\).*/\1/p' "$out" | head -1)
itl50=$(sed -n 's/.*ITL[[:space:]]*p50[[:space:]]*\([0-9.]*\).*/\1/p' "$out" | head -1)
steps=$(sed -n 's/.*steps[[:space:]]*\([0-9]*\).*/\1/p' "$out" | head -1)
if [ -z "$aggm0" ]; then
    echo "$id: REJECTED -- no AGGREGATE(M0) in output"
    tail -5 "$out"
    exit 1
fi

printf '{"id":"%s","block":%s,"cell":"%s","arm":"%s","bin":"%s","aggm0":%s,"itl50":%s,"steps":%s,"wall_s":%s,"q1":%s,"q2":%s}\n' \
    "$id" "$block" "$cell" "$arm" "$sha" "$aggm0" "${itl50:-0}" "${steps:-0}" \
    "$((end - start))" "${q1:-999}" "${q2:-999}" >> "$bank"

echo "$id: AGGREGATE(M0) $aggm0 tok/s   ITL p50 $itl50 ms   wall $((end - start))s   banked"
