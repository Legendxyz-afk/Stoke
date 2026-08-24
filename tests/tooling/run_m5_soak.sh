#!/bin/sh
# M5 gate item [5] -- the 1 h randomized soak, ASan + TSan, chunked and resumable.
#
# WHAT IT RUNS. `tests/test_sched.c`'s soak: a deterministic seed walk (seed 1,
# 2, 3, ... each generating a different trace over the real scheduler and the
# real block pool) with EVERY invariant asserted after EVERY step -- M4's I1-I12
# via bp_check plus M5's S1-S8. Gate item [1] banked 1,000,130 steps this way;
# [5] asks for an hour of it under sanitizers.
#
# WHY CHUNKED. The soak prints only when it finishes, so a single hour-long
# invocation banks nothing until the end and loses everything if it is killed --
# the shape that cost this project 76 minutes of probe compute on 2026-08-12.
# Each chunk here appends its result and its NEXT SEED to a bank file, so the
# run is resumable and its progress is legible while it is still going.
#
# WHY THE SEED IS THE WHOLE INPUT. A failing chunk prints the offending seed, and
# `test_sched <steps> <seed>` reproduces it in seconds. That is the entire reason
# the seed walk is deterministic rather than time- or /dev/urandom-seeded: a soak
# whose failures cannot be replayed is a soak that reports "something broke".
#
# WHY ASan AND TSan SEPARATELY, NEVER TOGETHER. They are mutually exclusive at
# link time, and each answers a different question: ASan/UBSan the memory and
# arithmetic questions, TSan the concurrency one. Run sequentially so neither
# competes with the other for the box -- these are correctness runs, so box state
# governs their DURATION, never their validity.
#
# setarch -R: TSan reserves fixed shadow-memory regions at startup and dies with
# "unexpected memory mapping" when ASLR places the binary where it wants to map
# (see the Makefile's note). Disabling randomization weakens nothing TSan checks.
#
# Usage, from the repo root inside WSL:
#   sh tests/tooling/run_m5_soak.sh [seconds_per_leg] [steps_per_chunk]
# Defaults: 1800 s per leg (2 legs = the 1 h gate), 200000 steps per chunk.

set -u

SECS=${1:-1800}
CHUNK=${2:-200000}
# LABEL keeps a pre-flight's artifacts out of the gate run's. The runner RESUMES
# from its bank file, so a short shakeout sharing the gate's stamp would both
# pollute the evidence and make the gate start mid-walk at a seed no one chose.
LABEL=${3:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
GR="$ROOT/docs/superpowers/gate-reports"
if [ -n "$LABEL" ]; then STAMP="$LABEL-m5-soak"; else STAMP=$(date +%Y-%m-%d)-m5-soak; fi
mkdir -p "$GR"

# ONE WRITER PER FILE. The playbook's ops lesson from M0: two writers sharing a
# log handle interleave and corrupt on Windows, and this bank is read back by the
# resume logic. Each leg owns its own bank; the summary is written once at the end.
for LEG in asan tsan; do
    BIN="$ROOT/build/linux-$LEG/test_sched"
    BANK="$GR/$STAMP-$LEG.jsonl"
    LOG="$GR/$STAMP-$LEG.txt"

    if [ ! -x "$BIN" ]; then
        echo "STOP: no $BIN -- build it with: make SAN=$LEG build/linux-$LEG/test_sched"
        exit 1
    fi

    # RESUME: the next seed is whatever the last banked chunk recorded. An absent
    # or unparsable bank starts at 1 rather than at 0 -- test_sched refuses seed 0
    # precisely so a mis-parse cannot silently masquerade as "start from the top".
    SEED=1
    if [ -f "$BANK" ]; then
        LAST=$(grep -o '"next":[0-9]*' "$BANK" | tail -1 | cut -d: -f2)
        if [ -n "${LAST:-}" ] && [ "$LAST" -ge 1 ] 2>/dev/null; then
            SEED=$LAST
            echo "resuming $LEG from seed $SEED"
        fi
    fi

    SETARCH=""
    if command -v setarch >/dev/null 2>&1; then SETARCH="setarch $(uname -m) -R"; fi

    T0=$(date +%s)
    CHUNKS=0
    STEPS=0
    FAILED=0
    echo "=== $LEG soak: ${SECS}s target, $CHUNK steps/chunk, from seed $SEED ===" >>"$LOG"
    while :; do
        NOW=$(date +%s)
        ELAPSED=$((NOW - T0))
        [ "$ELAPSED" -ge "$SECS" ] && break

        OUT=$($SETARCH "$BIN" "$CHUNK" "$SEED" 2>&1)
        RC=$?
        NEXT=$(printf '%s\n' "$OUT" | grep -o 'next=[0-9]*' | tail -1 | cut -d= -f2)
        DID=$(printf '%s\n' "$OUT" | grep -o 'soak: [0-9]* steps' | tail -1 | awk '{print $2}')
        PRE=$(printf '%s\n' "$OUT" | grep -o '[0-9]* preemptions' | tail -1 | awk '{print $1}')

        # A CHUNK THAT PRODUCED NO PARSABLE PROGRESS IS A FAILURE, NOT A ZERO.
        # A silent crash (sanitizer abort, OOM) leaves rc!=0 and no `next=`, and
        # treating that as "0 steps, carry on" is how a runner reports an hour of
        # green over a binary that died in the first minute.
        if [ "$RC" -ne 0 ] || [ -z "${NEXT:-}" ]; then
            printf '%s\n' "$OUT" >>"$LOG"
            echo "{\"leg\":\"$LEG\",\"kind\":\"fail\",\"seed\":$SEED,\"rc\":$RC,\"elapsed\":$ELAPSED}" >>"$BANK"
            echo "FAIL: $LEG chunk from seed $SEED (rc=$RC) -- see $LOG"
            FAILED=1
            break
        fi

        CHUNKS=$((CHUNKS + 1))
        STEPS=$((STEPS + ${DID:-0}))
        echo "{\"leg\":\"$LEG\",\"kind\":\"chunk\",\"seed\":$SEED,\"next\":$NEXT,\"steps\":${DID:-0},\"preemptions\":${PRE:-0},\"elapsed\":$ELAPSED}" >>"$BANK"
        echo "  $LEG chunk $CHUNKS: seeds [$SEED,$NEXT) ${DID:-0} steps ${PRE:-0} preemptions (${ELAPSED}s)" | tee -a "$LOG"
        SEED=$NEXT
    done

    T1=$(date +%s)
    if [ "$FAILED" -eq 1 ]; then
        echo "$LEG: STOPPED RED after $((T1 - T0))s, $CHUNKS chunks, $STEPS steps" | tee -a "$LOG"
        exit 1
    fi
    echo "$LEG: GREEN -- $((T1 - T0))s, $CHUNKS chunks, $STEPS steps, seeds up to $SEED" | tee -a "$LOG"
done

echo "SOAK GREEN on BOTH legs. Neither is evidence for the other: ASan/UBSan"
echo "answers the memory and arithmetic questions, TSan the concurrency one."
