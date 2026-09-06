#!/bin/sh
# M7 GATE ITEM [3] / rung B7 -- the 24 h serving soak. PREPARED, NOT RUN.
#
# A full day of randomized hostile traffic against ONE long-lived `stoke serve`,
# with a sanitizer watching from inside. What it hunts does not present as a
# wrong answer: a use-after-free on a path only a client disconnect reaches, a
# read past a request buffer, a leak per rejected connection, and RSS that
# climbs all day. Those present as nothing at all, until they present as a
# crash in front of somebody else.
#
# ▶▶ ONE SERVER, CHUNKED DRIVER -- AND THAT SHAPE IS THE WHOLE DESIGN.
# Two recorded lessons pull in opposite directions here:
#   (a) "never write a runner that banks only at the end" -- a 24 h invocation
#       that prints once loses everything when the box kills it, and this box
#       has killed long jobs with no message more than once.
#   (b) LeakSanitizer reports at process EXIT, and an RSS SLOPE is only
#       meaningful across a process that actually lived that long.
# Chunking the SERVER would satisfy (a) and destroy (b): every restart zeroes
# the RSS and throws away the leak evidence, so a chunked-server soak is a
# 24-long sequence of 1 h soaks wearing a day's costume. So: the SERVER runs
# once for the whole window and ends ITSELF via --max-seconds (never signalled
# -- a killed server is a server whose leaks were never checked), while the
# DRIVER runs in chunks and banks each chunk as it lands.
#
# ⚠⚠ THE SCAR THAT SHAPES THE GRADING. M6's soak passed every check it had
# while NEVER ENTERING THE SAMPLER: its traffic builder hardcoded
# "temperature":0, a 400 counts as an "answered exchange", and so a soak of
# pure refusals looked green. tests/serve_asan.sh already carries the fix --
# a floor of TEN completed generations read from the SERVER'S OWN log, with the
# floor measured against the -DSTOKE_MUT_FUZZ_BAD_BODY mutant (which logged
# two by accident, so `> 0` would have passed it). This runner keeps that and
# adds the other half the roadmap asks for: the driver must report WHAT ITS
# TRAFFIC ACTUALLY CONTAINED, banked per chunk, not just that it was busy.
#
# Linux/WSL only: the Windows toolchain here has no ASan.
#
# Usage, from the repo root inside WSL:
#   sh tests/tooling/run_m7_soak.sh [hours] [chunk_min] [builddir] [seed]
#   sh tests/tooling/run_m7_soak.sh --shakedown        # 6 min, proves the rig
#
# ⚠ BEFORE THE REAL RUN: 4.1 (code freeze) must have happened. Soaking a tree
# that is about to change spends a day of box time on a binary nobody will
# ship, and the roadmap's own sequencing note says a surprise in B5 AFTER the
# soak invalidates the soak.
set -u

if [ "${1:-}" = "--shakedown" ]; then
    HOURS=0; CHUNK_MIN=2; SHAKE=1; shift
    TOTAL_SECS=360
else
    SHAKE=0
    HOURS=${1:-24}
    CHUNK_MIN=${2:-60}
    TOTAL_SECS=$((HOURS * 3600))
fi
BUILDDIR=${3:-build/linux-asan}
SEED=${4:-20260906}
CHUNK_SECS=$((CHUNK_MIN * 60))
[ "$SHAKE" -eq 1 ] && CHUNK_SECS=120

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 90
GR="$ROOT/docs/superpowers/gate-reports"
mkdir -p "$GR"
STAMP=$(date +%Y-%m-%d)-m7-soak
[ "$SHAKE" -eq 1 ] && STAMP="SHAKEDOWN-m7-soak"
BANK="$GR/$STAMP.jsonl"
LOG="$GR/$STAMP.txt"
SRVLOG="$BUILDDIR/m7_soak.srv"
SANLOG="$BUILDDIR/m7_soak.sanlog"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$BUILDDIR/stoke"
fuzz="$BUILDDIR/serve_fuzz"
port="${STOKE_M7_SOAK_PORT:-18171}"

say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
bankline() { printf '%s\n' "$1" >> "$BANK"; }

# ---- PRE-FLIGHT. Every one of these has cost this project a run. ----
say "=== M7 soak pre-flight (stamp $STAMP) ==="
for f in "$model" "$bin" "$fuzz"; do
    [ -e "$f" ] || { say "ABORT -- missing $f"; exit 1; }
done
case "$BUILDDIR" in
    *asan*) : ;;
    *) say "⚠ NOTE -- $BUILDDIR is not a sanitizer build. The soak will still" \
            "run and the RSS slope is still meaningful, but the use-after-free" \
            "and leak classes this gate exists for will NOT be watched." ;;
esac

# Memory headroom. ASan roughly doubles footprint and this box is 15.6 GB; the
# project's own Makefile records sanitizer legs being OOM-killed inside
# `make test`, and three unrelated OOMs happened here on 2026-09-06.
free_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null)
say "memory available: ${free_mb:-unknown} MB"
if [ -n "${free_mb:-}" ] && [ "$free_mb" -lt 6000 ] && [ "$SHAKE" -eq 0 ]; then
    say "ABORT -- under 6 GB available; a 24 h ASan soak will be OOM-killed and"
    say "         an OOM-killed server is a server whose leaks were never checked."
    exit 1
fi
if command -v curl > /dev/null 2>&1; then : ; else say "ABORT -- curl missing"; exit 1; fi

rm -f "$SRVLOG" "$SANLOG".* "$BANK" "$LOG" 2>/dev/null

# ---- ONE server, for the whole window. Slack covers the last in-flight
# exchange only: --max-seconds starts when it begins LISTENING, so the model
# load sits outside it, which is what "serve for N seconds" should mean.
srv_secs=$((TOTAL_SECS + 120))
say "starting server: ${srv_secs}s budget, port $port, build $BUILDDIR"
ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:abort_on_error=0:log_path=$SANLOG" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
"$bin" serve "$model" --port "$port" --ctx 1024 --seqs 4 --threads 4 \
    --max-seconds "$srv_secs" -v > "$SRVLOG" 2>&1 &
srv_pid=$!

if ! curl -sS --retry 600 --retry-delay 1 --retry-connrefused \
        "http://127.0.0.1:$port/health" > /dev/null; then
    say "ABORT -- the server never answered /health"
    kill "$srv_pid" 2>/dev/null
    exit 1
fi
say "server up (pid $srv_pid)"

rss_mb() { awk '/VmRSS/ {print int($2/1024)}' "/proc/$1/status" 2>/dev/null; }
rss0=$(rss_mb "$srv_pid")
say "RSS at t=0: ${rss0:-?} MB"

# ---- chunked driver ----
n_chunks=$(( (TOTAL_SECS + CHUNK_SECS - 1) / CHUNK_SECS ))
[ "$n_chunks" -lt 1 ] && n_chunks=1
say "driver: $n_chunks chunk(s) of ${CHUNK_SECS}s"

manifest=$(mktemp)
find tests/data/http_corpus -type f 2>/dev/null | head -200 > "$manifest"
n_corpus=$(wc -l < "$manifest" | tr -d ' ')
say "corpus: $n_corpus files"

i=1
while [ "$i" -le "$n_chunks" ]; do
    if ! kill -0 "$srv_pid" 2>/dev/null; then
        say "⚠ server exited early, before chunk $i -- stopping the driver"
        break
    fi
    cseed=$((SEED + i))
    out=$("$fuzz" 127.0.0.1 "$port" "$CHUNK_SECS" "$manifest" 4 "$cseed" 20000 3 2>&1)
    frc=$?
    rss=$(rss_mb "$srv_pid")
    gens=$(grep -cE "stoke serve: (chat|stream) id=" "$SRVLOG" 2>/dev/null || echo 0)
    mix=$(printf '%s' "$out" | tr '\n' ' ' | sed 's/"/\\"/g')
    bankline "{\"chunk\":$i,\"seed\":$cseed,\"secs\":$CHUNK_SECS,\"fuzz_rc\":$frc,\"rss_mb\":${rss:-null},\"gens_cumulative\":$gens,\"mix\":\"$mix\"}"
    say "chunk $i/$n_chunks rc=$frc rss=${rss:-?}MB gens=$gens"
    say "  $(printf '%s' "$out" | grep -E 'serve_fuzz:' | tr '\n' ' ')"
    i=$((i + 1))
done
rm -f "$manifest"

say "waiting for the server to end ITSELF (never signalled -- this IS the leak check)"
wait "$srv_pid"; srv_rc=$?
say "server exited $srv_rc"

# ---- verdict ----
fail=0
if [ "$srv_rc" -ne 0 ]; then
    say "FAIL -- server exited $srv_rc (a sanitizer abort exits non-zero; so does a crash)"
    tail -40 "$SRVLOG" | sed 's/^/  /' | tee -a "$LOG"
    fail=1
fi
if ls "$SANLOG".* > /dev/null 2>&1; then
    say "FAIL -- the sanitizer reported:"
    for f in "$SANLOG".*; do sed 's/^/  /' "$f" | tee -a "$LOG"; done
    fail=1
fi
n_gen=$(grep -cE "stoke serve: (chat|stream) id=" "$SRVLOG" 2>/dev/null || echo 0)
if [ "${n_gen:-0}" -lt 10 ]; then
    say "FAIL -- only ${n_gen:-0} completed generations in the SERVER's own log."
    say "        Traffic that is all refusals looks like a clean soak to every"
    say "        other check here (M6's soak passed exactly that way)."
    fail=1
fi

# RSS slope: first banked chunk against the last. This is the number a chunked
# SERVER could never produce, and it is why the server is not chunked.
r_first=$(head -1 "$BANK" 2>/dev/null | sed -n 's/.*"rss_mb":\([0-9]*\).*/\1/p')
r_last=$(tail -1 "$BANK" 2>/dev/null | sed -n 's/.*"rss_mb":\([0-9]*\).*/\1/p')
if [ -n "${r_first:-}" ] && [ -n "${r_last:-}" ]; then
    say "RSS: ${r_first} MB -> ${r_last} MB over $n_chunks chunk(s)"
    say "⚠ RSS is REPORTED, not gated: a threshold picked now would be picked"
    say "  without a baseline, and this project does not fit bands to results."
    say "  The first full run establishes the baseline; the SECOND can gate it."
fi

say "traffic mix per chunk is banked in $(basename "$BANK") -- the roadmap"
say "requires the driver to report what its traffic ACTUALLY CONTAINED, and"
say "'it was busy' is not that."

if [ "$fail" -ne 0 ]; then
    say "=== M7 soak: FAILED ==="
    exit 1
fi
say "=== M7 soak: OK ($n_chunks chunks, ${TOTAL_SECS}s, $n_gen generations, no sanitizer reports) ==="
exit 0
