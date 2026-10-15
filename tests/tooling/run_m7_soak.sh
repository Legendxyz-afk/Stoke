#!/bin/sh
# M7 GATE ITEM [3] / rung B7 -- the 24 h serving soak. The 1 h baseline ran
# 2026-09-18 (OK); the 24 h has not run.
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
# ▶▶ THE RSS INSTRUMENT IS SPLIT, BECAUSE A TOTAL COULD NOT GIVE A VERDICT.
# The 2026-09-06 shakedown read VmRSS climbing 2568 -> 2761 MB in six minutes
# and the 1 h probe died at 2 of 12 chunks still climbing, and neither could
# say whether that was a leak: VmRSS is anonymous memory PLUS file-backed
# pages, and BOTH grow benignly here for a while --
#   - the model is mmap'd (gguf.c), so its pages enter RSS as they are first
#     read, up to the file size;
#   - the KV arena is xcalloc'd (kv.c) and the block pool's free list is a FIFO
#     (blockpool.c: pop head, push tail), so EVERY block in the pool is popped
#     once before any is recycled, and within a block every token row is its
#     own page -- anonymous RSS climbs one page per (block, position) first
#     written, until every row of every block has been written, then stops.
# A leak also climbs anonymous RSS and never stops. The only reading that
# separates them is the arena's OWN residency against its KNOWN size over a
# run long enough to saturate it, with mapped memory (VmData) watched
# alongside. So every sample here carries RssAnon / RssFile / VmData and the
# arena's residency separately, a sidecar samples them every 5 s for the
# whole server lifetime, and the bank keeps them per chunk. Saturation is a
# number this script computes from the model header before the run, and the
# probe mode below grades against it rather than against the result.
#
# ▶▶ THE 24 h RSS RULE (pre-registered 2026-09-23, before the run;
# docs/superpowers/design-notes/2026-09-23-m7-soak-rss-rule.md). Until then
# this script REPORTED RSS and graded nothing -- "the first full run
# establishes the baseline; the SECOND can gate it" -- and nobody wrote the
# rule after the first. Re-read, the 1 h baseline was not the flat reading it
# was banked as: VmData and the non-arena resident set BOTH climbed a steady
# ~18-20 MiB/h for the whole hour after a 5-min warm-up (plus one +21 MiB
# step), on the ASan build, where the 09-14 probe on the PLAIN build read
# flat. So the gate run (soak mode, >= 24 h) is now graded by
# tests/tooling/soak_rss_rule.sh: over the FINAL 6 h, VmData and RssAnon minus
# the arena must each grow by at most one KV block, fitted by least squares;
# anything but FLAT fails the soak. Its own test (17 cases, and a mutation
# check that every guard in it is load-bearing) runs in the pre-flight below,
# and ASan's quarantine is set to a stated size rather than left at its
# default, because the default is the likeliest cause of the climb (see the
# server launch).
#
# Linux/WSL only: the Windows toolchain here has no ASan.
#
# ▶▶ THE VM-LIFETIME HAZARD (2026-09-14; resolved from the record 09-16). The
# 1 h probe's server died FIRST, at 2.7 GB RSS, the sidecar outlived it by ten
# seconds, and two minutes later the VM shut itself down cleanly (Hyper-V
# event 18508: "by the guest operating system" -- an empty distro, then the
# idle timer). Best fit: host memory pressure (a game) inflates hv_balloon,
# the guest OOM killer picks the LARGEST process, and the largest process is
# the server whose lifetime IS the measurement. Three protections, each
# CHECKED below rather than assumed:
#   1. the server (and the witness) at oom_score_adj -1000, which the killer
#      never picks, so a CHUNK dies instead of the lifetime. Needs root --
#      run as `wsl.exe -u root -- sh tests/tooling/run_m7_soak.sh ...`. The
#      driver loop continues past a killed chunk (its fuzz rc is banked, the
#      chunk is reported as lost) and the fuzz, canary and sidecar keep the
#      default score, so the killer still has something to take.
#   2. a `dmesg -W` witness, line-flushed to /mnt/m for the whole run, so an
#      OOM kill or a balloon event is in the record even if the VM is gone by
#      morning. Grepped for the verdict.
#   3. an explicit .wslconfig (memory=, swap=), so the cap is a chosen number
#      and not the 50%-of-host default the 09-14 run had.
# And after ANY run, on the Windows side: Hyper-V-Worker-Admin event 18508
# says HOW the VM ended.
#
# Usage, from the repo root inside WSL:
#   sh tests/tooling/run_m7_soak.sh [hours] [chunk_min] [builddir] [seed]
#   sh tests/tooling/run_m7_soak.sh --shakedown              # 6 min, proves the rig
#   sh tests/tooling/run_m7_soak.sh --rss-probe [min] [builddir] [seed]
#       The RSS PROBE: a deliberately SMALL pool (--ctx 256 --seqs 4 = 64
#       blocks) so the arena saturates inside the run and the plateau is
#       observable, plain build by default (the question is not a sanitizer
#       question; the 09-06 plain build climbed the same). Default 30 min:
#       the v1 run saturated its arena at ~12-16 min under this traffic, and
#       the reading needs a post-saturation window of at least a third.
#       Pre-registered reading, v2. ⚠ v1 (run 2026-09-14 04:16, banked as
#       2026-09-14-m7-rss-probe.*) defined the walk's end as "n_blocks
#       completed generations" -- every generation pops at least one block,
#       so after n_blocks pops every block has been popped once -- and read
#       CLIMBING: post-walk RssAnon +26 MiB across halves. That reading stands
#       as banked. What it got wrong is granularity: kv.c lays a block out as
#       [16 token rows][n_head_kv vectors], and a token row is n_head_kv *
#       head_dim * 4 = 4096 B = ONE PAGE, so first-touch happens per (block,
#       position), not per block. A popped block is fully touched only once
#       some sequence has written all 16 of its rows, which the fuzz's
#       14-28-token requests do only when the block is their FIRST one. The
#       climb after the pop-walk was the remaining rows filling in: VmData
#       was flat to -3 MiB across the same halves, and RssAnon ended at 333
#       MiB against baseline 26 + heap growth 17 + arena 288 = 331. v2 stops
#       inferring the arena's residency from process totals and READS IT: the
#       arena's mapping(s) are located once at t=0 by size in
#       /proc/<pid>/smaps (asserted to add up to the arena) and their ADDRESS
#       RANGE is kept; from then on residency is read per page from
#       /proc/<pid>/pagemap over that range (the `arena_mb` column). See
#       arena_locate/arena_rss below for why not smaps' own Rss: the kernel
#       merges adjacent anonymous mappings, and a VMA found by size at t=0
#       stopped having that size a minute later. Run 2026-09-14 05:45 as
#       2026-09-14-0545-m7-rss-probe.*: FLAT -- the arena saturated at
#       exactly 288 of 288 MiB after 212 generations, the non-arena resident
#       set was 37 MiB in every chunk, VmData moved 1 MiB over 500
#       generations. SATURATION = the first sample whose arena residency
#       reaches arena - one block (the calloc'd arena starts 16 B into a
#       page, so its last row can straddle one).
#       The samples from there on are the POST-SATURATION window, split in
#       halves, and:
#         VOID     -- saturation never came, or the window is under a third
#                     of the run: too short to read. Says nothing.
#         CLIMBING -- across the halves, mean VmData up by more than one
#                     block (mapped growth: the leak signature, whatever the
#                     allocator), or mean RssAnon up by more than one block
#                     (touched growth past the arena), or the NON-arena
#                     resident at the end above baseline + mapped growth +
#                     one block (resident beyond anything mapped). Growth to
#                     find BEFORE B7.
#         FLAT     -- neither.
#       VmData is in the rule because it is the half RssAnon cannot see: a
#       leak that allocates without touching grows the mapping and not the
#       resident set, and a resident set the kernel swaps out shrinks while
#       the mapping does not.
#
# ⚠ BEFORE THE REAL RUN: 4.1 (code freeze) must have happened. Soaking a tree
# that is about to change spends a day of box time on a binary nobody will
# ship, and the roadmap's own sequencing note says a surprise in B5 AFTER the
# soak invalidates the soak.
set -u

MODE=soak
if [ "${1:-}" = "--shakedown" ]; then
    MODE=shake; shift
    HOURS=0; CHUNK_MIN=2
    TOTAL_SECS=360
elif [ "${1:-}" = "--rss-probe" ]; then
    MODE=probe; shift
    PROBE_MIN=${1:-30}
    HOURS=0; CHUNK_MIN=1
    TOTAL_SECS=$((PROBE_MIN * 60))
    # shift the minutes off so builddir/seed land in the usual slots
    [ $# -gt 0 ] && shift
    set -- "" "" "${1:-build/linux}" "${2:-20260914}"
else
    HOURS=${1:-24}
    CHUNK_MIN=${2:-60}
    TOTAL_SECS=$((HOURS * 3600))
fi
BUILDDIR=${3:-build/linux-asan}
SEED=${4:-20260906}
CHUNK_SECS=$((CHUNK_MIN * 60))
[ "$MODE" = shake ] && CHUNK_SECS=120

# ▶ THE GUEST CLOCK (design note 2026-10-12-m7-soak-4x6h-rule.md sections 9-10).
# serve_fuzz's chunk length and the server's --max-seconds both run on the WSL
# guest's MONOTONIC clock, which on this box runs FAST against the host: 9% on
# raw tsc, ~5% on both Hyper-V clocksources (2026-10-13; the host checked
# against time.windows.com). Every length in this script means REAL seconds,
# so the driver stretches what it hands to those two by the rate that
# tests/tooling/guest_clock_check.sh --rate measured just before the run (guest
# seconds per real second), and measures the rate again over the run's own
# span at the end. Required: an unmeasured clock is not "1".
CLOCK_RATE=${STOKE_SOAK_CLOCK_RATE:-}
# The server's budget covers the traffic, the driver's per-chunk work (fuzz
# teardown, canary, sampling -- 18 to 48 s a chunk in run 1, 2026-10-13, whose
# flat 120 s did not cover six of them) and the last in-flight exchange.
# Overridable ONLY so the rig's own test can starve the server on purpose.
SLACK_BASE=${STOKE_SOAK_SLACK_BASE_S:-120}
SLACK_PER_CHUNK=${STOKE_SOAK_SLACK_PER_CHUNK_S:-60}

# The serving shape. The probe shrinks ONLY the pool: same seqs, same threads,
# same traffic, so what it measures is the arena walk and not a different
# server.
CTX=1024
SEQS=4
THREADS=4
[ "$MODE" = probe ] && CTX=256

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 90
GR="$ROOT/docs/superpowers/gate-reports"
mkdir -p "$GR"
STAMP=$(date +%Y-%m-%d-%H%M)-m7-soak
[ "$MODE" = shake ] && STAMP="SHAKEDOWN-m7-soak"
[ "$MODE" = probe ] && STAMP="$(date +%Y-%m-%d-%H%M)-m7-rss-probe"
BANK="$GR/$STAMP.jsonl"
LOG="$GR/$STAMP.txt"
CSV="$GR/$STAMP.rss.csv"
WIT="$GR/$STAMP.dmesg"
SRVLOG="$BUILDDIR/m7_soak.srv"
SANLOG="$BUILDDIR/m7_soak.sanlog"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$BUILDDIR/stoke"
fuzz="$BUILDDIR/serve_fuzz"
port="${STOKE_M7_SOAK_PORT:-18171}"

say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
bankline() { printf '%s\n' "$1" >> "$BANK"; }
# grep -c prints a count even when it is 0 (and exits 1), so `|| echo 0` would
# print a SECOND zero; only a missing file leaves it empty.
gens_in() { g=$(grep -cE "stoke serve: (chat|stream) id=" "$1" 2>/dev/null); printf '%s' "${g:-0}"; }

# The series reader (v2, see the header). $1 = the saturation level in MiB
# (arena - 1 block), $2 = the CSV. `sat` is the first sample whose ARENA
# residency (column 8, read from the arena's own mappings) reaches that
# level; the post-saturation window is [sat, n], split in halves, and the
# means of RssAnon and VmData in each half are the slope reading. A function
# so the grader can be run on a synthetic series without a server (the grader
# is graded before it grades anything).
read_series() {
    awk -F, -v lvl="$1" 'NR>1 {n++; a[n]=$3+0; d[n]=$5+0; g[n]=$7+0; r[n]=$8+0}
    END {
        if (n < 6) { printf "n=%d sat=0", n; exit }
        sat=0; for (i=1;i<=n;i++) { if (r[i] >= lvl) { sat=i; break } }
        amax=0; amin=1e12; for (i=1;i<=n;i++){ if (a[i]>amax) amax=a[i]; if (a[i]<amin) amin=a[i] }
        printf "n=%d sat=%d anon0=%d data0=%d arena0=%d anon_min=%d anon_max=%d anon_last=%d data_last=%d arena_last=%d gens_last=%d",
               n, sat, a[1], d[1], r[1], amin, amax, a[n], d[n], r[n], g[n]
        if (sat == 0) exit
        m=n-sat+1; h=int(m/2); if (h < 1) exit
        sa1=0; sd1=0; for (i=sat;i<sat+h;i++){sa1+=a[i]; sd1+=d[i]}
        sa2=0; sd2=0; c2=0; for (i=sat+h;i<=n;i++){sa2+=a[i]; sd2+=d[i]; c2++}
        printf " anon_at_sat=%d arena_at_sat=%d gens_at_sat=%d post_n=%d anon_h1=%.0f anon_h2=%.0f data_h1=%.0f data_h2=%.0f",
               a[sat], r[sat], g[sat], m, sa1/h, sa2/c2, sd1/h, sd2/c2
    }' "$2"
}

# A previous run under the same stamp is stale, not history: it is the thing
# the numbers below would otherwise be appended to.
rm -f "$SRVLOG" "$SANLOG".* "$BANK" "$LOG" "$CSV" "$WIT" 2>/dev/null

# ---- PRE-FLIGHT. Every one of these has cost this project a run. ----
say "=== M7 soak pre-flight (stamp $STAMP, mode $MODE) ==="

# The clock rate and the slack (see where they are read, above). Refused, not
# defaulted: a run on an unmeasured clock is the run-1 failure again.
case "$CLOCK_RATE" in
    ''|*[!0-9.]*|*.*.*|.*|*.)
        say "ABORT -- STOKE_SOAK_CLOCK_RATE='$CLOCK_RATE' is not a measured rate: run"
        say "         sh tests/tooling/guest_clock_check.sh --rate first (the launcher does)"
        exit 2 ;;
esac
if ! awk -v r="$CLOCK_RATE" 'BEGIN { exit !(r >= 0.8 && r <= 1.25) }'; then
    say "ABORT -- clock rate $CLOCK_RATE is outside 0.8..1.25: no stretch makes that clock trustworthy"
    exit 2
fi
# And the clock's CAUSE (design note section 11): this distro's
# systemd-timesyncd steps the guest to NTP and WSL's own sync slews it back to
# the host at 1/12 for as long as both run, so the guest clock runs 8.3% fast
# from a few minutes after the distro starts -- after a short rate measurement
# may already have read ~1.00. A soak needs it stopped
# (tests/tooling/guest_clock_settle.sh, which the launcher runs).
tsd=$(systemctl is-active systemd-timesyncd 2>/dev/null)
say "guest clock: systemd-timesyncd ${tsd:-unknown}, stretching by $CLOCK_RATE"
if [ "$MODE" = soak ] && [ "$tsd" = active ]; then
    say "ABORT -- systemd-timesyncd is running: it and WSL's own time sync fight, and while"
    say "         they do the guest clock runs 8.3% fast. Run tests/tooling/guest_clock_settle.sh"
    say "         first (the launcher does)."
    exit 2
fi
for v in "$SLACK_BASE" "$SLACK_PER_CHUNK"; do
    case "$v" in
        ''|-|*[!0-9-]*|?*-*) say "ABORT -- a slack override '$v' is not a whole number of seconds"; exit 2 ;;
    esac
done
for f in "$model" "$bin" "$fuzz"; do
    [ -e "$f" ] || { say "ABORT -- missing $f"; exit 1; }
done
# The soak is graded on the FROZEN binary (SEQUENCE 4.1); the hash is the
# only thing that ties this log to the archive under build/stoke-m7-*.
say "binaries: $bin $(sha256sum "$bin" | cut -c1-16)  $fuzz $(sha256sum "$fuzz" | cut -c1-16)  (git $(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
case "$BUILDDIR" in
    *asan*) : ;;
    *) say "⚠ NOTE -- $BUILDDIR is not a sanitizer build. The soak will still" \
            "run and the RSS slope is still meaningful, but the use-after-free" \
            "and leak classes this gate exists for will NOT be watched." ;;
esac
[ -r /proc/self/status ] || { say "ABORT -- no /proc; the RSS instrument needs Linux"; exit 1; }

# Memory headroom. ASan roughly doubles footprint and this box is 15.6 GB; the
# project's own Makefile records sanitizer legs being OOM-killed inside
# `make test`, and three unrelated OOMs happened here on 2026-09-06.
free_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null)
say "memory available: ${free_mb:-unknown} MB"
if [ -n "${free_mb:-}" ] && [ "$free_mb" -lt 6000 ] && [ "$MODE" = soak ]; then
    say "ABORT -- under 6 GB available; a 24 h ASan soak will be OOM-killed and"
    say "         an OOM-killed server is a server whose leaks were never checked."
    exit 1
fi
if command -v curl > /dev/null 2>&1; then : ; else say "ABORT -- curl missing"; exit 1; fi

# The grader is graded before it grades anything. Gate [3]'s memory verdict
# comes from soak_rss_rule.sh, read at the END of a day: a grader that breaks
# under this VM's shell and awk would be found 24 h too late. Its test prints
# "test_soak_rss_rule: N/N OK"; the verdict is read from that line.
selftest=$(sh tests/tooling/test_soak_rss_rule.sh 2>&1 | tail -1)
case "$selftest" in
    "test_soak_rss_rule: "*" OK") say "RSS rule self-test: $selftest" ;;
    *) say "ABORT -- the RSS rule's own test did not pass: ${selftest:-no output}"
       say "         (sh tests/tooling/test_soak_rss_rule.sh prints every case)"
       exit 1 ;;
esac

# The VM-lifetime protections (header). Printed for every mode so a probe's
# log says what the VM looked like; ENFORCED only for the real soak, because
# a 6 min shakedown that dies teaches the same lesson for free.
uid=$(id -u)
balloon=$(dmesg 2>/dev/null | grep -o 'hv_balloon: Max. dynamic memory size: [0-9]* MB' | tail -1)
swap_mb=$(awk '/SwapTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null)
total_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null)
wslconf=""
for f in /mnt/c/Users/*/.wslconfig; do [ -f "$f" ] && wslconf="$f"; done
say "VM: MemTotal ${total_mb:-?} MB, swap ${swap_mb:-?} MB, ${balloon:-no hv_balloon line in dmesg}; .wslconfig: ${wslconf:-NONE}; uid $uid"
[ -n "$wslconf" ] && sed 's/^/  .wslconfig: /' "$wslconf" | tee -a "$LOG"
if [ "$MODE" = soak ]; then
    if [ "$uid" -ne 0 ]; then
        say "ABORT -- not root: oom_score_adj -1000 cannot be set on the server, so the"
        say "         OOM killer's first pick would be the server (the 2026-09-14 shape)."
        say "         Run as: wsl.exe -u root -- sh tests/tooling/run_m7_soak.sh ..."
        exit 1
    fi
    if [ -z "$wslconf" ] && [ -z "${STOKE_M7_SOAK_ALLOW_NO_WSLCONFIG:-}" ]; then
        say "ABORT -- no .wslconfig: the VM's cap is the 50%-of-host default and hv_balloon"
        say "         takes it back under host pressure. Write one ([wsl2] memory=, swap=),"
        say "         wsl --shutdown, re-run. STOKE_M7_SOAK_ALLOW_NO_WSLCONFIG=1 is the override."
        exit 1
    fi
fi

# ---- THE ARENA'S SIZE, FROM THE MODEL HEADER, BEFORE THE RUN. This is the
# formula in kv.c's kv_store_open (whole blocks of BP_BLOCK_TOKENS=16 tokens,
# f32 vectors, K and V arenas) with the pool sized as stoke.c sizes it:
# max_seqs * ceil(ctx / 16) blocks. If either changes, this must change with
# it -- which is why the derivation is printed and banked, not just the total.
hdr=$("$bin" info "$model" 2>/dev/null)
n_layer=$(printf '%s\n' "$hdr" | sed -n 's/.*\.block_count (UINT32) = \([0-9]*\).*/\1/p' | head -1)
n_kv=$(printf '%s\n' "$hdr" | sed -n 's/.*\.attention\.head_count_kv (UINT32) = \([0-9]*\).*/\1/p' | head -1)
hd=$(printf '%s\n' "$hdr" | sed -n 's/.*\.attention\.key_length (UINT32) = \([0-9]*\).*/\1/p' | head -1)
if [ -z "${hd:-}" ]; then
    n_embd=$(printf '%s\n' "$hdr" | sed -n 's/.*\.embedding_length (UINT32) = \([0-9]*\).*/\1/p' | head -1)
    n_head=$(printf '%s\n' "$hdr" | sed -n 's/.*\.attention\.head_count (UINT32) = \([0-9]*\).*/\1/p' | head -1)
    [ -n "${n_embd:-}" ] && [ -n "${n_head:-}" ] && hd=$((n_embd / n_head))
fi
if [ -z "${n_layer:-}" ] || [ -z "${n_kv:-}" ] || [ -z "${hd:-}" ]; then
    say "ABORT -- could not read n_layer/n_head_kv/head_dim from \`stoke info\`;"
    say "         the arena bound cannot be computed and the RSS reading would be"
    say "         a number with nothing to compare it to."
    exit 1
fi
blocks_per_seq=$(( (CTX + 15) / 16 ))
n_blocks=$((SEQS * blocks_per_seq))
block_bytes=$((2 * n_layer * 16 * n_kv * hd * 4))    # K + V, f32
arena_mib=$((n_blocks * block_bytes / 1048576))
block_mib_x10=$((block_bytes * 10 / 1048576))
file_mib=$(( $(wc -c < "$model") / 1048576 ))
say "arena: $SEQS seqs x $blocks_per_seq blocks = $n_blocks blocks x $block_bytes B" \
    "(n_layer=$n_layer n_head_kv=$n_kv head_dim=$hd f32 K+V) = $arena_mib MiB;" \
    "model file $file_mib MiB (mmap'd, file-backed)"

# ---- ONE server, for the whole window. Slack covers the last in-flight
# exchange only: --max-seconds starts when it begins LISTENING, so the model
# load sits outside it, which is what "serve for N seconds" should mean.
#
# ▶ AND THE BUDGET COVERS THE DRIVER, IN REAL SECONDS, STRETCHED TO GUEST ONES
# (design note section 10). It was TOTAL_SECS + 120 until run 1 (2026-10-13):
# six chunks of driver work outgrew the 120 s, the server ended itself ~1 min
# before chunk 6 did, and the driver died reading a server that was gone.
n_chunks=$(( (TOTAL_SECS + CHUNK_SECS - 1) / CHUNK_SECS ))
[ "$n_chunks" -lt 1 ] && n_chunks=1
srv_real=$((TOTAL_SECS + SLACK_BASE + n_chunks * SLACK_PER_CHUNK))
# --max-seconds 0 means NO limit to the server, so a starved budget is refused
# rather than turned into a server that never ends itself.
if [ "$srv_real" -lt 60 ]; then
    say "ABORT -- the server budget would be ${srv_real} s; the slack overrides starve it below a minute"
    exit 2
fi
srv_secs=$(awk -v s="$srv_real" -v r="$CLOCK_RATE" 'BEGIN { printf "%d", s * r + 0.5 }')
chunk_guest=$(awk -v s="$CHUNK_SECS" -v r="$CLOCK_RATE" 'BEGIN { printf "%d", s * r + 0.5 }')
say "clock: rate $CLOCK_RATE guest s per real s -- each chunk ${CHUNK_SECS} real s = ${chunk_guest} guest s;" \
    "server budget ${srv_real} real s (traffic + ${SLACK_BASE} + ${n_chunks} x ${SLACK_PER_CHUNK}) = ${srv_secs} guest s"

# The witness first: new kernel messages only (-W; the boot lines are in the
# pre-flight above), line-flushed so a record reaches /mnt/m before whatever
# comes next, -1000 so the killer skips it. Ended by us, at the end.
stdbuf -oL dmesg -W -T > "$WIT" 2>&1 &
wit_pid=$!
[ "$uid" -eq 0 ] && echo -1000 > "/proc/$wit_pid/oom_score_adj" 2>/dev/null
say "dmesg witness pid $wit_pid -> $(basename "$WIT") (oom_score_adj $(cat "/proc/$wit_pid/oom_score_adj" 2>/dev/null || echo '?'))"

# ASan's QUARANTINE IS SET, NOT DEFAULTED (2026-09-23, with the 24 h RSS
# rule). ASan holds freed chunks back from reuse until the quarantine --
# default 256 MiB -- is full, so an ASan server's VmData and RssAnon climb
# until it fills: the likeliest cause of the 1 h baseline's steady
# ~18-20 MiB/h, which the plain build did not show. At that rate 256 MiB
# fills at ~14 h, close to the rule's window at 18 h, and a slower day of
# traffic would push the fill INTO the window and fail a bounded fill as a
# leak. 64 MiB fills in ~3.5 h at that rate and before 18 h down to ~20% of
# it, and still holds hours of this server's frees -- a use-after-free has to
# reach back further than that to slip past it. If the 24 h still climbs in
# its last 6 h, the quarantine was not the cause, and the rule says so.
# THE 4 x 6 h SCHEME (Kabir 2026-10-12; design note
# docs/superpowers/design-notes/2026-10-12-m7-soak-4x6h-rule.md): a 6 h run
# is graded over its final 3 h, so its fill must end by hour 3. 16 MiB fills
# in ~50 min at the baseline's rate, and before hour 3 down to ~27% of it; a
# freed chunk still sits in quarantine for most of an hour of this server's
# frees.
ASAN_QUARANTINE_MB=64
[ "$MODE" = soak ] && [ "$HOURS" -eq 6 ] && ASAN_QUARANTINE_MB=16
say "starting server: ${srv_secs}s budget, port $port, build $BUILDDIR, --ctx $CTX --seqs $SEQS --threads $THREADS, ASan quarantine ${ASAN_QUARANTINE_MB} MiB"
ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:abort_on_error=0:quarantine_size_mb=$ASAN_QUARANTINE_MB:log_path=$SANLOG" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
"$bin" serve "$model" --port "$port" --ctx "$CTX" --seqs "$SEQS" --threads "$THREADS" \
    --max-seconds "$srv_secs" -v > "$SRVLOG" 2>&1 &
srv_pid=$!
# Before the model load, which is when RSS grows fastest: the server is the
# one process the killer must never pick.
[ "$uid" -eq 0 ] && echo -1000 > "/proc/$srv_pid/oom_score_adj" 2>/dev/null

if ! curl -sS --retry 600 --retry-delay 1 --retry-connrefused \
        "http://127.0.0.1:$port/health" > /dev/null; then
    say "ABORT -- the server never answered /health"
    kill "$srv_pid" 2>/dev/null; kill "$wit_pid" 2>/dev/null
    exit 1
fi
srv_oom=$(cat "/proc/$srv_pid/oom_score_adj" 2>/dev/null)
say "server up (pid $srv_pid, oom_score_adj ${srv_oom:-?})"
if [ "$MODE" = soak ] && [ "${srv_oom:-0}" != "-1000" ]; then
    say "ABORT -- the server's oom_score_adj is ${srv_oom:-?}, not -1000: it would be the"
    say "         OOM killer's first pick, and a killed server is a server whose leaks"
    say "         were never checked."
    kill "$srv_pid" 2>/dev/null; kill "$wit_pid" 2>/dev/null
    exit 1
fi

# /proc/<pid>/status, five fields, MiB. RssAnon + RssFile + RssShmem == VmRSS.
mem_fields() {
    awk '/^VmRSS/   {r=int($2/1024)}
         /^RssAnon/ {a=int($2/1024)}
         /^RssFile/ {f=int($2/1024)}
         /^VmData/  {d=int($2/1024)}
         /^VmHWM/   {h=int($2/1024)}
         END {printf "%d %d %d %d %d\n", r, a, f, d, h}' "/proc/$1/status" 2>/dev/null
}
rss_mb()  { mem_fields "$1" | cut -d' ' -f1; }
anon_mb() { mem_fields "$1" | cut -d' ' -f2; }
file_mb() { mem_fields "$1" | cut -d' ' -f3; }
data_mb() { mem_fields "$1" | cut -d' ' -f4; }

# THE ARENA'S OWN RESIDENCY. The K and V arenas are two xcalloc'd blocks of
# exactly total/2 bytes each (kv.c), which glibc serves as two anonymous
# mmaps of that size plus a 16 B header rounded up to a page. Two things the
# first two versions of this reader learned the hard way on 2026-09-14:
#   - the kernel MERGES adjacent anonymous mappings with the same flags into
#     one VMA (the two arenas showed as ONE mapping of 2 * (half + 4 KiB));
#   - and it keeps merging: a later mmap (the attention workspaces, ~37 MiB
#     of VmData in the first minute) landed adjacent and the VMA no longer
#     had the arena's size, so a reader that FOUND the arena by size at t=0
#     read 0 for the rest of the run.
# So the arena is located ONCE, at t=0, by size -- the anonymous mapping(s)
# (no path: 5 header fields) whose Size is within 64 KiB of half or of the
# whole, asserted to add up to the arena -- and its ADDRESS RANGE(S) are kept.
# From then on residency is read per page from /proc/<pid>/pagemap over
# exactly those ranges (bit 63 = present, bit 62 = swapped; the PFN is masked
# for an unprivileged reader but those two bits are not), which no later
# mapping can disturb. Prints "<present MiB> <swapped MiB>".
arena_half_kib=$((n_blocks * block_bytes / 2 / 1024))
arena_locate() {  # prints "<matched size KiB> <ranges...>"
    awk -v half="$arena_half_kib" '
        /^[0-9a-f]+-[0-9a-f]+ / { anon = (NF == 5); rng = $1; sz = 0 }
        /^Size:/ { sz = $2
                   if (anon && ((sz >= half - 64 && sz <= half + 64) ||
                                (sz >= 2 * half - 64 && sz <= 2 * half + 64))) { msz += sz; out = out " " rng } }
        END { printf "%d%s\n", msz, out }' "/proc/$1/smaps" 2>/dev/null
}
PAGEMAP_PY="$BUILDDIR/_pagemap_rss.py"
cat > "$PAGEMAP_PY" <<'PY'
import os, struct, sys
pid, ranges = sys.argv[1], sys.argv[2:]
fd = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
present = swapped = 0
for r in ranges:
    s, e = (int(x, 16) for x in r.split("-"))
    first, n = s >> 12, (e - s) >> 12
    buf = b""
    while len(buf) < n * 8:
        chunk = os.pread(fd, n * 8 - len(buf), first * 8 + len(buf))
        if not chunk:
            break
        buf += chunk
    for (v,) in struct.iter_unpack("<Q", buf):
        if v >> 63 & 1:
            present += 1
        elif v >> 62 & 1:
            swapped += 1
print(f"{present * 4096 // 1048576} {swapped * 4096 // 1048576}")
PY
arena_rss() { python3 "$PAGEMAP_PY" "$1" $ARENA_RANGES 2>/dev/null || printf '0 0\n'; }

set -- $(mem_fields "$srv_pid")
rss0=$1; anon0=$2; file0=$3; data0=$4
set -- $(arena_locate "$srv_pid")
arena_msz=$1; shift; ARENA_RANGES="$*"
if [ "${arena_msz:-0}" -lt $((2 * arena_half_kib - 128)) ] || [ "${arena_msz:-0}" -gt $((2 * arena_half_kib + 128)) ]; then
    say "ABORT -- the anonymous mappings matched in /proc/$srv_pid/smaps total ${arena_msz:-0} KiB, not the"
    say "         arena's $((2 * arena_half_kib)) KiB (ranges: ${ARENA_RANGES:-none}); the arena instrument"
    say "         is not reading the arena. (kv.c's layout or glibc's mmap threshold changed?"
    say "         A run whose instrument cannot see the arena would grade its own guess.)"
    kill "$srv_pid" 2>/dev/null; kill "$wit_pid" 2>/dev/null
    exit 1
fi
set -- $(arena_rss "$srv_pid")
arena0=$1; arena_sw0=$2
say "RSS at t=0: ${rss0:-?} MB (anon ${anon0:-?}, file ${file0:-?}, data ${data0:-?}); arena located at $ARENA_RANGES ($arena_msz KiB against 2 x $arena_half_kib), resident ${arena0:-?} MiB, swapped ${arena_sw0:-?} MiB -- before any request"

# Sidecar: one line every 5 s for as long as the server lives. Written by a
# subshell so a driver chunk that blocks cannot starve the series, and read
# back for the verdict below.
t_start=$(date +%s)
printf 't_s,rss_mb,anon_mb,file_mb,data_mb,hwm_mb,gens,arena_mb,arena_swapped_mb\n' > "$CSV"
(
    while kill -0 "$srv_pid" 2>/dev/null; do
        now=$(( $(date +%s) - t_start ))
        printf '%d,%s,%s,%s\n' "$now" "$(mem_fields "$srv_pid" | tr ' ' ',')" "$(gens_in "$SRVLOG")" \
               "$(arena_rss "$srv_pid" | tr ' ' ',')" >> "$CSV"
        sleep 5
    done
) &
sidecar_pid=$!

# ---- chunked driver ----
say "driver: $n_chunks chunk(s) of ${CHUNK_SECS}s (${chunk_guest} guest s each)"
# The run's own clock reading starts here and ends after the last chunk.
clk0_up=$(cut -d' ' -f1 /proc/uptime); clk0_wall=$(date +%s)

manifest=$(mktemp)
find tests/data/http_corpus -type f 2>/dev/null | head -200 > "$manifest"
n_corpus=$(wc -l < "$manifest" | tr -d ' ')
say "corpus: $n_corpus files"

srv_gone=""
i=1
while [ "$i" -le "$n_chunks" ]; do
    if ! kill -0 "$srv_pid" 2>/dev/null; then
        say "⚠ server exited early, before chunk $i -- stopping the driver"
        srv_gone="before chunk $i"
        break
    fi
    cseed=$((SEED + i))
    out=$("$fuzz" 127.0.0.1 "$port" "$chunk_guest" "$manifest" 4 "$cseed" 20000 3 2>&1)
    frc=$?
    # A killed fuzz is a lost CHUNK, not a lost run: the server (-1000) is
    # still there and the loop goes on. Said out loud, because a chunk line
    # with rc=137 and no exchange count would otherwise read like a quiet one.
    if [ "$frc" -ge 128 ]; then
        say "⚠ chunk $i: the fuzz was KILLED (rc $frc = signal $((frc - 128))) -- its traffic is lost, the server's lifetime is not; see the dmesg witness"
        killed=$((${killed:-0} + 1))
    fi
    # ⚠ THE STALL HALF OF THE GATE ("no crash, no leak slope, no stall") had no
    # instrument: serve_fuzz DISCARDS replies and counts any bytes read as an
    # exchange, so a server that answers headers and never finishes a
    # generation looks busy to it. Two readings close that. (1) The canary: one
    # plain non-streaming chat with a hard deadline, sent when the chunk's
    # traffic has drained, so a server that no longer completes generations
    # answers something other than 200 within 60 s. (2) The chunk's own
    # answered-exchange count from the fuzz line: a whole chunk with zero
    # answered exchanges is a server that stopped answering, whatever the
    # canary later says.
    canary=$(curl -sS --max-time 60 -o /dev/null -w '%{http_code} %{time_total}' \
        -H 'Content-Type: application/json' \
        -d '{"model":"stoke","messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"stream":false}' \
        "http://127.0.0.1:$port/v1/chat/completions" 2>/dev/null || printf '000 timeout')
    ccode=${canary%% *}; csecs=${canary#* }
    exch=$(printf '%s' "$out" | sed -n 's/.* \([0-9]*\) answered exchanges.*/\1/p' | head -1)
    mix=$(printf '%s' "$out" | tr '\n' ' ' | sed 's/"/\\"/g')
    # ▶ IS THE SERVER STILL THERE? (design note section 10.) Run 1 (2026-10-13)
    # died HERE without a verdict: the server's deadline came during chunk 6,
    # /proc held no status to read, `set --` set nothing, and `rss=$1` killed
    # the driver under `set -u`. A server that is gone is a READING: the chunk
    # is banked as far as it got, its stall readings are not counted (they
    # measure the absence), and the verdict below says whether the server
    # ended itself (VOID) or not (FAIL). gawk and mawk print nothing for a pid
    # that is gone; a zombie not yet reaped reads VmRSS 0 (the last sidecar
    # rows of run 1 and of this fix's red run).
    mf=$(mem_fields "$srv_pid")
    case "$mf" in
        ''|'0 '*)
            bankline "{\"chunk\":$i,\"seed\":$cseed,\"secs\":$CHUNK_SECS,\"guest_secs\":$chunk_guest,\"fuzz_rc\":$frc,\"server\":\"gone\",\"exchanges\":${exch:-null},\"canary_code\":\"$ccode\",\"canary_s\":\"$csecs\",\"mix\":\"$mix\"}"
            say "⚠ the server ended DURING chunk $i/$n_chunks (fuzz rc=$frc, exch=${exch:-?}, canary=$ccode) -- stopping the driver"
            say "  $(printf '%s' "$out" | grep -E 'serve_fuzz:' | tr '\n' ' ')"
            srv_gone="during chunk $i"
            break ;;
    esac
    set -- $mf
    rss=$1; anon=$2; filem=$3; datam=$4
    arena=$(arena_rss "$srv_pid" | cut -d' ' -f1)
    gens=$(gens_in "$SRVLOG")
    bankline "{\"chunk\":$i,\"seed\":$cseed,\"secs\":$CHUNK_SECS,\"guest_secs\":$chunk_guest,\"fuzz_rc\":$frc,\"rss_mb\":${rss:-null},\"anon_mb\":${anon:-null},\"file_mb\":${filem:-null},\"data_mb\":${datam:-null},\"arena_mb\":${arena:-null},\"gens_cumulative\":$gens,\"exchanges\":${exch:-null},\"canary_code\":\"$ccode\",\"canary_s\":\"$csecs\",\"mix\":\"$mix\"}"
    say "chunk $i/$n_chunks rc=$frc rss=${rss:-?}MB (anon ${anon:-?} file ${filem:-?} data ${datam:-?} arena ${arena:-?}/$arena_mib) gens=$gens exch=${exch:-?} canary=$ccode/${csecs}s"
    say "  $(printf '%s' "$out" | grep -E 'serve_fuzz:' | tr '\n' ' ')"
    if [ "$ccode" != "200" ]; then
        say "⚠ STALL -- the canary after chunk $i got '$ccode' in ${csecs}s (expected 200)"
        stalls=$((${stalls:-0} + 1))
    fi
    if [ "${exch:-1}" = "0" ]; then
        say "⚠ STALL -- chunk $i banked ZERO answered exchanges"
        stalls=$((${stalls:-0} + 1))
    fi
    i=$((i + 1))
done
rm -f "$manifest"

# The run's own clock reading (design note section 10): the guest's rate over
# the whole driver span, against the rate the chunks were stretched by. The
# guest WALL clock is kept on the host's by Hyper-V time sync (it is stepped
# back whenever the fast guest clock runs ahead), so over hours it is a
# real-seconds ruler; over a 6-minute shakedown its steps are too coarse to
# grade, so only soak mode is graded on it.
clk1_up=$(cut -d' ' -f1 /proc/uptime); clk1_wall=$(date +%s)
run_rate=$(awk -v u0="$clk0_up" -v u1="$clk1_up" -v w0="$clk0_wall" -v w1="$clk1_wall" \
    'BEGIN { d = w1 - w0; if (d <= 0) { print "?"; exit } printf "%.4f", (u1 - u0) / d }')
say "clock: over the driver's $((clk1_wall - clk0_wall)) real s the guest ran $run_rate guest s per real s (stretched by $CLOCK_RATE; systemd-timesyncd now $(systemctl is-active systemd-timesyncd 2>/dev/null || true))"
void=0
if [ "$MODE" = soak ]; then
    if [ "$run_rate" = "?" ] || ! awk -v a="$run_rate" -v b="$CLOCK_RATE" \
            'BEGIN { d = (a - b) / b; if (d < 0) d = -d; exit !(d <= 0.01) }'; then
        say "VOID -- the guest clock's rate over the run ($run_rate) is more than 1% from the rate the"
        say "        chunks were stretched by ($CLOCK_RATE): the chunks were not the registered length"
        void=1
    fi
fi

say "waiting for the server to end ITSELF (never signalled -- this IS the leak check)"
wait "$srv_pid"; srv_rc=$?
say "server exited $srv_rc"
kill "$sidecar_pid" 2>/dev/null; wait "$sidecar_pid" 2>/dev/null
kill "$wit_pid" 2>/dev/null; wait "$wit_pid" 2>/dev/null

# ---- verdict ----
fail=0
# What the kernel said while the server lived. An OOM kill of anything, or a
# balloon event, is a fact about the run whatever the other checks say.
# hv_balloon's boot-time "Max. dynamic memory size" line is information, not
# an event (it landed 45 s into the shakedown because the VM had just booted);
# the balloon's real requests ("Balloon request", "Unballoon") still count.
oom_pat='Out of memory|Killed process|oom-kill|oom_reaper|hv_balloon'
oomlines=$(grep -E "$oom_pat" "$WIT" 2>/dev/null | grep -v -c 'Max. dynamic memory size')
if [ "${oomlines:-0}" -gt 0 ]; then
    say "⚠ the dmesg witness recorded ${oomlines} OOM/balloon line(s) ($(wc -l < "$WIT" | tr -d ' ') kernel lines in all):"
    grep -E "$oom_pat" "$WIT" | grep -v 'Max. dynamic memory size' | tail -20 | sed 's/^/  /' | tee -a "$LOG"
else
    say "dmesg witness: no OOM or balloon lines over the run ($(wc -l < "$WIT" | tr -d ' ') kernel lines in all)"
fi
[ "${killed:-0}" -ne 0 ] && say "⚠ ${killed} chunk(s) lost to a killed fuzz (banked as fuzz_rc >= 128)"
say "host side, afterwards: Hyper-V-Worker-Admin event 18508 says how the VM ended --"
say "  powershell: Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-Hyper-V-Worker-Admin';Id=18508} -MaxEvents 3 | Format-List TimeCreated,Message"
if [ "$srv_rc" -ne 0 ]; then
    say "FAIL -- server exited $srv_rc (a sanitizer abort exits non-zero; so does a crash)"
    tail -40 "$SRVLOG" | sed 's/^/  /' | tee -a "$LOG"
    fail=1
fi
# ▶ A SERVER THAT ENDED BEFORE ITS DRIVER (design note section 10). Ended
# ITSELF -- exit 0, and its log's last line is the deadline's own "requests
# completed, exiting" -- its budget did not cover the driver: the traffic was
# cut short, the reading does not exist, and the run is VOID (re-run with its
# seed). Anything else that ended it is the engine's, and FAIL outranks VOID.
if [ -n "$srv_gone" ]; then
    srv_last=$(tail -n 1 "$SRVLOG" 2>/dev/null)
    case "$srv_rc:$srv_last" in
        "0:stoke serve: "*" requests completed, exiting")
            say "VOID -- the server ended ITSELF $srv_gone of $n_chunks ('$srv_last'): its budget,"
            say "        ${srv_real} real s = ${srv_secs} guest s, did not cover the driver, so the traffic was cut short"
            void=1 ;;
        *)
            say "FAIL -- the server ended $srv_gone of $n_chunks, and not by its own deadline"
            say "        (exit $srv_rc; its log ends: '$srv_last')"
            fail=1 ;;
    esac
fi
if ls "$SANLOG".* > /dev/null 2>&1; then
    say "FAIL -- the sanitizer reported:"
    for f in "$SANLOG".*; do sed 's/^/  /' "$f" | tee -a "$LOG"; done
    fail=1
fi
n_gen=$(gens_in "$SRVLOG")
if [ "${n_gen:-0}" -lt 10 ]; then
    say "FAIL -- only ${n_gen:-0} completed generations in the SERVER's own log."
    say "        Traffic that is all refusals looks like a clean soak to every"
    say "        other check here (M6's soak passed exactly that way)."
    fail=1
fi
if [ "${stalls:-0}" -ne 0 ]; then
    say "FAIL -- ${stalls} stall reading(s): a canary that did not get 200 within"
    say "        60 s, or a chunk with zero answered exchanges (see the ⚠ STALL lines)."
    fail=1
fi

# RSS slope: first banked chunk against the last, split. This is the number a
# chunked SERVER could never produce, and it is why the server is not chunked.
r_first=$(head -1 "$BANK" 2>/dev/null | sed -n 's/.*"rss_mb":\([0-9]*\).*/\1/p')
r_last=$(tail -1 "$BANK" 2>/dev/null | sed -n 's/.*"rss_mb":\([0-9]*\).*/\1/p')
a_first=$(head -1 "$BANK" 2>/dev/null | sed -n 's/.*"anon_mb":\([0-9]*\).*/\1/p')
a_last=$(tail -1 "$BANK" 2>/dev/null | sed -n 's/.*"anon_mb":\([0-9]*\).*/\1/p')
f_first=$(head -1 "$BANK" 2>/dev/null | sed -n 's/.*"file_mb":\([0-9]*\).*/\1/p')
f_last=$(tail -1 "$BANK" 2>/dev/null | sed -n 's/.*"file_mb":\([0-9]*\).*/\1/p')
if [ -n "${r_first:-}" ] && [ -n "${r_last:-}" ]; then
    say "RSS: ${r_first} MB -> ${r_last} MB over $n_chunks chunk(s)" \
        "(anon ${a_first:-?} -> ${a_last:-?}, file ${f_first:-?} -> ${f_last:-?};" \
        "arena bound $arena_mib MiB anon, file bound $file_mib MiB)"
fi

# The series, read against saturation (v2 -- see the header). Everything is
# printed so the reader can check the rule against the numbers rather than
# take the word.
one_block=$(( (block_mib_x10 + 9) / 10 ))
sat_level=$((arena_mib - one_block))
series=$(read_series "$sat_level" "$CSV")
say "series ($(basename "$CSV")): $series"
say "  (saturation level = arena $arena_mib - one block = $sat_level MiB resident in the arena's own mappings)"

if [ "$MODE" = probe ]; then
    # Pre-registered reading -- see the header. Every number below is either
    # from the header (arena, one block) or from the series line above.
    n_s=$(printf '%s' "$series" | sed -n 's/.*n=\([0-9]*\) sat.*/\1/p')
    sat=$(printf '%s' "$series" | sed -n 's/.* sat=\([0-9]*\).*/\1/p')     # NOT anon_at_sat=
    s_anon0=$(printf '%s' "$series" | sed -n 's/.*anon0=\([0-9]*\).*/\1/p')
    s_data0=$(printf '%s' "$series" | sed -n 's/.*data0=\([0-9]*\).*/\1/p')
    a_end=$(printf '%s' "$series" | sed -n 's/.*anon_last=\([0-9]*\).*/\1/p')
    d_end=$(printf '%s' "$series" | sed -n 's/.*data_last=\([0-9]*\).*/\1/p')
    r_end=$(printf '%s' "$series" | sed -n 's/.*arena_last=\([0-9]*\).*/\1/p')
    a_sat=$(printf '%s' "$series" | sed -n 's/.*anon_at_sat=\([0-9]*\).*/\1/p')
    r_sat=$(printf '%s' "$series" | sed -n 's/.*arena_at_sat=\([0-9]*\).*/\1/p')
    g_sat=$(printf '%s' "$series" | sed -n 's/.*gens_at_sat=\([0-9]*\).*/\1/p')
    post_n=$(printf '%s' "$series" | sed -n 's/.*post_n=\([0-9]*\).*/\1/p')
    a_h1=$(printf '%s' "$series" | sed -n 's/.*anon_h1=\([0-9]*\).*/\1/p')
    a_h2=$(printf '%s' "$series" | sed -n 's/.*anon_h2=\([0-9]*\).*/\1/p')
    d_h1=$(printf '%s' "$series" | sed -n 's/.*data_h1=\([0-9]*\).*/\1/p')
    d_h2=$(printf '%s' "$series" | sed -n 's/.*data_h2=\([0-9]*\).*/\1/p')
    verdict=""
    if [ -z "${n_s:-}" ] || [ "${n_s:-0}" -lt 6 ]; then
        verdict="VOID (fewer than 6 samples -- the sidecar did not run)"
    elif [ "${sat:-0}" -eq 0 ] || [ -z "${post_n:-}" ]; then
        verdict="VOID (the arena's residency never reached $sat_level MiB -- not saturated inside the run; arena ended $r_end, anon $a_end, data $d_end)"
    elif [ $((post_n * 3)) -lt "$n_s" ]; then
        verdict="VOID (post-saturation window is $post_n of $n_s samples, under a third of the run; saturation at sample $sat, $g_sat gens)"
    elif [ $((d_h2 - d_h1)) -gt "$one_block" ]; then
        verdict="CLIMBING (mapped growth: VmData $d_h1 -> $d_h2 MiB (+$((d_h2 - d_h1))) across the post-saturation halves; threshold one block = $one_block MiB)"
    elif [ $((a_h2 - a_h1)) -gt "$one_block" ]; then
        verdict="CLIMBING (touched growth past the arena: RssAnon $a_h1 -> $a_h2 MiB (+$((a_h2 - a_h1))) across the post-saturation halves with VmData $d_h1 -> $d_h2; threshold one block = $one_block MiB)"
    elif [ $((a_end - r_end)) -gt $((s_anon0 + (d_end - s_data0) + one_block)) ]; then
        verdict="CLIMBING (non-arena resident $((a_end - r_end)) MiB at the end, above baseline $s_anon0 + mapped growth $((d_end - s_data0)) + one block = $((s_anon0 + (d_end - s_data0) + one_block)) MiB: resident beyond anything mapped)"
    else
        verdict="FLAT (arena saturated at sample $sat of $n_s after $g_sat gens ($r_sat of $arena_mib MiB resident); post-saturation halves anon $a_h1 -> $a_h2 (+$((a_h2 - a_h1))), data $d_h1 -> $d_h2 (+$((d_h2 - d_h1))); threshold one block = $one_block MiB; ended anon $a_end = arena $r_end + non-arena $((a_end - r_end)) against baseline $s_anon0 + mapped growth $((d_end - s_data0)))"
    fi
    say "=== RSS PROBE: $verdict ==="
    case "$verdict" in
        FLAT*)     [ "$fail" -ne 0 ] && { say "=== RSS PROBE: FLAT but the run FAILED above ==="; exit 1; }; exit 0 ;;
        CLIMBING*) exit 1 ;;
        *)         exit 2 ;;
    esac
fi

# THE 24 h RSS RULE (header; pre-registered 2026-09-23). It grades the gate
# run -- soak mode, 24 h or more -- and a shorter run prints the same reading
# ungraded: the rule asks whether growth has STOPPED by hour 18, which a 1 h
# run cannot answer. The verdict is read from the grader's PRINTED line.
# Since 2026-10-12 a 6 h soak-mode run is ALSO graded: one run of the 4 x 6 h
# scheme, over its final 3 h, at the same rate (one block per 6 h of window;
# design note 2026-10-12-m7-soak-4x6h-rule.md). Gate [3] needs four of them.
rss_win=21600; rss_label="gate [3], final 6 h"
if [ "$MODE" = soak ] && [ "$HOURS" -eq 6 ]; then
    rss_win=10800; rss_label="gate [3] 4 x 6 h scheme, seed $SEED, final 3 h"
fi
rss_rule=$(sh tests/tooling/soak_rss_rule.sh "$CSV" "$TOTAL_SECS" "$block_bytes" "$rss_win" 2>&1)
if [ "$void" -ne 0 ]; then
    # A VOID run has no reading to grade (design note section 7): its memory
    # line is reported, and neither passes nor fails the run.
    say "RSS rule, REPORTED not graded (the run is VOID, above): $rss_rule"
    rss_word="RSS not graded"
elif [ "$MODE" = soak ] && { [ "$HOURS" -ge 24 ] || [ "$HOURS" -eq 6 ]; }; then
    say "RSS rule ($rss_label): $rss_rule"
    case "$rss_rule" in
        "FLAT ("*) rss_word="RSS FLAT" ;;
        *) say "FAIL -- the RSS rule did not read FLAT: CLIMBING is growth still under way"
           say "        in the graded window; VOID is a reading that does not exist. Neither is a pass."
           rss_word="RSS NOT FLAT"; fail=1 ;;
    esac
else
    say "RSS rule, REPORTED not graded (a ${HOURS} h $MODE run is not a gate run): $rss_rule"
    rss_word="RSS not graded"
fi

say "traffic mix per chunk is banked in $(basename "$BANK") -- the roadmap"
say "requires the driver to report what its traffic ACTUALLY CONTAINED, and"
say "'it was busy' is not that."

if [ "$fail" -ne 0 ]; then
    say "=== M7 soak: FAILED ==="
    exit 1
fi
if [ "$void" -ne 0 ]; then
    # Engine evidence of a failure outranks a rig error (FAILED is checked
    # first); with none, a run whose reading does not exist is VOID and is
    # re-run with its own seed (design note section 7).
    say "=== M7 soak: VOID (the reading does not exist -- see the VOID line above; re-run with seed $SEED) ==="
    exit 2
fi
say "=== M7 soak: OK ($n_chunks chunks, ${killed:-0} lost, ${TOTAL_SECS}s at clock rate $CLOCK_RATE (run $run_rate), $n_gen generations, no sanitizer reports, ${oomlines:-0} OOM/balloon lines, $rss_word) ==="
exit 0
