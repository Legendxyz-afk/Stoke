#!/bin/sh
# tests/serve_asan.sh -- M6 GATE ITEM [7], second half: the randomized serve
# smoke, under a sanitizer.
#
# Ten minutes of deliberately hostile, reproducible traffic against a real
# `stoke serve` while ASan watches from the inside. The failures this is hunting
# -- a use-after-free on a path only a client disconnect reaches, a read past a
# request buffer, a leak per rejected connection -- do not show up as a wrong
# answer. They show up as nothing at all, which is why every other serve gate
# can be green while one of these is present.
#
# ▶▶ THE SERVER ENDS ITSELF, AND THAT IS THE WHOLE DESIGN. LeakSanitizer reports
# at process EXIT, so a server killed by this script is a server whose leaks were
# never checked -- the gate would go green on precisely the class it exists to
# find. The traffic is random, so `--max-requests` cannot express "stop at the
# end"; `--max-seconds` can, and the script waits for the process rather than
# signalling it.
#
# ▶ ASAN_OPTIONS uses log_path, so a clean run creates NO file at all and the
# check below is "no log appeared". halt_on_error=1 makes the first report fatal
# rather than letting a damaged process keep serving and pile up noise.
#
# ⚠ THIS GATE IS A LINUX/WSL LEG. The Windows toolchain here has no ASan, so on
# Windows it SKIPS loudly rather than pretending. Run it as:
#     make SAN=asan && sh tests/serve_asan.sh build/linux-asan
#
# Usage: sh tests/serve_asan.sh <builddir> [exe-suffix] [seconds] [clients] [seed]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: serve_asan.sh <builddir> [exe-suffix] [seconds] [clients] [seed]}"
exe="${2:-}"
secs="${3:-600}"
clients="${4:-4}"
seed="${5:-20260829}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$builddir/stoke$exe"
fuzz="$builddir/serve_fuzz$exe"
port="${STOKE_SERVE_ASAN_PORT:-$(golden_lib_port 18147 "$builddir")}"

golden_lib_require_model "$model" "serve_asan"
golden_lib_require_binary "$bin" "serve_asan"
golden_lib_require_binary "$fuzz" "serve_asan"

# The A2 garbage corpus, verbatim, is the single best source of hostile input
# this project already owns -- 51 files whose whole purpose is to be malformed
# in a specific way. The manifest exists because listing a directory portably
# from C is more platform knowledge than a test client should carry.
manifest="$builddir/serve_asan.manifest"
ls tests/data/http_corpus/*.raw > "$manifest" 2> /dev/null
n_corpus=$(wc -l < "$manifest" 2> /dev/null || echo 0)
if [ "${n_corpus:-0}" -lt 10 ]; then
    echo "serve_asan: FAILED -- only ${n_corpus:-0} corpus files found; the" \
         "garbage half of this soak would be missing and a clean log would" \
         "mean less than it appears to" >&2
    exit 1
fi

log_prefix="$builddir/serve_asan.sanlog"
srvlog="$builddir/serve_asan.srv"
rm -f "$srvlog" "$log_prefix".*

# The server outlives the fuzzer by a margin, and the margin is small because
# its deadline starts when it STARTS LISTENING, not when the process starts --
# the model load is outside it, which is what "serve for N seconds" should mean.
# The slack only has to cover the fuzzer's own last in-flight exchange.
srv_secs=$((secs + 60))

ASAN_OPTIONS="detect_leaks=1:halt_on_error=1:abort_on_error=0:log_path=$log_prefix" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1" \
"$bin" serve "$model" --port "$port" --ctx 1024 --seqs 4 --threads 4 \
    --max-seconds "$srv_secs" -v > "$srvlog" 2>&1 &
srv_pid=$!

if ! curl -sS --retry 300 --retry-delay 1 --retry-connrefused \
        "http://127.0.0.1:$port/health" > /dev/null; then
    echo "serve_asan: FAILED -- the server never answered /health" >&2
    sed 's/^/  /' "$srvlog" >&2
    kill "$srv_pid" 2> /dev/null
    exit 1
fi

echo "serve_asan: server up on $port; ${secs}s of randomized traffic, $clients clients"
# read-ms and max-tok are gentler than the driver defaults because a
# sanitized forward pass is an order of magnitude slower: a client that
# gives up before the answer arrives contributes an action and no exchange,
# and a soak of nothing but timeouts is a clean log over no traffic.
"$fuzz" 127.0.0.1 "$port" "$secs" "$manifest" "$clients" "$seed" 20000 3
fuzz_rc=$?

# Waited for, never signalled -- see the header. This is the leak check.
wait "$srv_pid"
srv_rc=$?

fail=0
[ "$fuzz_rc" -eq 0 ] || fail=1
if [ "$srv_rc" -ne 0 ]; then
    echo "serve_asan: FAILED -- the server exited $srv_rc (a sanitizer abort" \
         "exits non-zero; so does a crash)" >&2
    tail -40 "$srvlog" >&2
    fail=1
fi

# A clean run creates no log file at all.
if ls "$log_prefix".* > /dev/null 2>&1; then
    echo "serve_asan: FAILED -- the sanitizer reported:" >&2
    for f in "$log_prefix".*; do
        echo "  ---- $f ----" >&2
        sed 's/^/  /' "$f" >&2
    done
    fail=1
fi

# ⚠ AND THE SERVER MUST HAVE DONE SOME WORK. A soak that loaded the model and
# then answered nothing would produce a clean sanitizer log, a zero exit, and no
# evidence whatsoever. The fuzzer asserts this from its side too; this is the
# server's own account of it.
if ! grep -q "stoke serve: " "$srvlog"; then
    echo "serve_asan: FAILED -- the server log is empty; nothing was served" >&2
    fail=1
fi

# ⚠⚠ AND IT MUST HAVE GENERATED, WHICH IS NOT THE SAME CLAIM. The fuzzer counts
# "answered exchanges", and a 400 is an answered exchange -- so a soak in which
# every single request was REFUSED passes every check above it while never
# entering the model, the scheduler, or the sampler. That is not hypothetical:
# the traffic builder constructs its JSON with snprintf, and one wrong comma
# there turns ten minutes of memory-safety evidence into ten minutes of testing
# the error path. This asserts the server's own record of work it can only have
# done by generating.
#
# ▶ PROVED, NOT ASSUMED, AND THE THRESHOLD IS THE MEASURED PART. Built with
# -DSTOKE_MUT_FUZZ_BAD_BODY, serve_fuzz drops exactly that comma. The mutant run
# (90 s, plain build) produced MORE traffic than the clean one -- 151 actions and
# 104 answered exchanges against 88 and 65, because refusals are cheap -- and
# serve_fuzz reported OK, since nothing was refused and plenty was answered.
# ⚠ It logged TWO completed generations, not zero: a handful of bodies survive
# the corruption by accident. So a `> 0` threshold would have passed this
# mutant, and only a floor well above the accidental survivors catches it. Ten
# is that floor; the clean arm cleared it at 90 s on the plain build and a real
# 600 s ASan soak is an order of magnitude past it.
n_gen=$(grep -cE "stoke serve: (chat|stream) id=" "$srvlog")
if [ "${n_gen:-0}" -lt 10 ]; then
    echo "serve_asan: FAILED -- the server logged only ${n_gen:-0} completed" \
         "generations. Traffic that is all refusals looks like a clean soak to" \
         "every other check here; the sanitizer saw the parser and nothing" \
         "downstream of it." >&2
    grep -m5 "stoke serve: " "$srvlog" | sed 's/^/  /' >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi

rm -f "$manifest" "$srvlog"
echo "serve_asan: OK (${secs}s, $clients clients, $n_corpus corpus files," \
     "seed $seed, no sanitizer reports, clean exit)"
exit 0
