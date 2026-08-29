#!/bin/sh
# tests/serve_raw.sh -- the three server behaviours the curl gate cannot reach.
# M6 rungs A3 (pipelining) and A4 (heartbeat, slow-client policy).
#
# tests/serve_golden.sh is gate item [2] and stays curl-only, exactly as the
# gate is written. This drives tests/serve_probe.c instead, because:
#   - curl has had no HTTP/1.1 pipelining since 7.62;
#   - a fifteen-second heartbeat cannot appear inside a gate that has to
#     finish, and its arrival time is not byte-stable, so it cannot live in a
#     golden either;
#   - and proving the server drops a client that stopped reading means BEING a
#     client that stopped reading, which curl will never do.
#
# ▶ ONE SERVER, THREE PROBES. The two tuning options below are what make the
# last two observable at all -- a 15 s heartbeat and a 256 KiB backlog are
# real defaults and unreachable ones. Neither affects the pipelining probe,
# whose requests are /health and never stream, so all three share one 2.5 GB
# model load instead of three.
#
# Usage: sh tests/serve_raw.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: serve_raw.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$builddir/stoke$exe"
probe="$builddir/serve_probe$exe"
port="${STOKE_SERVE_RAW_PORT:-18141}"

golden_lib_require_model "$model" "serve_raw"
golden_lib_require_binary "$bin" "serve_raw"
golden_lib_require_binary "$probe" "serve_raw"

srvlog="$builddir/serve_raw.srv"
rm -f "$srvlog"

guard=""
if command -v timeout > /dev/null 2>&1; then
    guard="timeout ${STOKE_SERVE_GUARD_SECS:-600}"
fi

# One to wait out the model load, two for the pipelined pair, one stream for
# the heartbeat. The slow reader is NOT counted: it is disconnected rather than
# answered, which is the whole point, and a server waiting for it to complete
# would never exit. So the count is 4 and the guard covers the rest.
n_requests=4

# ⚠ --sse-ping-ms 50: short enough that the comment fires during the PREFILL
# of the probe's deliberately long prompt, which is the only real silence this
# server has -- steady generation has no gap, because the tokens are the
# heartbeat. --sse-max-pending 4096: small enough that a client which stops
# reading trips the policy once the kernel buffers between us are full.
$guard "$bin" serve "$model" --port "$port" --ctx 1024 --seqs 2 --threads 8 \
    --max-requests "$n_requests" --sse-ping-ms 50 --sse-max-pending 4096 \
    --sse-send-buffer 4096 -v \
    > "$srvlog" 2>&1 &
srv_pid=$!

# The readiness wait, borrowed from the curl gate rather than reimplemented as
# a connect-retry inside the probe: a client with its own retry loop is a
# client that can hide a refused connection.
if ! curl -sS --retry 120 --retry-delay 1 --retry-connrefused \
        "http://127.0.0.1:$port/health" > /dev/null; then
    echo "serve_raw: FAILED -- the server never answered /health" >&2
    kill "$srv_pid" 2> /dev/null
    exit 1
fi

fail=0
# ⚠ THE ORDER IS FORCED BY THE COUNT. `slowread` is never ANSWERED -- it is
# dropped, which is the whole point -- so it never increments the completed
# count, and the count is what ends the server. Put it last and the fourth
# completion (the ping stream) would have already ended the run before it could
# connect. So it goes in the middle, and the stream that completes goes last.
# A sanitized build runs a forward pass an order of magnitude slower, so the
# probes need a deadline to match. Detected from the build directory rather
# than passed in, because `make test` invokes this with $(BUILDDIR) and
# nothing else knows which leg it is. A deadline tighter than the work is
# how you manufacture an intermittent red.
probe_deadline=120000
case "$builddir" in
    *asan*|*tsan*) probe_deadline=900000 ;;
esac

for mode in pipeline slowread ping; do
    if ! "$probe" "$mode" 127.0.0.1 "$port" "$probe_deadline"; then
        fail=1
    fi
done

wait "$srv_pid"
srv_rc=$?

if [ "$srv_rc" -ne 0 ]; then
    echo "serve_raw: FAILED -- the server exited $srv_rc" >&2
    fail=1
fi
if ! grep -q "^stoke serve: $n_requests requests completed, exiting$" "$srvlog"; then
    echo "serve_raw: FAILED -- the server did not report $n_requests completed" \
         "requests" >&2
    sed 's/^/  /' "$srvlog" >&2
    fail=1
fi
# The slow reader must show up in the log as a DROP, not as a completion. Its
# absence would mean the probe passed for some other reason -- a stream that
# simply ended, say -- and the policy went untested.
if ! grep -q "dropped --" "$srvlog"; then
    echo "serve_raw: FAILED -- the server never logged dropping a slow" \
         "client; the slowread probe passed without the policy firing" >&2
    sed 's/^/  /' "$srvlog" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi

rm -f "$srvlog"
echo "serve_raw: OK (pipelining, heartbeat, and the slow-client policy)"
exit 0
