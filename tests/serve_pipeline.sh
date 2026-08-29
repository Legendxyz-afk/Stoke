#!/bin/sh
# tests/serve_pipeline.sh -- M6 rung A3: HTTP/1.1 pipelining, which the curl
# gate structurally cannot reach.
#
# tests/serve_golden.sh is gate item [2] and it is driven by curl, which has had
# no pipelining since 7.62. src/server.c nonetheless re-enters its read path
# after every flush BECAUSE a pipelining client's next request may already be in
# the buffer -- so the gate that drives this server cannot see the one path that
# comment is about. This script closes that: tests/serve_pipeline.c writes two
# requests in a single send() over the same socket shim the server uses.
#
# Deliberately NOT folded into serve_golden.sh: that gate's transcript is the
# committed golden for item [2] and stays curl-only, exactly as the gate is
# written. This is a separate binary answer -- two responses, or not.
#
# Usage: sh tests/serve_pipeline.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: serve_pipeline.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$builddir/stoke$exe"
client="$builddir/serve_pipeline$exe"
port="${STOKE_SERVE_PIPELINE_PORT:-18141}"

golden_lib_require_model "$model" "serve_pipeline"
golden_lib_require_binary "$bin" "serve_pipeline"
golden_lib_require_binary "$client" "serve_pipeline"

srvlog="$builddir/serve_pipeline.srv"
rm -f "$srvlog"

guard=""
if command -v timeout > /dev/null 2>&1; then
    guard="timeout ${STOKE_SERVE_GUARD_SECS:-600}"
fi

# Three: one to wait out the model load, then the pipelined pair. A server that
# answered only the first of the pair never reaches three and the guard ends it,
# which is a failure with a cause rather than a hung gate.
n_requests=3

$guard "$bin" serve "$model" --port "$port" --ctx 512 --seqs 2 --threads 8 \
    --max-requests "$n_requests" -v > "$srvlog" 2>&1 &
srv_pid=$!

# The readiness wait, borrowed from the curl gate rather than reimplemented as a
# connect-retry inside the client: a client with its own retry loop is a client
# that can hide a refused connection.
if ! curl -sS --retry 120 --retry-delay 1 --retry-connrefused \
        "http://127.0.0.1:$port/health" > /dev/null; then
    echo "serve_pipeline: FAILED -- the server never answered /health" >&2
    kill "$srv_pid" 2> /dev/null
    exit 1
fi

"$client" 127.0.0.1 "$port"
client_rc=$?

wait "$srv_pid"
srv_rc=$?

fail=0
if [ "$client_rc" -ne 0 ]; then
    fail=1
fi
if [ "$srv_rc" -ne 0 ]; then
    echo "serve_pipeline: FAILED -- the server exited $srv_rc" >&2
    fail=1
fi
if ! grep -q "^stoke serve: $n_requests requests completed, exiting$" "$srvlog"; then
    echo "serve_pipeline: FAILED -- the server did not report $n_requests" \
         "completed requests" >&2
    sed 's/^/  /' "$srvlog" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi

rm -f "$srvlog"
echo "serve_pipeline: OK (two requests in one write, two responses)"
exit 0
