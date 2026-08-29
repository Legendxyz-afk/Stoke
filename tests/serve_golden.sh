#!/bin/sh
# tests/serve_golden.sh -- M6 GATE ITEM [2]: the curl goldens.
#
# Starts `stoke serve` on the real model, drives it with curl over a fixed
# sequence of requests, and diffs the whole transcript against a committed
# golden. This is the first thing in the project that proves an HTTP client
# talking to this engine gets bytes back -- every test before it stopped at a
# function boundary.
#
# ▶▶ THE SERVER EXITS ON ITS OWN, and that is load-bearing rather than tidy.
# `--max-requests N` makes it serve exactly N requests and return; the
# alternative is killing it from this script, which RACES -- the kill can land
# before the last response is flushed and the golden then records a truncated
# body as if that were the answer. The count is asserted below, so a request
# that never reached the server cannot pass as one that did.
#
# ▶ WHAT IS NORMALIZED, AND THE ONE THING THAT IS CHECKED SEPARATELY BECAUSE
# NORMALIZING IT WOULD HIDE IT:
#   - "created":<unix time>  -> a placeholder. It is the only volatile field in
#     the response; the chatcmpl-N id is the request counter and IS stable.
#   - the port in the server's own startup line, so the port stays overridable.
#   - CR bytes are stripped, because this repo is core.autocrlf=true and a
#     golden carrying CRLF would be rewritten on checkout into a different file
#     under the same name. ⚠ That strip would ALSO hide a server that ended its
#     header lines with a bare LF -- an HTTP/1.1 protocol violation -- so the
#     raw bytes of the first response are checked for CRLF BEFORE stripping.
#
# Same model policy as every other golden gate: models/ and *.gguf are
# gitignored, so this SKIPS loudly (exit 0) where the model is absent and gates
# the build where it is present.
#
# Usage: sh tests/serve_golden.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: serve_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$builddir/stoke$exe"
golden="tests/data/serve.golden"

# Overridable so a box with something already on the default can move it; the
# port is normalized out of the transcript, so moving it cannot change the
# golden.
port="${STOKE_SERVE_PORT:-18139}"

# ⚠ PINNED. The generated text is greedy and therefore reproducible, but
# reproducible at a FIXED REDUCTION SHAPE -- thread count is an input to the
# arithmetic, not decoration.
#
# Measured 2026-08-29: this gate's first prompt gives a byte-identical response
# at --threads 4 and --threads 8. That is one 10-token generation and it is NOT
# a proof of invariance -- a longer answer has more places for a regrouped sum
# to change one logit and then every token after it. The pin stays because it
# costs nothing and removes the variable; the measurement is recorded so nobody
# re-derives it, and so nobody mistakes it for a guarantee.
threads=8
ctx=1024
seqs=4

golden_lib_require_model "$model" "serve_golden"
golden_lib_require_binary "$bin" "serve_golden"
golden_lib_require_data_file "$golden" "serve_golden" "golden file"

out="$builddir/serve_golden.out"
raw="$builddir/serve_golden.raw"
srvlog="$builddir/serve_golden.srv"
diffout="$builddir/serve_golden.diff"
rm -f "$out" "$raw" "$srvlog" "$diffout"

url="http://127.0.0.1:$port"
n_requests=8

# The first request has to outwait a 2.5GB model load, so it retries a refused
# connection. Every later one does NOT: by then the server is up, and a retry
# there would paper over exactly the failure this gate exists to catch.
first_retry="--retry 120 --retry-delay 1 --retry-connrefused"

# `timeout` is coreutils and is present on both legs (msys and WSL); where it
# is not, the run is simply unguarded rather than refused. The guard is what
# turns "the server never reached its request count" from a hung gate into a
# failed one, so it is a bound on the whole session and not a per-request one.
guard=""
if command -v timeout > /dev/null 2>&1; then
    guard="timeout ${STOKE_SERVE_GUARD_SECS:-600}"
fi

$guard "$bin" serve "$model" --port "$port" --ctx "$ctx" --seqs "$seqs" \
    --threads "$threads" --max-requests "$n_requests" -v > "$srvlog" 2>&1 &
srv_pid=$!

# run_case <label> <curl-args...> -- appends the labelled, normalized response
# to $out. curl's own exit code is recorded: an empty reply and a 400 look the
# same in a transcript that only keeps bodies.
run_case() {
    rc_label="$1"
    shift
    printf '### %s\n' "$rc_label" >> "$out"
    curl -sS -i --http1.1 "$@" > "$raw" 2> "$builddir/serve_golden.err"
    rc_case=$?
    printf '[curl exit %d]\n' "$rc_case" >> "$out"
    # curl -sS reports every REFUSED attempt while it is still retrying, even
    # when the retry then succeeds -- and that line carries both the port and
    # how many milliseconds the model load happened to take, neither of which
    # is a fact about the server's answer. Dropped rather than normalized. The
    # verdict for a connection that never succeeded is the exit code on the
    # line above, which no filter here can touch; anything else curl has to say
    # is kept, and a green run says nothing.
    grep -v 'connect to 127\.0\.0\.1 port' "$builddir/serve_golden.err" \
        | sed 's/^/[curl stderr] /' >> "$out"
    tr -d '\r' < "$raw" | sed 's/"created":[0-9]*/"created":<EPOCH>/' >> "$out"
    printf '\n' >> "$out"
}

# 1. Liveness, and the wait for the model load.
run_case "GET /health" $first_retry "$url/health"

# The CRLF check, on the bytes of that first response, before anything strips
# them.
#
# ⚠⚠ BYTE-EXACT, VIA printf AND cmp, AND THAT IS NOT FUSSINESS. The obvious
# spelling -- `od -c "$raw" | grep '\\r'` -- READS GREEN AGAINST A SERVER THAT
# EMITS BARE LF. This grep matches a bare letter `r` (BRE reduces `\\r` to the
# unknown escape `\r`, which GNU grep treats as the literal `r`) and does NOT
# match a backslash-r, so it fires on the `r` in "Content-Type" and grades
# nothing. It was written that way here first, and it took three tries to catch
# because the first two controls happened to contain no letter `r` at all --
# the fixture could not tell the two hypotheses apart. printf emits the two
# bytes and cmp compares them; neither has an escape convention to get wrong.
printf 'HTTP/1.1 200 OK\r\n' > "$builddir/serve_golden.crlf"
if ! head -c 17 "$raw" | cmp -s - "$builddir/serve_golden.crlf"; then
    echo "serve_golden: FAILED -- the first response's status line does not" \
         "end with CRLF (HTTP/1.1 requires it, and the transcript below strips" \
         "CR so nothing later in this gate could see it)" >&2
    od -c "$raw" | head -3 >&2
    kill "$srv_pid" 2> /dev/null
    exit 1
fi

# 2. The endpoint this milestone exists for.
run_case "POST /v1/chat/completions (greedy, max_tokens 24)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24}'

# 3. A system turn ahead of the user turn: proves the template renders more
#    than one message, which one message can never show.
run_case "POST /v1/chat/completions (system + user)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke-golden","messages":[{"role":"system","content":"You are terse."},{"role":"user","content":"Name one color."}],"max_tokens":24}'

# 4. Streaming is refused rather than silently answered whole -- a client that
#    asked for a stream and got one blob would hang waiting for events.
run_case "POST /v1/chat/completions (stream:true -> 501)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"hi"}],"stream":true}'

# 5-8. The error surface. Each carries its own reason string, so a server that
#      refused all four for one wrong cause cannot pass -- the same argument
#      the A2 corpus makes with its reason slugs.
#
# ⚠ THE ORDER IS LOAD-BEARING AT THE END OF THE LIST, NOT ONLY AT THE START.
# The Nth request is the one that races the exit: a server that counted a
# request as completed when it BUILT the response rather than when it FLUSHED
# it would end the loop with the last body still in the buffer, and the client
# would see an empty reply. That is not hypothetical -- it is what this server
# did when it first ran, on 2026-08-29 -- so the last case here is a request
# whose whole answer must survive the count.
run_case "POST /v1/chat/completions (body is not JSON -> 400)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d 'not json at all'

run_case "POST /v1/chat/completions (no messages -> 400)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke"}'

run_case "GET /v1/chat/completions (wrong method -> 405)" "$url/v1/chat/completions"

run_case "GET /nope (unknown endpoint -> 404)" "$url/nope"

wait "$srv_pid"
srv_rc=$?

# The transcript ends with the server's own account of the session. It carries
# the token counts and the finish reason, which no response body states, and
# the completed-request count -- the one line that distinguishes "eight
# requests answered" from "five answered and three lost".
{
    printf '### server log\n'
    sed "s/:$port /:<PORT> /" "$srvlog"
    printf '[server exit %d]\n' "$srv_rc"
} >> "$out"

fail=0
if [ "$srv_rc" -ne 0 ]; then
    echo "serve_golden: FAILED -- the server exited $srv_rc" >&2
    fail=1
fi
if ! grep -q "^stoke serve: $n_requests requests completed, exiting$" "$srvlog"; then
    echo "serve_golden: FAILED -- the server did not report $n_requests" \
         "completed requests; a request that never reached it would otherwise" \
         "pass as one that did" >&2
    fail=1
fi

if ! diff -u "$golden" "$out" > "$diffout" 2>&1; then
    echo "serve_golden: FAILED -- transcript differs from $golden" >&2
    cat "$diffout" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi

rm -f "$out" "$raw" "$diffout" "$builddir/serve_golden.err" \
      "$builddir/serve_golden.crlf" "$srvlog"
echo "serve_golden: OK ($n_requests curl requests matched $golden)"
exit 0
