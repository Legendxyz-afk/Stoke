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
port="${STOKE_SERVE_PORT:-$(golden_lib_port 18139 "$builddir")}"

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
n_requests=19

# The first request has to outwait a 2.5GB model load, so it retries a refused
# connection. Every later one does NOT: by then the server is up, and a retry
# there would paper over exactly the failure this gate exists to catch.
first_retry="--retry 400 --retry-delay 1 --retry-connrefused"

# `timeout` is coreutils and is present on both legs (msys and WSL); where it
# is not, the run is simply unguarded rather than refused. The guard is what
# turns "the server never reached its request count" from a hung gate into a
# failed one, so it is a bound on the whole session and not a per-request one.
guard=$(golden_lib_guard "$builddir")

# ⚠ --idle-timeout 2 IS A REGRESSION GUARD, NOT TUNING. The reaper used to close
# CONN_WRITE connections too, and `last_active` is stamped when bytes move -- so
# a request whose GENERATION outlasted the timeout arrived at "ready to send"
# already stale, and was closed on the next turn, throwing away a response the
# engine had already produced. The client saw `curl: (52) Empty reply from
# server` while the server's own log said `finish=length`. Found on the TSan
# leg, where a forward pass is slow enough to cross sixty seconds; in production
# it is any long generation. Two seconds here means EVERY generating case in
# this gate crosses the timeout, so the bug cannot come back quietly.
$guard "$bin" serve "$model" --port "$port" --ctx "$ctx" --seqs "$seqs" \
    --threads "$threads" --max-requests "$n_requests" --idle-timeout 2 \
    -v > "$srvlog" 2>&1 &
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

# ---- the SSE assertions -------------------------------------------------
#
# ▶▶ NAMED, NOT ONLY DIFFED. The transcript diff below catches any change at
# all, which is what a golden is for -- and it explains nothing: every OpenAI
# compat trap that regresses reports as the same wall of red. These name the
# trap. It is the A2 corpus's reason-slug argument applied to a protocol
# instead of to a parser.
#
# ⚠ grep -F EVERYWHERE, deliberately. `data: [DONE]` read as a BRE is "data: "
# followed by one character out of D/O/N/E, which matches most lines in an SSE
# stream -- an assertion that can only pass. Fixed strings have no
# metacharacters to get wrong. This gate has already shipped one pattern whose
# escape it got wrong; see the CRLF check above.
sse_fail=0
sse_assert() { # <file> <what> <fixed-needle>
    if ! grep -qF -- "$3" "$1"; then
        echo "serve_golden: FAILED -- $2" >&2
        echo "               expected to find: $3" >&2
        sse_fail=1
    fi
}
sse_refute() { # <file> <what> <fixed-needle>
    if grep -qF -- "$3" "$1"; then
        echo "serve_golden: FAILED -- $2" >&2
        echo "               must not contain: $3" >&2
        sse_fail=1
    fi
}

# 4. THE ENDPOINT AS A STREAM. curl decodes the chunked framing here, so what
#    the transcript records is the event stream a client actually sees.
run_case "POST /v1/chat/completions (stream:true)" \
    -N -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24,"stream":true}'
cp "$raw" "$builddir/serve_golden.sse"
sse="$builddir/serve_golden.sse"

sse_assert "$sse" "a stream must be text/event-stream" \
    "Content-Type: text/event-stream"
sse_assert "$sse" "a stream is framed by chunked transfer-encoding, not a length" \
    "Transfer-Encoding: chunked"
sse_refute "$sse" "a stream cannot carry a Content-Length: nothing knows it when the head goes out" \
    "Content-Length:"
sse_assert "$sse" "the stream must terminate with the sentinel event" \
    "data: [DONE]"
sse_assert "$sse" "the final content chunk carries the finish reason" \
    '"finish_reason":"stop"'
sse_refute "$sse" "usage is sent ONLY under stream_options.include_usage" \
    '"usage"'

# The FIRST delta is its own trap twice over: openai-python reads `role` off it
# and off no other chunk, and expects finish_reason null until the last one.
# Both are properties of ONE chunk, so both are checked on that chunk -- "the
# stream carries a role somewhere" is a weaker claim a broken server can meet.
first_delta=$(grep -m1 '^data: {' "$sse")
case "$first_delta" in
    *'"role":"assistant"'*) ;;
    *) echo "serve_golden: FAILED -- the first delta must carry role=assistant" >&2
       echo "               first delta was: $first_delta" >&2
       sse_fail=1 ;;
esac
case "$first_delta" in
    *'"finish_reason":null'*) ;;
    *) echo "serve_golden: FAILED -- finish_reason stays null until the final chunk" >&2
       echo "               first delta was: $first_delta" >&2
       sse_fail=1 ;;
esac

# 5. stream_options.include_usage. OpenAI sends usage as its own final chunk
#    with an EMPTY choices array, AFTER the chunk carrying finish_reason: a
#    client reading usage off the finish chunk gets nothing, and one that stops
#    at finish_reason never sees it at all.
run_case "POST /v1/chat/completions (stream + include_usage)" \
    -N -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Name one color."}],"max_tokens":24,"stream":true,"stream_options":{"include_usage":true}}'
cp "$raw" "$builddir/serve_golden.usage"
usg="$builddir/serve_golden.usage"

sse_assert "$usg" "include_usage must produce a usage chunk" '"usage":{"prompt_tokens"'
sse_assert "$usg" "the usage chunk carries an empty choices array" '"choices":[]'
sse_assert "$usg" "the stream still terminates with the sentinel" "data: [DONE]"
fin_line=$(grep -nF -- '"finish_reason":"stop"' "$usg" | head -1 | cut -d: -f1)
usg_line=$(grep -nF -- '"usage":{"prompt_tokens"' "$usg" | head -1 | cut -d: -f1)
if [ -z "$fin_line" ] || [ -z "$usg_line" ] || [ "$usg_line" -le "$fin_line" ]; then
    echo "serve_golden: FAILED -- the usage chunk must come AFTER the chunk" \
         "carrying finish_reason (finish at line ${fin_line:-none}, usage at" \
         "line ${usg_line:-none})" >&2
    sse_fail=1
fi

# 6. THE WIRE, UNDECODED. curl --raw leaves the chunked framing in place, so
#    this case pins the bytes on the socket instead of what curl made of them:
#    the hex chunk sizes, the CRLF around each, and the terminating zero chunk.
#    Everything above would still pass against a server whose framing was
#    subtly wrong and whose client was forgiving.
run_case "POST /v1/chat/completions (stream, --raw wire bytes)" \
    -N --raw -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Name one color."}],"max_tokens":8,"stream":true}'

if [ "$sse_fail" -ne 0 ]; then
    echo "serve_golden: FAILED -- see the named SSE failures above" >&2
fi

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

# ---- A5: the two endpoints gate item [1] names, and the one parameter the M0
#      harness sends that this build does not implement -------------------------

# 9. /v1/models. openai-python calls it first in almost every script, and a
#    client that cannot list models cannot pick one.
run_case "GET /v1/models" "$url/v1/models"
sse_assert "$raw" "the models list is an object of type list" '"object":"list"'
sse_assert "$raw" "each entry is a model object" '"object":"model"'

# 10. /v1/completions, RAW -- no chat template. This is the M2 parity path's
#     shape: a prompt is bytes, not a conversation, and templating it would
#     silently change what the model was asked.
run_case "POST /v1/completions (raw, non-stream)" \
    -X POST "$url/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","prompt":"The capital of France is","max_tokens":8}'
sse_assert "$raw" "a completion is an object of type text_completion" \
    '"object":"text_completion"'
sse_assert "$raw" "a completion choice carries text, not a message" '"text":'
sse_refute "$raw" "a completion choice must NOT carry a chat message" '"message":'

# 11. The same endpoint streamed. Its chunks carry `text`, and a client that
#     reads delta.content off them gets nothing -- which is why the two shapes
#     are asserted apart rather than assumed to be the same code path.
run_case "POST /v1/completions (raw, streamed)" \
    -N -X POST "$url/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","prompt":"The capital of France is","max_tokens":8,"stream":true}'
sse_assert "$raw" "a streamed completion terminates with the sentinel" "data: [DONE]"
sse_assert "$raw" "streamed completion chunks carry text" '"text":'
sse_refute "$raw" "streamed completion chunks must NOT carry a chat delta" '"delta":'

# 12-13. temperature. The M0 harness sends `"temperature": 0.0` on EVERY
#     request, and greedy decoding IS temperature zero -- so zero is accepted
#     and nothing about it is special-cased. A NON-zero temperature is refused
#     rather than silently answered greedily: this build has no sampler, and
#     quietly returning the wrong distribution is the failure mode that takes
#     weeks to notice. Sampling arrives at A6 and this refusal goes with it.
run_case "POST /v1/chat/completions (temperature 0 -> 200)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Name one color."}],"max_tokens":8,"temperature":0.0}'
sse_assert "$raw" "temperature 0 is greedy and must be accepted" "HTTP/1.1 200 OK"

run_case "POST /v1/chat/completions (temperature 0.7 -> 400)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Name one color."}],"max_tokens":8,"temperature":0.7}'
sse_assert "$raw" "a non-zero temperature must be refused, not silently ignored" \
    "HTTP/1.1 400 Bad Request"

# ---- A6: stop strings ------------------------------------------------------
#
# ▶▶ THE THIRD CASE IS THE ONE THAT MATTERS AND IT IS PINNED ON PURPOSE. Case 2
# above establishes that this prompt answers "Hello! How can I help you today?"
# and that it arrives as the tokens "Hello" "!" " How" " can" " I" " help" ...
# -- the streamed transcript shows each one. So "an I" is a stop string that
# EXISTS IN THE OUTPUT AND IN NO SINGLE TOKEN: its "an" is the tail of " can"
# and its " I" is the whole of the next token. A server that checks each token
# against the stop list in isolation cannot see it, and that is precisely the
# defect the tail window exists to prevent. Everything else here would pass
# against such a server.

# 14. A stop string inside one token, non-streamed: the plain case.
run_case "POST /v1/chat/completions (stop, within one token)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24,"stop":["!"]}'
sse_assert "$raw" "a stop string truncates the answer before itself" '"content":"Hello"'
sse_assert "$raw" "stopping on a stop string reports finish_reason stop" \
    '"finish_reason":"stop"'

# 15. THE CROSS-TOKEN CASE, streamed. "an I" spans " can" and " I".
run_case "POST /v1/chat/completions (stop ACROSS a token boundary, streamed)" \
    -N -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24,"stream":true,"stop":["an I"]}'
sse_assert "$raw" "a stop string spanning two tokens must still be found" \
    '"finish_reason":"stop"'
sse_assert "$raw" "the stream still terminates with the sentinel after a stop" \
    "data: [DONE]"
# The overshoot: not one byte of the stop string, nor of anything after it, may
# reach the client. " help" is the token AFTER the match and its absence is the
# assertion -- a server that cancelled but emitted the token in flight would
# still show it.
sse_refute "$raw" "no text at or beyond the stop string may be emitted" '" help"'
sse_refute "$raw" "the stop string itself must not be emitted" '"an I"'

# 16-17. The rest of the sampling surface, on the same principle as
#     temperature: a parameter is accepted at the value where "no sampler" is
#     the CORRECT answer, and refused at any other. top_p 1 truncates nothing;
#     top_p 0.9 does, and answering it greedily would be answering a different
#     question. `seed` is deliberately accepted: greedy is already
#     deterministic, so any seed gives the identical stream, which is what a
#     seed promises.
run_case "POST /v1/chat/completions (top_p 1 + seed -> 200)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Name one color."}],"max_tokens":8,"top_p":1,"seed":42,"n":1}'
sse_assert "$raw" "the no-op value of every sampling parameter is accepted" \
    "HTTP/1.1 200 OK"

run_case "POST /v1/chat/completions (top_p 0.9 -> 400)" \
    -X POST "$url/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"stoke","messages":[{"role":"user","content":"Name one color."}],"max_tokens":8,"top_p":0.9}'
sse_assert "$raw" "a sampling parameter that would change the output is refused" \
    "HTTP/1.1 400 Bad Request"

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

fail=$sse_fail
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
      "$builddir/serve_golden.crlf" "$builddir/serve_golden.sse" "$builddir/serve_golden.usage" "$srvlog"
echo "serve_golden: OK ($n_requests curl requests matched $golden)"
exit 0
