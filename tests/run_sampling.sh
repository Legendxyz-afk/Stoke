#!/bin/sh
# tests/run_sampling.sh -- M7 rung B4: `stoke run` can sample, and it is the
# same sampler the server uses.
#
# ▶▶ WHY THE CLI NEEDED THIS. Until now the only way to exercise the sampler was
# to start a server and speak HTTP to it. `stoke run` is the parity and
# debugging path -- it is where a divergence gets bisected, where --probe-pos
# and --dump live -- and it could only decode greedily. A sampler you cannot
# reproduce from a one-shot command is a sampler you debug through a socket.
#
# ▶ EVERY ASSERTION IS A RELATION BETWEEN TWO RUNS, never a pinned string, and
# for the same reason tests/serve_openai.py works that way: a sampler that is
# parsed and then dropped produces perfectly fluent output. The only observable
# difference between a live sampler and a discarded flag is whether two runs
# that must differ do.
#
# ⚠ AND EACH IS PAIRED WITH THE CONTROL THAT MAKES IT NON-VACUOUS. "The same
# seed reproduces" is free for a binary that ignores every flag; its control is
# "a different seed differs". "top_k 1 collapses to the argmax" is free for a
# binary that ignores --temp; its control is "a sampled run differs from
# greedy". Neither half is trusted alone.
#
# ▶ COMPARISONS ARE ON THE ID LIST, not the decoded text. Ids are what a sampler
# chooses; text is what a detokenizer makes of them, and comparing text would
# put an unrelated component between the assertion and the thing under test.
#
# Usage: sh tests/run_sampling.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: run_sampling.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$builddir/stoke$exe"

golden_lib_require_model "$model" "run_sampling"
golden_lib_require_binary "$bin" "run_sampling"

fail=0
tmp="$builddir/run_sampling.out"

# ids_of <label> <extra args...> -- runs `stoke run` and echoes its id list,
# one id per line. `stoke run` prints the ids, then a `---` line, then the text.
ids_of() {
    rs_label="$1"
    shift
    if ! "$bin" run "$model" -p "$PROMPT" -n "$NGEN" --threads 4 "$@" \
            > "$tmp" 2>&1; then
        echo "run_sampling: FAILED -- '$rs_label' exited nonzero" >&2
        tail -3 "$tmp" >&2
        fail=1
        return
    fi
    sed -n '1,/^---$/p' "$tmp" | grep -E '^[0-9]+$'
}

# ⚠ AN EMPTY OPERAND IS A FAILURE IN BOTH DIRECTIONS, never a match. If the id
# extraction broke -- a changed output shape, a run that died -- every "these
# must agree" below would agree about nothing and pass.
same() { # <what> <a> <b>
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "run_sampling: FAILED -- $1: an operand was empty, so the" \
             "comparison graded nothing" >&2
        fail=1
    elif [ "$2" != "$3" ]; then
        echo "run_sampling: FAILED -- $1" >&2
        echo "               a: $(echo "$2" | tr '\n' ' ')" >&2
        echo "               b: $(echo "$3" | tr '\n' ' ')" >&2
        fail=1
    fi
}
differ() { # <what> <a> <b>
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "run_sampling: FAILED -- $1: an operand was empty, so the" \
             "comparison graded nothing" >&2
        fail=1
    elif [ "$2" = "$3" ]; then
        echo "run_sampling: FAILED -- $1" >&2
        echo "               both were: $(echo "$2" | tr '\n' ' ')" >&2
        fail=1
    fi
}

PROMPT="Write one sentence about the sea."
NGEN=16
SEED=20260830

greedy=$(ids_of "greedy baseline")
wild_a=$(ids_of "temp 1.4"             --temp 1.4 --seed "$SEED")
wild_b=$(ids_of "temp 1.4, same seed"  --temp 1.4 --seed "$SEED")
wild_c=$(ids_of "temp 1.4, other seed" --temp 1.4 --seed $((SEED + 1)))
k1=$(ids_of     "temp 1.4 + top_k 1"   --temp 1.4 --seed "$SEED" --top-k 1)

differ "a sampled run must not reproduce the greedy ids -- if it does, --temp \
never reached the sampler" "$greedy" "$wild_a"
same "the same seed must give the same ids; a seed that does not reproduce is \
not a seed" "$wild_a" "$wild_b"
differ "a different seed must give different ids -- without this, the \
reproduction above is satisfied by a binary that ignores every flag" \
    "$wild_a" "$wild_c"
same "top_k 1 leaves one candidate, so any temperature must still give the \
greedy ids" "$greedy" "$k1"

# The repeat penalty, on the greedy path and on a prompt whose answer repeats a
# token -- the same prompt the serve golden uses, chosen there by reading the
# token stream first. Without a repeat there is nothing for the penalty to bite
# and the case would pass against a binary that dropped the flag.
PROMPT="Repeat the word banana three times."
NGEN=24
plain=$(ids_of     "banana, greedy")
penal=$(ids_of     "banana, repeat_penalty 1.8" --repeat-penalty 1.8)
differ "a repeat penalty on an answer that repeats a token MUST change it" \
    "$plain" "$penal"

# ---- the CLI and the SERVER must agree, and this is the case that found a bug
#
# ▶▶ THE POINT OF TEACHING THE CLI TO SAMPLE is that a divergence can be
# bisected with a one-shot command instead of through a socket -- which is worth
# nothing unless the two paths actually agree. They did not: the CLI keyed each
# draw on `pos`, the position the NEXT forward will occupy, while the engine
# keys it on the position whose logits produced the id -- the prefill's LAST
# row, n_prompt - 1. Every draw was shifted by one and the CLI returned a
# different, perfectly fluent continuation for the same seed.
#
# ⚠ NOTHING ABOUT THAT WAS VISIBLE FROM EITHER SIDE ALONE. Both paths sampled,
# both reproduced their own seed, both passed every relation above. It was found
# by running the two and diffing them, and this case is that diff, kept.
PROMPT="Write one sentence about the sea."
NGEN=16
cli_ids=$(ids_of "cross-path CLI arm" --temp 1.4 --seed "$SEED")
cli_text=$(awk '/^---$/{f=1;next} f' "$tmp")

port="${STOKE_RUN_SAMPLING_PORT:-$(golden_lib_port 18151 "$builddir")}"
guard=$(golden_lib_guard "$builddir")
srvlog="$builddir/run_sampling.srv"
$guard "$bin" serve "$model" --port "$port" --ctx 1024 --seqs 2 --threads 4 \
    --max-requests 2 > "$srvlog" 2>&1 &
srv_pid=$!
if curl -sS --retry 400 --retry-delay 1 --retry-connrefused \
        "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
    srv_text=$(curl -sS -X POST "http://127.0.0.1:$port/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"stoke\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$NGEN,\"temperature\":1.4,\"seed\":$SEED}" \
        | sed -n 's/.*"content":"\([^"]*\)".*/\1/p')
    same "the CLI and the server must produce the SAME sampled text for the same \
prompt, seed and parameters -- they are one sampler keyed on (seed, absolute \
position), and a mismatch means one of them is keying it differently" \
        "$cli_text" "$srv_text"
else
    echo "run_sampling: FAILED -- the server never answered /health" >&2
    kill "$srv_pid" 2> /dev/null
    fail=1
fi
wait "$srv_pid" 2> /dev/null

if [ "$fail" -ne 0 ]; then
    exit 1
fi
rm -f "$tmp" "$srvlog"
echo "run_sampling: OK (the four flags reach the sampler, and the CLI matches" \
     "the server byte for byte on the same seed)"
exit 0
