#!/bin/sh
# tests/llama_gen_golden.sh -- GREEDY GENERATION parity for the second model.
#
# tests/tok_llama_golden.sh proves stoke's TOKENIZER matches llama.cpp on
# Llama-3.2-3B-Instruct. This gate proves the FORWARD PASS does: same prompt,
# greedy, and the decoded completion must match byte for byte. It is the
# standing form of B5's done-when, so that "Llama works" stops being a claim
# somebody made once and becomes something `make test` re-checks.
#
# ⚠ THE GOLDENS COME FROM llama.cpp, NOT FROM stoke. A golden regenerated from
# stoke's own output would pin stoke to itself and pass forever no matter how
# wrong both halves became -- it would be a change-detector, not a parity gate.
# tests/regen_llama_gen_goldens.sh regenerates them from the referent.
#
# ⚠⚠ THE REFERENT INVOCATION IS LOAD-BEARING AND EVERY FLAG IS THERE FOR A
# REASON THAT COST TIME TO FIND (2026-09-06):
#   llama-completion, NOT llama-cli   -- llama-cli AUTO-APPLIES THE CHAT
#       TEMPLATE for an instruct model, so it answers a DIFFERENT PROMPT and
#       every comparison against it is void.
#   --no-conversation, and NOT -st    -- `-st` (single turn) IMPLIES
#       conversation mode and silently re-enables the templating.
#   --override-kv ...add_bos_token=bool:false -- matches `stoke run --raw`
#       (add_special=false). Without it llama-bpe's raised add_bos default
#       prepends token 128000 and the streams differ at position 0.
#   -c sized per prompt               -- with no -c, llama.cpp allocates its KV
#       cache for the model's full 131072 training context (~15 GB).
#   --log-colors off + ANSI stripping -- the referent emits colour escapes into
#       stdout INTERMITTENTLY. Two defences, because one was not enough.
#   NOT --log-disable                 -- it suppresses the COMPLETION as well
#       as the logs, and yields an empty file that looks like a crash.
#
# Model-gated with a loud SKIP + exit 0 like every other golden gate.
#
# Usage: sh tests/llama_gen_golden.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: llama_gen_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
dir="tests/data/llama_gen"
stoke_bin="$builddir/stoke$exe"
ngen=64

golden_lib_require_model "$model" "llama_gen_golden"
golden_lib_require_binary "$stoke_bin" "llama_gen_golden"
golden_lib_require_data_file "$dir/cases.txt" "llama_gen_golden" "case list"

fail=0
n=0
while IFS=' ' read -r case ctx; do
    [ -n "${case:-}" ] || continue
    case "$case" in \#*) continue ;; esac
    prompt="$dir/$case.txt"
    golden="$dir/$case.golden"
    golden_lib_require_data_file "$prompt" "llama_gen_golden" "prompt"
    golden_lib_require_data_file "$golden" "llama_gen_golden" "golden"

    out="$builddir/llama_gen_$case.out"
    raw="$builddir/llama_gen_$case.raw"
    "$stoke_bin" run "$model" --raw -n "$ngen" -f "$prompt" > "$raw" 2>/dev/null
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "llama_gen_golden: FAILED -- stoke exited $rc on case '$case'" >&2
        fail=1; continue
    fi
    # `stoke run --raw` prints the generated ids, a bare "---", then the text.
    # This pipeline MUST match tests/regen_llama_gen_goldens.sh's exactly --
    # same ANSI strip, same trailing-whitespace strip, same trailing-blank-line
    # strip. Two normalisations that drift apart make the gate fail for a
    # reason that is neither engine's fault.
    sed -n '/^---$/,$p' "$raw" | sed '1d' | tr -d '\r' \
        | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[[:space:]]*$//' \
        | sed -e :a -e '/^$/{$d;N;ba' -e '}' > "$out"

    if diff -u "$golden" "$out" > "$builddir/llama_gen_$case.diff" 2>&1; then
        rm -f "$out" "$raw" "$builddir/llama_gen_$case.diff"
        n=$((n + 1))
    else
        echo "llama_gen_golden: FAILED -- case '$case' differs from $golden" >&2
        cat "$builddir/llama_gen_$case.diff" >&2
        fail=1
    fi
done < "$dir/cases.txt"

if [ "$fail" -ne 0 ]; then
    exit 1
fi
if [ "$n" -eq 0 ]; then
    # An empty case list would otherwise report a cheerful green having
    # checked nothing at all -- the exact shape of a vacuous gate.
    echo "llama_gen_golden: FAILED -- zero cases ran; $dir/cases.txt is empty" >&2
    exit 1
fi
echo "llama_gen_golden: OK ($n greedy completions match llama.cpp via $model)"
exit 0
