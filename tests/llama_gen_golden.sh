#!/bin/sh
# tests/llama_gen_golden.sh -- GREEDY GENERATION regression gate for the
# second model.
#
# ⚠⚠ READ THIS BEFORE CITING THIS GATE AS M7 PARITY. IT IS NOT THAT.
# The binding Fable ruling of 2026-08-05 (docs/ROADMAP.md:619-627, and the M3
# RULING memo it re-instantiates) says M7's FORWARD-PASS parity criterion is
# the §M3 v2 shape -- twins + a ref probe grid + targeted tests for the deltas,
# with τ re-frozen from M7's own entry probe over position bands
# [600,1200)/[1200,∞) BEFORE any parity evidence is graded -- and that it is
# "NEVER a 50/50 live-oracle token bar for any model". A byte-exact live-oracle
# generation comparison is structurally that forbidden shape.
#
# So what IS this? A REGRESSION DETECTOR with a referent-derived baseline. It
# pins stoke's current behaviour on four prompts against llama.cpp as of
# b10037, and it will catch a change in stoke that moves any of them. That is
# genuinely useful and it is why it is wired into `make test`.
#
# ⚠ AND ITS CASES ARE SELECTED ON SUCCESS, which must be stated because it
# bounds what a green here means: these four prompts are the ones that came
# back byte-identical. Others did NOT -- a 1426-token prompt diverges by one
# token, and on the unsloth Qwen file three of four prompts diverge. Those are
# not in here, so this gate's green says "these four still agree", never
# "stoke and llama.cpp agree".
#
# tests/tok_llama_golden.sh is different and IS the ruled criterion for its
# half: the tokenizer gate is M1-style byte-exact via llama-tokenize, exactly
# as the ruling requires, because tokenization is a discrete domain where
# exactness is reachable.
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
#
# ⚠⚠⚠ OPT-IN, AND THE REASON IS MEASURED, NOT CAUTIONARY. This gate was wired
# into `make test` on 2026-09-06 and the Linux leg immediately failed it while
# Windows passed: the SAME source and model produce different greedy text on
# the two platforms.
#     short2  Windows "if n <= 0: return 0"   Linux "if n <= 1: return n"
#     short3  Windows "three astronauts"      Linux "astronauts"
# `make` aborts on the first failure, so it also stopped serve_golden,
# run_sampling, serve_raw and serve_openai from running at all.
#
# In the SAME Linux run every discrete-domain golden passed -- golden_info,
# tok_golden, tok_special_golden, unicode_sweep_golden and tok_llama_golden.
# That contrast is the whole point: TOKENIZATION is exact across platforms and
# GENERATION is not, because the forward pass runs f32 transcendentals
# (rope's cosf/sinf/powf) through a different libm and a different compiler on
# each leg. A byte-exact generation bar is therefore not portable, which is
# precisely why the binding ruling says forward-pass parity must never be one.
#
# So it runs only when asked: STOKE_RUN_GEN_PARITY=1 sh tests/llama_gen_golden.sh ...
# and the goldens it compares against are WINDOWS-generated.
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: llama_gen_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

if [ "${STOKE_RUN_GEN_PARITY:-0}" != "1" ]; then
    echo "llama_gen_golden: SKIP -- opt-in only (set STOKE_RUN_GEN_PARITY=1)." \
         "Greedy text is NOT reproducible across platforms: this gate's" \
         "goldens are Windows-generated and the Linux leg differs on 2 of 4" \
         "cases, while every tokenizer golden passes on both."
    exit 0
fi

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
