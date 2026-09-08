#!/bin/sh
# tests/tok_unsloth_golden.sh -- B6's TOKENIZER half: the unsloth Q4_K_M file,
# byte-exact, M1-style.
#
# The binding ruling keeps the tokenizer gate "M1-style byte-exact via
# llama-tokenize (discrete domain)" for every model. This is that gate for the
# second FILE of the first architecture.
#
# ⭐ WHY IT REUSES THE BARTOWSKI GOLDENS AND WHY THAT IS THE RIGHT ANSWER, NOT A
# SHORTCUT. The two Q4_K_M files carry the SAME tokenizer -- same vocab, same
# merges, same specials; the 2026-09-05 metadata diff found the differences are
# `general.base_model.*`, `quantized_by`, `repo_url`, an extra tag and
# `padding_token_id`, and the WEIGHTS (different imatrix: 497 vs 79 calibration
# chunks). So the correct expectation is that this file reproduces the goldens
# EXACTLY, and asserting it is how "the tokenizer transfers" stops being an
# assumption. Verified against the live oracle before being written down
# (2026-09-08): all four legs match `llama-tokenize` on the unsloth file too.
#
# ⚠⚠ AND THEREFORE ITS GREEN IS WORTH LESS THAN IT LOOKS, WHICH IS WHY LEG 0
# EXISTS. A gate that passes because two files are identical would pass just as
# happily if it silently read the OTHER file -- the "one arm never ran" failure
# in [[fixture-can-be-blind-to-the-defect]] #11. Leg 0 asserts stoke is reading
# the file this gate names, using a metadata fact that is TRUE of the unsloth
# file and FALSE of bartowski's (42 kv, and general.name with hyphens).
#
# ⚠ It says nothing whatever about the FORWARD pass on this file. Different
# imatrix means different weights; that is B6's other half and is not here.
#
# Usage: sh tests/tok_unsloth_golden.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: tok_unsloth_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"
stoke_bin="$builddir/stoke$exe"
model="models/unsloth-Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

golden_lib_require_model "$model" "tok_unsloth_golden"
golden_lib_require_binary "$stoke_bin" "tok_unsloth_golden"

fail=0
info="$builddir/tok_unsloth_info.txt"

# ---- leg 0: prove we are reading THE FILE THIS GATE NAMES -------------------
"$stoke_bin" info "$model" > "$info" 2>/dev/null || {
    echo "tok_unsloth_golden: FAILED -- stoke info could not read $model" >&2
    exit 1
}
if ! grep -q "^== metadata (42 kv) ==" "$info"; then
    echo "tok_unsloth_golden: FAILED -- expected 42 metadata kv (the unsloth" \
         "file); bartowski's has 36. This gate is not reading the file it" \
         "names, so its other legs prove nothing" >&2
    head -1 "$info" >&2
    fail=1
fi
if ! grep -q "general.name (STRING) = Qwen3-4B-Instruct-2507$" "$info"; then
    echo "tok_unsloth_golden: FAILED -- general.name is not the unsloth" \
         "spelling (hyphens); bartowski's is 'Qwen3 4B Instruct 2507'" >&2
    fail=1
fi
[ "$fail" -eq 0 ] && echo "tok_unsloth_golden: leg 0 OK -- 42 kv and the unsloth general.name"
rm -f "$info"

# ---- legs 1-4: the four byte-exact comparisons ------------------------------
# Same corpora and same goldens as the bartowski gates; only the MODEL changes.
leg() {  # $1 corpus, $2 golden, $3 label, $4... extra stoke flags
    corpus="$1"; golden="$2"; label="$3"; shift 3
    golden_lib_require_data_file "$corpus" "tok_unsloth_golden" "corpus"
    golden_lib_require_data_file "$golden" "tok_unsloth_golden" "golden"
    out="$builddir/tok_unsloth_$label.ids"
    "$stoke_bin" tokenize "$@" "$model" "$corpus" 2>/dev/null \
        | tr -d '\r' | grep -o '^\[.*\]$' > "$out"
    if [ ! -s "$out" ]; then
        echo "tok_unsloth_golden: FAILED -- $label produced no bracketed id" \
             "line; an empty result must not compare equal to anything" >&2
        return 1
    fi
    if cmp -s "$out" "$golden"; then
        n=$(tr ',' '\n' < "$out" | wc -l | tr -d ' ')
        echo "tok_unsloth_golden: $label OK ($n ids match $(basename "$golden"))"
        rm -f "$out"
        return 0
    fi
    echo "tok_unsloth_golden: FAILED -- $label differs from" \
         "$(basename "$golden"); see $out" >&2
    return 1
}

leg tests/data/tok_corpus.txt tests/data/tok_corpus.ids.golden corpus || fail=1
leg tests/data/unicode_sweep.txt tests/data/unicode_sweep.ids.golden sweep || fail=1
leg tests/data/tok_special_corpus.txt tests/data/tok_special_corpus.plain.golden \
    special-plain || fail=1
leg tests/data/tok_special_corpus.txt tests/data/tok_special_corpus.ids.golden \
    special-on --special || fail=1

if [ "$fail" -ne 0 ]; then
    echo "tok_unsloth_golden: FAILED" >&2
    exit 1
fi
echo "tok_unsloth_golden: OK (B6's tokenizer half -- 4 legs byte-exact)"
exit 0
