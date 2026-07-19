#!/bin/sh
# tests/unicode_sweep_golden.sh -- script gate (mirrors tests/tok_golden.sh
# exactly): runs `stoke tokenize` (plain byte-BPE mode -- this gate is
# about the Unicode classification tables, not specials) on the real
# 2.5GB Qwen3 model over the committed fixed-stride Unicode sweep corpus
# (tests/unicode_sweep_fuzz.py; every codepoint 0x0000..0x10FFFF at stride
# 146, INCLUDING unassigned ones) and diffs it against the committed
# golden. This is the empirical close of the "src/unicode_data.h was
# generated against Python's Unicode 16.0, but the oracle may have been
# built against a different Unicode version" risk that the committed
# tok_corpus.txt corpus (built from a few hundred distinct codepoints in
# real-world text) never actually exercised across the whole codespace.
# models/ and *.gguf are gitignored, so the model exists on the reference
# box but not in a fresh clone -- this must SKIP cleanly (loud, explicit,
# exit 0) when the model is absent, and only actually gate the build when
# it's present.
#
# Shares its model-check/STOKE_REQUIRE_GOLDEN/binary-check/diff/cleanup
# body with golden_info.sh, tok_golden.sh and tok_special_golden.sh via
# tests/golden_lib.sh -- see that file for the shared policy this gate
# calls into.
#
# Usage: sh tests/unicode_sweep_golden.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: unicode_sweep_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
corpus="tests/data/unicode_sweep.txt"
golden="tests/data/unicode_sweep.ids.golden"
stoke_bin="$builddir/stoke$exe"

golden_lib_require_model "$model" "unicode_sweep_golden"
golden_lib_require_binary "$stoke_bin" "unicode_sweep_golden"
golden_lib_require_data_file "$corpus" "unicode_sweep_golden" "corpus file"
golden_lib_require_data_file "$golden" "unicode_sweep_golden" "golden file"

golden_lib_run_and_diff "unicode_sweep_golden" "$golden" \
    "$builddir/unicode_sweep.ids.out" "$builddir/unicode_sweep.ids.diff" \
    "$stoke_bin tokenize $model $corpus" \
    "$stoke_bin" tokenize "$model" "$corpus" || exit 1

echo "unicode_sweep_golden: OK ($corpus matches $golden via $model)"
exit 0
