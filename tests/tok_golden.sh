#!/bin/sh
# tests/tok_golden.sh -- script gate (mirrors tests/golden_info.sh
# exactly): runs `stoke tokenize` on the real 2.5GB
# Qwen3 model over the committed fuzz corpus and diffs it against the
# committed golden (produced by the real llama-tokenize.exe oracle --
# see tests/tok_fuzz.py's docstring for the exact regeneration command).
# models/ and *.gguf are gitignored, so the model exists on the reference
# box but not in a fresh clone -- this must SKIP cleanly (loud, explicit,
# exit 0) when the model is absent, and only actually gate the build when
# it's present. tests/test_tok.c's own unit tests use a tiny committed
# synthetic fixture and always run regardless of this gate.
#
# Shares its model-check/STOKE_REQUIRE_GOLDEN/binary-check/diff/cleanup
# body with golden_info.sh, tok_special_golden.sh and
# unicode_sweep_golden.sh via tests/golden_lib.sh -- see that file for the
# shared policy this gate calls into.
#
# Usage: sh tests/tok_golden.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: tok_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
corpus="tests/data/tok_corpus.txt"
golden="tests/data/tok_corpus.ids.golden"
stoke_bin="$builddir/stoke$exe"

golden_lib_require_model "$model" "tok_golden"
golden_lib_require_binary "$stoke_bin" "tok_golden"
golden_lib_require_data_file "$corpus" "tok_golden" "corpus file"
golden_lib_require_data_file "$golden" "tok_golden" "golden file"

golden_lib_run_and_diff "tok_golden" "$golden" \
    "$builddir/tok_corpus.ids.out" "$builddir/tok_corpus.ids.diff" \
    "$stoke_bin tokenize $model $corpus" \
    "$stoke_bin" tokenize "$model" "$corpus" || exit 1

echo "tok_golden: OK ($corpus matches $golden via $model)"
exit 0
