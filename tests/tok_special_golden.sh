#!/bin/sh
# tests/tok_special_golden.sh -- script gate (mirrors tests/tok_golden.sh
# exactly): runs `stoke tokenize` BOTH ways (plain, and --special) on the
# real 2.5GB Qwen3 model over the
# committed specials+ChatML corpus and diffs each against its own committed
# golden (produced by the real llama-tokenize.exe oracle -- see
# tests/tok_special_fuzz.py's docstring for the exact regeneration
# commands: plain `tokenize` <-> ...plain.golden is the oracle's
# --no-parse-special mode; `tokenize --special` <-> ...ids.golden is the
# oracle's default parse-special-ON mode). models/ and *.gguf are
# gitignored, so the model exists on the reference box but not in a fresh
# clone -- this must SKIP cleanly (loud, explicit, exit 0) when the model
# is absent, and only actually gate the build when it's present.
# tests/test_tok.c's own unit tests use a tiny committed synthetic fixture
# and always run regardless of this gate.
#
# Shares its model-check/STOKE_REQUIRE_GOLDEN/binary-check/diff/cleanup
# body with golden_info.sh, tok_golden.sh and unicode_sweep_golden.sh via
# tests/golden_lib.sh -- see that file for the shared policy this gate
# calls into. This is the one gate with TWO (cmd,golden) pairs to check,
# so unlike its three siblings it accumulates a `fail` flag across both
# legs instead of exiting on the first, so a run where both directions
# happen to fail reports both instead of only the first one found.
#
# Usage: sh tests/tok_special_golden.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: tok_special_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
corpus="tests/data/tok_special_corpus.txt"
golden_ids="tests/data/tok_special_corpus.ids.golden"
golden_plain="tests/data/tok_special_corpus.plain.golden"
stoke_bin="$builddir/stoke$exe"

golden_lib_require_model "$model" "tok_special_golden"
golden_lib_require_binary "$stoke_bin" "tok_special_golden"
golden_lib_require_data_file "$corpus" "tok_special_golden" "corpus file"
golden_lib_require_data_file "$golden_plain" "tok_special_golden" "golden file"
golden_lib_require_data_file "$golden_ids" "tok_special_golden" "golden file"

fail=0

# Direction 1: plain `tokenize` (tok_encode, USER_DEFINED-only match set)
# vs the oracle's --no-parse-special golden.
golden_lib_run_and_diff "tok_special_golden" "$golden_plain" \
    "$builddir/tok_special_corpus.plain.out" \
    "$builddir/tok_special_corpus.plain.diff" \
    "$stoke_bin tokenize $model $corpus" \
    "$stoke_bin" tokenize "$model" "$corpus" || fail=1

# Direction 2: `tokenize --special` (tok_encode_special, CONTROL union
# USER_DEFINED) vs the oracle's default parse-special-ON golden.
golden_lib_run_and_diff "tok_special_golden" "$golden_ids" \
    "$builddir/tok_special_corpus.ids.out" \
    "$builddir/tok_special_corpus.ids.diff" \
    "$stoke_bin tokenize --special $model $corpus" \
    "$stoke_bin" tokenize --special "$model" "$corpus" || fail=1

if [ "$fail" -ne 0 ]; then
    exit 1
fi

echo "tok_special_golden: OK ($corpus matches both goldens via $model)"
exit 0
