#!/bin/sh
# tests/golden_info.sh -- script gate: runs `stoke info` on the real
# 2.5GB Qwen3 model and diffs it against the
# committed golden. models/ and *.gguf are gitignored, so the model exists
# on the reference box but not in a fresh clone -- this must SKIP cleanly
# (loud, explicit, exit 0) when the model is absent, and only actually gate
# the build when it's present. tests/test_gguf.c's own unit tests use the
# tiny committed fixtures and always run regardless of this gate.
#
# Shares its model-check/STOKE_REQUIRE_GOLDEN/binary-check/diff/cleanup
# body with tok_golden.sh, tok_special_golden.sh and
# unicode_sweep_golden.sh via tests/golden_lib.sh -- see that file for the
# shared policy this gate calls into.
#
# Usage: sh tests/golden_info.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: golden_info.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
golden="tests/data/qwen3-4b-info.golden"
stoke_bin="$builddir/stoke$exe"

golden_lib_require_model "$model" "golden_info"
golden_lib_require_binary "$stoke_bin" "golden_info"
golden_lib_require_data_file "$golden" "golden_info" "golden file"

golden_lib_run_and_diff "golden_info" "$golden" \
    "$builddir/qwen3-4b-info.out" "$builddir/qwen3-4b-info.diff" \
    "$stoke_bin info $model" \
    "$stoke_bin" info "$model" || exit 1

echo "golden_info: OK ($model matches $golden)"
exit 0
