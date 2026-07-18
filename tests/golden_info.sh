#!/bin/sh
# tests/golden_info.sh -- script gate: runs `stoke info` on the real
# 2.5GB Qwen3 model and diffs it against the
# committed golden. models/ and *.gguf are gitignored, so the model exists
# on the reference box but not in a fresh clone -- this must SKIP cleanly
# (loud, explicit, exit 0) when the model is absent, and only actually gate
# the build when it's present. tests/test_gguf.c's own unit tests use the
# tiny committed fixtures and always run regardless of this gate.
#
# Usage: sh tests/golden_info.sh <builddir> [exe-suffix]
set -u

builddir="${1:?usage: golden_info.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
golden="tests/data/qwen3-4b-info.golden"
stoke_bin="$builddir/stoke$exe"

if [ ! -f "$model" ]; then
    if [ "${STOKE_REQUIRE_GOLDEN:-0}" = "1" ]; then
        echo "golden_info: FAILED -- $model not present but" \
             "STOKE_REQUIRE_GOLDEN=1 (this box is supposed to have the" \
             "real model; a moved/missing model must not silently" \
             "downgrade this gate to a no-op)" >&2
        exit 1
    fi
    echo "golden_info: SKIP -- $model not present (models/ is gitignored;" \
         "this gate only runs on boxes that have the real model file)"
    exit 0
fi

if [ ! -f "$stoke_bin" ]; then
    echo "golden_info: missing binary $stoke_bin (build it before running this gate)" >&2
    exit 1
fi

if [ ! -f "$golden" ]; then
    echo "golden_info: missing golden file $golden" >&2
    exit 1
fi

got="$builddir/qwen3-4b-info.out"
"$stoke_bin" info "$model" > "$got"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "golden_info: FAILED -- '$stoke_bin info $model' exited $rc" >&2
    exit 1
fi

if ! diff -u "$golden" "$got" > "$builddir/qwen3-4b-info.diff" 2>&1; then
    echo "golden_info: FAILED -- output differs from $golden" >&2
    cat "$builddir/qwen3-4b-info.diff" >&2
    exit 1
fi

rm -f "$got" "$builddir/qwen3-4b-info.diff"
echo "golden_info: OK ($model matches $golden)"
exit 0
