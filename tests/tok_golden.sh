#!/bin/sh
# tests/tok_golden.sh -- script gate (task-6a-brief.md B6, mirrors
# tests/golden_info.sh exactly): runs `stoke tokenize` on the real 2.5GB
# Qwen3 model over the committed fuzz corpus and diffs it against the
# committed golden (produced by the real llama-tokenize.exe oracle --
# see tests/tok_fuzz.py's docstring for the exact regeneration command).
# models/ and *.gguf are gitignored, so the model exists on the reference
# box but not in a fresh clone -- this must SKIP cleanly (loud, explicit,
# exit 0) when the model is absent, and only actually gate the build when
# it's present. tests/test_tok.c's own unit tests use a tiny committed
# synthetic fixture and always run regardless of this gate.
#
# Usage: sh tests/tok_golden.sh <builddir> [exe-suffix]
set -u

builddir="${1:?usage: tok_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
corpus="tests/data/tok_corpus.txt"
golden="tests/data/tok_corpus.ids.golden"
stoke_bin="$builddir/stoke$exe"

if [ ! -f "$model" ]; then
    echo "tok_golden: SKIP -- $model not present (models/ is gitignored;" \
         "this gate only runs on boxes that have the real model file)"
    exit 0
fi

if [ ! -f "$stoke_bin" ]; then
    echo "tok_golden: missing binary $stoke_bin (build it before running this gate)" >&2
    exit 1
fi

if [ ! -f "$corpus" ]; then
    echo "tok_golden: missing corpus file $corpus" >&2
    exit 1
fi

if [ ! -f "$golden" ]; then
    echo "tok_golden: missing golden file $golden" >&2
    exit 1
fi

got="$builddir/tok_corpus.ids.out"
"$stoke_bin" tokenize "$model" "$corpus" > "$got"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "tok_golden: FAILED -- '$stoke_bin tokenize $model $corpus' exited $rc" >&2
    exit 1
fi

if ! diff -u "$golden" "$got" > "$builddir/tok_corpus.ids.diff" 2>&1; then
    echo "tok_golden: FAILED -- output differs from $golden" >&2
    cat "$builddir/tok_corpus.ids.diff" >&2
    exit 1
fi

rm -f "$got" "$builddir/tok_corpus.ids.diff"
echo "tok_golden: OK ($corpus matches $golden via $model)"
exit 0
