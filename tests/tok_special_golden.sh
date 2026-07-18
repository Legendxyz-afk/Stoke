#!/bin/sh
# tests/tok_special_golden.sh -- script gate (task-6b-brief.md C7/"Gate
# mechanism", mirrors tests/tok_golden.sh exactly): runs `stoke tokenize`
# BOTH ways (plain, and --special) on the real 2.5GB Qwen3 model over the
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
# Usage: sh tests/tok_special_golden.sh <builddir> [exe-suffix]
set -u

builddir="${1:?usage: tok_special_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
corpus="tests/data/tok_special_corpus.txt"
golden_ids="tests/data/tok_special_corpus.ids.golden"
golden_plain="tests/data/tok_special_corpus.plain.golden"
stoke_bin="$builddir/stoke$exe"

if [ ! -f "$model" ]; then
    echo "tok_special_golden: SKIP -- $model not present (models/ is" \
         "gitignored; this gate only runs on boxes that have the real" \
         "model file)"
    exit 0
fi

if [ ! -f "$stoke_bin" ]; then
    echo "tok_special_golden: missing binary $stoke_bin (build it before running this gate)" >&2
    exit 1
fi

if [ ! -f "$corpus" ]; then
    echo "tok_special_golden: missing corpus file $corpus" >&2
    exit 1
fi

if [ ! -f "$golden_ids" ] || [ ! -f "$golden_plain" ]; then
    echo "tok_special_golden: missing golden file(s) ($golden_ids, $golden_plain)" >&2
    exit 1
fi

fail=0

# Direction 1: plain `tokenize` (tok_encode, USER_DEFINED-only match set)
# vs the oracle's --no-parse-special golden.
got_plain="$builddir/tok_special_corpus.plain.out"
"$stoke_bin" tokenize "$model" "$corpus" > "$got_plain"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "tok_special_golden: FAILED -- '$stoke_bin tokenize $model $corpus' exited $rc" >&2
    fail=1
elif ! diff -u "$golden_plain" "$got_plain" > "$builddir/tok_special_corpus.plain.diff" 2>&1; then
    echo "tok_special_golden: FAILED -- plain tokenize output differs from $golden_plain" >&2
    cat "$builddir/tok_special_corpus.plain.diff" >&2
    fail=1
else
    rm -f "$builddir/tok_special_corpus.plain.diff"
fi
rm -f "$got_plain"

# Direction 2: `tokenize --special` (tok_encode_special, CONTROL union
# USER_DEFINED) vs the oracle's default parse-special-ON golden.
got_ids="$builddir/tok_special_corpus.ids.out"
"$stoke_bin" tokenize --special "$model" "$corpus" > "$got_ids"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "tok_special_golden: FAILED -- '$stoke_bin tokenize --special $model $corpus' exited $rc" >&2
    fail=1
elif ! diff -u "$golden_ids" "$got_ids" > "$builddir/tok_special_corpus.ids.diff" 2>&1; then
    echo "tok_special_golden: FAILED -- --special output differs from $golden_ids" >&2
    cat "$builddir/tok_special_corpus.ids.diff" >&2
    fail=1
else
    rm -f "$builddir/tok_special_corpus.ids.diff"
fi
rm -f "$got_ids"

if [ "$fail" -ne 0 ]; then
    exit 1
fi

echo "tok_special_golden: OK ($corpus matches both goldens via $model)"
exit 0
