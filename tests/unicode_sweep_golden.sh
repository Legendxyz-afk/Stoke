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
# Usage: sh tests/unicode_sweep_golden.sh <builddir> [exe-suffix]
set -u

builddir="${1:?usage: unicode_sweep_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
corpus="tests/data/unicode_sweep.txt"
golden="tests/data/unicode_sweep.ids.golden"
stoke_bin="$builddir/stoke$exe"

if [ ! -f "$model" ]; then
    if [ "${STOKE_REQUIRE_GOLDEN:-0}" = "1" ]; then
        echo "unicode_sweep_golden: FAILED -- $model not present but" \
             "STOKE_REQUIRE_GOLDEN=1 (this box is supposed to have the" \
             "real model; a moved/missing model must not silently" \
             "downgrade this gate to a no-op)" >&2
        exit 1
    fi
    echo "unicode_sweep_golden: SKIP -- $model not present (models/ is" \
         "gitignored; this gate only runs on boxes that have the real" \
         "model file)"
    exit 0
fi

if [ ! -f "$stoke_bin" ]; then
    echo "unicode_sweep_golden: missing binary $stoke_bin (build it before running this gate)" >&2
    exit 1
fi

if [ ! -f "$corpus" ]; then
    echo "unicode_sweep_golden: missing corpus file $corpus" >&2
    exit 1
fi

if [ ! -f "$golden" ]; then
    echo "unicode_sweep_golden: missing golden file $golden" >&2
    exit 1
fi

got="$builddir/unicode_sweep.ids.out"
"$stoke_bin" tokenize "$model" "$corpus" > "$got"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "unicode_sweep_golden: FAILED -- '$stoke_bin tokenize $model $corpus' exited $rc" >&2
    exit 1
fi

if ! diff -u "$golden" "$got" > "$builddir/unicode_sweep.ids.diff" 2>&1; then
    echo "unicode_sweep_golden: FAILED -- output differs from $golden" >&2
    cat "$builddir/unicode_sweep.ids.diff" >&2
    exit 1
fi

rm -f "$got" "$builddir/unicode_sweep.ids.diff"
echo "unicode_sweep_golden: OK ($corpus matches $golden via $model)"
exit 0
