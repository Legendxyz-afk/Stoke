#!/bin/sh
# tests/regen_goldens.sh -- mechanizes golden regeneration: regenerates all
# three tok_*.golden files straight from their corpora via the real oracle
# (bench/vendor/llama-b10037/llama-tokenize.exe over the real Qwen3 model),
# replacing the "run the oracle and strip stdout by hand" manual step that
# used to be described only in prose (tok_fuzz.py / tok_special_fuzz.py
# docstrings).
#
# This script REGENERATES the committed goldens; it does not gate anything
# and is deliberately NOT wired into `make test`. Running it always
# overwrites tests/data/*.golden with fresh oracle output -- review the
# diff (`git diff tests/data/*.golden`) before committing. If the diff is
# non-empty, that means tokenizer *behavior* changed (or the oracle/corpus
# did) -- do not commit a golden change you can't explain.
#
# Model-gated with a loud SKIP + exit 0, exactly like the other gate
# scripts (tests/golden_info.sh, tests/tok_golden.sh,
# tests/tok_special_golden.sh): models/ and bench/vendor/ are gitignored,
# so the model and oracle binary exist on the reference box but not in a
# fresh clone.
#
# Filter (verified to reproduce all three committed goldens byte-exactly):
# strip the oracle's own "\r\n" console line endings, then keep only the
# one line that is the flat "[id, id, ...]"-shaped ids output.
#   <oracle> -m <model> -f <corpus> --ids --no-escape --no-bos [--no-parse-special] \
#       | tr -d '\r' | grep -o '^\[.*\]$'
#
# Usage: sh tests/regen_goldens.sh
set -eu

cd "$(dirname "$0")/.."

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
oracle="bench/vendor/llama-b10037/llama-tokenize.exe"

if [ ! -f "$model" ]; then
    echo "regen_goldens: SKIP -- $model not present (models/ is" \
         "gitignored; this script only runs on boxes that have the real" \
         "model file)"
    exit 0
fi

if [ ! -f "$oracle" ]; then
    echo "regen_goldens: SKIP -- $oracle not present (bench/vendor/ is" \
         "gitignored; this script only runs on boxes that have the" \
         "vendored oracle binary)"
    exit 0
fi

# regen <corpus> <golden> [oracle-flag...]
regen() {
    corpus="$1"
    golden="$2"
    shift 2
    echo "regen_goldens: regenerating $golden ..."
    "$oracle" -m "$model" -f "$corpus" --ids --no-escape --no-bos "$@" \
        | tr -d '\r' | grep -o '^\[.*\]$' > "$golden"
}

# --no-parse-special: tok_encode's mode (plain `stoke tokenize`).
regen tests/data/tok_corpus.txt tests/data/tok_corpus.ids.golden \
    --no-parse-special

# Oracle default (parse-special ON): tok_encode_special's mode
# (`stoke tokenize --special`).
regen tests/data/tok_special_corpus.txt \
    tests/data/tok_special_corpus.ids.golden

# --no-parse-special again, same corpus: tok_encode's corrected OFF-mode
# behavior (`stoke tokenize`, no --special) on the specials+ChatML corpus.
regen tests/data/tok_special_corpus.txt \
    tests/data/tok_special_corpus.plain.golden --no-parse-special

echo "regen_goldens: OK -- regenerated 3 goldens; review with" \
     "'git diff tests/data/*.golden' before committing"
