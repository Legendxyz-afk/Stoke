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
# Safety: the oracle runs to a TEMP file first, its own exit status is
# checked explicitly (a pipeline's exit status under `set -e` is only ever
# its LAST command's -- `set -o pipefail` isn't POSIX-portable to msys/WSL
# `sh` either, so this doesn't lean on it at all), and the filtered result
# is checked for the expected shape (exactly one line, starting '[' and
# ending ']') -- all BEFORE the committed golden is touched. Only a
# known-good replacement is ever `mv`'d over a committed golden; a failure
# anywhere in between leaves it byte-unchanged (previously, `> "$golden"`
# truncated the committed file up front, so an oracle failure partway
# through could leave a 0-byte or wrong golden in place while the script
# still reported OK).
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

    raw="$golden.tmp_raw"
    filtered="$golden.tmp_filtered"
    rm -f "$raw" "$filtered"

    # Run the oracle to a temp file and capture ITS OWN exit status
    # explicitly, via the `cmd || rc=$?` idiom (the only portable way to
    # both inspect a failing command's exit code AND stop `set -e` from
    # aborting the script right here, before the check below ever runs).
    rc=0
    "$oracle" -m "$model" -f "$corpus" --ids --no-escape --no-bos "$@" \
        > "$raw" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "regen_goldens: FAILED -- oracle exited $rc regenerating" \
             "$golden (corpus $corpus) -- $golden left untouched" >&2
        rm -f "$raw"
        exit 1
    fi

    # `|| true`: a pipeline whose last command (grep -o) finds zero
    # matches exits 1 -- a real failure mode this script must report with
    # its own message below, not have `set -e` kill the script
    # unceremoniously first.
    tr -d '\r' < "$raw" | grep -o '^\[.*\]$' > "$filtered" || true
    rm -f "$raw"

    # Assert the filtered result is non-empty and has the expected shape
    # (exactly one line, starting '[' and ending ']') BEFORE the committed
    # golden is touched at all.
    n_lines=$(wc -l < "$filtered" | tr -d ' ')
    if [ "$n_lines" -ne 1 ]; then
        echo "regen_goldens: FAILED -- filtered oracle output for $golden" \
             "has $n_lines bracket-shaped line(s), expected exactly 1" \
             "(corpus $corpus) -- see $filtered -- $golden left untouched" >&2
        exit 1
    fi
    line=$(cat "$filtered")
    case $line in
        \[*\]) : ;;
        *)
            echo "regen_goldens: FAILED -- $golden's sole filtered line" \
                 "does not start with '[' and end with ']' -- $golden" \
                 "left untouched" >&2
            exit 1
            ;;
    esac

    # Only NOW, with a known-good replacement sitting in $filtered, replace
    # the committed golden -- a same-directory `mv` is a rename, so this
    # is atomic: $golden is never observably truncated or partially
    # written.
    mv -f "$filtered" "$golden"
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
