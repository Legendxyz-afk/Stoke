#!/bin/sh
# tests/regen_llama_gen_goldens.sh -- regenerates tests/data/llama_gen/*.golden
# from the REFERENT (llama.cpp), for tests/llama_gen_golden.sh.
#
# ⚠ THE WHOLE POINT IS THAT THE GOLDENS DO NOT COME FROM stoke. Regenerating
# them from stoke's own output would turn a parity gate into a change-detector
# that passes forever however wrong both halves become. If a golden changes
# here, either llama.cpp changed or the prompt did -- and either way the diff
# needs explaining before it is committed.
#
# Like tests/regen_goldens.sh this REGENERATES committed data and is
# deliberately NOT wired into `make test`. Review `git diff tests/data/` first.
#
# Every flag below is load-bearing; see tests/llama_gen_golden.sh's header for
# what each one cost to discover. In short: llama-completion (NOT llama-cli,
# which chat-templates the prompt), --no-conversation (and NOT -st, which
# re-enables it), an explicit -c per case, --log-colors off, and NOT
# --log-disable (it suppresses the completion itself).
#
# Usage: sh tests/regen_llama_gen_goldens.sh
set -eu

cd "$(dirname "$0")/.."

model="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
ref="bench/vendor/llama-b10037/llama-completion.exe"
dir="tests/data/llama_gen"
ngen=64

if [ ! -f "$model" ]; then
    echo "regen_llama_gen_goldens: SKIP -- $model not present (models/ is" \
         "gitignored; this only runs on a box that has the real model)"
    exit 0
fi
if [ ! -f "$ref" ]; then
    echo "regen_llama_gen_goldens: SKIP -- $ref not present (bench/vendor/ is" \
         "gitignored)"
    exit 0
fi

while IFS=' ' read -r case ctx; do
    [ -n "${case:-}" ] || continue
    case "$case" in \#*) continue ;; esac
    prompt="$dir/$case.txt"
    golden="$dir/$case.golden"
    [ -f "$prompt" ] || { echo "missing prompt $prompt" >&2; exit 1; }
    echo "regen_llama_gen_goldens: $case (ctx=$ctx) ..."

    tmp="$golden.tmp"
    rc=0
    "$ref" -m "$model" \
        --override-kv tokenizer.ggml.add_bos_token=bool:false \
        -f "$prompt" -n "$ngen" --temp 0 --seed 1 -c "$ctx" -t 8 \
        --no-warmup --simple-io --no-conversation --no-display-prompt \
        --log-colors off < /dev/null > "$tmp" 2>/dev/null || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "regen_llama_gen_goldens: FAILED -- referent exited $rc on '$case'" >&2
        rm -f "$tmp"; exit 1
    fi

    # Strip CR, ANSI escapes (emitted intermittently even with colours off),
    # trailing whitespace, and trailing BLANK LINES -- the referent ends its
    # output with an empty line and stoke does not, which is console
    # formatting rather than a difference in what either engine generated.
    # LEADING whitespace is preserved: " Paris" and "Paris" are a real
    # difference. tests/llama_gen_golden.sh applies the IDENTICAL pipeline to
    # stoke's side; if these two ever drift apart the gate compares apples to
    # pears and fails for a reason that is nobody's bug.
    tr -d '\r' < "$tmp" \
        | sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' -e 's/[[:space:]]*$//' \
        | sed -e :a -e '/^$/{$d;N;ba' -e '}' > "$tmp.clean"
    rm -f "$tmp"

    if [ ! -s "$tmp.clean" ]; then
        echo "regen_llama_gen_goldens: FAILED -- '$case' produced an EMPTY" \
             "completion; $golden left untouched" >&2
        rm -f "$tmp.clean"; exit 1
    fi
    mv -f "$tmp.clean" "$golden"
done < "$dir/cases.txt"

echo "regen_llama_gen_goldens: OK -- review 'git diff tests/data/llama_gen'" \
     "before committing; a changed golden means llama.cpp or the prompt moved"
