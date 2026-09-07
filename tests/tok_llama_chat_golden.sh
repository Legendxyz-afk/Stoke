#!/bin/sh
# tests/tok_llama_chat_golden.sh -- the Llama-3 HEADER TEMPLATE gate.
#
# M7 §8.8's FOURTH targeted delta ("header-format template #2"). The other
# three -- QK-norm gating, llama3 rope scaling, the llama-bpe regex -- were
# graded by the 2026-09-07 entry probe; this one is not a forward-pass property
# and a probe grid says nothing about it.
#
# ⚠ THIS GATE IS M1-STYLE BYTE-EXACT AND THAT IS ALLOWED. A chat template is a
# DISCRETE domain -- messages in, token ids out, no floating point anywhere --
# exactly like the tokenizer gate. The binding ruling's "never a 50/50 live
# oracle bar" is about the FORWARD PASS, whose output is continuous. Getting
# that distinction backwards in either direction is how a gate ends up either
# unreachable or vacuous.
#
# ⚠⚠ THE DATE IS PINNED WITH --chat-date AND THAT IS LOAD-BEARING. The template
# emits "Today Date: {strftime_now('%d %b %Y')}", and llama.cpp really does
# render TODAY (it defines strftime_now over the current local time --
# common/jinja/value.cpp). An unpinned golden over this output would pass on the
# day it was made and fail every day after, for a reason nobody can fix. The
# renderer therefore takes the date as a PARAMETER; production passes NULL and
# gets today, this gate passes the date the goldens were built with.
#
# ▶ THE GOLDENS COME FROM THE REFERENT. `*.render.txt` is llama.cpp's own
# rendering of the model's tokenizer.chat_template (transcribed from
# llama-template-analysis, which renders it through minja), and `*.ids.golden`
# is `llama-tokenize --ids` over that render. stoke's renderer was written
# independently from the Jinja source; the gate is that the two agree.
#
# ⚠⚠⚠ THE `inject` CASE IS NOT A PARITY CASE AND MUST NOT BE TURNED INTO ONE.
# When the user's own message CONTAINS the literal text "<|eot_id|>", the naive
# pipeline -- render everything to one string, then tokenize with specials
# matched -- emits the REAL 128009 id from user content and forges a turn
# boundary. `inject.naive-unsafe.ids` is that naive result, kept as a CONTRAST,
# and stoke must NOT equal it. stoke splices the template's own special ids in
# directly and encodes content with CONTROL specials NOT matched, so the same
# text becomes ordinary byte-BPE tokens.
#   This says nothing about llama.cpp's SERVER, which is not what produced the
#   contrast file -- `llama-tokenize -f` is this harness. The claim is exactly:
#   the naive pipeline is injectable and stoke does not use it.
#
# Model-gated with a loud SKIP + exit 0 like every other golden gate.
#
# Usage: sh tests/tok_llama_chat_golden.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: tok_llama_chat_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
dir="tests/data/llama_chat"
stoke_bin="$builddir/stoke$exe"
# The date the goldens were generated with. Changing it means regenerating them.
chat_date="07 Sep 2026"

golden_lib_require_model "$model" "tok_llama_chat_golden"
golden_lib_require_binary "$stoke_bin" "tok_llama_chat_golden"

fail=0
n=0

# --- parity cases: stoke must EQUAL the referent -----------------------------
for case in plain trim; do
    user="$dir/$case.user.txt"
    golden="$dir/$case.ids.golden"
    golden_lib_require_data_file "$user" "tok_llama_chat_golden" "user message"
    golden_lib_require_data_file "$golden" "tok_llama_chat_golden" "golden"

    out="$builddir/tok_llama_chat_$case.out"
    "$stoke_bin" tokenize --chat --chat-date "$chat_date" "$model" "$user" \
        2>/dev/null | tr -d '\r' > "$out"
    if [ ! -s "$out" ]; then
        echo "tok_llama_chat_golden: FAILED -- '$case' produced no output" >&2
        fail=1; continue
    fi
    if diff -u "$golden" "$out" > "$builddir/tok_llama_chat_$case.diff" 2>&1; then
        rm -f "$out" "$builddir/tok_llama_chat_$case.diff"
        n=$((n + 1))
    else
        echo "tok_llama_chat_golden: FAILED -- '$case' differs from $golden" >&2
        cat "$builddir/tok_llama_chat_$case.diff" >&2
        fail=1
    fi
done

# --- the injection case: stoke must DIFFER from the naive pipeline -----------
user="$dir/inject.user.txt"
naive="$dir/inject.naive-unsafe.ids"
golden_lib_require_data_file "$user" "tok_llama_chat_golden" "user message"
golden_lib_require_data_file "$naive" "tok_llama_chat_golden" "naive contrast"

out="$builddir/tok_llama_chat_inject.out"
"$stoke_bin" tokenize --chat --chat-date "$chat_date" "$model" "$user" \
    2>/dev/null | tr -d '\r' > "$out"
if [ ! -s "$out" ]; then
    echo "tok_llama_chat_golden: FAILED -- 'inject' produced no output" >&2
    fail=1
elif diff -q "$naive" "$out" >/dev/null 2>&1; then
    echo "tok_llama_chat_golden: FAILED -- 'inject' MATCHES the naive unsafe" \
         "tokenization: a literal <|eot_id|> in user content became the real" \
         "special id, which forges a turn boundary" >&2
    fail=1
else
    # Differing is necessary and not sufficient: assert the specials the user
    # typed are ABSENT from the content region. The template's OWN 128009s are
    # legitimate, so count them -- one per closed turn (system, user) -- and
    # require exactly that. A stray one means content leaked through.
    got_eot=$(tr ',' '\n' < "$out" | tr -d ' []' | grep -c '^128009$')
    got_hdr=$(tr ',' '\n' < "$out" | tr -d ' []' | grep -c '^128006$')
    if [ "$got_eot" -ne 2 ] || [ "$got_hdr" -ne 3 ]; then
        echo "tok_llama_chat_golden: FAILED -- 'inject' has $got_eot <|eot_id|>" \
             "and $got_hdr <|start_header_id|>; the template alone emits 2 and 3," \
             "so content leaked a special id" >&2
        fail=1
    else
        rm -f "$out"
        n=$((n + 1))
    fi
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi
if [ "$n" -eq 0 ]; then
    echo "tok_llama_chat_golden: FAILED -- no cases ran" >&2
    exit 1
fi
echo "tok_llama_chat_golden: OK ($n cases: 2 referent-parity, 1 injection-safety)"
exit 0
