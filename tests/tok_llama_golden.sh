#!/bin/sh
# tests/tok_llama_golden.sh -- the SECOND MODEL's byte-exact tokenizer gate.
# Everything the three Qwen gates check, checked again on
# Llama-3.2-3B-Instruct: a different architecture, a different 128k vocab, a
# different merge table, and -- the reason this file exists -- a different
# PRE-TOKENIZER (tokenizer.ggml.pre = "llama-bpe", i.e. `\p{N}{1,3}`,
# add_bos defaulting to true, and ignore_merges).
#
# ⚠ This gate is the only thing that grades two of those three deltas on a
# REAL vocab, and one of them is invisible to the synthetic unit fixture:
# dropping the GPT-2 byte->unicode remap from the ignore_merges lookup
# leaves tests/test_tok.c fully green (its probe token "zqx" is ASCII, where
# the remap is the identity) and is caught HERE, because real fragments
# start with 'Ġ'. See tests/test_tok.c's llama-bpe section.
#
# ⚠ WHAT THIS GATE DELIBERATELY DOES NOT COVER: tests/data/unicode_sweep.txt.
# stoke and the oracle disagree on exactly ONE of its 139,375 ids, and the
# cause is measured, not suspected -- a UNICODE VERSION SKEW, which
# tests/gen_unicode_data.py's docstring predicted in writing:
#
#   U+116E2 MYANMAR EASTERN PWO KAREN DIGIT EIGHT is category Nd in Unicode
#   16.0.0 (which src/unicode_data.h is generated from) and UNDEFINED in the
#   table llama.cpp b10037 ships (unicode-data.cpp: the range at 0x116CA
#   carries flags 0x0001 = UNDEFINED). So for the input " <U+116E2>" the
#   oracle takes alt4 (' '?[^\s\p{L}\p{N}]+) and gets ONE fragment, while
#   stoke sees a number, takes alt7 for the space alone and alt3 for the
#   digit, and gets TWO. Both are internally correct for their own Unicode
#   version. It only becomes VISIBLE in the ids because Llama's merge table
#   happens to carry the rule that spans that boundary ('Ġ' + 'ð', rank
#   251501 of 280,147); Qwen's 151,387 merges do not have it, which is why
#   tests/unicode_sweep_golden.sh has been passing over the identical
#   classification difference all along -- measured on both models, not
#   assumed.
#
# Adding the sweep here would mean either committing a red gate or
# hand-editing src/unicode_data.h to mirror one vendored oracle build's
# Unicode version. That is a policy decision about which Unicode stoke
# targets, it is not B5's to take, and it is written up in
# docs/superpowers/design-notes/2026-09-05-m7-b5-entry.md rather than
# silently absorbed here.
#
# Shares its model-check/STOKE_REQUIRE_GOLDEN/binary-check/diff/cleanup body
# with the other gates via tests/golden_lib.sh, and accumulates a `fail`
# flag across all three legs (like tok_special_golden.sh) so one run reports
# every failing direction instead of stopping at the first.
#
# Usage: sh tests/tok_llama_golden.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: tok_llama_golden.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
corpus="tests/data/tok_corpus.txt"
scorpus="tests/data/tok_special_corpus.txt"
golden="tests/data/tok_corpus.llama.ids.golden"
sgolden_plain="tests/data/tok_special_corpus.llama.plain.golden"
sgolden_ids="tests/data/tok_special_corpus.llama.ids.golden"
stoke_bin="$builddir/stoke$exe"

golden_lib_require_model "$model" "tok_llama_golden"
golden_lib_require_binary "$stoke_bin" "tok_llama_golden"
golden_lib_require_data_file "$corpus" "tok_llama_golden" "corpus file"
golden_lib_require_data_file "$scorpus" "tok_llama_golden" "corpus file"
golden_lib_require_data_file "$golden" "tok_llama_golden" "golden file"
golden_lib_require_data_file "$sgolden_plain" "tok_llama_golden" "golden file"
golden_lib_require_data_file "$sgolden_ids" "tok_llama_golden" "golden file"

fail=0

# Leg 1: the 2,078-line multilingual corpus, plain mode. This is the leg
# that grades \p{N}{1,3} and ignore_merges on a real vocab: before
# ignore_merges landed it diverged at 77 places on one token
# (' مدينة', id 117188), and reverting the digit bound takes that to 145.
golden_lib_run_and_diff "tok_llama_golden" "$golden" \
    "$builddir/tok_corpus.llama.ids.out" \
    "$builddir/tok_corpus.llama.ids.diff" \
    "$stoke_bin tokenize $model $corpus" \
    "$stoke_bin" tokenize "$model" "$corpus" || fail=1

# Leg 2: the specials/ChatML corpus, plain mode (USER_DEFINED-only match
# set). Its special strings are Qwen's, which Llama's vocab does NOT
# contain -- so on this model they are ordinary text and this leg grades the
# partition path's behaviour when NOTHING matches, which is the case the
# Qwen gate can never exercise.
golden_lib_run_and_diff "tok_llama_golden" "$sgolden_plain" \
    "$builddir/tok_special_corpus.llama.plain.out" \
    "$builddir/tok_special_corpus.llama.plain.diff" \
    "$stoke_bin tokenize $model $scorpus" \
    "$stoke_bin" tokenize "$model" "$scorpus" || fail=1

# Leg 3: the same corpus with CONTROL specials matched too. Llama's own
# CONTROL tokens (<|begin_of_text|>, <|eot_id|>, ...) are a different set
# from Qwen's, so the two goldens differ and the mode is doing real work
# here rather than being a copy of leg 2.
golden_lib_run_and_diff "tok_llama_golden" "$sgolden_ids" \
    "$builddir/tok_special_corpus.llama.ids.out" \
    "$builddir/tok_special_corpus.llama.ids.diff" \
    "$stoke_bin tokenize --special $model $scorpus" \
    "$stoke_bin" tokenize --special "$model" "$scorpus" || fail=1

if [ "$fail" -ne 0 ]; then
    exit 1
fi

echo "tok_llama_golden: OK (3 legs match their goldens via $model)"
exit 0
