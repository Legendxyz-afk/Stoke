/* src/tok.h -- byte-level BPE tokenizer: qwen2 pre-tokenizer regex split ->
 * GPT-2 byte<->unicode remap -> merge-rank BPE, and the exact reverse for
 * decode. This is the encode/decode CORE only (task-6a-brief.md) -- special-
 * token parsing and ChatML templating are a separate later task (6b) and
 * are deliberately NOT declared here.
 *
 * Contract (task-6a-brief.md, "Interfaces" + B1/B6):
 *   - tok_from_gguf() never returns NULL; fatal() on any malformed or
 *     unexpected tokenizer metadata (wrong tokenizer.ggml.model/pre, a byte
 *     with no single-byte vocab entry, a merge half missing from the
 *     vocab, bos/eos ids out of range, ...). `f` is not retained past the
 *     call -- the returned Tok is fully self-contained, so the caller may
 *     gguf_close(f) immediately after this returns.
 *   - tok_encode() must be byte-exact against
 *     `llama-tokenize.exe -m <model> -f <file> --ids --no-escape
 *      --no-parse-special --no-bos` (b10037) run over the same bytes -- the
 *     gate tests/tok_golden.sh checks this on a 2000-line multilingual
 *     corpus (task-6a-brief.md B6).
 *   - Every failure path is fatal(): "tok: <what> -- <why> -- <hint>",
 *     nonzero exit, no silent truncation or fallback.
 */
#ifndef STOKE_TOK_H
#define STOKE_TOK_H

#include "common.h"

#include "gguf.h"

#include <stdbool.h>

typedef struct Tok Tok;

/* Reads tokenizer.ggml.* tables from `f` (task-6a-brief.md B1: model/pre
 * assertions, tokens/merges arrays, bos/eos ids, add_bos_token) and builds
 * a fully self-contained tokenizer. Never returns NULL. */
Tok* tok_from_gguf(const GgufFile* f);

/* Frees everything owned by `t`. `t` may be NULL (no-op). */
void tok_free(Tok* t);

/* Encodes raw UTF-8 bytes `utf8[0..len)`: qwen2 pre-tokenizer split, then
 * per fragment, GPT-2 byte->unicode remap + merge-rank BPE. Writes token
 * ids to `ids` and returns the count written.
 *
 * Cap sizing: BPE only ever MERGES symbols (never splits), so the token
 * count can never exceed the CANONICAL byte count `utf8` decodes to (see
 * src/tok.c's block comment above utf8_decode_one) -- which is at most
 * 3*len (achieved only when every byte is independently invalid UTF-8,
 * each substituting U+FFFD's 3-byte canonical encoding; well-formed input
 * never expands). So cap >= 3*len is always sufficient; cap >= len is
 * enough for any input that's entirely well-formed UTF-8. Fatal if `cap`
 * is exceeded anyway (defensive; every attacker-influenced size in this
 * codebase is bounds-checked, not just documented as the caller's
 * responsibility).
 *
 * add_special: when true, prepends BOS iff the model's
 * tokenizer.ggml.add_bos_token is true (Qwen3: false, so this adds
 * nothing for this model -- the general path is still implemented). No
 * special-token *parsing* of `utf8` happens here (pure byte-BPE only,
 * matching `--no-parse-special`); that is task 6b. */
u32 tok_encode(const Tok* t, const char* utf8, u64 len, u32* ids, u32 cap,
               bool add_special);

/* Decodes ONE token id to raw UTF-8 bytes into `buf` (a token piece may be
 * a partial UTF-8 sequence -- BPE tokens don't have to land on codepoint
 * boundaries) and returns the byte count written. Fatal if `id` is out of
 * vocab range, or if the decoded byte count would exceed `cap`. */
u32 tok_decode(const Tok* t, u32 id, char* buf, u32 cap);

u32 tok_bos(const Tok* t); /* tokenizer.ggml.bos_token_id */
u32 tok_eos(const Tok* t); /* tokenizer.ggml.eos_token_id */

#endif /* STOKE_TOK_H */
