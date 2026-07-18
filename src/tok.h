/* src/tok.h -- byte-level BPE tokenizer: qwen2 pre-tokenizer regex split ->
 * GPT-2 byte<->unicode remap -> merge-rank BPE, and the exact reverse for
 * decode (task-6a-brief.md), plus the special-token parse path and the
 * ChatML chat template (task-6b-brief.md), layered on the same pipeline.
 *
 * Contract (task-6a-brief.md "Interfaces" + B1/B6; task-6b-brief.md
 * "Interfaces" + C1-C6):
 *   - tok_from_gguf() never returns NULL; fatal() on any malformed or
 *     unexpected tokenizer metadata (wrong tokenizer.ggml.model/pre, a byte
 *     with no single-byte vocab entry, a merge half missing from the
 *     vocab, bos/eos ids out of range, a missing/wrong-length/out-of-range
 *     tokenizer.ggml.token_type, ...). `f` is not retained past the call --
 *     the returned Tok is fully self-contained, so the caller may
 *     gguf_close(f) immediately after this returns.
 *   - tok_encode() must be byte-exact against
 *     `llama-tokenize.exe -m <model> -f <file> --ids --no-escape
 *      --no-parse-special --no-bos` (b10037) run over the same bytes --
 *     tests/tok_golden.sh checks this on a 2000-line multilingual corpus
 *     (task-6a-brief.md B6). tok_encode_special() must be byte-exact
 *     against the same oracle WITHOUT `--no-parse-special` (parse-special
 *     ON, the oracle's default) -- tests/tok_special_golden.sh checks this,
 *     plus tok_encode()'s corrected OFF-mode behavior, on a dedicated
 *     specials+ChatML corpus (task-6b-brief.md C7/C8).
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

/* Encodes raw UTF-8 bytes `utf8[0..len)`: partitions the input against the
 * USER_DEFINED-type entries of the model's special-token table (exact,
 * leftmost-longest match -- task-6b-brief.md C2-C4), then, independently
 * per gap between/around matches, qwen2 pre-tokenizer split + per-fragment
 * GPT-2 byte->unicode remap + merge-rank BPE (task-6a-brief.md B2/B3/B5).
 * Writes token ids to `ids` and returns the count written.
 *
 * This matches the oracle's `--no-parse-special` mode: CONTROL-type
 * specials (e.g. `<|im_start|>`) are NOT matched and fall through to
 * byte-BPE like ordinary text, but USER_DEFINED-type specials (e.g.
 * `<think>`) ARE always matched regardless of this flag (task-6b-brief.md
 * C2 -- a correction over task-6a's original tok_encode, which matched
 * neither, invisibly: its gate corpus happened to contain zero
 * USER_DEFINED strings). Use tok_encode_special() for the oracle's default
 * parse-special-ON mode, which also matches CONTROL.
 *
 * Cap sizing: BPE only ever MERGES symbols (never splits), and a matched
 * special consumes >=1 input byte to emit exactly 1 id -- never more ids
 * than BPEing those same bytes would have produced -- so the token count
 * can never exceed the CANONICAL byte count `utf8` decodes to (see
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
 * nothing for this model -- the general path is still implemented). */
u32 tok_encode(const Tok* t, const char* utf8, u64 len, u32* ids, u32 cap,
               bool add_special);

/* Same as tok_encode, but ALSO matches CONTROL-type special tokens (e.g.
 * `<|im_start|>`, `<|endoftext|>`) in the input text -- the oracle's
 * default mode (parse-special ON, i.e. `--no-parse-special` OMITTED;
 * task-6b-brief.md C2). Same cap-sizing bound as tok_encode: a matched
 * CONTROL special is likewise 1 emitted id for several consumed bytes,
 * never more ids than pure byte-BPE of the same bytes would emit. */
u32 tok_encode_special(const Tok* t, const char* utf8, u64 len, u32* ids,
                        u32 cap, bool add_special);

/* One chat message. Explicit lengths (not NUL-terminated) to match
 * tok_encode's (ptr,len) discipline: content is untrusted server input at
 * M6. */
typedef struct {
    const char* role;
    u64         role_len; /* "system" | "user" | "assistant" | "tool" */
    const char* content;
    u64         content_len; /* arbitrary UTF-8, may contain anything */
} StokeMsg;

/* Renders the ChatML prompt for `msgs[0..n)` and encodes it; returns id
 * count. Always appends the generation prompt `<|im_start|>assistant\n`
 * (this is an inference engine; a non-generating variant is deferred until
 * a caller needs one). Llama-3 header format is M7. No BOS is ever added
 * (this model's tokenizer.ggml.add_bos_token is false; unlike tok_encode/
 * tok_encode_special, this function has no add_special parameter at all --
 * task-6b-brief.md C5).
 *
 * Injection safety (task-6b-brief.md C5): this does NOT render the whole
 * conversation to one string and parse-special-encode it -- that would let
 * a user whose message CONTENT contains the literal text `<|im_end|>`
 * forge a turn boundary. Instead, each message's `{role}\n{content}` is
 * encoded through tok_encode's match set (USER_DEFINED matched, CONTROL
 * NOT matched), and only the template's own `<|im_start|>`/`<|im_end|>`
 * ids -- looked up in the model's special-token table, never hardcoded --
 * are spliced in around it. For benign content (none of the 6 USER_DEFINED
 * strings present) this is byte-identical to render-then-parse-special-
 * encode, since specials are hard boundaries there too; it diverges only
 * on adversarial content, which is the point. Fatal if the vocab has no
 * `<|im_start|>` or `<|im_end|>` CONTROL/USER_DEFINED special token.
 *
 * Cap sizing: safe if cap >= 8 + 3*(role_len+content_len) summed over
 * every message, + 31 for the fixed generation-prompt tail (same
 * 3x-canonical-bytes-worst-case reasoning as tok_encode, plus a fixed +2
 * per message for its <|im_start|>/<|im_end|> ids and +1 tail id). Fatal
 * if `cap` is exceeded anyway, in the same style as tok_encode. */
u32 tok_apply_chatml(const Tok* t, const StokeMsg* msgs, u32 n, u32* ids,
                      u32 cap);

/* Decodes ONE token id to raw UTF-8 bytes into `buf` (a token piece may be
 * a partial UTF-8 sequence -- BPE tokens don't have to land on codepoint
 * boundaries) and returns the byte count written. Fatal if `id` is out of
 * vocab range, or if the decoded byte count would exceed `cap`.
 *
 * Policy (task-6b-brief.md C6): tok_decode is the mechanical inverse of
 * the vocab table -- id in, that id's exact stored bytes out -- for EVERY
 * id, special or not; it always yields literal text (e.g. id 151644 ->
 * the 12 bytes "<|im_start|>"). It does not know or care whether an id is
 * a CONTROL/USER_DEFINED special. Deciding whether a control token should
 * be shown to an API client as-is, suppressed, or treated as a stop token
 * is a policy choice belonging to the server layer (M6), not here -- so
 * there is deliberately no `special`/`skip_special` flag on this function;
 * add one only when a caller actually needs the distinction. */
u32 tok_decode(const Tok* t, u32 id, char* buf, u32 cap);

u32 tok_bos(const Tok* t); /* tokenizer.ggml.bos_token_id */
u32 tok_eos(const Tok* t); /* tokenizer.ggml.eos_token_id */

#endif /* STOKE_TOK_H */
