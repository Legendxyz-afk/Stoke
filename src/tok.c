/* src/tok.c -- byte-level BPE tokenizer. See src/tok.h for the public
 * contract and .superpowers/sdd/task-6a-brief.md (B1-B6) for the exact
 * algorithm this implements -- every constant/regex/algorithm step here is
 * taken verbatim from that brief, which was probed from the real Qwen3
 * tokenizer + llama-tokenize.exe (b10037) at dispatch time.
 *
 * Pipeline (task-6a-brief.md B2/B3/B5), per tok_encode() call:
 *   1. UTF-8-decode the whole input into codepoints (src/tok.c
 *      utf8_decode_one) -- invalid/overlong/lone-surrogate bytes fall back
 *      to one pseudo-codepoint per raw byte (B4), so this step never fails.
 *   2. Split those codepoints into fragments via the qwen2 pre-tokenizer
 *      regex (B3, 7 alternatives, hand-implemented as match_alt1..7 below
 *      since C has no \p{L}/\p{N} Unicode regex engine) -- classification
 *      uses the generated src/unicode_data.h table.
 *   3. Each fragment's RAW BYTES (not its classification codepoints --
 *      see the block comment above bpe_encode_fragment for why these are
 *      different things) are remapped byte-by-byte through the GPT-2
 *      byte<->unicode table (B2, built at tok_from_gguf() time) into
 *      initial BPE symbols, then repeatedly merged by lowest merge rank
 *      (B5) until no adjacent pair has a rule. Final symbols are vocab
 *      ids.
 * tok_decode() reverses step 3's remap per character of the vocab string.
 *
 * Two hash tables, both built once in tok_from_gguf():
 *   - a TEMPORARY vocab string->id map (open addressing, FNV-1a), used
 *     only to resolve merge-rule halves to ids at init time, then freed;
 *   - a PERMANENT (id_left,id_right)->{rank,merged_id} map (open
 *     addressing, integer hash), used by every tok_encode() call's merge
 *     loop. Resolving merges to id-pairs once at init means the runtime
 *     hot path never hashes a string -- see the block comment above
 *     merge_hash_init for the invariant this relies on.
 */
#include "common.h"

#include "tok.h"

#include "unicode_data.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* ===========================================================================
 * GPT-2 byte<->unicode map (task-6a-brief.md B2)
 * ===========================================================================
 */

/* Max value bytes_to_unicode() can ever assign: 256 "nice" bytes map to
 * themselves (identity, max 255); the remaining 256-188=68 "not nice" bytes
 * map to 256+n for n in 0..67, so max cpt is 256+67=323. Sized generously
 * (512) rather than tightly at 324 -- the margin costs nothing and removes
 * any temptation to re-derive the exact count under time pressure. */
#define TOK_CPT_TABLE_SIZE 512

/* Builds both directions of the GPT-2 "byte-level" remap (task-6a-brief.md
 * B2), generated at init rather than hardcoded, per the brief's explicit
 * requirement:
 *     bs = ['!'..'~'] + ['\xa1'..'\xac'] + ['\xae'..'\xff']   (as byte values)
 *     cs = bs[:]; n = 0
 *     for b in range(256):
 *         if b not in bs: bs.append(b); cs.append(256+n); n += 1
 * byte b -> unicode codepoint cs[i] where bs[i]==b. The Python loop walks
 * b=0..255 IN ORDER and only appends a byte the first time it's seen not-
 * yet-nice, so `n` increments in strictly increasing byte-value order --
 * the two-pass version below (pass 1: mark identity for the three "nice"
 * ranges; pass 2: walk b=0..255 assigning 256+n to whatever wasn't marked)
 * reproduces that exact assignment order. */
static void build_byte_unicode_maps(u32 byte_to_cpt[256],
                                     i16 cpt_to_byte[TOK_CPT_TABLE_SIZE]) {
    bool nice[256];
    for (u32 i = 0; i < 256; i++) nice[i] = false;
    for (u32 b = (u32)'!'; b <= (u32)'~'; b++) nice[b] = true;   /* 0x21..0x7E */
    for (u32 b = 0xA1; b <= 0xAC; b++) nice[b] = true;           /* 0xA1..0xAC */
    for (u32 b = 0xAE; b <= 0xFF; b++) nice[b] = true;           /* 0xAE..0xFF */

    for (u32 b = 0; b < 256; b++) {
        if (nice[b]) byte_to_cpt[b] = b;
    }
    u32 n = 0;
    for (u32 b = 0; b < 256; b++) {
        if (!nice[b]) {
            byte_to_cpt[b] = 256 + n;
            n++;
        }
    }

    for (u32 i = 0; i < TOK_CPT_TABLE_SIZE; i++) cpt_to_byte[i] = -1;
    for (u32 b = 0; b < 256; b++) {
        u32 cpt = byte_to_cpt[b];
        cpt_to_byte[cpt] = (i16)b; /* cpt < TOK_CPT_TABLE_SIZE always: max
                                       323 < 512, see the #define above */
    }
}

/* UTF-8-encodes `cp` (< 0x110000) into `out` (>=4 bytes); returns the byte
 * length (1..4). Only ever called here with byte-remap codepoints (0..323,
 * always 1 or 2 bytes in practice), but implemented generally correct. */
static u32 utf8_encode_cpt(u32 cp, u8* out) {
    if (cp < 0x80) {
        out[0] = (u8)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (u8)(0xC0 | (cp >> 6));
        out[1] = (u8)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (u8)(0xE0 | (cp >> 12));
        out[1] = (u8)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (u8)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        out[0] = (u8)(0xF0 | (cp >> 18));
        out[1] = (u8)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (u8)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (u8)(0x80 | (cp & 0x3F));
        return 4;
    }
}

/* ===========================================================================
 * UTF-8 decode -- STRUCTURAL validation only, U+FFFD substitution on
 * failure. This deliberately does NOT match task-6a-brief.md B4's literal
 * description ("decodes each undecodable byte as its own codepoint
 * 0x00..0xFF") -- seven rounds of isolated oracle probes at dispatch (see
 * task-6a-report.md self-review for the full transcript) established that
 * llama-tokenize.exe b10037 actually:
 *   1. Accepts ANY structurally well-formed sequence (right lead-byte
 *      pattern + the exact right count of 0x80-0xBF continuation bytes)
 *      and computes its codepoint via the plain bit-packing formula --
 *      WITHOUT rejecting overlong encodings or the surrogate range
 *      (0xD800-0xDFFF). Smoking-gun proof: C0 81, E0 80 81, and a lone raw
 *      byte 0x01 all tokenize IDENTICALLY (id 189) -- the two "overlong"
 *      encodings of U+0001 are accepted and normalized down to the same
 *      single byte, not rejected.
 *   2. On a genuine STRUCTURAL failure (bad lead byte, missing/invalid
 *      continuation byte, or truncation), substitutes codepoint U+FFFD
 *      (NOT the failing byte's own value) and consumes exactly the ONE
 *      lead byte, retrying at the very next byte -- proof: a lone byte
 *      0x80 tokenizes the same (id 5691) as any other single invalid
 *      byte, and its piece decodes to literal bytes EF BF BD (U+FFFD's
 *      canonical UTF-8), not back to 0x80.
 * Point 1 means every decoded codepoint (valid-structural OR the U+FFFD
 * substitute) then needs to be re-encoded to its CANONICAL minimal UTF-8
 * form before byte-remapping for BPE -- see decode_utf8_all() below, which
 * is where that re-encoding happens. This also means encode is NOT
 * lossless for overlong/invalid input (unlike B4's claim) -- confirmed
 * against the real oracle, so tests/tok_golden.sh's decode(encode(x))==x
 * roundtrip is scoped to well-formed corpus lines only (see
 * tests/test_tok.c and tok_golden.sh for where this is called out).
 *
 * 0xF5-0xFF are still rejected outright (fall back, 1 byte): a full
 * structural 0xF5-0xF7 sequence (codepoint > 0x10FFFF) makes the oracle
 * itself crash (exit 127) -- there is no real behavior to match, so this
 * engine stays defensive rather than feeding an out-of-range codepoint
 * into its own classification table.
 */
static u32 utf8_decode_one(const u8* buf, u64 len, u64 pos, u32* out_len) {
    u8 b0 = buf[pos];

    if (b0 < 0x80) {
        *out_len = 1;
        return b0;
    }
    if (b0 < 0xC0) {
        /* 0x80..0xBF: a continuation byte can't start a sequence. */
        *out_len = 1;
        return 0xFFFD;
    }
    if (b0 < 0xE0) {
        /* 2-byte structural pattern: 0xC0..0xDF (including 0xC0/0xC1,
           which can only ever produce an overlong value -- accepted per
           the block comment above). */
        if (pos + 1 >= len || (buf[pos + 1] & 0xC0) != 0x80) {
            *out_len = 1;
            return 0xFFFD;
        }
        u32 cp = ((u32)(b0 & 0x1F) << 6) | (u32)(buf[pos + 1] & 0x3F);
        *out_len = 2;
        return cp;
    }
    if (b0 < 0xF0) {
        /* 3-byte: 0xE0..0xEF, no overlong/surrogate rejection (ditto). */
        if (pos + 2 >= len || (buf[pos + 1] & 0xC0) != 0x80 ||
            (buf[pos + 2] & 0xC0) != 0x80) {
            *out_len = 1;
            return 0xFFFD;
        }
        u32 cp = ((u32)(b0 & 0x0F) << 12) | ((u32)(buf[pos + 1] & 0x3F) << 6) |
                 (u32)(buf[pos + 2] & 0x3F);
        *out_len = 3;
        return cp;
    }
    if (b0 < 0xF5) {
        /* 4-byte: 0xF0..0xF4. The >0x10FFFF guard below is defensive
           (unreachable for F0..F3; for F4 it can trip, see block comment --
           the oracle has no defined behavior there either). */
        if (pos + 3 >= len || (buf[pos + 1] & 0xC0) != 0x80 ||
            (buf[pos + 2] & 0xC0) != 0x80 || (buf[pos + 3] & 0xC0) != 0x80) {
            *out_len = 1;
            return 0xFFFD;
        }
        u32 cp = ((u32)(b0 & 0x07) << 18) | ((u32)(buf[pos + 1] & 0x3F) << 12) |
                 ((u32)(buf[pos + 2] & 0x3F) << 6) | (u32)(buf[pos + 3] & 0x3F);
        if (cp > 0x10FFFF) {
            *out_len = 1;
            return 0xFFFD;
        }
        *out_len = 4;
        return cp;
    }
    /* b0 >= 0xF5: no structural pattern this engine accepts. */
    *out_len = 1;
    return 0xFFFD;
}

/* ===========================================================================
 * qwen2 pre-tokenizer regex (task-6a-brief.md B3), hand-implemented as 7
 * alternative matchers tried in order at each position -- first match
 * wins, matching standard ordered-alternation regex semantics. Every
 * codepoint is exactly one of {letter, number, whitespace, other}
 * (mutually exclusive by construction of the generated table), and every
 * one of those buckets is guaranteed to match SOME alternative (letter ->
 * alt2 at worst; number -> alt3; whitespace -> alt5/6/7, alt7 always
 * succeeds; other -> alt4 at worst, matching itself with 0 lead). So
 * match_fragment() below never fails to consume at least 1 codepoint --
 * there is no character class this regex leaves unhandled.
 * ===========================================================================
 */

typedef struct {
    u32 cp;
    u64 canon_off; /* offset into the CANONICAL re-encoded buffer (see
                       decode_utf8_all) -- NOT the original utf8 buffer;
                       these differ whenever cp came from an overlong or
                       U+FFFD-substituted sequence (block comment above
                       utf8_decode_one) */
    u32 canon_len; /* 1..4: bytes `cp` occupies in the canonical buffer */
    u8  flags;     /* UNICODE_FLAG_* */
} DecodedCp;

static bool cp_is_letter(const DecodedCp* c) {
    return (c->flags & UNICODE_FLAG_LETTER) != 0;
}
static bool cp_is_number(const DecodedCp* c) {
    return (c->flags & UNICODE_FLAG_NUMBER) != 0;
}
static bool cp_is_ws(const DecodedCp* c) {
    return (c->flags & UNICODE_FLAG_WHITESPACE) != 0;
}
static bool cp_is_other(const DecodedCp* c) {
    return !cp_is_letter(c) && !cp_is_number(c) && !cp_is_ws(c);
}
static bool cp_is_crlf(const DecodedCp* c) {
    return c->cp == 0x0D || c->cp == 0x0A;
}

/* alt1: (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]) */
static u64 match_alt1(const DecodedCp* cps, u64 n, u64 i) {
    if (cps[i].cp != (u32)'\'') return 0;
    if (i + 1 >= n) return 0;
    u32 c1 = cps[i + 1].cp;
    if (c1 == 's' || c1 == 'S' || c1 == 't' || c1 == 'T' || c1 == 'm' ||
        c1 == 'M' || c1 == 'd' || c1 == 'D') {
        return 2;
    }
    if (i + 2 < n) {
        u32 c2 = cps[i + 2].cp;
        if ((c1 == 'r' || c1 == 'R') && (c2 == 'e' || c2 == 'E')) return 3;
        if ((c1 == 'v' || c1 == 'V') && (c2 == 'e' || c2 == 'E')) return 3;
        if ((c1 == 'l' || c1 == 'L') && (c2 == 'l' || c2 == 'L')) return 3;
    }
    return 0;
}

/* alt2: [^\r\n\p{L}\p{N}]?\p{L}+ -- an optional lead char (anything except
 * CR/LF/letter/number) followed by one-or-more letters. The lead is only
 * taken if a letter actually follows it (standard backtracking: the
 * optional group can't help if \p{L}+ would then match zero letters). */
static u64 match_alt2(const DecodedCp* cps, u64 n, u64 i) {
    u64 j = i;
    if (!cp_is_crlf(&cps[i]) && !cp_is_letter(&cps[i]) && !cp_is_number(&cps[i])) {
        if (i + 1 < n && cp_is_letter(&cps[i + 1])) {
            j = i + 1;
        }
    }
    u64 k = j;
    while (k < n && cp_is_letter(&cps[k])) k++;
    u64 letters = k - j;
    if (letters == 0) return 0;
    return (j - i) + letters;
}

/* alt3: \p{N} -- exactly one number character (digits emit one id each). */
static u64 match_alt3(const DecodedCp* cps, u64 n, u64 i) {
    (void)n;
    return cp_is_number(&cps[i]) ? 1 : 0;
}

/* alt4: ' '?[^\s\p{L}\p{N}]+[\r\n]* -- an optional literal space (not any
 * \s -- specifically U+0020), one-or-more "other" chars (not whitespace,
 * letter, or number), then trailing CR/LF. */
static u64 match_alt4(const DecodedCp* cps, u64 n, u64 i) {
    u64 j = i;
    if (cps[i].cp == 0x20) {
        if (i + 1 < n && cp_is_other(&cps[i + 1])) {
            j = i + 1;
        }
    }
    u64 k = j;
    while (k < n && cp_is_other(&cps[k])) k++;
    u64 other_count = k - j;
    if (other_count == 0) return 0;
    u64 trail = 0;
    while (k + trail < n && cp_is_crlf(&cps[k + trail])) trail++;
    return (j - i) + other_count + trail;
}

/* alt5: \s*[\r\n]+ -- \s* is greedy but must backtrack to leave >=1 CR/LF
 * for the mandatory tail. Since \s* first tries the WHOLE maximal
 * whitespace run, and everything after the run's rightmost CR/LF is (by
 * definition of "rightmost") not CR/LF, the tail can only ever end up
 * matching exactly that one position. So: match length = (rightmost
 * CR/LF's index within the run) - i + 1; no match if the run has no CR/LF
 * at all. */
static u64 match_alt5(const DecodedCp* cps, u64 n, u64 i) {
    if (!cp_is_ws(&cps[i])) return 0;
    u64 j = i;
    i64 last_break = -1;
    while (j < n && cp_is_ws(&cps[j])) {
        if (cp_is_crlf(&cps[j])) last_break = (i64)j;
        j++;
    }
    if (last_break < 0) return 0;
    return (u64)last_break - i + 1;
}

/* alt6: \s+(?!\S) -- the maximal whitespace run, minus its last char,
 * UNLESS the run reaches end-of-input (nothing there to be \S, so the
 * lookahead trivially holds and the whole run matches). Fails (0) when the
 * run is exactly 1 char and more input follows (can't backtrack \s+ below
 * 1 occurrence). */
static u64 match_alt6(const DecodedCp* cps, u64 n, u64 i) {
    if (!cp_is_ws(&cps[i])) return 0;
    u64 j = i;
    while (j < n && cp_is_ws(&cps[j])) j++;
    u64 run = j - i;
    if (j == n) return run;
    return (run >= 2) ? (run - 1) : 0;
}

/* alt7: \s+ -- any remaining whitespace run, unconditional. */
static u64 match_alt7(const DecodedCp* cps, u64 n, u64 i) {
    if (!cp_is_ws(&cps[i])) return 0;
    u64 j = i;
    while (j < n && cp_is_ws(&cps[j])) j++;
    return j - i;
}

/* Returns how many codepoints starting at cps[i] the first matching
 * alternative consumes (always > 0 for i < n -- see the block comment
 * above this section). */
static u64 match_fragment(const DecodedCp* cps, u64 n, u64 i) {
    u64 m;
    if ((m = match_alt1(cps, n, i)) > 0) return m;
    if ((m = match_alt2(cps, n, i)) > 0) return m;
    if ((m = match_alt3(cps, n, i)) > 0) return m;
    if ((m = match_alt4(cps, n, i)) > 0) return m;
    if ((m = match_alt5(cps, n, i)) > 0) return m;
    if ((m = match_alt6(cps, n, i)) > 0) return m;
    if ((m = match_alt7(cps, n, i)) > 0) return m;
    fatal("tok: internal -- qwen2 pre-tokenizer matched nothing at "
          "codepoint index %llu (cp=U+%04X) -- every codepoint must be "
          "letter/number/whitespace/other and alt4/alt7 always match "
          "the last two; this should be unreachable",
          (unsigned long long)i, cps[i].cp);
    return 0; /* unreachable */
}

/* Decodes utf8[0..len) into `out` (caller-sized >= len entries -- codepoint
 * count can never exceed byte count) AND, in the same pass, writes each
 * codepoint's CANONICAL (minimal, non-overlong) UTF-8 re-encoding into
 * `canon` (caller-sized >= len*3 bytes -- see the block comment above
 * utf8_decode_one: the worst-case expansion is 1 original byte -> one
 * U+FFFD -> 3 canonical bytes, achieved when every input byte is
 * independently invalid). This canonical buffer, not the original `buf`,
 * is what fragments are sliced from for BPE (tok_encode) -- necessary
 * because an overlong or U+FFFD-substituted codepoint's canonical bytes
 * differ from its original raw bytes, and that's what the oracle actually
 * byte-remaps. Returns the codepoint count; *canon_len_out receives the
 * total bytes written to `canon`. */
static u64 decode_utf8_all(const u8* buf, u64 len, DecodedCp* out, u8* canon,
                            u64* canon_len_out) {
    u64 n = 0;
    u64 pos = 0;
    u64 cw = 0;
    while (pos < len) {
        u32 clen;
        u32 cp = utf8_decode_one(buf, len, pos, &clen);
        u8  enc[4];
        u32 enc_len = utf8_encode_cpt(cp, enc);
        memcpy(canon + cw, enc, enc_len);
        out[n].cp = cp;
        out[n].canon_off = cw;
        out[n].canon_len = enc_len;
        out[n].flags = unicode_cpt_flags(cp);
        n++;
        cw += enc_len;
        pos += clen;
    }
    *canon_len_out = cw;
    return n;
}

/* ===========================================================================
 * Hash tables
 * ===========================================================================
 */

static u64 fnv1a(const char* s, u64 len) {
    u64 h = 14695981039346656037ULL;
    for (u64 i = 0; i < len; i++) {
        h ^= (u8)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* splitmix64 finalizer -- a standard, well-mixed integer hash. */
static u64 hash_u64(u64 x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static u32 next_pow2_at_least(u32 want) {
    u32 cap = 16;
    while (cap < want) cap <<= 1;
    return cap;
}

/* ---- temporary string(bytes,len) -> vocab id map (init-time only) ----
 * Keys point into Tok->vocab_str[], which the map does not own -- callers
 * must free the map (strmap_free) before freeing anything the keys point
 * into stops being safe, but since this map is scoped entirely inside
 * tok_from_gguf() and never outlives it, that's automatic. */

typedef struct {
    const char* key;
    u32         key_len;
    u32         id;
    bool        used;
} StrSlot;

typedef struct {
    StrSlot* slots;
    u32      cap;
    u32      mask;
} StrMap;

static void strmap_init(StrMap* m, u32 want_entries) {
    u32 cap = next_pow2_at_least(want_entries < (UINT32_MAX / 2)
                                      ? want_entries * 2
                                      : UINT32_MAX);
    m->slots = (StrSlot*)xcalloc(cap, sizeof(StrSlot));
    m->cap = cap;
    m->mask = cap - 1;
}

/* Duplicate keys keep the FIRST id inserted (silently) -- a well-formed
 * BPE vocab shouldn't have duplicate token strings, but tolerating one
 * rather than fataling keeps this map's only job (string->id resolution
 * for merge-rule halves) robust to that theoretical edge case. */
static void strmap_insert(StrMap* m, const char* key, u32 key_len, u32 id) {
    u64 h = fnv1a(key, key_len);
    u32 i = (u32)h & m->mask;
    for (;;) {
        if (!m->slots[i].used) {
            m->slots[i].used = true;
            m->slots[i].key = key;
            m->slots[i].key_len = key_len;
            m->slots[i].id = id;
            return;
        }
        if (m->slots[i].key_len == key_len &&
            memcmp(m->slots[i].key, key, key_len) == 0) {
            return; /* duplicate key: keep first */
        }
        i = (i + 1) & m->mask;
    }
}

static bool strmap_find(const StrMap* m, const char* key, u32 key_len,
                         u32* out_id) {
    u64 h = fnv1a(key, key_len);
    u32 i = (u32)h & m->mask;
    for (u32 probes = 0; probes <= m->mask; probes++) {
        if (!m->slots[i].used) return false;
        if (m->slots[i].key_len == key_len &&
            memcmp(m->slots[i].key, key, key_len) == 0) {
            *out_id = m->slots[i].id;
            return true;
        }
        i = (i + 1) & m->mask;
    }
    return false;
}

static void strmap_free(StrMap* m) { free(m->slots); }

/* ---- permanent (id_left,id_right) -> {rank,merged_id} map (runtime) ----
 * Invariant this relies on (standard BPE construction property): every
 * symbol that can ever exist mid-merge is either one of the 256 base-byte
 * vocab tokens or the result of some earlier merge rule -- so a merge
 * rule's two halves, and their concatenation, are ALWAYS present in the
 * vocab by the time tok_from_gguf() processes tokenizer.ggml.merges. This
 * lets init time resolve (left_string,right_string)->merged_id once via
 * the vocab string map, so the runtime merge loop never touches a string,
 * only these 3 integers. */

typedef struct {
    u64  key; /* (id_left << 32) | id_right */
    u32  rank;
    u32  merged_id;
    bool used;
} MergeSlot;

typedef struct {
    MergeSlot* slots;
    u32        cap;
    u32        mask;
} MergeMap;

static void mergemap_init(MergeMap* m, u32 want_entries) {
    u32 cap = next_pow2_at_least(want_entries < (UINT32_MAX / 2)
                                      ? want_entries * 2
                                      : UINT32_MAX);
    m->slots = (MergeSlot*)xcalloc(cap, sizeof(MergeSlot));
    m->cap = cap;
    m->mask = cap - 1;
}

static void mergemap_insert(MergeMap* m, u32 id_left, u32 id_right, u32 rank,
                             u32 merged_id) {
    u64 key = ((u64)id_left << 32) | (u64)id_right;
    u64 h = hash_u64(key);
    u32 i = (u32)h & m->mask;
    for (;;) {
        if (!m->slots[i].used) {
            m->slots[i].used = true;
            m->slots[i].key = key;
            m->slots[i].rank = rank;
            m->slots[i].merged_id = merged_id;
            return;
        }
        if (m->slots[i].key == key) {
            fatal("tok: tokenizer.ggml.merges -- duplicate rule for pair "
                  "(%u,%u) at ranks %u and %u",
                  id_left, id_right, m->slots[i].rank, rank);
        }
        i = (i + 1) & m->mask;
    }
}

static bool mergemap_find(const MergeMap* m, u32 id_left, u32 id_right,
                           u32* out_rank, u32* out_merged) {
    u64 key = ((u64)id_left << 32) | (u64)id_right;
    u64 h = hash_u64(key);
    u32 i = (u32)h & m->mask;
    for (u32 probes = 0; probes <= m->mask; probes++) {
        if (!m->slots[i].used) return false;
        if (m->slots[i].key == key) {
            *out_rank = m->slots[i].rank;
            *out_merged = m->slots[i].merged_id;
            return true;
        }
        i = (i + 1) & m->mask;
    }
    return false;
}

/* ===========================================================================
 * Tok
 * ===========================================================================
 */

struct Tok {
    u32    vocab_size;
    char** vocab_str; /* [vocab_size], owned, NUL-terminated */
    u32*   vocab_len; /* [vocab_size] */

    u32 byte_to_cpt[256];
    i16 cpt_to_byte[TOK_CPT_TABLE_SIZE];
    u32 byte_to_id[256]; /* vocab id of each raw byte's single-byte token */

    MergeMap merges;

    u32  bos_id;
    u32  eos_id;
    bool add_bos_token;
};

Tok* tok_from_gguf(const GgufFile* f) {
    const char* model = gguf_meta_str(f, "tokenizer.ggml.model");
    if (strcmp(model, "gpt2") != 0) {
        fatal("tok: tokenizer.ggml.model is '%s' -- expected 'gpt2' -- "
              "only the gpt2/qwen2 byte-BPE tokenizer is implemented "
              "(task-6a-brief.md B1)",
              model);
    }
    const char* pre = gguf_meta_str(f, "tokenizer.ggml.pre");
    if (strcmp(pre, "qwen2") != 0) {
        fatal("tok: tokenizer.ggml.pre is '%s' -- expected 'qwen2' -- only "
              "the qwen2 pre-tokenizer regex is implemented "
              "(task-6a-brief.md B1)",
              pre);
    }

    u64 vocab_size64 = gguf_meta_arr_len(f, "tokenizer.ggml.tokens");
    if (vocab_size64 == 0 || vocab_size64 > UINT32_MAX) {
        fatal("tok: tokenizer.ggml.tokens has implausible length %llu",
              (unsigned long long)vocab_size64);
    }
    u32 vocab_size = (u32)vocab_size64;

    Tok* t = (Tok*)xcalloc(1, sizeof(Tok));
    t->vocab_size = vocab_size;
    t->vocab_str = (char**)xcalloc(vocab_size, sizeof(char*));
    t->vocab_len = (u32*)xcalloc(vocab_size, sizeof(u32));

    StrMap smap;
    strmap_init(&smap, vocab_size);

    for (u32 id = 0; id < vocab_size; id++) {
        const char* s = gguf_meta_arr_str(f, "tokenizer.ggml.tokens", id);
        size_t      slen = strlen(s);
        char*       owned = (char*)xmalloc(slen + 1);
        memcpy(owned, s, slen + 1);
        t->vocab_str[id] = owned;
        t->vocab_len[id] = (u32)slen;
        strmap_insert(&smap, owned, (u32)slen, id);
    }

    build_byte_unicode_maps(t->byte_to_cpt, t->cpt_to_byte);

    /* byte -> vocab id (also asserts B2's "all 256 bytes are in the vocab
     * as single tokens" invariant, which the merge loop's base case and
     * tok_encode both depend on). */
    for (u32 b = 0; b < 256; b++) {
        u8  enc[4];
        u32 enc_len = utf8_encode_cpt(t->byte_to_cpt[b], enc);
        u32 id;
        if (!strmap_find(&smap, (const char*)enc, enc_len, &id)) {
            fatal("tok: byte 0x%02x's GPT-2 remap codepoint U+%04X has no "
                  "matching single-byte vocab entry -- task-6a-brief.md B2 "
                  "requires every raw byte to be representable",
                  b, t->byte_to_cpt[b]);
        }
        t->byte_to_id[b] = id;
    }

    u64 merge_count64 = gguf_meta_arr_len(f, "tokenizer.ggml.merges");
    if (merge_count64 > UINT32_MAX) {
        fatal("tok: tokenizer.ggml.merges has implausible length %llu",
              (unsigned long long)merge_count64);
    }
    u32 merge_count = (u32)merge_count64;
    mergemap_init(&t->merges, merge_count > 0 ? merge_count : 1);

    for (u32 rank = 0; rank < merge_count; rank++) {
        const char* line = gguf_meta_arr_str(f, "tokenizer.ggml.merges", rank);
        size_t      llen = strlen(line);
        const char* sp = (const char*)memchr(line, ' ', llen);
        if (!sp) {
            fatal("tok: tokenizer.ggml.merges[%u] = '%s' has no space "
                  "separator between its two halves",
                  rank, line);
        }
        size_t alen = (size_t)(sp - line);
        size_t blen = llen - alen - 1;
        const char* a = line;
        const char* b = sp + 1;

        u32 id_a, id_b, id_ab;
        if (!strmap_find(&smap, a, (u32)alen, &id_a)) {
            fatal("tok: tokenizer.ggml.merges[%u] left half '%.*s' not "
                  "found in vocab",
                  rank, (int)alen, a);
        }
        if (!strmap_find(&smap, b, (u32)blen, &id_b)) {
            fatal("tok: tokenizer.ggml.merges[%u] right half '%.*s' not "
                  "found in vocab",
                  rank, (int)blen, b);
        }
        char* ab = (char*)xmalloc(alen + blen);
        memcpy(ab, a, alen);
        memcpy(ab + alen, b, blen);
        if (!strmap_find(&smap, ab, (u32)(alen + blen), &id_ab)) {
            fatal("tok: tokenizer.ggml.merges[%u] merged form '%.*s' not "
                  "found in vocab",
                  rank, (int)(alen + blen), ab);
        }
        free(ab);

        mergemap_insert(&t->merges, id_a, id_b, rank, id_ab);
    }

    strmap_free(&smap);

    i64 bos64 = gguf_meta_i64(f, "tokenizer.ggml.bos_token_id");
    if (bos64 < 0 || (u64)bos64 >= vocab_size) {
        fatal("tok: tokenizer.ggml.bos_token_id %lld out of vocab range "
              "[0,%u)",
              (long long)bos64, vocab_size);
    }
    i64 eos64 = gguf_meta_i64(f, "tokenizer.ggml.eos_token_id");
    if (eos64 < 0 || (u64)eos64 >= vocab_size) {
        fatal("tok: tokenizer.ggml.eos_token_id %lld out of vocab range "
              "[0,%u)",
              (long long)eos64, vocab_size);
    }
    t->bos_id = (u32)bos64;
    t->eos_id = (u32)eos64;
    t->add_bos_token =
        gguf_meta_i64_or(f, "tokenizer.ggml.add_bos_token", 0) != 0;

    return t;
}

void tok_free(Tok* t) {
    if (!t) return;
    for (u32 i = 0; i < t->vocab_size; i++) free(t->vocab_str[i]);
    free(t->vocab_str);
    free(t->vocab_len);
    free(t->merges.slots);
    free(t);
}

/* ===========================================================================
 * BPE merge loop (task-6a-brief.md B5)
 * ===========================================================================
 * IMPORTANT: this operates on the fragment's CANONICAL bytes (from
 * decode_utf8_all's `canon` buffer), not the fragment's ORIGINAL raw bytes
 * and not the "classification" codepoints match_fragment() used to find
 * the fragment boundary -- three related but distinct things:
 *   - classification codepoint: what match_alt1..7 look at (letter/
 *     number/whitespace/other) to decide fragment boundaries.
 *   - canonical bytes: that same codepoint's minimal UTF-8 re-encoding --
 *     identical to the original input bytes for well-formed UTF-8, but
 *     DIFFERENT for an overlong sequence (shrinks) or a structural-decode
 *     failure (becomes U+FFFD's 3 bytes EF BF BD) -- see the block comment
 *     above utf8_decode_one for the oracle probing that established this.
 *   - BPE symbols: each canonical byte, individually remapped through
 *     byte_to_id[] (B2) as the STARTING symbol -- e.g. a 2-canonical-byte
 *     codepoint like U+00E9 (UTF-8 C3 A9, one letter, one fragment) starts
 *     BPE as TWO symbols, which may or may not merge back together
 *     depending on the trained merge table.
 * Byte-level BPE is intentionally agnostic to whether those canonical
 * bytes came from valid, overlong, or altogether-invalid input -- every
 * byte 0..255 has a byte_to_id[] entry, so this can never fail on content.
 */

static void bpe_encode_fragment(const Tok* t, const u8* bytes, u64 nbytes,
                                 u32* ids, u32* n_out, u32 cap) {
    if (nbytes == 0) return;

    u32* sym = (u32*)xcalloc(nbytes, sizeof(u32));
    u64  sym_count = nbytes;
    for (u64 i = 0; i < nbytes; i++) {
        sym[i] = t->byte_to_id[bytes[i]];
    }

    for (;;) {
        bool have_best = false;
        u32  best_rank = 0;
        u64  best_pos = 0;
        u32  best_merged = 0;
        for (u64 i = 0; i + 1 < sym_count; i++) {
            u32 rank, merged_id;
            if (mergemap_find(&t->merges, sym[i], sym[i + 1], &rank,
                               &merged_id)) {
                if (!have_best || rank < best_rank) {
                    have_best = true;
                    best_rank = rank;
                    best_pos = i;
                    best_merged = merged_id;
                }
            }
        }
        if (!have_best) break;
        sym[best_pos] = best_merged;
        for (u64 k = best_pos + 1; k + 1 < sym_count; k++) {
            sym[k] = sym[k + 1];
        }
        sym_count--;
    }

    for (u64 i = 0; i < sym_count; i++) {
        if (*n_out >= cap) {
            fatal("tok: tok_encode -- output cap %u exceeded (BPE never "
                  "emits more tokens than input CANONICAL bytes decode to, "
                  "so cap >= 3*<input byte length> is always sufficient -- "
                  "see src/tok.h's tok_encode contract -- check the "
                  "caller's cap sizing)",
                  cap);
        }
        ids[(*n_out)++] = sym[i];
    }
    free(sym);
}

/* ===========================================================================
 * Public encode / decode
 * ===========================================================================
 */

u32 tok_encode(const Tok* t, const char* utf8, u64 len, u32* ids, u32 cap,
               bool add_special) {
    u32 n_out = 0;

    if (add_special && t->add_bos_token) {
        if (n_out >= cap) {
            fatal("tok: tok_encode -- cap %u too small to hold BOS", cap);
        }
        ids[n_out++] = t->bos_id;
    }

    if (len == 0) return n_out;

    /* canon: worst case 3 canonical bytes per original byte (see the block
     * comment above utf8_decode_one and decode_utf8_all's own comment). */
    DecodedCp* cps = (DecodedCp*)xcalloc(len, sizeof(DecodedCp));
    u8*        canon = (u8*)xcalloc(len, 3);
    u64        canon_len = 0;
    u64        n_cp = decode_utf8_all((const u8*)utf8, len, cps, canon, &canon_len);

    u64 ci = 0;
    while (ci < n_cp) {
        u64 m = match_fragment(cps, n_cp, ci);
        u64 frag_end_ci = ci + m;
        u64 byte_start = cps[ci].canon_off;
        u64 byte_end =
            (frag_end_ci < n_cp) ? cps[frag_end_ci].canon_off : canon_len;

        bpe_encode_fragment(t, canon + byte_start, byte_end - byte_start,
                             ids, &n_out, cap);

        ci = frag_end_ci;
    }

    free(cps);
    free(canon);
    return n_out;
}

u32 tok_decode(const Tok* t, u32 id, char* buf, u32 cap) {
    if (id >= t->vocab_size) {
        fatal("tok: tok_decode -- id %u out of range (vocab size %u)", id,
              t->vocab_size);
    }
    const char* s = t->vocab_str[id];
    u32         slen = t->vocab_len[id];

    u32 w = 0;
    u64 pos = 0;
    while (pos < slen) {
        u32 clen;
        u32 cp = utf8_decode_one((const u8*)s, slen, pos, &clen);
        if (cp >= TOK_CPT_TABLE_SIZE || t->cpt_to_byte[cp] < 0) {
            fatal("tok: tok_decode -- id %u ('%.*s') contains codepoint "
                  "U+%04X which is not a valid GPT-2 byte-remap codepoint "
                  "-- decode in this task (6a) only covers byte-BPE-"
                  "derived tokens; special/added tokens are 6b's scope",
                  id, (int)slen, s, cp);
        }
        if (w >= cap) {
            fatal("tok: tok_decode -- cap %u exceeded decoding id %u", cap,
                  id);
        }
        buf[w++] = (char)(u8)t->cpt_to_byte[cp];
        pos += clen;
    }
    return w;
}

u32 tok_bos(const Tok* t) { return t->bos_id; }
u32 tok_eos(const Tok* t) { return t->eos_id; }
