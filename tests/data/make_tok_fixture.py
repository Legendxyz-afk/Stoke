#!/usr/bin/env python3
"""tests/data/make_tok_fixture.py -- generates tests/data/tiny_tok.gguf: a
tiny, handcrafted GGUF v3 fixture carrying a real (but minimal) gpt2/qwen2
byte-BPE tokenizer, for tests/test_tok.c's always-run (no 2.5GB real-model
dependency) unit tests. Run manually; output is committed binary data:

    python3 tests/data/make_tok_fixture.py

Contains:
  - all 256 GPT-2 byte-remap single-byte tokens (ids 0..255), computed via
    the same bytes_to_unicode() algorithm as the real gpt2/qwen2 tokenizer
    -- an INDEPENDENT implementation of that published spec (this script
    does not import src/tok.c; it exercises the real tokenizer against
    externally-known-correct bytes, same spirit as
    tests/data/make_fixtures.py's GGUF fixtures not reusing src/gguf.c).
  - a handful of extra multi-byte tokens + merge rules on top of those, so
    tests/test_tok.c can exercise: multi-step BPE merging ("hello" built
    from 4 chained merges), a leading-space ('Ġ') merge chain (mirrors the
    real "Hello world" example), and leftmost-on-tied-rank merge order
    ("on equal rank pick the leftmost" -- see test_tok.c's tie-break test
    for the worked-out reasoning: "aaa" via rules "a a"->"aa" then
    "aa a"->"aaa" only reconstructs correctly if the FIRST of two
    same-rank-candidate positions is merged first).
  - bos/eos ids and add_bos_token=true (deliberately the opposite of the
    real Qwen3 model's `false`, so tok_encode's add_special=true path has
    something to actually exercise here).
  - a tokenizer.ggml.token_type array (REQUIRED by tok_from_gguf) covering
    all 271 base+extra tokens above as NORMAL(1), plus 4 synthetic
    specials layered on top (ids 271..274):
    two CONTROL(3) (the literal ChatML strings "<|im_start|>"/"<|im_end|>",
    so tok_apply_chatml has something to exercise on this fixture too, not
    just tok_encode_special's generic partition path), one USER_DEFINED(4)
    ("<think>"), and one UNUSED(5) ("[UNUSED0]", mirroring the real
    model's "[PAD151669]"-style naming). All 271 base ids keep their
    ORIGINAL values (appended-not-inserted), so every id test_tok.c already
    hardcodes (259 "hello", 264 " world", 266 "aaa"/eos, ...) stays valid
    unchanged -- only the vocab_size-dependent "first invalid id" constant
    (used by the decode-out-of-range negative test) moves from 271 to 275.

build_base_vocab() (the shared 271-token/15-merge-rule base) is factored
out so every fixture below -- positive and negative -- builds on the exact
same underlying byte-BPE vocabulary, varying only what's layered on top
(specials / token_type), matching this file's existing negative-fixture
style (make_bad_model/make_bad_pre/make_missing_byte: minimal, deliberately
broken in exactly ONE way each, so each negative test exercises exactly
the fatal it claims to).

GGUF-writing Writer class mirrors tests/data/make_fixtures.py's (same wire
format), kept local rather than imported so each test-data generator stays
self-contained (matches make_fixtures.py's own no-external-deps-beyond-
struct style).
"""
import os
import struct

T_STRING = 8
T_ARRAY = 9
T_UINT32 = 4
T_INT32 = 5
T_BOOL = 7

GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


class Writer:
    def __init__(self):
        self.buf = bytearray()

    def tell(self):
        return len(self.buf)

    def i32(self, v):
        self.buf += struct.pack('<i', v)

    def u32(self, v):
        self.buf += struct.pack('<I', v & 0xFFFFFFFF)

    def u64(self, v):
        self.buf += struct.pack('<Q', v & 0xFFFFFFFFFFFFFFFF)

    def raw(self, b):
        self.buf += bytes(b)

    def gstr_bytes(self, b):
        self.u64(len(b))
        self.raw(b)

    def gstr(self, s):
        self.gstr_bytes(s.encode('utf-8'))

    def header(self, tensor_count, kv_count):
        self.raw(GGUF_MAGIC)
        self.u32(GGUF_VERSION)
        self.u64(tensor_count)
        self.u64(kv_count)

    def kv_string(self, key, val):
        self.gstr(key)
        self.u32(T_STRING)
        self.gstr(val)

    def kv_u32(self, key, val):
        self.gstr(key)
        self.u32(T_UINT32)
        self.u32(val)

    def kv_bool(self, key, val):
        self.gstr(key)
        self.u32(T_BOOL)
        self.raw(bytes([1 if val else 0]))

    def kv_array_str(self, key, vals):
        self.gstr(key)
        self.u32(T_ARRAY)
        self.u32(T_STRING)
        self.u64(len(vals))
        for v in vals:
            self.gstr(v)

    def kv_array_i32(self, key, vals):
        self.gstr(key)
        self.u32(T_ARRAY)
        self.u32(T_INT32)
        self.u64(len(vals))
        for v in vals:
            self.i32(v)


def pad_to(cur, align):
    return (align - (cur % align)) % align


def bytes_to_unicode():
    """Verbatim GPT-2 byte<->unicode remap algorithm (the published
    bytes_to_unicode() spec)."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


BYTE_TO_CPT = bytes_to_unicode()


def remap(s):
    """Byte-encodes an ASCII string through the GPT-2 byte-remap table (for
    building readable extra tokens/merges on top of the 256 base ones).
    Ordinary byte-BPE vocab entries are stored this way; special-token
    strings are NOT -- they're stored literally, so
    fixtures add them directly rather than through this function (in
    practice a no-op difference here, since every special string used below
    is pure "nice" printable ASCII 0x21-0x7E, remap()'s identity range --
    but written as literal strings anyway to mirror the real GGUF
    structure and not rely on that coincidence)."""
    return ''.join(chr(BYTE_TO_CPT[ord(c)]) for c in s)


BASE_VOCAB_SIZE = 271  # 256 base bytes + 15 extra tokens, asserted below


def build_base_vocab():
    """The 271-token base byte-BPE vocab (ids 0..270) + its 15 merge rules,
    shared by EVERY fixture in this file. Specials (ids 271+) are layered
    on top per-fixture, since the negative fixtures below each need a
    different token_type/specials configuration on top of this same
    base. Returns fresh (tokens, merges, id_of, bos_id, eos_id) -- no
    aliasing across separate calls."""
    base_tokens = [chr(BYTE_TO_CPT[b]) for b in range(256)]

    extra = [
        remap("he"),      # 256
        remap("hel"),     # 257
        remap("hell"),    # 258
        remap("hello"),   # 259
        remap(" w"),      # 260  (' ' remaps to 'Ġ')
        remap(" wo"),     # 261
        remap(" wor"),    # 262
        remap(" worl"),   # 263
        remap(" world"),  # 264
        remap("aa"),      # 265
        remap("aaa"),     # 266
        remap("12"),      # 267  (alt3 discriminator: exactly-one-digit)
        remap("()"),      # 268  (alt4 discriminator: one-or-more "other")
        remap(")\n"),     # 269  (alt4 discriminator: trailing [\r\n]*)
        remap("lo"),      # 270  (alt1 discriminator: "'llo" priority)
    ]
    tokens = base_tokens + extra
    assert len(tokens) == BASE_VOCAB_SIZE
    id_of = {s: i for i, s in enumerate(tokens)}

    def mrule(a, b):
        return f"{a} {b}"

    merges = [
        mrule(remap("h"), remap("e")),      # 0: h e -> he
        mrule(remap("he"), remap("l")),     # 1: he l -> hel
        mrule(remap("hel"), remap("l")),    # 2: hel l -> hell
        mrule(remap("hell"), remap("o")),   # 3: hell o -> hello
        mrule(remap(" "), remap("w")),      # 4: Gw -> ' w'
        mrule(remap(" w"), remap("o")),     # 5
        mrule(remap(" wo"), remap("r")),    # 6
        mrule(remap(" wor"), remap("l")),   # 7
        mrule(remap(" worl"), remap("d")),  # 8
        mrule(remap("a"), remap("a")),      # 9: a a -> aa
        mrule(remap("aa"), remap("a")),     # 10: aa a -> aaa
        mrule(remap("1"), remap("2")),      # 11: 1 2 -> 12
        mrule(remap("("), remap(")")),      # 12: ( ) -> ()
        mrule(remap(")"), remap("\n")),     # 13: ) \n -> )\n
        mrule(remap("l"), remap("o")),      # 14: l o -> lo
    ]

    for m in merges:
        a, b = m.split(' ', 1)
        assert a in id_of, f"merge half {a!r} missing from vocab"
        assert b in id_of, f"merge half {b!r} missing from vocab"
        assert (a + b) in id_of, f"merge result {(a + b)!r} missing from vocab"

    bos_id = id_of[remap("hello")]
    eos_id = id_of[remap("aaa")]
    return tokens, merges, id_of, bos_id, eos_id


# ===========================================================================
# tiny_tok.gguf -- the main positive fixture: base vocab + 4 synthetic
# specials + a well-formed token_type array covering all of it.
# ===========================================================================

# (string, llama_token_type) -- CONTROL(3), CONTROL(3), USER_DEFINED(4),
# UNUSED(5). The two CONTROL entries are the literal ChatML boundary
# strings so tok_apply_chatml has real ids to look up on this fixture, not
# just tok_encode_special's generic partition path.
SPECIALS = [
    ("<|im_start|>", 3),
    ("<|im_end|>", 3),
    ("<think>", 4),
    ("[UNUSED0]", 5),
]


def make_tiny_tok():
    tokens, merges, id_of, bos_id, eos_id = build_base_vocab()

    for s, _ty in SPECIALS:
        assert s not in id_of, f"special {s!r} collides with base vocab"
        id_of[s] = len(tokens)
        tokens.append(s)
    assert len(tokens) == BASE_VOCAB_SIZE + len(SPECIALS)

    token_type = [1] * BASE_VOCAB_SIZE + [ty for _s, ty in SPECIALS]
    assert len(token_type) == len(tokens)

    w = Writer()
    w.header(tensor_count=0, kv_count=8)
    w.kv_string("tokenizer.ggml.model", "gpt2")
    w.kv_string("tokenizer.ggml.pre", "qwen2")
    w.kv_array_str("tokenizer.ggml.tokens", tokens)
    w.kv_array_str("tokenizer.ggml.merges", merges)
    w.kv_u32("tokenizer.ggml.bos_token_id", bos_id)
    w.kv_u32("tokenizer.ggml.eos_token_id", eos_id)
    w.kv_bool("tokenizer.ggml.add_bos_token", True)
    w.kv_array_i32("tokenizer.ggml.token_type", token_type)
    cur = w.tell()
    pad = pad_to(cur, 32)
    w.raw(b'\x00' * pad)
    return w.buf, tokens, merges, id_of, bos_id, eos_id


# ===========================================================================
# 6a negative fixtures -- UNCHANGED from task-6a (each fatals before ever
# reaching the token_type check added in 6b, so none of them need one; see
# src/tok.c's tok_from_gguf -- the token_type/specials-table block is
# deliberately built LAST, after model/pre/tokens/merges/byte_to_id are all
# already validated).
# ===========================================================================

def make_bad_model():
    """tokenizer.ggml.model wrong -- tok_from_gguf must fatal on the very
    first check, before touching tokens/merges/pre, so this fixture needs
    nothing beyond that one KV."""
    w = Writer()
    w.header(tensor_count=0, kv_count=1)
    w.kv_string("tokenizer.ggml.model", "bert-wordpiece")
    cur = w.tell()
    w.raw(b'\x00' * pad_to(cur, 32))
    return w.buf


def make_bad_pre():
    """tokenizer.ggml.model correct (so that check passes) but
    tokenizer.ggml.pre wrong -- tok_from_gguf must fatal on the second
    check, before touching tokens/merges."""
    w = Writer()
    w.header(tensor_count=0, kv_count=2)
    w.kv_string("tokenizer.ggml.model", "gpt2")
    w.kv_string("tokenizer.ggml.pre", "llama3")
    cur = w.tell()
    w.raw(b'\x00' * pad_to(cur, 32))
    return w.buf


def make_missing_byte():
    """A vocab missing ONE of the 256 required single-byte tokens (skips
    byte 0x41 'A') -- tok_from_gguf must fatal building byte_to_id[], per
    the "all 256 bytes are in the vocab as single tokens" invariant.
    model/pre are correct so the fatal under test is specifically the
    missing-byte one, not an earlier check firing first. No token_type
    array (still not needed -- this fatal fires before the token_type
    check is ever reached)."""
    skip_byte = 0x41
    tokens = [chr(BYTE_TO_CPT[b]) for b in range(256) if b != skip_byte]
    w = Writer()
    w.header(tensor_count=0, kv_count=6)
    w.kv_string("tokenizer.ggml.model", "gpt2")
    w.kv_string("tokenizer.ggml.pre", "qwen2")
    w.kv_array_str("tokenizer.ggml.tokens", tokens)
    w.kv_array_str("tokenizer.ggml.merges", [])
    w.kv_u32("tokenizer.ggml.bos_token_id", 0)
    w.kv_u32("tokenizer.ggml.eos_token_id", 0)
    cur = w.tell()
    w.raw(b'\x00' * pad_to(cur, 32))
    return w.buf


# ===========================================================================
# Special-token/token_type negative fixtures: each is the SAME valid
# 271-token base vocab, varying only the
# token_type array or specials set in exactly the one way needed to
# exercise its fatal.
# ===========================================================================

def _write_base_tokenizer_kvs(w, tokens, merges, bos_id, eos_id, kv_count):
    """The 7 tokenizer.ggml.* KVs every 6b fixture below starts with
    (model/pre/tokens/merges/bos/eos/add_bos_token) -- callers add their
    own token_type (or not) afterward and pass the resulting total
    kv_count up front, since GGUF's header declares kv_count before any KV
    bytes follow."""
    w.header(tensor_count=0, kv_count=kv_count)
    w.kv_string("tokenizer.ggml.model", "gpt2")
    w.kv_string("tokenizer.ggml.pre", "qwen2")
    w.kv_array_str("tokenizer.ggml.tokens", tokens)
    w.kv_array_str("tokenizer.ggml.merges", merges)
    w.kv_u32("tokenizer.ggml.bos_token_id", bos_id)
    w.kv_u32("tokenizer.ggml.eos_token_id", eos_id)
    w.kv_bool("tokenizer.ggml.add_bos_token", True)


def make_no_token_type():
    """Otherwise-fully-valid tokenizer (same 271-token base as tiny_tok,
    no specials needed) but NO tokenizer.ggml.token_type key at all --
    tok_from_gguf must fatal on the "required" check."""
    tokens, merges, _id_of, bos_id, eos_id = build_base_vocab()
    w = Writer()
    _write_base_tokenizer_kvs(w, tokens, merges, bos_id, eos_id, kv_count=7)
    cur = w.tell()
    w.raw(b'\x00' * pad_to(cur, 32))
    return w.buf


def make_bad_token_type_len():
    """token_type array present but ONE SHORT of vocab_size -- must fatal
    on the length check."""
    tokens, merges, _id_of, bos_id, eos_id = build_base_vocab()
    token_type = [1] * (len(tokens) - 1)
    w = Writer()
    _write_base_tokenizer_kvs(w, tokens, merges, bos_id, eos_id, kv_count=8)
    w.kv_array_i32("tokenizer.ggml.token_type", token_type)
    cur = w.tell()
    w.raw(b'\x00' * pad_to(cur, 32))
    return w.buf


def make_bad_token_type_value():
    """token_type array is the correct length, but ONE element (id 100,
    an arbitrary base-byte id) is 7 -- outside llama_token_type's 0..6 --
    must fatal on the per-element range check."""
    tokens, merges, _id_of, bos_id, eos_id = build_base_vocab()
    token_type = [1] * len(tokens)
    token_type[100] = 7
    w = Writer()
    _write_base_tokenizer_kvs(w, tokens, merges, bos_id, eos_id, kv_count=8)
    w.kv_array_i32("tokenizer.ggml.token_type", token_type)
    cur = w.tell()
    w.raw(b'\x00' * pad_to(cur, 32))
    return w.buf


def make_no_imstart():
    """Fully valid tokenizer, INCLUDING a well-formed token_type array and
    a non-empty specials table (one USER_DEFINED, one UNUSED) -- but no
    token literally equal to "<|im_start|>" anywhere in the vocab.
    tok_from_gguf() must SUCCEED (loading a model with no ChatML-style
    specials is not itself an error); tok_apply_chatml() must fatal when
    actually called (the documented "model lacks <|im_start|>" case)."""
    tokens, merges, id_of, bos_id, eos_id = build_base_vocab()
    specials = [("<think>", 4), ("[UNUSED0]", 5)]
    for s, _ty in specials:
        assert s not in id_of
        id_of[s] = len(tokens)
        tokens.append(s)
    token_type = [1] * BASE_VOCAB_SIZE + [ty for _s, ty in specials]

    w = Writer()
    _write_base_tokenizer_kvs(w, tokens, merges, bos_id, eos_id, kv_count=8)
    w.kv_array_i32("tokenizer.ggml.token_type", token_type)
    cur = w.tell()
    w.raw(b'\x00' * pad_to(cur, 32))
    return w.buf


def main():
    data, tokens, merges, id_of, bos_id, eos_id = make_tiny_tok()
    path = os.path.join(OUT_DIR, "tiny_tok.gguf")
    with open(path, 'wb') as f:
        f.write(data)
    print(f"wrote {path} ({len(data)} bytes, {len(tokens)} tokens, "
          f"{len(merges)} merges, bos={bos_id}, eos={eos_id})")
    # Print the ids test_tok.c hardcodes, so its expectations are visibly
    # derived from this same generator rather than hand-guessed.
    for name in ("he", "hel", "hell", "hello", " w", " wo", " wor", " worl",
                 " world", "aa", "aaa", "12", "()", ")\n", "lo"):
        print(f"  id({name!r}) = {id_of[remap(name)]}")
    for b in (0x00, 0x20, 0x41, 0xFF):
        print(f"  id(byte 0x{b:02x}) = {id_of[chr(BYTE_TO_CPT[b])]}")
    for s, _ty in SPECIALS:
        print(f"  id({s!r}) = {id_of[s]}")
    print(f"  vocab_size = {len(tokens)} (first invalid id for decode)")

    for name, fn in (("tiny_tok_bad_model.gguf", make_bad_model),
                      ("tiny_tok_bad_pre.gguf", make_bad_pre),
                      ("tiny_tok_missing_byte.gguf", make_missing_byte),
                      ("tiny_tok_no_token_type.gguf", make_no_token_type),
                      ("tiny_tok_bad_token_type_len.gguf",
                       make_bad_token_type_len),
                      ("tiny_tok_bad_token_type_value.gguf",
                       make_bad_token_type_value),
                      ("tiny_tok_no_imstart.gguf", make_no_imstart)):
        p = os.path.join(OUT_DIR, name)
        d = fn()
        with open(p, 'wb') as f:
            f.write(d)
        print(f"wrote {p} ({len(d)} bytes)")


if __name__ == "__main__":
    main()
