#!/usr/bin/env python3
"""tests/data/make_tok_fixture.py -- generates tests/data/tiny_tok.gguf: a
tiny, handcrafted GGUF v3 fixture carrying a real (but minimal) gpt2/qwen2
byte-BPE tokenizer, for tests/test_tok.c's always-run (no 2.5GB real-model
dependency) unit tests. Run manually; output is committed binary data:

    python3 tests/data/make_tok_fixture.py

Contains:
  - all 256 GPT-2 byte-remap single-byte tokens (ids 0..255), computed via
    the same bytes_to_unicode() algorithm as task-6a-brief.md B2 -- an
    INDEPENDENT implementation of that published spec (this script does not
    import src/tok.c; it exercises the real tokenizer against externally-
    known-correct bytes, same spirit as tests/data/make_fixtures.py's GGUF
    fixtures not reusing src/gguf.c).
  - a handful of extra multi-byte tokens + merge rules on top of those, so
    tests/test_tok.c can exercise: multi-step BPE merging ("hello" built
    from 4 chained merges), a leading-space ('Ġ') merge chain (mirrors the
    real "Hello world" example), and leftmost-on-tied-rank merge order
    (task-6a-brief.md B5's "on equal rank pick the leftmost" -- see
    test_tok.c's tie-break test for the worked-out reasoning: "aaa" via
    rules "a a"->"aa" then "aa a"->"aaa" only reconstructs correctly if the
    FIRST of two same-rank-candidate positions is merged first).
  - bos/eos ids and add_bos_token=true (deliberately the opposite of the
    real Qwen3 model's `false`, so tok_encode's add_special=true path has
    something to actually exercise here).

GGUF-writing Writer class mirrors tests/data/make_fixtures.py's (same wire
format, task-5-brief.md appendix A1/A2), kept local rather than imported so
each test-data generator stays self-contained (matches make_fixtures.py's
own no-external-deps-beyond-struct style).
"""
import os
import struct

T_STRING = 8
T_ARRAY = 9
T_UINT32 = 4
T_BOOL = 7

GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


class Writer:
    def __init__(self):
        self.buf = bytearray()

    def tell(self):
        return len(self.buf)

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


def pad_to(cur, align):
    return (align - (cur % align)) % align


def bytes_to_unicode():
    """Verbatim task-6a-brief.md B2 algorithm."""
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
    building readable extra tokens/merges on top of the 256 base ones)."""
    return ''.join(chr(BYTE_TO_CPT[ord(c)]) for c in s)


def make_tiny_tok():
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

    w = Writer()
    w.header(tensor_count=0, kv_count=7)
    w.kv_string("tokenizer.ggml.model", "gpt2")
    w.kv_string("tokenizer.ggml.pre", "qwen2")
    w.kv_array_str("tokenizer.ggml.tokens", tokens)
    w.kv_array_str("tokenizer.ggml.merges", merges)
    w.kv_u32("tokenizer.ggml.bos_token_id", bos_id)
    w.kv_u32("tokenizer.ggml.eos_token_id", eos_id)
    w.kv_bool("tokenizer.ggml.add_bos_token", True)
    cur = w.tell()
    pad = pad_to(cur, 32)
    w.raw(b'\x00' * pad)
    return w.buf, tokens, merges, id_of, bos_id, eos_id


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
    task-6a-brief.md B2's "all 256 bytes are in the vocab as single
    tokens" invariant. model/pre are correct so the fatal under test is
    specifically the missing-byte one, not an earlier check firing first."""
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

    for name, fn in (("tiny_tok_bad_model.gguf", make_bad_model),
                      ("tiny_tok_bad_pre.gguf", make_bad_pre),
                      ("tiny_tok_missing_byte.gguf", make_missing_byte)):
        p = os.path.join(OUT_DIR, name)
        d = fn()
        with open(p, 'wb') as f:
            f.write(d)
        print(f"wrote {p} ({len(d)} bytes)")


if __name__ == "__main__":
    main()
