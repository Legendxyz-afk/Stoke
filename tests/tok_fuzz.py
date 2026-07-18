#!/usr/bin/env python3
"""tests/tok_fuzz.py -- generates tests/data/tok_corpus.txt: a ~2000-line
deterministic (fixed seed) multilingual fuzz corpus for the byte-exact BPE
golden gate. Run manually; output is committed:

    python3 tests/tok_fuzz.py

Regenerating the golden after a corpus change (see tests/regen_goldens.sh
to do this mechanically for all three goldens at once):
    bench/vendor/llama-b10037/llama-tokenize.exe \\
        -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \\
        -f tests/data/tok_corpus.txt --ids --no-escape --no-parse-special --no-bos
    then strip the raw stdout to just the "[...]" list (LF-terminated, no
    other bytes) and save as tests/data/tok_corpus.ids.golden.

Corpus categories: plain ASCII prose, source code
(punctuation clusters), emoji incl. ZWJ sequences, CJK (Chinese/Japanese/
Korean), Devanagari, Arabic (RTL), long whitespace runs, tabs, a 10 KB
single line, an empty line, control chars, and lone UTF-8 surrogate/
continuation bytes. Deterministic across OSes: `\\n` line endings only,
UTF-8, no platform paths, fixed random seed.
"""
import random

SEED = 20260717
OUT_PATH = "tests/data/tok_corpus.txt"
TARGET_LINES = 2000

random.seed(SEED)

# ---- category source material ----

ASCII_WORDS = (
    "the quick brown fox jumps over the lazy dog while the sun sets "
    "behind rolling hills and a gentle breeze carries the scent of pine "
    "through the valley where children used to play until autumn came "
    "and painted every leaf a different shade of amber orange and red"
).split()

CODE_LINES = [
    "def foo(): return 1",
    "for (int i = 0; i < n; i++) { sum += arr[i]; }",
    "const x = {a: 1, b: [2, 3], c: () => null};",
    "SELECT * FROM users WHERE id = 42 AND active = true;",
    'if (!ptr) { fatal("null pointer at %s:%d", __FILE__, __LINE__); }',
    "class Foo(Bar, Baz):\n    def __init__(self, **kwargs):\n        pass",
    "let result = a.map(x => x * 2).filter(x => x > 0).reduce((a,b)=>a+b,0);",
    '#include <stdio.h>\nint main(void) { printf("hi\\n"); return 0; }',
    "std::vector<int> v = {1,2,3}; v.push_back(4); // comment",
    "x = (a+b)*(c-d)/e % f; assert(x >= 0 and x <= 100)",
    "curl -sSL https://example.com/api?key=val&other=1 | jq '.data[]'",
    'git commit -m "fix: handle edge-case in tokenizer (#42)"',
]

CJK_CHARS = list("日本語中国語한국어漢字文字言語東京北京首爾平仮名片仮名")
DEVANAGARI_WORDS = ["नमस्ते", "धन्यवाद", "भारत", "हिन्दी", "योग", "संस्कृत", "गणेश", "आकाश"]
ARABIC_WORDS = ["مرحبا", "شكرا", "العربية", "السلام", "كتاب", "مدينة", "جميل", "سعيد"]
EMOJI = ["🙂", "🚀", "🔥", "✨", "🌍", "🎉", "🐍", "☕", "💡", "🧠"]
ZWJ_FAMILY = "👨‍👩‍👧"  # man + ZWJ + woman + ZWJ + girl
ZWJ_SEQS = [ZWJ_FAMILY, "👍\U0001f3fd", "🏳️‍🌈", "👩‍💻"]


def line_ascii_prose():
    n = random.randint(4, 20)
    words = [random.choice(ASCII_WORDS) for _ in range(n)]
    return " ".join(words) + random.choice(["", ".", "!", "?"])


def line_code():
    return random.choice(CODE_LINES)


def line_emoji():
    n = random.randint(1, 5)
    parts = []
    for _ in range(n):
        if random.random() < 0.3:
            parts.append(random.choice(ZWJ_SEQS))
        else:
            parts.append(random.choice(EMOJI))
    return "text " + " ".join(parts) + " more text"


def line_cjk():
    n = random.randint(3, 15)
    return "".join(random.choice(CJK_CHARS) for _ in range(n))


def line_devanagari():
    n = random.randint(2, 8)
    return " ".join(random.choice(DEVANAGARI_WORDS) for _ in range(n))


def line_arabic():
    n = random.randint(2, 8)
    return " ".join(random.choice(ARABIC_WORDS) for _ in range(n))


def line_whitespace_runs():
    parts = []
    for _ in range(random.randint(2, 6)):
        parts.append(random.choice(ASCII_WORDS))
        parts.append(" " * random.randint(1, 12))
    return "".join(parts).rstrip(" ") + " " * random.randint(0, 3)


def line_tabs():
    words = [random.choice(ASCII_WORDS) for _ in range(random.randint(3, 8))]
    return "\t".join(words)


def line_control_chars():
    # a scatter of low control bytes (not \r\n -- those are tested via
    # RAW_BYTE_LINES/whitespace-run lines) mixed with ordinary ASCII text,
    # staying valid UTF-8 (control chars 0x00-0x1F minus \t\n\r are all
    # valid 1-byte UTF-8 codepoints on their own).
    ctrl = [chr(c) for c in (0x01, 0x02, 0x05, 0x0B, 0x0C, 0x1B, 0x1F)]
    words = [random.choice(ASCII_WORDS) for _ in range(4)]
    out = []
    for w in words:
        out.append(w)
        out.append(random.choice(ctrl))
    return "".join(out)


def line_mixed_multilingual():
    parts = [
        random.choice(ASCII_WORDS),
        random.choice(CJK_CHARS) * random.randint(1, 3),
        random.choice(DEVANAGARI_WORDS),
        random.choice(EMOJI),
        random.choice(ARABIC_WORDS),
    ]
    random.shuffle(parts)
    return " ".join(parts)


def line_long():
    words = []
    total = 0
    while total < 10000:
        w = random.choice(ASCII_WORDS)
        words.append(w)
        total += len(w) + 1
    return " ".join(words)


# ---- lines that must be written as raw bytes (invalid UTF-8 on purpose) ----
# Lone UTF-8 surrogate/continuation bytes: write raw bytes 0x80-0xBF and
# 0xED 0xA0 0x80-style sequences to stress invalid-UTF-8 handling. Written
# directly as bytes (Python str can't hold unpaired surrogates as UTF-8);
# harmless surrounding ASCII keeps each line easy to eyeball in a hex dump
# when debugging a golden mismatch.
RAW_BYTE_LINES = [
    b"lone continuation bytes: " + bytes([0x80, 0x81, 0x9F, 0xBF]) + b" end",
    b"surrogate-style 3-byte seq: " + bytes([0xED, 0xA0, 0x80]) + b" end",
    b"surrogate-style high: " + bytes([0xED, 0xBF, 0xBF]) + b" end",
    b"overlong 2-byte NUL: " + bytes([0xC0, 0x80]) + b" end",
    b"overlong 3-byte: " + bytes([0xE0, 0x80, 0x80]) + b" end",
    b"truncated 3-byte at EOL: text " + bytes([0xE2, 0x82]),
    b"truncated 4-byte: " + bytes([0xF0, 0x9F, 0x98]) + b" end",
    b"lead with no continuation: " + bytes([0xC3]) + b"x" + bytes([0xE0]) + b"y",
    b"bare high continuation run: " + bytes([0x80]) * 5 + b" end",
    b"invalid lead 0xFF/0xFE: " + bytes([0xFF, 0xFE]) + b" end",
]

GENERATORS = [
    (line_ascii_prose, 5),
    (line_code, 3),
    (line_emoji, 2),
    (line_cjk, 2),
    (line_devanagari, 1),
    (line_arabic, 1),
    (line_whitespace_runs, 2),
    (line_tabs, 1),
    (line_control_chars, 1),
    (line_mixed_multilingual, 2),
]


def build_weighted_choices():
    choices = []
    for fn, weight in GENERATORS:
        choices.extend([fn] * weight)
    return choices


def main():
    choices = build_weighted_choices()
    n_text = TARGET_LINES - len(RAW_BYTE_LINES) - 2  # -2: empty line + 10KB line
    text_lines = [random.choice(choices)() for _ in range(n_text)]
    empty_at = random.randint(10, len(text_lines) - 10)

    with open(OUT_PATH, "wb") as f:
        n_written = 0
        for i, line in enumerate(text_lines):
            if i == empty_at:
                f.write(b"\n")
                n_written += 1
            f.write(line.encode("utf-8"))
            f.write(b"\n")
            n_written += 1
        f.write(line_long().encode("utf-8"))
        f.write(b"\n")
        n_written += 1
        for rl in RAW_BYTE_LINES:
            f.write(rl)
            f.write(b"\n")
            n_written += 1

    print(f"wrote {OUT_PATH}: {n_written} lines")


if __name__ == "__main__":
    main()
