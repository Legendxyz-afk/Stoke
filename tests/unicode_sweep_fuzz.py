#!/usr/bin/env python3
"""tests/unicode_sweep_fuzz.py -- generates tests/data/unicode_sweep.txt: a
deterministic, fixed-stride sweep across the ENTIRE Unicode codepoint space
(0x0000..0x10FFFF, including unassigned codepoints), closing the
documented "Unicode 16.0 (src/unicode_data.h's generation source) vs
b10037 (the oracle's build)" version-skew risk that the committed
tok_corpus.txt / tok_special_corpus.txt corpora -- built from a few
hundred distinct codepoints found in real-world text -- never actually
exercised. Run manually; output is committed:

    python3 tests/unicode_sweep_fuzz.py

Sweep parameters (fixed; do not change without re-validating against the
oracle -- a different stride is a different, unvalidated experiment):
  - cp from 0x0000 to 0x10FFFF inclusive, stride 146, starting at cp=0.
  - Skip cp in 0xD800..0xDFFF (UTF-16 surrogates -- not valid UTF-8 on
    their own) and cp in {0x0000, 0x000A, 0x000D} (NUL and the line
    terminators would corrupt a line-oriented corpus and its golden).
  - This yields 7616 surviving codepoints: floor(0x10FFFF / 146) + 1 =
    7631 raw stride candidates (cp = 0, 146, 292, ..., 1113980), minus 1
    for cp=0 itself, minus 14 further candidates that land in the
    surrogate range (the multiples of 146 from 379*146=55334 through
    392*146=57232) = 7616. (0x000A/0x000D are never stride candidates in
    the first place -- neither is a multiple of 146 -- so excluding them
    changes nothing; only cp=0 needed the explicit skip.)
  - 3 lines per surviving codepoint, in this exact order, X being that
    codepoint: (1) X alone; (2) " Xa" (space, X, ASCII 'a'); (3) "aX1"
    (ASCII 'a', X, ASCII '1'). These are the fragmentation-discriminating
    contexts: if src/unicode_data.h classifies X in a different
    Letter/Number/Whitespace/Other category than the oracle does, at
    least one of the three splits differently and the ids diverge.
  - 7616 * 3 = 22848 lines total.

Deterministic: no randomness, no set/dict iteration order (plain integer
stride), binary-mode write with `\\n` endings only (matches
tok_fuzz.py/tok_special_fuzz.py -- a CRLF slip changes every id downstream
and makes the golden platform-dependent).

Unassigned codepoints are DELIBERATELY INCLUDED (not skipped): the whole
point of this sweep is to catch cases where src/unicode_data.h (generated
from this box's Python `unicodedata`, Unicode 16.0 -- see
tests/gen_unicode_data.py) disagrees with the oracle's own Unicode-version
classification of some codepoint, and "unassigned in 16.0 but classified
differently by whatever Unicode version the oracle's build used" is
exactly the kind of codepoint that could disagree.

Regenerating the golden after a sweep-parameter change (do not do this
lightly -- see the "fixed; do not change" note above):
    bench/vendor/llama-b10037/llama-tokenize.exe \\
        -m models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \\
        -f tests/data/unicode_sweep.txt --ids --no-escape --no-bos \\
        --no-parse-special
    then strip the raw stdout to just the "[...]" list (LF-terminated, no
    other bytes) and save as tests/data/unicode_sweep.ids.golden. Plain
    byte-BPE mode only (--no-parse-special, gating plain `stoke
    tokenize`): this gate is about the Unicode tables, not specials.
"""

OUT_PATH = "tests/data/unicode_sweep.txt"

MAX_CP = 0x10FFFF
STRIDE = 146
SURROGATE_LO = 0xD800
SURROGATE_HI = 0xDFFF
EXCLUDED = {0x0000, 0x000A, 0x000D}

EXPECTED_CODEPOINTS = 7616
EXPECTED_LINES = 22848


def surviving_codepoints():
    cp = 0
    while cp <= MAX_CP:
        if cp not in EXCLUDED and not (SURROGATE_LO <= cp <= SURROGATE_HI):
            yield cp
        cp += STRIDE


def main():
    cps = list(surviving_codepoints())
    assert len(cps) == EXPECTED_CODEPOINTS, (
        f"codepoint count drifted: got {len(cps)}, expected "
        f"{EXPECTED_CODEPOINTS} -- this is a sanity check on the stride "
        f"math, not a number to force; if it fires, the sweep parameters "
        f"changed and the golden is no longer valid for them")

    lines = []
    for cp in cps:
        ch = chr(cp)
        lines.append(ch)          # (1) X alone
        lines.append(f" {ch}a")   # (2) space, X, ASCII 'a'
        lines.append(f"a{ch}1")   # (3) ASCII 'a', X, ASCII '1'

    with open(OUT_PATH, "wb") as f:
        n_written = 0
        for line in lines:
            f.write(line.encode("utf-8"))
            f.write(b"\n")
            n_written += 1

    assert n_written == EXPECTED_LINES, (
        f"line count drifted: got {n_written}, expected {EXPECTED_LINES}")

    print(f"wrote {OUT_PATH}: {len(cps)} codepoints, {n_written} lines")


if __name__ == "__main__":
    main()
