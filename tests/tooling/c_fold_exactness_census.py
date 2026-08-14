#!/usr/bin/env python3
"""c_fold_exactness_census.py -- can CONTRACT v2's (f32)S_b ever lose bits on
THIS model?

WHY THIS EXISTS. M4.5 rung C replaced a fold that converted eight i32 LANES to
f32 with one that converts their single integer TOTAL, S_b. f32 represents
every integer below 2^24 = 16,777,216 and only some above it, so the new
conversion can round where the old one -- each lane carrying about an eighth
of the total -- could not. tests/test_c_oracle.c settles the integer core
(totals bit-identical m4-vs-HEAD) and censuses S_b over a RANDOM corpus, which
is not this model's distribution. This tool answers the question for the
shipping weights.

WHAT IT COMPUTES, and why it is a bound rather than a sample. For each Q4_K /
Q6_K super-block it computes

    B = sum_j |scale_j| * sum_{k in sub-block j} |w_k| * 127

which is the maximum |S_b| over EVERY possible activation, since |q8| <= 127
is guaranteed by quantize_row_q8_k (iscale = -127/max). If B < 2^24 for every
super-block in the file, then (f32)S_b is exact for every input this model can
ever see -- a statement no sampling run could make. Only if B crosses 2^24
does the question become empirical.

Usage:  python tests/tooling/c_fold_exactness_census.py <model.gguf> [--selfcheck]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from gguf_reader import (GGML_NAME, GGML_Q4_K, GGML_Q6_K,  # noqa: E402
                         GGUFFile)

F32_EXACT_INT_LIMIT = 1 << 24
Q8_MAX = 127
CHUNK_BLOCKS = 1 << 17  # ~131k blocks: 19 MB of Q4_K raw per chunk

Q4K_DT = np.dtype([("d", "<f2"), ("dmin", "<f2"),
                   ("scales", "u1", (12,)), ("qs", "u1", (128,))])
Q6K_DT = np.dtype([("ql", "u1", (128,)), ("qh", "u1", (64,)),
                   ("scales", "i1", (16,)), ("d", "<f2")])


def scale_min_k4(scales, j):
    """ggml's get_scale_min_k4, vectorised -- the scale half only.

    Deliberately a second transcription rather than an import: gguf_reader's
    copy is verified against llama.cpp's own dequant output, and if these two
    disagree the census is wrong in a way worth finding. --selfcheck compares
    them.
    """
    if j < 4:
        return (scales[:, j] & 63).astype(np.int32)
    return (((scales[:, j + 4] & 0xF) |
             ((scales[:, j - 4] >> 6) << 4))).astype(np.int32)


def q4k_bounds(raw, nblocks):
    """Upper bound on |S_b| per super-block, shape [nblocks]."""
    b = np.frombuffer(raw, dtype=Q4K_DT, count=nblocks)
    scales, qs = b["scales"], b["qs"]
    total = np.zeros(nblocks, dtype=np.int64)
    for j in range(8):
        g = j // 2
        span = qs[:, g * 32:(g + 1) * 32]
        nib = (span & 0xF) if (j % 2 == 0) else (span >> 4)
        total += scale_min_k4(scales, j).astype(np.int64) * \
            nib.sum(axis=1, dtype=np.int64)
    return total * Q8_MAX


def q6k_aux8(b, nblocks):
    """The centred i8 quants in the order the kernels build them."""
    ql, qh = b["ql"], b["qh"]
    a = np.empty((nblocks, 256), dtype=np.int16)
    for h in range(2):
        base = h * 128
        ql_h = ql[:, h * 64:(h + 1) * 64]
        qh_h = qh[:, h * 32:(h + 1) * 32]
        a[:, base + 0:base + 32] = (ql_h[:, 0:32] & 0xF) | (((qh_h >> 0) & 3) << 4)
        a[:, base + 32:base + 64] = (ql_h[:, 32:64] & 0xF) | (((qh_h >> 2) & 3) << 4)
        a[:, base + 64:base + 96] = (ql_h[:, 0:32] >> 4) | (((qh_h >> 4) & 3) << 4)
        a[:, base + 96:base + 128] = (ql_h[:, 32:64] >> 4) | (((qh_h >> 6) & 3) << 4)
    return a - 32


def q6k_bounds(raw, nblocks):
    b = np.frombuffer(raw, dtype=Q6K_DT, count=nblocks)
    a = np.abs(q6k_aux8(b, nblocks)).reshape(nblocks, 16, 16)
    sub = a.sum(axis=2, dtype=np.int64)                       # [nb,16]
    sc = np.abs(b["scales"].astype(np.int64))                 # [nb,16]
    return (sub * sc).sum(axis=1) * Q8_MAX


def census(path, selfcheck):
    with GGUFFile(path) as f:
        tensors = [t for t in f.tensors.values()
                   if t.type in (GGML_Q4_K, GGML_Q6_K)]
        if not tensors:
            print("no Q4_K/Q6_K tensors in this file")
            return 0
        print(f"{path}\n{len(tensors)} K-quant tensors, 2^24 = "
              f"{F32_EXACT_INT_LIMIT}\n")

        if selfcheck:
            run_selfcheck(f, tensors)

        grand = {}
        for t in sorted(tensors, key=lambda x: x.name):
            block_bytes = 144 if t.type == GGML_Q4_K else 210
            fn = q4k_bounds if t.type == GGML_Q4_K else q6k_bounds
            nb_total = t.nbytes // block_bytes
            worst, over, blocks = 0, 0, 0
            for start in range(0, nb_total, CHUNK_BLOCKS):
                n = min(CHUNK_BLOCKS, nb_total - start)
                off = t.abs_offset + start * block_bytes
                raw = f._mm[off:off + n * block_bytes]
                bounds = fn(raw, n)
                worst = max(worst, int(bounds.max()))
                over += int((bounds >= F32_EXACT_INT_LIMIT).sum())
                blocks += n
            name = GGML_NAME[t.type]
            g = grand.setdefault(name, [0, 0, 0])
            g[0] = max(g[0], worst)
            g[1] += over
            g[2] += blocks
            flag = "  <-- CROSSES 2^24" if over else ""
            print(f"  {t.name:44s} {name:5s} blocks={blocks:9d} "
                  f"max_bound={worst:12d} over={over}{flag}")

        print()
        crossed = 0
        for name, (worst, over, blocks) in sorted(grand.items()):
            pct = 100.0 * worst / F32_EXACT_INT_LIMIT
            print(f"{name}: {blocks} super-blocks, worst possible |S_b| = "
                  f"{worst} ({pct:.2f}% of 2^24), super-blocks that could "
                  f"reach 2^24: {over}")
            crossed += over
        print()
        if crossed == 0:
            print("VERDICT: no super-block in this file can reach 2^24 under "
                  "ANY activation, so (f32)S_b is EXACT everywhere and "
                  "CONTRACT v2's fold loses nothing the m4 fold kept.")
        else:
            print(f"VERDICT: {crossed} super-blocks can exceed 2^24 -- the "
                  "conversion CAN round on this model; measure how often it "
                  "does on real activations before concluding anything.")
        return 0


def run_selfcheck(f, tensors):
    """Two independent checks on the vectorised bounds above, on one real
    super-block of each type: the scale unpack against gguf_reader's own
    (llama.cpp-verified) copy, and the bound against a plain Python loop."""
    print("selfcheck:")
    for want_type, block_bytes in ((GGML_Q4_K, 144), (GGML_Q6_K, 210)):
        t = next((x for x in tensors if x.type == want_type), None)
        if t is None:
            continue
        raw = bytes(f._mm[t.abs_offset:t.abs_offset + block_bytes])
        if want_type == GGML_Q4_K:
            b = np.frombuffer(raw, dtype=Q4K_DT, count=1)
            mine = [int(scale_min_k4(b["scales"], j)[0]) for j in range(8)]
            theirs = [int(GGUFFile._scale_min_k4(np, b["scales"], j)[0][0])
                      for j in range(8)]
            assert mine == theirs, f"scale unpack differs: {mine} != {theirs}"
            qs = b["qs"][0]
            slow = 0
            for j in range(8):
                g, s = j // 2, 0
                for k in range(32):
                    v = qs[g * 32 + k]
                    s += int(v & 0xF) if j % 2 == 0 else int(v >> 4)
                slow += mine[j] * s
            slow *= Q8_MAX
            fast = int(q4k_bounds(raw, 1)[0])
            assert slow == fast, f"Q4_K bound: slow {slow} != fast {fast}"
            print(f"  Q4_K  scales {mine} agree with gguf_reader; "
                  f"bound {fast} agrees with a per-element loop")
        else:
            b = np.frombuffer(raw, dtype=Q6K_DT, count=1)
            a = q6k_aux8(b, 1)[0]
            sc = b["scales"][0]
            slow = 0
            for j in range(16):
                s = sum(abs(int(a[j * 16 + k])) for k in range(16))
                slow += abs(int(sc[j])) * s
            slow *= Q8_MAX
            fast = int(q6k_bounds(raw, 1)[0])
            assert slow == fast, f"Q6_K bound: slow {slow} != fast {fast}"
            assert int(a.min()) >= -32 and int(a.max()) <= 31, \
                f"Q6_K centred quants out of range: [{a.min()},{a.max()}]"
            print(f"  Q6_K  centred quants in [{int(a.min())},{int(a.max())}]; "
                  f"bound {fast} agrees with a per-element loop")
    print()


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    return census(argv[1], "--selfcheck" in argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
