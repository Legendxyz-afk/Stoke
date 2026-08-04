#!/usr/bin/env python3
"""k4_dequant_check.py -- prove our Q4_K/Q6_K dequant against llama.cpp's own.

WHY THIS SHAPE OF CHECK. The playbook names the 6-bit scale/min unpack
(`get_scale_min_k4`) as "the classic transcription bug", and K2 already
recorded what the strongest available check looks like:

    the strongest check on a transcription is a FIXED POINT of the original
    system's own output -- it requires no agreement about intent.

For Q8_0 that was dequant->requant reproducing the original bytes. K-quants
have no such round trip (the quantizer is lossy and iterative), so the fixed
point is taken one step out instead: llama.cpp is asked to dequantize the
REAL model file itself,

    llama-quantize --allow-requantize <Q4_K_M>.gguf <out>.gguf F32

and our python must reproduce those f32 values BIT FOR BIT. Nothing here
rests on reading the C correctly -- it rests on the same binary that ships
the engine we are chasing having already computed the answer.

WHY BIT-EXACT AND NOT A TOLERANCE. Dequantization is a fixed sequence of f32
operations with no reduction in it, so two correct implementations must agree
exactly. A tolerance would accept a wrong operand grouping (f32 multiply is
not associative) and, worse, would accept a scale/min unpack that is right
for most blocks -- which is exactly how the classic bug presents. Comparison
is on the raw f32 BIT PATTERNS, so -0.0 vs 0.0 and any NaN payload are
differences too.

Usage:
    python k4_dequant_check.py --q4km models/...Q4_K_M.gguf \\
        --f32ref build/k4ref/...-f32.gguf [--layers 0,17,35] [--rows 8]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import (  # noqa: E402
    GGUFFile, GgufError, GGML_NAME, GGML_Q4_K, GGML_Q6_K,
)

# The playbook's mandated layer sample. 0 and 35 are the boundary layers
# (first/last), 17 is interior -- a layout bug that happened to be harmless at
# a block boundary still has to survive the middle of the tensor.
DEFAULT_LAYERS = (0, 17, 35)

KQUANT = (GGML_Q4_K, GGML_Q6_K)


def bit_diff(np, a, b):
    """(n_differing, max_abs_diff) comparing raw f32 bit patterns.

    Bits, not values: a tolerance is what lets a wrong operand grouping and a
    mostly-right scale unpack through, and those are the two failure modes
    this file exists to catch.
    """
    ab = a.astype(np.float32).view(np.uint32)
    bb = b.astype(np.float32).view(np.uint32)
    ne = int((ab != bb).sum())
    if ne == 0:
        return 0, 0.0
    return ne, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def select_tensors(gf, layers):
    """Every K-quant tensor in the sampled layers, plus token_embd."""
    want = []
    for name, t in gf.tensors.items():
        if t.type not in KQUANT:
            continue
        if name.startswith("blk."):
            try:
                n = int(name.split(".")[1])
            except (IndexError, ValueError):
                continue
            if n in layers:
                want.append(name)
        elif name.startswith("token_embd"):
            want.append(name)
    return sorted(want)


def emit_golden(np, src, ref, names, path, meta):
    """Freeze a few real blocks + llama.cpp's own f32 for them.

    The 15 GB F32 reference is a build artifact, not something to keep: it is
    reproducible in ~4 minutes from the committed recipe. But deleting it
    would leave these formulas with no permanent net, so a thin slice of it is
    kept instead -- REAL block bytes paired with the ORIGINAL SYSTEM'S OWN
    output for them. Values are stored as f32 hex so the golden is exact and
    still diffable by eye.
    """
    recs = []
    for name in names:
        t = src.get_tensor(name)
        r = ref.get_tensor(name)
        blk_elems = 256
        row = 0
        got = src.tensor_row_dequant(t, row)
        want = ref.tensor_row_dequant(r, row)
        if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
            fatalish = f"{name}: refusing to freeze a golden from a row that does not match"
            raise GgufError(fatalish)
        stride = src.row_stride_bytes(t)
        raw = src.row_raw_bytes(t, row, stride)
        blk_bytes = stride // (t.dims[0] // blk_elems)
        recs.append((GGML_NAME[t.type], name, row,
                     raw[:blk_bytes], want[:blk_elems]))
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("# k4 dequant golden -- first block of row 0, and the f32 "
                 "llama.cpp itself produced for it\n")
        fh.write("# NOT hand-computed: every value here came out of "
                 "llama-quantize F32 on the real file.\n")
        for k, v in meta.items():
            fh.write(f"# {k}: {v}\n")
        fh.write("# record: <type> <tensor> <row> <raw-hex> then one line of "
                 "f32 values as hex bit patterns\n")
        for tname, name, row, raw, vals in recs:
            fh.write(f"{tname} {name} {row} {raw.hex()}\n")
            fh.write(" ".join(f"{x:08x}"
                              for x in vals.astype(np.float32).view(np.uint32))
                     + "\n")
    return len(recs)


def check_golden(np, src, path):
    """Verify the live dequant against the frozen slice. Needs only the
    Q4_K_M file -- no 15 GB reference, so this runs in CI-time forever."""
    lines = [l.rstrip("\n") for l in open(path, encoding="ascii")
             if not l.startswith("#") and l.strip()]
    if len(lines) % 2 or not lines:
        print(f"golden {path}: malformed ({len(lines)} payload lines)")
        return 1
    bad = 0
    for i in range(0, len(lines), 2):
        tname, name, row, rawhex = lines[i].split()
        want = np.array([int(x, 16) for x in lines[i + 1].split()],
                        dtype=np.uint32).view(np.float32)
        t = src.get_tensor(name)
        if GGML_NAME.get(t.type) != tname:
            print(f"  FAIL {name}: type is {GGML_NAME.get(t.type)}, golden says {tname}")
            bad += 1
            continue
        raw = src.row_raw_bytes(t, int(row), len(rawhex) // 2)
        if raw.hex() != rawhex:
            print(f"  FAIL {name}: raw block bytes moved -- the MODEL changed, "
                  f"not the code")
            bad += 1
            continue
        got = src.tensor_row_dequant(t, int(row))[:want.shape[0]]
        nb, mx = bit_diff(np, got, want)
        if nb:
            print(f"  FAIL {name} ({tname}): {nb}/{want.shape[0]} differ, "
                  f"max |d| {mx:.6g}")
            bad += 1
        else:
            print(f"  ok   {name} ({tname}): {want.shape[0]} values bit-exact")
    print(f"\ngolden: {len(lines) // 2} records, {bad} failing")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q4km", required=True, help="the real Q4_K_M GGUF")
    ap.add_argument("--f32ref",
                    help="llama-quantize F32 dequant of the SAME file")
    ap.add_argument("--emit-golden", metavar="PATH",
                    help="freeze a slice of the verified comparison (needs --f32ref)")
    ap.add_argument("--check-golden", metavar="PATH",
                    help="verify against a frozen slice; needs no F32 reference")
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--rows", type=int, default=8,
                    help="rows sampled per tensor (default 8)")
    a = ap.parse_args()

    layers = {int(x) for x in a.layers.split(",") if x.strip() != ""}
    if not os.path.exists(a.q4km):
        ap.error(f"not found: {a.q4km}")
    import numpy as np
    src = GGUFFile(a.q4km)

    if a.check_golden:
        if not os.path.exists(a.check_golden):
            ap.error(f"not found: {a.check_golden}")
        return check_golden(np, src, a.check_golden)

    if not a.f32ref:
        ap.error("--f32ref is required unless --check-golden is used")
    if not os.path.exists(a.f32ref):
        ap.error(f"not found: {a.f32ref}")
    ref = GGUFFile(a.f32ref)

    names = select_tensors(src, layers)
    if not names:
        print("NO K-QUANT TENSORS SELECTED -- nothing was proven.")
        return 2

    print(f"layers {sorted(layers)}, {a.rows} rows/tensor, "
          f"{len(names)} tensors\n")
    tot_vals = tot_bad = 0
    per_type = {}
    failures = []

    for name in names:
        t = src.get_tensor(name)
        r = ref.get_tensor(name)
        tname = GGML_NAME.get(t.type, str(t.type))
        if r.type not in (0,):  # GGML_F32
            failures.append(f"{name}: reference is {GGML_NAME.get(r.type)}, not F32")
            continue
        if list(r.dims) != list(t.dims):
            failures.append(f"{name}: dims {list(t.dims)} vs ref {list(r.dims)}")
            continue
        n_out = t.dims[1] if t.n_dims == 2 else 1
        # Rows spread across the tensor, always including the first and last:
        # a stride bug shows at the ends, a scale-unpack bug shows anywhere.
        if a.rows >= n_out:
            rows = list(range(n_out))
        else:
            rows = sorted({0, n_out - 1} | {
                (i * n_out) // a.rows for i in range(a.rows)})
        bad = vals = 0
        worst = 0.0
        for row in rows:
            got = src.tensor_row_dequant(t, row)
            want = ref.tensor_row_dequant(r, row)
            nb, mx = bit_diff(np, got, want)
            bad += nb
            worst = max(worst, mx)
            vals += got.shape[0]
        st = per_type.setdefault(tname, [0, 0, 0])
        st[0] += vals
        st[1] += bad
        st[2] += 1
        tot_vals += vals
        tot_bad += bad
        mark = "ok  " if bad == 0 else "FAIL"
        extra = "" if bad == 0 else f"  {bad} differ, max |d| {worst:.6g}"
        print(f"  {mark} {name:<34} {tname:<5} {len(rows):>3} rows "
              f"{vals:>7} values{extra}")
        if bad:
            failures.append(f"{name}: {bad}/{vals} values differ")

    print()
    for tname, (v, b, nt) in sorted(per_type.items()):
        print(f"  {tname}: {nt} tensors, {v} values, {b} differing")
    ok = not failures and tot_bad == 0 and tot_vals > 0
    print(f"\n{tot_vals} values compared, {tot_bad} differing")
    if ok:
        print("K4 DEQUANT CHECK: BIT-EXACT vs llama.cpp's own dequant -- PASS")
        if a.emit_golden:
            # Only ever frozen from a PASSING comparison -- a golden taken
            # from a failing one would pin the bug as the expectation.
            n = emit_golden(np, src, ref, names, a.emit_golden, {
                "source": os.path.basename(a.q4km),
                "reference": os.path.basename(a.f32ref),
                "recipe": "llama-quantize --allow-requantize <src> <ref> F32 8",
                "verified": f"{tot_vals} values bit-exact over layers "
                            f"{sorted(layers)}",
            })
            print(f"golden written: {a.emit_golden} ({n} records)")
        return 0
    print("K4 DEQUANT CHECK: FAIL")
    for f in failures[:20]:
        print(f"  - {f}")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GgufError as e:
        print(f"gguf error: {e}")
        sys.exit(2)
