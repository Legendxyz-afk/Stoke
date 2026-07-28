#!/usr/bin/env python3
"""tests/tooling/l2_diff.py -- the L2 rung of the M2 parity-debug ladder,
as a reproducible step instead of an ad-hoc one.

L2 compares the post-block hidden state after each of the 36 layers
between stoke's C forward pass (`stoke run ... --dump DIR`) and the numpy
f32 reference (`ref_forward.py ... --dump-layers DIR`), for the LAST
prompt position. Both write L00.f32 .. L35.f32, raw little-endian float32,
n_embd (2560) values each.

Kabir's 2026-07-28 option-3 decision makes M2's gate "C-vs-ref on a
stratified subset + the L2 ladder", so the ladder is now gate evidence and
needs to be re-runnable on demand and mechanically judged. W3 established
the reference result: 36/36 layers max-abs-diff < 1e-3, worst L34 6.1e-4.

WHY THE CURVE MATTERS AS MUCH AS THE MAX. Two very different things both
show up as "a nonzero diff": float32 non-associativity, which accumulates
SMOOTHLY as depth grows because each layer's small rounding differences
compound; and a structural bug in one op, which appears as a STEP -- a
layer whose diff jumps far above the trend and stays there. A pure
max-abs-diff threshold cannot tell them apart, so this tool reports the
per-layer curve and flags any layer whose diff jumps more than JUMP_FACTOR
over the running trend even when the absolute threshold still passes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

N_LAYERS = 36
THRESHOLD = 1e-3      # M2 note section 9's L2 bar. NEVER loosen without escalation.
JUMP_FACTOR = 10.0    # a layer diff this many x the running max = a STEP, not drift


def load_dir(d: Path) -> list:
    out = []
    for i in range(N_LAYERS):
        f = d / f"L{i:02d}.f32"
        if not f.exists():
            sys.exit(f"fatal: {f} missing -- expected L00.f32..L{N_LAYERS-1:02d}.f32")
        a = np.fromfile(f, dtype=np.float32)
        if a.size == 0:
            sys.exit(f"fatal: {f} is empty")
        out.append(a)
    sizes = {a.size for a in out}
    if len(sizes) != 1:
        sys.exit(f"fatal: {d} has inconsistent layer widths: {sorted(sizes)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", help="one dump dir (e.g. stoke's --dump)")
    ap.add_argument("dir_b", help="the other (e.g. ref's --dump-layers)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--quiet", action="store_true", help="only the verdict")
    a = ap.parse_args()

    A, B = load_dir(Path(a.dir_a)), load_dir(Path(a.dir_b))
    if A[0].size != B[0].size:
        sys.exit(f"fatal: width mismatch {A[0].size} vs {B[0].size}")

    rows, running_max, jumps = [], 0.0, []
    for i, (x, y) in enumerate(zip(A, B)):
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        mx, mean = float(d.max()), float(d.mean())
        # A step is judged against the trend BEFORE this layer, so layer 0
        # (no history) can never be flagged.
        if i > 0 and running_max > 0 and mx > JUMP_FACTOR * running_max:
            jumps.append((i, mx, running_max))
        running_max = max(running_max, mx)
        rows.append((i, mx, mean))

    if not a.quiet:
        print(f"{'layer':>5} {'max_abs_diff':>14} {'mean_abs_diff':>14}")
        for i, mx, mean in rows:
            mark = "  <-- STEP" if any(j[0] == i for j in jumps) else ""
            print(f"L{i:02d}   {mx:>14.3e} {mean:>14.3e}{mark}")
        print()

    worst_i, worst = max(((i, mx) for i, mx, _ in rows), key=lambda t: t[1])
    over = [(i, mx) for i, mx, _ in rows if mx >= a.threshold]

    print(f"layers compared : {len(rows)} x {A[0].size} values")
    print(f"worst layer     : L{worst_i:02d} at {worst:.3e}")
    print(f"threshold       : {a.threshold:g}")
    if over:
        print(f"OVER THRESHOLD  : {len(over)} layer(s) -- "
              f"{', '.join(f'L{i:02d}={m:.3e}' for i, m in over[:6])}")
    if jumps:
        print(f"STEP DETECTED   : {len(jumps)} layer(s) jumped >{JUMP_FACTOR:g}x "
              f"the running trend -- {', '.join(f'L{i:02d} ({m:.3e} vs {r:.3e})' for i, m, r in jumps)}")
        print("                  A step is the signature of a STRUCTURAL bug in one "
              "op, not float32 drift. Bisect that layer before reading anything else.")

    ok = not over and not jumps
    print()
    print("L2: PASS -- smooth accumulation, all layers under threshold" if ok
          else "L2: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
