#!/usr/bin/env python3
"""tests/tooling/step_trace_phases.py -- decompose a serving run's wall from
the STOKE_STEP_TRACE=1 lines (src/serve.c, 2026-09-15) into its phases.

    python tests/tooling/step_trace_phases.py <stderr-file> [<stderr-file> ...]

Per file: wall (first step to last), forward vs logits time, the prefill phase
(steps carrying a chunk: prefill rows, riding decode rows, ms per prefill row),
the decode-only tail (steps, rows, ms per row), and the per-step cost by
decode width N in the tail. Every number is read from the trace lines; nothing
is modelled. A ratio between two files is only readable if the box state was
the same (2026-09-15: the SLOW state moved both arms ~35%)."""
import re
import statistics as st
import sys
from collections import defaultdict

PAT = re.compile(r"step trace: k=(\d+) t=([\d.]+) rows=(\d+) decode=(\d+) prefill=(\d+) "
                 r"yield=(\d+) fwd=([\d.]+) logits=([\d.]+)")


def parse(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PAT.search(line)
            if m:
                rows.append(dict(k=int(m[1]), t=float(m[2]), rows=int(m[3]), dec=int(m[4]),
                                 pre=int(m[5]), yld=int(m[6]), fwd=float(m[7]), lg=float(m[8])))
    return rows


def report(path):
    rows = parse(path)
    if not rows:
        print(f"{path}: no step trace lines"); return
    wall = rows[-1]["t"] + rows[-1]["fwd"] + rows[-1]["lg"]
    fwd = sum(r["fwd"] for r in rows); lg = sum(r["lg"] for r in rows)
    pre = [r for r in rows if r["pre"] > 0]; dec = [r for r in rows if r["pre"] == 0]
    print(f"== {path}")
    print(f"  steps {len(rows)}  wall {wall/1000:.1f} s = forward {fwd/1000:.1f} + logits {lg/1000:.1f} "
          f"+ other {(wall-fwd-lg)/1000:.1f}")
    if pre:
        p_rows = sum(r["pre"] for r in pre); d_ride = sum(r["dec"] for r in pre)
        p_end = pre[-1]["t"] + pre[-1]["fwd"] + pre[-1]["lg"]
        pure = [r["fwd"] / r["pre"] for r in pre if r["dec"] == 0 and r["pre"] >= 64]
        print(f"  prefill phase: {len(pre)} steps to t={p_end/1000:.1f} s; {p_rows} prefill rows "
              f"+ {d_ride} riding decode rows; {sum(r['fwd'] for r in pre)/1000:.1f} s forward"
              + (f"; pure chunk cost {st.median(pure):.2f} ms/row ({1000/st.median(pure):.1f} tok/s)" if pure else ""))
    if dec:
        d_rows = sum(r["dec"] for r in dec)
        tot = sum(r["fwd"] + r["lg"] for r in dec)
        print(f"  decode-only tail: {len(dec)} steps from t={dec[0]['t']/1000:.1f} s; {d_rows} rows; "
              f"{tot/1000:.1f} s; {tot/max(d_rows,1):.1f} ms/row")
        by = defaultdict(list)
        for r in dec:
            by[r["dec"]].append(r["fwd"] + r["lg"])
        cells = []
        for n in sorted(by):
            m = st.median(by[n])
            cells.append(f"N={n}:{m:.0f}ms/{m/n:.1f}per-row(n={len(by[n])})")
        print("  tail steps by N: " + "  ".join(cells))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
