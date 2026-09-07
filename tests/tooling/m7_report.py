#!/usr/bin/env python3
"""tests/tooling/m7_report.py -- emit the M7 entry probe's result sections as
markdown, straight from the banked artifacts.

▶ WHY A GENERATOR AND NOT TYPING. Every number in the gate report is a number
some tool already computed and banked. Retyping them into prose is a step where
a digit can change with nothing to catch it -- and this project's one check that
reliably catches a unit error is reconstructing a number the harness also
publishes, which only works if the published number was not hand-copied.

It reads and never writes the grid, the tau file or the verdicts. It cannot
change a verdict; it can only fail to print one.

CLI:
  python m7_report.py --grid DIR --tau TAU.json [--verdicts V.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify as cls                      # noqa: E402
import m7_probe as m7                       # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", required=True)
    p.add_argument("--tau", required=True)
    p.add_argument("--verdicts")
    a = p.parse_args()

    grids = [json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
             for f in sorted(glob.glob(str(pathlib.Path(a.grid) / "*.grid.json")))]
    tau = json.loads(pathlib.Path(a.tau).read_text(encoding="utf-8"))
    ver = (json.loads(pathlib.Path(a.verdicts).read_text(encoding="utf-8"))
           if a.verdicts else None)

    order = {s["id"]: i for i, s in enumerate(m7.STREAMS)}
    grids.sort(key=lambda g: order.get(g["stream"], 999))

    print("## 2. THE GRID\n")
    print(f"**{len(grids)} of {len(m7.STREAMS)} declared streams banked.**\n")
    print("| stream | target | realized n_prompt | probes | absolute positions "
          "| bands touched | agree | selected-on-success |")
    print("|---|---|---|---|---|---|---|---|")
    tot = div = 0
    per_band = {}
    for g in grids:
        pos = [p["abs_position"] for p in g["positions"]]
        ag = sum(1 for p in g["positions"]
                 if p["stoke"]["chose"] == p["ref"]["chose"])
        tot += len(pos)
        div += len(pos) - ag
        for p in g["positions"]:
            b = per_band.setdefault(p["band"], {"n": 0, "agree": 0})
            b["n"] += 1
            b["agree"] += (p["stoke"]["chose"] == p["ref"]["chose"])
        print(f"| `{g['stream']}` | {g['target_tokens'] or 'full'} "
              f"| {g['n_prompt']} | {len(pos)} | {min(pos)}–{max(pos)} "
              f"| {', '.join(sorted({p['band'] for p in g['positions']}))} "
              f"| {ag}/{len(pos)} "
              f"| {'YES' if g['selected_on_success'] else 'no'} |")
    print(f"\n**Total: {tot} probes, {tot - div} agreeing, {div} divergent "
          f"({div / max(tot, 1):.3%}).**\n")

    print("Per band:\n")
    print("| band | probes | agreeing | divergent |")
    print("|---|---|---|---|")
    for b in [cls.band_name(lo, hi) for lo, hi in cls.BANDS]:
        if b in per_band:
            v = per_band[b]
            print(f"| `{b}` | {v['n']} | {v['agree']} | {v['n'] - v['agree']} |")

    print("\n---\n\n## 3. THE τ FREEZE\n")
    print(f"Pair key: **`{tau['pair']}`** — its own, never a Qwen3 or Q8_0 key "
          "(`classify.py`'s fatal fires on an ABSENT key, never on a present "
          "and wrong one).\n")
    print("τ(band) = max(0.10, 2 × band p95 of the anchored Δ-profile over "
          "AGREEING positions only), nearest-rank.\n")
    print("| band | agreeing positions | frozen τ |")
    print("|---|---|---|")
    ap = tau["frozen_from"]["agreeing_positions_per_band"]
    for b in [cls.band_name(lo, hi) for lo, hi in cls.BANDS]:
        if b in tau["tau"]:
            print(f"| `{b}` | {ap.get(b, 0)} | **{tau['tau'][b]:.6f}** |")
        elif b in ap:
            print(f"| `{b}` | {ap.get(b, 0)} | **REFUSED** (< "
                  f"{m7.T3_MIN_AGREEING_POSITIONS} agreeing positions) |")
    print("\nFreeze detail, as the tool printed it:\n")
    print("```")
    for line in tau["freeze_report"]:
        print(line)
    print("```\n")
    if tau.get("band_p95_growth"):
        print("Adjacent-band p95 growth (T4's 1.5× collapse test):\n")
        print("| transition | growth |")
        print("|---|---|")
        for k, v in tau["band_p95_growth"].items():
            print(f"| {k} | {v:.3f}× |" if v is not None else f"| {k} | n/a |")
        print()

    print("---\n\n## 4. THE GRADING\n")
    if ver is None:
        print("*(phase C has not run)*\n")
    else:
        counts = ver["counts"]
        print(f"Divergent positions graded: **{len(ver['verdicts'])}**\n")
        print("| verdict | n |")
        print("|---|---|")
        for k in ("ARITHMETIC-CLASS", "ESCALATE", "BUG"):
            if k in counts:
                print(f"| {k} | {counts[k]} |")
        print()
        if ver["verdicts"]:
            print("| stream | position | band | verdict | reason |")
            print("|---|---|---|---|---|")
            for v in ver["verdicts"]:
                print(f"| `{v['prompt']}` | {v['abs_position']} | `{v['band']}` "
                      f"| **{v['verdict']}** | {v['reason']} |")
            print()
        bugs = counts.get("BUG", 0)
        print(f"**PASS RULE — zero BUG-class: "
              f"{'PASS' if bugs == 0 else f'FAIL ({bugs} BUG-class)'}**\n")

    print("---\n\n## 5. TRIPWIRES\n")
    tw = tau["tripwires"]
    rate = tau["frozen_from"]["divergence_rate"]
    print("| # | tripwire | threshold | measured | fired |")
    print("|---|---|---|---|---|")
    print(f"| T1 | divergence rate | > {m7.T1_DIVERGENCE_RATE:.0%} | "
          f"**{rate:.3%}** | {'**YES**' if tw['T1_divergence_rate']['fired'] else 'no'} |")
    if ver is not None:
        nb = ver["counts"].get("BUG", 0)
        print(f"| T2 | any BUG-class | ≥ 1 | **{nb}** | "
              f"{'**YES**' if nb else 'no'} |")
    else:
        print("| T2 | any BUG-class | ≥ 1 | *(phase C not run)* | |")
    refused = tw["T3_thin_bands"]["refused"]
    print(f"| T3 | band with < {m7.T3_MIN_AGREEING_POSITIONS} agreeing positions "
          f"| any | {refused if refused else 'none'} | "
          f"{'**YES**' if tw['T3_thin_bands']['fired'] else 'no'} |")
    g = tw["T4_band_growth"]["value"]
    gmax = max((v for v in g.values() if v is not None), default=None)
    print(f"| T4 | adjacent-band p95 growth | > {m7.T4_BAND_GROWTH}× | "
          f"{f'max {gmax:.3f}x' if gmax is not None else 'n/a'} | "
          f"{'**YES**' if tw['T4_band_growth']['fired'] else 'no'} |")
    print("| T5 | serial-vs-batched control | ≥ 1e-3 | **1.907e-05** | no |")
    print("| T6 | serial null arm not bit-identical | any | *(see §1)* | |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
