#!/usr/bin/env python3
"""tests/tooling/run_e2e_interleaved.py -- clause 2's end-to-end ratio, measured
the way this box actually permits.

▶ WHY THIS EXISTS, AND IT IS A CORRECTION. The first attempt (2026-09-08,
`bench/results/2026-09-08-e2e-block`) ran five sequential cells of three runs
each and its pre-registered bracket FIRED: the same `llamacpp-tuned x
s2-closed-8` cell read 67.6 s at t+0 and 81.3 s at t+32, a **18.4% drift**
against a 5% threshold. Every cross-cell ratio from that block is therefore
VOID -- not wrong, unresolvable, because a 20% referent move swamps the ~3%
effect under test.

[[measurement-needs-interleaving]] already said this: *one unit per PROCESS,
counterbalance the order, detect drift from the arms' own repeats.* A
block-sequential matrix violates all three -- it couples three runs to one
thermal state and separates the arms by half an hour.

▶ WHAT THIS DOES INSTEAD.
  - ONE measured run per unit, one adapter lifetime per unit (fresh server).
  - Arms ALTERNATE (A B A B ...), so drift is shared rather than assigned.
  - A cool-down between units, because the drift above is thermal in shape --
    a step at the end of a 35-minute sustained load, not a gradual slope.
  - Drift is read from EACH ARM'S OWN repeats across the sitting, which is the
    only control that works when both arms move together.
  - Every unit is banked as it completes: a run that dies at unit 7 keeps 6.

▶ IT ADDS NO MEASUREMENT LOGIC. Every number comes from bench.run_scenario via
run_matrix.run_cell -- the same code the referents are measured by, which is the
entire point of the exercise it serves.

Usage:
  python tests/tooling/run_e2e_interleaved.py --scenario s2-closed-8 \
      --pairs 3 --cooldown 45 --out bench/results/<block>/interleaved-8.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "bench"))

import run_matrix  # noqa: E402
import scenarios as scen_mod  # noqa: E402


def unit(config_label, scenario, timeout_s):
    """One adapter lifetime, one discarded warmup, ONE measured run."""
    cell = run_matrix.run_cell(config_label, scenario, runs=1, timeout_s=timeout_s)
    if cell["error"]:
        return {"config": config_label, "error": cell["error"]}
    r = cell["runs"][0]
    pcs = r["per_client"]
    if r.get("errors"):
        return {"config": config_label, "error": str(r["errors"])}
    # ⚠ The verdict comes from reconstructing a number the harness also
    # publishes -- the one check that catches a unit error. A unit that cannot
    # reconstruct is refused, not banked.
    num = sum(p["n_tokens"] - 1 for p in pcs)
    span = max(p["last_token_abs"] for p in pcs) - min(p["first_token_abs"] for p in pcs)
    if abs(num / span - r["aggregate_decode_tps"]) > 1e-9:
        return {"config": config_label,
                "error": f"reconstruction failed: {num/span} vs {r['aggregate_decode_tps']}"}
    if {p["n_tokens"] for p in pcs} != {scenario.max_tokens}:
        return {"config": config_label,
                "error": f"token counts {sorted({p['n_tokens'] for p in pcs})} "
                         f"!= {scenario.max_tokens}"}
    ttfts = sorted(p["ttft_ms"] / 1000.0 for p in pcs)
    return {
        "config": config_label,
        "scenario": scenario.name,
        "wall_s": r["wall_s"],
        "aggregate_decode_tps": r["aggregate_decode_tps"],
        "ttft50_s": ttfts[len(ttfts) // 2],
        "ttft_min_s": ttfts[0],
        "completed": r["completed"],
        "adapter_log_head": (cell.get("adapter_log_tail") or "").splitlines()[:2],
        "error": None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--arm-a", default="llamacpp-tuned")
    p.add_argument("--arm-b", default="stoke-tuned")
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=45.0)
    p.add_argument("--timeout-s", type=int, default=900)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    scenario = run_matrix.SCENARIOS_BY_NAME[a.scenario]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")

    banked = []
    t_start = time.time()
    for k in range(a.pairs):
        # ABBA within a pair: the order flips every pair, so neither arm
        # systematically occupies the hotter half of the sitting.
        order = (a.arm_a, a.arm_b) if k % 2 == 0 else (a.arm_b, a.arm_a)
        for label in order:
            if a.cooldown > 0 and banked:
                time.sleep(a.cooldown)
            rec = unit(label, scenario, a.timeout_s)
            rec["pair"] = k
            rec["t_min"] = round((time.time() - t_start) / 60.0, 2)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            banked.append(rec)
            if rec.get("error"):
                print(f"  [{rec['t_min']:5.1f}m] {label:16s} ERROR: {rec['error']}",
                      flush=True)
            else:
                print(f"  [{rec['t_min']:5.1f}m] {label:16s} wall={rec['wall_s']:7.1f}s "
                      f"agg={rec['aggregate_decode_tps']:6.3f} "
                      f"ttft50={rec['ttft50_s']:5.1f}s", flush=True)
    fh.close()

    ok = [r for r in banked if not r.get("error")]
    print("\n== per arm ==")
    walls = {}
    for label in (a.arm_a, a.arm_b):
        w = [r["wall_s"] for r in ok if r["config"] == label]
        if not w:
            print(f"  {label}: NO UNITS"); continue
        walls[label] = w
        drift = 100 * (max(w) - min(w)) / st.median(w)
        print(f"  {label:16s} n={len(w)} wall median={st.median(w):7.1f}s "
              f"own spread={drift:5.1f}%  units={[round(x,1) for x in w]}")
    if len(walls) == 2 and all(walls.values()):
        ma, mb = st.median(walls[a.arm_a]), st.median(walls[a.arm_b])
        print(f"\n  RATIO {a.arm_a} / {a.arm_b} = {ma/mb:.4f}x "
              f"(>1 means {a.arm_b} finishes the work first)")
        worst = max(100 * (max(w) - min(w)) / st.median(w) for w in walls.values())
        print(f"  ⚠ largest single-arm spread this sitting: {worst:.1f}% -- a ratio "
              f"is only readable if it exceeds this")
    return 0


if __name__ == "__main__":
    sys.exit(main())
