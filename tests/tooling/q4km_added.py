"""Run ONE added-sample leg for a band that cannot freeze from the fixed grid.

▶ WHY THIS IS NOT A GRID LEG, AND WHY THAT DISTINCTION IS LOAD-BEARING.
Ruling section 5 FIXES the grid at p00/p10/p20/p30/p45 + p41. Its tripwire
threshold (>=8 graded flips) was calibrated against THAT grid's ~384 covered
positions, so a prompt added to the grid would move a frozen constant's basis.
The same ruling's tau fail-safe, however, names this case explicitly:

    a graded position landing in a tau-less band stops grading pending a
    freeze from ADDED SAMPLES of that band

So added samples exist to FEED A FREEZE and nothing else. Concretely, and this
is the whole contract:

  - they contribute agreeing positions to the tau envelope for their band;
  - they do NOT enter the coverage total, the flip count, or the tripwire;
  - they are banked as `{pid}-added.json`, a name no grid consumer matches,
    and q4km_tau.gather_coverage walks RUN_A_PROMPTS + RUN_B_PROMPTS only, so
    the separation is structural rather than a convention to remember.

▶ IT REUSES q4km_grid.run_leg RATHER THAN REBUILDING THE INVOCATION. The
ref_forward argv, the --probe-all wiring and the guards are already validated
there; a second copy is a second thing that can drift from the leg the grid
actually ran, which would make the added samples incomparable with the
positions they are pooled against.

▶ TAU POOLS ACROSS PROMPTS BY CONSTRUCTION. tau is per PAIR, per FILE, per
BAND -- never per prompt (ruling 9.4-2, G6). The Q8_0 freeze pooled 31 probes
across prompts the same way. And the freeze stays disjoint from what it
grades: these are AGREEING positions, the graded divergence is another leg's.

CLI:
  python q4km_added.py --prompt p32 --out DIR [--dry-run]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q4km_grid as qg          # noqa: E402
import classify as cls          # noqa: E402


def build_leg(pid: str, tokens):
    return {"leg": f"{pid}-added", "prompt": pid, "kind": "probe",
            "ids": list(tokens), "n": qg.N_PREDICT,
            "positions": qg.leg_positions(len(tokens))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--additional-instance", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    def log(s):
        print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)

    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    stoke_hash = qg.check_freeze()
    log(f"stoke.exe  : {stoke_hash[:16]}... (G11 freeze verified)")

    tokens = qg.tokenize(a.prompt)
    head = qg.banked_head_ids(a.prompt)
    leg = build_leg(a.prompt, tokens)
    plen = len(tokens)

    in_band = {}
    for i in range(qg.N_PREDICT):
        in_band.setdefault(cls.band_of(plen + i), []).append(i)
    log(f"prompt     : {a.prompt}, {plen} tokens, HEAD stream {len(head)} ids")
    log(f"positions  : abs {plen}..{plen + qg.N_PREDICT - 1}")
    for b in sorted(in_band):
        log(f"  {b}: {len(in_band[b])} of the 64 generated positions")
    log(f"cost       : {leg['positions']} forward positions")

    out_json = outdir / f"{leg['leg']}.json"
    if out_json.exists():
        log(f"{out_json.name} already banked -- nothing to do")
        return 0
    if a.dry_run:
        log("dry run: leg constructed, nothing executed")
        return 0

    qg.check_no_q8_holder()
    qg.check_ram(qg.MODEL_SIZE_EXPECTED, a.additional_instance)

    dt, ids, _ = qg.run_leg(leg, outdir, log)
    log(f"{leg['leg']}: done in {dt / 60:.1f} min "
        f"({dt / leg['positions']:.2f} s/pos), {len(ids)} ids")

    j = None
    for i in range(min(len(ids), len(head))):
        if ids[i] != head[i]:
            j = i
            break
    if j is None:
        agree = list(range(len(ids)))
        log(f"{leg['leg']}: agrees with HEAD at every generated position")
    else:
        agree = list(range(j))
        log(f"{leg['leg']}: diverges at index {j} (abs {plen + j})")
    per_band = {}
    for i in agree:
        per_band[cls.band_of(plen + i)] = per_band.get(cls.band_of(plen + i), 0) + 1
    for b in sorted(per_band):
        log(f"  contributes {per_band[b]} agreeing positions to {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
