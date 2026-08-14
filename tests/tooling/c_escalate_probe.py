#!/usr/bin/env python3
"""Characterise WHY each ESCALATE escalated: a substantive set difference, or
churn at the tau cut?

Same question g4_escalate_probe.py answers for gate item [4], but for a pair of
BINARIES rather than one binary in two KV modes -- which is what M4.5 rung C is
graded as. That tool is left alone on purpose: it is the banked evidence behind
a closed gate item, and editing it would make that record unreproducible.

Section 6 discards, on each side, candidates within tau of that side's rank-5
logit and requires the survivors to be equal. That is a HARD cut, so a token at
rank5+tau+eps on one side and rank5+tau-eps on the other flips membership
without anything substantive differing. This prints both sides' top-5 with each
candidate's distance from its own cut, so the two cases are distinguishable by
eye instead of assumed.

THE TEST THAT MATTERS is not the symmetric difference's size but whether the
tokens in it are ranked by BOTH sides. Section 6 defines its BUG trigger as a
candidate the other implementation never ranked; a token both sides rank, that
merely lands on opposite sides of a hard cut, is the instrument's own boundary
churn.

Usage:
  python tests/tooling/c_escalate_probe.py --stoke-a A.exe --stoke-b B.exe \\
      --verdicts <...>-verdict.jsonl --out <...>-escalate-characterisation.json
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
PDIR = ROOT / "tests" / "data" / "parity"
PROBE_RE = re.compile(r"probe pos=(\d+): id=(\d+) logit=(-?[\d.]+)")


def probe(stoke, model, prompt, gp, kv, threads):
    p = subprocess.run([stoke, "run", str(model), "--raw", "-n", "64",
                        "-f", str(prompt), "--kv", kv,
                        "--threads", str(threads), "--probe-pos", str(gp)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    se = p.stderr.decode("utf-8", "replace")
    top = [(int(m.group(2)), float(m.group(3))) for m in PROBE_RE.finditer(se)]
    if not top:
        raise SystemExit(f"no probe output for {prompt} gp={gp} from {stoke}")
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stoke-a", required=True)
    ap.add_argument("--stoke-b", required=True)
    ap.add_argument("--kv-a", default="f32")
    ap.add_argument("--kv-b", default="f32")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    verdicts = [json.loads(l) for l in
                pathlib.Path(args.verdicts).open(encoding="utf-8")]
    esc = [v for v in verdicts if v["verdict"] == "ESCALATE"]
    print(f"{len(esc)} ESCALATE positions\n")

    out = []
    for v in esc:
        tau = v["tau"]
        path = PDIR / f"{v['prompt']}.txt"
        a = probe(args.stoke_a, args.model, path, v["gp"], args.kv_a, args.threads)
        b = probe(args.stoke_b, args.model, path, v["gp"], args.kv_b, args.threads)
        ar5, br5 = min(x[1] for x in a), min(x[1] for x in b)
        acut, bcut = ar5 + tau, br5 + tau
        seta = {i for i, val in a if val > acut}
        setb = {i for i, val in b if val > bcut}
        symdiff = sorted(seta ^ setb)
        margins = ([abs(val - acut) for _, val in a] +
                   [abs(val - bcut) for _, val in b])
        # The BUG language: is every disputed token ranked by BOTH sides?
        ranked_a, ranked_b = {i for i, _ in a}, {i for i, _ in b}
        unranked = [t for t in symdiff if t not in ranked_a or t not in ranked_b]
        g = None if v["g_x"] is None else max(v["g_x"], v["g_y"])
        rec = {"prompt": v["prompt"], "gp": v["gp"], "tau": tau,
               "x": v["x"], "y": v["y"], "g_max": g,
               "margin_breach": (g is not None and g >= tau),
               "a_top5": a, "b_top5": b,
               "set_a": sorted(seta), "set_b": sorted(setb),
               "symmetric_difference": symdiff,
               "symdiff_tokens_not_ranked_by_both": unranked,
               "closest_candidate_to_a_cut": round(min(margins), 6)}
        out.append(rec)
        why = []
        if rec["margin_breach"]:
            why.append(f"MARGIN BREACH g={g:.4f} >= tau={tau:.4f}")
        if seta != setb:
            why.append("set difference")
        print(f"=== {v['prompt']} gp={v['gp']} tau={tau:.4f} "
              f"(chose {v['x']} vs {v['y']}) -- {'; '.join(why)} ===")
        print(f"  A top5: {[(i, round(val,4)) for i,val in a]}")
        print(f"  B top5: {[(i, round(val,4)) for i,val in b]}")
        print(f"  survivors A={sorted(seta)}  B={sorted(setb)}  symdiff={symdiff}")
        print(f"  symdiff tokens NOT ranked by both: {unranked or 'none'}")
        print(f"  nearest candidate sits {min(margins):.4f} from its own cut\n")

    pathlib.Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    n_breach = sum(1 for r in out if r["margin_breach"])
    n_unranked = sum(1 for r in out if r["symdiff_tokens_not_ranked_by_both"])
    print(f"summary: {n_breach}/{len(out)} margin breaches; "
          f"{n_unranked}/{len(out)} with a disputed token the other side "
          f"never ranked")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
