"""Characterise WHY each ESCALATE escalated: substantive set difference, or
churn at the tau cut?

Section 6 discards, on each side, candidates within tau of that side's rank-5
logit and requires the survivors to be equal. That is a HARD cut, so a token
sitting at rank5+tau+eps on one side and rank5+tau-eps on the other flips
membership without anything substantive differing. This prints both sides'
top-5 with each candidate's distance above its own cut, so the two cases are
distinguishable by eye instead of assumed.
"""
import json
import pathlib
import re
import subprocess
import sys

STOKE = r"M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe"
MODEL = r"M:\New Drive\Desktop\Programming\stoke\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
PDIR = pathlib.Path(r"M:\New Drive\Desktop\Programming\stoke\tests\data\parity")
GR = pathlib.Path(r"M:\New Drive\Desktop\Programming\stoke\docs\superpowers\gate-reports")
PROBE_RE = re.compile(r"probe pos=(\d+): id=(\d+) logit=(-?[\d.]+)")


def probe(prompt, gp, kv):
    p = subprocess.run([STOKE, "run", MODEL, "--raw", "-n", "64", "-f", prompt,
                        "--kv", kv, "--threads", "6", "--probe-pos", str(gp)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    se = p.stderr.decode("utf-8", "replace")
    return [(int(m.group(2)), float(m.group(3))) for m in PROBE_RE.finditer(se)]


def main():
    verdicts = [json.loads(l) for l in
                (GR / "2026-08-12-m4-g4-verdict.jsonl").open(encoding="utf-8")]
    esc = [v for v in verdicts if v["verdict"] == "ESCALATE"]
    print(f"{len(esc)} ESCALATE positions\n")
    out = []
    for v in esc:
        tau = v["tau"]
        path = str(PDIR / f"{v['prompt']}.txt")
        a = probe(path, v["gp"], "f32")
        b = probe(path, v["gp"], "q8")
        ar5, br5 = min(x[1] for x in a), min(x[1] for x in b)
        acut, bcut = ar5 + tau, br5 + tau
        seta = {i for i, val in a if val > acut}
        setb = {i for i, val in b if val > bcut}
        # how close is the nearest candidate to its own cut, on either side?
        margins = ([abs(val - acut) for _, val in a] +
                   [abs(val - bcut) for _, val in b])
        rec = {"prompt": v["prompt"], "gp": v["gp"], "tau": tau,
               "x": v["x"], "y": v["y"],
               "f32_top5": a, "q8_top5": b,
               "set_f32": sorted(seta), "set_q8": sorted(setb),
               "symmetric_difference": sorted(seta ^ setb),
               "closest_candidate_to_a_cut": round(min(margins), 6)}
        out.append(rec)
        print(f"=== {v['prompt']} gp={v['gp']} tau={tau:.4f} "
              f"(chose {v['x']} vs {v['y']}) ===")
        print(f"  f32 top5: {[(i, round(val,4)) for i,val in a]}")
        print(f"  q8  top5: {[(i, round(val,4)) for i,val in b]}")
        print(f"  survivors f32={sorted(seta)}  q8={sorted(setb)}  "
              f"symdiff={sorted(seta ^ setb)}")
        print(f"  nearest candidate sits {min(margins):.4f} from its own cut")
        print()
    (GR / "2026-08-13-m4-g4-escalate-characterisation.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print("written -> 2026-08-13-m4-g4-escalate-characterisation.json")


if __name__ == "__main__":
    sys.exit(main())
