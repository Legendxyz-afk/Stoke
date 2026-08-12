#!/usr/bin/env python3
"""M4 gate item 4 -- the Q8-KV ON/OFF divergence count, graded by ruling S6.

THE CRITERION (2026-08-04 ruling S8.1, quoted): "KV-ON vs KV-OFF, within-stoke,
same engine, only the KV representation differing -- graded by S6 with its own
tau (frozen from an M4 entry probe of the ON<->OFF pair), zero BUG-class, and
class divergences bounded by: first-divergence prompt rate <= the M2 census
rate (19/50) AND flip rate <= 0.5%/token, both on the frozen 50-corpus."

S6's rule, applied at the first divergent position where side X chose x and
side Y chose y:

    g_X = logit_X(x) - logit_X(y)      g_Y = logit_Y(y) - logit_Y(x)

  ARITHMETIC-CLASS  iff max(g_X,g_Y) < tau AND the top-5 sets agree after
                    discarding, on each side, candidates within tau of that
                    side's rank-5 logit
  BUG               iff max(g_X,g_Y) >= 2*tau, OR either side's chosen token
                    sits more than tau below the other side's rank-5 logit
  ESCALATE          otherwise -> the L2/L3 ladder decides

tau = max(0.10, 2 x band p95), per position band, FROZEN BY PHASE 1 BEFORE IT
GRADES ANYTHING. That ordering is not a nicety: a tau chosen after seeing the
divergences it must judge is a fitted constant, and the ruling calls moving a
frozen tau "the fix-the-bar move".

TWO THINGS THIS INHERITS AND DOES NOT PRETEND OTHERWISE.
  - tau is frozen from AGREEING positions of the SAME pair it grades, so a
    defect that uniformly inflated logit differences would inflate tau and the
    margins together. That headroom is already carried in the M3 ledger; it is
    restated here because this instrument has the same shape.
  - The top-5 probe cannot supply logit_X(y) when y is outside X's top 5. That
    case is NOT silently treated as small: it goes to ESCALATE, which is the
    fail-safe direction the ruling specifies.

Usage:
  python -u tests/tooling/m4_q8kv_gate.py --stoke build/win/stoke.exe \\
      --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -n 64 \\
      --out docs/superpowers/gate-reports/2026-08-12-m4-g4
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

# Same three bands M3 used, so the two instruments' constants are comparable.
BANDS = [(0, 100), (100, 300), (300, 10 ** 9)]
TAU_FLOOR = 0.10

PROBE_RE = re.compile(r"probe pos=(\d+): id=(\d+) logit=(-?[\d.]+)")


def band_of(abs_pos):
    for lo, hi in BANDS:
        if lo <= abs_pos < hi:
            return f"[{lo},{hi if hi < 10**9 else 'inf'})"
    raise AssertionError(abs_pos)


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"FAILED (exit {p.returncode}): {' '.join(cmd)}")
    return p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def ids_of(stdout):
    out = []
    for line in stdout.split("\n"):
        line = line.strip()
        if line == "---":
            break
        if line.isdigit():
            out.append(int(line))
    return out


def prompt_len(stoke, model, path):
    """Prompt token count, for the position BAND.

    `stoke tokenize` prints ONE bracketed list, not one id per line -- reading
    it line-wise silently yields 0 and files every position into the lowest
    band, which is a wrong tau applied without any error. Parsed from the
    bracket so a format change fails loudly instead.
    """
    so, _ = run([stoke, "tokenize", model, path])
    m = re.search(r"\[([\d,\s]*)\]", so)
    if not m:
        raise SystemExit(f"could not parse `stoke tokenize` output: {so[:120]!r}")
    body = m.group(1).strip()
    return 0 if not body else len(body.split(","))


def gen(stoke, model, prompt, n, kv, probe_pos=None):
    cmd = [stoke, "run", model, "--raw", "-n", str(n), "-f", prompt,
           "--kv", kv]
    if probe_pos is not None:
        cmd += ["--probe-pos", str(probe_pos)]
    so, se = run(cmd)
    top5 = [(int(m.group(2)), float(m.group(3)))
            for m in PROBE_RE.finditer(se)]
    return ids_of(so), top5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stoke", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-dir", default="tests/data/parity")
    ap.add_argument("-n", "--n-gen", type=int, default=64)
    ap.add_argument("--corpus", type=int, default=50)
    # PER-PROMPT positions, `name:gp+gp+gp`, because probe cost is per RUN and
    # a 447-token prompt costs ~20x a 30-token one. Band coverage is what the
    # default encodes: short prompts sit in [0,100), p32 (104 tok) lands every
    # generated position in [100,300), and p44 (447 tok) is the cheapest of the
    # five prompts that reach [300,inf). A band with no samples falls back to
    # the 0.10 floor, which is defined but weak, so coverage is chosen rather
    # than hoped for.
    ap.add_argument("--probe-spec",
                    default="p00:0+15+31+47+63,p10:0+31+63,"
                            "p32:0+15+31+47+63,p44:0+63")
    ap.add_argument("--out", required=True, help="path prefix for the artifacts")
    args = ap.parse_args()

    pdir = pathlib.Path(args.prompt_dir)
    corpus = [f"p{i:02d}" for i in range(args.corpus)]
    for c in corpus:
        if not (pdir / f"{c}.txt").exists():
            raise SystemExit(f"missing {pdir / (c + '.txt')}")

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- phase 1
    # THE TAU FREEZE. Agreeing positions only: a position where the two sides
    # chose different tokens tells us about the divergence, not about the
    # background noise the threshold is supposed to describe.
    print("=== PHASE 1: entry probe, freezing tau ===")
    spec = []
    for item in args.probe_spec.split(","):
        name, _, poss = item.partition(":")
        spec.append((name, [int(x) for x in poss.split("+") if x]))
    deltas = {band_of(b[0]): [] for b in [(lo, hi) for lo, hi in BANDS]}
    n_agree = n_seen = 0

    for name, positions in spec:
        path = str(pdir / f"{name}.txt")
        plen = prompt_len(args.stoke, args.model, path)

        for gp in positions:
            if gp >= args.n_gen:
                continue
            a_ids, a_top = gen(args.stoke, args.model, path, args.n_gen, "f32", gp)
            b_ids, b_top = gen(args.stoke, args.model, path, args.n_gen, "q8", gp)
            n_seen += 1
            if gp >= len(a_ids) or gp >= len(b_ids):
                continue
            if a_ids[gp] != b_ids[gp]:
                continue                      # not an agreeing position
            n_agree += 1
            amap = dict(a_top)
            bmap = dict(b_top)
            band = band_of(plen + gp)
            for tok in set(amap) & set(bmap):
                deltas[band].append(abs(amap[tok] - bmap[tok]))
            print(f"  {name} gp={gp} band={band} agree "
                  f"({len(set(amap) & set(bmap))} shared top-5)")

    tau = {}
    for band, ds in deltas.items():
        if ds:
            ds.sort()
            p95 = ds[min(len(ds) - 1, int(0.95 * len(ds)))]
        else:
            p95 = 0.0
        tau[band] = max(TAU_FLOOR, 2.0 * p95)
        print(f"  band {band}: n={len(ds)} p95={p95:.6f} -> tau={tau[band]:.6f}")

    frozen = {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "pair": "kv_f32 <-> kv_q8_0", "model": args.model,
              "tau": tau, "n_probe_positions": n_seen,
              "n_agreeing": n_agree,
              "rule": "max(0.10, 2 x band p95) -- ruling 9.4-2"}
    tau_path = outp.with_name(outp.name + "-tau.json")
    tau_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    print(f"  tau FROZEN -> {tau_path}\n")

    # ---------------------------------------------------------------- phase 2
    print(f"=== PHASE 2: {args.corpus}-corpus sweep, {args.n_gen} tokens ===")
    rows, n_div, tok_total = [], 0, 0
    for name in corpus:
        path = str(pdir / f"{name}.txt")
        t0 = time.time()
        a_ids, _ = gen(args.stoke, args.model, path, args.n_gen, "f32")
        b_ids, _ = gen(args.stoke, args.model, path, args.n_gen, "q8")
        first = None
        for i in range(min(len(a_ids), len(b_ids))):
            if a_ids[i] != b_ids[i]:
                first = i
                break
        if first is None and len(a_ids) != len(b_ids):
            first = min(len(a_ids), len(b_ids))
        tok_total += min(len(a_ids), len(b_ids))
        if first is not None:
            n_div += 1
        rows.append({"prompt": name, "n_f32": len(a_ids), "n_q8": len(b_ids),
                     "first_divergence": first,
                     "f32_ids": a_ids, "q8_ids": b_ids})
        print(f"  {name}: {'SAME' if first is None else f'diverges at {first}'}"
              f"  ({time.time() - t0:.1f}s)")

    sweep_path = outp.with_name(outp.name + "-sweep.json")
    sweep_path.write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- phase 3
    print(f"\n=== PHASE 3: grading {n_div} first divergences (ruling S6) ===")
    verdicts = []
    for r in rows:
        if r["first_divergence"] is None:
            continue
        gp = r["first_divergence"]
        path = str(pdir / f"{r['prompt']}.txt")
        _, a_top = gen(args.stoke, args.model, path, args.n_gen, "f32", gp)
        _, b_top = gen(args.stoke, args.model, path, args.n_gen, "q8", gp)
        band = band_of(prompt_len(args.stoke, args.model, path) + gp)
        t = tau[band]

        x, y = r["f32_ids"][gp], r["q8_ids"][gp]
        amap, bmap = dict(a_top), dict(b_top)
        a_r5 = min(v for _, v in a_top) if a_top else None
        b_r5 = min(v for _, v in b_top) if b_top else None

        note = ""
        if y not in amap or x not in bmap:
            verdict = "ESCALATE"
            note = ("the other side's token is outside this side's top-5, so "
                    "its logit is unknown; fail-safe per the ruling")
            gx = gy = None
        else:
            gx = amap[x] - amap[y]
            gy = bmap[y] - bmap[x]
            g = max(gx, gy)
            top_a = {i for i, v in a_top if v > a_r5 + t}
            top_b = {i for i, v in b_top if v > b_r5 + t}
            sets_agree = (top_a == top_b)
            below_a = (b_r5 is not None and amap[x] < b_r5 - t)
            below_b = (a_r5 is not None and bmap[y] < a_r5 - t)
            if g >= 2 * t or below_a or below_b:
                verdict = "BUG"
            elif g < t and sets_agree:
                verdict = "ARITHMETIC-CLASS"
            else:
                verdict = "ESCALATE"
                note = ("in the [tau, 2tau) band or a top-5 set difference; "
                        "the L2/L3 ladder decides")
        verdicts.append({"prompt": r["prompt"], "gp": gp, "band": band,
                         "tau": t, "x": x, "y": y, "g_x": gx, "g_y": gy,
                         "verdict": verdict, "note": note})
        print(f"  {r['prompt']} gp={gp} band={band} tau={t:.4f} "
              f"g=({gx if gx is None else round(gx,4)},"
              f"{gy if gy is None else round(gy,4)}) -> {verdict}")

    # ---------------------------------------------------------------- verdict
    n_bug = sum(1 for v in verdicts if v["verdict"] == "BUG")
    n_esc = sum(1 for v in verdicts if v["verdict"] == "ESCALATE")
    flip_rate = (n_div / tok_total * 100.0) if tok_total else 0.0

    summary = {"tau": frozen, "n_prompts": len(rows), "n_diverging": n_div,
               "tokens_compared": tok_total, "flip_rate_pct_per_token": flip_rate,
               "n_bug": n_bug, "n_escalate": n_esc, "verdicts": verdicts}
    outp.with_name(outp.name + "-verdict.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")

    print("\n=== GATE ITEM 4 ===")
    print(f"  first-divergence prompts : {n_div}/{len(rows)}   (bound: <= 19/50)")
    print(f"  flip rate                : {flip_rate:.3f}%/token  (bound: <= 0.5)")
    print(f"  BUG-class                : {n_bug}              (bound: 0)")
    print(f"  ESCALATE (ladder owed)   : {n_esc}")
    ok = (n_bug == 0 and n_div <= 19 and flip_rate <= 0.5 and n_esc == 0)
    if n_esc:
        print("  -> NOT a verdict yet: every ESCALATE needs its L2/L3 ladder run.")
    print("GATE ITEM 4: " + ("PASS" if ok else "NOT PASSED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
