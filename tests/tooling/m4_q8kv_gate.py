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

EVERY UNIT OF WORK IS BANKED THE MOMENT IT LANDS, and that is not a nicety
either. The first attempt at this gate wrote NOTHING until a whole phase
finished, was stopped 76 minutes in, and lost all of it -- the same shape that
cost 37 minutes on 2026-08-10. A probe pair, a sweep prompt and a graded
divergence are each appended (and fsync'd) to a `.jsonl` as they complete, so
an interrupted run costs only the unit in flight, and `--resume` picks up from
the bank.

Resuming is refused across a changed configuration. Rows produced by two
different binaries are not evidence about either one, so the run records a
fingerprint (stoke sha256, model identity, n, corpus, probe spec) and stops if
it does not match. Nothing is ever deleted to make room: a bank that does not
belong to this run means a new `--out` prefix, not an overwrite.

Usage:
  python -u tests/tooling/m4_q8kv_gate.py --stoke build/win/stoke.exe \\
      --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -n 64 \\
      --out docs/superpowers/gate-reports/2026-08-12-m4-g4 [--resume]
"""

import argparse
import hashlib
import json
import os
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


def jsonl_load(path):
    """Banked records, REPAIRING a tail left half-written by a kill.

    Everything up to the first unparseable line is kept and the file is cut
    back to that boundary. The cut is not tidiness: caught by the smoke test,
    an append onto a surviving fragment FUSES with it, so the record is lost,
    and every record appended after it is discarded by the next load too. A
    bank that quietly stops absorbing progress is worse than no bank -- it
    would have re-run the same prompt on every resume.
    """
    out = []
    if not path.exists():
        return out
    raw = path.read_bytes()
    good = 0
    for line in raw.splitlines(keepends=True):
        text = line.strip()
        if text:
            try:
                out.append(json.loads(text))
            except json.JSONDecodeError:
                break
        good += len(line)
    if good != len(raw):
        with path.open("r+b") as fh:
            fh.truncate(good)
        print(f"  [resume] {path.name}: discarded {len(raw) - good} bytes of a "
              f"record cut short by the interruption, after {len(out)} good ones")
    return out


def jsonl_append(path, obj):
    """Append one record and get it ONTO THE DISK before the next unit starts.

    Without the fsync the bank is only as durable as the OS cache, which is
    exactly what a terminated task does not flush. The newline guard covers
    the case a repair cannot see -- a final record that parses but whose
    trailing newline never landed.
    """
    with path.open("a+b") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell():
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")     # append mode: writes land at the end
        fh.write((json.dumps(obj) + "\n").encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())


def fingerprint(args):
    """What a banked record is evidence ABOUT.

    The binary is hashed because it is the thing under test. The model is
    identified by size+mtime rather than a hash: it is 2.3 GB, and reading it
    end to end would evict the page cache this run depends on -- which is not
    a correctness question but does decide whether the sweep takes 2.6 hours
    or fifteen.
    """
    h = hashlib.sha256()
    with open(args.stoke, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = pathlib.Path(args.model).stat()
    return {"stoke": str(args.stoke), "stoke_sha256": h.hexdigest(),
            "model": str(args.model), "model_bytes": st.st_size,
            "model_mtime": int(st.st_mtime), "n_gen": args.n_gen,
            "corpus": args.corpus, "prompt_dir": str(args.prompt_dir),
            "probe_spec": args.probe_spec, "threads": args.threads}


# What a resume compares. `threads` is recorded but deliberately NOT here:
# the ids are byte-identical across default/4/6/10 threads (checked before
# --threads was plumbed through), so the thread count buys duration and
# cannot reach the verdict. Refusing a resume over it would block exactly the
# duration tuning it is there for.
IDENTITY_KEYS = ("stoke", "stoke_sha256", "model", "model_bytes",
                 "model_mtime", "n_gen", "corpus", "prompt_dir", "probe_spec")


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


def gen(stoke, model, prompt, n, kv, probe_pos=None, threads=None):
    cmd = [stoke, "run", model, "--raw", "-n", str(n), "-f", prompt,
           "--kv", kv]
    if threads:
        cmd += ["--threads", str(threads)]
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
    ap.add_argument("--threads", type=int, default=0,
                    help="forwarded to every `stoke run`; 0 leaves the "
                         "engine default. Duration only -- ids are "
                         "thread-invariant, so this is outside the resume "
                         "identity")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the banked .jsonl units of an "
                         "interrupted run with the SAME configuration")
    args = ap.parse_args()

    pdir = pathlib.Path(args.prompt_dir)
    corpus = [f"p{i:02d}" for i in range(args.corpus)]
    for c in corpus:
        if not (pdir / f"{c}.txt").exists():
            raise SystemExit(f"missing {pdir / (c + '.txt')}")

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    probe_path = outp.with_name(outp.name + "-probe.jsonl")
    sweep_jsonl = outp.with_name(outp.name + "-sweep.jsonl")
    verdict_jsonl = outp.with_name(outp.name + "-verdict.jsonl")
    tau_path = outp.with_name(outp.name + "-tau.json")
    run_path = outp.with_name(outp.name + "-run.json")
    banks = [probe_path, sweep_jsonl, verdict_jsonl]

    # ----------------------------------------------------------------- resume
    # Decided BEFORE anything runs, and never by deleting: an existing bank is
    # either this run's (resume it) or someone else's (refuse and say so).
    fp = fingerprint(args)
    present = [p.name for p in banks + [tau_path] if p.exists()]
    if args.resume:
        if not run_path.exists():
            raise SystemExit(
                f"--resume: no {run_path.name}; there is no banked run to "
                f"continue under this --out prefix.")
        old = json.loads(run_path.read_text(encoding="utf-8"))
        differ = sorted(k for k in IDENTITY_KEYS if old.get(k) != fp.get(k))
        if differ:
            raise SystemExit(
                "REFUSING to resume: the bank was produced by a different "
                f"configuration ({', '.join(differ)}). Records from two "
                "configurations are not evidence about either one -- use a "
                "new --out prefix.")
        print(f"=== RESUMING from {', '.join(present) or 'an empty bank'} ===")
    elif present:
        raise SystemExit(
            f"{outp.name}* already carries banked units ({', '.join(present)}). "
            "Pass --resume to continue that run, or choose a new --out prefix. "
            "Nothing is overwritten.")
    else:
        run_path.write_text(json.dumps(fp, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- phase 1
    # THE TAU FREEZE. Agreeing positions only: a position where the two sides
    # chose different tokens tells us about the divergence, not about the
    # background noise the threshold is supposed to describe.
    spec = []
    for item in args.probe_spec.split(","):
        name, _, poss = item.partition(":")
        spec.append((name, [int(x) for x in poss.split("+") if x]))

    if tau_path.exists():
        # A frozen tau is frozen. Re-deriving it on a resume -- after some of
        # the sweep it must judge has already been seen -- is the fix-the-bar
        # move the ruling names, even if the probes would give the same answer.
        frozen = json.loads(tau_path.read_text(encoding="utf-8"))
        tau = frozen["tau"]
        print("=== PHASE 1: SKIPPED, tau was frozen "
              f"{frozen['frozen_utc']} from {frozen['n_agreeing']}/"
              f"{frozen['n_probe_positions']} probe positions ===")
        for band in sorted(tau):
            print(f"  band {band}: tau={tau[band]:.6f}")
        print()
    else:
        print("=== PHASE 1: entry probe, freezing tau ===")
        records = {(r["prompt"], r["gp"]): r for r in jsonl_load(probe_path)}

        for name, positions in spec:
            path = str(pdir / f"{name}.txt")
            plen = None
            for gp in positions:
                if gp >= args.n_gen:
                    continue
                if (name, gp) in records:
                    continue
                if plen is None:
                    plen = prompt_len(args.stoke, args.model, path)
                t0 = time.time()
                a_ids, a_top = gen(args.stoke, args.model, path, args.n_gen,
                                   "f32", gp, args.threads)
                b_ids, b_top = gen(args.stoke, args.model, path, args.n_gen,
                                   "q8", gp, args.threads)
                covered = gp < len(a_ids) and gp < len(b_ids)
                rec = {"prompt": name, "gp": gp, "plen": plen,
                       "covered": covered,
                       "agree": covered and a_ids[gp] == b_ids[gp],
                       "a_top": a_top, "b_top": b_top,
                       "secs": round(time.time() - t0, 1)}
                jsonl_append(probe_path, rec)
                records[(name, gp)] = rec
                print(f"  {name} gp={gp} probed ({rec['secs']:.1f}s)")

        # tau is derived from the BANKED records in one place, so a resumed run
        # and an uninterrupted one cannot arrive at different thresholds from
        # the same probes.
        deltas = {band_of(lo): [] for lo, _ in BANDS}
        n_agree = n_seen = 0
        for name, positions in spec:
            for gp in positions:
                rec = records.get((name, gp))
                if rec is None:
                    continue
                n_seen += 1
                if not rec["agree"]:
                    continue          # tells us about the divergence, not the
                n_agree += 1          # background noise tau describes
                amap = {int(i): float(v) for i, v in rec["a_top"]}
                bmap = {int(i): float(v) for i, v in rec["b_top"]}
                band = band_of(rec["plen"] + gp)
                shared = set(amap) & set(bmap)
                for tok in shared:
                    deltas[band].append(abs(amap[tok] - bmap[tok]))
                print(f"  {name} gp={gp} band={band} agree "
                      f"({len(shared)} shared top-5)")

        tau = {}
        for band, ds in deltas.items():
            if ds:
                ds.sort()
                p95 = ds[min(len(ds) - 1, int(0.95 * len(ds)))]
            else:
                p95 = 0.0
            tau[band] = max(TAU_FLOOR, 2.0 * p95)
            print(f"  band {band}: n={len(ds)} p95={p95:.6f} -> "
                  f"tau={tau[band]:.6f}")

        frozen = {"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()),
                  "pair": "kv_f32 <-> kv_q8_0", "model": args.model,
                  "tau": tau, "n_probe_positions": n_seen,
                  "n_agreeing": n_agree,
                  "rule": "max(0.10, 2 x band p95) -- ruling 9.4-2"}
        tau_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
        print(f"  tau FROZEN -> {tau_path}\n")

    # ---------------------------------------------------------------- phase 2
    print(f"=== PHASE 2: {args.corpus}-corpus sweep, {args.n_gen} tokens ===")
    banked_rows = {r["prompt"]: r for r in jsonl_load(sweep_jsonl)}
    rows = []
    for name in corpus:
        row = banked_rows.get(name)
        if row is None:
            path = str(pdir / f"{name}.txt")
            t0 = time.time()
            a_ids, _ = gen(args.stoke, args.model, path, args.n_gen, "f32",
                           threads=args.threads)
            b_ids, _ = gen(args.stoke, args.model, path, args.n_gen, "q8",
                           threads=args.threads)
            first = None
            for i in range(min(len(a_ids), len(b_ids))):
                if a_ids[i] != b_ids[i]:
                    first = i
                    break
            if first is None and len(a_ids) != len(b_ids):
                first = min(len(a_ids), len(b_ids))
            row = {"prompt": name, "n_f32": len(a_ids), "n_q8": len(b_ids),
                   "first_divergence": first,
                   "f32_ids": a_ids, "q8_ids": b_ids,
                   "secs": round(time.time() - t0, 1)}
            jsonl_append(sweep_jsonl, row)
            tag = f"({row['secs']:.1f}s)"
        else:
            tag = "[banked]"
        rows.append(row)
        first = row["first_divergence"]
        print(f"  {name}: "
              f"{'SAME' if first is None else f'diverges at {first}'}  {tag}")

    # Counted from the rows, not accumulated in the loop, so the totals do not
    # depend on which prompts THIS invocation happened to run.
    n_div = sum(1 for r in rows if r["first_divergence"] is not None)
    tok_total = sum(min(r["n_f32"], r["n_q8"]) for r in rows)

    sweep_path = outp.with_name(outp.name + "-sweep.json")
    sweep_path.write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- phase 3
    print(f"\n=== PHASE 3: grading {n_div} first divergences (ruling S6) ===")
    graded = {v["prompt"]: v for v in jsonl_load(verdict_jsonl)}
    verdicts = []
    for r in rows:
        if r["first_divergence"] is None:
            continue
        if r["prompt"] in graded:
            v = graded[r["prompt"]]
            verdicts.append(v)
            print(f"  {v['prompt']} gp={v['gp']} -> {v['verdict']} [banked]")
            continue
        gp = r["first_divergence"]
        path = str(pdir / f"{r['prompt']}.txt")
        _, a_top = gen(args.stoke, args.model, path, args.n_gen, "f32", gp,
                       args.threads)
        _, b_top = gen(args.stoke, args.model, path, args.n_gen, "q8", gp,
                       args.threads)
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
        v = {"prompt": r["prompt"], "gp": gp, "band": band,
             "tau": t, "x": x, "y": y, "g_x": gx, "g_y": gy,
             "verdict": verdict, "note": note}
        jsonl_append(verdict_jsonl, v)
        verdicts.append(v)
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
