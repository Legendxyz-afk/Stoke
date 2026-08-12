#!/usr/bin/env python3
"""M4 gate item 5 -- the static N in {1..16} aggregate scaling curve.

THE BAR (2026-08-04 ruling section 8.4, item 5): "monotone non-decreasing
aggregate within within-cycle spread through N=8; any genuine inversion
(beyond spread) = results-vs-physics memo before the stamp, not a silent
recording -- a static-batch inversion is the M0 disease stoke exists to not
have, and it would indict the design, not the tuning."

CONFIGURATION IS THE PLAYBOOK'S LOCK 2, not a choice made here: Q4_K_M,
per-slot ctx 1024, ALL N THE SAME CTX. At f32 KV that is 288 KiB/token, so
N=16 x 1024 reserves 4.83 GB of pool beside a 2.33 GB model -- which is why
the lock exists and why 2048 was ruled out for this box.

▶ THE MEASURAND IS THE DECODE AGGREGATE, not end-to-end. ModelTiming splits
prefill from decode and the rate stoke prints divides decode tokens by decode
seconds -- the same unit llama-bench calls tg, and the same denominator at
every N. Prompts are deliberately SHORT: prefill here is per-sequence and
sequential, so a long prompt would spend the run's wall clock on work the
curve does not measure.

▶ INTERLEAVED CYCLES, NOT BLOCKS. The sweep runs 1..16, then 1..16 again,
rather than repeating each N before moving on. Box state drifts over minutes;
blocked repeats turn that drift into an apparent N effect, which is exactly
the error mode that produced this project's retracted "10 threads are 15%
worse" claim. The spread reported per N is the spread ACROSS cycles, and the
monotonicity test is applied to the medians with that spread as its
tolerance.

▶ WHAT THIS SCRIPT DOES NOT DO: quiesce the box. Stop ISLC, warm the page
cache, and confirm the standby cache is about the model size BEFORE running
this -- a stoke rate near 1.4 tok/s means the disk is being measured, and
free RAM reads healthy while the model is entirely absent from cache.

Usage:
  python tests/tooling/m4_curve.py --stoke build/win/stoke.exe \
      --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
      -n 32 --ctx 1024 --cycles 3
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time

# Short and unremarkable: the curve measures decode, and prefill is charged
# separately. Five of them so slots do not all run the same stream.
PROMPTS = [
    "Explain how a rainbow forms.",
    "List the planets in order.",
    "What is the capital of Peru?",
    "Describe photosynthesis briefly.",
    "Why is the sky blue?",
]

RATE_RE = re.compile(r"decode (\d+) tok in ([\d.]+) ms \(([\d.]+) tok/s aggregate\)")


def run_one(stoke, model, n_seq, n_gen, ctx, threads):
    cmd = [stoke, "batch", model, "--raw", "-n", str(n_gen)]
    for p in PROMPTS:
        cmd += ["-p", p]
    cmd += ["--seqs", str(n_seq), "--ctx", str(ctx), "--timing"]
    if threads:
        cmd += ["--threads", str(threads)]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    wall = time.time() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"FAILED (exit {proc.returncode}) at N={n_seq}")
    err = proc.stderr.decode("utf-8", "replace")
    m = RATE_RE.search(err)
    if not m:
        raise SystemExit(f"no aggregate rate in stderr at N={n_seq}:\n{err}")
    n_dec, ms, rate = int(m.group(1)), float(m.group(2)), float(m.group(3))
    return {"n_seq": n_seq, "n_decode": n_dec, "decode_ms": ms,
            "tok_s": rate, "wall_s": round(wall, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stoke", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", "--n-gen", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--max-n", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    widths = list(range(1, args.max_n + 1))
    print(f"model  : {args.model}")
    print(f"config : ctx {args.ctx}, n_gen {args.n_gen}, "
          f"{args.cycles} interleaved cycles over N=1..{args.max_n}")
    print()

    runs = []
    for c in range(args.cycles):
        for n in widths:
            r = run_one(args.stoke, args.model, n, args.n_gen, args.ctx,
                        args.threads)
            r["cycle"] = c
            runs.append(r)
            print(f"  cycle {c} N={n:<3} {r['tok_s']:8.3f} tok/s "
                  f"({r['n_decode']} tok in {r['decode_ms']/1000:.1f}s, "
                  f"wall {r['wall_s']}s)")
        print()

    by_n = {n: sorted(r["tok_s"] for r in runs if r["n_seq"] == n)
            for n in widths}
    med = {n: statistics.median(by_n[n]) for n in widths}
    spread = {n: (max(by_n[n]) - min(by_n[n])) for n in widths}

    print("  N   median tok/s   spread   samples")
    for n in widths:
        print(f"  {n:<3} {med[n]:12.3f}   {spread[n]:6.3f}   "
              + " ".join(f"{v:.2f}" for v in by_n[n]))
    print()

    # The bar is monotone non-decreasing THROUGH N=8, judged against the
    # within-cycle spread rather than against exact ordering: two adjacent Ns
    # differing by less than the noise the box itself shows are not ordered
    # by this instrument, and pretending otherwise manufactures a finding.
    inversions = []
    for i in range(1, min(8, args.max_n)):
        a, b = widths[i - 1], widths[i]
        tol = max(spread[a], spread[b])
        if med[b] < med[a] - tol:
            inversions.append((a, b, med[a], med[b], tol))

    tail = []
    for i in range(min(8, args.max_n), args.max_n):
        a, b = widths[i - 1], widths[i]
        tol = max(spread[a], spread[b])
        if med[b] < med[a] - tol:
            tail.append((a, b, med[a], med[b], tol))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"runs": runs,
                       "median": med,
                       "spread": spread,
                       "inversions_through_8": inversions,
                       "inversions_past_8": tail}, f, indent=1)
        print(f"  wrote {args.json_out}")

    if tail:
        print("  NOTE past N=8 (outside the bar, recorded not judged):")
        for a, b, ma, mb, tol in tail:
            print(f"    N={a} {ma:.3f} -> N={b} {mb:.3f} "
                  f"(drop {ma - mb:.3f} > spread {tol:.3f})")

    if inversions:
        print()
        print("GATE ITEM 5: FAIL -- genuine inversion through N=8")
        for a, b, ma, mb, tol in inversions:
            print(f"    N={a} {ma:.3f} -> N={b} {mb:.3f} "
                  f"(drop {ma - mb:.3f} > spread {tol:.3f})")
        print("  Ruling 8.4 item 5: this is a results-vs-physics memo BEFORE "
              "the stamp, not a re-run until green.")
        return 1

    print(f"GATE ITEM 5: PASS -- aggregate monotone non-decreasing through "
          f"N=8 within spread ({med[1]:.3f} -> {med[min(8, args.max_n)]:.3f} "
          f"tok/s, {med[1] and med[min(8, args.max_n)] / med[1]:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
