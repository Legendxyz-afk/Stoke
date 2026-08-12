#!/usr/bin/env python3
"""M4 gate item 3 -- prefix dedup, on the REAL model.

THE BAR (2026-08-04 ruling section 8.4, item 3): "dedup'd shared prefix
produces streams identical to un-dedup'd, refcounts prove out in sim and one
real-model spot check."

  - refcounts in sim ....... tests/test_blockpool.c (the 1e6-op storm adopts
                             and reports dedup_hit/adopt coverage; bp_check
                             asserts I1/I2 after EVERY op)
  - physical sharing ....... tests/test_kvshare.c (kv_view returns the SAME
                             pointer for the adopted span, a different one
                             after it)
  - identical streams ...... here, on the real model

WHY THE PROMPTS ARE BUILT RATHER THAN PICKED. The parity corpus prompts share
no prefix with each other, so a dedup run over them would adopt nothing and
pass this gate while proving nothing -- the exact shape that made the block
pool's first op storm report dedup_hit=0. So the run builds sequences that
share a long preamble and then DIVERGE, which is also M5's own trace (N
concurrent requests over one 1-2k system prompt).

THREE WITNESSES, because "dedup was enabled" is not "dedup fired":
  1. the adoption count stoke reports per sequence is > 0 after the first
  2. the prefill token count DROPS by exactly that many positions -- work
     skipped, not merely bookkeeping touched
  3. stdout is byte-identical to the un-dedup'd run

Usage:
  python tests/tooling/m4_prefix_dedup.py --stoke build/win/stoke.exe \
      --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -n 16 --seqs 4
"""

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile
import time

SUFFIXES = [
    " Answer in exactly one word.",
    " Answer at length, with examples.",
    " Reply with a single number and nothing else.",
    " Summarize the above for a child.",
    " List three bullet points.",
    " Explain the opposite view instead.",
    " Give a one-sentence definition.",
    " Translate your answer into French.",
]


def run(cmd):
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dt = time.time() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"FAILED (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.stdout, proc.stderr.decode("utf-8", "replace"), dt


def parse_dedup(stderr):
    m = re.search(r"dedup: (\d+) of (\d+) prompt positions adopted \(([^)]*)\)",
                  stderr)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), [int(x) for x in m.group(3).split()]


def parse_prefill(stderr):
    m = re.search(r"prefill (\d+) tok", stderr)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stoke", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--preamble", default="tests/data/parity/p20.txt")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("-n", "--n-gen", type=int, default=16)
    args = ap.parse_args()

    if args.seqs > len(SUFFIXES):
        raise SystemExit(f"--seqs at most {len(SUFFIXES)}")

    preamble = pathlib.Path(args.preamble).read_text(encoding="utf-8").rstrip("\n")
    tmp = tempfile.mkdtemp(prefix="m4dedup")
    paths = []
    for i in range(args.seqs):
        p = pathlib.Path(tmp) / f"seq{i}.txt"
        p.write_text(preamble + SUFFIXES[i], encoding="utf-8")
        paths.append(str(p))

    print(f"model    : {args.model}")
    print(f"preamble : {args.preamble} ({len(preamble)} bytes)")
    print(f"seqs     : {args.seqs} sharing that preamble, diverging after it")
    print(f"n_gen    : {args.n_gen}")
    print()

    base = [args.stoke, "batch", args.model, "--raw", "-n", str(args.n_gen)]
    for p in paths:
        base += ["-f", p]
    base += ["--timing"]

    out_off, err_off, t_off = run(base)
    out_on, err_on, t_on = run(base + ["--dedup"])

    pre_off, pre_on = parse_prefill(err_off), parse_prefill(err_on)
    ded = parse_dedup(err_on)

    print(f"  dedup OFF: prefill {pre_off} tok  ({t_off:.1f}s)")
    print(f"  dedup ON : prefill {pre_on} tok  ({t_on:.1f}s)")
    if ded is None:
        print("  no dedup line reported -- stoke did not run with --dedup?")
        print("\nGATE ITEM 3: FAIL")
        return 1
    total, prompt_total, per_seq = ded
    print(f"  adopted  : {total} of {prompt_total} prompt positions {per_seq}")
    print()

    fails = []

    # 1. It fired, and not on the first sequence (nothing is resident yet).
    if total <= 0:
        fails.append("nothing was adopted -- dedup did not fire")
    if per_seq and per_seq[0] != 0:
        fails.append(f"seq 0 adopted {per_seq[0]}, but nothing can be "
                     f"resident before the first sequence prefills")
    if len(per_seq) > 1 and not all(a > 0 for a in per_seq[1:]):
        fails.append(f"a sequence after the first adopted nothing: {per_seq}")

    # 2. Work was actually skipped, by exactly the reported amount.
    if pre_off is None or pre_on is None:
        fails.append("could not read prefill counts from --timing")
    elif pre_off - pre_on != total:
        fails.append(f"prefill dropped by {pre_off - pre_on} but "
                     f"{total} positions were reported adopted")

    # 3. And nothing about the output changed.
    if out_on != out_off:
        fails.append("stdout differs between the dedup'd and un-dedup'd runs")
        goff, gon = out_off.split(b"\n"), out_on.split(b"\n")
        for i in range(max(len(goff), len(gon))):
            a = goff[i] if i < len(goff) else b"<missing>"
            b = gon[i] if i < len(gon) else b"<missing>"
            if a != b:
                print(f"       first difference at line {i}: "
                      f"off {a!r} on {b!r}")
                break

    for f in fails:
        print(f"  FAIL: {f}")
    if fails:
        print("\nGATE ITEM 3: FAIL")
        return 1
    print(f"GATE ITEM 3: PASS -- {total} prompt positions served from the "
          f"prefix cache, prefill fell {pre_off} -> {pre_on}, "
          f"streams byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
