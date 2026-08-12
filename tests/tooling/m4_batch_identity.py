#!/usr/bin/env python3
"""M4 gate item 2 -- batch-composition independence on the REAL model.

THE BAR (2026-08-04 ruling section 8.4, item 2): "same seq => token-identical
at batch sizes {1,2,4,8,16}, real model, EXACT (bit-level ids). Hard fail."

WHAT THIS COMPARES, AND WHY IT IS A BYTE COMPARE. `stoke batch` prints, for
each sequence, a `== seq i ==` header followed by exactly what `stoke run`
prints for that prompt: the generated ids one per line, `---`, then the
decoded text. So a sequence's block with its header removed IS a whole
`stoke run` stdout, and the check is `==` on bytes -- no re-tokenizing, no
re-formatting, nothing in between that could absorb a difference.

WIDTH 1 IS NOT REDUNDANT. `stoke run` drives model_forward_position over
m->scratch; `stoke batch --seqs 1` drives model_forward_batch over
m->scratch_n[0]. They are the same body (model.c) but not the same scratch
and not the same orchestration loop, so N=1 is the case that isolates the
driver from the batching.

PROMPTS REPEAT ONCE N EXCEEDS THE CORPUS, by construction: sequence s runs
prompt s % len(prompts). At N=16 over 5 prompts that means three or four
slots run the SAME prompt, which is the case a slot-indexed bug would leave
looking self-consistent -- they must each match the solo reference, not
merely each other.

Usage:
  python tests/tooling/m4_batch_identity.py --stoke build/win/stoke.exe \
      --model models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -n 16
"""

import argparse
import pathlib
import subprocess
import sys
import time

DEFAULT_PROMPTS = ["p00", "p10", "p20", "p30", "p45"]
DEFAULT_WIDTHS = [1, 2, 4, 8, 16]


def run(cmd):
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    dt = time.time() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(f"FAILED (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.stdout, dt


def split_seq_blocks(stdout, n_seq):
    """Blocks in order, header lines removed, BY BYTE OFFSET.

    Deliberately not a line split-and-rejoin. The first version of this did
    that and dropped the newline the following header line had consumed, so
    every block except the LAST came up one byte short -- which presented as
    "N-1 of N slots differ at every width", i.e. exactly like a real batching
    bug, and cost a full gate run to diagnose. Slicing the original bytes
    cannot lose or invent a byte.

    Headers are matched only at a line boundary: decoded model output is
    arbitrary text and could contain anything, including a header lookalike.
    """
    spans = []
    pos = 0
    for i in range(n_seq):
        hdr = b"== seq %d ==\n" % i
        idx = stdout.find(hdr, pos)
        while idx > 0 and stdout[idx - 1:idx] != b"\n":
            idx = stdout.find(hdr, idx + 1)
        if idx < 0:
            raise SystemExit(f"no `== seq {i} ==` header at a line boundary")
        spans.append((idx, idx + len(hdr)))
        pos = idx + len(hdr)

    blocks = []
    for i in range(n_seq):
        end = spans[i + 1][0] if i + 1 < n_seq else len(stdout)
        blocks.append(stdout[spans[i][1]:end])
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stoke", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-dir", default="tests/data/parity")
    ap.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    ap.add_argument("--widths", default=",".join(str(w) for w in DEFAULT_WIDTHS))
    ap.add_argument("-n", "--n-gen", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=0, help="0 = per-run default")
    args = ap.parse_args()

    prompts = args.prompts.split(",")
    widths = [int(w) for w in args.widths.split(",")]
    pdir = pathlib.Path(args.prompt_dir)
    paths = [str(pdir / f"{p}.txt") for p in prompts]
    for p in paths:
        if not pathlib.Path(p).exists():
            raise SystemExit(f"missing prompt file {p}")

    print(f"model   : {args.model}")
    print(f"prompts : {', '.join(prompts)}")
    print(f"widths  : {widths}   n_gen={args.n_gen}")
    print()

    # Solo references, one per prompt.
    solo = {}
    for name, path in zip(prompts, paths):
        out, dt = run([args.stoke, "run", args.model, "--raw",
                       "-n", str(args.n_gen), "-f", path])
        solo[name] = out
        n_ids = len([l for l in out.split(b"\n") if l.isdigit()])
        print(f"  solo {name}: {n_ids} ids in {dt:.1f}s")
    print()

    failures = []
    for w in widths:
        cmd = [args.stoke, "batch", args.model, "--raw", "-n", str(args.n_gen)]
        for path in paths:
            cmd += ["-f", path]
        cmd += ["--seqs", str(w)]
        if args.ctx:
            cmd += ["--ctx", str(args.ctx)]
        out, dt = run(cmd)
        blocks = split_seq_blocks(out, w)

        bad = []
        for s in range(w):
            name = prompts[s % len(prompts)]
            if blocks[s] != solo[name]:
                bad.append((s, name))
        status = "OK  " if not bad else "FAIL"
        print(f"  N={w:<3} {status} {w}/{w} slots compared in {dt:.1f}s"
              if not bad else
              f"  N={w:<3} {status} {len(bad)} of {w} slots differ: {bad}")
        if bad:
            failures.append((w, bad))
            # Show the first divergence concretely -- a gate that only counts
            # is a gate nobody can act on.
            s, name = bad[0]
            got, want = blocks[s], solo[name]
            if len(got) != len(want):
                print(f"       slot {s}: {len(got)} bytes vs {len(want)} "
                      f"in the solo reference")
            gl, wl = got.split(b"\n"), want.split(b"\n")
            for i in range(max(len(gl), len(wl))):
                g = gl[i] if i < len(gl) else b"<missing>"
                e = wl[i] if i < len(wl) else b"<missing>"
                if g != e:
                    print(f"       first difference at line {i}: "
                          f"got {g!r} want {e!r}")
                    break

    print()
    if failures:
        print(f"GATE ITEM 2: FAIL -- {len(failures)} width(s) diverged")
        return 1
    total = sum(widths)
    print(f"GATE ITEM 2: PASS -- {total} sequence streams over "
          f"{len(widths)} widths, all byte-identical to solo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
