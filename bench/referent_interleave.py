#!/usr/bin/env python3
"""bench/referent_interleave.py -- measure stoke against llama.cpp, interleaved.

WHY THIS EXISTS, and why it is a separate tool rather than a bench mode.

This project has been bitten three times by the same error: comparing two
performance figures that were not measured together (K1's %ref denominator;
the 2026-08-03 "residue", which subtracted a quiet-window bench from a
loaded-window e2e; and the 2026-08-03 -O ruling's own central comparison,
which measured llama on this box and quoted stoke's rate from a different
session). Each time the arithmetic was right and the basis was not.

The Fable K3 ruling (2026-08-03, S4) made the rule binding:

    Two performance figures may be compared -- ratio, difference, or
    percentage -- only if measured in the same session under the same
    recorded conditions, interleaved whenever both sides are runnable.

This tool is where that rule is mechanised for the one comparison the M3
attribution ladder actually turns on: stoke's decode rate against the
vendored llama.cpp's, on the same file, same box, same pinned core. It runs
A,B,A,B and reports the WITHIN-CYCLE ratio, so box drift between cycles
moves both sides together and cannot pass for a difference.

WHY NOT INSIDE `stoke bench --ops`. The ruling denied that (S5): the bench is
the self-instrument of an embeddable engine and must run on boxes with no
vendor tree, and an instrument that executes its own comparison target
conflates measuring with refereeing. The referent belongs at the protocol
layer -- here.

WHY PYTHON AND NOT POWERSHELL. It imports box_sample/sidecar_delta/box_grade
from append_history.py rather than reimplementing them. Two copies of a
threshold in two languages is how a basis drifts, which is the failure this
whole tool exists to prevent.

WHAT A RECORD IS GOOD FOR. A record is valid for exactly the session that
took it. The attribution ladder may cite paired records and nothing else; a
referent figure re-quoted from another day is precisely what the rule
forbids, however carefully it was measured on that day.
"""

import argparse
import ctypes
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from append_history import (  # noqa: E402
    box_grade, box_sample, sidecar_delta, git_provenance, sha256_of,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The protocol both sides are held to. Changing any of these changes what a
# record MEANS, so the record carries the version and a reader can tell.
PROTOCOL = 2

# stoke's rate line, written to stderr by `stoke run --timing`.
TIMING_RE = re.compile(
    r"decode\s+(\d+)\s+tok in\s+([0-9.]+)\s+ms\s+\(([0-9.]+)\s+tok/s\)")


def pin(proc, affinity_mask, high_priority=True):
    """Pin a running process to `affinity_mask` and raise its priority.

    Windows only; a no-op elsewhere. Applied immediately after spawn rather
    than at creation, so the first milliseconds of process startup may run
    unpinned. That is deliberate and harmless HERE: what is compared is the
    decode loop, which begins after the model is open (~150 ms later) and
    runs for seconds. It would NOT be acceptable for a load-time measurement.
    """
    if sys.platform != "win32":
        return
    PROCESS_SET_INFORMATION = 0x0200
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = k32.OpenProcess(PROCESS_SET_INFORMATION, False, proc.pid)
    if not h:
        return
    try:
        k32.SetProcessAffinityMask(h, ctypes.c_size_t(affinity_mask))
        if high_priority:
            k32.SetPriorityClass(h, 0x00000080)  # HIGH_PRIORITY_CLASS
    finally:
        k32.CloseHandle(h)


def run_stoke(exe, model, prompt, n_gen, affinity):
    """One pinned stoke generation. Returns (tok_s, detail) or (None, error).

    The rate is READ FROM THE BINARY, not derived from wall clock out here:
    `stoke run --timing` reports its own decode loop, so load and prefill are
    excluded at the source instead of being differenced away afterwards --
    which is the method this tool exists to retire.
    """
    cmd = [exe, "run", model, "-p", prompt, "-n", str(n_gen), "--timing"]
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    pin(p, affinity)
    _, err = p.communicate()
    if p.returncode != 0:
        return None, f"stoke exited {p.returncode}: {err.strip()[:300]}"
    m = TIMING_RE.search(err)
    if not m:
        return None, f"no --timing rate in stoke stderr: {err.strip()[:300]}"
    return float(m.group(3)), {"n_decode": int(m.group(1)),
                               "decode_ms": float(m.group(2))}


def run_llama(bench_exe, model, n_gen, affinity, reps, threads):
    """One pinned llama-bench tg run. Returns (tok_s, detail) or (None, error).

    `-p 0` so no prompt processing is included and the reported figure is the
    tg (token generation) rate alone -- the same unit stoke's decode_ms
    covers. llama-bench does its own pinning via -C/--cpu-strict/--prio, so
    both sides are pinned by their own supported mechanism rather than one
    being wrapped in something the other is not.
    """
    cmd = [bench_exe, "-m", model, "-t", str(threads), "-p", "0",
           "-n", str(n_gen), "-r", str(reps), "-C", hex(affinity),
           "--cpu-strict", "1", "--prio", "2", "-o", "jsonl"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        return None, f"llama-bench exited {r.returncode}: {r.stderr.strip()[:300]}"
    rows = [json.loads(ln) for ln in r.stdout.splitlines() if ln.strip().startswith("{")]
    tg = [row for row in rows if row.get("n_gen", 0) > 0]
    if not tg:
        return None, f"no tg row in llama-bench jsonl output: {r.stdout[:300]}"
    row = tg[-1]
    return float(row["avg_ts"]), {"stddev_ts": row.get("stddev_ts"),
                                  "samples": row.get("samples_ts")}


def one_side(fn, *args):
    """Run one side of a cycle bracketed by box samples."""
    before = box_sample()
    t0 = time.time()
    value, detail = fn(*args)
    return {
        "tok_s": value,
        "detail": detail if value is not None else None,
        "error": None if value is not None else detail,
        "wall_s": round(time.time() - t0, 2),
        "sidecar": sidecar_delta(before, box_sample()),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="GGUF both sides run (must be the SAME file)")
    ap.add_argument("--stoke-exe",
                    default=os.path.join(REPO, "build", "win", "stoke.exe"))
    ap.add_argument("--llama-bench",
                    default=os.path.join(REPO, "bench", "vendor",
                                         "llama-b10037", "llama-bench.exe"))
    ap.add_argument("--cycles", type=int, default=3,
                    help="A,B cycles (default 3; the ruling's K3 exit claim "
                         "needs at least 3)")
    ap.add_argument("--tokens", type=int, default=32,
                    help="tokens generated per side, per cycle (default 32)")
    ap.add_argument("--affinity", default="0x1",
                    help="logical-processor mask BOTH sides are pinned to "
                         "(default 0x1)")
    ap.add_argument("--threads", type=int, default=1,
                    help="llama threads; stoke is single-threaded until K5")
    ap.add_argument("--llama-reps", type=int, default=1,
                    help="llama-bench -r within one cycle (default 1: cycles "
                         "are the repetition unit here, not -r)")
    ap.add_argument("--prompt", default="Hi",
                    help="stoke prompt; kept short so prefill is a small "
                         "share of the run (it is excluded from the rate "
                         "either way)")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="write the paired record here as JSON")
    a = ap.parse_args()

    for path, what in ((a.model, "model"), (a.stoke_exe, "stoke exe"),
                       (a.llama_bench, "llama-bench")):
        if not os.path.exists(path):
            ap.error(f"{what} not found: {path}")
    if a.cycles < 1:
        ap.error("--cycles must be >= 1")

    affinity = int(a.affinity, 16) if a.affinity.startswith("0x") else int(a.affinity)

    print(f"interleaving {a.cycles} cycle(s), {a.tokens} tok/side, "
          f"affinity {hex(affinity)}, model {os.path.basename(a.model)}")

    cycles = []
    for i in range(a.cycles):
        # A then B, every cycle, in the same order: the ratio is taken within
        # a cycle, so a monotonic drift across the session shifts both sides
        # of every ratio together instead of accumulating into one of them.
        s = one_side(run_stoke, a.stoke_exe, a.model, a.prompt, a.tokens,
                     affinity)
        l = one_side(run_llama, a.llama_bench, a.model, a.tokens, affinity,
                     a.llama_reps, a.threads)
        ratio = (l["tok_s"] / s["tok_s"]
                 if s["tok_s"] and l["tok_s"] and s["tok_s"] > 0 else None)
        cycles.append({"cycle": i + 1, "stoke": s, "llama": l,
                       "llama_over_stoke": round(ratio, 4) if ratio else None})
        st = f"{s['tok_s']:.3f}" if s["tok_s"] else f"FAILED ({s['error']})"
        lt = f"{l['tok_s']:.3f}" if l["tok_s"] else f"FAILED ({l['error']})"
        rt = f"{ratio:.3f}x" if ratio else "n/a"
        print(f"  cycle {i + 1}/{a.cycles}: stoke {st} | llama {lt} | {rt}")

    ratios = [c["llama_over_stoke"] for c in cycles
              if c["llama_over_stoke"] is not None]
    sidecars = [c[side]["sidecar"] for c in cycles for side in ("stoke", "llama")]
    grade, reasons = box_grade(sidecars, os.path.getsize(a.model),
                               os.cpu_count())

    record = {
        "kind": "referent_interleave",
        "protocol": PROTOCOL,
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": a.label,
        "git": git_provenance(),
        "config": {
            "model": os.path.basename(a.model),
            "model_size_bytes": os.path.getsize(a.model),
            "tokens_per_side": a.tokens,
            "affinity": hex(affinity),
            "threads": a.threads,
            "stoke_sha256": sha256_of(a.stoke_exe),
            "llama_build": os.path.basename(os.path.dirname(a.llama_bench)),
        },
        # THE FIGURE. A median of within-cycle ratios -- never a ratio of
        # medians, which would silently re-admit the cross-window comparison
        # by pairing a stoke number from one cycle with a llama number from
        # another.
        "llama_over_stoke_median": (round(statistics.median(ratios), 4)
                                    if ratios else None),
        "llama_over_stoke_cycles": ratios,
        # Spread across cycles is the honesty check on the ratio itself: it
        # says how much the RATIO moved while the box moved under it.
        "ratio_rel_spread": (round((max(ratios) - min(ratios)) / min(ratios), 4)
                             if len(ratios) > 1 and min(ratios) > 0 else 0.0),
        "box": {"platform": sys.platform, "logical_cpus": os.cpu_count(),
                "grade": grade, "grade_reasons": reasons},
        "cycles": cycles,
    }

    if not ratios:
        print("\nNO CYCLE PRODUCED A RATIO -- nothing here is a measurement.")
    else:
        print(f"\nllama/stoke = {record['llama_over_stoke_median']:.3f}x "
              f"(median of {len(ratios)} within-cycle ratios, "
              f"spread {record['ratio_rel_spread'] * 100:.1f}%)")
        if grade == "gate":
            print("box grade: GATE -- quiesced by measurement.")
        else:
            print("box grade: AMBIENT -- this record describes a loaded box "
                  "and is not comparable to a quiesced one:")
            for r in reasons:
                print(f"  - {r}")
        if a.cycles < 3:
            print(f"NOTE: {a.cycles} cycle(s). The ruling's K3 exit claim "
                  f"requires at least 3.")

    if a.out:
        with open(a.out, "w", encoding="ascii", newline="\n") as fh:
            json.dump(record, fh, indent=2)
            fh.write("\n")
        print(f"\nrecord written to {a.out}")
    else:
        print("\n" + json.dumps(record, separators=(",", ":")))
    return 0 if ratios else 2


if __name__ == "__main__":
    sys.exit(main())
