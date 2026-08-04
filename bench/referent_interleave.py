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

TWO MODES, ONE PROTOCOL.

  stoke-vs-llama  (default)  -- the K3 comparison this tool was built for:
      both sides run THE SAME FILE, and what varies is the engine.

  llama-vs-llama  (--mode)   -- the K4-entry measurement: both sides are the
      SAME ENGINE and what varies is the QUANT FILE. It answers "what does a
      K-quant actually cost to unpack", which is the last unmeasured factor
      in M3's bar projection.

The pinning, `-p 0`, A,B,A,B interleaving and median-of-within-cycle-ratios
are identical in both, so PROTOCOL does not move -- a mode is not a different
discipline. But note what DOES relax: the header rule above says "on the same
file", and in llama-vs-llama the file is deliberately the variable. That is
sound only because the engine is then held fixed; a record that varied BOTH
would measure nothing. The record carries `mode` so a reader never has to
infer which of the two it is holding.
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


def run_cycles(n_cycles, side_a, side_b, name_a, name_b, ratio_key):
    """Run `n_cycles` of A,B and return (cycles, ratios).

    A then B, every cycle, in the same order: the ratio is taken WITHIN a
    cycle, so a monotonic drift across the session shifts both sides of every
    ratio together instead of accumulating into one of them.

    `side_a`/`side_b` are (fn, *args) tuples for `one_side`. Both modes go
    through here so neither can quietly acquire a different discipline.
    """
    cycles, ratios = [], []
    for i in range(n_cycles):
        a = one_side(*side_a)
        b = one_side(*side_b)
        ratio = (b["tok_s"] / a["tok_s"]
                 if a["tok_s"] and b["tok_s"] and a["tok_s"] > 0 else None)
        if ratio is not None:
            ratios.append(round(ratio, 4))
        cycles.append({"cycle": i + 1, name_a: a, name_b: b,
                       ratio_key: round(ratio, 4) if ratio else None})
        at = f"{a['tok_s']:.3f}" if a["tok_s"] else f"FAILED ({a['error']})"
        bt = f"{b['tok_s']:.3f}" if b["tok_s"] else f"FAILED ({b['error']})"
        rt = f"{ratio:.3f}x" if ratio else "n/a"
        print(f"  cycle {i + 1}/{n_cycles}: {name_a} {at} | "
              f"{name_b} {bt} | {rt}")
    return cycles, ratios


def bandwidth_win(size_a_bytes, size_b_bytes, measured_ratio):
    """What fraction of a smaller file's bandwidth advantage actually arrives.

    A decode step at batch 1 re-reads every weight, so an engine that were
    purely bandwidth-bound would speed up by exactly `size_a/size_b` when the
    file shrinks -- that is the win on offer, for free, before any kernel runs.
    A K-quant does not collect all of it: the superblock scale/min unpack is
    real work the Q8_0 path does not do.

    Returns (size_ratio, retained), retained = measured / size_ratio. 1.0 means
    the smaller file was free; 0.8 means a fifth of the offered win was spent
    unpacking it. Values ABOVE 1.0 are not "better than free" -- they mean the
    baseline side was not bandwidth-bound either, and the number stops being
    readable as a K-quant cost.
    """
    if not size_a_bytes or not size_b_bytes or size_b_bytes <= 0:
        return None, None
    size_ratio = size_a_bytes / float(size_b_bytes)
    if measured_ratio is None or size_ratio <= 0:
        return round(size_ratio, 4), None
    return round(size_ratio, 4), round(measured_ratio / size_ratio, 4)


def selftest():
    """Check the two pieces of arithmetic a reader would otherwise take on
    trust. Both have a failure mode that is silent and directional."""
    fails = []

    def check(name, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            fails.append(name)

    # A purely bandwidth-bound engine collects the whole size ratio.
    sr, ret = bandwidth_win(4_280_405_216, 2_497_280_736, 1.714)
    check("size ratio of the two real model files is ~1.714", abs(sr - 1.7141) < 0.001)
    check("a measured ratio equal to the size ratio retains ~1.0",
          abs(ret - 1.0) < 0.001)

    # The direction that matters: a shortfall must read BELOW 1.0.
    _, ret_short = bandwidth_win(4_280_405_216, 2_497_280_736, 1.40)
    check("a measured ratio below the size ratio retains < 1.0", ret_short < 1.0)
    check("and it is the honest quotient, not a difference",
          abs(ret_short - 1.40 / 1.7141) < 0.001)

    # Degenerate inputs must yield None, never a number a reader would cite.
    check("zero baseline size yields no figure", bandwidth_win(0, 1, 1.0) == (None, None))
    check("a missing measured ratio yields a size ratio but no retention",
          bandwidth_win(4, 2, None) == (2.0, None))

    # The ratio discipline: median of within-cycle ratios, never a ratio of
    # medians. Constructed so the two DISAGREE -- if they agreed, a regression
    # here would pass while the tool did the forbidden thing.
    a_vals, b_vals = [10.0, 20.0, 30.0], [20.0, 60.0, 30.0]
    within = statistics.median([b / a for a, b in zip(a_vals, b_vals)])
    of_medians = statistics.median(b_vals) / statistics.median(a_vals)
    check("the two methods differ on this fixture (else it proves nothing)",
          abs(within - of_medians) > 0.1)
    check("within-cycle median is the 2.0 one", abs(within - 2.0) < 1e-9)

    print("SELFTEST: " + ("ALL GREEN" if not fails
                          else f"{len(fails)} FAILED"))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("stoke-vs-llama", "llama-vs-llama"),
                    default="stoke-vs-llama",
                    help="what varies between the two sides: the ENGINE "
                         "(default, same file both sides) or the QUANT FILE "
                         "(llama both sides, --model-b required)")
    ap.add_argument("--model",
                    help="GGUF both sides run in stoke-vs-llama mode; the "
                         "BASELINE file in llama-vs-llama mode")
    ap.add_argument("--model-b",
                    help="llama-vs-llama only: the CANDIDATE file, measured "
                         "against --model")
    ap.add_argument("--selftest", action="store_true",
                    help="check this tool's own arithmetic and exit")
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

    if a.selftest:
        return selftest()

    if not a.model:
        ap.error("--model is required")
    needed = [(a.model, "model"), (a.llama_bench, "llama-bench")]
    if a.mode == "stoke-vs-llama":
        needed.append((a.stoke_exe, "stoke exe"))
        if a.model_b:
            ap.error("--model-b is only meaningful in --mode llama-vs-llama")
    else:
        if not a.model_b:
            ap.error("--mode llama-vs-llama requires --model-b")
        needed.append((a.model_b, "model-b"))
    for path, what in needed:
        if not os.path.exists(path):
            ap.error(f"{what} not found: {path}")
    if a.cycles < 1:
        ap.error("--cycles must be >= 1")

    affinity = int(a.affinity, 16) if a.affinity.startswith("0x") else int(a.affinity)
    size_a = os.path.getsize(a.model)

    if a.mode == "stoke-vs-llama":
        name_a, name_b, ratio_key = "stoke", "llama", "llama_over_stoke"
        side_a = (run_stoke, a.stoke_exe, a.model, a.prompt, a.tokens, affinity)
        side_b = (run_llama, a.llama_bench, a.model, a.tokens, affinity,
                  a.llama_reps, a.threads)
        size_b = size_a
        config = {
            "model": os.path.basename(a.model),
            "model_size_bytes": size_a,
            "tokens_per_side": a.tokens,
            "affinity": hex(affinity),
            "threads": a.threads,
            "stoke_sha256": sha256_of(a.stoke_exe),
            "llama_build": os.path.basename(os.path.dirname(a.llama_bench)),
        }
        what = f"model {os.path.basename(a.model)}"
    else:
        name_a, name_b = "baseline", "candidate"
        ratio_key = "candidate_over_baseline"
        side_a = (run_llama, a.llama_bench, a.model, a.tokens, affinity,
                  a.llama_reps, a.threads)
        side_b = (run_llama, a.llama_bench, a.model_b, a.tokens, affinity,
                  a.llama_reps, a.threads)
        size_b = os.path.getsize(a.model_b)
        config = {
            "baseline_model": os.path.basename(a.model),
            "baseline_size_bytes": size_a,
            "candidate_model": os.path.basename(a.model_b),
            "candidate_size_bytes": size_b,
            "tokens_per_side": a.tokens,
            "affinity": hex(affinity),
            "threads": a.threads,
            "llama_build": os.path.basename(os.path.dirname(a.llama_bench)),
        }
        what = (f"{os.path.basename(a.model)} -> "
                f"{os.path.basename(a.model_b)}")

    print(f"interleaving {a.cycles} cycle(s), {a.tokens} tok/side, "
          f"affinity {hex(affinity)}, threads {a.threads}, {what}")

    cycles, ratios = run_cycles(a.cycles, side_a, side_b, name_a, name_b,
                                ratio_key)

    sidecars = [c[side]["sidecar"] for c in cycles for side in (name_a, name_b)]
    # The RAM bar is set by the larger file: a row is only as comparable as
    # its most demanding side.
    grade, reasons = box_grade(sidecars, max(size_a, size_b), os.cpu_count())
    median_ratio = round(statistics.median(ratios), 4) if ratios else None

    record = {
        "kind": "referent_interleave",
        "protocol": PROTOCOL,
        "mode": a.mode,
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": a.label,
        "git": git_provenance(),
        "config": config,
        # THE FIGURE. A median of within-cycle ratios -- never a ratio of
        # medians, which would silently re-admit the cross-window comparison
        # by pairing one side's number from one cycle with the other's from
        # another.
        f"{ratio_key}_median": median_ratio,
        f"{ratio_key}_cycles": ratios,
        # Spread across cycles is the honesty check on the ratio itself: it
        # says how much the RATIO moved while the box moved under it.
        "ratio_rel_spread": (round((max(ratios) - min(ratios)) / min(ratios), 4)
                             if len(ratios) > 1 and min(ratios) > 0 else 0.0),
        "box": {"platform": sys.platform, "logical_cpus": os.cpu_count(),
                "grade": grade, "grade_reasons": reasons},
        "cycles": cycles,
    }

    if a.mode == "llama-vs-llama":
        size_ratio, retained = bandwidth_win(size_a, size_b, median_ratio)
        record["size_ratio"] = size_ratio
        record["bandwidth_win_retained"] = retained

    if not ratios:
        print("\nNO CYCLE PRODUCED A RATIO -- nothing here is a measurement.")
    else:
        print(f"\n{name_b}/{name_a} = {median_ratio:.3f}x "
              f"(median of {len(ratios)} within-cycle ratios, "
              f"spread {record['ratio_rel_spread'] * 100:.1f}%)")
        if a.mode == "llama-vs-llama":
            print(f"size ratio {record['size_ratio']:.3f}x on offer; "
                  f"{record['bandwidth_win_retained'] * 100:.1f}% of it "
                  f"retained after the unpack cost")
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
