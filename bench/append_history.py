#!/usr/bin/env python3
"""bench/append_history.py -- append one row to bench/history.jsonl.

WHY THIS EXISTS. `bench/history.jsonl` is a standing requirement, not an M3
convenience: the Phase-1 spec ("Perf regression tracking") and the M3-M7
forward note ("history.jsonl discipline") both say every green commit touching
a hot path appends a row, and that regressions are caught by DIFFING HISTORY
rather than by remembering. A file written at that cadence has to be produced
mechanically -- a format a human retypes by hand drifts within a week.

TWO DESIGN DECISIONS, both forced by what M2 and K1 measured rather than by
taste. They are the whole reason this is a tool and not an echo command.

1. THE ROW STORES ITS OWN REFERENCE STREAM -- BUT THE STABILITY GUARD JUDGES
   ABSOLUTES, AND THAT ORDERING IS THE OPPOSITE OF WHAT WAS FIRST ASSUMED.
   Storing the reference stream beside every absolute figure is what lets two
   rows taken weeks apart be RE-NORMALIZED against each other, and %ref really
   is the right lens for comparing a LOADED run to a QUIESCED one, where the
   whole box is uniformly slower and numerator and denominator move together.

   It is NOT a run-to-run noise reducer, and measurement on 2026-08-02 said so
   plainly: over five runs, %ref was LESS stable than the absolute median on
   8 of 9 measurable ops (ffn_gate 33.3% vs 22.8%, attn_q 27.2% vs 20.7%).
   The reason is structural -- the reference stream is measured ONCE, at the
   start of the run, while each op is measured later, so the ratio only
   cancels box state that persists across the entire run. Against run-to-run
   jitter the denominator is simply a second noisy measurement, and dividing
   by it ADDS its variance to every op instead of removing it. So the guard
   below is applied to absolute medians on purpose.

2. A ROW IS A MEDIAN ACROSS RUNS, NOT WITHIN ONE RUN, AND STABILITY IS
   ENFORCED RATHER THAN HOPED FOR. A single bench run cannot self-diagnose
   sustained interference: on 2026-08-02 a run reported LOGITS at 463.8 ms
   (2.6x wrong) while its own max/med ratio was 1.33, which the table's footer
   reads as HEALTHY. The spread heuristic inside one run catches TRANSIENT
   interference and is blind to SUSTAINED interference -- and sustained is the
   kind that moves a median. So the M0 protocol's "median of 3" is implemented
   here as three separate RUNS whose medians must agree, and disagreement is a
   REFUSAL to write: a box that cannot measure consistently right now must not
   silently deposit a number that a future regression hunt will trust.

WHAT THIS TOOL DELIBERATELY DOES NOT DO. It does not derive tok/s. That needs
the model's layer count and turns a recording tool into one that must also
know architecture; the arithmetic belongs in the ledger where its assumptions
are visible. It also does not hash the multi-GB model file -- name plus byte
size distinguishes every model this project uses, and a 20-second hash on
every append is a cost with no reader.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY = os.path.join(REPO, "bench", "history.jsonl")

SCHEMA = 1

# Ops below this median cost sit on the timer's resolution floor (the ~1 us
# primitives), where a 10% run-to-run difference is the clock, not the box.
# Stability is therefore judged only on ops large enough to be measured --
# which is also exactly the set the attribution ladder reads.
STABILITY_MIN_US = 100.0

# Cross-run spread above this means the box is not fit to measure right now.
# Calibrated against real data: the post-fix LOGITS spread was 8%, the
# pre-fix one 162%. 20% admits the former and refuses the latter.
STABILITY_MAX_REL = 0.20


# ---------------------------------------------------------------- pure logic


def cross_run(values):
    """Median/min/max of one op's per-run medians. Pure; see --selftest."""
    if not values:
        raise ValueError("cross_run: no values")
    s = sorted(values)
    return {
        "med": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "runs": list(values),
    }


def rel_spread(values):
    """(max-min)/min, the run-to-run disagreement. 0.0 for a single run.

    Relative to MIN rather than to the median: the question being asked is
    "how much slower did the worst run look than the best", and anchoring on
    the fastest observation is the conservative reading of interference.
    """
    if len(values) < 2:
        return 0.0
    lo, hi = min(values), max(values)
    if lo <= 0:
        return float("inf")
    return (hi - lo) / lo


def stability_violations(per_op, min_us=STABILITY_MIN_US,
                         max_rel=STABILITY_MAX_REL):
    """Ops whose cross-run medians disagree too much to be worth recording.

    Returns a list of (op, spread) sorted worst-first. Empty means the box
    measured consistently. Ops cheaper than `min_us` are exempt, with the
    reason given at STABILITY_MIN_US.
    """
    out = []
    for op, values in per_op.items():
        if min(values) < min_us:
            continue
        s = rel_spread(values)
        if s > max_rel:
            out.append((op, s))
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out


# ------------------------------------------------------------- provenance


def _run(cmd, cwd=REPO):
    """Best-effort command capture. Provenance must never fail the append."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def git_provenance():
    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    tag = _run(["git", "describe", "--tags", "--exact-match", "HEAD"])
    return {
        "commit": commit,
        # A row recorded from a dirty tree is not reproducible from the commit
        # it names, so the flag is part of the row rather than a warning that
        # scrolls away.
        "dirty": bool(dirty) if dirty is not None else None,
        "tag": tag,
    }


def sha256_of(path, limit=None):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and fh.tell() >= limit:
                break
    return h.hexdigest()


def free_ram_mb():
    """Best-effort, None when unavailable. Box state explains an odd row."""
    try:
        if sys.platform == "win32":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_OperatingSystem)"
                        ".FreePhysicalMemory"])
            return round(int(out) / 1024.0) if out else None
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024.0)
    except Exception:
        return None
    return None


# ------------------------------------------------------------------ driving


def one_run(exe, model, extra_args):
    """Run the bench once and return its parsed final JSON line.

    The JSON is the LAST line by construction (bench.c emit_jsonl), so the
    human table above it is ignored rather than parsed -- no coupling to a
    format meant for eyes.
    """
    cmd = [exe, "bench", "--ops", model, "--jsonl"] + list(extra_args)
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            f"bench failed (rc={r.returncode}):\n{r.stderr.strip()[:2000]}")
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("bench produced no output")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"bench's final line is not JSON ({e}). Was it built with the "
            f"--jsonl support? Line was:\n{lines[-1][:400]}")


def build_row(runs, label, milestone, exe, model):
    per_op = {}
    per_op_meta = {}
    for r in runs:
        for op in r["ops"]:
            per_op.setdefault(op["op"], []).append(op["med_us"])
            per_op_meta[op["op"]] = op
    ref_runs = [r["ref_gbs"] for r in runs]

    ops_out = {}
    for op, values in per_op.items():
        meta = per_op_meta[op]
        c = cross_run(values)
        ops_out[op] = {
            "shape": meta["shape"],
            "bytes": meta["bytes"],
            "reps": meta["reps"],
            "med_us": round(c["med"], 3),
            "med_us_runs": [round(v, 3) for v in c["runs"]],
            "rel_spread": round(rel_spread(values), 4),
            # Recomputed against the CROSS-RUN reference median so this figure
            # and ref_gbs below are the same basis. Taking the last run's
            # pct_ref would silently mix bases.
            "pct_ref": None,
        }

    ref_med = statistics.median(ref_runs)
    for op, o in ops_out.items():
        gbs = (o["bytes"] / 1e9) / (o["med_us"] / 1e6) if o["med_us"] > 0 else 0.0
        o["gbs"] = round(gbs, 4)
        o["pct_ref"] = round(100.0 * gbs / ref_med, 3) if ref_med > 0 else None

    return {
        "schema": SCHEMA,
        "kind": "ops",
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "milestone": milestone,
        "label": label,
        "git": git_provenance(),
        "build": {
            "binary_sha256": sha256_of(exe),
            "binary_basename": os.path.basename(exe),
        },
        # Model IDENTITY, not a path: models/ is gitignored and a committed
        # file must not point at something absent from the repo.
        "model": {
            "name": os.path.basename(model),
            "size_bytes": os.path.getsize(model),
        },
        "box": {
            "platform": sys.platform,
            "free_ram_mb": free_ram_mb(),
        },
        "runs": len(runs),
        "ref_gbs": round(ref_med, 4),
        "ref_gbs_runs": [round(v, 4) for v in ref_runs],
        "ref_rel_spread": round(rel_spread(ref_runs), 4),
        "ops": ops_out,
    }


# ----------------------------------------------------------------- selftest


def selftest():
    checks, failed = 0, 0

    def ck(name, cond):
        nonlocal checks, failed
        checks += 1
        if not cond:
            failed += 1
            print(f"  FAIL {name}")
        else:
            print(f"  ok   {name}")

    c = cross_run([3.0, 1.0, 2.0])
    ck("cross_run median of 3", c["med"] == 2.0)
    ck("cross_run min/max", c["min"] == 1.0 and c["max"] == 3.0)
    ck("cross_run preserves run order", c["runs"] == [3.0, 1.0, 2.0])

    ck("rel_spread single run is 0", rel_spread([5.0]) == 0.0)
    ck("rel_spread equal runs is 0", rel_spread([5.0, 5.0, 5.0]) == 0.0)
    ck("rel_spread 8pct", abs(rel_spread([159.5, 162.8, 172.3]) - 0.0802) < 1e-3)

    # THE REAL DATA THAT MOTIVATED THE GUARD. Pre-fix LOGITS must be refused,
    # post-fix must be admitted -- if this check ever inverts, the guard has
    # stopped doing the one job it was calibrated for.
    pre = {"matvec LOGITS": [193.8e3, 463.8e3, 176.8e3]}
    post = {"matvec LOGITS": [162.8e3, 159.5e3, 172.3e3]}
    ck("guard REFUSES the pre-fix 2.6x swing", len(stability_violations(pre)) == 1)
    ck("guard ADMITS the post-fix 8pct spread", stability_violations(post) == [])

    # A MATCHED PAIR: identical 33% spread, opposite verdicts, and the ONLY
    # difference is op size. Both must be well ABOVE STABILITY_MAX_REL or the
    # pair proves nothing -- an earlier version of this check used 11% for
    # both, so the "exempt" case passed merely for being under the threshold
    # and would have kept passing with the exemption deleted.
    floor = {"vec_dot_q8_0_f32": [0.9, 1.2, 0.9]}      # ~33%, sub-timer-floor
    big = {"matvec ffn_up": [9000.0, 12000.0, 9000.0]}  # ~33%, real op
    ck("both halves of the pair exceed the threshold",
       rel_spread([0.9, 1.2, 0.9]) > STABILITY_MAX_REL
       and rel_spread([9000.0, 12000.0, 9000.0]) > STABILITY_MAX_REL)
    ck("timer-floor op exempt despite 33pct spread",
       stability_violations(floor) == [])
    ck("large op with the SAME 33pct spread is NOT exempt",
       len(stability_violations(big)) == 1)

    ck("violations sorted worst-first",
       [op for op, _ in stability_violations(
           {"a": [100.0, 130.0], "b": [100.0, 200.0]})] == ["b", "a"])

    print(f"\nselftest: {checks - failed}/{checks} checks passed")
    return 1 if failed else 0


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path to the .gguf to bench")
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "win",
                                                  "stoke.exe"))
    ap.add_argument("--runs", type=int, default=3,
                    help="separate bench RUNS to median across (default 3)")
    ap.add_argument("--label", default="", help="what this row records")
    ap.add_argument("--milestone", default="M3")
    ap.add_argument("--bench-args", nargs="*", default=[],
                    help="extra args passed through to stoke bench")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the row, append nothing")
    ap.add_argument("--force", action="store_true",
                    help="append even if the stability guard fails")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if not a.model:
        ap.error("--model is required (or use --selftest)")
    if not os.path.exists(a.exe):
        ap.error(f"binary not found: {a.exe}")
    if not os.path.exists(a.model):
        ap.error(f"model not found: {a.model}")
    if a.runs < 1:
        ap.error("--runs must be >= 1")
    if a.runs < 3 and not a.force:
        ap.error("--runs < 3 cannot detect sustained interference (see this "
                 "file's header, decision 2). Use --force if you mean it.")

    print(f"running {a.runs} bench runs ...")
    runs = []
    for i in range(a.runs):
        r = one_run(a.exe, a.model, a.bench_args)
        runs.append(r)
        print(f"  run {i + 1}/{a.runs}: ref {r['ref_gbs']:.2f} GB/s, "
              f"{len(r['ops'])} ops")

    row = build_row(runs, a.label, a.milestone, a.exe, a.model)

    per_op = {op: o["med_us_runs"] for op, o in row["ops"].items()}
    violations = stability_violations(per_op)

    # The reference stream is checked separately and unconditionally: it is
    # the DENOMINATOR of every %ref in the row, so if it wobbles the whole
    # ratio basis wobbles with it, and no per-op exemption should apply.
    ref_spread = rel_spread(row["ref_gbs_runs"])
    if ref_spread > STABILITY_MAX_REL:
        violations.append(("reference stream", ref_spread))

    if violations:
        print("\nSTABILITY GUARD FAILED -- this box is not measuring "
              "consistently right now:")
        for op, s in violations:
            print(f"  {op:<26} {s * 100:6.1f}% spread across runs")
        print("\nA row written from these runs would be a number a future "
              "regression hunt trusts and should not.\nQuiesce the box and "
              "re-run, or pass --force if you accept it.")
        if not a.force:
            return 2

    line = json.dumps(row, separators=(",", ":"), sort_keys=False)
    if a.dry_run:
        print("\n--dry-run, nothing appended. Row would be:\n")
        print(json.dumps(row, indent=2))
        return 0

    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    with open(HISTORY, "a", encoding="ascii", newline="\n") as fh:
        fh.write(line + "\n")
    n = sum(1 for _ in open(HISTORY, encoding="ascii"))
    print(f"\nappended to bench/history.jsonl ({n} row(s) total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
