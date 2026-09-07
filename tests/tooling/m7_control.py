#!/usr/bin/env python3
"""tests/tooling/m7_control.py -- the instrument controls for M7's entry probe.

Compares two banked ref_forward.py probe files at their COMMON positions and
reports what separates them. It grades nothing about stoke; it grades the
REFERENCE, which is the thing the whole probe grid rests on.

Declared in
docs/superpowers/design-notes/2026-09-07-m7-entry-probe-preregistration.md §2.1-2.3:

  C1  serial(250) vs batched(250)          licenses the batched arm.
                                           T5 refuses at max anchored |Δ| >= 1e-3.
  C2  serial(40)  vs serial(250)@0..39     the instrument's own NOISE FLOOR.
                                           Expect BIT-IDENTICAL; anything else
                                           is a floor below which no Δ is
                                           readable, and it goes in the report.
  C3  batched(T=2915) vs batched(T=250)    whether the batched answer depends on
                                           the BATCH it was computed in. Causal
                                           attention means position p depends
                                           only on ids[0..p], so they MUST agree;
                                           only the GEMM shape differs.

▶ WHY A TOOL AND NOT A SCRIPT. This is the measurement that decides whether an
instrument 148x faster than the one of record may be used at all. A number that
licenses a 1040-probe grid is not something to compute in a shell one-liner and
retype.

▶ THE ANCHOR. Δ is reported both anchored (on side A's chosen id, the memo §3.6
method that m2_classify_probes.delta_profile implements) and raw. Anchored is
what tau is frozen from and therefore what T5 is stated in; raw is printed
beside it so a reader can see whether the distinction mattered here. Only ids
BOTH sides ranked can be differenced -- an id outside the other side's top-5 has
an unknown logit, not a large one.

⚠ IT ALSO REPORTS THE TREND, because "max over 250 positions" hides whether the
difference GROWS. The mechanism under test (BLAS blocking) scales with the
matrix, so a flat trend and a rising one license very different extrapolations.

CLI:
  python m7_control.py --a A.json --b B.json --label C1 [--threshold 1e-3]
  python m7_control.py --a A.json --b B.json --label C2 --require-exact
  python m7_control.py --selftest
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import m2_classify_probes as mcp            # noqa: E402
import classify as cls                      # noqa: E402

T5_THRESHOLD = 1e-3


def load(path):
    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return d, {int(p["pos"]): p for p in d["probes"]}


def _read_ids(path):
    txt = pathlib.Path(path).read_text(encoding="ascii")
    return [int(t) for t in txt.replace(",", " ").split() if t]


def compare(a_path, b_path, ids_a=None, ids_b=None):
    da, pa = load(a_path)
    db, pb = load(b_path)

    # ⚠⚠ THE STREAMS MUST BE THE SAME STREAM WHERE THEY OVERLAP, AND THAT IS
    # CHECKED, NOT ASSUMED. Two probe files over DIFFERENT ids compare position
    # by position perfectly happily and produce a beautifully small delta that
    # means nothing at all -- the failure mode is a licensed instrument and a
    # green report. C3 compares runs of three different LENGTHS, so a plain
    # sha256 equality cannot serve: the digest covers the whole id list and
    # differs by design when one run is a prefix of the other.
    #
    # So when the id files are given, the prefix relationship is verified
    # element by element over the compared range. When they are NOT given the
    # tool says so in the report rather than implying a check it did not make.
    common = sorted(set(pa) & set(pb))
    if not common:
        raise SystemExit("fatal: the two probe files share no position")

    ids_checked = False
    if ids_a and ids_b:
        A, B = _read_ids(ids_a), _read_ids(ids_b)
        if len(A) != da.get("n_ids") or len(B) != db.get("n_ids"):
            raise SystemExit(
                f"fatal: id file lengths {len(A)}/{len(B)} do not match the "
                f"probe files' n_ids {da.get('n_ids')}/{db.get('n_ids')} -- "
                f"the ids handed here are not the ids that were probed")
        n = max(common) + 1
        if len(A) < n or len(B) < n:
            raise SystemExit(f"fatal: a stream is shorter ({len(A)}/{len(B)}) "
                             f"than the compared range (0..{n - 1})")
        bad = [i for i in range(n) if A[i] != B[i]]
        if bad:
            raise SystemExit(
                f"fatal: the two streams differ at {len(bad)} of the first {n} "
                f"ids (first at index {bad[0]}: {A[bad[0]]} vs {B[bad[0]]}) -- "
                f"these probe files are not comparable")
        ids_checked = True

    rows = []
    n_argmax_diff = 0
    n_bitexact = 0
    for pos in common:
        A, B = pa[pos], pb[pos]
        if A["argmax"] != B["argmax"]:
            n_argmax_diff += 1
        ta = [tuple(t) for t in A["top5"]]
        tb = [tuple(t) for t in B["top5"]]
        if ta == tb:
            n_bitexact += 1
        prof = mcp.delta_profile(tb, ta, A["argmax"])
        rows.append({
            "pos": pos,
            "band": cls.band_of(pos),
            "argmax_a": A["argmax"], "argmax_b": B["argmax"],
            "bit_identical": ta == tb,
            "anchored_max": max(prof["anchored"]) if prof else None,
            "raw_max": max(prof["raw"]) if prof else None,
            "n_shared": len(prof["shared_ids"]) if prof else 0,
        })

    anchored = [r["anchored_max"] for r in rows if r["anchored_max"] is not None]
    raw = [r["raw_max"] for r in rows if r["raw_max"] is not None]
    unanchorable = sum(1 for r in rows if r["anchored_max"] is None)

    return {
        "a": {"file": str(a_path), "mode": da.get("mode"), "n_ids": da.get("n_ids"),
              "seconds": da.get("seconds"), "env": da.get("env")},
        "b": {"file": str(b_path), "mode": db.get("mode"), "n_ids": db.get("n_ids"),
              "seconds": db.get("seconds"), "env": db.get("env")},
        "ids_verified_identical_over_range": ids_checked,
        "common_positions": len(common),
        "position_range": [common[0], common[-1]],
        "argmax_disagreements": n_argmax_diff,
        "bit_identical_positions": n_bitexact,
        "unanchorable_positions": unanchorable,
        "anchored_max": max(anchored) if anchored else None,
        "anchored_p95": mcp.percentile(anchored, 95) if anchored else None,
        "raw_max": max(raw) if raw else None,
        "trend": _trend(rows),
        "rows": rows,
    }


def _trend(rows):
    """Max anchored |Δ| in each fifth of the probed range, in position order.

    Reported ALWAYS, not only when it looks alarming: a growth figure that
    appears only when someone is worried cannot be compared across sittings."""
    vals = [r for r in rows if r["anchored_max"] is not None]
    if len(vals) < 5:
        return None
    n = len(vals)
    out = []
    for i in range(5):
        chunk = vals[i * n // 5:(i + 1) * n // 5]
        if chunk:
            out.append({"positions": [chunk[0]["pos"], chunk[-1]["pos"]],
                        "anchored_max": max(c["anchored_max"] for c in chunk)})
    return out


def report(res, label, threshold, require_exact):
    print(f"=== {label} ===")
    print(f"  A: {res['a']['mode']} n_ids={res['a']['n_ids']} "
          f"{res['a']['seconds']}s  threads="
          f"{res['a']['env'].get('OPENBLAS_NUM_THREADS')} "
          f"affinity={res['a']['env'].get('affinity_mask')}")
    print(f"  B: {res['b']['mode']} n_ids={res['b']['n_ids']} "
          f"{res['b']['seconds']}s  threads="
          f"{res['b']['env'].get('OPENBLAS_NUM_THREADS')} "
          f"affinity={res['b']['env'].get('affinity_mask')}")
    print(f"  common positions: {res['common_positions']} "
          f"({res['position_range'][0]}..{res['position_range'][1]})")
    print(f"  ids verified identical over that range: "
          f"{'YES' if res['ids_verified_identical_over_range'] else 'NOT CHECKED (no --ids-a/--ids-b)'}")
    print(f"  argmax disagreements: {res['argmax_disagreements']}")
    print(f"  bit-identical top-5:  {res['bit_identical_positions']}"
          f"/{res['common_positions']}")
    if res["unanchorable_positions"]:
        print(f"  !! unanchorable positions (no shared chosen id): "
              f"{res['unanchorable_positions']}")
    print(f"  anchored |d|: max={res['anchored_max']!r} p95={res['anchored_p95']!r}")
    print(f"  raw      |d|: max={res['raw_max']!r}")
    if res["trend"]:
        print("  trend (max anchored |d| by fifth of the range):")
        for t in res["trend"]:
            print(f"    {t['positions'][0]:>5}..{t['positions'][1]:<5} "
                  f"{t['anchored_max']:.3e}")

    ok = True
    if require_exact:
        ok = (res["bit_identical_positions"] == res["common_positions"]
              and res["argmax_disagreements"] == 0)
        print(f"  VERDICT: {'BIT-IDENTICAL' if ok else 'NOT BIT-IDENTICAL'} "
              f"(required)")
    else:
        m = res["anchored_max"]
        ok = (m is not None and m < threshold and res["argmax_disagreements"] == 0)
        print(f"  VERDICT vs T5 ({threshold:g}): "
              f"{'PASS -- batched arm LICENSED' if ok else 'FAIL -- NOT licensed'}")
    return ok


def run_selftest() -> int:
    fails = []

    def check(label, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        if not cond:
            fails.append(label)

    check("T5 is the pre-registered 1e-3", T5_THRESHOLD == 1e-3)

    import tempfile
    def write(tmp, name, probes, mode="teacher-force/serial"):
        p = pathlib.Path(tmp) / name
        p.write_text(json.dumps({"mode": mode, "n_ids": len(probes), "seconds": 1.0,
                                 "env": {}, "probes": probes}), encoding="utf-8")
        return p

    t5 = [[1, 5.0], [2, 4.0], [3, 3.0], [4, 2.0], [5, 1.0]]
    same = [{"pos": i, "argmax": 1, "next_id": None, "top5": t5} for i in range(6)]
    with tempfile.TemporaryDirectory() as tmp:
        a = write(tmp, "a.json", same)
        b = write(tmp, "b.json", same)
        r = compare(a, b)
        check("identical files: zero delta, all bit-identical",
              r["anchored_max"] == 0.0
              and r["bit_identical_positions"] == r["common_positions"])
        check("identical files pass the exactness requirement",
              report(r, "selftest-exact", T5_THRESHOLD, True))

        # A perturbation must be SEEN. A control that cannot detect a difference
        # would report "licensed" for an instrument that is not.
        moved = [dict(p) for p in same]
        moved[3] = dict(moved[3])
        moved[3]["top5"] = [[1, 5.0], [2, 4.0 + 0.5], [3, 3.0], [4, 2.0], [5, 1.0]]
        c = write(tmp, "c.json", moved)
        r2 = compare(a, c)
        check("a 0.5 perturbation is detected", r2["anchored_max"] == 0.5)
        check("and it FAILS T5", not report(r2, "selftest-perturbed",
                                            T5_THRESHOLD, False))
        check("and it is not reported bit-identical",
              r2["bit_identical_positions"] == r2["common_positions"] - 1)

        # An argmax flip must fail even when the logit Δ is tiny -- the two
        # conditions are ANDed for a reason.
        flipped = [dict(p) for p in same]
        flipped[2] = dict(flipped[2])
        flipped[2]["argmax"] = 2
        flipped[2]["top5"] = [[2, 5.0], [1, 4.0], [3, 3.0], [4, 2.0], [5, 1.0]]
        d = write(tmp, "d.json", flipped)
        r3 = compare(a, d)
        check("an argmax flip fails T5 even at small delta",
              not report(r3, "selftest-flip", T5_THRESHOLD, False))

        # Disjoint positions must refuse rather than silently compare nothing.
        far = [{"pos": 900 + i, "argmax": 1, "next_id": None, "top5": t5}
               for i in range(3)]
        e = write(tmp, "e.json", far)
        try:
            compare(a, e)
            check("disjoint probe files are refused", False)
        except SystemExit:
            check("disjoint probe files are refused", True)

        # The stream guard must REFUSE a mismatch and ACCEPT a genuine prefix.
        # Both directions, because a guard that refuses everything is as useless
        # as one that accepts everything, and only one of those is obvious.
        ia = pathlib.Path(tmp) / "ia.txt"; ia.write_text("11 12 13 14 15 16", encoding="ascii")
        ib = pathlib.Path(tmp) / "ib.txt"; ib.write_text("11 12 99 14 15 16", encoding="ascii")
        try:
            compare(a, b, ia, ib)
            check("a stream mismatch inside the compared range is refused", False)
        except SystemExit as ex:
            check("a stream mismatch inside the compared range is refused",
                  "not comparable" in str(ex))
        ic = pathlib.Path(tmp) / "ic.txt"; ic.write_text("11 12 13 14 15 16", encoding="ascii")
        r4 = compare(a, b, ia, ic)
        check("identical streams verify and are marked as verified",
              r4["ids_verified_identical_over_range"] is True)
        check("without id files the report says NOT CHECKED, never implies one",
              compare(a, b)["ids_verified_identical_over_range"] is False)
        # An id file that is not the file that was probed must be refused too --
        # handing the tool the wrong ids is the same failure wearing a check.
        iw = pathlib.Path(tmp) / "iw.txt"; iw.write_text("11 12 13 14 15", encoding="ascii")
        try:
            compare(a, b, iw, ic)
            check("an id file whose length contradicts n_ids is refused", False)
        except SystemExit:
            check("an id file whose length contradicts n_ids is refused", True)

    print(f"\n{'ALL GREEN' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a")
    p.add_argument("--b")
    p.add_argument("--label", default="control")
    p.add_argument("--threshold", type=float, default=T5_THRESHOLD)
    p.add_argument("--require-exact", action="store_true")
    p.add_argument("--ids-a", help="id file behind --a; with --ids-b, verifies "
                                   "the two streams are identical over the "
                                   "compared range instead of assuming it")
    p.add_argument("--ids-b")
    p.add_argument("--out")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return run_selftest()
    if not args.a or not args.b:
        print("fatal: --a and --b are required", file=sys.stderr)
        return 1
    res = compare(args.a, args.b, args.ids_a, args.ids_b)
    ok = report(res, args.label, args.threshold, args.require_exact)
    if args.out:
        res["label"] = args.label
        res["threshold"] = args.threshold
        res["require_exact"] = args.require_exact
        res["passed"] = ok
        pathlib.Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
