#!/usr/bin/env python3
"""tests/tooling/classify.py -- the ruling's section 6 classification rule,
applied mechanically to one divergent position.

BINDING SOURCE: docs/superpowers/memos/2026-08-04-m3-gate-item-2-RULING-FABLE.md
section 6. This file implements that rule and nothing else; where a reader
thinks the code and the ruling differ, THE RULING WINS and this file is the
bug. Every deviation the implementation was forced into is recorded in
DEVIATIONS below rather than being silently resolved in code.

THE RULE. At the first divergent position, sides X and Y, X chose x, Y chose
y, x != y. With tau = tau(pair, band) from the frozen table:

  ARITHMETIC-CLASS  iff max(g_X, g_Y) < tau  AND the top-5 sets agree after
                    discarding, on each side, candidates within tau of that
                    side's rank-5 logit.
  BUG               iff max(g_X, g_Y) >= 2*tau, or a chosen token was never
                    ranked by the other side (see DEVIATION 1).
  ESCALATE          otherwise -- the L2/L3 ladder decides, and the ladder,
                    not the constant, is what settles the borderline band.

where g_X = logit_X(x) - logit_X(y) and g_Y = logit_Y(y) - logit_Y(x), each
>= 0 by construction because each side chose its own argmax.

WHY THE FAIL-SAFE POINTS AT ESCALATE. The ruling sets no "smallest genuine bug
signal" because none has ever been measured. So this classifier NEVER passes a
position it cannot affirmatively classify: an unmeasurable quantity, an
out-of-range bound, a malformed probe -- all land in ESCALATE, which costs a
ladder run. Nothing here can turn "I could not tell" into ARITHMETIC-CLASS,
which is the only direction that could hide a defect behind a constant.

DEVIATIONS from a literal reading of the ruling's prose, each deliberate:

  1. "either side's chosen token sits more than tau below the other side's
     rank-5 logit" IS NOT OBSERVABLE from a top-5 probe when that token is
     absent from the other side's top-5 -- the probe reports five logits and
     the token is not one of them, so its logit is bounded (< rank-5) but not
     known. This implementation therefore returns ESCALATE, not BUG, for the
     unobservable case, UNLESS the bound alone already proves >= 2*tau (which
     is affirmative and is reported as BUG). Escalating cannot pass a bug; a
     guessed BUG would block a stamp on a number nobody measured. FLAGGED TO
     FABLE as G9.
  2. "the top-5 sets are equal after discarding ..." admits two readings:
     strict set equality of the two cores, or each core being contained in the
     other side's full top-5. This implements CONTAINMENT, because the ruling's
     own BUG rationale is "a candidate the other implementation never ranked",
     which is a containment test. Strict equality would additionally fail on
     pure boundary churn, which the tolerance exists to forgive. FLAGGED AS G10.

Both deviations are recorded in the output record, so a reader of a banked
classification sees which reading produced it.

CLI:
  python classify.py --records FILE.json --tau-table FILE.json [--out FILE]
  python classify.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys

CLASS = "ARITHMETIC-CLASS"
BUG = "BUG"
ESCALATE = "ESCALATE"

# Position bands (ruling section 8.9). Keyed by ABSOLUTE position =
# prompt_len + generated_index, because the error term the bands exist to
# track (rope_neox theta accumulation) grows with absolute position.
BANDS = ((0, 100), (100, 300), (300, 600))


def band_of(abs_position: int) -> str:
    """Name the band an absolute position falls in. Positions at or beyond the
    last band's end are NOT silently folded into it -- M7 adds bands there
    (ruling 8.9) and a Qwen3-era tau must not be stretched over an M7-era
    context by accident."""
    if abs_position < 0:
        raise ValueError(f"absolute position {abs_position} is negative")
    for lo, hi in BANDS:
        if lo <= abs_position < hi:
            return f"[{lo},{hi})"
    raise ValueError(
        f"absolute position {abs_position} is beyond the last frozen band "
        f"{BANDS[-1]}; ruling 8.9 requires a tau re-freeze before grading here")


def _as_pairs(top5, who):
    if not isinstance(top5, (list, tuple)) or len(top5) != 5:
        raise ValueError(f"{who}: expected exactly 5 (id, logit) pairs, got "
                         f"{len(top5) if hasattr(top5, '__len__') else type(top5)}")
    out = []
    for entry in top5:
        if len(entry) != 2:
            raise ValueError(f"{who}: malformed top-5 entry {entry!r}")
        out.append((int(entry[0]), float(entry[1])))
    # The probe contract on BOTH sides is rank-0 first, descending. A list that
    # is not sorted is a parse error, not something to sort into shape here.
    logits = [lg for _, lg in out]
    if any(logits[i] < logits[i + 1] for i in range(4)):
        raise ValueError(f"{who}: top-5 is not in descending logit order: {logits}")
    return out


def _margin(top5, chose_self, chose_other):
    """Return (margin, known). margin is g = logit(self) - logit(other) when the
    other side's token appears in this side's top-5, and otherwise a LOWER
    BOUND computed against rank-5 -- which is sound for proving BUG (a lower
    bound >= 2*tau proves the true margin is too) and insufficient for proving
    CLASS, which is exactly how it is used below."""
    table = dict(top5)
    self_lg = table[chose_self]
    if chose_other in table:
        return self_lg - table[chose_other], True
    return self_lg - top5[4][1], False


def _core(top5, tau):
    """Candidates far enough above this side's own rank-5 logit that noise of
    size tau could not have put them there. Rank-5 itself is never core."""
    cut = top5[4][1] + tau
    return {tid for tid, lg in top5 if lg > cut}


def classify(x_top5, x_chose, y_top5, y_chose, tau, x_name="X", y_name="Y"):
    """Apply the rule. Returns a dict; never raises for a merely surprising
    input, only for a malformed one."""
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    x5 = _as_pairs(x_top5, x_name)
    y5 = _as_pairs(y_top5, y_name)
    if x_chose != x5[0][0]:
        raise ValueError(f"{x_name} chose {x_chose} but its top-5 rank-0 is "
                         f"{x5[0][0]}; probe contract says rank-0 == argmax")
    if y_chose != y5[0][0]:
        raise ValueError(f"{y_name} chose {y_chose} but its top-5 rank-0 is "
                         f"{y5[0][0]}; probe contract says rank-0 == argmax")
    if x_chose == y_chose:
        raise ValueError(f"not a divergence: both sides chose {x_chose}")

    g_x, x_known = _margin(x5, x_chose, y_chose)
    g_y, y_known = _margin(y5, y_chose, x_chose)
    worst = max(g_x, g_y)

    x_core, y_core = _core(x5, tau), _core(y5, tau)
    x_ids, y_ids = {t for t, _ in x5}, {t for t, _ in y5}
    foreign = sorted((x_core - y_ids) | (y_core - x_ids))
    sets_agree = not foreign

    rec = {
        "tau": tau, "g_x": g_x, "g_y": g_y, "worst_margin": worst,
        "x_margin_known": x_known, "y_margin_known": y_known,
        "sets_agree": sets_agree, "foreign_core_candidates": foreign,
        "x_name": x_name, "y_name": y_name,
        "x_chose": x_chose, "y_chose": y_chose,
        "deviation_notes": [],
    }

    # BUG, affirmatively. Sound on bounds: a LOWER bound >= 2*tau proves the
    # true margin is >= 2*tau, whether or not the margin itself is known.
    if worst >= 2 * tau:
        rec["verdict"] = BUG
        rec["reason"] = (f"worst margin {worst:.6f} >= 2*tau ({2 * tau:.6f}); "
                         f"a perturbation of the measured class size cannot "
                         f"move a decision by this much")
        return rec

    # CLASS requires BOTH margins actually measured. A bound is not a margin.
    if x_known and y_known and worst < tau and sets_agree:
        rec["verdict"] = CLASS
        rec["reason"] = (f"worst margin {worst:.6f} < tau ({tau:.6f}) and the "
                         f"top-5 sets agree modulo the rank-5 tolerance")
        return rec

    rec["verdict"] = ESCALATE
    why = []
    if not (x_known and y_known):
        absent = []
        if not x_known:
            absent.append(f"{y_name}'s choice {y_chose} is absent from {x_name}'s top-5")
        if not y_known:
            absent.append(f"{x_name}'s choice {x_chose} is absent from {y_name}'s top-5")
        why.append("; ".join(absent) +
                   " -- 'more than tau below rank-5' is NOT observable from a "
                   "top-5 probe, so this is not affirmatively classifiable")
        rec["deviation_notes"].append("DEVIATION 1 (G9): unobservable -> ESCALATE, not BUG")
    if not sets_agree:
        why.append(f"core candidates {foreign} are not ranked by the other side "
                   f"(containment reading)")
        rec["deviation_notes"].append("DEVIATION 2 (G10): containment, not strict set equality")
    if x_known and y_known and sets_agree and tau <= worst < 2 * tau:
        why.append(f"worst margin {worst:.6f} lies in the borderline band "
                   f"[tau, 2*tau) = [{tau:.6f}, {2 * tau:.6f})")
    rec["reason"] = "; ".join(why) + " -- run the L2/L3 ladder: <=1e-3 and no STEP => CLASS, a STEP => BUG"
    return rec


def classify_records(records, tau_table):
    """records: list of dicts with prompt, abs_position, pair, and both sides.
    tau_table: {pair: {band: tau}}. A missing entry is fatal, never defaulted --
    grading against a tau nobody froze is the failure this whole rule prevents."""
    out = []
    for r in records:
        band = band_of(int(r["abs_position"]))
        pair = r["pair"]
        try:
            tau = float(tau_table[pair][band])
        except KeyError:
            raise SystemExit(f"fatal: no frozen tau for pair {pair!r} band {band}; "
                             f"freeze it from the probe BEFORE grading (ruling 2.3)")
        rec = classify(r["x"]["top5"], int(r["x"]["chose"]),
                       r["y"]["top5"], int(r["y"]["chose"]), tau,
                       r["x"].get("name", "X"), r["y"].get("name", "Y"))
        rec.update({"prompt": r["prompt"], "position": r.get("position"),
                    "abs_position": r["abs_position"], "band": band, "pair": pair})
        out.append(rec)
    return out


# ============================================================================
# selftest
# ============================================================================

def _t5(pairs):
    return [[i, l] for i, l in pairs]


def run_selftest() -> int:
    fails = []

    def check(label, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            fails.append(label)

    print("classify.py selftest")
    TAU = 0.10

    # --- the shape the M2 census actually recorded: a clean top-2 swap -------
    # 18 of 18 classifiable M2 divergences were exactly this, at gaps of
    # 0.001-0.204 nats. It must classify as CLASS at a tau above the gap.
    x = _t5([(438, 19.79), (714, 19.75), (22, 18.0), (33, 17.0), (44, 16.0)])
    y = _t5([(714, 19.78), (438, 19.76), (22, 18.0), (33, 17.0), (44, 16.0)])
    r = classify(x, 438, y, 714, TAU)
    check("clean top-2 swap inside tau -> ARITHMETIC-CLASS", r["verdict"] == CLASS)
    check("  and both margins were actually measured",
          r["x_margin_known"] and r["y_margin_known"])

    # --- a margin no class effect could produce -----------------------------
    x = _t5([(438, 25.0), (714, 19.0), (22, 18.0), (33, 17.0), (44, 16.0)])
    y = _t5([(714, 24.0), (438, 19.0), (22, 18.0), (33, 17.0), (44, 16.0)])
    r = classify(x, 438, y, 714, TAU)
    check("margin >= 2*tau -> BUG", r["verdict"] == BUG)

    # --- the borderline band belongs to the ladder, not to the constant -----
    x = _t5([(438, 19.94), (714, 19.79), (22, 18.0), (33, 17.0), (44, 16.0)])
    y = _t5([(714, 19.90), (438, 19.85), (22, 18.0), (33, 17.0), (44, 16.0)])
    r = classify(x, 438, y, 714, TAU)
    check("margin in [tau, 2*tau) -> ESCALATE", r["verdict"] == ESCALATE)
    check("  and the reason names the ladder", "ladder" in r["reason"])

    # --- unobservable: the other side never ranked the token ----------------
    # Bound is small, so BUG cannot be affirmed -> ESCALATE (DEVIATION 1).
    x = _t5([(438, 19.79), (22, 19.75), (33, 19.74), (44, 19.73), (55, 19.72)])
    y = _t5([(714, 19.78), (438, 19.77), (22, 19.0), (33, 18.0), (44, 17.0)])
    r = classify(x, 438, y, 714, TAU)
    check("chosen token absent from the other top-5, small bound -> ESCALATE",
          r["verdict"] == ESCALATE)
    check("  and it is recorded as unmeasured, not as a measured margin",
          r["x_margin_known"] is False)
    check("  and the deviation is disclosed in the record",
          any("G9" in n for n in r["deviation_notes"]))

    # --- unobservable, but the BOUND ALONE already proves BUG ---------------
    # X's rank-5 is 6.0 below its choice, so g_x >= 6.0 >= 2*tau regardless of
    # where the absent token actually sits. Affirmative on a bound.
    x = _t5([(438, 25.0), (22, 19.4), (33, 19.3), (44, 19.2), (55, 19.0)])
    y = _t5([(714, 19.78), (438, 19.77), (22, 19.0), (33, 18.0), (44, 17.0)])
    r = classify(x, 438, y, 714, TAU)
    check("absent token but lower bound >= 2*tau -> BUG on the bound",
          r["verdict"] == BUG)

    # --- a foreign core candidate is not boundary churn ---------------------
    # CONSTRUCTED CAREFULLY, and the first attempt was wrong in a way worth
    # recording: it dropped 714 out of X's top-5 too, which makes g_x a LOWER
    # BOUND of 3.79 and correctly trips BUG-on-the-bound before the set test is
    # ever reached -- so it passed for the wrong reason and proved nothing about
    # the set test. To isolate the set condition BOTH margins must stay known
    # and small, so 714 stays in X's list and only 999 is foreign.
    x = _t5([(438, 19.79), (714, 19.75), (999, 19.70), (22, 18.0), (33, 17.0)])
    y = _t5([(714, 19.78), (438, 19.76), (22, 18.0), (33, 17.0), (44, 16.0)])
    r = classify(x, 438, y, 714, TAU)
    check("core candidate the other side never ranked -> ESCALATE",
          r["verdict"] == ESCALATE and r["foreign_core_candidates"] == [999])
    check("  and it escalated on the SET test, both margins being known and small",
          r["x_margin_known"] and r["y_margin_known"] and r["worst_margin"] < TAU)

    # --- boundary churn IS forgiven, which is what the tolerance is for -----
    # 999 must sit STRICTLY INSIDE the tolerance band (rank5, rank5+tau] on X's
    # side: 17.02 against a rank-5 of 17.00 with tau 0.10. The obvious-looking
    # construction -- putting the odd candidate AT rank-5 -- proves nothing,
    # because `>` excludes a rank-5 candidate whether the tolerance is there or
    # not, so the case passes with the tolerance DELETED. Mutation M3 caught
    # exactly that; same family as the M0-era timer-floor exemption whose
    # matched pair sat under the threshold and would have passed with the
    # exemption removed.
    x = _t5([(438, 19.79), (714, 19.75), (22, 18.0), (999, 17.02), (33, 17.0)])
    y = _t5([(714, 19.78), (438, 19.76), (22, 18.0), (33, 17.0), (44, 16.9)])
    r = classify(x, 438, y, 714, TAU)
    check("rank-5 churn within tau is forgiven -> ARITHMETIC-CLASS",
          r["verdict"] == CLASS)
    check("  and the forgiven candidate really was inside the tolerance band",
          17.00 < 17.02 <= 17.00 + TAU)

    # --- tau scales the verdict, which is what makes freezing it load-bearing
    x = _t5([(438, 19.79), (714, 19.75), (22, 18.0), (33, 17.0), (44, 16.0)])
    y = _t5([(714, 19.78), (438, 19.76), (22, 18.0), (33, 17.0), (44, 16.0)])
    check("the SAME position flips to BUG at a 50x smaller tau",
          classify(x, 438, y, 714, 0.002)["verdict"] == BUG)

    # --- malformed input is refused, never coerced --------------------------
    for label, fn in (
        ("non-divergent position refused",
         lambda: classify(x, 438, x, 438, TAU)),
        ("rank-0 disagreeing with the stated choice refused",
         lambda: classify(x, 714, y, 714, TAU)),
        ("a top-5 that is not descending refused",
         lambda: classify(_t5([(1, 1.0), (2, 5.0), (3, 0.0), (4, -1.0), (5, -2.0)]),
                          1, y, 714, TAU)),
        ("a top-4 refused", lambda: classify(x[:4], 438, y, 714, TAU)),
        ("tau <= 0 refused", lambda: classify(x, 438, y, 714, 0.0)),
    ):
        try:
            fn()
            check(label, False)
        except ValueError:
            check(label, True)

    # --- band assignment ----------------------------------------------------
    check("band [0,100)", band_of(87) == "[0,100)")
    check("band [100,300)", band_of(100) == "[100,300)")
    check("band [300,600)", band_of(553) == "[300,600)")
    try:
        band_of(1200)
        check("a position past the frozen bands is refused, not folded in", False)
    except ValueError:
        check("a position past the frozen bands is refused, not folded in", True)

    # --- a tau that was never frozen is fatal, not defaulted ----------------
    rec = {"prompt": "p02", "abs_position": 39, "pair": "f32_intcore",
           "x": {"top5": x, "chose": 438}, "y": {"top5": y, "chose": 714}}
    try:
        classify_records([rec], {"some_other_pair": {"[0,100)": 0.1}})
        check("an unfrozen tau is fatal", False)
    except SystemExit:
        check("an unfrozen tau is fatal", True)
    got = classify_records([rec], {"f32_intcore": {"[0,100)": TAU}})
    check("a frozen tau grades and carries its band", got[0]["band"] == "[0,100)")

    print(f"\n{'ALL GREEN' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--records")
    p.add_argument("--tau-table")
    p.add_argument("--out")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return run_selftest()
    if not (a.records and a.tau_table):
        print("fatal: --records and --tau-table are both required", file=sys.stderr)
        return 2
    records = json.load(open(a.records, encoding="utf-8"))
    tau_table = json.load(open(a.tau_table, encoding="utf-8"))
    out = classify_records(records, tau_table)
    counts = {}
    for r in out:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    for r in out:
        print(f"{r['prompt']} pos={r['position']} abs={r['abs_position']} "
              f"{r['band']} {r['verdict']}: {r['reason']}")
    print(f"\n{counts}")
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    # A BUG anywhere is a hard block (ruling 7.3-B); ESCALATE is unfinished
    # work, not a pass, so neither may exit 0.
    return 1 if (counts.get(BUG) or counts.get(ESCALATE)) else 0


if __name__ == "__main__":
    sys.exit(main())
