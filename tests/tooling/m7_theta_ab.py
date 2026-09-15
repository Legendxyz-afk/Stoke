#!/usr/bin/env python3
"""tests/tooling/m7_theta_ab.py -- the 2026-09-08 rope-theta A/B readout.

QUESTION (gate report 2026-09-07-m7-entry-probe.md, left open there): the tau
freeze's band p95 rises 1.322x from [600,1200) into [1200,inf). Is that the
closed-vs-accum rope-theta difference between the reference and stoke, or is it
a longer-context population?

PROTOCOL: docs/superpowers/design-notes/2026-09-08-m7-rope-theta-ab-preregistration.md
-- committed before the arms were compared. The thresholds below are arithmetic
on numbers already banked in 2026-09-07-m7-entry-probe-tau.json.

WHAT IT IS NOT. It does not re-freeze tau. The banked freeze stands as the
graded criterion; picking whichever arm gives the smaller tolerance after
seeing both is the fit-to-result this project withdraws results for.

EVERYTHING SHARED IS IMPORTED: build_envelope and _band_growth from m7_probe
(the same construction the freeze used), percentile/delta_profile from
m2_classify_probes, band_of from classify.

CLI:
  python m7_theta_ab.py --closed DIR --accum DIR
        [--banked-tau JSON] [--null-a REF.json --null-b REF.json] [--out JSON]

⚠ ASCII only in every printed string: this console is cp1252 and a tau or a
delta in a report line is an unhandled exception, not a cosmetic problem.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import classify as cls                      # noqa: E402
import m2_classify_probes as mcp            # noqa: E402
import m7_probe as mp                       # noqa: E402

# The transition under examination. Named constants so a reader can see they
# were not chosen after the fact.
LO_BAND = "[600,1200)"
HI_BAND = "[1200,inf)"
EXPLAINED_AT = 0.50      # excess_removed >= this  -> EXPLAINED
REFUTED_BELOW = 0.10     # excess_removed <  this  -> REFUTED
R2_TOL = 1e-6            # closed arm must reproduce the banked p95 to this
BOUND_THRESHOLD = 0.0125  # a quarter of the 09-07 Llama freeze's rise (0.049911)
BOUND_QUARTER = 0.25     # --bound-threshold banked: this fraction of the
                         # banked rise, so a second pair gets ITS OWN quarter
                         # (2026-09-15, B6) instead of the Llama literal


class AbError(RuntimeError):
    pass


# ---------------------------------------------------------------------------


def grade(lo_closed, hi_closed, lo_accum, hi_accum):
    """The pre-registered rule, as one pure function so --selftest can drive it
    with numbers whose answer is known by hand."""
    g_closed = hi_closed / lo_closed
    g_accum = hi_accum / lo_accum
    if g_closed == 1.0:
        return g_closed, g_accum, None, "UNDEFINED (closed arm shows no rise)"
    excess = (g_closed - g_accum) / (g_closed - 1.0)
    if excess >= EXPLAINED_AT:
        v = "EXPLAINED"
    elif excess < REFUTED_BELOW:
        v = "REFUTED"
    else:
        v = "PARTIAL"
    return g_closed, g_accum, excess, v


def band_p95(envelope):
    """p95 of the anchored profile per band, over the whole envelope. The same
    aggregation m7_probe._band_growth performs, exposed per band rather than as
    a ratio, and computed with the same imported percentile()."""
    by_band = {}
    for s in envelope:
        by_band.setdefault(cls.band_of(int(s["abs_position"])), []).extend(
            s["profile"]["anchored"])
    return {b: {"n": len(v), "p95": mcp.percentile(v, 95), "max": max(v)}
            for b, v in by_band.items()}


def stoke_sides_identical(ga, gb):
    """R1. The two arms each ran stoke independently; if those runs differ, the
    p95 difference between the arms is not attributable to the ref switch."""
    diffs = []
    ia = {g["stream"]: g for g in ga}
    ib = {g["stream"]: g for g in gb}
    if set(ia) != set(ib):
        raise AbError(f"arms cover different streams: "
                      f"closed-only={sorted(set(ia) - set(ib))} "
                      f"accum-only={sorted(set(ib) - set(ia))}")
    for name in sorted(ia):
        pa, pb = ia[name]["positions"], ib[name]["positions"]
        if len(pa) != len(pb):
            diffs.append(f"{name}: {len(pa)} vs {len(pb)} probes")
            continue
        for x, y in zip(pa, pb):
            if x["abs_position"] != y["abs_position"]:
                diffs.append(f"{name}: position {x['abs_position']} vs "
                             f"{y['abs_position']}")
            elif [tuple(t) for t in x["stoke"]["top5"]] != \
                    [tuple(t) for t in y["stoke"]["top5"]]:
                diffs.append(f"{name} pos={x['abs_position']}: stoke top5 differs")
    return diffs


def ref_delta(ga, gb):
    """R4 / the bound. Per probed position, the largest |logit_closed -
    logit_accum| over the ids present in BOTH arms' ref top-5.

    An id in one top-5 and not the other has an UNKNOWN logit on the other
    side, not a large one -- the same asymmetry delta_profile enforces."""
    ia = {g["stream"]: g for g in ga}
    ib = {g["stream"]: g for g in gb}
    rows = []
    for name in sorted(ia):
        for x, y in zip(ia[name]["positions"], ib[name]["positions"]):
            dx = dict((int(i), float(v)) for i, v in x["ref"]["top5"])
            dy = dict((int(i), float(v)) for i, v in y["ref"]["top5"])
            shared = sorted(set(dx) & set(dy))
            d = max((abs(dx[i] - dy[i]) for i in shared), default=None)
            rows.append({"stream": name, "abs_position": x["abs_position"],
                         "band": cls.band_of(int(x["abs_position"])),
                         "shared_ids": len(shared), "delta": d,
                         "ref_chose_closed": x["ref"]["chose"],
                         "ref_chose_accum": y["ref"]["chose"]})
    return rows


def delta_by_band(rows):
    out = {}
    for r in rows:
        if r["delta"] is None:
            continue
        out.setdefault(r["band"], []).append(r)
    return {b: {"n": len(v),
                "max": max(r["delta"] for r in v),
                "p95": mcp.percentile([r["delta"] for r in v], 95),
                "median_position": sorted(r["abs_position"] for r in v)[len(v) // 2]}
            for b, v in out.items()}


def null_arm(path_a, path_b):
    """R3. Two `closed` runs of the same stream must agree in every probed
    logit -- bit-identical, not 'small'. C2 established this for the SERIAL
    path; the batched path at a fixed T is where these probes actually come
    from and had never been checked."""
    a = json.loads(pathlib.Path(path_a).read_text(encoding="utf-8"))
    b = json.loads(pathlib.Path(path_b).read_text(encoding="utf-8"))
    if a["theta_mode"] != "closed" or b["theta_mode"] != "closed":
        raise AbError(f"null arm needs two closed runs, got "
                      f"{a['theta_mode']}/{b['theta_mode']}")
    if a["ids_sha256"] != b["ids_sha256"]:
        raise AbError("null arm ran over different id streams -- the digests "
                      "differ, so a per-position comparison means nothing")
    pa = {p["pos"]: p["top5"] for p in a["probes"]}
    pb = {p["pos"]: p["top5"] for p in b["probes"]}
    if set(pa) != set(pb):
        raise AbError("null arm probed different positions")
    worst, n_diff = 0.0, 0
    for pos in sorted(pa):
        xa = dict((int(i), float(v)) for i, v in pa[pos])
        xb = dict((int(i), float(v)) for i, v in pb[pos])
        if sorted(xa) != sorted(xb):
            n_diff += 1
            continue
        for i in xa:
            d = abs(xa[i] - xb[i])
            if d > worst:
                worst = d
            if d != 0.0:
                n_diff += 1
                break
    return {"positions": len(pa), "positions_differing": n_diff,
            "max_abs_delta": worst,
            "bit_identical": (n_diff == 0 and worst == 0.0)}


# ---------------------------------------------------------------------------


def markdown(report) -> str:
    """The report's result section, GENERATED from the artifact.

    Same rule m7_report.py follows: a number that a tool published is never
    retyped into prose by hand. ⚠ Needs PYTHONIOENCODING=utf-8 when piped --
    this console is cp1252 and the text legitimately contains tau and Delta."""
    L = []
    g = report["growth"]
    L.append("| band | arm | n | p95 | max |")
    L.append("|---|---|---:|---:|---:|")
    order = [cls.band_name(lo, hi) for lo, hi in cls.BANDS]
    for b in sorted(set(report["closed"]["bands"]) | set(report["accum"]["bands"]),
                    key=order.index):
        for arm in ("closed", "accum"):
            r = report[arm]["bands"].get(b)
            if r is None:
                L.append(f"| `{b}` | {arm} | — | — | — |")
            else:
                L.append(f"| `{b}` | {arm} | {r['n']} | **{r['p95']:.6f}** | "
                         f"{r['max']:.6f} |")
    L.append("")
    if "closed" in g:
        L.append(f"| | growth `{LO_BAND}`→`{HI_BAND}` |")
        L.append("|---|---|")
        L.append(f"| closed (the ruled default, the freeze's arm) | "
                 f"**{g['closed']:.4f}×** |")
        L.append(f"| accum (the diagnostic arm) | **{g['accum']:.4f}×** |")
        L.append(f"| excess removed | **{g['excess_removed']:.4f}** |")
        L.append(f"| **verdict** | **{g['verdict']}** |")
        L.append("")
    L.append("Reference closed-vs-accum logit Δ (the bound):")
    L.append("")
    L.append("| band | positions | median position | max \\|Δ\\| | p95 \\|Δ\\| |")
    L.append("|---|---:|---:|---:|---:|")
    for b in sorted(report["delta"]["by_band"], key=order.index):
        r = report["delta"]["by_band"][b]
        L.append(f"| `{b}` | {r['n']} | {r['median_position']} | "
                 f"{r['max']:.6e} | {r['p95']:.6e} |")
    L.append("")
    L.append(f"Reference argmax flips between the arms: "
             f"**{report['delta']['ref_argmax_flips']} of "
             f"{report['delta']['n_positions']}** probed positions.")
    L.append("")
    bd = report["bound"]
    if bd["2x_delta_max_hi_band"] is not None:
        src = bd.get("threshold_source", "literal")
        L.append(f"2 × max Δ in `{HI_BAND}` = **{bd['2x_delta_max_hi_band']:.6e}** "
                 f"against the {bd['threshold']:.6g} refusal threshold "
                 f"({'a quarter of this pair’s banked rise' if src == 'banked' else 'the 09-08 literal'}) ⇒ "
                 f"**{'CANNOT account for the rise' if bd['cannot_explain_the_rise'] else 'could contribute'}**.")
        L.append("")
    L.append("| # | tripwire | fired |")
    L.append("|---|---|---|")
    names = {"R1_stoke_side_differs": "R1 — stoke's own probes differ between arms",
             "R2_closed_arm_does_not_reproduce": "R2 — closed arm does not reproduce the banked p95/n",
             "R3_null_arm_not_bit_identical": "R3 — null arm not bit-identical",
             "R4_switch_changed_nothing": "R4 — the switch changed nothing (vacuous A/B)",
             "R5_theta_mode_mismatch": "R5 — a probe file's theta_mode is not its arm's"}
    for k, v in report["tripwires"].items():
        L.append(f"| | {names.get(k, k)} | {'**FIRED**' if v else 'no'} |")
    if isinstance(report.get("R3"), dict):
        n = report["R3"]
        L.append("")
        L.append(f"R3 detail: {n['positions']} probed positions, "
                 f"{n['positions_differing']} differing, max \\|Δ\\| = "
                 f"**{n['max_abs_delta']:.6e}**"
                 f"{' — **bit-identical**' if n['bit_identical'] else ''}.")
    return "\n".join(L) + "\n"


def run_selftest() -> int:
    """The grading arithmetic and the banked-freeze parser, on numbers whose
    answer is known by hand. Neither needs a model, a binary or a probe."""
    fails = []

    def check(label, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            fails.append(label)

    # The banked transition: p95 0.154794 -> 0.204705 = 1.3224x.
    LO, HI = 0.154794, 0.204705
    g, _, _, _ = grade(LO, HI, LO, HI)
    # ⚠ 1e-5, not 1e-12: the banked 1.3224369... was computed from the
    # full-precision p95 values, and freeze_report prints them rounded to six
    # decimals. Demanding bit-agreement here would be asserting that a printed
    # number is the number it was printed from.
    check("the banked pair reproduces the banked growth 1.3224x",
          abs(g - 1.3224369308326396) < 1e-5)

    # accum removes the whole rise -> EXPLAINED
    _, _, e, v = grade(LO, HI, LO, LO)
    check("an accum arm with no rise at all is EXPLAINED (excess 1.0)",
          v == "EXPLAINED" and abs(e - 1.0) < 1e-12)
    # accum identical to closed -> REFUTED, excess exactly 0
    _, _, e, v = grade(LO, HI, LO, HI)
    check("an accum arm identical to closed is REFUTED (excess 0.0)",
          v == "REFUTED" and e == 0.0)
    # Either side of the EXPLAINED boundary. Deliberately NOT exactly 0.5: the
    # rule is a threshold on a measured ratio, so a test that lands on the
    # boundary is testing float rounding rather than the rule.
    def at(frac):
        return LO * (1.0 + (HI / LO - 1.0) * (1.0 - frac))
    _, _, e, v = grade(LO, HI, LO, at(0.5 + 1e-6))
    check("just above half the excess removed is EXPLAINED", v == "EXPLAINED")
    _, _, e, v = grade(LO, HI, LO, at(0.5 - 1e-6))
    check("just below half the excess removed is PARTIAL", v == "PARTIAL")
    # a quarter -> PARTIAL
    q = LO * (1.0 + (HI / LO - 1.0) * 0.75)
    _, _, e, v = grade(LO, HI, LO, q)
    check("removing a quarter of the excess is PARTIAL",
          v == "PARTIAL" and abs(e - 0.25) < 1e-9)
    # an accum arm that is WORSE than closed -> excess negative -> REFUTED
    _, _, e, v = grade(LO, HI, LO, HI * 1.10)
    check("an accum arm that widens the rise is REFUTED, not EXPLAINED",
          v == "REFUTED" and e < 0.0)
    # a closed arm with no rise leaves nothing to explain
    _, _, e, v = grade(LO, LO, LO, HI)
    check("a closed arm with no rise returns UNDEFINED, not a division by zero",
          e is None and v.startswith("UNDEFINED"))

    # The freeze_report parser, against the real banked file.
    tau = HERE.parent.parent / "docs" / "superpowers" / "gate-reports" / \
        "2026-09-07-m7-entry-probe-tau.json"
    if tau.exists():
        doc = json.loads(tau.read_text(encoding="utf-8"))
        got = {}
        for line in doc["freeze_report"]:
            s = line.strip()
            if ":" not in s or "p95=" not in s:
                continue
            band = s.split(":", 1)[0].strip()
            kv = dict(t.split("=", 1) for t in s.split(":", 1)[1].split()
                      if "=" in t)
            got[band] = {"n": int(kv["n"]), "p95": float(kv["p95"])}
        check("the banked freeze_report parses to all five bands", len(got) == 5)
        check("the parsed [600,1200) matches the banked table",
              got.get(LO_BAND) == {"n": 926, "p95": 0.154794})
        check("the parsed [1200,inf) matches the banked table",
              got.get(HI_BAND) == {"n": 1232, "p95": 0.204705})
        # R2_TOL must exceed the rounding in the banked line (6 decimals).
        check("R2_TOL is loose enough for the banked 6-decimal rounding",
              R2_TOL > 5e-7)
        # --bound-threshold banked must be the rule the 0.0125 literal was
        # computed by: a quarter of the Llama freeze's own rise. 1e-4, not
        # exact: the literal was rounded to four decimals.
        if LO_BAND in got and HI_BAND in got:
            derived = BOUND_QUARTER * (got[HI_BAND]["p95"] - got[LO_BAND]["p95"])
            check("'banked' derives the 09-08 literal from the 09-07 freeze",
                  abs(derived - BOUND_THRESHOLD) < 1e-4)
    else:
        check(f"banked tau file present at {tau}", False)

    print(f"\n{'ALL GREEN' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--closed", required=False)
    p.add_argument("--accum", required=False)
    p.add_argument("--banked-tau",
                   default=str(HERE.parent.parent / "docs" / "superpowers" /
                               "gate-reports" / "2026-09-07-m7-entry-probe-tau.json"))
    p.add_argument("--null-a")
    p.add_argument("--null-b")
    p.add_argument("--bound-threshold", default=str(BOUND_THRESHOLD),
                   help="the refusal threshold for 2 x max ref delta in the "
                        "high band: a number, or 'banked' = a quarter of the "
                        "rise in --banked-tau's own freeze report (the rule "
                        "the 0.0125 literal was computed by, for the Llama "
                        "pair). Default: the 09-08 literal, so that run "
                        "reproduces.")
    p.add_argument("--out")
    p.add_argument("--set", choices=tuple(mp.MODEL_SETS), default="llama32",
                   help="which model set the grids belong to (m7_probe refuses "
                        "to read one set's grid as another's; 2026-09-15, B6)")
    p.add_argument("--markdown", help="write the report's result section, "
                                      "generated from the artifact")
    p.add_argument("--shakedown", action="store_true",
                   help="pipeline check on a corpus that does not cover the "
                        "transition: prints the per-band p95, the delta and the "
                        "tripwires, and REFUSES to compute a growth ratio or a "
                        "verdict. A shakedown is not a result.")
    a = p.parse_args()
    if a.selftest:
        return run_selftest()
    if not a.closed or not a.accum:
        print("fatal: --closed and --accum are required", file=sys.stderr)
        return 1
    mp.select_set(a.set)

    ga = mp.load_grids(pathlib.Path(a.closed))
    gb = mp.load_grids(pathlib.Path(a.accum))
    if ga[0]["ref_theta_mode"] != "closed":
        raise AbError(f"--closed dir is arm {ga[0]['ref_theta_mode']!r}")
    if gb[0]["ref_theta_mode"] != "accum":
        raise AbError(f"--accum dir is arm {gb[0]['ref_theta_mode']!r}")

    fired = {}
    report = {"streams": sorted(g["stream"] for g in ga)}

    # ---- R1 -------------------------------------------------------------
    r1 = stoke_sides_identical(ga, gb)
    fired["R1_stoke_side_differs"] = bool(r1)
    report["R1"] = {"differences": r1[:10], "n_differences": len(r1)}

    # ---- the envelopes, through the freeze's own construction -----------
    env_a, agree_a, tot_a, div_a = mp.build_envelope(ga)
    env_b, agree_b, tot_b, div_b = mp.build_envelope(gb)
    pa, pb = band_p95(env_a), band_p95(env_b)
    report["closed"] = {"bands": pa, "probes": tot_a, "divergent": div_a,
                        "agreeing_positions": agree_a}
    report["accum"] = {"bands": pb, "probes": tot_b, "divergent": div_b,
                       "agreeing_positions": agree_b}

    covered = all(b in tbl for tbl in (pa, pb) for b in (LO_BAND, HI_BAND))
    if not covered and not a.shakedown:
        raise AbError(f"the streams do not cover {LO_BAND} -> {HI_BAND} in both "
                      f"arms -- closed has {sorted(pa)}, accum has {sorted(pb)}. "
                      f"Pass --shakedown to check the pipeline without a verdict.")
    if covered:
        g_closed, g_accum, excess, verdict = grade(
            pa[LO_BAND]["p95"], pa[HI_BAND]["p95"],
            pb[LO_BAND]["p95"], pb[HI_BAND]["p95"])
        report["growth"] = {"closed": g_closed, "accum": g_accum,
                            "excess_removed": excess, "verdict": verdict}
    else:
        g_closed = g_accum = excess = None
        verdict = "SHAKEDOWN -- no verdict, the corpus does not cover the transition"
        report["growth"] = {"verdict": verdict}

    # ---- R2: the closed arm must reproduce the banked freeze -------------
    banked = json.loads(pathlib.Path(a.banked_tau).read_text(encoding="utf-8"))
    bfr = banked["freeze_report"]
    banked_p95 = {}
    for line in bfr:
        # "  [600,1200): n=926 max=0.984710 p95=0.154794 => tau=0.309588"
        s = line.strip()
        if ":" not in s or "p95=" not in s:
            continue
        band = s.split(":", 1)[0].strip()
        kv = dict(t.split("=", 1) for t in s.split(":", 1)[1].split() if "=" in t)
        banked_p95[band] = {"n": int(kv["n"]), "p95": float(kv["p95"])}
    r2 = []
    for b in (LO_BAND, HI_BAND):
        if b not in pa:
            continue          # shakedown corpus; nothing to reproduce here
        if b not in banked_p95:
            r2.append(f"{b}: absent from the banked freeze report")
            continue
        if banked_p95[b]["n"] != pa[b]["n"]:
            r2.append(f"{b}: n {pa[b]['n']} vs banked {banked_p95[b]['n']}")
        if abs(banked_p95[b]["p95"] - pa[b]["p95"]) > R2_TOL:
            r2.append(f"{b}: p95 {pa[b]['p95']:.6f} vs banked "
                      f"{banked_p95[b]['p95']:.6f}")
    fired["R2_closed_arm_does_not_reproduce"] = bool(r2)
    report["R2"] = {"banked": banked_p95, "mismatches": r2}

    # ---- the bound's threshold: the literal, or this pair's own quarter ---
    if a.bound_threshold == "banked":
        if LO_BAND not in banked_p95 or HI_BAND not in banked_p95:
            raise AbError(f"--bound-threshold banked needs both {LO_BAND} and "
                          f"{HI_BAND} in the banked freeze report")
        bound_threshold = BOUND_QUARTER * (banked_p95[HI_BAND]["p95"]
                                           - banked_p95[LO_BAND]["p95"])
        if bound_threshold <= 0.0:
            raise AbError("--bound-threshold banked: the banked freeze shows "
                          "no rise, so a quarter of it is not a threshold")
        bound_source = "banked"
    else:
        bound_threshold = float(a.bound_threshold)
        bound_source = "literal"

    # ---- R4 / the bound --------------------------------------------------
    rows = ref_delta(ga, gb)
    dbb = delta_by_band(rows)
    all_d = [r["delta"] for r in rows if r["delta"] is not None]
    fired["R4_switch_changed_nothing"] = (max(all_d, default=0.0) == 0.0)
    # How often the switch moved a TOKEN DECISION, not just a logit. A delta
    # that never crosses an argmax is a different kind of small from one that
    # does, and the aggregate cannot distinguish them.
    flipped = [r for r in rows if r["ref_chose_closed"] != r["ref_chose_accum"]]
    report["delta"] = {"by_band": dbb,
                       "max_anywhere": max(all_d, default=None),
                       "n_positions": len(all_d),
                       "ref_argmax_flips": len(flipped),
                       "ref_argmax_flip_detail": flipped[:20],
                       # banked per position so "it grows with position" is
                       # auditable from the artifact, not only from two
                       # band aggregates
                       "positions": rows}
    bound_hi = 2.0 * dbb[HI_BAND]["max"] if HI_BAND in dbb else None
    report["bound"] = {"2x_delta_max_hi_band": bound_hi,
                       "threshold": bound_threshold,
                       "threshold_source": bound_source,
                       "cannot_explain_the_rise":
                           bound_hi is not None and bound_hi < bound_threshold}

    # ---- R3: the null arm -------------------------------------------------
    if a.null_a and a.null_b:
        n = null_arm(a.null_a, a.null_b)
        fired["R3_null_arm_not_bit_identical"] = not n["bit_identical"]
        report["R3"] = n
    else:
        report["R3"] = "NOT RUN"

    fired["R5_theta_mode_mismatch"] = False  # enforced at write time in m7_probe
    report["tripwires"] = fired

    # ---- print ------------------------------------------------------------
    print("ROPE-THETA A/B -- closed (ruled default) vs accum (diagnostic)")
    print(f"  streams: {', '.join(report['streams'])}")
    print("")
    print(f"  {'band':<14} {'arm':<7} {'n':>6} {'p95':>10} {'max':>10}")
    _order = [cls.band_name(lo, hi) for lo, hi in cls.BANDS]
    for b in sorted(set(pa) | set(pb), key=_order.index):
        for label, tbl in (("closed", pa), ("accum", pb)):
            r = tbl.get(b)
            if r is None:
                print(f"  {b:<14} {label:<7} {'-':>6} {'-':>10} {'-':>10}")
                continue
            print(f"  {b:<14} {label:<7} {r['n']:>6} {r['p95']:>10.6f} "
                  f"{r['max']:>10.6f}")
    print("")
    if g_closed is not None:
        print(f"  growth {LO_BAND}->{HI_BAND}:  closed {g_closed:.4f}x   "
              f"accum {g_accum:.4f}x")
        print(f"  excess removed by the accum arm: "
              f"{'n/a' if excess is None else format(excess, '.4f')}"
              f"   (EXPLAINED >= {EXPLAINED_AT}, REFUTED < {REFUTED_BELOW})")
    print(f"  VERDICT: {verdict}")
    print("")
    print("  ref closed-vs-accum logit delta (the bound):")
    order = [cls.band_name(lo, hi) for lo, hi in cls.BANDS]
    for b in sorted(dbb, key=order.index):
        r = dbb[b]
        print(f"    {b:<14} n={r['n']:>4} median_pos={r['median_position']:>5} "
              f"max={r['max']:.6e} p95={r['p95']:.6e}")
    if bound_hi is not None:
        print(f"    2 x max delta in {HI_BAND} = {bound_hi:.6e} vs the "
              f"{bound_threshold:.6g} threshold ({bound_source}) -> "
              f"{'CANNOT explain the rise' if bound_hi < bound_threshold else 'could contribute'}")
    print(f"    ref argmax flips between the arms: "
          f"{report['delta']['ref_argmax_flips']} of "
          f"{report['delta']['n_positions']} probed positions")
    print("")
    print("  tripwires:")
    for k, v in fired.items():
        print(f"    {k}: {'FIRED' if v else 'quiet'}")
    if r1:
        for d in r1[:5]:
            print(f"      R1: {d}")
    if r2:
        for d in r2:
            print(f"      R2: {d}")
    if isinstance(report["R3"], dict):
        n = report["R3"]
        print(f"    R3 null arm: {n['positions']} positions, "
              f"{n['positions_differing']} differing, max |delta| = "
              f"{n['max_abs_delta']:.6e}")

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(report, indent=1),
                                       encoding="utf-8")
        print(f"\n  -> {a.out}")
    if a.markdown:
        pathlib.Path(a.markdown).write_text(markdown(report), encoding="utf-8")
        print(f"  -> {a.markdown}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AbError, mp.ProbeError, mcp.ProbeError) as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)
