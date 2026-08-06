#!/usr/bin/env python3
"""tests/tooling/m2_classify_probes.py -- collect the section 6 classification
INPUT for [2v2-ii]'s divergences. It does not grade, and that is the point.

BINDING SOURCE: docs/superpowers/memos/2026-08-04-m3-gate-item-2-RULING-FABLE.md
(sections 2.3 and 6). classify.py implements the rule; this file only feeds it.

▶▶ WHY COLLECTION AND GRADING ARE SEPARATE PROGRAMS. Ruling section 2.3 carries
an ORDERING RULE: "the probe grid runs and freezes tau BEFORE [2v2-ii]'s
divergences are graded". The probe grid is [2v2-iii], which is blocked on the
G1-G8 cost brief. So the VERDICT is blocked -- but the top-5 logits at a
divergent position do not depend on tau in any way, so COLLECTING them is not.
Splitting the two lets the blocked half block only itself, and it makes the
ordering rule structural rather than a thing to remember: this tool cannot
grade even if asked, and classify.py fatals on a tau nobody froze.

▶ WHAT A RECORD IS. For each prompt whose M2 and HEAD streams differ, at the
FIRST divergent generated index k: both binaries' top-5 (id, logit) at k, each
side's chosen id, and abs_position = prompt_tokens + k. Absolute position is
what section 8.9's bands are keyed on, because the error term they track
(rope_neox theta accumulation) grows with absolute position, not with k.

▶ THE CROSS-CHECK THAT MAKES THE PROBES MEAN ANYTHING. A probe re-runs
generation from scratch, so nothing structurally guarantees that the
distribution it reports is the one that produced the banked id. This tool
therefore requires each side's probe rank-0 to EQUAL that side's banked id at
k, and treats a mismatch as fatal. Greedy decode makes rank-0 the chosen token
(model.c: top-5 descending, ties by lower id), so a mismatch means the stream
is not reproducible or the probe indexes a different position than the sweep
did -- either way a finding, and neither is something to average over. The
assumption is cheap to state and would be invisible if left unstated: exactly
the shape this project keeps getting caught by.

▶ LENGTH-MISMATCH DIVERGENCES HAVE NO TOP-5 ON ONE SIDE, BY CONSTRUCTION.
m2_sweep.py sets first_diff = min(len) when the streams agree everywhere but
one side stopped early (EOS). At that index the shorter side never produced a
distribution, so `--probe-pos` there fatals rather than printing five lines.
Such rows are recorded as unprobeable and reported as ESCALATE-BY-CONSTRUCTION
-- they are NOT written into the records file, because a fabricated fifth pair
would be a number nobody measured entering a criterion. They are also not
dropped: the section 6 fail-safe points at investigation, so an unclassifiable
position must survive into the report as unclassifiable.

▶ THE FREEZE GUARD IS RE-CHECKED HERE, NOT INHERITED. parity_gate.py keys its
cache on stoke_hash and m2_sweep.py refuses a stream from another binary; this
tool probes a LIVE binary and compares it against ids banked earlier, so it
re-verifies the sha256 itself. A rebuild between the sweep and the probes would
otherwise produce top-5s from an engine that never generated the stream being
explained.

▶ TAU KEY CONTRACT. Records carry pair = "f32_intcore_q8_0": the ruling's tau is
per PAIR and per FILE (section 9.4-2), and the M2 binary cannot read K-quants at
all, so [2v2-ii] is inherently the Q8_0 leg. The probe grid must freeze tau under
this exact key. A key mismatch is loud -- classify.py raises SystemExit on a
missing entry rather than defaulting -- which is why the contract is written
down here rather than coordinated by memory.

CLI:
  python m2_classify_probes.py --sweep FILE.json --out RECORDS.json [--limit N]
  python m2_classify_probes.py --selftest
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
HEAD_EXE = ROOT / "build" / "win" / "stoke.exe"
M2_EXE = ROOT / "build" / "stoke-m2-gate-d2b17c53.exe"
PROMPTS = ROOT / "tests" / "data" / "parity"
LENS = ROOT / "tests" / "tooling" / ".gate_cache" / "prompt_token_lens.json"

# Same sha the sweep credits its banked streams to. Not a copy to keep in sync
# by hand -- m2_sweep.py:31 carries the identical constant and both refuse
# rather than proceed, so a divergence between them fails loudly on first use.
HEAD_SHA = "876ce53fd4a943fdd820f0bf9e8328b8aa751fa1269df9fd20f4fedbfb6390ff"

PAIR = "f32_intcore_q8_0"

# src/main.c cmd_run(): fprintf(stderr, "probe pos=%d: id=%u logit=%.6f\n").
# %.6f is always fixed-point, never scientific, so the pattern is strict.
# Transcribed from the source, not guessed -- same contract parity_gate.py:773
# already relies on; a second spelling of it here would be a second thing to
# get wrong, so it is deliberately identical.
_PROBE_RE = re.compile(r"^probe pos=(-?\d+): id=(\d+) logit=(-?\d+\.\d{6})$")

# ▶ SAME P-CORE PIN AND PRIORITY AS m2_sweep.py, AND DUPLICATED ON PURPOSE.
# A probe IS a full re-generation, so it is the identical workload under the
# identical physics: measured 2026-08-06, work left unplaced bounces onto
# E-cores and second-hyperthreads for ~1.6x. The lines are copied rather than
# imported because m2_sweep.py is a SCRIPT -- importing it would run a sweep.
# ctypes argtypes are declared because HANDLE/DWORD_PTR are 64-bit here and
# ctypes defaults to c_int, which truncates the pseudo-handle silently.
# Priority is passed at creation because it is NOT reliably inherited by
# children (observed directly: parent High, fresh children Normal).
P_CORE_MASK = 0x555
_HIGH = getattr(subprocess, "HIGH_PRIORITY_CLASS", 0) if sys.platform == "win32" else 0

if sys.platform == "win32":
    _k32 = ctypes.WinDLL("kernel32")
    _k32.GetCurrentProcess.restype = ctypes.c_void_p
    _k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not _k32.SetProcessAffinityMask(_k32.GetCurrentProcess(), P_CORE_MASK):
        sys.exit(f"fatal: could not pin to P-cores (mask 0x{P_CORE_MASK:X})")

# ▶ RAM CEILING, and it is the second time this session that ignoring it cost
# real time. Each probe process holds ~1.2 GB private KV on top of ONE shared
# ~4.0 GB model. When the total exceeds what is free, the model starts being
# trimmed and re-faulted: measured signature is Cache Faults/sec jumping from
# ~60 to ~7,000 with Pages Input/sec still near zero, and prompts taking 3x
# longer. The arithmetic that matters: 3 jobs under that pressure completed 3
# prompts per 513 s where 2 jobs without it did 2 per 140 s -- FEWER JOBS IS
# 2.5x FASTER once the model stops fitting.
#
# ▶ THIS GUARD IS DELIBERATELY PARTIAL, and saying so is the point: it fires at
# LAUNCH only. Both collapses this session were caused by an application
# opening AFTER the run started (Chrome once, VS Code the next), which no
# launch-time check can see. Mid-run pressure needs an operator restart at a
# lower --jobs; the run is resumable per prompt precisely so that costs one
# probe.
MODEL_GB, PER_JOB_GB = 4.0, 1.3


def _free_gb():
    if sys.platform != "win32":
        return None
    class _MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    ms = _MS()
    ms.dwLength = ctypes.sizeof(_MS)
    ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullAvailPhys / (1024.0 ** 3)


PROBE_TIMEOUT_S = 2400.0   # p40-p44 re-generate ~550 positions on the M2
                           # binary at ~1.7 s each; the sweep measured 940 s
                           # worst case, so this is a ~2.5x margin. Loose is
                           # the safe direction for a ceiling (the 07-29 run
                           # lost a day to a tight one).


class ProbeError(Exception):
    """Any no-silent-fallback condition. Caught once in main()."""


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_probe(stderr_text: str, pos: int):
    """The five (id, logit) pairs for `pos`, in the order printed.

    Lines for OTHER positions are ignored rather than tolerated silently in the
    count: the filter is on pos, and the arity check that follows is what makes
    a malformed or short probe fatal instead of quietly producing a 4-entry
    top-5 that classify.py would then reject far from the cause."""
    out = []
    for line in stderr_text.splitlines():
        m = _PROBE_RE.match(line.strip())
        if m and int(m.group(1)) == pos:
            out.append((int(m.group(2)), float(m.group(3))))
    if len(out) != 5:
        raise ProbeError(f"expected exactly 5 'probe pos={pos}' lines, got {len(out)}")
    logits = [lg for _, lg in out]
    if any(logits[i] < logits[i + 1] for i in range(4)):
        raise ProbeError(f"probe at pos={pos} not descending: {logits}")
    return out


def check_rank0(top5, banked_chosen: int, who: str, pos: int) -> None:
    """A probe explains the stream it came from, or it explains nothing.

    Greedy decode picks the argmax and the probe prints descending with ties
    broken by lower id, so rank-0 IS the generated token. If it is not, the
    re-generation did not reproduce the banked stream or `pos` does not mean
    the same thing to the probe as it did to the sweep -- both are findings,
    and neither may be smoothed over by taking the probe at face value."""
    if top5[0][0] != banked_chosen:
        raise ProbeError(
            f"{who}: probe rank-0 id {top5[0][0]} != banked id {banked_chosen} "
            f"at generated index {pos} -- the probe does not explain the banked "
            f"stream; do not grade this position")


def divergent_rows(sweep: dict):
    """Rows that need a probe, in prompt order. A row with identical=True has no
    divergent position to grade; an ERROR row never decided anything."""
    out = []
    for r in sweep.get("rows", []):
        if r.get("error") or r.get("identical"):
            continue
        if r.get("first_diff") is None:
            raise ProbeError(f"{r.get('prompt')}: not identical but first_diff is null")
        out.append(r)
    return out


def comparison_index(row) -> int:
    """The only position at which the two streams are comparable.

    For a DIVERGENT prompt that is first_diff: past it the two sides are
    generating from different prefixes, so they are no longer answering the same
    question and a logit difference there measures nothing. For an AGREEING
    prompt every position is comparable, and the last generated index is used --
    which is also the index the ruling's own grid samples (63)."""
    if row.get("first_diff") is not None:
        return int(row["first_diff"])
    return len(row["head_ids"]) - 1


def probeable(row) -> bool:
    """False when one side stopped before the comparison index -- that side has
    no distribution there, so no top-5 exists to collect (see header)."""
    k = comparison_index(row)
    return k < len(row["m2_ids"]) and k < len(row["head_ids"])


def probe_pair(row):
    """Both binaries' top-5 at the comparison index. Worker body: subprocesses
    only, no shared state, no file writes -- the parent stays the single writer
    for the same reason m2_sweep.py's does."""
    k = comparison_index(row)
    prompt = PROMPTS / f"{row['prompt']}.txt"
    head5 = run_probe(HEAD_EXE, prompt, k, len(row["head_ids"]))
    check_rank0(head5, int(row["head_ids"][k]), "HEAD", k)
    m2_5 = run_probe(M2_EXE, prompt, k, len(row["m2_ids"]))
    check_rank0(m2_5, int(row["m2_ids"][k]), "M2", k)
    return row, k, m2_5, head5


def run_probe(exe: pathlib.Path, prompt: pathlib.Path, pos: int, n_predict: int):
    argv = [str(exe), "run", str(MODEL), "--raw", "-f", str(prompt),
            "-n", str(n_predict), "--probe-pos", str(pos)]
    try:
        p = subprocess.run(argv, capture_output=True, timeout=PROBE_TIMEOUT_S,
                           creationflags=_HIGH)
    except subprocess.TimeoutExpired:
        raise ProbeError(f"probe timed out after {PROBE_TIMEOUT_S}s: {' '.join(argv)}")
    if p.returncode != 0:
        raise ProbeError(f"probe failed (exit {p.returncode}): {' '.join(argv)}\n"
                         f"stderr: {p.stderr.decode('utf-8', 'replace')[-400:]}")
    return parse_probe(p.stderr.decode("utf-8", "replace"), pos)


def delta_profile(m2_top5, head_top5, head_chose: int):
    """|Δ| between the two sides' logits, over the ids BOTH sides ranked.

    ▶ ANCHORED, and the choice is disclosed rather than resolved silently. Two
    implementations of the same forward pass can in principle carry a constant
    offset across a whole logit vector, which would inflate a raw difference
    without any token decision being at risk. Anchoring on HEAD's chosen id (the
    memo section 3.6 method) subtracts that offset out and measures what the
    classification rule actually consumes: how much the two sides disagree about
    MARGINS. Both are returned -- `anchored` is what tau is frozen from,
    `raw` is recorded beside it so a reader can see whether the distinction
    mattered on this corpus. If they differ materially that is itself a finding.

    Only ids present on BOTH sides can be differenced: a top-5 reports five
    logits, and an id outside the other side's five has an unknown logit, not a
    large one. Excluding them is the same asymmetry classify.py enforces -- a
    bound may prove BUG, never CLASS."""
    x, y = dict(m2_top5), dict(head_top5)
    shared = sorted(set(x) & set(y))
    if head_chose not in x or head_chose not in y:
        # No common anchor => no offset-free comparison exists here. Refusing is
        # the fail-safe direction: a sample that cannot be anchored must not
        # quietly enter the envelope that sets tau for everything else.
        return None
    ax, ay = x[head_chose], y[head_chose]
    return {
        "shared_ids": shared,
        "anchored": [abs((x[i] - ax) - (y[i] - ay)) for i in shared],
        "raw": [abs(x[i] - y[i]) for i in shared],
    }


def build_envelope_sample(row, plen: int, m2_top5, head_top5) -> dict:
    """One class-noise sample, taken at a position where the two streams AGREE.

    ▶ WHY AGREEING POSITIONS, AND WHY THAT IS NOT CIRCULAR. tau exists to say
    how large a logit disagreement is ordinary for this implementation pair. It
    is measured here on positions where M2 and HEAD chose the SAME token, and
    spent on positions where they chose differently -- disjoint sets, so no
    divergence contributes to the threshold that grades it. This is the same
    shape as the ruling's ref grid (which samples index 63 on all 50 prompts,
    overwhelmingly agreeing positions); only the referent differs, and section
    2.3 names M2<->HEAD and ref<->HEAD as the same tau family.

    ▶ ITS BLIND SPOT, stated because it is real: M2 and HEAD share every line of
    composition logic (attention, rope, layer order), so a bug there is invisible
    to this pair -- but it is equally invisible to the COMPARISON being graded,
    which is the same pair. This measures exactly the arithmetic-class question
    it is asked. Independent coverage of the shared logic is what the ref
    referent supplies, and for Q8_0 it already exists at 1.4e-5 (memo 3.6)."""
    k = int(row["first_diff"]) if row.get("first_diff") is not None else len(row["head_ids"]) - 1
    prof = delta_profile(m2_top5, head_top5, int(row["head_ids"][k]))
    return {"prompt": row["prompt"], "position": k, "abs_position": plen + k,
            "pair": PAIR, "profile": prof}


def build_record(row, plen: int, m2_top5, head_top5) -> dict:
    """Assemble one classify.py record. abs_position is prompt tokens + the
    generated index; src/tok.c is unchanged between m2 and HEAD (verified by
    git diff), so ONE prompt length is correct for both sides."""
    k = int(row["first_diff"])
    return {
        "prompt": row["prompt"],
        "position": k,
        "abs_position": plen + k,
        "pair": PAIR,
        "x": {"name": "M2", "chose": int(row["m2_ids"][k]),
              "top5": [[i, l] for i, l in m2_top5]},
        "y": {"name": "HEAD", "chose": int(row["head_ids"][k]),
              "top5": [[i, l] for i, l in head_top5]},
    }


TAU_FLOOR = 0.10          # ruling 9.4-2: the floor exists because 0.0430 is a
                          # SINGLE-position measurement and a knife-edge tau
                          # would classify on noise.
TAU_MULTIPLIER = 2.0


def _classify_mod():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import classify
    return classify


def percentile(values, q):
    """Nearest-rank percentile on sorted values. Nearest-rank, not interpolated:
    with a handful of samples per band an interpolated p95 invents a value that
    was never measured, and tau is a threshold that must be traceable to an
    observation."""
    if not values:
        raise ProbeError("percentile of an empty sample")
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(-(-q * len(s) // 100)) - 1))
    return s[idx]


def freeze_tau(envelope, min_per_band=8):
    """tau(pair, band) = max(0.10, 2 x band p95), ruling 9.4-2, applied
    mechanically to the measured envelope.

    ▶ A BAND WITH TOO FEW SAMPLES IS REFUSED, NOT FLOORED. It is tempting to let
    a thin band fall back on the 0.10 floor and carry on -- but that silently
    converts "we did not measure this band" into "this band measured quiet",
    which is precisely the substitution the freeze exists to prevent. A refused
    band means positions in it cannot be graded until it is measured, and
    classify.py already fatals on a missing entry, so the refusal propagates
    instead of evaporating."""
    classify = _classify_mod()
    by_band = {}
    for s in envelope:
        prof = s.get("profile")
        if not prof:
            continue
        band = classify.band_of(int(s["abs_position"]))
        by_band.setdefault(band, []).extend(prof["anchored"])

    table, report = {PAIR: {}}, []
    for band, vals in sorted(by_band.items()):
        p95 = percentile(vals, 95)
        tau = max(TAU_FLOOR, TAU_MULTIPLIER * p95)
        if len(vals) < min_per_band:
            report.append(f"  {band}: REFUSED -- {len(vals)} samples < {min_per_band} "
                          f"(p95 would be {p95:.6f}, tau {tau:.6f})")
            continue
        table[PAIR][band] = tau
        report.append(f"  {band}: n={len(vals)} max={max(vals):.6f} p95={p95:.6f} "
                      f"=> tau={tau:.6f}{'  (floor)' if tau == TAU_FLOOR else ''}")
    return table, report


# ============================================================================
# selftest -- everything above that does not need a model
# ============================================================================

def run_selftest() -> int:
    fails = []
    total = [0]

    def check(label, ok):
        total[0] += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            fails.append(label)

    def raises(fn):
        """BaseException, not Exception -- classify.py signals an unfrozen tau
        with SystemExit, which does NOT derive from Exception. Caught by this
        selftest on its first run: the narrow spelling let the SystemExit escape
        and kill the harness rather than register as the pass it was."""
        try:
            fn()
        except BaseException:
            return True
        return False

    good = ("probe pos=7: id=100 logit=12.500000\n"
            "probe pos=7: id=200 logit=11.000000\n"
            "probe pos=7: id=300 logit=10.000000\n"
            "probe pos=7: id=400 logit=9.000000\n"
            "probe pos=7: id=500 logit=8.000000\n")

    got = parse_probe(good, 7)
    check("parse_probe returns 5 pairs", len(got) == 5)
    check("parse_probe keeps printed order", [i for i, _ in got] == [100, 200, 300, 400, 500])
    check("parse_probe reads negative logits",
          parse_probe(good.replace("logit=8.000000", "logit=-8.000000"), 7)[4][1] == -8.0)

    # Lines for another position must not be counted -- a probe run emits only
    # one position's worth, but the filter is what guarantees it.
    mixed = good + "probe pos=8: id=999 logit=99.000000\n"
    check("parse_probe filters other positions", len(parse_probe(mixed, 7)) == 5)
    check("parse_probe fatal on wrong position", raises(lambda: parse_probe(good, 8)))
    check("parse_probe fatal on 4 lines",
          raises(lambda: parse_probe("\n".join(good.splitlines()[:4]), 7)))
    check("parse_probe fatal on non-descending",
          raises(lambda: parse_probe(good.replace("id=200 logit=11.000000",
                                                  "id=200 logit=13.000000"), 7)))
    # %.6f is the contract; scientific notation means the format changed under us.
    check("parse_probe rejects scientific notation",
          raises(lambda: parse_probe(good.replace("logit=8.000000", "logit=8.0e00"), 7)))

    check("check_rank0 accepts matching id", check_rank0(got, 100, "X", 7) is None)
    check("check_rank0 fatal on mismatch", raises(lambda: check_rank0(got, 200, "X", 7)))

    sweep = {"rows": [
        {"prompt": "p00", "identical": True, "first_diff": None,
         "m2_ids": [1], "head_ids": [1]},
        {"prompt": "p01", "identical": False, "first_diff": 3,
         "m2_ids": [1, 2, 3, 4, 5], "head_ids": [1, 2, 3, 9, 5]},
        {"prompt": "p02", "error": "boom", "secs": 1.0},
        {"prompt": "p03", "identical": False, "first_diff": 2,
         "m2_ids": [1, 2], "head_ids": [1, 2, 7]},
    ]}
    d = divergent_rows(sweep)
    check("divergent_rows skips identical and ERROR", [r["prompt"] for r in d] == ["p01", "p03"])
    check("divergent_rows fatal on null first_diff when not identical",
          raises(lambda: divergent_rows({"rows": [{"prompt": "pX", "identical": False,
                                                   "first_diff": None}]})))
    check("probeable true when both sides reached the index", probeable(d[0]) is True)
    check("probeable FALSE on a length-mismatch divergence", probeable(d[1]) is False)

    rec = build_record(d[0], plen=40,
                       m2_top5=[(4, 5.0), (9, 4.9), (3, 1.0), (2, 0.5), (1, 0.0)],
                       head_top5=[(9, 5.0), (4, 4.95), (3, 1.0), (2, 0.5), (1, 0.0)])
    check("record abs_position = prompt tokens + index", rec["abs_position"] == 43)
    check("record x chose M2's banked id", rec["x"]["chose"] == 4)
    check("record y chose HEAD's banked id", rec["y"]["chose"] == 9)
    check("record carries the frozen-tau pair key", rec["pair"] == PAIR)
    check("record top5 is JSON-shaped pairs",
          rec["x"]["top5"][0] == [4, 5.0] and len(rec["x"]["top5"]) == 5)

    # --- comparison index -------------------------------------------------
    check("comparison_index = first_diff when divergent", comparison_index(d[0]) == 3)
    check("comparison_index = last index when agreeing",
          comparison_index({"identical": True, "first_diff": None,
                            "head_ids": [1, 2, 3, 4]}) == 3)

    # --- delta profile ----------------------------------------------------
    # A pure additive offset on one side must vanish under anchoring and must
    # NOT vanish in the raw figures -- that contrast is the whole reason the
    # anchored form is the one tau is frozen from.
    xs = [(10, 5.0), (20, 4.0), (30, 3.0), (40, 2.0), (50, 1.0)]
    ys_off = [(10, 6.0), (20, 5.0), (30, 4.0), (40, 3.0), (50, 2.0)]
    prof = delta_profile(xs, ys_off, 10)
    check("anchored delta cancels a constant offset", max(prof["anchored"]) == 0.0)
    check("raw delta does NOT cancel it", min(prof["raw"]) == 1.0)
    ys_real = [(10, 5.0), (20, 4.25), (30, 3.0), (40, 2.0), (50, 1.0)]
    check("anchored delta sees a real disagreement",
          abs(max(delta_profile(xs, ys_real, 10)["anchored"]) - 0.25) < 1e-9)
    check("delta profile uses only ids BOTH sides ranked",
          delta_profile(xs, [(10, 5.0), (99, 4.0), (30, 3.0), (40, 2.0), (50, 1.0)],
                        10)["shared_ids"] == [10, 30, 40, 50])
    check("delta profile REFUSES when the anchor is not on both sides",
          delta_profile(xs, [(11, 5.0), (20, 4.0), (30, 3.0), (40, 2.0), (50, 1.0)],
                        10) is None)

    # --- percentile / tau freeze -----------------------------------------
    check("percentile is nearest-rank, never interpolated",
          percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0)
    check("percentile fatal on empty sample", raises(lambda: percentile([], 95)))

    def env(n, val, absolute):
        return [{"prompt": f"p{i:02d}", "abs_position": absolute, "pair": PAIR,
                 "profile": {"shared_ids": [1], "anchored": [val], "raw": [val]}}
                for i in range(n)]

    tbl, _ = freeze_tau(env(10, 0.01, 50))
    check("tau takes the 0.10 floor when the band is quiet",
          tbl[PAIR]["[0,100)"] == 0.10)
    tbl2, _ = freeze_tau(env(10, 0.20, 50))
    check("tau is 2x p95 when that exceeds the floor",
          abs(tbl2[PAIR]["[0,100)"] - 0.40) < 1e-9)
    tbl3, rep3 = freeze_tau(env(3, 0.20, 50))
    check("a thin band is REFUSED, not floored", "[0,100)" not in tbl3[PAIR])
    check("a refused band says so in the report",
          any("REFUSED" in line for line in rep3))
    tbl4, _ = freeze_tau(env(10, 0.20, 50) + env(10, 0.05, 350))
    check("tau is frozen PER BAND, not globally",
          abs(tbl4[PAIR]["[0,100)"] - 0.40) < 1e-9 and tbl4[PAIR]["[300,600)"] == 0.10)

    # The record must be something classify.py actually accepts -- a shape that
    # only this file agrees with would fail an hour into a probe run.
    try:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import classify
        out = classify.classify_records([rec], {PAIR: {"[0,100)": 0.10}})
        check("classify.py accepts the emitted record", out and "verdict" in out[0])
        check("classify.py fatals when the pair key is not frozen",
              raises(lambda: classify.classify_records([rec], {"other": {"[0,100)": 0.1}})))
    except ImportError:
        check("classify.py importable", False)

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED'} "
          f"({total[0] - len(fails)}/{total[0]})")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep")
    ap.add_argument("--out")
    ap.add_argument("--envelope",
                    help="where to write the class-noise samples taken at AGREEING "
                         "positions; these are what tau is later frozen from")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="probe at most N prompts (banking is per-prompt, so a "
                         "limited run is a prefix of the full one, not a "
                         "different experiment)")
    ap.add_argument("--freeze-tau", metavar="ENVELOPE.json",
                    help="SEPARATE INVOCATION, deliberately: compute the tau table "
                         "from an envelope file. Keeping the freeze out of the "
                         "collecting run is what makes ruling 2.3's ordering rule "
                         "structural -- the numbers exist, and are fixed, before "
                         "anything is graded against them.")
    ap.add_argument("--tau-out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return run_selftest()

    if a.freeze_tau:
        envelope = json.loads(pathlib.Path(a.freeze_tau).read_text())
        table, report = freeze_tau(envelope)
        print(f"tau frozen from {len(envelope)} envelope samples "
              f"(agreeing positions only):")
        for line in report:
            print(line)
        if a.tau_out:
            pathlib.Path(a.tau_out).write_text(json.dumps(table, indent=1))
            print(f"\n-> {a.tau_out}")
        return 0 if table[PAIR] else 1

    if not (a.sweep and a.out):
        print("fatal: --sweep and --out are both required", file=sys.stderr)
        return 2

    if a.jobs > 1:
        free, need = _free_gb(), MODEL_GB + a.jobs * PER_JOB_GB
        if free is not None and free < need:
            print(f"fatal: --jobs {a.jobs} needs ~{need:.1f} GB (model {MODEL_GB} + "
                  f"{a.jobs}x{PER_JOB_GB} private KV), {free:.1f} GB free. Under that, "
                  f"the model is trimmed and re-faulted and FEWER JOBS RUN FASTER.",
                  file=sys.stderr)
            return 2

    live = sha256_of(HEAD_EXE)
    if live != HEAD_SHA:
        print(f"fatal: build/win/stoke.exe is {live[:8]}..., banked streams are "
              f"{HEAD_SHA[:8]}... -- probes from a rebuilt engine cannot explain "
              f"streams it never generated. Re-bank or restore.", file=sys.stderr)
        return 2

    sweep = json.loads(pathlib.Path(a.sweep).read_text())
    lens = json.loads(LENS.read_text())
    all_rows = [r for r in sweep.get("rows", []) if not r.get("error")]
    div = divergent_rows(sweep)
    same = [r for r in all_rows if r.get("identical")]
    print(f"prompts: {len(all_rows)} usable of {sweep.get('of', '?')} "
          f"-- {len(div)} divergent (graded), {len(same)} agreeing (envelope)",
          flush=True)

    out_path = pathlib.Path(a.out)
    side_path = out_path.with_name(out_path.stem + "_unprobeable.json")
    env_path = pathlib.Path(a.envelope) if a.envelope else None
    records, unprobeable, envelope = [], [], []
    done = set()
    if out_path.exists():
        records = json.loads(out_path.read_text())
        done |= {r["prompt"] for r in records}
    if side_path.exists():
        unprobeable = json.loads(side_path.read_text())
        done |= {r["prompt"] for r in unprobeable}
    if env_path and env_path.exists():
        envelope = json.loads(env_path.read_text())
        done |= {r["prompt"] for r in envelope}
    if done:
        print(f"resuming: {len(done)} prompts already probed", flush=True)

    rows = div + (same if env_path else [])
    todo = [r for r in sorted(rows, key=lambda r: r["prompt"]) if r["prompt"] not in done]
    if a.limit:
        todo = todo[:a.limit]

    runnable = []
    for row in todo:
        pid = row["prompt"]
        plen = lens.get(pid)
        if plen is None:
            # Regenerable, and the recipe travels with the error: `stoke
            # tokenize` calls tok_encode(add_special=false), the exact call
            # `run --raw` makes, so its counts are the same tokenization the
            # banked streams were generated from -- not an approximation of it.
            raise ProbeError(
                f"{pid}: no measured prompt token length in {LENS.name}. "
                f"Regenerate with: build\\win\\stoke.exe tokenize <model.gguf> "
                f"tests\\data\\parity\\{pid}.txt  (count the ids)")
        k = comparison_index(row)
        if not probeable(row):
            short = "M2" if k >= len(row["m2_ids"]) else "HEAD"
            unprobeable.append({"prompt": pid, "position": k, "abs_position": plen + k,
                                "pair": PAIR, "reason": f"{short} stopped before index {k}; "
                                f"no top-5 exists on that side",
                                "verdict": "ESCALATE-BY-CONSTRUCTION"})
            side_path.write_text(json.dumps(unprobeable, indent=1))
            print(f"{pid}: UNPROBEABLE ({short} stopped before {k}) -> ESCALATE", flush=True)
            continue
        runnable.append(row)

    t_all = time.time()
    if a.jobs > 1:
        pool = ThreadPoolExecutor(max_workers=a.jobs)
        futs = [pool.submit(probe_pair, r) for r in runnable]
        results = (f.result() for f in as_completed(futs))
    else:
        results = (probe_pair(r) for r in runnable)

    for row, k, m2_5, head5 in results:
        pid = row["prompt"]
        plen = lens[pid]
        if row.get("identical"):
            sample = build_envelope_sample(row, plen, m2_5, head5)
            envelope.append(sample)
            envelope.sort(key=lambda r: r["prompt"])
            env_path.write_text(json.dumps(envelope, indent=1))
            prof = sample["profile"]
            worst = max(prof["anchored"]) if prof else float("nan")
            print(f"{pid}: ENVELOPE at k={k} abs={plen + k} "
                  f"n={len(prof['shared_ids']) if prof else 0} "
                  f"worst|d|={worst:.6f}", flush=True)
        else:
            records.append(build_record(row, plen, m2_5, head5))
            records.sort(key=lambda r: r["prompt"])
            out_path.write_text(json.dumps(records, indent=1))
            print(f"{pid}: RECORD at k={k} abs={plen + k} "
                  f"(M2 chose {row['m2_ids'][k]}, HEAD chose {row['head_ids'][k]})",
                  flush=True)

    print(f"\n{len(records)} grading records, {len(envelope)} envelope samples, "
          f"{len(unprobeable)} unprobeable, {time.time() - t_all:.0f}s")
    print(f"records -> {out_path}")
    if env_path:
        print(f"envelope -> {env_path}")
    if unprobeable:
        print(f"unprobeable -> {side_path}")
    print("\nNOT GRADED, and this tool has no path that would. Next, in order:"
          f"\n  1. python m2_classify_probes.py --freeze-tau {env_path} --tau-out tau.json"
          f"\n  2. python classify.py --records {out_path} --tau-table tau.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ProbeError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)
