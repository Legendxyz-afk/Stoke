#!/usr/bin/env python3
"""tests/tooling/m7_probe.py -- M7's entry probe grid: the ruled ref-vs-stoke
comparison for Llama-3.2-3B.

BINDING SOURCES, in order:
  docs/ROADMAP.md:619-627                                  (Fable, 2026-08-05)
  docs/superpowers/memos/2026-08-04-m3-gate-item-2-RULING-FABLE.md
      §2.3 (the probe protocol), §6 (the classification rule),
      §8.8 (M7's parity criterion), §8.9 (per-band tau, M7's two new bands)
  docs/superpowers/design-notes/2026-09-07-m7-entry-probe-preregistration.md
      (the corpus, the grid, the tripwires -- written before any comparison)

▶ WHAT THIS IS NOT. It never consults llama.cpp. The two sides are
ref_forward.py (f32 numpy) and the stoke binary. A live-oracle token bar is the
one shape the ruling forbids for a forward pass, and staying out of it is
structural here, not a matter of discipline.

▶ THE THREE PHASES ARE SEPARATE INVOCATIONS AND THAT IS THE DESIGN. Phase B
freezes tau; phase C spends it. A tool that could do both can always be run in
the order that passes. The tau file on disk between them is what makes the
ordering auditable afterwards -- the same split q4km_tau.py and classify.py
already keep.

▶ EVERYTHING SHARED IS IMPORTED, NEVER REIMPLEMENTED: parse_probe, check_rank0,
delta_profile, freeze_tau, percentile from m2_classify_probes; classify and
band_of from classify. A second implementation of a frozen constant's formula is
a second thing that can drift from the ruling.

CLI:
  python m7_probe.py --phase a --out-dir DIR [--only STREAM,...]
  python m7_probe.py --phase b --out-dir DIR --tau-out TAU.json
  python m7_probe.py --phase c --out-dir DIR --tau TAU.json [--verdicts V.json]
  python m7_probe.py --selftest
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import classify as cls                      # noqa: E402
import m2_classify_probes as mcp            # noqa: E402

MODEL = ROOT / "models" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
STOKE = ROOT / "build" / "win" / "stoke.exe"
LAUNCHER = HERE / "run_ref_probe.ps1"
N_GEN = 64

# ⚠ THE PAIR KEY IS ITS OWN, AND THAT IS LOAD-BEARING. tau is per pair and per
# file (ruling 9.4-2, G6). A Llama-3.2 envelope frozen under a Qwen3 or Q8_0 key
# would NOT be loud: classify.py's fatal fires on a key that is ABSENT, never on
# one that is present and wrong.
PAIR = "f32_intcore_llama32_q4km"

SHORT = "tests/data/llama_gen/{}.txt"
LONG = "tests/data/llama_gen/long2.txt"

# ============================================================================
# THE CORPUS -- must equal §3.3 of the pre-registration, which is authoritative.
# `target` is a TARGET token count realized as a byte cut (see the amendment in
# §3.2 of that document); the count that actually falls out is measured and
# banked. `selected` flags the three streams inherited from a corpus that was
# selected on success.
# ============================================================================
STREAMS = [
    {"id": "S-a1",   "src": SHORT.format("short1"), "target": None, "selected": True},
    {"id": "S-a2",   "src": SHORT.format("short2"), "target": None, "selected": True},
    {"id": "S-a3",   "src": SHORT.format("short3"), "target": None, "selected": True},
    {"id": "L-40",   "src": LONG, "target": 40,   "selected": False},
    {"id": "L-80",   "src": LONG, "target": 80,   "selected": False},
    {"id": "L-160",  "src": LONG, "target": 160,  "selected": False},
    {"id": "L-240",  "src": LONG, "target": 240,  "selected": False},
    {"id": "L-380",  "src": LONG, "target": 380,  "selected": False},
    {"id": "L-520",  "src": LONG, "target": 520,  "selected": False},
    {"id": "L-660",  "src": LONG, "target": 660,  "selected": False},
    {"id": "L-900",  "src": LONG, "target": 900,  "selected": False},
    {"id": "L-1120", "src": LONG, "target": 1120, "selected": False},
    {"id": "L-1300", "src": LONG, "target": 1300, "selected": False},
    {"id": "L-1900", "src": LONG, "target": 1900, "selected": False},
    {"id": "L-2500", "src": LONG, "target": 2500, "selected": False},
    {"id": "L-2851", "src": LONG, "target": None, "selected": True},
]

# Tripwires, from §5 of the pre-registration. Named constants so a later reader
# can see they were set before the grid ran, not tuned to it.
T1_DIVERGENCE_RATE = 0.02
T3_MIN_AGREEING_POSITIONS = 20
T4_BAND_GROWTH = 1.5


class ProbeError(RuntimeError):
    pass


# ============================================================================
# phase A -- bank the probes
# ============================================================================

def _run(argv, **kw):
    r = subprocess.run(argv, capture_output=True, text=True, **kw)
    return r


def tokenize(path: pathlib.Path):
    r = _run([str(STOKE), "tokenize", str(MODEL), str(path)])
    if r.returncode != 0:
        raise ProbeError(f"stoke tokenize exited {r.returncode}: {r.stderr.strip()[:300]}")
    line = r.stdout.strip().splitlines()[0]
    return [int(s) for s in line.strip("[]").replace(" ", "").split(",") if s]


def cut_text(src: pathlib.Path, target_tokens: int, work: pathlib.Path,
             stream_id: str) -> pathlib.Path:
    """Byte-cut `src` near `target_tokens`, snapped FORWARD to the next space so
    a stream never begins... or rather ends... mid-word.

    The bytes-per-token ratio is measured from the file itself, not assumed:
    the same target on a different text would land somewhere else entirely, and
    a hardcoded 5.5 would be a constant nobody measured."""
    raw = src.read_bytes()
    all_ids = tokenize(src)
    bpt = len(raw) / max(len(all_ids), 1)
    off = min(len(raw), max(1, int(round(target_tokens * bpt))))
    nxt = raw.find(b" ", off)
    if nxt != -1:
        off = nxt
    out = work / f"{stream_id}.prompt.txt"
    out.write_bytes(raw[:off])
    return out


def stoke_probe_all(prompt_file: pathlib.Path):
    """One `stoke run --probe-all`. Returns (gen_ids, {gen_index: top5}).

    ⚠ THE VERDICT IS TAKEN FROM WHAT THE RUN PRINTS. An exit code says the
    process ended, not that it generated anything: a run that produced zero ids
    exits 0 and would otherwise enter the grid as an empty stream."""
    r = _run([str(STOKE), "run", str(MODEL), "--raw", "-n", str(N_GEN),
              "-f", str(prompt_file), "--probe-all"])
    if r.returncode != 0:
        raise ProbeError(f"stoke run exited {r.returncode}: {r.stderr.strip()[:300]}")
    head = r.stdout.split("\n---\n", 1)[0]
    gen = [int(s) for s in head.split() if s.strip().isdigit()]
    if not gen:
        raise ProbeError("stoke run produced no ids")
    probes = {}
    for k in range(len(gen)):
        probes[k] = mcp.parse_probe(r.stderr, k)
        # rank-0 IS the generated token under greedy decode. If it is not, the
        # probe does not explain the stream it came from and nothing downstream
        # may treat it as if it did.
        mcp.check_rank0(probes[k], gen[k], "stoke", k)
    return gen, probes


def ref_probe(ids, positions, out_json: pathlib.Path, work: pathlib.Path,
              stream_id: str, mode="batched"):
    ids_file = work / f"{stream_id}.stream.ids"
    ids_file.write_text("\n".join(str(i) for i in ids), encoding="ascii")
    argv = ["powershell", "-NoProfile", "-File", str(LAUNCHER),
            "-Gguf", str(MODEL), "-IdsFile", str(ids_file),
            "-Out", str(out_json), "-Mode", mode,
            "-ProbeAt", ",".join(str(p) for p in positions)]
    r = _run(argv)
    if not out_json.exists():
        raise ProbeError(f"ref probe wrote no file for {stream_id} "
                         f"(rc={r.returncode}): {r.stderr.strip()[:400]}")
    return json.loads(out_json.read_text(encoding="utf-8"))


def phase_a(out_dir: pathlib.Path, only=None):
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    banked = []
    for st in STREAMS:
        if only and st["id"] not in only:
            continue
        src = ROOT / st["src"]
        pf = src if st["target"] is None else cut_text(src, st["target"], work, st["id"])
        prompt_ids = tokenize(pf)
        gen, s_probes = stoke_probe_all(pf)
        stream_ids = prompt_ids + gen
        n_prompt = len(prompt_ids)
        # generated index k lives at ABSOLUTE position n_prompt-1+k: the
        # distribution that produced gen[k] was computed after consuming
        # stream_ids[n_prompt-1+k].
        positions = [n_prompt - 1 + k for k in range(len(gen))]
        ref = ref_probe(stream_ids, positions,
                        out_dir / f"{st['id']}.ref.json", work, st["id"])
        r_probes = {int(p["pos"]): [tuple(t) for t in p["top5"]] for p in ref["probes"]}
        missing = [p for p in positions if p not in r_probes]
        if missing:
            raise ProbeError(f"{st['id']}: ref returned no probe at {missing[:5]}")

        rec = {"stream": st["id"], "src": st["src"], "selected_on_success": st["selected"],
               "target_tokens": st["target"], "n_prompt": n_prompt,
               "n_gen": len(gen), "gen_ids": gen, "ref_env": ref["env"],
               "ref_seconds": ref["seconds"], "ids_sha256": ref["ids_sha256"],
               "positions": []}
        for k, pos in enumerate(positions):
            rec["positions"].append({
                "abs_position": pos, "gen_index": k,
                "band": cls.band_of(pos),
                "stoke": {"top5": s_probes[k], "chose": s_probes[k][0][0]},
                "ref": {"top5": r_probes[pos], "chose": r_probes[pos][0][0]},
            })
        (out_dir / f"{st['id']}.grid.json").write_text(
            json.dumps(rec, indent=1), encoding="utf-8")
        agree = sum(1 for p in rec["positions"] if p["stoke"]["chose"] == p["ref"]["chose"])
        print(f"{st['id']}: n_prompt={n_prompt} probes={len(positions)} "
              f"bands={sorted({p['band'] for p in rec['positions']})} "
              f"agree={agree}/{len(positions)} ref={ref['seconds']:.1f}s", flush=True)
        banked.append(rec["stream"])
    return banked


# ============================================================================
# phase B -- freeze tau, from AGREEING positions only
# ============================================================================

def load_grids(out_dir: pathlib.Path):
    grids = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(out_dir.glob("*.grid.json"))]
    if not grids:
        raise ProbeError(f"no *.grid.json in {out_dir} -- phase A has not run")
    return grids


def phase_b(out_dir: pathlib.Path, tau_out: pathlib.Path):
    grids = load_grids(out_dir)
    envelope, agreeing_positions = [], {}
    total = divergent = 0
    for g in grids:
        for p in g["positions"]:
            total += 1
            if p["stoke"]["chose"] != p["ref"]["chose"]:
                divergent += 1
                continue
            # anchored on STOKE's chosen id -- the side under test, matching
            # delta_profile's "head_chose" role in the M2/M3 legs.
            prof = mcp.delta_profile(p["ref"]["top5"], p["stoke"]["top5"],
                                     p["stoke"]["chose"])
            if prof is None:
                continue
            envelope.append({"abs_position": p["abs_position"], "profile": prof,
                             "stream": g["stream"]})
            agreeing_positions.setdefault(p["band"], 0)
            agreeing_positions[p["band"]] += 1

    table, report = mcp.freeze_tau(envelope, min_per_band=8, pair=PAIR)
    # ⚠ THE RULING'S FLOOR IS 20 AGREEING POSITIONS, AND freeze_tau's
    # min_per_band COUNTS ANCHORED VALUES (up to five per position), so it
    # cannot enforce it. Enforced here, in positions, exactly as q4km_tau.py
    # records for the same reason.
    refused = []
    for band in list(table[PAIR]):
        if agreeing_positions.get(band, 0) < T3_MIN_AGREEING_POSITIONS:
            refused.append((band, agreeing_positions.get(band, 0)))
            del table[PAIR][band]

    rate = divergent / max(total, 1)
    growth = _band_growth(envelope)
    out = {"pair": PAIR, "tau": table[PAIR],
           "frozen_from": {"probes_total": total, "agreeing": total - divergent,
                           "divergent": divergent, "divergence_rate": rate,
                           "agreeing_positions_per_band": agreeing_positions},
           "refused_bands": refused,
           "band_p95_growth": growth,
           "tripwires": {
               "T1_divergence_rate": {"threshold": T1_DIVERGENCE_RATE,
                                      "value": rate, "fired": rate > T1_DIVERGENCE_RATE},
               "T3_thin_bands": {"threshold": T3_MIN_AGREEING_POSITIONS,
                                 "refused": refused, "fired": bool(refused)},
               "T4_band_growth": {"threshold": T4_BAND_GROWTH, "value": growth,
                                  "fired": any(v is not None and v > T4_BAND_GROWTH
                                               for v in growth.values())}},
           "freeze_report": report}
    tau_out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for line in report:
        print(line)
    for band, n in refused:
        print(f"  {band}: REFUSED after the position floor -- {n} agreeing "
              f"positions < {T3_MIN_AGREEING_POSITIONS}")
    print(f"divergence rate {rate:.4%} of {total} probes "
          f"(T1 threshold {T1_DIVERGENCE_RATE:.2%}"
          f"{' -- FIRED' if rate > T1_DIVERGENCE_RATE else ''})")
    return out


def _band_growth(envelope):
    """p95 growth from each band to the next OCCUPIED band (§8.9's collapse
    test, and T4). Reported whether or not it fires: a growth number that only
    appears when it is alarming cannot be compared across sittings."""
    by_band = {}
    for s in envelope:
        by_band.setdefault(cls.band_of(int(s["abs_position"])), []).extend(
            s["profile"]["anchored"])
    order = [cls.band_name(lo, hi) for lo, hi in cls.BANDS]
    occupied = [b for b in order if by_band.get(b)]
    out = {}
    for a, b in zip(occupied, occupied[1:]):
        pa = mcp.percentile(by_band[a], 95)
        pb = mcp.percentile(by_band[b], 95)
        out[f"{a}->{b}"] = (pb / pa) if pa > 0 else None
    return out


# ============================================================================
# phase C -- grade
# ============================================================================

def phase_c(out_dir: pathlib.Path, tau_file: pathlib.Path,
            verdicts_out: pathlib.Path | None):
    tau_doc = json.loads(tau_file.read_text(encoding="utf-8"))
    tau_table = {tau_doc["pair"]: tau_doc["tau"]}
    grids = load_grids(out_dir)
    records = []
    for g in grids:
        for p in g["positions"]:
            if p["stoke"]["chose"] == p["ref"]["chose"]:
                continue
            records.append({
                "prompt": g["stream"], "abs_position": p["abs_position"],
                "pair": tau_doc["pair"],
                "x": {"top5": p["ref"]["top5"], "chose": p["ref"]["chose"]},
                "y": {"top5": p["stoke"]["top5"], "chose": p["stoke"]["chose"]},
            })
    if not records:
        print("no divergent positions -- nothing to grade")
        verdicts = []
    else:
        verdicts = cls.classify_records(records, tau_table)
    counts = {}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    for v in verdicts:
        print(f"  {v['prompt']} pos={v['abs_position']} {v['band']} "
              f"{v['verdict']}: {v['reason']}")
    print(f"verdicts: {counts or '{}'}")
    bugs = counts.get("BUG", 0)
    print(f"PASS RULE: zero BUG-class -> "
          f"{'PASS' if bugs == 0 else f'FAIL ({bugs} BUG-class)'}")
    if verdicts_out:
        verdicts_out.write_text(json.dumps(
            {"pair": tau_doc["pair"], "tau": tau_doc["tau"],
             "counts": counts, "verdicts": verdicts}, indent=1), encoding="utf-8")
    return counts


# ============================================================================
# selftest -- everything above that needs neither model nor binary
# ============================================================================

def run_selftest() -> int:
    fails = []

    def check(label, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            fails.append(label)

    check("the corpus has 16 streams", len(STREAMS) == 16)
    check("stream ids are unique",
          len({s["id"] for s in STREAMS}) == len(STREAMS))
    # FOUR, not three: short1/2/3 plus long2 at its full length. The
    # pre-registration's prose said three while its own table flagged four, and
    # this check is what found the disagreement.
    check("exactly four streams are flagged selected-on-success",
          sum(1 for s in STREAMS if s["selected"]) == 4)
    check("the pair key is not a Qwen3/Q8_0 key",
          PAIR != mcp.PAIR and "q8_0" not in PAIR and "qwen" not in PAIR)

    # The position mapping is the one thing that, if wrong, silently misaligns
    # every comparison in the grid while looking perfectly healthy.
    n_prompt, k = 240, 7
    check("generated index k maps to absolute n_prompt-1+k",
          (n_prompt - 1 + k) == 246)
    check("gen index 0 of a 5-token prompt is absolute position 4",
          (5 - 1 + 0) == 4)

    # Band coverage of the declared targets, computed rather than asserted from
    # the table's prose: a target list that no longer tiles the bands is a
    # corpus defect and should be caught here, not after an hour of probing.
    covered = set()
    for s in STREAMS:
        t = s["target"] or {"S-a1": 5, "S-a2": 12, "S-a3": 17, "L-2851": 2851}[s["id"]]
        for k in range(N_GEN):
            covered.add(cls.band_of(t - 1 + k))
    want = {cls.band_name(lo, hi) for lo, hi in cls.BANDS}
    check(f"the declared targets tile every band {sorted(want)}", covered == want)

    # Tripwires must be constants, not expressions evaluated against results.
    check("T1 is the M3 rate (~1.8%) rounded up, not a post-hoc number",
          T1_DIVERGENCE_RATE == 0.02)
    check("T3 is the ruling's 20 agreeing POSITIONS", T3_MIN_AGREEING_POSITIONS == 20)
    check("T4 is §8.9's 1.5x collapse test", T4_BAND_GROWTH == 1.5)

    # delta_profile must refuse an unanchorable sample rather than admit it.
    a = [(1, 5.0), (2, 4.0), (3, 3.0), (4, 2.0), (5, 1.0)]
    b = [(9, 5.0), (8, 4.0), (7, 3.0), (6, 2.0), (5, 1.0)]
    check("an unanchorable sample is refused, not admitted to the envelope",
          mcp.delta_profile(a, b, 1) is None)
    prof = mcp.delta_profile(a, a, 1)
    check("an identical pair anchors to all-zero deltas",
          prof is not None and max(prof["anchored"]) == 0.0)

    print(f"\n{'ALL GREEN' if not fails else str(len(fails)) + ' FAILED'}")
    return 1 if fails else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("a", "b", "c"))
    p.add_argument("--out-dir")
    p.add_argument("--only")
    p.add_argument("--tau-out")
    p.add_argument("--tau")
    p.add_argument("--verdicts")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return run_selftest()
    if not a.phase or not a.out_dir:
        print("fatal: --phase and --out-dir are required", file=sys.stderr)
        return 1
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if a.phase == "a":
            phase_a(out_dir, set(a.only.split(",")) if a.only else None)
        elif a.phase == "b":
            if not a.tau_out:
                print("fatal: --phase b needs --tau-out", file=sys.stderr)
                return 1
            phase_b(out_dir, pathlib.Path(a.tau_out))
        else:
            if not a.tau:
                print("fatal: --phase c needs --tau (frozen by phase b FIRST)",
                      file=sys.stderr)
                return 1
            phase_c(out_dir, pathlib.Path(a.tau),
                    pathlib.Path(a.verdicts) if a.verdicts else None)
    except (ProbeError, mcp.ProbeError) as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
