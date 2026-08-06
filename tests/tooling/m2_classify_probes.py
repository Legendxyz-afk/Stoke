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
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

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


def probeable(row) -> bool:
    """False when one side stopped before the divergent index -- that side has
    no distribution there, so no top-5 exists to collect (see header)."""
    k = int(row["first_diff"])
    return k < len(row["m2_ids"]) and k < len(row["head_ids"])


def run_probe(exe: pathlib.Path, prompt: pathlib.Path, pos: int, n_predict: int):
    argv = [str(exe), "run", str(MODEL), "--raw", "-f", str(prompt),
            "-n", str(n_predict), "--probe-pos", str(pos)]
    try:
        p = subprocess.run(argv, capture_output=True, timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise ProbeError(f"probe timed out after {PROBE_TIMEOUT_S}s: {' '.join(argv)}")
    if p.returncode != 0:
        raise ProbeError(f"probe failed (exit {p.returncode}): {' '.join(argv)}\n"
                         f"stderr: {p.stderr.decode('utf-8', 'replace')[-400:]}")
    return parse_probe(p.stderr.decode("utf-8", "replace"), pos)


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
    ap.add_argument("--limit", type=int, default=0,
                    help="probe at most N divergent prompts (banking is per-prompt, "
                         "so a limited run is a prefix of the full one, not a "
                         "different experiment)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return run_selftest()
    if not (a.sweep and a.out):
        print("fatal: --sweep and --out are both required", file=sys.stderr)
        return 2

    live = sha256_of(HEAD_EXE)
    if live != HEAD_SHA:
        print(f"fatal: build/win/stoke.exe is {live[:8]}..., banked streams are "
              f"{HEAD_SHA[:8]}... -- probes from a rebuilt engine cannot explain "
              f"streams it never generated. Re-bank or restore.", file=sys.stderr)
        return 2

    sweep = json.loads(pathlib.Path(a.sweep).read_text())
    lens = json.loads(LENS.read_text())
    rows = divergent_rows(sweep)
    print(f"divergent prompts: {len(rows)} of {sweep.get('of', '?')}", flush=True)

    out_path = pathlib.Path(a.out)
    side_path = out_path.with_name(out_path.stem + "_unprobeable.json")
    records, unprobeable = [], []
    if out_path.exists():
        records = json.loads(out_path.read_text())
        done = {r["prompt"] for r in records}
        print(f"resuming: {len(done)} records already collected", flush=True)
    else:
        done = set()
    if side_path.exists():
        unprobeable = json.loads(side_path.read_text())
        done |= {r["prompt"] for r in unprobeable}

    todo = [r for r in rows if r["prompt"] not in done]
    if a.limit:
        todo = todo[:a.limit]

    for row in todo:
        pid = row["prompt"]
        k = int(row["first_diff"])
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
        if not probeable(row):
            short = "M2" if k >= len(row["m2_ids"]) else "HEAD"
            unprobeable.append({"prompt": pid, "position": k, "abs_position": plen + k,
                                "pair": PAIR, "reason": f"{short} stopped before index {k}; "
                                f"no top-5 exists on that side",
                                "verdict": "ESCALATE-BY-CONSTRUCTION"})
            side_path.write_text(json.dumps(unprobeable, indent=1))
            print(f"{pid}: UNPROBEABLE ({short} stopped before {k}) -> ESCALATE", flush=True)
            continue
        prompt = PROMPTS / f"{pid}.txt"
        t0 = time.time()
        head5 = run_probe(HEAD_EXE, prompt, k, len(row["head_ids"]))
        check_rank0(head5, int(row["head_ids"][k]), "HEAD", k)
        m2_5 = run_probe(M2_EXE, prompt, k, len(row["m2_ids"]))
        check_rank0(m2_5, int(row["m2_ids"][k]), "M2", k)
        records.append(build_record(row, plen, m2_5, head5))
        out_path.write_text(json.dumps(records, indent=1))
        print(f"{pid}: probed at k={k} abs={plen + k} "
              f"(M2 chose {row['m2_ids'][k]}, HEAD chose {row['head_ids'][k]}, "
              f"{time.time() - t0:.0f}s)", flush=True)

    print(f"\ncollected {len(records)} records, {len(unprobeable)} unprobeable")
    print(f"records -> {out_path}")
    if unprobeable:
        print(f"unprobeable -> {side_path}")
    print("\nNOT GRADED. Ruling 2.3 requires the probe grid to freeze tau first; "
          f"then:\n  python classify.py --records {out_path} --tau-table <frozen>.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ProbeError as e:
        print(f"fatal: {e}", file=sys.stderr)
        sys.exit(1)
