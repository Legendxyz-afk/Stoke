#!/usr/bin/env python3
"""tests/tooling/ref_gate.py -- M2's SUBSTITUTE token gate: stoke's C
forward pass vs ref_forward.py, both pure float32.

WHY THIS EXISTS (read before changing anything): M2's original gate was
greedy 64-id parity against llama-server. The 2026-07-28 survey measured
19 of 50 prompts diverging, 18 of 18 classifiable hits being a clean
top-2 swap at oracle gaps of 0.001-0.204 nats -- the oracle quantizes
activations to Q8 blocks for integer dot products while M2's stoke is
pure f32, so the two rank near-ties differently. Kabir took memo option 3
(docs/superpowers/memos/2026-07-20-m2-oracle-near-tie-policy.md): M2's
token gate is re-scoped to C-vs-ref, and ORACLE token-parity is DEFERRED
TO M3, where LOCKED decision #2's integer core puts stoke in the oracle's
own numeric class and the forward note already mandates re-running the
50-prompt oracle gate.

WHAT THIS COMPARES: `stoke run --raw -f <prompt> -n 64` against
`ref_forward.py <gguf> --ids <same ids> -n 64`. Both sides are pure f32,
so unlike the oracle gate there is no numeric-class gap between them --
only float32 non-associativity (numpy BLAS summation order vs stoke's
scalar C loops), which W3's L2 ladder bounds at 6.1e-4 in hidden state.

TOKENIZATION IS PROVEN IDENTICAL, NOT ASSUMED: ref takes --ids, so the
prompt must be tokenized here. src/main.c cmd_run's `--raw` branch calls
tok_encode(tok, text, text_len, ids, cap, add_special=false); cmd_tokenize
(without --special) calls tok_encode(t, buf, len, ids, cap,
add_special=false). Byte-for-byte the same call, verified by reading both
call sites. So `stoke tokenize` yields exactly the ids `stoke run --raw`
consumed.

STOKE IS NOT RE-RUN. Its 64 ids for all 50 corpus prompts are already
banked in parity_gate.py's .gate_cache, under a key containing the
stoke.exe sha256 -- this tool verifies that hash matches the current
binary and FATALS if it does not, rather than silently comparing against
a stale generation.

The ops lessons this file is built to respect (each was paid for once
already, see the ledger): the per-prompt timeout is DERIVED from the work
the input implies, never a flat guess (a flat 1800s ceiling below the
long-context tier's own 49-min cost killed four prompts on 07-27); every
prompt result is cached atomically so a kill costs only in-flight work;
progress is printed per prompt and unbuffered so a hung run is
distinguishable from a working one; stdout is forced to UTF-8 because
cp437 is a real Windows console codepage and an unmappable character in a
normal print path aborts a detached run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# cp437 lesson: force UTF-8 before anything can print.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

from parity_gate import compare_ids, GateError  # same comparator as the oracle gate

STOKE_EXE = REPO / "build" / "win" / "stoke.exe"
MODEL_PATH = REPO / "models" / "Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
CORPUS = REPO / "tests" / "data" / "parity"
REF = HERE / "ref_forward.py"
ORACLE_CACHE = HERE / ".gate_cache"        # parity_gate's, read-only here
CACHE = HERE / ".ref_gate_cache"           # this tool's own
N_PREDICT = 64

# Cost model. 13.8 s/position is W1's own end-to-end benchmark
# (.superpowers/sdd/m2-w1-notes.md: "--ids 1,2,3 -n 2, 5 total positions:
# 69s wall clock =~ 13.8s/position"). NOTE ref_forward.py's own comment
# says "~10-11s" -- that figure is only the per-layer dequant portion and
# understates the end-to-end cost; do not use it for sizing.
SEC_PER_POSITION = 13.8
TIMEOUT_SAFETY = 2.0     # >=2x measured cost, per the derived-ceiling lesson
TIMEOUT_FLOOR_S = 1800.0


def fatal(msg: str):
    raise GateError(msg)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def timeout_for(n_prompt: int, n_predict: int) -> float:
    """DERIVED ceiling, never a flat guess. Positions = prompt + generated;
    each costs SEC_PER_POSITION; multiply by a safety factor so a loaded
    box cannot manufacture a false timeout."""
    return max(TIMEOUT_FLOOR_S,
               (n_prompt + n_predict) * SEC_PER_POSITION * TIMEOUT_SAFETY)


# ---------------------------------------------------------------- tokenize

_IDS_RE = re.compile(r"^\[\s*(\d+(?:\s*,\s*\d+)*)?\s*\]\s*$")


def parse_ids_line(text: str) -> List[int]:
    """Parse cmd_tokenize's `[a, b, c]` output. Fails loud on any other
    shape -- a silently-misparsed prompt would invalidate the whole run."""
    line = None
    for cand in reversed(text.strip().splitlines()):
        if cand.strip():
            line = cand.strip()
            break
    if line is None:
        fatal("stoke tokenize produced no output")
    m = _IDS_RE.match(line)
    if not m:
        fatal(f"stoke tokenize output is not an id list: {line!r}")
    body = m.group(1)
    if not body:
        fatal("stoke tokenize produced an EMPTY id list")
    return [int(s) for s in body.split(",")]


def tokenize(prompt_path: Path) -> List[int]:
    argv = [str(STOKE_EXE), "tokenize", str(MODEL_PATH), str(prompt_path)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        fatal(f"stoke tokenize failed on {prompt_path.name} "
              f"(rc={r.returncode}): {r.stderr.strip()[:400]}")
    return parse_ids_line(r.stdout)


# --------------------------------------------------------------------- ref

def run_ref(ids: List[int], n_predict: int, timeout_s: float) -> List[int]:
    """Greedy-decode n_predict ids from the numpy f32 reference. ref prints
    one id per line and may stop early on EOS (which it prints)."""
    argv = [sys.executable, "-u", str(REF), str(MODEL_PATH),
            "--ids", ",".join(str(i) for i in ids), "-n", str(n_predict)]
    try:
        r = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout_s, cwd=str(HERE))
    except subprocess.TimeoutExpired:
        fatal(f"ref_forward timed out after {timeout_s:.0f}s "
              f"({len(ids)}+{n_predict} positions) -- the DERIVED ceiling "
              f"was exceeded, which means the cost model "
              f"({SEC_PER_POSITION}s/position) is wrong, not that the "
              f"process is necessarily wedged. Re-measure before raising it.")
    if r.returncode != 0:
        fatal(f"ref_forward failed (rc={r.returncode}): "
              f"{r.stderr.strip()[:600]}")
    out: List[int] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.lstrip("-").isdigit():
            fatal(f"ref_forward printed a non-id line: {line!r}")
        out.append(int(line))
    if not out:
        fatal("ref_forward printed no ids")
    return out


# ------------------------------------------------------------------- cache

def cache_key(prompt_path: Path, stoke_hash: str, ref_hash: str,
              n_predict: int) -> dict:
    return {
        "prompt_sha256": sha256_file(prompt_path),
        "stoke_sha256": stoke_hash,
        "ref_sha256": ref_hash,
        "model": MODEL_PATH.name,
        "n_predict": n_predict,
        "tool": "ref_gate/1",
    }


def cache_path(key: dict) -> Path:
    h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return CACHE / f"{h}.json"


def cache_load(key: dict) -> Optional[dict]:
    p = cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))["data"]
    except Exception:
        return None


def cache_store(key: dict, prompt_id: str, data: dict):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = cache_path(key)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"prompt_id": prompt_id, "key": key,
                               "data": data}, indent=1), encoding="utf-8")
    os.replace(tmp, p)          # atomic


def load_stoke_ids(prompt_id: str, stoke_hash: str) -> List[int]:
    """Read stoke's banked 64 ids out of parity_gate's cache. FATALS if the
    entry is missing or was produced by a different stoke.exe -- never
    silently compares against a stale generation."""
    if not ORACLE_CACHE.is_dir():
        fatal(f"{ORACLE_CACHE} does not exist -- stoke ids come from the "
              f"parity_gate survey cache; run the survey first")
    for f in ORACLE_CACHE.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(d.get("prompt_path", "")).stem != prompt_id:
            continue
        key = d.get("key", {})
        if key.get("stoke_hash") != stoke_hash:
            fatal(f"{prompt_id}: cached stoke ids were produced by "
                  f"stoke.exe {key.get('stoke_hash','?')[:12]}… but the "
                  f"current binary is {stoke_hash[:12]}… -- refusing to "
                  f"compare against a stale generation")
        # The cache is keyed on n_predict too; an entry generated at a
        # different length would silently supply a shorter/longer id stream.
        if key.get("n_predict") != N_PREDICT:
            fatal(f"{prompt_id}: cached stoke ids were generated at "
                  f"n_predict={key.get('n_predict')}, not {N_PREDICT}")
        ids = d.get("data", {}).get("stoke_ids")
        if not ids:
            fatal(f"{prompt_id}: cache entry has no stoke_ids")
        return ids
    fatal(f"{prompt_id}: no entry in {ORACLE_CACHE.name} -- stoke has not "
          f"generated this prompt")


# -------------------------------------------------------------------- work

def one_prompt(prompt_id: str, stoke_hash: str, ref_hash: str,
               n_predict: int) -> dict:
    path = CORPUS / f"{prompt_id}.txt"
    if not path.exists():
        fatal(f"missing corpus file {path}")
    key = cache_key(path, stoke_hash, ref_hash, n_predict)
    hit = cache_load(key)
    if hit is not None:
        hit["cached"] = True
        return hit

    t0 = time.time()
    prompt_ids = tokenize(path)
    ref_ids = run_ref(prompt_ids, n_predict, timeout_for(len(prompt_ids), n_predict))
    stoke_ids = load_stoke_ids(prompt_id, stoke_hash)

    c = compare_ids(ref_ids, stoke_ids)
    data = {
        "prompt_id": prompt_id,
        "n_prompt": len(prompt_ids),
        "n_ref": len(ref_ids),
        "n_stoke": len(stoke_ids),
        "identical": c.identical,
        "first_diff_pos": c.first_diff_pos,
        "ref_ids": ref_ids,
        "stoke_ids": stoke_ids,
        "elapsed_s": round(time.time() - t0, 1),
        "sec_per_position": round((time.time() - t0) / max(1, len(prompt_ids) + n_predict), 2),
        "cached": False,
    }
    if not c.identical and c.first_diff_pos is not None:
        p = c.first_diff_pos
        data["ref_id_at_diff"] = ref_ids[p] if p < len(ref_ids) else None
        data["stoke_id_at_diff"] = stoke_ids[p] if p < len(stoke_ids) else None
    cache_store(key, prompt_id, data)
    return data


def run_gate(prompt_ids: List[str], jobs: int, n_predict: int) -> int:
    if not STOKE_EXE.exists():
        fatal(f"missing {STOKE_EXE}")
    if not MODEL_PATH.exists():
        fatal(f"missing {MODEL_PATH}")
    stoke_hash = sha256_file(STOKE_EXE)
    ref_hash = sha256_file(REF)

    print(f"REF GATE -- stoke C vs ref_forward.py, both pure f32")
    print(f"  model      : {MODEL_PATH.name}")
    print(f"  stoke.exe  : {stoke_hash[:16]}…")
    print(f"  ref_forward: {ref_hash[:16]}…")
    print(f"  prompts    : {len(prompt_ids)} -- {','.join(prompt_ids)}")
    print(f"  jobs       : {jobs}   n_predict: {n_predict}")
    print(f"  ceiling    : max({TIMEOUT_FLOOR_S:.0f}s, positions x "
          f"{SEC_PER_POSITION}s x {TIMEOUT_SAFETY}) -- DERIVED per prompt")
    sys.stdout.flush()

    # Longest first: the makespan is set by the long-context prompts, so
    # submitting them last would strand the tail (same lesson as the survey).
    # Ordered on MANIFEST's real token counts, the same source the ceiling
    # uses -- bytes would misrank multibyte scripts against ASCII.
    counts = manifest_token_counts()
    ordered = sorted(prompt_ids, key=lambda p: counts.get(p, 0), reverse=True)

    results: Dict[str, dict] = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one_prompt, p, stoke_hash, ref_hash, n_predict): p
                for p in ordered}
        # AS COMPLETED, not submission order: submission is longest-first, so
        # draining in submission order would hold every progress line hostage
        # to the slowest prompt and make a working run look hung.
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                results[pid] = fut.result()
            except GateError as e:
                results[pid] = {"prompt_id": pid, "error": str(e)}
            done += 1
            r = results[pid]
            el = (time.time() - t0) / 60.0
            if "error" in r:
                print(f"[{done}/{len(ordered)}] {pid} ERROR -- {r['error'][:200]}")
            else:
                tag = "EXACT" if r["identical"] else f"DIVERGED@{r['first_diff_pos']}"
                cc = " (cached)" if r.get("cached") else f" {r['sec_per_position']}s/pos"
                print(f"[{done}/{len(ordered)}] {pid} {tag}{cc} -- elapsed {el:.1f} min")
            sys.stdout.flush()

    print()
    exact = [p for p, r in results.items() if r.get("identical")]
    diverged = [p for p, r in results.items() if "error" not in r and not r.get("identical")]
    errors = [p for p, r in results.items() if "error" in r]
    print(f"RESULT: {len(exact)} exact, {len(diverged)} diverged, {len(errors)} error "
          f"of {len(prompt_ids)}")
    if diverged:
        print("DIVERGED:")
        for p in sorted(diverged):
            r = results[p]
            print(f"  {p} at position {r['first_diff_pos']}: "
                  f"ref={r.get('ref_id_at_diff')} stoke={r.get('stoke_id_at_diff')}")
    if errors:
        print("ERRORS:")
        for p in sorted(errors):
            print(f"  {p}: {results[p]['error'][:300]}")

    report = CACHE / "ref_gate_report.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "model": MODEL_PATH.name, "stoke_sha256": stoke_hash,
        "ref_sha256": ref_hash, "n_predict": n_predict,
        "jobs": jobs, "prompts": prompt_ids,
        "sec_per_position_model": SEC_PER_POSITION,
        "results": results,
    }, indent=1), encoding="utf-8")
    print(f"\nreport: {report}")
    return 0 if (not diverged and not errors) else 1


# ============================================================================
# WAIVER PROTOCOL -- RATIFIED AS BUILT by the actual Fable session,
# 2026-07-28: docs/superpowers/memos/2026-07-28-m2-substitute-gate-RULING-
# FABLE.md is the authority (it also fixes what this block alone does not:
# stop-index equality, first-divergence-only adjudication, no-chaining for
# reserve replacements, the calibration rationale). Provenance, corrected:
# this block was transcribed from the 07-28 draft that a silent model-
# override bug had produced under Sonnet, not Fable; the ruling ratified the
# four conditions unchanged, so the code needed no change -- only this header.
#
# Built BEFORE the measurement run produced any result, deliberately: fixing
# adjudication logic after seeing which prompt flipped is gate-loosening by
# another name. Nothing below may be relaxed without a fresh Fable ruling.
#
# A divergence is waivable iff ALL FOUR hold at the first-divergence position p:
#   1. p is identified. Trivial on this pair -- both sides emit one id per
#      line, so none of the oracle's coalesced-entry alignment ambiguity
#      exists here.
#   2. Both sides' top-2 are the SAME PAIR, SWAPPED.
#   3. BOTH sides' gap(top1,top2) < EPSILON = 1e-3 nats. This is §8's
#      already-approved constant reused verbatim, not a new number. A raw
#      logit difference IS the nats gap -- the softmax normalizer cancels in
#      the subtraction, identically on both sides.
#   4. Both sides' top-5 have IDENTICAL IDS IN IDENTICAL RANK ORDER at ranks
#      3, 4 and 5 (0-indexed 2,3,4). Ranks 1-2 may differ in order (that is
#      condition 2's business) but not in membership.
#      Rationale, recorded: a last-ulp coincidence at the argmax boundary has
#      no mechanism to also permute the ordering three ranks deep into a
#      151,936-way distribution; a real formula or accumulation bug generically
#      would. Fable's original condition 4 (re-feed the prefix and diff hidden
#      state) was withdrawn because `stoke run` accepts no token-id input --
#      only -p/-f text -- so it could not be done without either rebuilding
#      stoke.exe (forking the hash the whole evidentiary chain is pinned to)
#      or detokenizing, which can silently change the prefix.
#
# If EITHER side's probe cannot be taken (early EOS before p, so the position
# does not exist), the divergence is NOT waivable: absence of evidence
# defaults to FAIL, never to a free pass.
# ============================================================================

EPSILON = 1e-3        # §8 nats. NEVER loosen.
WAIVER_BUDGET = 2     # §8. NEVER exceed.
RESERVE_IDS = [f"r{i:02d}" for i in range(10)]

_PROBE_RE = re.compile(r"^probe pos=(\d+): id=(\d+) logit=(-?\d+\.\d+)\s*$")


def parse_probe(text: str, pos: int) -> List[Tuple[int, float]]:
    """Parse the 5 stderr probe lines. ONE parser for both sides -- stoke's
    src/main.c and ref_forward.py emit byte-identical
    `probe pos=N: id=U logit=F`. Fails loud on any other shape."""
    rows = []
    for line in text.splitlines():
        m = _PROBE_RE.match(line.strip())
        if m:
            if int(m.group(1)) != pos:
                fatal(f"probe line reports pos={m.group(1)}, expected {pos}")
            rows.append((int(m.group(2)), float(m.group(3))))
    if len(rows) != 5:
        fatal(f"expected exactly 5 probe lines at pos {pos}, parsed "
              f"{len(rows)} -- refusing to adjudicate on partial evidence.\n"
              f"raw stderr:\n{text.strip()[:800]}")
    for i in range(4):
        if rows[i][1] < rows[i + 1][1]:
            fatal(f"probe lines are not rank-ordered at {i}: {rows}")
    return rows


def probe_lines(text: str) -> List[str]:
    """The `probe pos=...` lines exactly as the process printed them. The
    gate report records THESE, not a re-rendering of the parsed numbers --
    ruling §3.2 requires both 5-line blocks verbatim, and a re-render of a
    float would not be verbatim."""
    return [ln.rstrip("\r\n") for ln in text.splitlines()
            if _PROBE_RE.match(ln.strip())]


def stoke_probe(prompt_path: Path, pos: int, n_predict: int,
                n_tok: int) -> Tuple[List[Tuple[int, float]], List[str]]:
    """`--probe-pos N` indexes the GENERATED ids on both sides -- the same
    space as compare_ids' first_diff_pos (src/main.c:527 bounds it against
    the generated count; ref_forward.py:356 against --n). Verified by
    reading both, not assumed: probing the wrong space would adjudicate a
    different position than the one that diverged.

    The ceiling is DERIVED from this prompt's real MANIFEST token count --
    a probe re-generates the whole prompt, so the cost is the generation
    cost, and a flat guess here would repeat the 07-27 timeout lesson in
    miniature. SEC_PER_POSITION is ref's rate (13.8s), ~2.7x stoke's
    measured 5.2s, so the stoke side is bounded generously on purpose."""
    argv = [str(STOKE_EXE), "run", str(MODEL_PATH), "--raw",
            "-f", str(prompt_path), "-n", str(n_predict), "--probe-pos", str(pos)]
    try:
        r = subprocess.run(argv, capture_output=True, text=True,
                           timeout=timeout_for(n_tok, n_predict))
    except subprocess.TimeoutExpired:
        fatal(f"stoke probe on {prompt_path.name} at pos {pos} exceeded its "
              f"derived ceiling ({timeout_for(n_tok, n_predict):.0f}s)")
    if r.returncode != 0:
        fatal(f"stoke probe failed on {prompt_path.name} at pos {pos} "
              f"(rc={r.returncode}): {r.stderr.strip()[:400]}")
    return parse_probe(r.stderr, pos), probe_lines(r.stderr)


def ref_probe(ids: List[int], pos: int,
              n_predict: int) -> Tuple[List[Tuple[int, float]], List[str]]:
    argv = [sys.executable, "-u", str(REF), str(MODEL_PATH),
            "--ids", ",".join(str(i) for i in ids), "-n", str(n_predict),
            "--probe-pos", str(pos)]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, cwd=str(HERE),
                           timeout=timeout_for(len(ids), n_predict))
    except subprocess.TimeoutExpired:
        fatal(f"ref probe at pos {pos} exceeded its derived ceiling "
              f"({timeout_for(len(ids), n_predict):.0f}s)")
    if r.returncode != 0:
        fatal(f"ref probe failed at pos {pos} (rc={r.returncode}): "
              f"{r.stderr.strip()[:400]}")
    return parse_probe(r.stderr, pos), probe_lines(r.stderr)


def ref_waiver_decision(stoke_top5, ref_top5) -> dict:
    """Pure. Fable's four conditions; no I/O, no budget -- see WaiverLedger."""
    if len(stoke_top5) != 5 or len(ref_top5) != 5:
        fatal("waiver: both sides need exactly 5 probe entries")
    s_gap = stoke_top5[0][1] - stoke_top5[1][1]
    r_gap = ref_top5[0][1] - ref_top5[1][1]
    s_pair = (stoke_top5[0][0], stoke_top5[1][0])
    r_pair = (ref_top5[0][0], ref_top5[1][0])

    c2 = (s_pair[0] == r_pair[1] and s_pair[1] == r_pair[0])
    c3 = (s_gap < EPSILON) and (r_gap < EPSILON)
    s_tail = [i for i, _ in stoke_top5[2:5]]
    r_tail = [i for i, _ in ref_top5[2:5]]
    c4 = (s_tail == r_tail)

    if c2 and c3 and c4:
        eligible, why = True, "genuine near-tie: top-2 pair swapped, both gaps < eps, ranks 3-5 identical"
    elif not c2:
        eligible, why = False, f"top-2 not a clean swap: stoke{s_pair} vs ref{r_pair}"
    elif not c3:
        eligible, why = False, f"gap >= eps ({EPSILON}): stoke_gap={s_gap:.6g}, ref_gap={r_gap:.6g}"
    else:
        eligible, why = False, f"ranks 3-5 differ: stoke{s_tail} vs ref{r_tail}"

    return {"eligible": eligible, "reason": why, "stoke_gap": s_gap,
            "ref_gap": r_gap, "stoke_pair": s_pair, "ref_pair": r_pair,
            "stoke_tail": s_tail, "ref_tail": r_tail,
            "cond2_pair_swapped": c2, "cond3_both_gaps_under_eps": c3,
            "cond4_ranks_3_5_identical": c4}


class WaiverLedger:
    """Budget is a hard global ceiling across the whole gate run, and each
    waived prompt consumes the next reserve prompt in order."""

    def __init__(self, budget: int = WAIVER_BUDGET, reserve=None):
        self.budget = budget
        self._reserve = iter(reserve if reserve is not None else RESERVE_IDS)
        self.records: List[dict] = []

    def used(self) -> int:
        return sum(1 for r in self.records if r["waived"])

    def try_waive(self, prompt_id: str, pos: int, decision: dict) -> dict:
        rec = {"prompt_id": prompt_id, "position": pos, "waived": False,
               "replacement": None, **decision}
        if not decision["eligible"]:
            rec["outcome"] = f"NOT WAIVABLE -- {decision['reason']}"
        elif self.used() >= self.budget:
            rec["outcome"] = (f"eligible but BUDGET EXHAUSTED "
                              f"({self.used()}/{self.budget} already used)")
        else:
            try:
                rec["replacement"] = next(self._reserve)
            except StopIteration:
                fatal("reserve pool exhausted -- unreachable with budget 2 "
                      "and 10 reserve prompts; something is very wrong")
            rec["waived"] = True
            rec["outcome"] = (f"WAIVED -- replaced by {rec['replacement']}, "
                              f"which must itself pass")
        self.records.append(rec)
        return rec


# ============================================================================
# ADJUDICATION -- ruling §3 and §4.1.
#
# The measurement run records comparisons and NOTHING else, deliberately
# (§3.1). This is the separate step that turns them into a verdict, and its
# one hard rule is that it never re-derives the bar: every divergence goes
# through the committed ref_waiver_decision / WaiverLedger above, and this
# code only sequences them and reports what they returned.
#
# What it enforces, straight off ruling §2 -- PASS requires ALL of:
#   1. zero unresolved tooling errors (§2.1),
#   2. token-exact incl. stop index on all 24, <=2 waivers, each waived
#      prompt replaced by the next reserve IN ORDER, replacements
#      token-exact with NO waiver eligibility (§2.2, no chaining),
#   3. the L2 ladder re-run at gate HEAD (§2.3) -- a SEPARATE leg run by
#      l2_diff.py, which this tool deliberately does not fold in: it never
#      reports "GATE PASS", only "TOKEN LEG PASS", so the report cannot be
#      stamped off this exit code alone.
#
# Stop-index equality needs no code here: compare_ids already reports a
# length mismatch as a divergence AT the shorter sequence's length, so an
# early EOS on one side arrives as an ordinary first-divergence position --
# and the probe at that position is then unobtainable on the short side,
# which §2 says must FAIL rather than be waived. That path is exercised by
# the probe-unobtainable branch below.
#
# TIE-BREAK, recorded because the ruling does not pin it: when more than one
# prompt diverges, they are adjudicated in ascending prompt-id order, so the
# reserve draw (r00, then r01) is deterministic and reproducible. This is a
# sequencing choice, not a criterion -- no outcome depends on it, since
# every divergence must be individually eligible regardless of draw order.
# ============================================================================

# Ruling §1, verbatim -- and identical to what the launched task carries
# (run_ref_gate_detached.cmd line 56). Adjudicating a run whose prompt set
# is not exactly this is refused: a partial or edited subset is not the
# gate, whatever its results say.
RULING_SUBSET = ["p01", "p02", "p05", "p08", "p09", "p11", "p12", "p16",
                 "p20", "p24", "p25", "p26", "p28", "p29", "p37", "p39",
                 "p41", "p43", "p44", "p45", "p46", "p47", "p48", "p49"]


def adjudicate_report(report: dict, probe_both, subset=None) -> dict:
    """Pure sequencing over an already-recorded report. `probe_both` is
    injected so the whole verdict path is testable without live processes:
    probe_both(prompt_id, pos) -> {"stoke_top5", "ref_top5", "stoke_lines",
    "ref_lines"} or {"error": str}.

    Returns a verdict dict; never raises on a gate outcome (only on a
    malformed report), because "FAIL" is a result to record, not an
    exception to escape with."""
    subset = list(subset if subset is not None else RULING_SUBSET)
    results = report.get("results")
    if not isinstance(results, dict) or not results:
        fatal("report has no results block")

    out = {"subset": subset, "verdict": None, "fail_reasons": [],
           "pending": [], "divergences": [], "waivers": [],
           "n_exact": 0, "blocked_by": None}

    errors = sorted(p for p, r in results.items() if "error" in r)
    if errors:
        out["verdict"] = "FAIL"
        out["blocked_by"] = "tooling"
        out["fail_reasons"].append(
            f"unresolved tooling error(s) on {','.join(errors)} -- ruling "
            f"§2.1: fix the tooling and re-run those prompts; the gate can "
            f"never pass around an unresolved error, and adjudicating one "
            f"would spend hours of probes on an invalid run")
        return out

    missing = [p for p in subset if p not in results]
    if missing:
        out["verdict"] = "INCOMPLETE"
        out["blocked_by"] = "coverage"
        out["fail_reasons"].append(
            f"{len(missing)} subset prompt(s) absent from the report: "
            f"{','.join(missing)}")
        return out

    diverged = sorted(p for p in subset if not results[p].get("identical"))
    out["n_exact"] = len(subset) - len(diverged)

    ledger = WaiverLedger()
    for pid in diverged:
        r = results[pid]
        pos = r.get("first_diff_pos")
        rec = {"prompt_id": pid, "position": pos,
               "ref_id_at_diff": r.get("ref_id_at_diff"),
               "stoke_id_at_diff": r.get("stoke_id_at_diff"),
               "n_ref": r.get("n_ref"), "n_stoke": r.get("n_stoke")}
        if pos is None:
            out["fail_reasons"].append(
                f"{pid}: diverged but the report carries no first_diff_pos")
            rec["outcome"] = "NOT WAIVABLE -- no divergence position recorded"
            out["divergences"].append(rec)
            continue

        pr = probe_both(pid, pos)
        if "error" in pr:
            # §2: "any probe unobtainable on either side at p (fatal or
            # position never generated) -- absence of evidence defaults to
            # FAIL, never a free pass." This is also where a stop-index
            # divergence lands: the short side cannot be probed at p.
            rec["probe_error"] = pr["error"]
            rec["outcome"] = "NOT WAIVABLE -- probe unobtainable"
            out["fail_reasons"].append(
                f"{pid} at position {pos}: probe unobtainable -- {pr['error'][:200]}")
            out["divergences"].append(rec)
            continue

        rec["stoke_probe_lines"] = pr["stoke_lines"]
        rec["ref_probe_lines"] = pr["ref_lines"]
        decision = ref_waiver_decision(pr["stoke_top5"], pr["ref_top5"])
        waiver = ledger.try_waive(pid, pos, decision)
        rec["decision"] = decision
        rec["outcome"] = waiver["outcome"]
        rec["waived"] = waiver["waived"]
        rec["replacement"] = waiver["replacement"]
        out["divergences"].append(rec)
        out["waivers"].append(waiver)
        if not waiver["waived"]:
            out["fail_reasons"].append(f"{pid} at position {pos}: {waiver['outcome']}")

    # Replacements: token-exact, and NOT waiver-eligible (§2.2, no chaining).
    for w in out["waivers"]:
        if not w["waived"]:
            continue
        rep = w["replacement"]
        if rep not in results:
            out["pending"].append(rep)
            continue
        rr = results[rep]
        if "error" in rr:
            out["fail_reasons"].append(f"replacement {rep}: tooling error -- {rr['error'][:200]}")
        elif not rr.get("identical"):
            out["fail_reasons"].append(
                f"replacement {rep} diverged at position {rr.get('first_diff_pos')} "
                f"-- replacements carry NO waiver eligibility (ruling §2.2, "
                f"no chaining), so this fails the gate outright")

    if out["fail_reasons"]:
        out["verdict"] = "FAIL"
    elif out["pending"]:
        out["verdict"] = "INCOMPLETE"
        out["blocked_by"] = "replacement"
    else:
        out["verdict"] = "PASS"
    return out


def probe_cache_key(prompt_id: str, pos: int, stoke_hash: str,
                    ref_hash: str, n_predict: int) -> dict:
    return {"prompt_id": prompt_id, "probe_pos": pos,
            "stoke_sha256": stoke_hash, "ref_sha256": ref_hash,
            "model": MODEL_PATH.name, "n_predict": n_predict,
            "tool": "ref_gate/probe/1"}


def run_adjudication(n_predict: int) -> int:
    if not STOKE_EXE.exists():
        fatal(f"missing {STOKE_EXE}")
    report_path = CACHE / "ref_gate_report.json"
    if not report_path.exists():
        fatal(f"no report at {report_path} -- adjudication runs on a "
              f"completed measurement run, not instead of one")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    stoke_hash = sha256_file(STOKE_EXE)
    ref_hash = sha256_file(REF)
    # The evidentiary pin (ruling §4.3). If either hash moved, the report
    # describes a different pair of binaries than the probes would exercise.
    if report.get("stoke_sha256") != stoke_hash:
        fatal(f"stoke.exe is {stoke_hash[:12]}… but the report was produced "
              f"by {str(report.get('stoke_sha256'))[:12]}… -- the freeze was "
              f"broken; probes would not describe the recorded run")
    if report.get("ref_sha256") != ref_hash:
        fatal(f"ref_forward.py is {ref_hash[:12]}… but the report was "
              f"produced by {str(report.get('ref_sha256'))[:12]}… -- frozen "
              f"until the gate report is stamped")
    if report.get("n_predict") != n_predict:
        fatal(f"report n_predict={report.get('n_predict')}, adjudicating at "
              f"{n_predict}")
    ran = [p for p in report.get("prompts", []) if p.startswith("p")]
    if sorted(ran) != sorted(RULING_SUBSET):
        fatal(f"the report's corpus prompts are not the ruling's subset "
              f"(§1). ran={','.join(sorted(ran))}")

    counts = manifest_token_counts()

    def probe_both(prompt_id: str, pos: int) -> dict:
        key = probe_cache_key(prompt_id, pos, stoke_hash, ref_hash, n_predict)
        hit = cache_load(key)
        if hit is not None:
            print(f"  probes {prompt_id}@{pos}: cached")
            sys.stdout.flush()
            return hit
        path = CORPUS / f"{prompt_id}.txt"
        n_tok = counts.get(prompt_id)
        if n_tok is None:
            return {"error": f"{prompt_id} has no MANIFEST token count"}
        try:
            print(f"  probing {prompt_id}@{pos} (stoke ~{(n_tok+n_predict)*5.2/60:.0f} min, "
                  f"ref ~{(n_tok+pos)*SEC_PER_POSITION/60:.0f} min) …")
            sys.stdout.flush()
            ids = tokenize(path)
            s_rows, s_lines = stoke_probe(path, pos, n_predict, n_tok)
            r_rows, r_lines = ref_probe(ids, pos, n_predict)
        except GateError as e:
            # Errors are NEVER cached (survey lesson): a cached failure would
            # make a fixable probe permanently unobtainable.
            return {"error": str(e)}
        data = {"stoke_top5": s_rows, "ref_top5": r_rows,
                "stoke_lines": s_lines, "ref_lines": r_lines}
        cache_store(key, prompt_id, data)
        return data

    print("REF GATE ADJUDICATION -- ruling "
          "docs/superpowers/memos/2026-07-28-m2-substitute-gate-RULING-FABLE.md")
    print(f"  stoke.exe  : {stoke_hash[:16]}…")
    print(f"  ref_forward: {ref_hash[:16]}…")
    print(f"  subset     : {len(RULING_SUBSET)} prompts (ruling §1)")
    print(f"  probes run SERIALLY -- a ref probe is a full 2.3 GB process "
          f"(ruling §3.2)\n")
    sys.stdout.flush()

    v = adjudicate_report(report, probe_both)

    print(f"\nexact: {v['n_exact']}/{len(v['subset'])}   "
          f"divergences: {len(v['divergences'])}")
    for rec in v["divergences"]:
        print(f"\n--- {rec['prompt_id']} @ position {rec['position']} "
              f"(ref={rec.get('ref_id_at_diff')} stoke={rec.get('stoke_id_at_diff')})")
        for label, lines in (("stoke", rec.get("stoke_probe_lines")),
                             ("ref  ", rec.get("ref_probe_lines"))):
            if lines:
                print(f"  {label} probe (verbatim):")
                for ln in lines:
                    print(f"    {ln}")
        if "decision" in rec:
            print(f"  decision: {json.dumps(rec['decision'], sort_keys=True)}")
        print(f"  OUTCOME: {rec['outcome']}")

    adj = CACHE / "ref_gate_adjudication.json"
    adj.write_text(json.dumps(v, indent=1), encoding="utf-8")
    print(f"\nadjudication: {adj}")

    print(f"\nTOKEN LEG: {v['verdict']}")
    for r in v["fail_reasons"]:
        print(f"  - {r}")
    for p in v["pending"]:
        print(f"  - replacement {p} not yet run. Bank its stoke ids through "
              f"the hash-guarded path (ruling §4.2), then re-run this gate "
              f"and re-adjudicate:\n"
              f"      python tests/tooling/parity_gate.py --only {p} …\n"
              f"      python tests/tooling/ref_gate.py --subset "
              f"{','.join(RULING_SUBSET)},{p} --jobs 3\n"
              f"      python tests/tooling/ref_gate.py --adjudicate")
    if v["verdict"] == "PASS":
        print("\nThis is the TOKEN leg only. GATE PASS additionally requires "
              "the L2 ladder re-run at this HEAD (ruling §2.3): ids "
              "[785,6722,315,9625,374], 36/36 layers <=1e-3, no STEP, via "
              "l2_diff.py. Do not stamp the gate report off this line alone.")
    return 0 if v["verdict"] == "PASS" else 1


_MANIFEST_ROW = re.compile(r"^\|\s*([pr]\d\d)\s*\|[^|]*\|\s*(\d+)\s*\|")


def manifest_token_counts() -> Dict[str, int]:
    """Real tokenizer counts from tests/data/parity/MANIFEST.md, whose table
    records the actual `stoke tokenize` output (verified in the corpus
    report). Used for sizing so no ceiling rests on a bytes/token guess."""
    man = CORPUS / "MANIFEST.md"
    if not man.exists():
        fatal(f"missing {man}")
    out: Dict[str, int] = {}
    for line in man.read_text(encoding="utf-8").splitlines():
        m = _MANIFEST_ROW.match(line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    if not out:
        fatal("MANIFEST.md parsed to zero token counts -- table shape changed")
    return out


def run_selftest() -> int:
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"[{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and bool(cond)

    check("parse_ids_line basic", parse_ids_line("[1, 2, 3]") == [1, 2, 3])
    check("parse_ids_line single", parse_ids_line("[42]") == [42])
    check("parse_ids_line ignores preamble",
          parse_ids_line("noise\n[10234, 30]") == [10234, 30])
    for bad in ("", "[]", "1,2,3", "[1, x]", "[1 2]"):
        try:
            parse_ids_line(bad)
            check(f"parse_ids_line rejects {bad!r}", False)
        except GateError:
            check(f"parse_ids_line rejects {bad!r}", True)

    # Ceiling checks run off MANIFEST's REAL tokenizer counts, not a
    # bytes/token guess -- the whole derived-ceiling lesson was that sizing
    # must come from the work the input actually implies.
    counts = manifest_token_counts()
    n_corpus = sum(1 for k in counts if k.startswith("p"))
    n_reserve = sum(1 for k in counts if k.startswith("r"))
    # Both classes matter: the corpus is what the gate runs, the reserve is
    # what a waiver replacement is drawn from. The row regex covers [pr]\d\d
    # precisely so a reserve replacement can be sized like any other prompt.
    check(f"MANIFEST parsed 50 corpus + 10 reserve token counts "
          f"(got {n_corpus} + {n_reserve})",
          n_corpus == 50 and n_reserve == 10)
    flat_would_fail = [pid for pid, n in counts.items()
                       if (n + N_PREDICT) * SEC_PER_POSITION > 1800.0]
    check(f"a flat 1800s ceiling WOULD have failed {len(flat_would_fail)} "
          f"prompts ({','.join(sorted(flat_would_fail)[:6])}) -- so the "
          f"derived formula is not equivalent to a constant",
          len(flat_would_fail) > 0)
    # The real assertion: the ceiling for the WORST prompt must exceed an
    # independently-stated figure, so this check can actually fail if the
    # cost model or safety factor is ever weakened.
    worst_pid = max(counts, key=lambda p: counts[p])
    worst_ceiling = timeout_for(counts[worst_pid], N_PREDICT)
    check(f"worst prompt {worst_pid} ({counts[worst_pid]} tok) gets a ceiling "
          f"of {worst_ceiling/3600:.2f} h, > the 4.1 h its 554 positions imply "
          f"at 2x", worst_ceiling >= 4.1 * 3600)

    # ---- adjudication (ruling §3/§4.1). Every check below can fail: the
    # verdict paths are exercised against synthetic reports with an injected
    # probe function, so no live process is needed and no outcome is assumed.
    check(f"RULING_SUBSET is the 24 of ruling §1 (got {len(RULING_SUBSET)})",
          len(RULING_SUBSET) == 24 and len(set(RULING_SUBSET)) == 24)
    cmd = HERE / "run_ref_gate_detached.cmd"
    m = re.search(r"--subset\s+([pr\d,]+)", cmd.read_text(encoding="utf-8")) if cmd.exists() else None
    check("the launched task's --subset is exactly RULING_SUBSET "
          "(so adjudication cannot describe a different run than the one "
          "the scheduled task performs)",
          m is not None and m.group(1).split(",") == RULING_SUBSET)
    check("every subset prompt has a corpus file",
          all((CORPUS / f"{p}.txt").exists() for p in RULING_SUBSET))

    probe_block = ("probe pos=7: id=11 logit=1.000000\n"
                   "probe pos=7: id=22 logit=0.999500\n"
                   "probe pos=7: id=33 logit=0.500000\n"
                   "probe pos=7: id=44 logit=0.400000\n"
                   "probe pos=7: id=55 logit=0.300000\n")
    check("parse_probe reads a well-formed block",
          parse_probe(probe_block, 7)[0] == (11, 1.0))
    check("probe_lines keeps the 5 lines verbatim",
          probe_lines("loading…\n" + probe_block) == probe_block.strip().splitlines())
    for label, bad, pos in (("4 lines", "\n".join(probe_block.splitlines()[:4]), 7),
                            ("wrong pos", probe_block, 8),
                            ("unordered", probe_block.replace("1.000000", "0.100000"), 7)):
        try:
            parse_probe(bad, pos)
            check(f"parse_probe rejects {label}", False)
        except GateError:
            check(f"parse_probe rejects {label}", True)

    def _top5(pair, gap, tail=(33, 44, 55)):
        return ([(pair[0], 1.0), (pair[1], 1.0 - gap)]
                + [(t, 0.5 - i * 0.01) for i, t in enumerate(tail)])

    swap_ok = {"stoke_top5": _top5((11, 22), 5e-4), "ref_top5": _top5((22, 11), 6e-4),
               "stoke_lines": ["s"], "ref_lines": ["r"]}
    check("ref_waiver_decision: clean sub-eps swap is eligible",
          ref_waiver_decision(swap_ok["stoke_top5"], swap_ok["ref_top5"])["eligible"])
    check("ref_waiver_decision: gap >= eps is NOT eligible",
          not ref_waiver_decision(_top5((11, 22), 5e-4), _top5((22, 11), 2e-3))["eligible"])
    check("ref_waiver_decision: top-2 not swapped is NOT eligible",
          not ref_waiver_decision(_top5((11, 22), 5e-4), _top5((11, 22), 5e-4))["eligible"])
    check("ref_waiver_decision: ranks 3-5 differing is NOT eligible",
          not ref_waiver_decision(_top5((11, 22), 5e-4),
                                  _top5((22, 11), 5e-4, tail=(33, 55, 44)))["eligible"])

    def _report(overrides, prompts=None):
        res = {p: {"identical": True} for p in (prompts or RULING_SUBSET)}
        res.update(overrides)
        return {"n_predict": N_PREDICT, "results": res}

    def _diverged(pos=7):
        return {"identical": False, "first_diff_pos": pos,
                "ref_id_at_diff": 22, "stoke_id_at_diff": 11}

    def _probe(kind):
        return lambda pid, pos: (dict(swap_ok) if kind == "ok"
                                 else {"error": "probe pos never reached"} if kind == "err"
                                 else {"stoke_top5": _top5((11, 22), 5e-4),
                                       "ref_top5": _top5((22, 11), 2e-3),
                                       "stoke_lines": ["s"], "ref_lines": ["r"]})

    v = adjudicate_report(_report({}), _probe("ok"))
    check("adjudicate: all exact => PASS", v["verdict"] == "PASS" and v["n_exact"] == 24)

    v = adjudicate_report(_report({"p05": {"error": "timed out"}}), _probe("ok"))
    check("adjudicate: a tooling error => FAIL without probing (ruling §2.1)",
          v["verdict"] == "FAIL" and v["blocked_by"] == "tooling")

    v = adjudicate_report(_report({}, prompts=RULING_SUBSET[:-1]), _probe("ok"))
    check("adjudicate: a prompt missing from the report => INCOMPLETE",
          v["verdict"] == "INCOMPLETE" and v["blocked_by"] == "coverage")

    v = adjudicate_report(_report({"p02": _diverged()}), _probe("ok"))
    check("adjudicate: waivable flip draws r00 and blocks on the replacement",
          v["verdict"] == "INCOMPLETE" and v["blocked_by"] == "replacement"
          and v["pending"] == ["r00"] and v["waivers"][0]["waived"])
    check("adjudicate: the verbatim probe blocks reach the record",
          v["divergences"][0]["stoke_probe_lines"] == ["s"]
          and v["divergences"][0]["ref_probe_lines"] == ["r"])

    v = adjudicate_report(_report({"p02": _diverged(), "r00": {"identical": True}}),
                          _probe("ok"))
    check("adjudicate: waived + exact replacement => PASS", v["verdict"] == "PASS")

    v = adjudicate_report(_report({"p02": _diverged(),
                                   "r00": {"identical": False, "first_diff_pos": 3}}),
                          _probe("ok"))
    check("adjudicate: a diverging replacement FAILS outright (no chaining)",
          v["verdict"] == "FAIL"
          and any("no chaining" in r for r in v["fail_reasons"]))

    v = adjudicate_report(_report({"p02": _diverged(), "p46": _diverged()}), _probe("ok"))
    check("adjudicate: two waivers draw r00,r01 in ascending prompt order",
          [w["replacement"] for w in v["waivers"]] == ["r00", "r01"])

    v = adjudicate_report(_report({"p02": _diverged(), "p46": _diverged(),
                                   "p48": _diverged()}), _probe("ok"))
    check("adjudicate: a 3rd flip exhausts the budget => FAIL",
          v["verdict"] == "FAIL"
          and any("BUDGET EXHAUSTED" in r for r in v["fail_reasons"]))

    v = adjudicate_report(_report({"p02": _diverged()}), _probe("wide"))
    check("adjudicate: an ineligible flip => FAIL",
          v["verdict"] == "FAIL" and not v["waivers"][0]["waived"])

    v = adjudicate_report(_report({"p02": _diverged()}), _probe("err"))
    check("adjudicate: an unobtainable probe => FAIL, never a free pass",
          v["verdict"] == "FAIL"
          and any("probe unobtainable" in r for r in v["fail_reasons"]))

    check("stoke.exe present", STOKE_EXE.exists())
    check("model present", MODEL_PATH.exists())
    check("ref_forward present", REF.exists())
    check("corpus present (50 p-prompts)",
          len(list(CORPUS.glob("p*.txt"))) == 50)

    print("\n" + ("REF GATE SELFTEST: ALL GREEN" if ok else "REF GATE SELFTEST: FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", help="comma-separated prompt ids, e.g. p48,p20")
    ap.add_argument("--jobs", type=int, default=3,
                    help="concurrent ref processes (RAM-bound; ~3 on a 16GB box)")
    ap.add_argument("--n-predict", type=int, default=N_PREDICT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adjudicate", action="store_true",
                    help="turn a COMPLETED run's report into a verdict: probe "
                         "both sides at each first divergence and route it "
                         "through the committed ref_waiver_decision (ruling "
                         "§4.1). Serial, and it re-runs nothing that passed.")
    a = ap.parse_args(argv)

    if a.selftest:
        return run_selftest()
    if a.adjudicate:
        try:
            return run_adjudication(a.n_predict)
        except GateError as e:
            print(f"fatal: {e}", file=sys.stderr)
            return 1
    if not a.subset:
        print("fatal: --subset is required (the subset is a decided gate "
              "parameter, not a default this tool may invent)", file=sys.stderr)
        return 2
    ids = [s.strip() for s in a.subset.split(",") if s.strip()]
    for pid in ids:
        if not (CORPUS / f"{pid}.txt").exists():
            print(f"fatal: no corpus file for {pid}", file=sys.stderr)
            return 2
    if a.dry_run:
        stoke_hash = sha256_file(STOKE_EXE)
        counts = manifest_token_counts()      # real counts, not a bytes guess
        total = 0.0
        longest = 0.0
        print(f"{'prompt':8} {'tokens':>7} {'est_min':>8} {'ceiling_s':>10}")
        for pid in ids:
            n_tok = counts.get(pid)
            if n_tok is None:
                print(f"fatal: {pid} has no MANIFEST token count", file=sys.stderr)
                return 2
            cost = (n_tok + a.n_predict) * SEC_PER_POSITION
            total += cost
            longest = max(longest, cost)
            print(f"{pid:8} {n_tok:>7} {cost/60:>8.1f} "
                  f"{timeout_for(n_tok, a.n_predict):>10.0f}")
        wall = max(total / a.jobs, longest)   # cannot beat the longest prompt
        print(f"\nestimated serial cost: {total/3600:.1f} h; "
              f"at --jobs {a.jobs}: ~{wall/3600:.1f} h wall "
              f"(makespan floor = longest single prompt, {longest/60:.0f} min)")
        print(f"stoke.exe {stoke_hash[:16]}…")
        return 0
    try:
        return run_gate(ids, a.jobs, a.n_predict)
    except GateError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
