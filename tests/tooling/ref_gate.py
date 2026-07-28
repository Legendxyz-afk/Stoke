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
        if d.get("key", {}).get("stoke_hash") != stoke_hash:
            fatal(f"{prompt_id}: cached stoke ids were produced by "
                  f"stoke.exe {d['key'].get('stoke_hash','?')[:12]}… but the "
                  f"current binary is {stoke_hash[:12]}… -- refusing to "
                  f"compare against a stale generation")
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
    def n_tokens(pid: str) -> int:
        return (CORPUS / f"{pid}.txt").stat().st_size
    ordered = sorted(prompt_ids, key=n_tokens, reverse=True)

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


_MANIFEST_ROW = re.compile(r"^\|\s*(p\d\d)\s*\|[^|]*\|\s*(\d+)\s*\|")


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
    check(f"MANIFEST parsed 50 real token counts (got {len(counts)})",
          len(counts) == 50)
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
    a = ap.parse_args(argv)

    if a.selftest:
        return run_selftest()
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
