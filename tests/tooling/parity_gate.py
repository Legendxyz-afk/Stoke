#!/usr/bin/env python3
"""tests/tooling/parity_gate.py -- M2 W4 parity gate harness.

Implements the gate mechanics of
docs/superpowers/design-notes/2026-07-19-m2-op-spec-and-parity-protocol.md
section 8 (VERBATIM contract) against the oracle invocation SETTLED there
(default KV, no -ctk/-ctv -- A5 outcome E3) and the near-tie policy of
docs/superpowers/memos/2026-07-20-m2-oracle-near-tie-policy.md (the §8
waiver protocol is implemented EXACTLY as written; this harness does NOT
implement memo option 2's "bounded near-tie recognition" -- that is a
policy change pending Kabir, not yet adopted, so a v1-class near-tie is
GATE FAIL here, full stop).

SCOPE BOUNDARY (binding on this worker, not on whoever runs this later):
this module was authored WITHOUT ever starting `llama-server` or issuing
a live oracle HTTP request -- the manager runs the actual gate. The only
verification performed while writing it was `--selftest` (server-free,
against the committed `tests/tooling/validation/oracle_v3_multilingual.json`
fixture), `--dry-run` (prints the plan, touches no process/socket), and a
plain `ast.parse` syntax check. See `.superpowers/sdd/m2-w4-report.md`.

THE ALIGNER (the one subtle piece -- see `align_tokens_to_entries()` and
`harvest_token_bytes()` docstrings for the full derivation): llama-server's
`completion_probabilities` coalesces consecutive raw generation steps that
individually split a multi-byte UTF-8 character into ONE reported entry,
so entry-index drifts from token-position after any coalescing. The
subtlety proven against the v3 fixture before writing a line of the walk:
a coalesced entry's OWN top-level `bytes` field is the CUMULATIVE span of
every swallowed step (unusable as a per-token byte length), while the
SAME token id, when it appears as a `top_logprobs` CANDIDATE anywhere in
the response, always reports its true single-token byte encoding. The
aligner therefore harvests id -> bytes exclusively from `top_logprobs`
candidates, never from an entry's own top-level `bytes`.

Python idiom: stdlib only (no numpy, no `gguf` package needed -- this
tool never touches the GGUF file itself, only oracle JSON and stoke.exe's
stdout/stderr), explicit loud failure via `GateError`/`fatal()` (mirrors
tests/tooling/gguf_reader.py's `GgufError`/`fatal()` exactly), a
`--selftest` that prints PASS/FAIL per assertion and a single-line banner
the brief pins verbatim: `GATE SELFTEST: ALL GREEN`.

MANAGER CORRECTIONS incorporated 2026-07-27 (both measured live by the
manager against the real server; this worker did not re-verify by
starting anything itself, per instruction):
  1. Build identity does NOT come from `/props` (it has no build/commit
     field) -- it comes from `llama-server.exe --version` instead (see
     `capture_build_identity()`/`parse_version_output()`). Health-poll
     timeout raised from a guessed 180s to a measured-informed 300s
     floor (cold start measured at 69s), poll interval 1s, and a dead
     child now surfaces the tail of its captured log instead of being
     polled for the full timeout (see `OracleServer.wait_healthy()`).
  2. Added `--survey`: a diagnostic, never-early-stops, budget/reserve-
     free census mode for the near-tie divergence-class measurement
     (see `run_survey()`/`write_survey_report()`), plus longest-first
     (by corpus file byte size) stoke job-pool scheduling (see
     `_order_longest_first()`) to avoid the long-context tier (p40-p44,
     r07) stranding the makespan in the tail.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Constants -- probed THIS session (2026-07-27, brief header) and re-verified
# by this worker via direct filesystem stat before writing a line of the
# request/process-spawn code. The harness re-checks the load-bearing ones
# (model size, exe presence, corpus completeness) again at run time and
# FATALS if live reality disagrees -- never a silent fallback.
# ============================================================================

# --- Console encoding hardening (manager fix 2026-07-27, found in review) ---
# This module prints the section sign U+00A7 in several normal, non-error
# paths (e.g. the --dry-run "KV setting ... M2 note" line). Python encodes
# stdout using the console codepage, and U+00A7 is UNMAPPABLE in cp437 --
# the default codepage of a Windows console -- so those prints raise
# UnicodeEncodeError and abort the process. VERIFIED first-hand, not
# theorised: `PYTHONIOENCODING=cp437 python parity_gate.py --dry-run` exited
# 1 with 'charmap' codec can't encode character '\xa7', while cp1252 and
# cp850 (which do map it) exited 0. --selftest passes under ALL of them
# because it happens not to print that character -- i.e. verification does
# not catch this, but a multi-hour detached run driven from a .cmd would
# die on it. Forcing UTF-8 with errors='replace' makes the harness immune
# to the ambient codepage; 'replace' rather than 'strict' guarantees that a
# character can never abort a gate run, since garbled output is always
# preferable to a destroyed 4-hour measurement. Report/cache files are
# unaffected -- those already pin encoding='utf-8' at every write site.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass   # best-effort: a redirected/replaced stream may refuse


REPO_ROOT = Path(__file__).resolve().parents[2]   # tests/tooling/parity_gate.py -> repo root

ORACLE_EXE = REPO_ROOT / "bench" / "vendor" / "llama-b10037" / "llama-server.exe"
STOKE_EXE = REPO_ROOT / "build" / "win" / "stoke.exe"
MODEL_PATH = REPO_ROOT / "models" / "Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
MODEL_SIZE_EXPECTED = 4280405216   # bytes -- probed constant, brief header

CORPUS_DIR = REPO_ROOT / "tests" / "data" / "parity"
CACHE_DIR = REPO_ROOT / "tests" / "tooling" / ".gate_cache"          # dev cache; manager gitignores it
VALIDATION_FIXTURE = REPO_ROOT / "tests" / "tooling" / "validation" / "oracle_v3_multilingual.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "tests" / "tooling" / "parity_gate_dev_report.md"
DEFAULT_SURVEY_REPORT_PATH = REPO_ROOT / "tests" / "tooling" / "parity_survey_dev_report.md"

N_PROMPTS = 50
N_RESERVE = 10
EPSILON = 1e-3       # nats -- §8 waiver epsilon. NEVER loosen.
WAIVER_BUDGET = 2    # §8 -- NEVER exceed.

N_PREDICT_DEFAULT = 64
DEFAULT_PORT = 8090
DEFAULT_JOBS = 4

# Generous ceilings, not CLI-tunable (the brief's CLI contract does not
# include timeout flags; §1 sets the expectation "seconds per token" for
# scalar M2 code, and the 400-512 token long-context prompts (p40-p44,
# r07) need one-position-at-a-time prefill on top of the 64 generated --
# these are ceilings against a truly wedged process, not a tuning knob).
STOKE_TIMEOUT_S = 1800.0
# MANAGER-MEASURED 2026-07-27 (live server, settled invocation, 4.3GB Q8_0,
# -t 1): cold start reported healthy after 69s. 300s is a >4x margin over
# that measurement, per the manager's explicit "at least 300 s" floor --
# NOT the brief's original 180s guess, which the live measurement showed
# was too tight.
ORACLE_HEALTH_TIMEOUT_S = 300.0
ORACLE_REQUEST_TIMEOUT_S = 900.0


class GateError(Exception):
    """Any fatal, no-silent-fallback condition: gate-integrity violations,
    malformed oracle/stoke output, a probed constant contradicted by live
    reality. Callers catch this ONCE at the top (main()) and print
    'fatal: <msg>' to stderr, exit 1 -- mirrors gguf_reader.py's
    GgufError/fatal() idiom exactly."""


def fatal(msg: str):
    raise GateError(msg)


def _display_argv(argv: List[str]) -> str:
    """Human-readable argv for dry-run/report/error text only (actual
    subprocess calls always use the list form, never a shell string --
    this repo's own path contains a space ('New Drive'), so naive
    ' '.join() would print something copy-pasteable-wrong)."""
    return " ".join(f'"{a}"' if " " in a else a for a in argv)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Corpus
# ============================================================================

def default_prompt_ids() -> List[str]:
    return [f"p{idx:02d}" for idx in range(N_PROMPTS)]


def default_reserve_ids() -> List[str]:
    return [f"r{idx:02d}" for idx in range(N_RESERVE)]


def corpus_path(prompt_id: str) -> Path:
    return CORPUS_DIR / f"{prompt_id}.txt"


def load_prompt_text(prompt_path: Path) -> str:
    """Exact file bytes, decoded UTF-8. These corpus files are eol-pinned
    `-text` (MANIFEST.md: last byte deliberately not 0x0A on most of
    them) -- read_bytes()+decode NEVER touches \\r\\n, unlike a text-mode
    open(), so this cannot silently normalize a newline the gate depends
    on being exact."""
    if not prompt_path.exists():
        fatal(f"corpus file missing: {prompt_path}")
    raw = prompt_path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        fatal(f"{prompt_path}: not valid UTF-8 -- {e}")


def parse_only(only_arg: Optional[str]) -> List[str]:
    """Validates --only against the KNOWN id namespace (p00..p49,
    r00..r09) -- a bad id STRING is a usage error, fatal immediately,
    independent of whether the corresponding file happens to exist on
    disk (that existence check is separate, see run_gate/run_dry_run)."""
    if not only_arg:
        return default_prompt_ids()
    ids = [s.strip() for s in only_arg.split(",") if s.strip()]
    if not ids:
        fatal("--only was given but resolved to an empty list")
    valid = set(default_prompt_ids()) | set(default_reserve_ids())
    bad = [i for i in ids if i not in valid]
    if bad:
        fatal(f"--only contains unknown prompt id(s): {bad!r} (valid ids: "
              f"p00..p49, r00..r09)")
    return ids


def _order_longest_first(prompt_ids: List[str]) -> List[str]:
    """Sort by corpus FILE BYTE SIZE descending -- a robust, dependency-
    free proxy for token count (an exact count would need invoking a
    tokenizer, which this harness deliberately never does). MANAGER-
    MEASURED 2026-07-27: ~5.3s per forward pass, corpus mean 77.8 / max
    490 prompt tokens; with a bounded --jobs pool, submitting the
    longest prompts FIRST means the long-context tier (p40-p44, r07:
    447-490 tokens, ~553 forward passes, ~49 min each) starts immediately
    on multiple workers instead of landing in the tail after 45 short
    prompts drain, which would otherwise strand the whole run's
    makespan waiting on stragglers with idle workers. Byte-size ordering
    cannot misfire on the case that matters here: no short prompt in
    this corpus is anywhere close to the long-context files' byte count,
    even though byte-per-token ratio varies by script (CJK vs Latin)
    among the short prompts themselves."""
    return sorted(prompt_ids, key=lambda pid: corpus_path(pid).stat().st_size, reverse=True)


# ============================================================================
# The aligner -- tokens[] (ground truth, 64 ids) vs completion_probabilities
# (coalesced, fewer entries after any multi-byte UTF-8 split). See module
# docstring for the top-level summary; full derivation below.
# ============================================================================

def harvest_token_bytes(entries: List[dict]) -> Dict[int, bytes]:
    """Build a GLOBAL id -> bytes map from this response's own data, using
    ONLY `top_logprobs` candidates -- never an entry's own top-level
    'id'/'bytes' pair.

    Why not the top-level field: for a COALESCED entry (one that swallowed
    >1 raw generation step because the earlier step(s) alone were invalid
    UTF-8), the top-level 'bytes' is the CUMULATIVE span of every
    swallowed step, tagged with only the LAST step's id. Empirically
    against tests/tooling/validation/oracle_v3_multilingual.json: token
    id 235 appears as a top-level entry id with bytes=[32,240,159,140,141]
    (the full 5-byte ' <U+1F30D>' span covering 3 underlying tokens), but
    as a top_logprobs CANDIDATE (an alternative single-token choice at
    some step) it always reports bytes=[141] -- its true one-byte
    encoding. Using the top-level field as "this token's byte length"
    would overcount by every swallowed predecessor's bytes and corrupt
    the whole downstream cumulative-offset walk. top_logprobs candidates
    are always single-vocab-token alternatives, so their (id, bytes)
    pairs are internally consistent and reusable anywhere in the response.

    FATAL (GateError) if the SAME id reports two different byte encodings
    as a candidate in this response -- that would mean the tokenizer
    itself is not a fixed id->bytes mapping, a much deeper problem this
    harness refuses to paper over.
    """
    id_bytes: Dict[int, bytes] = {}
    for entry in entries:
        candidates = entry.get("top_logprobs")
        if candidates is None:
            fatal("harvest_token_bytes: an entry is missing 'top_logprobs'")
        for cand in candidates:
            cid = cand["id"]
            cbytes = bytes(cand["bytes"])
            prev = id_bytes.get(cid)
            if prev is not None and prev != cbytes:
                fatal(f"harvest_token_bytes: token id {cid} has conflicting "
                      f"byte encodings within one response's top_logprobs "
                      f"candidates: {prev!r} vs {cbytes!r}")
            id_bytes[cid] = cbytes
    return id_bytes


def align_tokens_to_entries(tokens: List[int], entries: List[dict]) -> List[int]:
    """Returns, for each of the len(tokens) generated-token positions, the
    index into `entries` whose byte-span covers that position (multiple
    consecutive positions map to the SAME entry index whenever that
    stretch was coalesced -- e.g. positions 5,6,7 -> entry 5 in the v3
    fixture). Positions never map to a PARTIAL entry -- coalescing only
    ever merges whole consecutive raw tokens, never splits one.

    Algorithm: resolve each token's own byte length via
    harvest_token_bytes(), take cumulative byte offsets on both the
    token side and the entry side (entries' own top-level 'bytes' field
    IS trustworthy for entry-span LENGTH -- that field's only problem is
    being misused as a per-token length; as "how many bytes did this
    reported entry emit in total" it is exactly right), then walk a
    single advancing entry-pointer so each token's [start,end) byte span
    lands inside some entry's [start,end) span.

    FATAL (no silent fallback) if: a token id's bytes can't be resolved
    from this response's own top_logprobs candidates at all (see the
    m2-w4-report.md flag on this -- there is no proven guarantee every
    ground-truth id always resurfaces as a candidate somewhere in its own
    response, only an empirical confirmation on the v3 fixture); the
    total byte counts disagree; the walk runs off the end of `entries`;
    or the resulting mapping is not monotonic.
    """
    if not tokens:
        fatal("align_tokens_to_entries: tokens is empty")
    if not entries:
        fatal("align_tokens_to_entries: entries is empty")

    id_bytes = harvest_token_bytes(entries)
    tok_lens: List[int] = []
    for pos, tid in enumerate(tokens):
        b = id_bytes.get(tid)
        if b is None:
            fatal(f"align_tokens_to_entries: token id {tid} at position {pos} "
                  f"never appears as a top_logprobs candidate anywhere in "
                  f"this response -- cannot independently determine its "
                  f"byte length (no stoke.exe decode path is wired up; see "
                  f"m2-w4-report.md aligner flag). Refusing to guess.")
        tok_lens.append(len(b))

    entry_lens = []
    for e in entries:
        eb = e.get("bytes")
        if eb is None:
            fatal("align_tokens_to_entries: an entry is missing 'bytes'")
        entry_lens.append(len(bytes(eb)))

    total_tok = sum(tok_lens)
    total_entry = sum(entry_lens)
    if total_tok != total_entry:
        fatal(f"align_tokens_to_entries: total token bytes ({total_tok}) != "
              f"total entry bytes ({total_entry}) -- byte reconciliation "
              f"failed, refusing to align")

    entry_start = []
    entry_end = []
    c = 0
    for l in entry_lens:
        entry_start.append(c)
        c += l
        entry_end.append(c)

    mapping: List[int] = []
    ei = 0
    tok_start = 0
    for pos, tl in enumerate(tok_lens):
        tok_end = tok_start + tl
        while not (entry_start[ei] <= tok_start and tok_end <= entry_end[ei]):
            ei += 1
            if ei >= len(entries):
                fatal(f"align_tokens_to_entries: walk ran past the last "
                      f"entry while placing token position {pos} "
                      f"(byte span [{tok_start},{tok_end}))")
        mapping.append(ei)
        tok_start = tok_end

    for i in range(len(mapping) - 1):
        if mapping[i] > mapping[i + 1]:
            fatal(f"align_tokens_to_entries: non-monotonic mapping at "
                  f"position {i} ({mapping[i]} -> {mapping[i + 1]})")
    return mapping


def top5_for_position(tokens: List[int], entries: List[dict], pos: int) -> List[Tuple[int, float]]:
    """The oracle's (id, logprob) top-5 for the entry covering token
    position `pos`, rank-0 (highest logprob) first, exactly as
    top_logprobs reported it. Per §8: "For a divergence at token position
    p, return that entry's top_logprobs" -- if p falls inside a coalesced
    run, every position in that run shares the SAME entry (and therefore
    the same top-5), which is the only data the oracle's response format
    makes available; there is no finer-grained per-swallowed-step
    distribution to recover."""
    mapping = align_tokens_to_entries(tokens, entries)
    if not (0 <= pos < len(mapping)):
        fatal(f"top5_for_position: position {pos} out of range [0,{len(mapping)})")
    entry = entries[mapping[pos]]
    return [(c["id"], c["logprob"]) for c in entry["top_logprobs"]]


# ============================================================================
# Waiver decision (§8 near-tie protocol) -- pure function + a small stateful
# ledger for the budget/reserve-consumption side.
# ============================================================================

def waiver_decision(oracle_top5: List[Tuple[int, float]],
                     stoke_top5: List[Tuple[int, float]]) -> dict:
    """Pure, no I/O -- the §8 per-divergence eligibility check. Each input
    is exactly 5 (id, value) pairs, rank-0 (highest value) first: oracle's
    values are logprobs, stoke's are raw logits. A top1-top2 GAP is
    IDENTICAL in both spaces for the same softmax distribution (logprob_i
    = logit_i - logsumexp(logits), so logprob_top1 - logprob_top2 =
    logit_top1 - logit_top2 exactly -- the logsumexp normalizer cancels in
    the subtraction), so comparing the two gaps directly, unit for unit,
    is exact, not an approximation.

    Conditions (§8, ALL required for eligible=True):
      1. oracle_gap < EPSILON AND stoke_gap < EPSILON (both nats, strict <)
      2. the two top-2 ids are the SAME PAIR, SWAPPED (oracle's rank0 ==
         stoke's rank1 AND oracle's rank1 == stoke's rank0)
    Budget/reserve consumption is NOT this function's job -- see
    WaiverLedger.try_waive(), which wraps this with the ≤2 budget check.
    NEVER loosens EPSILON or weakens the swap condition -- doing so is a
    gate-integrity violation per §8 / the near-tie memo.
    """
    if len(oracle_top5) != 5:
        fatal(f"waiver_decision: oracle_top5 must have exactly 5 entries, got {len(oracle_top5)}")
    if len(stoke_top5) != 5:
        fatal(f"waiver_decision: stoke_top5 must have exactly 5 entries, got {len(stoke_top5)}")

    oracle_gap = oracle_top5[0][1] - oracle_top5[1][1]
    stoke_gap = stoke_top5[0][1] - stoke_top5[1][1]
    if oracle_gap < 0:
        fatal(f"waiver_decision: oracle_top5 not sorted descending "
              f"(rank0={oracle_top5[0]} < rank1={oracle_top5[1]})")
    if stoke_gap < 0:
        fatal(f"waiver_decision: stoke_top5 not sorted descending "
              f"(rank0={stoke_top5[0]} < rank1={stoke_top5[1]})")

    oracle_pair = (oracle_top5[0][0], oracle_top5[1][0])
    stoke_pair = (stoke_top5[0][0], stoke_top5[1][0])
    same_pair_swapped = (oracle_pair[0] == stoke_pair[1] and oracle_pair[1] == stoke_pair[0])
    gaps_ok = (oracle_gap < EPSILON) and (stoke_gap < EPSILON)

    if gaps_ok and same_pair_swapped:
        eligible, reason, cls = True, "both gaps < epsilon and top-2 pair swapped -- genuine near-tie", "waivable"
    elif not gaps_ok:
        eligible, reason, cls = False, (f"gap >= epsilon ({EPSILON}): "
                                         f"oracle_gap={oracle_gap:.6g}, stoke_gap={stoke_gap:.6g}"), "non_waivable_gap"
    else:
        eligible, reason, cls = False, (f"both gaps < epsilon but top-2 pairs are not a clean "
                                         f"swap: oracle_pair={oracle_pair}, stoke_pair={stoke_pair}"), "non_waivable_pair"

    return {
        "eligible": eligible,
        "reason": reason,
        "class": cls,   # "waivable" | "non_waivable_gap" | "non_waivable_pair" -- §8 census/survey use
        "oracle_gap": oracle_gap,
        "stoke_gap": stoke_gap,
        "oracle_pair": oracle_pair,
        "stoke_pair": stoke_pair,
        "oracle_top5": oracle_top5,
        "stoke_top5": stoke_top5,
    }


class WaiverLedger:
    """Stateful budget/reserve tracking around the pure waiver_decision().
    Reserve prompts are consumed strictly in order (r00, r01, ...);
    budget is a hard ≤N ceiling shared across the WHOLE gate run (not
    per-prompt) -- the 3rd otherwise-eligible waiver is downgraded to
    ineligible with a budget-exhausted reason, which is exactly a GATE
    FAIL at the call site."""

    def __init__(self, budget: int, reserve_ids: List[str]):
        self.budget = budget
        self._reserve_iter = iter(reserve_ids)
        self.records: List[dict] = []

    def _next_reserve_or_fatal(self) -> str:
        try:
            return next(self._reserve_iter)
        except StopIteration:
            fatal("waiver ledger: reserve pool exhausted -- this should be "
                  "impossible given budget<=2 and 10 reserve prompts, so "
                  "something upstream is wrong (gate-integrity violation, "
                  "not a normal gate outcome)")

    def try_waive(self, prompt_id: str, position: int,
                  oracle_top5: List[Tuple[int, float]],
                  stoke_top5: List[Tuple[int, float]]) -> Tuple[bool, dict]:
        decision = waiver_decision(oracle_top5, stoke_top5)
        if not decision["eligible"]:
            return False, decision
        if len(self.records) >= self.budget:
            decision = dict(decision)
            decision["eligible"] = False
            decision["reason"] = (f"otherwise-waivable divergence REJECTED: waiver "
                                   f"budget ({self.budget}) already used "
                                   f"({len(self.records)} waiver(s) recorded)")
            return False, decision
        replacement = self._next_reserve_or_fatal()
        record = {
            "prompt_id": prompt_id,
            "position": position,
            "replacement": replacement,
            "oracle_gap": decision["oracle_gap"],
            "stoke_gap": decision["stoke_gap"],
            "oracle_pair": decision["oracle_pair"],
            "stoke_pair": decision["stoke_pair"],
        }
        self.records.append(record)
        return True, {**decision, "replacement": replacement}


# ============================================================================
# Id-sequence comparison
# ============================================================================

@dataclass
class IdComparison:
    identical: bool
    first_diff_pos: Optional[int]
    oracle_len: int
    stoke_len: int


def compare_ids(oracle_ids: List[int], stoke_ids: List[int]) -> IdComparison:
    """Full-sequence comparison (§8 harness contract: "compare the FULL id
    sequences; lengths must agree...; PASS iff identical"). A length
    mismatch where one sequence is a strict prefix of the other is
    reported as a divergence AT the shorter sequence's length (the point
    where it ran out) -- see run_gate/run_stoke_probe for how that
    interacts with the waiver protocol (probing a position beyond what a
    side actually generated is a hard failure there, by construction)."""
    n = min(len(oracle_ids), len(stoke_ids))
    for i in range(n):
        if oracle_ids[i] != stoke_ids[i]:
            return IdComparison(False, i, len(oracle_ids), len(stoke_ids))
    if len(oracle_ids) != len(stoke_ids):
        return IdComparison(False, n, len(oracle_ids), len(stoke_ids))
    return IdComparison(True, None, len(oracle_ids), len(stoke_ids))


# ============================================================================
# Stoke invocation (§ probed constants; src/main.c cmd_run is the authority
# on exact output format -- verified by reading it directly, not guessed)
# ============================================================================

def stoke_generate_argv(prompt_path: Path, n_predict: int) -> List[str]:
    return [str(STOKE_EXE), "run", str(MODEL_PATH), "--raw",
            "-f", str(prompt_path), "-n", str(n_predict)]


def stoke_probe_argv(prompt_path: Path, n_predict: int, pos) -> List[str]:
    return [str(STOKE_EXE), "run", str(MODEL_PATH), "--raw",
            "-f", str(prompt_path), "-n", str(n_predict), "--probe-pos", str(pos)]


# src/main.c cmd_run(): `fprintf(stderr, "probe pos=%d: id=%u logit=%.6f\n", ...)`
# -- %.6f is always fixed-point (never scientific), so the logit pattern is
# strict: optional '-', digits, '.', exactly 6 digits.
_PROBE_LINE_RE = re.compile(r"^probe pos=(-?\d+): id=(\d+) logit=(-?\d+\.\d{6})$")


def run_stoke_generate(prompt_path: Path, n_predict: int,
                        timeout_s: float = STOKE_TIMEOUT_S) -> List[int]:
    """`stoke run <model> --raw -f <prompt> -n <n_predict>` -> the
    generated ids, one per source line, BEFORE the literal '---' line
    (src/main.c: printf("%u\\n") per id, then printf("---\\n")). A
    non-zero exit means stoke's own fatal() already wrote 'fatal: ...' to
    stderr -- never attempt to parse stdout in that case."""
    argv = stoke_generate_argv(prompt_path, n_predict)
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        fatal(f"stoke run timed out after {timeout_s}s: {_display_argv(argv)}")
    if proc.returncode != 0:
        fatal(f"stoke run failed (exit {proc.returncode}): {_display_argv(argv)}\n"
              f"stderr: {proc.stderr.decode('utf-8', errors='replace')}")
    ids: List[int] = []
    for line in proc.stdout.decode("utf-8").splitlines():
        if line == "---":
            break
        try:
            ids.append(int(line))
        except ValueError:
            fatal(f"stoke run: unparseable id line {line!r} before the '---' "
                  f"separator: {_display_argv(argv)}")
    return ids


def run_stoke_probe(prompt_path: Path, n_predict: int, pos: int,
                     timeout_s: float = STOKE_TIMEOUT_S) -> List[Tuple[int, float]]:
    """`--probe-pos <pos>` re-run -> the 5 (id, logit) stderr lines for
    generated position `pos`. If stoke's OWN generation stopped before
    reaching `pos` (e.g. an early EOS), main.c fatals with a nonzero exit
    instead of emitting probe lines -- that is treated as a hard failure
    here (a divergence at a position stoke never reached has no stoke-
    side top-5 evidence, so it cannot be waived by construction; see
    m2-w4-report.md's flag on length-mismatch divergences)."""
    argv = stoke_probe_argv(prompt_path, n_predict, pos)
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        fatal(f"stoke --probe-pos {pos} timed out after {timeout_s}s: {_display_argv(argv)}")
    if proc.returncode != 0:
        fatal(f"stoke --probe-pos {pos} failed (exit {proc.returncode}) -- likely "
              f"means stoke's own generation stopped before position {pos} (no "
              f"stoke-side top-5 evidence exists there -- non-waivable by "
              f"construction): {_display_argv(argv)}\n"
              f"stderr: {proc.stderr.decode('utf-8', errors='replace')}")
    stderr_text = proc.stderr.decode("utf-8", errors="replace")
    results: List[Tuple[int, float]] = []
    for line in stderr_text.splitlines():
        m = _PROBE_LINE_RE.match(line.strip())
        if m and int(m.group(1)) == pos:
            results.append((int(m.group(2)), float(m.group(3))))
    if len(results) != 5:
        fatal(f"expected exactly 5 'probe pos={pos}: ...' lines on stderr, got "
              f"{len(results)}: {_display_argv(argv)}\nstderr: {stderr_text!r}")
    return results


# ============================================================================
# Build identity -- MANAGER-MEASURED 2026-07-27 (live server): /props carries
# model_path/n_ctx/total_slots/chat_template etc but NO build/commit field,
# contradicting the brief's original "GET /props and capture its build/commit
# field" instruction. The reliable, INSTANT (no model load), no-server-needed
# source is `llama-server.exe --version`, stderr:
#   version: 10037 (56d6e9dde)
#   built with Clang 20.1.8 for Windows x86_64
# This does not start a listening server and is not an oracle HTTP request,
# so it is not excluded by the "never start llama-server / never run a live
# gate" constraint -- but per the manager's explicit instruction this worker
# did NOT invoke it to re-verify; the parser below is proven instead by a
# --selftest unit test against the manager's own captured example string.
# ============================================================================

_VERSION_LINE_RE = re.compile(r"version:\s*(\S+)\s*\((\S+)\)")


def parse_version_output(stderr_text: str, stdout_text: str) -> dict:
    """Pure parser for `llama-server.exe --version` output. FATAL (no
    silent fallback) if no 'version: <build> (<commit>)' line is found in
    either stream -- this header is a gate-report checklist item, never
    silently left blank."""
    combined = stderr_text + "\n" + stdout_text
    m = _VERSION_LINE_RE.search(combined)
    if not m:
        fatal(f"'--version' output did not contain a 'version: N (hash)' line "
              f"-- refusing to record an empty build-identity field. "
              f"stdout={stdout_text!r} stderr={stderr_text!r}")
    return {
        "raw_stderr": stderr_text.strip(),
        "raw_stdout": stdout_text.strip(),
        "build_number": m.group(1),
        "commit": m.group(2),
    }


def capture_build_identity(exe: Path = ORACLE_EXE, timeout_s: float = 30.0) -> dict:
    """`<exe> --version` -- instant, no model load, no port, no listening
    server (manager-confirmed 2026-07-27 by direct measurement)."""
    argv = [str(exe), "--version"]
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout_s,
                               cwd=str(exe.parent))
    except subprocess.TimeoutExpired:
        fatal(f"{_display_argv(argv)} timed out after {timeout_s}s")
    return parse_version_output(proc.stderr.decode("utf-8", errors="replace"),
                                 proc.stdout.decode("utf-8", errors="replace"))


# ============================================================================
# Oracle server lifecycle (§8 SETTLED invocation -- default KV, no -ctk/-ctv,
# plus the -c context cap approved by Kabir 2026-07-27; see ORACLE_CTX below)
# ============================================================================

# Context cap on the ORACLE only. APPROVED BY KABIR 2026-07-27 (option (a) of
# docs/superpowers/memos/2026-07-27-m2-oracle-context-cap.md) on the strength
# of a byte-identical proof, NOT on an assumption that a context bound "should
# not" matter:
#   - Without it, llama-server sizes KV from the model's advertised default
#     context and holds 11,038 MB PRIVATE, making the survey ~20 GB at
#     --jobs 4 and ~16.5 GB even at --jobs 1 on a 16 GB box -- unrunnable,
#     and NOT fixable by lowering --jobs, since the oracle is the consumer.
#   - With -c 2048: 501 MB private (22x less) and /health in 8s instead of 69s.
#   - PROOF the gate's ground truth is unchanged: the three committed captures
#     (oracle_v1_english / oracle_v2_code / oracle_v3_multilingual) were
#     re-issued under the pinned request body with -c 2048 and returned 64/64
#     BYTE-IDENTICAL greedy ids each.
# 2048 is ~3.7x the gate's true maximum need of 554 positions (the 490-token
# longest corpus prompt + 64 generated), so no gate prompt can approach it.
# This value is part of the recorded gate's provenance -- it is written into
# every report's identity block. Do not change it without redoing the proof.
ORACLE_CTX = 2048


def oracle_argv(port: int) -> List[str]:
    return [str(ORACLE_EXE), "-m", str(MODEL_PATH), "-t", "1",
            "--parallel", "1", "--port", str(port), "-c", str(ORACLE_CTX)]


class OracleServer:
    """Owns exactly one llama-server subprocess. Every exit path -- normal
    return, GateError, an uncaught exception, SIGINT/SIGTERM, or Python
    process exit via atexit -- must reach stop(); stop() is idempotent so
    it is safe to be reached from more than one of those simultaneously."""

    def __init__(self, port: int):
        self.port = port
        self.proc: Optional[subprocess.Popen] = None
        self.log_path: Optional[Path] = None
        self._stopped = False

    def argv(self) -> List[str]:
        return oracle_argv(self.port)

    def _port_free(self) -> bool:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", self.port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    def start(self):
        global _ACTIVE_SERVER
        if not ORACLE_EXE.exists():
            fatal(f"oracle binary not found at {ORACLE_EXE} (probed constant "
                  f"contradicted by live reality)")
        if not MODEL_PATH.exists():
            fatal(f"model file not found at {MODEL_PATH} (probed constant "
                  f"contradicted by live reality)")
        actual_size = MODEL_PATH.stat().st_size
        if actual_size != MODEL_SIZE_EXPECTED:
            fatal(f"model file {MODEL_PATH} is {actual_size} bytes, expected "
                  f"{MODEL_SIZE_EXPECTED} (probed constant contradicted -- "
                  f"refusing to gate against a different/corrupt file)")
        if not self._port_free():
            fatal(f"port {self.port} is already in use -- free it and rerun "
                  f"(this harness never silently picks a different port)")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = CACHE_DIR / f"oracle_server_port{self.port}.log"
        log_fh = open(self.log_path, "wb")
        try:
            self.proc = subprocess.Popen(
                self.argv(), cwd=str(ORACLE_EXE.parent),
                stdout=log_fh, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        finally:
            log_fh.close()   # Popen holds its own dup of the fd
        _ACTIVE_SERVER = self

    def _tail_log(self, n_lines: int = 40) -> str:
        if self.log_path is None or not self.log_path.exists():
            return "(no log captured)"
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"(failed to read log: {e})"
        lines = text.splitlines()
        return "\n".join(lines[-n_lines:])

    def wait_healthy(self, timeout_s: float = ORACLE_HEALTH_TIMEOUT_S, interval_s: float = 1.0):
        """MANAGER-MEASURED 2026-07-27: cold start (4.3GB Q8_0, -t 1) took
        69s to report healthy -- timeout_s defaults to a >4x margin over
        that. Checks the child process's liveness on every iteration so a
        crashed server fails immediately, with the tail of its captured
        log, rather than being polled for the full timeout."""
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + timeout_s
        last_err = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                fatal(f"llama-server exited early (code {self.proc.returncode}) "
                      f"while waiting for /health -- last error: {last_err}\n"
                      f"--- tail of {self.log_path} ---\n{self._tail_log()}")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        body = json.loads(resp.read().decode("utf-8"))
                        if body.get("status") == "ok":
                            return
            except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as e:
                last_err = e
            time.sleep(interval_s)
        fatal(f"llama-server did not become healthy within {timeout_s}s at "
              f"{url} -- last error: {last_err}\n"
              f"--- tail of {self.log_path} ---\n{self._tail_log()}")

    def capture_identity(self) -> dict:
        """Build identity comes from `--version` (MANAGER-MEASURED
        2026-07-27: /props carries model_path/n_ctx/total_slots/
        chat_template etc but NO build/commit field -- the brief's
        original "/props has a build field" instruction was wrong,
        corrected live). /props is still queried, but only for selected
        LIVE-SERVER fields (model_path, n_ctx, total_slots) that cross-
        check which file the running server actually opened. Also folds
        in the oracle exe's own sha256 (argv always uses the FULL
        absolute path, so classic $PATH-shadowing is structurally
        impossible here; the hash additionally catches a binary swapped
        in-place at that same path)."""
        build = capture_build_identity(ORACLE_EXE)

        url = f"http://127.0.0.1:{self.port}/props"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                props = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as e:
            fatal(f"failed to GET {url} for live-server identity capture: {e}")
        live_fields = {k: props.get(k) for k in ("model_path", "n_ctx", "total_slots")}

        return {
            "argv": self.argv(),
            "oracle_exe_path": str(ORACLE_EXE.resolve()),
            "oracle_exe_sha256": _sha256_file(ORACLE_EXE),
            "kv_setting": "DEFAULT (no -ctk/-ctv) -- settled A5 outcome E3, M2 note §8 resolved block",
            "context_cap": (
                f"-c {ORACLE_CTX} (oracle only; gate's true max need is 554 positions "
                f"= 490-token longest prompt + 64 generated). Approved by Kabir "
                f"2026-07-27, memo docs/superpowers/memos/2026-07-27-m2-oracle-context-cap.md, "
                f"on a byte-identical proof: the committed oracle_v1_english / "
                f"oracle_v2_code / oracle_v3_multilingual captures re-issued under the "
                f"pinned body with this cap returned 64/64 identical greedy ids each. "
                f"Without it llama-server holds 11,038 MB private (default-context KV) "
                f"vs 501 MB with it."
            ),
            "build": build,
            "props_live_fields": live_fields,
            "props_full": props,
        }

    def request_completion(self, prompt_text: str, n_predict: int,
                            timeout_s: float = ORACLE_REQUEST_TIMEOUT_S) -> dict:
        """W1-PROBED exact request body (§8 PROBED block) -- do NOT add,
        remove, or rename a field."""
        body = {
            "prompt": prompt_text,
            "n_predict": n_predict,
            "temperature": 0,
            "n_probs": 5,
            "seed": 0,
            "return_tokens": True,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as e:
            fatal(f"/completion request failed: {e}")

    def stop(self, reason: str = "normal"):
        global _ACTIVE_SERVER
        if self._stopped:
            return
        self._stopped = True
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                except Exception:
                    pass
        if _ACTIVE_SERVER is self:
            _ACTIVE_SERVER = None

    def verify_dead_or_exit(self):
        proc = self.proc
        if proc is not None and proc.poll() is None:
            print(f"fatal: oracle server (pid {proc.pid}, port {self.port}) "
                  f"survived cleanup -- kill it manually: "
                  f"taskkill /PID {proc.pid} /F", file=sys.stderr)
            sys.exit(1)


# Process-wide cleanup wiring -- the brief requires try/finally AND atexit
# AND a signal handler, all three, so a stray killed-mid-run harness never
# strands the 4.3GB-mmap server process.
_ACTIVE_SERVER: Optional[OracleServer] = None


def _atexit_cleanup():
    if _ACTIVE_SERVER is not None:
        _ACTIVE_SERVER.stop(reason="atexit")


atexit.register(_atexit_cleanup)


def _signal_cleanup(signum, frame):
    if _ACTIVE_SERVER is not None:
        _ACTIVE_SERVER.stop(reason=f"signal {signum}")
    sys.exit(130 if signum == getattr(signal, "SIGINT", None) else 143)


# SIGTERM's actual delivery on native Windows differs from POSIX (Python's
# os.kill(pid, SIGTERM) there calls TerminateProcess directly, bypassing
# handlers), but registering it is harmless and keeps this portable to the
# project's WSL leg; SIGBREAK is the Windows-specific Ctrl-Break signal.
for _sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
    _sig = getattr(signal, _sig_name, None)
    if _sig is not None:
        try:
            signal.signal(_sig, _signal_cleanup)
        except (ValueError, OSError):
            pass   # e.g. not the main thread -- best-effort


# ============================================================================
# Dev cache (never used by the RECORDED run -- that run always passes
# --clean, per the brief: "THE recorded gate run is --clean")
# ============================================================================

def invocation_signature(args) -> str:
    """The part of "how would this prompt be processed" that affects the
    RESULT -- deliberately EXCLUDES --port (a pure networking detail that
    must not bust the cache) and the actual prompt path (parameterized
    out as a placeholder, since the prompt's own file hash is a separate,
    already-named cache-key component)."""
    # DERIVED from the real oracle_argv(), never a hand-written literal:
    # the whole purpose of this signature is to bust the cache when the
    # oracle invocation changes, and a literal cannot do that -- it silently
    # keeps saying "same invocation" while the actual flags move underneath
    # it. (Caught 2026-07-27, when -c 2048 was added and the literal did not
    # move.) --port is dropped as a pure networking detail that must not bust
    # the cache; the model path is collapsed to a placeholder since the model
    # is separately pinned by MODEL_SIZE_EXPECTED at every server start.
    argv = oracle_argv(port=0)
    flags = []
    skip_next = False
    for i, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if tok == "--port":
            skip_next = True
            continue
        if i == 0:
            continue                      # exe path -- covered by its sha256
        flags.append("<model>" if tok == str(MODEL_PATH) else tok)
    parts = [
        f"oracle_exe_sha256={_sha256_file(ORACLE_EXE)}",
        f"oracle_flags={' '.join(flags)}",
        f"stoke_argv_template={_display_argv(stoke_generate_argv(Path('<prompt>'), args.n_predict))}",
    ]
    return "|".join(parts)


def _cache_key_parts(prompt_path: Path, args) -> dict:
    """The brief's 4 named components: (prompt-file hash, stoke.exe hash,
    n_predict, invocation string) -- 'invocation string' here folds in
    the oracle binary's own identity and its fixed flags plus the stoke
    argv template, so a swapped oracle binary at the same vendored path
    also busts the cache (see invocation_signature() docstring)."""
    return {
        "prompt_hash": _sha256_file(prompt_path),
        "stoke_hash": _sha256_file(STOKE_EXE),
        "n_predict": args.n_predict,
        "invocation": invocation_signature(args),
    }


def _cache_key_hash(parts: dict) -> str:
    raw = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cache_data(prompt_path: Path, args) -> dict:
    if args.clean:
        return {}
    parts = _cache_key_parts(prompt_path, args)
    p = CACHE_DIR / f"{_cache_key_hash(parts)}.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("data", {})
    except (OSError, json.JSONDecodeError):
        # A corrupt/partial dev-cache entry is treated as a miss and
        # refreshed. Safe: the RECORDED gate run never reads this cache
        # at all (always --clean), so no gate DECISION can ever depend
        # on a stale or corrupt entry here -- only dev-loop convenience.
        return {}


def _store_cache_data(prompt_path: Path, args, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts = _cache_key_parts(prompt_path, args)
    p = CACHE_DIR / f"{_cache_key_hash(parts)}.json"
    payload = {"prompt_path": str(prompt_path), "key": parts, "data": data}
    tmp = p.parent / (p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, p)   # atomic on both Windows and POSIX


def fetch_oracle_result(server: OracleServer, prompt_id: str, args) -> dict:
    prompt_path = corpus_path(prompt_id)
    data = _load_cache_data(prompt_path, args)
    if "oracle" not in data:
        prompt_text = load_prompt_text(prompt_path)
        resp = server.request_completion(prompt_text, args.n_predict)
        tokens = resp.get("tokens")
        if not tokens:
            fatal(f"{prompt_id}: oracle response missing/empty top-level "
                  f"'tokens' field -- was return_tokens actually honored? "
                  f"(§8 PROBED block)")
        completion_probabilities = resp.get("completion_probabilities")
        if completion_probabilities is None:
            fatal(f"{prompt_id}: oracle response missing "
                  f"'completion_probabilities' (needed for the waiver aligner)")
        data["oracle"] = {"tokens": tokens, "completion_probabilities": completion_probabilities}
        _store_cache_data(prompt_path, args, data)
    return data["oracle"]


def fetch_stoke_result(prompt_id: str, args) -> List[int]:
    prompt_path = corpus_path(prompt_id)
    data = _load_cache_data(prompt_path, args)
    if "stoke_ids" not in data:
        ids = run_stoke_generate(prompt_path, args.n_predict)
        data["stoke_ids"] = ids
        _store_cache_data(prompt_path, args, data)
    return data["stoke_ids"]


# ============================================================================
# Report
# ============================================================================

def write_report(path: Path, identity: dict, args, slot_status: Dict[str, dict],
                  slot_history: Dict[str, List[dict]], gate_fail: Optional[dict],
                  waivers: List[dict]):
    lines = []
    lines.append(f"# M2 parity gate report -- {'GATE FAIL' if gate_fail else 'GATE PASS'}")
    lines.append("")
    lines.append("## Run configuration")
    lines.append(f"- n_predict: {args.n_predict}")
    lines.append(f"- jobs (stoke parallelism): {args.jobs}")
    lines.append(f"- clean: {args.clean}")
    lines.append(f"- waiver epsilon: {EPSILON} nats, budget: {WAIVER_BUDGET}")
    lines.append("")
    lines.append("## Oracle identity")
    lines.append(f"- argv: `{_display_argv(identity['argv'])}`")
    lines.append(f"- exe path: `{identity['oracle_exe_path']}`")
    lines.append(f"- exe sha256: `{identity['oracle_exe_sha256']}`")
    lines.append(f"- KV setting: {identity['kv_setting']}")
    lines.append(f"- context cap: {identity['context_cap']}")
    lines.append(f"- build (from `--version`): {identity['build']['raw_stderr']!r}")
    lines.append(f"  - parsed: build_number={identity['build']['build_number']} "
                  f"commit={identity['build']['commit']}")
    lines.append(f"- live /props fields (model_path/n_ctx/total_slots): "
                  f"`{json.dumps(identity['props_live_fields'])}`")
    lines.append("")
    lines.append("## stoke.exe identity")
    lines.append(f"- path: `{STOKE_EXE}`")
    lines.append(f"- sha256: `{_sha256_file(STOKE_EXE)}`")
    lines.append("")
    lines.append("## Corpus")
    lines.append(f"- directory: `{CORPUS_DIR}`")
    lines.append(f"- {N_PROMPTS} gate prompts (p00..p{N_PROMPTS - 1:02d}) + "
                  f"{N_RESERVE} reserve (r00..r{N_RESERVE - 1:02d})")
    lines.append("")
    lines.append("## Waivers")
    if not waivers:
        lines.append("(none)")
    else:
        for i, w in enumerate(waivers):
            lines.append(f"{i + 1}. prompt={w['prompt_id']} pos={w['position']} "
                          f"oracle_gap={w['oracle_gap']:.6g} stoke_gap={w['stoke_gap']:.6g} "
                          f"oracle_pair={w['oracle_pair']} stoke_pair={w['stoke_pair']} "
                          f"replaced_by={w['replacement']}")
    lines.append("")
    lines.append("## Per-slot waive trail")
    any_trail = False
    for slot, trail in slot_history.items():
        if trail:
            any_trail = True
            for step in trail:
                lines.append(f"- slot {slot}: prompt {step['prompt_id']} diverged at pos "
                              f"{step['position']}, waived, replaced_by={step['replacement']}")
    if not any_trail:
        lines.append("(none)")
    lines.append("")
    lines.append("## Per-prompt results")
    for slot in sorted(slot_status):
        r = slot_status[slot]
        lines.append(f"- {slot}: {r['status']} (final prompt id {r['prompt_id']}, "
                      f"{r['oracle_len']} generated ids)")
    if gate_fail:
        lines.append("")
        lines.append("## GATE FAIL evidence")
        lines.append("```json")
        lines.append(json.dumps(gate_fail, indent=2, default=str))
        lines.append("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_survey_report(path: Path, identity: dict, args, per_prompt: List[dict], census: dict):
    """--survey's report. Structurally SEPARATE from write_report() above
    -- shares no code, no header wording, no PASS/FAIL vocabulary -- so
    there is no path by which a survey run could emit anything a human
    could mistake for a gate stamp (manager instruction 2026-07-27: "if
    you find any way --survey could produce output that gets mistaken
    for a gate stamp, close that hole"). The loud marker appears in the
    title itself AND as a bolded restatement on the next line."""
    lines = []
    lines.append("# M2 parity SURVEY report -- DIAGNOSTIC ONLY, NOT A GATE RESULT")
    lines.append("")
    lines.append("**MODE: SURVEY -- DIAGNOSTIC ONLY, NOT A GATE RESULT.** This is a "
                  "measurement for the divergence-class census "
                  "(docs/superpowers/memos/2026-07-20-m2-oracle-near-tie-policy.md). "
                  "It NEVER enforces the §8 waiver budget, NEVER consumes a reserve "
                  "prompt, and NEVER early-stops on a non-waivable divergence -- every "
                  "prompt is evaluated independently. It carries NO PASS/FAIL verdict.")
    lines.append("")
    lines.append("## Run configuration")
    lines.append(f"- n_predict: {args.n_predict}")
    lines.append(f"- jobs (stoke parallelism): {args.jobs}")
    lines.append(f"- clean: {args.clean}")
    lines.append("- prompt scheduling: longest-first by corpus file byte size "
                  "(see _order_longest_first() -- makespan note)")
    lines.append("")
    lines.append("## Oracle identity")
    lines.append(f"- argv: `{_display_argv(identity['argv'])}`")
    lines.append(f"- build (from `--version`): {identity['build']['raw_stderr']!r}")
    lines.append(f"- KV setting: {identity['kv_setting']}")
    lines.append(f"- context cap: {identity['context_cap']}")
    lines.append("")
    lines.append("## Census")
    n = len(per_prompt)
    n_exact = len(census["exact"])
    n_diverged = len(census["diverged"])
    n_error = len(census.get("error", []))
    lines.append(f"- total prompts surveyed: {n}")
    lines.append(f"- token-exact: {n_exact}")
    lines.append(f"- diverged: {n_diverged}")
    lines.append(f"- tooling errors (not classified as a divergence): {n_error}")
    lines.append("")
    lines.append("### Divergence breakdown by class")
    class_counts: Dict[str, int] = {}
    for row in per_prompt:
        if row["status"] == "DIVERGED":
            c = row.get("verdict_class", "unknown")
            class_counts[c] = class_counts.get(c, 0) + 1
    if not class_counts:
        lines.append("(no divergences)")
    else:
        for c, n_c in sorted(class_counts.items()):
            lines.append(f"- {c}: {n_c}")
    lines.append("")
    lines.append("## Per-prompt detail")
    for row in per_prompt:
        if row["status"] == "EXACT":
            lines.append(f"- {row['prompt_id']}: EXACT ({row['oracle_len']} ids)")
        elif row["status"] == "ERROR":
            lines.append(f"- {row['prompt_id']}: ERROR oracle_error={row.get('oracle_error')!r} "
                          f"stoke_error={row.get('stoke_error')!r}")
        else:
            lines.append(f"- {row['prompt_id']}: DIVERGED at pos {row['position']} "
                          f"class={row.get('verdict_class')} reason={row.get('reason')!r}")
            lines.append(f"  oracle_id={row.get('oracle_id_at_pos')} "
                          f"stoke_id={row.get('stoke_id_at_pos')}")
            if "oracle_top5" in row:
                lines.append(f"  oracle_top5={row['oracle_top5']}")
                lines.append(f"  stoke_top5={row['stoke_top5']}")
    lines.append("")
    lines.append("**Reminder: MODE=SURVEY. This file is diagnostic input to a human "
                  "policy decision, not a gate result. It never implies PASS or FAIL.**")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================================
# The live gate (NEVER invoked by the W4 worker itself -- manager-only, per
# the brief's hard constraint #2)
# ============================================================================

def run_gate(args) -> int:
    if args.jobs < 1:
        fatal(f"--jobs must be >= 1, got {args.jobs}")
    if args.n_predict < 1:
        fatal(f"--n-predict must be >= 1, got {args.n_predict}")
    if not (1 <= args.port <= 65535):
        fatal(f"--port {args.port} out of range")

    selected = parse_only(args.only)
    reserve_ids = default_reserve_ids()

    all_needed = sorted(set(selected) | set(reserve_ids))
    missing = [pid for pid in all_needed if not corpus_path(pid).exists()]
    if missing:
        fatal(f"corpus file(s) missing for id(s) {missing!r} under {CORPUS_DIR} "
              f"(probed constant contradicted: expected p00..p49 + r00..r09 "
              f"all present)")

    report_path = Path(args.report) if args.report else DEFAULT_REPORT_PATH

    server = OracleServer(args.port)
    server.start()
    try:
        server.wait_healthy()
        identity = server.capture_identity()

        ledger = WaiverLedger(WAIVER_BUDGET, reserve_ids)
        slot_active: Dict[str, str] = {slot: slot for slot in selected}
        slot_status: Dict[str, dict] = {}
        slot_history: Dict[str, List[dict]] = {slot: [] for slot in selected}

        pending_slots = list(selected)
        while pending_slots:
            # Longest-first (by corpus file byte size): keeps the bounded
            # stoke job pool from stranding its makespan on long-context
            # stragglers landing in the tail (manager-measured 2026-07-27;
            # see _order_longest_first() docstring). Oracle calls are
            # strictly sequential regardless of order, so this only
            # matters for the stoke pool below, but the SAME ordered list
            # is reused for both loops for simplicity.
            need_ids = _order_longest_first(sorted({slot_active[s] for s in pending_slots}))

            # Oracle calls: sequential -- the server runs --parallel 1, so
            # concurrent /completion calls would queue/interfere anyway.
            oracle_cache: Dict[str, dict] = {}
            for pid in need_ids:
                oracle_cache[pid] = fetch_oracle_result(server, pid, args)

            # Stoke calls: bounded parallel pool (independent processes,
            # shared mmap page cache across them per §11 rationale).
            stoke_cache: Dict[str, List[int]] = {}
            with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                futs = {ex.submit(fetch_stoke_result, pid, args): pid for pid in need_ids}
                for fut in as_completed(futs):
                    pid = futs[fut]
                    stoke_cache[pid] = fut.result()   # re-raises GateError, if any

            next_pending = []
            for slot in pending_slots:
                pid = slot_active[slot]
                oracle_data = oracle_cache[pid]
                stoke_ids = stoke_cache[pid]
                cmp = compare_ids(oracle_data["tokens"], stoke_ids)
                if cmp.identical:
                    slot_status[slot] = {"status": "PASS", "prompt_id": pid,
                                          "oracle_len": cmp.oracle_len, "stoke_len": cmp.stoke_len}
                    continue

                pos = cmp.first_diff_pos
                oracle_top5 = top5_for_position(oracle_data["tokens"],
                                                 oracle_data["completion_probabilities"], pos)
                stoke_top5 = run_stoke_probe(corpus_path(pid), args.n_predict, pos)
                waived, decision = ledger.try_waive(pid, pos, oracle_top5, stoke_top5)
                if waived:
                    slot_history[slot].append({"prompt_id": pid, "position": pos, **decision})
                    slot_active[slot] = decision["replacement"]
                    next_pending.append(slot)
                else:
                    fail_info = {
                        "slot": slot, "prompt_id": pid, "position": pos,
                        "oracle_ids_len": cmp.oracle_len, "stoke_ids_len": cmp.stoke_len,
                        "oracle_id_at_pos": oracle_data["tokens"][pos] if pos < len(oracle_data["tokens"]) else None,
                        "stoke_id_at_pos": stoke_ids[pos] if pos < len(stoke_ids) else None,
                        "oracle_top5": oracle_top5, "stoke_top5": stoke_top5,
                        "decision": decision,
                    }
                    print("GATE FAIL", file=sys.stderr)
                    print(json.dumps(fail_info, indent=2, default=str), file=sys.stderr)
                    write_report(report_path, identity, args, slot_status, slot_history,
                                 gate_fail=fail_info, waivers=ledger.records)
                    return 1
            pending_slots = next_pending

        write_report(report_path, identity, args, slot_status, slot_history,
                     gate_fail=None, waivers=ledger.records)
        print(f"GATE PASS -- {len(selected)} prompts, {len(ledger.records)} "
              f"waiver(s) used (budget {WAIVER_BUDGET}). Report: {report_path}")
        return 0
    finally:
        server.stop()
        server.verify_dead_or_exit()


# ============================================================================
# --survey: DIAGNOSTIC ONLY -- manager addition 2026-07-27. Measures the
# divergence-class census across the corpus for the near-tie policy memo.
# Structurally SEPARATE from run_gate(): does not import/use WaiverLedger at
# all (so budget/reserve-consumption is not merely "disabled by a flag", it
# is ABSENT from this code path by construction), never calls write_report()
# (uses write_survey_report() instead), and never prints GATE PASS/FAIL.
#
# Per-prompt isolation: unlike run_gate (which FAILS LOUD immediately on any
# GateError, correct for a gate authority), run_survey catches GateError
# PER PROMPT during both the raw oracle/stoke fetch and the divergence-
# evidence gathering, recording an ERROR/EVIDENCE_UNAVAILABLE row and
# continuing -- because "every prompt is evaluated independently" and a
# multi-hour unattended run must not be voided by one flaky prompt.
#
# On the "incompatible with anything that implies the recorded run"
# instruction: the current CLI has no separate flag that means "this IS the
# recorded run" (the recorded run is procedurally defined as "--clean
# without --survey", not a distinct flag) -- the guarantee here is
# structural instead: --survey is dispatched to this wholly separate
# function in main(), so it can never ALSO produce a run_gate() result in
# the same invocation. --selftest + --survey together is rejected explicitly
# below since selftest is fixture-only and unrelated to the corpus/oracle.
# Flagged in m2-w4-report.md for the manager to confirm this reading.
# ============================================================================

def run_survey(args) -> int:
    if args.jobs < 1:
        fatal(f"--jobs must be >= 1, got {args.jobs}")
    if args.n_predict < 1:
        fatal(f"--n-predict must be >= 1, got {args.n_predict}")
    if not (1 <= args.port <= 65535):
        fatal(f"--port {args.port} out of range")

    selected = parse_only(args.only)
    missing = [pid for pid in selected if not corpus_path(pid).exists()]
    if missing:
        fatal(f"corpus file(s) missing for id(s) {missing!r} under {CORPUS_DIR}")

    report_path = Path(args.report) if args.report else DEFAULT_SURVEY_REPORT_PATH

    server = OracleServer(args.port)
    server.start()
    try:
        server.wait_healthy()
        identity = server.capture_identity()

        ordered = _order_longest_first(selected)

        oracle_results: Dict[str, dict] = {}
        oracle_errors: Dict[str, str] = {}
        print(f"MODE: SURVEY -- DIAGNOSTIC ONLY, NOT A GATE RESULT "
              f"({len(ordered)} prompts, jobs={args.jobs}, n_predict={args.n_predict})")
        for i, pid in enumerate(ordered, 1):
            try:
                oracle_results[pid] = fetch_oracle_result(server, pid, args)
                print(f"[oracle {i}/{len(ordered)}] {pid} ok")
            except GateError as e:
                oracle_errors[pid] = str(e)
                print(f"[oracle {i}/{len(ordered)}] {pid} ERROR: {e}")

        def _safe_fetch_stoke(pid: str):
            try:
                return "ok", fetch_stoke_result(pid, args)
            except GateError as e:
                return "err", str(e)

        stoke_results: Dict[str, List[int]] = {}
        stoke_errors: Dict[str, str] = {}
        # Progress is printed per completed prompt, with a wall clock and a
        # running count. This is NOT cosmetic: the stoke phase is hours long
        # (~5.3s per forward pass), and without it the operator cannot tell a
        # working run from a hung one, or see where a killed run stopped.
        # Pair with `python -u` (see run_survey_detached.cmd) -- buffered
        # stdout redirected to a file would defeat this entirely.
        t0 = time.monotonic()
        done = 0
        total = len(ordered)
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_safe_fetch_stoke, pid): pid for pid in ordered}
            for fut in as_completed(futs):
                pid = futs[fut]
                status, payload = fut.result()
                if status == "ok":
                    stoke_results[pid] = payload
                else:
                    stoke_errors[pid] = payload
                done += 1
                el = time.monotonic() - t0
                rate = el / done
                print(f"[stoke {done}/{total}] {pid} {status} -- "
                      f"elapsed {el/60:.1f} min, ~{rate/60:.1f} min/prompt, "
                      f"eta ~{(rate * (total - done))/60:.0f} min")

        census = {"exact": [], "diverged": [], "error": []}
        per_prompt: List[dict] = []
        for pid in ordered:
            if pid in oracle_errors or pid in stoke_errors:
                census["error"].append(pid)
                per_prompt.append({
                    "prompt_id": pid, "status": "ERROR",
                    "oracle_error": oracle_errors.get(pid),
                    "stoke_error": stoke_errors.get(pid),
                })
                continue

            oracle_data = oracle_results[pid]
            stoke_ids = stoke_results[pid]
            cmp = compare_ids(oracle_data["tokens"], stoke_ids)
            if cmp.identical:
                census["exact"].append(pid)
                per_prompt.append({"prompt_id": pid, "status": "EXACT",
                                    "oracle_len": cmp.oracle_len, "stoke_len": cmp.stoke_len})
                continue

            pos = cmp.first_diff_pos
            row = {
                "prompt_id": pid, "status": "DIVERGED", "position": pos,
                "oracle_ids_len": cmp.oracle_len, "stoke_ids_len": cmp.stoke_len,
                "oracle_id_at_pos": oracle_data["tokens"][pos] if pos < len(oracle_data["tokens"]) else None,
                "stoke_id_at_pos": stoke_ids[pos] if pos < len(stoke_ids) else None,
            }
            try:
                oracle_top5 = top5_for_position(oracle_data["tokens"],
                                                 oracle_data["completion_probabilities"], pos)
                stoke_top5 = run_stoke_probe(corpus_path(pid), args.n_predict, pos)
                decision = waiver_decision(oracle_top5, stoke_top5)
                row["oracle_top5"] = oracle_top5
                row["stoke_top5"] = stoke_top5
                row["would_waive"] = decision["eligible"]
                row["verdict_class"] = decision["class"]
                row["reason"] = decision["reason"]
            except GateError as e:
                row["would_waive"] = False
                row["verdict_class"] = "evidence_unavailable"
                row["reason"] = str(e)
            census["diverged"].append(pid)
            per_prompt.append(row)

        write_survey_report(report_path, identity, args, per_prompt, census)
        print("MODE: SURVEY -- DIAGNOSTIC ONLY, NOT A GATE RESULT")
        print(f"SURVEY COMPLETE -- {len(census['exact'])}/{len(ordered)} token-exact, "
              f"{len(census['diverged'])}/{len(ordered)} diverged, "
              f"{len(census['error'])}/{len(ordered)} tooling error(s). "
              f"Report: {report_path}")
        print("(this is a measurement, not a gate verdict -- no PASS/FAIL is implied)")
        return 0
    finally:
        server.stop()
        server.verify_dead_or_exit()


def run_dry_run(args) -> int:
    """Prints the plan and exits -- starts nothing, sends nothing (no
    server, no --version probe, no requests)."""
    selected = parse_only(args.only)
    reserve_ids = default_reserve_ids()
    server = OracleServer(args.port)

    print("=== parity_gate.py --dry-run (no server started, nothing executed) ===")
    if args.survey:
        print()
        print("MODE: SURVEY -- DIAGNOSTIC ONLY, NOT A GATE RESULT")
        print("  budget/reserve enforcement: ABSENT (WaiverLedger is not used in survey mode)")
        print("  early-stop on non-waivable divergence: DISABLED (every prompt evaluated independently)")
        print(f"  default survey report path: {DEFAULT_SURVEY_REPORT_PATH}")
    print()
    print("Oracle server argv:")
    print(f"  {_display_argv(server.argv())}")
    print(f"  oracle exe exists: {ORACLE_EXE.exists()}  ({ORACLE_EXE})")
    print(f"  model exists: {MODEL_PATH.exists()}  ({MODEL_PATH})")
    if MODEL_PATH.exists():
        actual = MODEL_PATH.stat().st_size
        ok = "OK" if actual == MODEL_SIZE_EXPECTED else "MISMATCH"
        print(f"  model size: {actual} bytes (expected {MODEL_SIZE_EXPECTED}, {ok})")
    print(f"  health url: http://127.0.0.1:{args.port}/health "
          f"(timeout {ORACLE_HEALTH_TIMEOUT_S}s, ~1s poll interval)")
    print(f"  KV setting: DEFAULT (no -ctk/-ctv) -- settled A5 outcome E3, M2 note §8")
    print("Build-identity probe (instant, no model load, no port):")
    print(f"  {_display_argv([str(ORACLE_EXE), '--version'])}")
    print()
    print(f"stoke exe exists: {STOKE_EXE.exists()}  ({STOKE_EXE})")
    print()
    print(f"Resolved prompt slots ({len(selected)}): {', '.join(selected)}")
    missing = [p for p in selected if not corpus_path(p).exists()]
    print(f"  missing: {missing if missing else '(none)'}")
    if not args.survey:
        print(f"Reserve pool ({len(reserve_ids)}): {', '.join(reserve_ids)}")
        r_missing = [p for p in reserve_ids if not corpus_path(p).exists()]
        print(f"  missing: {r_missing if r_missing else '(none)'}")
    existing_selected = [p for p in selected if corpus_path(p).exists()]
    ordered_preview = _order_longest_first(existing_selected)
    preview_str = ", ".join(ordered_preview[:10]) + (" ..." if len(ordered_preview) > 10 else "")
    print(f"Stoke job-pool submission order (longest-first by file size): {preview_str}")
    print()
    print("Stoke generate command template:")
    print(f"  {_display_argv(stoke_generate_argv(Path('<prompt-file>'), args.n_predict))}")
    print("Stoke probe command template:")
    print(f"  {_display_argv(stoke_probe_argv(Path('<prompt-file>'), args.n_predict, '<pos>'))}")
    print()
    print(f"n_predict={args.n_predict}  jobs={args.jobs}  port={args.port}  clean={args.clean}  "
          f"survey={args.survey}")
    if not args.survey:
        print(f"waiver epsilon={EPSILON} nats  waiver budget={WAIVER_BUDGET}")
    print(f"cache dir: {CACHE_DIR}")
    default_path = DEFAULT_SURVEY_REPORT_PATH if args.survey else DEFAULT_REPORT_PATH
    report_path = Path(args.report) if args.report else default_path
    print(f"report path: {report_path}")
    print()
    print("Parallelism: oracle /completion calls are SERIALIZED (server runs "
          "--parallel 1); only stoke.exe invocations are parallelized, bounded "
          "by --jobs. NOTE: --jobs is a RAM ceiling, not a core count -- each "
          "stoke.exe holds ~1,181 MB private (eager K+V allocation at "
          "model_open); see README-gate.md.")
    return 0


# ============================================================================
# --selftest -- SERVER-FREE, against committed fixtures only
# ============================================================================

def run_selftest() -> int:
    failures = 0

    def check(label: str, ok: bool):
        nonlocal failures
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures += 1

    # ---- 1. Aligner vs the committed v3 fixture -----------------------
    print("--- 1. Aligner vs tests/tooling/validation/oracle_v3_multilingual.json ---")
    try:
        with open(VALIDATION_FIXTURE, "r", encoding="utf-8") as f:
            fixture = json.load(f)
    except OSError as e:
        check(f"load fixture {VALIDATION_FIXTURE}", False)
        print(f"  error: {e}")
        fixture = None

    if fixture is not None:
        tokens = fixture["tokens"]
        entries = fixture["completion_probabilities"]
        check("fixture has 64 tokens", len(tokens) == 64)
        check("fixture has 54 completion_probabilities entries", len(entries) == 54)

        try:
            mapping = align_tokens_to_entries(tokens, entries)
        except GateError as e:
            check(f"aligner completes without a fatal ({e})", False)
            mapping = None

        if mapping is not None:
            check("every one of the 64 token positions maps to a valid entry index",
                  len(mapping) == 64 and all(0 <= ei < len(entries) for ei in mapping))
            check("the mapping is monotonic",
                  all(mapping[i] <= mapping[i + 1] for i in range(len(mapping) - 1)))

            id_bytes = harvest_token_bytes(entries)
            total_tok_bytes = sum(len(id_bytes[t]) for t in tokens)
            total_entry_bytes = sum(len(bytes(e["bytes"])) for e in entries)
            check(f"total mapped bytes ({total_entry_bytes}) == total token bytes "
                  f"({total_tok_bytes})", total_tok_bytes == total_entry_bytes)

            # HARD-CODED expected values read directly from the committed
            # JSON (m2-w4-report.md documents the derivation): position 0
            # (no coalescing yet) and position 8, the first position AFTER
            # the FIRST coalescing run (tokens[5:8] = [11162,234,235], the
            # split ' <U+1F30D>' emoji, all merge into entry index 5).
            e0 = entries[mapping[0]]
            check("position 0 -> top_logprobs[0].id == 271",
                  e0["top_logprobs"][0]["id"] == 271)
            check("position 0 -> top_logprobs[0].logprob == -0.3127651810646057",
                  e0["top_logprobs"][0]["logprob"] == -0.3127651810646057)

            e8 = entries[mapping[8]]
            check("position 8 (after 1st coalescing point) -> top_logprobs[0].id == 144232",
                  e8["top_logprobs"][0]["id"] == 144232)
            check("position 8 -> top_logprobs[0].logprob == -0.06248762086033821",
                  e8["top_logprobs"][0]["logprob"] == -0.06248762086033821)

            # Not vacuous: confirm the coalescing actually happened as
            # documented -- positions 5,6,7 (the swallowed UTF-8 split)
            # share one entry, distinct from their neighbors.
            check("positions 5,6,7 (the swallowed UTF-8 split) map to the same entry",
                  mapping[5] == mapping[6] == mapping[7])
            check("that shared entry differs from positions 4's and 8's entries",
                  mapping[5] != mapping[4] and mapping[5] != mapping[8])

    # ---- 2. Waiver decision unit tests (synthetic, no server) ----------
    print("\n--- 2. Waiver decision unit tests (synthetic top-5 dicts, no server) ---")

    # (a) oracle_gap 5e-4, stoke_gap 8e-4, same pair swapped -> WAIVE
    o_a = [(100, 0.0), (200, -5e-4), (9, -9.0), (8, -8.0), (7, -7.0)]
    s_a = [(200, 0.0), (100, -8e-4), (9, -9.0), (8, -8.0), (7, -7.0)]
    d_a = waiver_decision(o_a, s_a)
    check("(a) oracle_gap=5e-4, stoke_gap=8e-4, same pair swapped -> WAIVE", d_a["eligible"])

    # (b) oracle_gap 2.6e-2, stoke_gap 2.6e-2, same pair (v1-class magnitude) -> FAIL (gap >= eps)
    o_b = [(100, 0.0), (200, -2.6e-2), (9, -9.0), (8, -8.0), (7, -7.0)]
    s_b = [(200, 0.0), (100, -2.6e-2), (9, -9.0), (8, -8.0), (7, -7.0)]
    d_b = waiver_decision(o_b, s_b)
    check("(b) oracle_gap=2.6e-2, stoke_gap=2.6e-2, same pair -> FAIL (gap >= eps)",
          not d_b["eligible"] and "gap" in d_b["reason"])

    # (c) both gaps < eps but DIFFERENT top-2 pairs -> FAIL
    o_c = [(100, 0.0), (200, -5e-4), (9, -9.0), (8, -8.0), (7, -7.0)]
    s_c = [(300, 0.0), (400, -5e-4), (9, -9.0), (8, -8.0), (7, -7.0)]
    d_c = waiver_decision(o_c, s_c)
    check("(c) both gaps < eps but different top-2 pairs -> FAIL",
          not d_c["eligible"] and "pair" in d_c["reason"])

    # Bonus: the ACTUAL v1 near-tie numbers from the 2026-07-20 memo
    # (oracle logprobs vs ref/stoke logits -- gap-in-logit-space equals
    # gap-in-logprob-space for the same distribution, see waiver_decision
    # docstring) -- confirms this real, motivating case is non-waivable
    # for BOTH reasons (gap >= eps AND not a clean pair swap), matching
    # the memo's own conclusion.
    o_v1 = [(2326, -1.4927), (1378, -1.5191), (825, -1.5204), (264, -1.9065), (4236, -3.0839)]
    s_v1 = [(825, 22.9451), (2326, 22.8749), (1378, 22.8518), (264, 22.4663), (4236, 21.3111)]
    d_v1 = waiver_decision(o_v1, s_v1)
    check("(bonus) real v1 near-tie (memo 2026-07-20) is non-waivable", not d_v1["eligible"])

    # ---- 3. Reserve ordering + waiver budget ---------------------------
    print("\n--- 3. Reserve ordering + waiver budget (stateful WaiverLedger) ---")
    ledger = WaiverLedger(budget=WAIVER_BUDGET, reserve_ids=["r00", "r01", "r02"])
    w1, d1 = ledger.try_waive("p01", 10, o_a, s_a)
    check("waiver 1 succeeds", w1)
    check("waiver 1 consumes r00 (reserve ordering)", d1.get("replacement") == "r00")
    w2, d2 = ledger.try_waive("p02", 20, o_a, s_a)
    check("waiver 2 succeeds", w2)
    check("waiver 2 consumes r01 (reserve ordering)", d2.get("replacement") == "r01")
    # (d) a 3rd waiver in one run -> FAIL (budget), even though otherwise eligible
    w3, d3 = ledger.try_waive("p03", 30, o_a, s_a)
    check("(d) 3rd waiver in one run -> FAIL (budget)",
          (not w3) and "budget" in d3["reason"].lower())
    check("budget rejection leaves the reserve pointer at r02 (unconsumed)",
          len(ledger.records) == 2)

    # ---- 4. Build-identity parser (manager-measured 2026-07-27 example) --
    print("\n--- 4. parse_version_output() vs the manager's live-captured example ---")
    manager_stderr = "version: 10037 (56d6e9dde)\nbuilt with Clang 20.1.8 for Windows x86_64\n"
    parsed = parse_version_output(manager_stderr, "")
    check("build_number == '10037'", parsed["build_number"] == "10037")
    check("commit == '56d6e9dde'", parsed["commit"] == "56d6e9dde")
    check("raw_stderr preserved", "built with Clang 20.1.8" in parsed["raw_stderr"])

    # negative test: must FATAL on output with no 'version: N (hash)' line,
    # not silently return an empty/default build-identity dict.
    caught = False
    try:
        parse_version_output("some unrelated error text\n", "")
    except GateError:
        caught = True
    check("parse_version_output raises GateError on unparseable input "
          "(not a dead check -- proves the fatal path is real)", caught)

    print()
    if failures == 0:
        print("GATE SELFTEST: ALL GREEN")
        return 0
    print(f"GATE SELFTEST: {failures} FAILURE(S)")
    return 1


# ============================================================================
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parity_gate.py",
        description="M2 W4 parity gate harness -- stoke vs llama-server greedy "
                     "token parity, §8 waiver protocol.")
    p.add_argument("--clean", action="store_true",
                    help="ignore + refresh the dev cache; THE recorded gate run uses this")
    p.add_argument("--only", metavar="p00,p07,...", default=None,
                    help="comma-separated prompt ids to run (default: all 50 p00..p49)")
    p.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help=f"bounded parallel pool size for stoke.exe invocations "
                         f"(oracle calls are always serialized) (default {DEFAULT_JOBS})")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"port for the llama-server oracle (default {DEFAULT_PORT})")
    p.add_argument("--n-predict", type=int, default=N_PREDICT_DEFAULT, dest="n_predict",
                    help=f"tokens to generate per prompt, both sides (default {N_PREDICT_DEFAULT})")
    p.add_argument("--report", metavar="PATH", default=None,
                    help="report output path (default: a dev path under tests/tooling/)")
    p.add_argument("--selftest", action="store_true",
                    help="run the SERVER-FREE selftest (aligner + waiver-decision unit tests) and exit")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="print the plan and exit without starting the server or running anything")
    p.add_argument("--survey", action="store_true",
                    help="DIAGNOSTIC ONLY: run all selected prompts to completion, never "
                         "early-stop, no budget/reserve enforcement -- for the near-tie "
                         "divergence-class census. NOT a gate result (manager addition 2026-07-27).")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.selftest and args.survey:
            fatal("--selftest and --survey are mutually exclusive (selftest is "
                  "fixture-only and touches no corpus/oracle)")
        if args.selftest:
            return run_selftest()
        if args.dry_run:
            return run_dry_run(args)
        if args.survey:
            return run_survey(args)
        return run_gate(args)
    except GateError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
