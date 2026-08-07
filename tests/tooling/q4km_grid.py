#!/usr/bin/env python3
"""tests/tooling/q4km_grid.py -- execute [2v2-iii]'s Q4_K_M reference grid.

BINDING SOURCE: docs/superpowers/memos/2026-08-06-m3-probe-grid-RULING-FABLE.md
section 5 (the grid is FIXED there), as published with one band correction in
docs/superpowers/gate-reports/2026-08-07-m3-q4km-grid-published.md. Where a
reader thinks this file and the ruling differ, THE RULING WINS and this file is
the bug.

▶▶ WHAT THIS PROGRAM DOES AND WHAT IT REFUSES TO DO. It runs `ref_forward.py`
over the fixed grid, banks one probe file per leg, and locates the FIRST
position where ref's greedy stream leaves the banked HEAD stream. It does not
freeze tau, it does not grade, and it never invokes stoke.exe to produce a
number -- it only hashes it. That split is the ruling's ordering rule made
structural rather than remembered (section 2.3: the grid runs and freezes tau
BEFORE divergences are graded), and it is the same split m2_classify_probes.py
made for [2v2-ii] and for the same reason.

▶▶ WHY EVERY LEG COSTS `prompt_len + 64` POSITIONS, INCLUDING A CONTINUATION.
`ref_forward.py` prefills the WHOLE prefix on every invocation and
re-dequantizes every weight at every position (the deliberate RAM trade-off
recorded 2026-08-05: caching a 4B model's f32 weights is ~16 GB on a 15.63 GB
box). A continuation therefore does NOT resume cheaply from index j -- it
re-prefills prompt + (j+1) banked ids and then generates the remaining
63-j. Total positions are the same as the original leg. Costing a continuation
as "the tail only" is the arithmetic error that would blow the window.

▶▶ THE WINDOW BOUND IS STRUCTURAL, NOT WATCHED. Section 5 authorizes windows of
<=4 h. This driver refuses to START a leg whose projected finish would pass the
budget, and stops cleanly at that boundary with every completed leg banked and
resumable. A run that ends early is a correct run, not an interrupted one --
the alternative is an operator watching a 3 h detached job, which is how the
<=4 h authorization gets exceeded by accident.

▶ THE RATE PRIOR IS MEASURED ON THIS FILE, NOT INHERITED FROM Q8_0.
`REF_SEC_PER_POS_PRIOR` below is a Q4_K_M measurement (see its comment). The
2026-08-05 fit (23.8 + 18.42*P) was taken on the Q8_0 file, and Q4_K/Q6_K
superblock unpacking in numpy is not the same work as Q8_0's scale-multiply --
inheriting that constant across files is exactly the "quantity asserted from
structure rather than computed" failure this milestone has recorded three
times. The prior is only ever used for the FIRST leg; from the second leg on
the driver projects from what it has actually observed this run.

▶ RESUME IS THE OUTPUT DIRECTORY'S. A leg is done iff its output file exists
AND revalidates (every structural check re-run). There is no separate progress
store to drift out of sync with the artifacts, and a corrupted half-written
file is re-run rather than trusted -- the same reasoning bank_streams.py gives
for leaning on parity_gate's cache instead of inventing one.

▶ THE G11 FREEZE IS RE-CHECKED HERE, NOT INHERITED. The banked HEAD streams
this driver compares against are keyed on stoke.exe's sha256; the grid's later
stoke-side probes must come from that same binary. A rebuild between banking
and this run would silently compare ref against a stream no live binary
produces, so the hash is verified before any leg starts.

CLI:
  python q4km_grid.py --run A [--out DIR] [--budget-s N] [--dry-run]
  python q4km_grid.py --run B [...]
  python q4km_grid.py --continue-from RUNMANIFEST.json [...]
  python q4km_grid.py --selftest
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

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT / "tests" / "tooling" / "ref_forward.py"
STOKE_EXE = ROOT / "build" / "win" / "stoke.exe"
MODEL = ROOT / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MODEL_SIZE_EXPECTED = 2497280736          # ruling section 5's launch-condition operand
CORPUS = ROOT / "tests" / "data" / "parity"
OUT_DEFAULT = ROOT / "tests" / "tooling" / ".gate_cache" / "q4km_grid"

# The frozen binary (ruling G11). Full hash, not a prefix: a prefix comparison
# is a check that gets weaker the more it is copied.
STOKE_HASH_FROZEN = "876ce53fd4a943fdd820f0bf9e8328b8aa751fa1269df9fd20f4fedbfb6390ff"

PAIR = "f32_intcore_q4_k_m"   # per-pair AND per-file (ruling 9.4-2, G6). Q8_0's
                              # key is f32_intcore_q8_0 and its tau NEVER
                              # transfers here in either direction.

N_PREDICT = 64                # section 5's grid; parity_gate.N_PREDICT_DEFAULT

# Grid prompts, in the ruling's stated order. RUN A's p00 pair is first because
# it is the section 8A discharge gate: the tool is validated before ~3 h is
# spent with it, and a mismatch stops the leg.
RUN_A_PROMPTS = ["p00", "p10", "p20", "p30", "p45"]
RUN_B_PROMPTS = ["p41"]

# Q4_K_M rate prior, measured on THIS file 2026-08-07. Two points were taken
# and they DO NOT AGREE, which is recorded rather than averaged away:
#
#   P=16  (p45, -n 2)   906 s   ullAvailPhys ~2.5 GB, selftests running
#   P=2   (1 tok, -n 1)  77 s   ullAvailPhys ~6.0 GB, box otherwise idle
#
# An affine fit through those two gives a NEGATIVE intercept (-41 s), which is
# not a cost model -- it is the signature of two measurements taken in
# different box states, the 2026-08-05 lesson that box state governs VALIDITY
# for a perf number. Both are therefore AMBIENT-GRADE and neither is banked as
# a cost constant. `fit_cost_model` refuses the negative-intercept fit by rule.
#
# The prior is set to the CONSERVATIVE of the two (the origin fit at the larger
# P), and it matters less than it looks: it gates only the FIRST leg's start
# decision, after which the driver re-fits from legs it has actually run at
# real sizes. Chasing a better prior with more 15-minute probes would be paying
# for a number the run itself produces for free.
REF_SEC_PER_POS_PRIOR = 56.6      # 906/16, loaded box -- deliberately the high one
REF_INTERCEPT_S = 23.8            # process start + GGUF header parse (Q8_0 fit)

WINDOW_BUDGET_S = 4 * 3600        # section 5: every window <=4 h

# Section 5's ops constraint is "free RAM > model + 2 GB". MEASURED on this
# file 2026-08-07: one ref_forward.py run on Q4_K_M holds a ~4.0 GB working set
# and consumed 4.62 GB of ullAvailPhys (7.09 -> 2.47) with the model resident.
# So model + 2.0 = 4.326 GB is about 0.3 GB optimistic: a launch at exactly the
# ruling's bar would run with the model being trimmed, which IS the recorded
# page-cache collapse the constraint exists to prevent.
#
# DECIDED (2026-08-07, Opus, under the governance change): raise the margin to
# 2.5 GB. Direction check -- this is an OPS constraint, not a section 8.4
# criterion constant, and the change is strictly stricter: it refuses launches
# the ruling would have allowed and permits none the ruling forbids.
RAM_MARGIN_GB = 2.5               # was 2.0; see the measurement above

# Private (non-shared) footprint of one ref_forward.py instance: the 1.56 GB
# f32 _embd_cache plus per-layer dequant transients. Measured 2026-08-07 as
# 3.9 GB working set minus the ~2.33 GB mapped model, which a second instance
# on the SAME file shares rather than duplicates.
PRIVATE_PER_INSTANCE_GB = 1.9

# Applied to every window-guard projection. The fit is a two-point line from
# this run's own legs; the margin covers box drift over a multi-hour window,
# not a modelling gap. It is deliberately small -- a large factor would make
# the guard refuse legs that fit, which costs a whole window.
WINDOW_SAFETY = 1.15


class GridError(RuntimeError):
    pass


# ============================================================================
# environment guards -- each one is a precondition the ruling names by hand
# ============================================================================

def free_gb():
    """ullAvailPhys -- the guard's own number, and no other.

    Win32_OperatingSystem.FreePhysicalMemory read 3.31 GB against this
    function's 7.10 GB on this box on 2026-08-06: it counts the free list only
    and excludes standby pages, which is exactly where an mmapped model wants
    to live. Section 5 says "by ullAvailPhys (the guard's own number -- quote
    no other)"; this is that function, copied from bank_streams.py so the two
    launch conditions cannot drift apart.
    """
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


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_freeze():
    """G11: stoke.exe is frozen until `git tag m3`."""
    if not STOKE_EXE.exists():
        raise GridError(f"stoke.exe not found at {STOKE_EXE}")
    got = sha256_file(STOKE_EXE)
    if got != STOKE_HASH_FROZEN:
        raise GridError(
            f"G11 FREEZE VIOLATED: stoke.exe is {got[:16]}..., expected "
            f"{STOKE_HASH_FROZEN[:16]}.... The banked Q4_K_M HEAD streams this "
            f"grid compares against were generated by the frozen binary and are "
            f"keyed on its hash; comparing ref against a stream no live binary "
            f"produces is not evidence. Restore the frozen build before running.")
    return got


def pin_model_size():
    """Fail loudly if the model file is not the one the size constant names.

    Same reasoning bank_streams.py records: a truncated or swapped file would
    produce legs that look exactly like real ones, and the launch condition is
    computed FROM this constant, so a wrong file also silently wrongs the RAM
    guard.
    """
    if not MODEL.exists():
        raise GridError(f"model not found at {MODEL}")
    actual = MODEL.stat().st_size
    if actual != MODEL_SIZE_EXPECTED:
        raise GridError(f"model size mismatch: expected {MODEL_SIZE_EXPECTED} "
                        f"bytes, found {actual}")
    return actual


def running_ref_instances():
    """PIDs of live ref_forward.py processes, found by COMMAND LINE.

    By command line, never by image name: every one of these is `python.exe`,
    and the 2026-07-28 kill-order lesson is that identifying them by name is
    how the wrong process gets touched.
    """
    if sys.platform != "win32":
        return []
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*ref_forward.py*' } | "
             "ForEach-Object { $_.ProcessId }"],
            capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]


def ram_requirement(model_bytes: int, additional_instance: bool, live_instances):
    """The launch bar, as a PURE function of its inputs.

    Split out from check_ram because the version that read the live process
    list inline could only be tested against whatever happened to be running:
    the selftest for "the discount is refused with nothing running" passed
    vacuously while a leg was in flight, and the mutation harness caught it
    granting the discount unconditionally. Taking `live_instances` as an
    argument makes both branches decidable on demand. Fourth recording of a
    test that passed for a reason unrelated to the thing it names.
    """
    if not additional_instance:
        return model_bytes / (1024.0 ** 3) + RAM_MARGIN_GB
    if not live_instances:
        raise GridError(
            "--additional-instance was passed but no ref_forward.py process is "
            "running. The reduced bar is justified ONLY by the model already "
            "being mapped by another instance; with none running this instance "
            "must pay the full launch condition.")
    return PRIVATE_PER_INSTANCE_GB + 0.5


def check_ram(model_bytes: int, additional_instance: bool = False):
    """Launch condition. `additional_instance` charges only the PRIVATE
    footprint, and only when the fact that justifies it is verified.

    ▶ WHY A SECOND INSTANCE IS CHEAPER, AND WHY THE DISCOUNT IS CONDITIONAL.
    gguf_reader mmaps the model, so two processes reading the SAME file share
    those pages -- the second instance's marginal cost is its private working
    set (the ~1.56 GB f32 _embd_cache plus transients), not another copy of the
    model. Measured 2026-08-07: 3.9 GB working set of which ~2.33 GB is the
    mapped file. But that discount is only real if another instance is
    ACTUALLY mapping the file, so this VERIFIES one is running rather than
    trusting the flag. A flag that weakens a guard on an unchecked assertion is
    the failure this project keeps recording.
    """
    free = free_gb()
    need = ram_requirement(model_bytes, additional_instance,
                           running_ref_instances() if additional_instance else [])
    if free is None:
        return None, need
    if free < need:
        raise GridError(
            f"refusing to launch: needs > {need:.3f} GB by ullAvailPhys "
            f"(model {model_bytes / 1024.0 ** 3:.3f} + {RAM_MARGIN_GB} margin), "
            f"free {free:.2f} GB. The recorded failure mode is a page-cache "
            f"collapse (cache faults ~60 -> ~7,000, ~3x slower), which takes the "
            f"run past the window it is authorized under.")
    return free, need


def check_no_q8_holder():
    """Section 5: never concurrent with anything mapping the Q8_0 file.

    Alternating two large mmapped models is the recorded 2026-08-04 page-cache
    churn. This cannot prove nothing maps the file -- it checks the processes
    this tree is capable of starting, which is where the recorded collapses
    came from -- so it reports rather than certifies, and says so.
    """
    if sys.platform != "win32":
        return []
    try:
        r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    watch = ("stoke.exe", "llama-server.exe", "llama-cli.exe", "main.exe")
    found = []
    for line in r.stdout.splitlines():
        for name in watch:
            if f'"{name}"' in line.lower():
                found.append(line.split(",")[0].strip('"'))
    return sorted(set(found))


# ============================================================================
# pure logic -- everything the selftest can exercise without a model run
# ============================================================================

def leg_positions(prompt_len: int, n_predict: int = N_PREDICT) -> int:
    """Forward positions one invocation costs: the whole prefill plus every
    generated position. A continuation re-prefills, so it costs the same as
    the leg it continues -- see the module docstring."""
    return prompt_len + n_predict


def project_cost_s(positions: int, rate_s_per_pos: float,
                   intercept_s: float = REF_INTERCEPT_S) -> float:
    return intercept_s + rate_s_per_pos * positions


def observed_rate(observations, prior: float) -> float:
    """Through-origin rate in s/position. Kept as the ONE-POINT fallback.

    For a true cost a + b*P this returns b + a/P, which OVERESTIMATES b. With
    one measurement that error is the safe direction for a window bound. With
    two it is simply the worse estimator, and `fit_cost_model` is used instead.
    """
    if not observations:
        return prior
    tot_s = sum(s for s, _ in observations)
    tot_p = sum(p for _, p in observations)
    if tot_p <= 0:
        return prior
    return tot_s / tot_p


def fit_cost_model(observations, prior_rate: float,
                   prior_intercept: float = REF_INTERCEPT_S):
    """Fit cost(P) = intercept + rate*P from this run's completed legs.

    ▶ WHY THE INTERCEPT IS NOT A ROUNDING DETAIL HERE. `logits()` is tied to
    `token_embd` on this file (`output.weight` is absent), so it reads
    `Qwen3Ref._embd_cache` -- built ONCE, on the first embed() call, by
    dequantizing a Q6_K 2560x151936 tensor into 1.56 GB of f32. That one-time
    cost lands entirely in the intercept. A through-origin fit folds it into
    the slope, and a 16-position measurement then reports a per-position rate
    inflated by a/16. Projected onto p41's 541 positions that is not a
    conservative rounding -- it is large enough to refuse to start a leg that
    fits, which would block RUN B on an artifact of the fit.

    Returns (intercept_s, rate_s_per_pos). With fewer than two distinct leg
    sizes there is nothing to separate, so the origin fit stands.
    """
    pts = [(p, s) for s, p in observations if p > 0]
    if not pts:
        return prior_intercept, prior_rate
    if len({p for p, _ in pts}) < 2:
        return 0.0, observed_rate(observations, prior_rate)
    n = len(pts)
    sx = sum(p for p, _ in pts)
    sy = sum(s for _, s in pts)
    sxx = sum(p * p for p, _ in pts)
    sxy = sum(p * s for p, s in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, observed_rate(observations, prior_rate)
    rate = (n * sxy - sx * sy) / denom
    intercept = (sy - rate * sx) / n
    # A non-positive slope or a negative intercept means the data does not
    # support a line; refusing the fit is better than projecting from one.
    if rate <= 0 or intercept < 0:
        return 0.0, observed_rate(observations, prior_rate)
    return intercept, rate


def first_divergence(ref_ids, head_ids):
    """Locate where ref's greedy stream leaves the banked HEAD stream.

    Returns (kind, index):
      ("agree", None)   -- identical over the full compared length
      ("diverge", j)    -- both sides produced index j and chose differently
      ("length", L)     -- agree on every shared index; one side stopped at L

    The length case is NOT folded into "diverge". At index L the shorter side
    never produced a distribution, so no top-5 exists there on that side and
    section 6 cannot be applied -- m2_classify_probes.py reports exactly this
    shape as ESCALATE-BY-CONSTRUCTION rather than fabricating a fifth pair.
    Merging the two would let an unclassifiable position be graded.
    """
    n = min(len(ref_ids), len(head_ids))
    for j in range(n):
        if ref_ids[j] != head_ids[j]:
            return "diverge", j
    if len(ref_ids) != len(head_ids):
        return "length", n
    return "agree", None


def probe_pos_line(pos: int, tid: int, logit: float) -> str:
    """src/main.c's format, which ref_forward.py mirrors byte-for-byte."""
    return f"probe pos={pos}: id={tid} logit={logit:.6f}"


_PROBE_RE = re.compile(r"^probe pos=(\d+): id=(\d+) logit=(-?\d+\.\d{6})$")


def parse_probe_pos_stderr(text: str):
    """Parse the five `probe pos=` lines. Anything else on stderr is ignored,
    but a line that ALMOST matches is a parse failure rather than a skip --
    silently dropping a malformed probe line would turn a tooling defect into
    a passing invariance check."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("probe pos="):
            continue
        m = _PROBE_RE.match(line)
        if not m:
            raise GridError(f"malformed probe line: {line!r}")
        out.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return out


def check_invariance(bare_ids, probe_data, probe_pos, stderr_text):
    """Section 8A's OWED item, discharged empirically. Returns a record.

    TWO claims, and they are different claims:

      (1) STREAM INVARIANCE -- `--probe-all` must not change the ids. The
          argument was structural (append-only list, read by nobody, next_id
          still computed from lg) and `--selftest` passed, but it had never
          been confirmed on a model run because the box had 0.31 GB free.
          This compares the bare run's ids to the flagged run's ids.

      (2) INDEX AGREEMENT -- `--probe-all`'s record at index N and
          `--probe-pos N` must describe the same position. Both come from
          `top5_from_logits(lg)` on the same array, so a disagreement means
          the two paths index different positions, which would silently
          misalign every graded position in the grid.

    PRECISION, STATED RATHER THAN GLOSSED: --probe-pos prints %.6f, so the
    strongest available comparison is exact ids plus logits equal to six
    decimals. This is not bit-equality of the underlying floats and is not
    claimed as such; six decimals is what the printout carries.
    """
    rec = {"probe_pos": probe_pos, "checks": [], "ok": True}

    def add(name, ok, detail=""):
        rec["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            rec["ok"] = False

    flag_ids = probe_data["ids"]
    add("stream_invariance", bare_ids == flag_ids,
        "" if bare_ids == flag_ids else
        f"bare produced {len(bare_ids)} ids, --probe-all produced {len(flag_ids)}; "
        f"first difference at index "
        f"{next((i for i, (a, b) in enumerate(zip(bare_ids, flag_ids)) if a != b), 'n/a')}")

    recs = [p for p in probe_data["probes"] if p["index"] == probe_pos]
    add("probe_all_has_index", len(recs) == 1,
        f"found {len(recs)} records at index {probe_pos}")
    if len(recs) != 1:
        return rec

    printed = parse_probe_pos_stderr(stderr_text)
    add("probe_pos_line_count", len(printed) == 5, f"got {len(printed)} lines")
    if len(printed) != 5:
        return rec

    add("probe_pos_index", all(p == probe_pos for p, _, _ in printed),
        f"positions printed: {sorted({p for p, _, _ in printed})}")

    from_json = [probe_pos_line(probe_pos, tid, lg) for tid, lg in recs[0]["top5"]]
    from_err = [probe_pos_line(p, t, 0.0).rsplit(" logit=", 1)[0] + f" logit={l}"
                for p, t, l in printed]
    add("index63_matches_probe_pos", from_json == from_err,
        "" if from_json == from_err else
        f"json={from_json!r} stderr={from_err!r}")
    return rec


def expected_n_input(leg) -> int:
    """What `n_input` MUST be for this leg, derived from the leg rather than
    read back out of the artifact.

    Exists as its own function because the first version of the resume path
    passed `data["n_input"]` as the expected value -- comparing a field to
    itself, a check that cannot fail. `leg["ids"]` is the invocation's entire
    input on every leg kind (the prompt for a probe leg, prompt + forced banked
    ids for a continuation), so it is the independent quantity.
    """
    return len(leg["ids"])


def validate_probe_file(data, prompt_len: int):
    """Every structural claim a later consumer would otherwise assume.

    The rank-0 cross-check is the one that makes the probes mean anything: a
    probe set is only interpretable against the stream it was taken along, and
    `top5_from_logits` breaks ties by the LOWER id while the stream is chosen
    by np.argmax, which returns the first maximum -- the same token. So rank-0
    MUST equal the generated id at every index, and a mismatch means the two
    are not describing the same decision. m2_classify_probes.py requires the
    same identity of its two binaries and treats a mismatch as fatal.
    """
    errs = []
    for key in ("ids", "n_input", "probes"):
        if key not in data:
            errs.append(f"missing key {key!r}")
    if errs:
        return errs
    ids, probes = data["ids"], data["probes"]
    if data["n_input"] != prompt_len:
        errs.append(f"n_input {data['n_input']} != prompt length {prompt_len}")
    if len(probes) != len(ids):
        errs.append(f"{len(probes)} probes for {len(ids)} ids")
    seen = [p["index"] for p in probes]
    if seen != list(range(len(probes))):
        errs.append(f"probe indexes are not 0..{len(probes) - 1} in order")
    for p in probes:
        t5 = p["top5"]
        if len(t5) != 5:
            errs.append(f"index {p['index']}: {len(t5)} candidates, expected 5")
            continue
        lg = [l for _, l in t5]
        if any(lg[i] < lg[i + 1] for i in range(4)):
            errs.append(f"index {p['index']}: top5 not in descending logit order")
    for i, (tid, p) in enumerate(zip(ids, probes)):
        if p["top5"][0][0] != tid:
            errs.append(f"index {i}: rank-0 is {p['top5'][0][0]} but the stream "
                        f"chose {tid}; probe contract says rank-0 == argmax")
            break
    return errs


# ============================================================================
# leg construction and execution
# ============================================================================

def tokenize(pid: str) -> list:
    """Tokenize with the FROZEN binary against the Q4_K_M file.

    Not read from .gate_cache/prompt_token_lens.json: that file was produced
    against the Q8_0 model, and the band assignment this grid is graded under
    is keyed on prompt_len. Re-deriving it from the file the run actually uses
    costs seconds and removes an assumption that the two tokenizations agree.
    """
    argv = [str(STOKE_EXE), "tokenize", str(MODEL), str(CORPUS / f"{pid}.txt")]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise GridError(f"stoke tokenize failed on {pid} (rc={r.returncode}): "
                        f"{r.stderr.strip()[:400]}")
    line = None
    for cand in reversed(r.stdout.strip().splitlines()):
        if cand.strip():
            line = cand.strip()
            break
    if line is None:
        raise GridError(f"stoke tokenize produced no output for {pid}")
    m = re.match(r"^\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]\s*$", line)
    if not m:
        raise GridError(f"stoke tokenize output is not an id list for {pid}: {line!r}")
    return [int(s) for s in m.group(1).split(",")]


def banked_head_ids(pid: str) -> list:
    """HEAD's Q4_K_M stream for `pid`, from parity_gate's cache.

    Read through parity_gate rather than by opening the cache file directly:
    the four-part key (prompt hash, stoke hash, n_predict, invocation
    signature) is the thing that makes the entry trustworthy, and a second
    reader that reconstructs the path by hand is a second thing to keep in
    sync with the freeze rule.
    """
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import parity_gate as pg
    import bank_streams as bs
    bs.bind_model("q4_k_m")
    args = bs._Args("q4_k_m", N_PREDICT)
    data = pg._load_cache_data(pg.corpus_path(pid), args)
    ids = data.get("stoke_ids")
    if not ids:
        raise GridError(
            f"{pid}: no banked HEAD stream on the Q4_K_M file. The grid is "
            f"teacher-forced against these ids, so they must exist first: "
            f"python tests\\tooling\\bank_streams.py --model q4_k_m --only {pid}")
    return list(ids)


def filter_legs(legs, only):
    """Restrict to named prompts or legs, for running disjoint subsets as
    SEPARATE processes.

    ▶ WHY PARALLELISM IS A FILTER AND NOT A THREAD POOL IN HERE. Each leg is
    an independent ref_forward.py process that is ~1 core busy on a 10-core
    box, so the speedup is available from the OS for free. Putting a worker
    pool inside a gate tool at the point where its output becomes evidence
    buys the same wall-clock for new concurrency bugs in the one program that
    must not have any. ref_gate.py's --jobs did the same thing for M2.
    Disjointness is the caller's to arrange; the per-leg output files make an
    accidental overlap loud rather than silent, since the second finisher
    rewrites a file the first already validated.

    Accepts prompt ids (p20) or full leg names (p00-probe).
    """
    want = {s.strip() for s in only.split(",") if s.strip()}
    kept = [l for l in legs if l["leg"] in want or l["prompt"] in want]
    unknown = want - {l["leg"] for l in legs} - {l["prompt"] for l in legs}
    if unknown:
        raise GridError(f"--only names nothing in this run: {sorted(unknown)}")
    if not kept:
        raise GridError(f"--only {only!r} selected no legs")
    # The 8A pair is a DEPENDENCY, not a preference: p00-probe's discharge
    # compares against p00-bare's ids, and selecting the probe without the
    # bare run would silently produce a probe leg that cannot discharge 8A.
    names = {l["leg"] for l in kept}
    if "p00-probe" in names and "p00-bare" not in names:
        raise GridError(
            "--only selected p00-probe without p00-bare. Section 8A's discharge "
            "compares the flagged run's ids against the bare run's; select both "
            "(they are sequential) or neither.")
    return kept


def make_legs(prompts, tokens, run_name):
    """Build the leg list for a run. RUN A's p00 pair comes first, in the
    ruling's order, because it is the section 8A discharge gate."""
    legs = []
    for pid in prompts:
        plen = len(tokens[pid])
        if run_name == "A" and pid == "p00":
            legs.append({"leg": "p00-bare", "prompt": pid, "kind": "bare",
                         "ids": tokens[pid], "n": N_PREDICT,
                         "positions": leg_positions(plen)})
            legs.append({"leg": "p00-probe", "prompt": pid, "kind": "probe",
                         "ids": tokens[pid], "n": N_PREDICT,
                         "probe_pos": N_PREDICT - 1,
                         "positions": leg_positions(plen)})
        else:
            legs.append({"leg": f"{pid}-probe", "prompt": pid, "kind": "probe",
                         "ids": tokens[pid], "n": N_PREDICT,
                         "positions": leg_positions(plen)})
    return legs


def continuation_leg(pid, prompt_tokens, head_ids, j):
    """Section 5's REQUIRED continuation: re-lock the stream past the first
    divergence so the grid's position coverage returns to ~384 and the
    tripwire denominator holds.

    Forced prefix is prompt + banked HEAD ids[0..j] INCLUSIVE (j+1 ids) and
    the run generates 64-(j+1) more. j == 63 needs none: there is nothing
    after the last generated index to recover.
    """
    if not 0 <= j < N_PREDICT:
        raise GridError(f"continuation index {j} outside [0,{N_PREDICT})")
    n_left = N_PREDICT - (j + 1)
    if n_left <= 0:
        return None
    forced = list(prompt_tokens) + list(head_ids[:j + 1])
    return {"leg": f"{pid}-cont-j{j}", "prompt": pid, "kind": "cont",
            "ids": forced, "n": n_left, "forced_from": j,
            "positions": len(forced) + n_left}


def run_leg(leg, outdir: pathlib.Path, log):
    """One ref_forward.py invocation. Returns (seconds, generated ids, stderr).

    stderr is returned rather than only written because section 8A's index
    check reads the `--probe-pos` lines from THIS run -- comparing against a
    file on disk would let a stale printout from an earlier invocation
    discharge the check.
    """
    out_json = outdir / f"{leg['leg']}.json"
    argv = [sys.executable, "-u", str(REF), str(MODEL),
            "--ids", ",".join(str(i) for i in leg["ids"]), "-n", str(leg["n"])]
    if leg["kind"] != "bare":
        argv += ["--probe-all", str(out_json)]
    if leg.get("probe_pos") is not None:
        argv += ["--probe-pos", str(leg["probe_pos"])]

    shown = " ".join(argv[3:])          # everything after `python -u ref_forward.py`
    shown = re.sub(r"--ids [\d,]+", f"--ids <{len(leg['ids'])} ids>", shown)
    log(f"    exec: {shown}")
    t0 = time.time()
    r = subprocess.run(argv, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise GridError(f"{leg['leg']}: ref_forward.py exited {r.returncode} "
                        f"after {dt:.1f}s\n{r.stderr.strip()[-2000:]}")
    ids = [int(s) for s in r.stdout.split()]
    (outdir / f"{leg['leg']}.ids").write_text(
        ",".join(str(i) for i in ids), encoding="utf-8")
    if r.stderr.strip():
        (outdir / f"{leg['leg']}.err").write_text(r.stderr, encoding="utf-8")
    return dt, ids, r.stderr


# ============================================================================
# driver
# ============================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", choices=["A", "B"], help="which fixed run to execute")
    ap.add_argument("--continuations", action="store_true",
                    help="run the REQUIRED continuation for every divergence "
                         "already recorded in --out's manifest")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--budget-s", type=float, default=WINDOW_BUDGET_S)
    ap.add_argument("--rate-s-per-pos", type=float, default=REF_SEC_PER_POS_PRIOR,
                    help="prior for the FIRST leg only; measured on Q4_K_M")
    ap.add_argument("--only", default=None, metavar="p20,p30 or p00-bare",
                    help="run only these prompts/legs, so disjoint subsets can "
                         "run as separate concurrent processes")
    ap.add_argument("--additional-instance", action="store_true",
                    help="another ref_forward.py is already mapping the model, "
                         "so charge only this instance's private footprint "
                         "(verified, not trusted)")
    ap.add_argument("--dry-run", action="store_true",
                    help="guards, tokenization, banked ids, projections -- no model run")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return run_selftest()
    if not (a.run or a.continuations):
        ap.error("one of --run or --continuations is required")

    outdir = pathlib.Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    logpath = outdir / "grid.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(logpath, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    try:
        stoke_hash = check_freeze()
        size = pin_model_size()
        free, need = check_ram(size, a.additional_instance)
        holders = check_no_q8_holder()
    except GridError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 2

    log(f"=== q4km_grid run={a.run or 'continuations'} ===")
    log(f"model      : {MODEL.name} ({size} bytes, pinned)")
    log(f"stoke.exe  : {stoke_hash[:16]}... (G11 freeze verified)")
    log(f"free RAM   : {free:.2f} GB by ullAvailPhys (need > {need:.3f})"
        if free is not None else "free RAM   : n/a")
    log(f"budget     : {a.budget_s / 3600:.2f} h")
    if holders:
        log(f"WARNING    : processes that may map a model file are running: {holders}")

    prompts = RUN_A_PROMPTS if a.run == "A" else RUN_B_PROMPTS if a.run == "B" else []
    # The manifest name carries --only so two CONCURRENT instances cannot
    # clobber each other's record. The probe files are already per leg; this
    # was the one shared artifact, and a lost manifest would lose the recorded
    # divergence indexes that the required continuations are built from.
    slug = "" if not a.only else "-" + re.sub(r"[^A-Za-z0-9]+", "_", a.only)
    manifest_path = outdir / f"manifest-{a.run or 'cont'}{slug}.json"

    try:
        if a.continuations:
            legs, tokens, head = _continuation_legs(outdir, log)
        else:
            tokens = {pid: tokenize(pid) for pid in prompts}
            head = {pid: banked_head_ids(pid) for pid in prompts}
            legs = make_legs(prompts, tokens, a.run)
            if a.only:
                legs = filter_legs(legs, a.only)
                prompts = [p for p in prompts if p in {l["prompt"] for l in legs}]
            for pid in prompts:
                log(f"  {pid}: {len(tokens[pid])} prompt tokens, "
                    f"{len(head[pid])} banked HEAD ids")
    except GridError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 2

    rate = a.rate_s_per_pos
    total_pos = sum(l["positions"] for l in legs)
    log(f"legs       : {len(legs)}, {total_pos} forward positions")
    log(f"projection : {project_cost_s(total_pos, rate) / 3600:.2f} h at the "
        f"{rate:.2f} s/pos prior")

    if a.dry_run:
        log("dry run -- no model invoked")
        for l in legs:
            log(f"  {l['leg']:16s} {l['kind']:5s} {l['positions']:4d} pos  "
                f"~{project_cost_s(l['positions'], rate) / 60:.1f} min")
        return 0

    t_start = time.time()
    observations = []
    results = {}
    bare_ids = None

    for leg in legs:
        pid = leg["prompt"]
        out_json = outdir / f"{leg['leg']}.json"
        elapsed = time.time() - t_start
        icept, rate = fit_cost_model(observations, a.rate_s_per_pos)
        est = project_cost_s(leg["positions"], rate, icept) * WINDOW_SAFETY

        # RESUME. `leg["ids"]` is the invocation's whole input -- the prompt for
        # a probe leg, prompt + forced banked ids for a continuation -- so it is
        # the right expected n_input for both. Passing data["n_input"] here
        # would compare the field to itself: a check that cannot fail, which is
        # the shape this tree has been caught by before.
        expect_n_input = expected_n_input(leg)
        if leg["kind"] == "bare" and (outdir / f"{leg['leg']}.ids").exists():
            txt = (outdir / f"{leg['leg']}.ids").read_text(encoding="utf-8").strip()
            if txt:
                bare_ids = [int(s) for s in txt.split(",")]
                log(f"  {leg['leg']}: already banked ({len(bare_ids)} ids) -- skipping")
                continue
        if leg["kind"] != "bare" and out_json.exists():
            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
                errs = validate_probe_file(data, expect_n_input)
                if not errs:
                    log(f"  {leg['leg']}: already banked, revalidated -- skipping")
                    results[leg["leg"]] = _summarise(leg, data, head.get(pid))
                    continue
                log(f"  {leg['leg']}: banked file failed revalidation {errs}; re-running")
            except (OSError, ValueError) as e:
                log(f"  {leg['leg']}: banked file unreadable ({e}); re-running")

        if elapsed + est > a.budget_s:
            if est > a.budget_s:
                # Distinct from "no room left in THIS window": a leg that does
                # not fit an EMPTY window is a cost finding about the ruling's
                # own estimate, and it must be recorded and decided rather than
                # retried forever by a driver that will refuse it every time.
                log(f"  LEG DOES NOT FIT A FRESH WINDOW: {leg['leg']} projects "
                    f"{est / 3600:.2f} h against a {a.budget_s / 3600:.2f} h "
                    f"authorization (fit: {icept:.0f}s + {rate:.2f}s/pos over "
                    f"{leg['positions']} positions). Section 5 estimated this "
                    f"leg at ~2.8 h. This is a COST FINDING -- record it and "
                    f"decide the window split; do not widen the budget silently.")
            else:
                log(f"  STOPPING AT A LEG BOUNDARY: {leg['leg']} projects "
                    f"{est / 60:.1f} min, elapsed {elapsed / 60:.1f} min, budget "
                    f"{a.budget_s / 60:.1f} min. Every completed leg is banked; "
                    f"re-run this command in a new window to continue.")
            break

        log(f"  {leg['leg']}: {leg['positions']} pos, ~{est / 60:.1f} min projected")
        try:
            dt, ids, stderr_text = run_leg(leg, outdir, log)
        except GridError as e:
            log(f"  {leg['leg']}: FAILED -- {e}")
            _write_manifest(manifest_path, a, legs, results, observations, log)
            return 1
        observations.append((dt, leg["positions"]))
        log(f"  {leg['leg']}: done in {dt / 60:.1f} min "
            f"({dt / leg['positions']:.2f} s/pos), {len(ids)} ids")

        if leg["kind"] == "bare":
            bare_ids = ids
            continue

        data = json.loads(out_json.read_text(encoding="utf-8"))
        errs = validate_probe_file(data, expect_n_input)
        if errs:
            log(f"  {leg['leg']}: PROBE FILE INVALID -- {errs}")
            _write_manifest(manifest_path, a, legs, results, observations, log)
            return 1

        if leg.get("probe_pos") is not None:
            if bare_ids is None:
                log("  section 8A: no bare run in this window -- cannot discharge")
            else:
                inv = check_invariance(bare_ids, data, leg["probe_pos"], stderr_text)
                (outdir / "section8A-invariance.json").write_text(
                    json.dumps(inv, indent=1), encoding="utf-8")
                for c in inv["checks"]:
                    log(f"    8A {c['name']}: {'PASS' if c['ok'] else 'FAIL'} {c['detail']}")
                if not inv["ok"]:
                    log("  SECTION 8A DISCHARGE FAILED -- STOPPING THE LEG. "
                        "Ruling section 5: any mismatch stops the leg, tooling "
                        "defect memo before any use of this grid.")
                    _write_manifest(manifest_path, a, legs, results, observations, log)
                    return 3
                log("  section 8A OWED item DISCHARGED (stream invariance + index agreement)")

        results[leg["leg"]] = _summarise(leg, data, head.get(pid))
        r = results[leg["leg"]]
        log(f"  {leg['leg']}: vs banked HEAD -> {r['divergence']['kind']}"
            f"{'' if r['divergence']['index'] is None else ' at index ' + str(r['divergence']['index'])}"
            f"{'' if r['divergence']['index'] is None else ' (abs ' + str(r['divergence']['abs_position']) + ')'}")

    _write_manifest(manifest_path, a, legs, results, observations, log)
    log(f"=== window done in {(time.time() - t_start) / 3600:.2f} h ===")
    return 0


def _summarise(leg, data, head_ids):
    ref_ids = data["ids"]
    rec = {"leg": leg["leg"], "prompt": leg["prompt"], "kind": leg["kind"],
           "n_input": data["n_input"], "n_generated": len(ref_ids),
           "positions": leg["positions"], "pair": PAIR}
    if leg["kind"] == "cont":
        rec["forced_from"] = leg["forced_from"]
        rec["divergence"] = {"kind": "n/a-continuation", "index": None,
                             "abs_position": None}
        return rec
    if head_ids is None:
        rec["divergence"] = {"kind": "no-banked-stream", "index": None,
                             "abs_position": None}
        return rec
    kind, j = first_divergence(ref_ids, head_ids)
    rec["divergence"] = {
        "kind": kind, "index": j,
        "abs_position": None if j is None else data["n_input"] + j,
        "ref_chose": None if j is None or j >= len(ref_ids) else ref_ids[j],
        "head_chose": None if j is None or j >= len(head_ids) else head_ids[j],
    }
    return rec


def _write_manifest(path, args, legs, results, observations, log):
    rate = observed_rate(observations, args.rate_s_per_pos)
    doc = {
        "pair": PAIR,
        "model": MODEL.name,
        "model_bytes": MODEL_SIZE_EXPECTED,
        "stoke_hash_frozen": STOKE_HASH_FROZEN,
        "n_predict": N_PREDICT,
        "run": args.run or "continuations",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "legs_planned": [{k: v for k, v in l.items() if k != "ids"} for l in legs],
        "legs_completed": results,
        "observed_rate_s_per_pos": round(rate, 4),
        "observations": [{"secs": round(s, 2), "positions": p} for s, p in observations],
    }
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    log(f"manifest   : {path}")


def _continuation_legs(outdir, log):
    """Build the REQUIRED continuations from manifests already in `outdir`."""
    legs, tokens, head = [], {}, {}
    for mpath in sorted(outdir.glob("manifest-*.json")):
        doc = json.loads(mpath.read_text(encoding="utf-8"))
        for leg_name, rec in doc.get("legs_completed", {}).items():
            d = rec.get("divergence", {})
            if d.get("kind") != "diverge":
                continue
            pid = rec["prompt"]
            j = int(d["index"])
            if pid not in tokens:
                tokens[pid] = tokenize(pid)
                head[pid] = banked_head_ids(pid)
            leg = continuation_leg(pid, tokens[pid], head[pid], j)
            if leg is None:
                log(f"  {pid}: divergence at index {j} == last generated position; "
                    f"no continuation exists (section 5)")
                continue
            legs.append(leg)
    if not legs:
        log("  no recorded divergences need a continuation")
    return legs, tokens, head


# ============================================================================
# selftest
# ============================================================================

def run_selftest() -> int:
    fails = []

    def check(label, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            fails.append(label)

    # ---- leg cost arithmetic ----
    check("a leg costs prompt_len + n_predict positions",
          leg_positions(23) == 87 and leg_positions(477) == 541)
    check("RUN A's published position total is 585",
          sum(leg_positions(n) for n in (23, 23, 24, 64, 53, 14)) == 585)
    check("RUN B's published position total is 541",
          leg_positions(477) == 541)

    # ---- the one-point fallback is conservative in the stated direction ----
    # true cost 23.8 + 18.42*P observed at P=87; projecting P=541 must come out
    # HIGH, or the window bound leans the wrong way.
    true87 = project_cost_s(87, 18.42)
    r = observed_rate([(true87, 87)], 1.0)
    check("origin fit overestimates a longer leg (window bound leans safe)",
          project_cost_s(541, r) > project_cost_s(541, 18.42))
    check("no observations falls back to the prior",
          observed_rate([], 7.5) == 7.5)
    check("zero positions cannot divide by zero",
          observed_rate([(10.0, 0)], 3.0) == 3.0)

    # ---- the two-point fit separates the embedding-cache build ----
    # Exact data from a known line: a large intercept plus a modest slope, the
    # shape a tied-output model actually has.
    A, B = 400.0, 12.0
    obs = [(A + B * 16, 16), (A + B * 87, 87)]
    ia, ib = fit_cost_model(obs, 99.0)
    check("a two-point fit recovers the intercept and slope",
          abs(ia - A) < 1e-6 and abs(ib - B) < 1e-9)
    check("the fit projects a long leg far better than the origin rate",
          abs(project_cost_s(541, ib, ia) - (A + B * 541)) <
          abs(project_cost_s(541, observed_rate(obs, 99.0)) - (A + B * 541)))
    check("the origin rate would have overstated p41 by more than an hour",
          project_cost_s(541, observed_rate(obs, 99.0)) - (A + B * 541) > 3600)
    check("one distinct size cannot be fitted, so the origin rate stands",
          fit_cost_model([(A + B * 16, 16)], 99.0)[0] == 0.0)
    check("no observations falls back to both priors",
          fit_cost_model([], 7.5, 21.0) == (21.0, 7.5))
    check("a negative slope is refused rather than projected from",
          fit_cost_model([(100.0, 16), (50.0, 87)], 7.5)[0] == 0.0)
    check("a negative intercept is refused rather than projected from",
          fit_cost_model([(160.0, 16), (900.0, 87)], 7.5)[0] == 0.0)
    check("the window safety factor is small enough not to refuse fitting legs",
          1.0 < WINDOW_SAFETY <= 1.25)

    # ---- divergence location ----
    check("identical streams agree", first_divergence([1, 2, 3], [1, 2, 3]) == ("agree", None))
    check("first difference is found, not the last",
          first_divergence([1, 9, 9], [1, 2, 3]) == ("diverge", 1))
    check("index 0 divergence is found", first_divergence([5], [6]) == ("diverge", 0))
    check("a short-but-agreeing stream is 'length', never 'diverge'",
          first_divergence([1, 2], [1, 2, 3]) == ("length", 2))
    check("a divergence inside a short stream still reports 'diverge'",
          first_divergence([1, 7], [1, 2, 3]) == ("diverge", 1))

    # ---- continuation construction ----
    toks, headids = [100, 101], list(range(200, 264))
    leg = continuation_leg("pXX", toks, headids, 5)
    check("continuation forces prompt + banked ids[0..j] INCLUSIVE",
          leg["ids"] == [100, 101] + list(range(200, 206)))
    check("continuation generates the remaining 64-(j+1)", leg["n"] == 58)
    check("continuation re-prefills, so it costs prompt_len+64 positions",
          leg["positions"] == leg_positions(2))
    check("a divergence at the last index needs no continuation",
          continuation_leg("pXX", toks, headids, 63) is None)

    # ---- expected n_input is derived from the leg, never read back ----
    check("a probe leg expects n_input == the prompt length",
          expected_n_input({"ids": [1, 2, 3], "kind": "probe"}) == 3)
    check("a continuation expects n_input == prompt + forced ids",
          expected_n_input(continuation_leg("pXX", [100, 101], list(range(200, 264)), 5)) == 8)
    check("the expected n_input is independent of the artifact "
          "(a file claiming any n_input is still checked)",
          validate_probe_file({"ids": [7], "n_input": 999,
                               "probes": [{"index": 0, "top5": [[7, 5.0], [8, 4.0],
                                                                [9, 3.0], [10, 2.0],
                                                                [11, 1.0]]}]}, 2) != [])

    # ---- probe file validation ----
    def mk(ids, n_input=2, mangle=None):
        probes = [{"index": i, "top5": [[t, 10.0 - k] for k, t in
                                        enumerate([tid, 900, 901, 902, 903])]}
                  for i, tid in enumerate(ids)]
        d = {"ids": list(ids), "n_input": n_input, "probes": probes}
        if mangle:
            mangle(d)
        return d

    check("a well-formed probe file validates clean", validate_probe_file(mk([7, 8]), 2) == [])
    check("rank-0 disagreeing with the stream is caught",
          any("rank-0" in e for e in
              validate_probe_file(mk([7, 8], mangle=lambda d:
                                     d["probes"][1]["top5"].insert(0, [55, 99.0]) or
                                     d["probes"][1]["top5"].pop()), 2)))
    check("a wrong n_input is caught", any("n_input" in e for e in
                                           validate_probe_file(mk([7, 8], n_input=9), 2)))
    check("a missing probe is caught",
          validate_probe_file(mk([7, 8], mangle=lambda d: d["probes"].pop()), 2) != [])
    check("out-of-order top5 logits are caught",
          any("descending" in e for e in
              validate_probe_file(mk([7, 8], mangle=lambda d:
                                     d["probes"][0].__setitem__(
                                         "top5", [[7, 1.0], [900, 2.0], [901, 0.0],
                                                  [902, -1.0], [903, -2.0]])), 2)))
    check("a truncated candidate list is caught",
          any("expected 5" in e for e in
              validate_probe_file(mk([7], mangle=lambda d:
                                     d["probes"][0]["top5"].pop()), 2)))

    # ---- section 8A ----
    top5 = [[11, 1.5], [12, 1.25], [13, 1.0], [14, 0.5], [15, 0.25]]
    data = {"ids": [11, 11], "n_input": 2,
            "probes": [{"index": 0, "top5": top5}, {"index": 1, "top5": top5}]}
    err = "\n".join(probe_pos_line(1, t, l) for t, l in top5)
    inv = check_invariance([11, 11], data, 1, err)
    check("8A passes when the stream is unchanged and the records agree", inv["ok"])
    check("8A fails when --probe-all changed the id stream",
          not check_invariance([11, 12], data, 1, err)["ok"])
    bad = "\n".join(probe_pos_line(1, t + 1, l) for t, l in top5)
    check("8A fails when probe-pos reports different ids",
          not check_invariance([11, 11], data, 1, bad)["ok"])
    shifted = "\n".join(probe_pos_line(1, t, l + 0.001) for t, l in top5)
    check("8A fails on a logit difference at the printed precision",
          not check_invariance([11, 11], data, 1, shifted)["ok"])
    check("8A fails when fewer than five lines are printed",
          not check_invariance([11, 11], data, 1,
                               "\n".join(err.splitlines()[:4]))["ok"])
    check("8A fails when the requested index is absent from probe-all",
          not check_invariance([11, 11], data, 5, err)["ok"])

    # a logit difference BELOW the printed precision is invisible, and the
    # docstring says so rather than the check pretending otherwise
    tiny = "\n".join(probe_pos_line(1, t, l + 1e-9) for t, l in top5)
    check("8A's precision limit is real and stated (1e-9 is invisible at %.6f)",
          check_invariance([11, 11], data, 1, tiny)["ok"])

    # ---- probe line parsing ----
    check("a malformed probe line raises rather than being skipped",
          _raises(lambda: parse_probe_pos_stderr("probe pos=1: id=7 logit=nope")))
    check("non-probe stderr noise is ignored",
          parse_probe_pos_stderr("loading model\n" + probe_pos_line(3, 7, -1.5)) ==
          [(3, 7, "-1.500000")])

    # ---- leg ordering: the 8A gate is first ----
    legs = make_legs(RUN_A_PROMPTS, {p: [0] * n for p, n in
                                     zip(RUN_A_PROMPTS, (23, 24, 64, 53, 14))}, "A")
    check("RUN A opens with the p00 bare/flag pair, bare first",
          [l["leg"] for l in legs[:2]] == ["p00-bare", "p00-probe"])
    check("RUN A has six legs for five prompts", len(legs) == 6)
    check("only the p00 probe leg carries --probe-pos",
          [l["leg"] for l in legs if l.get("probe_pos") is not None] == ["p00-probe"])
    check("the probe-pos index is the last generated position",
          legs[1]["probe_pos"] == N_PREDICT - 1)

    # ---- the leg filter that makes concurrent subsets possible ----
    allA = make_legs(RUN_A_PROMPTS, {p: [0] * n for p, n in
                                     zip(RUN_A_PROMPTS, (23, 24, 64, 53, 14))}, "A")
    check("--only selects by prompt id",
          [l["leg"] for l in filter_legs(allA, "p20,p30")] == ["p20-probe", "p30-probe"])
    check("--only selects by full leg name",
          [l["leg"] for l in filter_legs(allA, "p00-bare,p00-probe")] ==
          ["p00-bare", "p00-probe"])
    check("--only p00 keeps the 8A pair together",
          [l["leg"] for l in filter_legs(allA, "p00")] == ["p00-bare", "p00-probe"])
    check("selecting p00-probe WITHOUT p00-bare is refused (8A dependency)",
          _raises(lambda: filter_legs(allA, "p00-probe")))
    check("an unknown --only name is refused, not silently ignored",
          _raises(lambda: filter_legs(allA, "p20,p99")))
    check("a filter selecting nothing is refused",
          _raises(lambda: filter_legs(allA, "")))
    check("disjoint subsets partition the run exactly",
          sorted(l["leg"] for l in filter_legs(allA, "p00,p10")) +
          sorted(l["leg"] for l in filter_legs(allA, "p20,p30,p45")) ==
          sorted(l["leg"] for l in allA))

    # ---- the additional-instance discount is conditional on a VERIFIED fact ----
    # Decidable on demand, in BOTH directions, regardless of what is running.
    check("--additional-instance is refused when nothing is mapping the model",
          _raises(lambda: ram_requirement(MODEL_SIZE_EXPECTED, True, [])))
    check("--additional-instance charges only the private footprint when one is",
          ram_requirement(MODEL_SIZE_EXPECTED, True, [4321]) ==
          PRIVATE_PER_INSTANCE_GB + 0.5)
    check("the full bar still applies without the flag, whatever is running",
          ram_requirement(MODEL_SIZE_EXPECTED, False, [4321]) ==
          MODEL_SIZE_EXPECTED / 1024.0 ** 3 + RAM_MARGIN_GB)
    check("the discounted bar is strictly below the full one",
          ram_requirement(MODEL_SIZE_EXPECTED, True, [1]) <
          ram_requirement(MODEL_SIZE_EXPECTED, False, []))
    check("the private-footprint figure is below a full instance's working set",
          0 < PRIVATE_PER_INSTANCE_GB < MODEL_SIZE_EXPECTED / 1024.0 ** 3 + RAM_MARGIN_GB)

    # ---- constants that must not drift from the ruling ----
    check("the pair key is per-file and not Q8_0's", PAIR == "f32_intcore_q4_k_m")
    check("the window budget is the ruling's 4 h", WINDOW_BUDGET_S == 4 * 3600)
    check("the RAM margin is >= the ruling's model + 2 GB (tightening only)",
          RAM_MARGIN_GB >= 2.0)
    check("the RAM margin covers the measured 4.62 GB Q4_K_M footprint",
          MODEL_SIZE_EXPECTED / 1024.0 ** 3 + RAM_MARGIN_GB >= 4.62)
    check("the pinned model size is section 5's launch-condition operand",
          MODEL_SIZE_EXPECTED == 2497280736)

    print(f"\n{len(fails)} failed" if fails else "\nALL PASS")
    return 1 if fails else 0


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
