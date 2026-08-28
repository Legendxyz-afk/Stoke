#!/usr/bin/env python3
"""bench/append_history.py -- append one row to bench/history.jsonl.

WHY THIS EXISTS. `bench/history.jsonl` is a standing requirement, not an M3
convenience: the Phase-1 spec ("Perf regression tracking") and the M3-M7
forward note ("history.jsonl discipline") both say every green commit touching
a hot path appends a row, and that regressions are caught by DIFFING HISTORY
rather than by remembering. A file written at that cadence has to be produced
mechanically -- a format a human retypes by hand drifts within a week.

TWO DESIGN DECISIONS, both forced by what M2 and K1 measured rather than by
taste. They are the whole reason this is a tool and not an echo command.

1. THE ROW STORES ITS OWN REFERENCE STREAM -- BUT THE STABILITY GUARD JUDGES
   ABSOLUTES, AND THAT ORDERING IS THE OPPOSITE OF WHAT WAS FIRST ASSUMED.
   Storing the reference stream beside every absolute figure is what lets two
   rows taken weeks apart be RE-NORMALIZED against each other, and %ref really
   is the right lens for comparing a LOADED run to a QUIESCED one, where the
   whole box is uniformly slower and numerator and denominator move together.

   It is NOT a run-to-run noise reducer, and measurement on 2026-08-02 said so
   plainly: over five runs, %ref was LESS stable than the absolute median on
   8 of 9 measurable ops (ffn_gate 33.3% vs 22.8%, attn_q 27.2% vs 20.7%).
   The reason is structural -- the reference stream is measured ONCE, at the
   start of the run, while each op is measured later, so the ratio only
   cancels box state that persists across the entire run. Against run-to-run
   jitter the denominator is simply a second noisy measurement, and dividing
   by it ADDS its variance to every op instead of removing it. So the guard
   below is applied to absolute medians on purpose.

2. A ROW IS A MEDIAN ACROSS RUNS, NOT WITHIN ONE RUN, AND STABILITY IS
   ENFORCED RATHER THAN HOPED FOR. A single bench run cannot self-diagnose
   sustained interference: on 2026-08-02 a run reported LOGITS at 463.8 ms
   (2.6x wrong) while its own max/med ratio was 1.33, which the table's footer
   reads as HEALTHY. The spread heuristic inside one run catches TRANSIENT
   interference and is blind to SUSTAINED interference -- and sustained is the
   kind that moves a median. So the M0 protocol's "median of 3" is implemented
   here as three separate RUNS whose medians must agree, and disagreement is a
   REFUSAL to write: a box that cannot measure consistently right now must not
   silently deposit a number that a future regression hunt will trust.

3. THE BOX SIDECAR, AND WHY IT IS SAMPLED HERE RATHER THAN IN bench.c
   (2026-08-03, Fable K3 ruling S4). Decision 2's guard compares three runs to
   EACH OTHER, so a SUSTAINED co-load makes all three agree and be equally
   wrong -- which is exactly what happened: a row taken 2026-08-03 passed the
   guard at 13.8% spread while its matvecs implied 0.987 tok/s against the same
   binary's 1.644 the same morning, and it had to be deleted unrecorded. The
   cause was found only afterwards, by forensics: a game had been resident
   since 09:57 that day, averaging 5.08 cores, spanning every contested
   measurement window of two sessions. Nobody had recorded the process table.

   The guard could not have caught it and no amount of re-running would have.
   What closes it is recording the ONE fact nobody wrote down, mechanically,
   in the row itself. So each run is BRACKETED by a box sample and the row
   carries `box.sidecar` plus a mechanical `box.grade`:

     "gate"    -- quiesced by measurement: foreign CPU below 5% of the box AND
                  free RAM above the model + 2 GB. Comparison-grade.
     "ambient" -- anything else. A real number about a loaded box; NOT a diff
                  base, NOT a referent, NOT quotable against another row.

   bench.c deliberately emits no box state (see its emit_jsonl comment: a
   process that guesses its own provenance writes confident falsehoods). That
   stands. This layer is not guessing -- it is the PARENT of the bench run and
   samples the window from outside, which is the one place the co-load during
   the timed region is observable at all.

   `foreign_cpu_cores` is a FLOOR: it sums per-process CPU-time deltas and
   silently undercounts processes this user cannot open. A floor is the right
   direction for a guard -- it can fail to raise an alarm, but it cannot
   invent one, so "gate" is never granted on an overestimate of quiet.

WHAT THIS TOOL DELIBERATELY DOES NOT DO. It does not derive tok/s. That needs
the model's layer count and turns a recording tool into one that must also
know architecture; the arithmetic belongs in the ledger where its assumptions
are visible. It also does not hash the multi-GB model file -- name plus byte
size distinguishes every model this project uses, and a 20-second hash on
every append is a cost with no reader.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY = os.path.join(REPO, "bench", "history.jsonl")

# SCHEMA 2 (2026-08-03) -- rows at schema 1 ARE NOT COMPARABLE TO THESE, and
# the version bump is the marker that says so rather than a note someone has
# to remember. Two things changed at once:
#
#   1. The build moved -O1 -> -O3. Every matvec got ~3.1x faster with
#      byte-identical output, so a schema-1 median and a schema-2 median
#      differ by the build before they differ by anything about the code.
#      `build.cflags` now records the flags, read back from the binary
#      itself (bench.c's STOKE_BUILD_CFLAGS), never from this process's
#      environment -- the appender's environment is not the binary's.
#   2. `ref_gbs` changed meaning. Through schema 1 it was a serial f32
#      reduction measuring FP-add latency (~7 GB/s), roughly 2.15x below
#      this box's real read bandwidth; every schema-1 pct_ref is therefore
#      against a denominator that low. From schema 2 it is
#      max(split8_f32, xor_u64), an honest read-rate lower bound.
#
# The two schema-1 rows stay in the file UNEDITED. Recorded evidence is
# append-only: a row is what was measured on the day it was measured, and
# rewriting history to make it comparable is how a history file stops being
# evidence. `ref_gbs_serial` is carried in every schema-2 row precisely so
# the old rows can be CONVERTED when a comparison is genuinely needed.
SCHEMA = 2

# Ops below this median cost sit on the timer's resolution floor (the ~1 us
# primitives), where a 10% run-to-run difference is the clock, not the box.
# Stability is therefore judged only on ops large enough to be measured --
# which is also exactly the set the attribution ladder reads.
STABILITY_MIN_US = 100.0

# Cross-run spread above this means the box is not fit to measure right now.
# Calibrated against real data: the post-fix LOGITS spread was 8%, the
# pre-fix one 162%. 20% admits the former and refuses the latter.
STABILITY_MAX_REL = 0.20

# ---- box sidecar thresholds (header decision 3) ----
# "all-LP busy < 5%" from the ruling's quiesced definition, expressed as cores
# so it reads the same on any core count: 5% of a 16-LP box is 0.8 cores.
QUIESCED_MAX_FOREIGN_FRAC = 0.05
# Headroom above the model so a mmap'd file can stay resident. The 2026-08-03
# sessions ran a 4.28 GB model with ~5 GB free and spent a day arguing about
# whether the shortfall was paging; it was not, but the margin is cheap.
QUIESCED_MIN_FREE_HEADROOM_MB = 2048
# A process holding at least this many cores is named in the sidecar, so the
# culprit is in the row rather than in someone's memory of the day.
SIDECAR_NAME_PROC_CORES = 0.05


# ---------------------------------------------------------------- pure logic


def cross_run(values):
    """Median/min/max of one op's per-run medians. Pure; see --selftest."""
    if not values:
        raise ValueError("cross_run: no values")
    s = sorted(values)
    return {
        "med": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "runs": list(values),
    }


def rel_spread(values):
    """(max-min)/min, the run-to-run disagreement. 0.0 for a single run.

    Relative to MIN rather than to the median: the question being asked is
    "how much slower did the worst run look than the best", and anchoring on
    the fastest observation is the conservative reading of interference.
    """
    if len(values) < 2:
        return 0.0
    lo, hi = min(values), max(values)
    if lo <= 0:
        return float("inf")
    return (hi - lo) / lo


def stability_violations(per_op, min_us=STABILITY_MIN_US,
                         max_rel=STABILITY_MAX_REL):
    """Ops whose cross-run medians disagree too much to be worth recording.

    Returns a list of (op, spread) sorted worst-first. Empty means the box
    measured consistently. Ops cheaper than `min_us` are exempt, with the
    reason given at STABILITY_MIN_US.
    """
    out = []
    for op, values in per_op.items():
        if min(values) < min_us:
            continue
        s = rel_spread(values)
        if s > max_rel:
            out.append((op, s))
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out


def sidecar_delta(before, after):
    """Derive one run's box sidecar from the two samples bracketing it. Pure.

    `before`/`after` are box_sample() dicts. Returns None when either sample
    is missing -- an absent sidecar is reported as absent, never as quiet,
    because "no evidence of load" and "evidence of no load" are the exact
    pair this whole mechanism exists to stop conflating.

    CPU is attributed by PID across the two samples. A process alive in both
    contributes its delta; one that appears only in `after` started during the
    window and contributes all of it. The bench child itself is excluded for
    free: it is spawned and reaped strictly between the samples, so its PID is
    in neither. A process that EXITS during the window loses its share, which
    is part of why the total is documented as a floor.
    """
    if not before or not after:
        return None
    wall_s = (after["t_ms"] - before["t_ms"]) / 1000.0
    if wall_s <= 0:
        return None

    prev = {p["pid"]: p["cpu_s"] for p in before["procs"]}
    per_proc = []
    for p in after["procs"]:
        if p["pid"] in prev:
            used = p["cpu_s"] - prev[p["pid"]]
        elif p.get("start_ms") is not None and p["start_ms"] >= before["t_ms"]:
            # Started inside the window, so all of its CPU belongs to the
            # window. Verified by START TIME rather than by absence from the
            # first sample: a process can also be absent there because it was
            # unreadable at that moment, and charging its whole lifetime's CPU
            # to this window would turn the total into an over-estimate.
            used = p["cpu_s"]
        else:
            continue
        # Negative means PID reuse across the window -- a different process
        # wearing a recycled number. Drop it rather than subtract nonsense.
        if used > 0:
            per_proc.append((p["name"], used / wall_s))

    per_proc.sort(key=lambda nc: nc[1], reverse=True)
    total = sum(cores for _, cores in per_proc)
    named = [{"name": n, "cores": round(c, 3)}
             for n, c in per_proc if c >= SIDECAR_NAME_PROC_CORES]

    free_before = before.get("free_ram_mb")
    free_after = after.get("free_ram_mb")
    frees = [f for f in (free_before, free_after) if f is not None]

    return {
        "wall_s": round(wall_s, 2),
        # A FLOOR (header decision 3): inaccessible processes are invisible.
        "foreign_cpu_cores": round(total, 3),
        # Non-null means this sampler could not observe the load that matters
        # here; box_grade refuses "gate" on it. Carried from the sample rather
        # than re-derived so the reason travels into the row.
        "blind": after.get("blind") or before.get("blind"),
        "top_foreign": named[0] if named else None,
        "busy_procs": named[:5],
        # The MINIMUM across the window, not an endpoint: the question a
        # future reader asks is whether the model could stay resident
        # throughout, and the worst moment is what answers it.
        "free_ram_mb_min": min(frees) if frees else None,
        "power_plan": after.get("power_plan"),
        "on_ac": after.get("on_ac"),
        # A plan or power-source change DURING the run means the run spanned
        # two machine configurations, so its numbers describe neither.
        "conditions_changed": (
            before.get("power_plan") != after.get("power_plan")
            or before.get("on_ac") != after.get("on_ac")),
    }


def box_grade(sidecars, model_size_bytes, n_logical,
              max_frac=QUIESCED_MAX_FOREIGN_FRAC,
              min_headroom_mb=QUIESCED_MIN_FREE_HEADROOM_MB):
    """Mechanical comparison-grade for a row. Pure. Returns (grade, reasons).

    "gate" is granted only on POSITIVE evidence of quiet from every run in the
    row. Missing sidecars, an unknown core count, or one loaded run out of
    three all yield "ambient" -- the grade is a claim about the whole row, and
    a row is only as comparable as its worst run.
    """
    reasons = []
    if not sidecars or any(s is None for s in sidecars):
        return "ambient", ["no box sidecar -- box state during the run is unrecorded"]
    if not n_logical:
        return "ambient", ["logical core count unknown -- cannot judge foreign load"]

    for s in sidecars:
        if s.get("blind"):
            reasons.append(f"box sampler is blind: {s['blind']}")
            break
    if any(s.get("conditions_changed") for s in sidecars):
        reasons.append("power plan or power source CHANGED mid-run -- the run "
                       "spans two machine configurations")
    if any(s.get("on_ac") is False for s in sidecars):
        reasons.append("running on battery")

    max_cores = max_frac * n_logical
    worst = max(s["foreign_cpu_cores"] for s in sidecars)
    if worst > max_cores:
        top = max((s["top_foreign"] for s in sidecars if s["top_foreign"]),
                  key=lambda t: t["cores"], default=None)
        who = f" (top: {top['name']} at {top['cores']} cores)" if top else ""
        reasons.append(
            f"foreign CPU {worst:.2f} cores exceeds {max_cores:.2f} "
            f"({max_frac * 100:.0f}% of {n_logical} LPs){who}")

    need_mb = model_size_bytes / (1024.0 * 1024.0) + min_headroom_mb
    frees = [s["free_ram_mb_min"] for s in sidecars
             if s["free_ram_mb_min"] is not None]
    if not frees:
        reasons.append("free RAM unrecorded")
    elif min(frees) < need_mb:
        reasons.append(
            f"free RAM fell to {min(frees)} MB, under the "
            f"{need_mb:.0f} MB the model + {min_headroom_mb} MB headroom needs")

    return ("ambient" if reasons else "gate"), reasons


# ------------------------------------------------------------- provenance


def _run(cmd, cwd=REPO):
    """Best-effort command capture. Provenance must never fail the append."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def git_provenance():
    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    tag = _run(["git", "describe", "--tags", "--exact-match", "HEAD"])
    return {
        "commit": commit,
        # A row recorded from a dirty tree is not reproducible from the commit
        # it names, so the flag is part of the row rather than a warning that
        # scrolls away.
        "dirty": bool(dirty) if dirty is not None else None,
        "tag": tag,
    }


def sha256_of(path, limit=None):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and fh.tell() >= limit:
                break
    return h.hexdigest()


def free_ram_mb():
    """Best-effort, None when unavailable. Box state explains an odd row."""
    try:
        if sys.platform == "win32":
            out = _run(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_OperatingSystem)"
                        ".FreePhysicalMemory"])
            return round(int(out) / 1024.0) if out else None
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024.0)
    except Exception:
        return None
    return None


# Cumulative per-process CPU seconds + free RAM + a clock, in one shot. The
# counters must be CUMULATIVE, not instantaneous: bracketing the run with two
# cumulative reads gives the average over exactly the timed window, whereas two
# instantaneous rate samples would only describe the two instants at its edges
# -- and a co-load that starts after the first sample would read as quiet.
_WIN_SAMPLE_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$procs = @(Get-Process | Where-Object { $_.CPU -gt 0 } | ForEach-Object {
    $st = $null
    if ($_.StartTime) { $st = [DateTimeOffset]::new($_.StartTime).ToUnixTimeMilliseconds() }
    @{ pid = $_.Id; name = $_.ProcessName; cpu_s = $_.CPU; start_ms = $st }
})
$free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory
$plan = ((powercfg /getactivescheme) -replace '.*\((.*)\).*', '$1').Trim()
$batt = (Get-CimInstance Win32_Battery).BatteryStatus
[pscustomobject]@{
    t_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    free_ram_mb = [math]::Round($free / 1024.0)
    power_plan = $plan
    on_ac = ($batt -eq $null -or $batt -ne 1)
    procs = $procs
} | ConvertTo-Json -Compress -Depth 4
"""


def box_sample():
    """Snapshot of cumulative box counters. None when unavailable.

    Never raises: a box that cannot be sampled produces an ungraded row, which
    is a worse row but still a true one. Failing the append over telemetry
    would trade a recorded measurement for no measurement at all.
    """
    try:
        if sys.platform == "win32":
            out = _run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command", _WIN_SAMPLE_PS])
            if not out:
                return None
            d = json.loads(out)
            procs = d.get("procs") or []
            # PowerShell's ConvertTo-Json emits a bare object, not an array,
            # when the pipeline yields exactly one item.
            if isinstance(procs, dict):
                procs = [procs]
            return {
                "t_ms": int(d["t_ms"]),
                "free_ram_mb": d.get("free_ram_mb"),
                # The power plan is part of the recorded conditions, not
                # trivia: it changed from "Performance" to "Turbo" between two
                # checks 30 minutes apart on 2026-08-03 without anyone
                # touching a benchmark. A row that does not name it invites
                # exactly the cross-condition comparison the grade exists to
                # prevent, with nothing visible to catch it.
                "power_plan": d.get("power_plan"),
                "on_ac": d.get("on_ac"),
                "procs": [{"pid": int(p["pid"]), "name": str(p["name"]),
                           "cpu_s": float(p["cpu_s"]),
                           "start_ms": (int(p["start_ms"])
                                        if p.get("start_ms") is not None
                                        else None)} for p in procs],
            }

        # Linux: /proc/stat is authoritative and exact, but it is aggregate
        # only. Per-process attribution would mean walking /proc/*/stat, and
        # the measurement discipline this serves (the referent interleave, the
        # K3-entry baseline) is a Windows-box activity. The aggregate is what
        # decides the grade; naming the culprit is a convenience.
        #
        # UNDER WSL THIS SAMPLER IS BLIND TO THE THING IT EXISTS TO CATCH.
        # /proc/stat inside WSL2 describes the utility VM, not the Windows
        # host, so a host running two games reads as an idle box -- measured
        # 2026-08-03: the host at 5.35 foreign cores, WSL's own view 0.01, and
        # it graded "gate". A sampler that cannot see the load must say so
        # rather than certify quiet, which is the same failure the grade was
        # built to prevent, one layer down.
        blind = None
        try:
            with open("/proc/version") as fh:
                if "microsoft" in fh.read().lower():
                    blind = ("running under WSL -- /proc/stat sees the VM, "
                             "not the Windows host, so host load is invisible")
        except Exception:
            pass

        with open("/proc/stat") as fh:
            f = fh.readline().split()
        vals = [int(v) for v in f[1:11]]
        idle = vals[3] + vals[4]           # idle + iowait
        ticks = os.sysconf("SC_CLK_TCK")
        return {
            "t_ms": int(time.time() * 1000),
            "free_ram_mb": free_ram_mb(),
            "blind": blind,
            # Synthesized as one pseudo-process so sidecar_delta's PID
            # arithmetic is identical on both platforms. Busy jiffies are
            # already the sum across all cores, which is what "cores" means.
            "procs": [{"pid": -1, "name": "(system busy)",
                       "cpu_s": (sum(vals) - idle) / float(ticks),
                       "start_ms": None}],
        }
    except Exception:
        return None


# ------------------------------------------------------------------ driving


def prefer_performance(pid):
    """Opt one process out of Windows EcoQoS. True/False on Windows, None off it.

    ▶▶ WHY A LEDGER TOOL TOUCHES POWER MANAGEMENT AT ALL. On 2026-08-28 the
    first M5 row came out reporting every one of 15 ops ~3x slower than the M3
    rows -- attn_q 1020.9 -> 3069.9 us, and rmsnorm 1.6 -> 7.9 on an op no
    milestone since M3 has touched. It was not a regression. `bench --ops` is
    the ONLY path in this project that does not get the 2026-08-17 EcoQoS fix:
    that fix lives in `thread_prefer_performance()` and is called from
    `worker_main` and from `pool_start` ("the caller runs chunks too"), and
    `bench_ops` never calls `pool_start` because the pool is deliberately inert
    for this measurement. So the one instrument whose whole job is catching
    regressions was the one thread Windows was free to park on an E-core.

    Measured, A/B, one binary, order ABBA, on the row that was withdrawn:

        op        as-launched   opt-out   recovery   the M3 row it is diffed against
        attn_q       3038.4 us    936.1     3.25x    1020.9
        rmsnorm         7.9 us      1.6     4.94x        1.6

    The opt-out does not merely improve the numbers, it lands them ON the M3
    values -- which is what says the code was never the problem.

    ⚠ AND THE STABILITY GUARD COULD NOT HAVE CAUGHT IT. Throttled runs are
    stable; they are consistently slow. `ref_gbs` was 21.5 against M3's 21.0,
    because the reference stream is memory-bound and a frequency cap barely
    touches it -- which also means %ref normalisation would have made this
    WORSE, dividing a throttled op by an unthrottled denominator. This is the
    concrete case the header's decision 1 argues about in the abstract.

    The state is NOT inherited by children (measured the same day: opting this
    interpreter out left the bench child throttled at 2943 us), so it has to be
    applied per spawned process, which is why this is here and not in a
    wrapper.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        class _PPT(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong),
                        ("ControlMask", ctypes.c_ulong),
                        ("StateMask", ctypes.c_ulong)]

        PROCESS_SET_INFORMATION = 0x0200
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
        PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
        ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        h = k32.OpenProcess(PROCESS_SET_INFORMATION, False, int(pid))
        if not h:
            return False
        try:
            s = _PPT(PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                     PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
                     0)  # bit CONTROLLED, state 0 == throttling OFF
            ok = k32.SetProcessInformation(h, ProcessPowerThrottling,
                                           ctypes.byref(s), ctypes.sizeof(s))
            return bool(ok)
        finally:
            k32.CloseHandle(h)
    except Exception:
        # Best-effort, exactly like pool.c's version: on a Windows without the
        # entry point the behaviour is what it was before.
        return False


def one_run(exe, model, extra_args):
    """Run the bench once; return (parsed final JSON line, box sidecar).

    The JSON is the LAST line by construction (bench.c emit_jsonl), so the
    human table above it is ignored rather than parsed -- no coupling to a
    format meant for eyes.

    The run is BRACKETED by box samples (header decision 3). The samples are
    taken as close to the subprocess as possible so the window they describe
    is the window that was timed; the sampling cost itself lands outside the
    bench's own timers either way.
    """
    before = box_sample()
    cmd = [exe, "bench", "--ops", model, "--jsonl"] + list(extra_args)
    # Popen, not subprocess.run: the EcoQoS opt-out has to be applied to a
    # process that already exists, and run() does not come back until it is
    # gone. The window between spawn and the call is covered by the 4.3 GB
    # model load, which happens before anything is timed.
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    qos = prefer_performance(p.pid)
    out, err = p.communicate()
    sidecar = sidecar_delta(before, box_sample())
    if qos is not None:
        sidecar["qos_optout"] = qos
    if p.returncode != 0:
        raise SystemExit(
            f"bench failed (rc={p.returncode}):\n{err.strip()[:2000]}")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("bench produced no output")
    try:
        return json.loads(lines[-1]), sidecar
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"bench's final line is not JSON ({e}). Was it built with the "
            f"--jsonl support? Line was:\n{lines[-1][:400]}")


def build_row(runs, label, milestone, exe, model, sidecars=None):
    per_op = {}
    per_op_meta = {}
    for r in runs:
        for op in r["ops"]:
            per_op.setdefault(op["op"], []).append(op["med_us"])
            per_op_meta[op["op"]] = op
    ref_runs = [r["ref_gbs"] for r in runs]

    # Every run in a row comes from ONE binary, so disagreeing flags mean the
    # exe was swapped mid-append and the runs are not a median of the same
    # thing. Refuse rather than record the first one and hope.
    cflags = {r.get("build_cflags") for r in runs}
    if len(cflags) != 1:
        raise SystemExit(
            f"runs disagree about build_cflags ({sorted(map(str, cflags))}) -- "
            "the binary changed mid-append; a row must be one build")
    build_cflags = cflags.pop()
    if build_cflags is None:
        raise SystemExit(
            "bench emitted no build_cflags -- this is a schema-1 binary. "
            "Rebuild before appending: a schema-2 row must record the flags "
            "it was measured with, or it is exactly the ambiguity the bump "
            "exists to remove.")

    ops_out = {}
    for op, values in per_op.items():
        meta = per_op_meta[op]
        c = cross_run(values)
        ops_out[op] = {
            "shape": meta["shape"],
            "bytes": meta["bytes"],
            "reps": meta["reps"],
            "med_us": round(c["med"], 3),
            "med_us_runs": [round(v, 3) for v in c["runs"]],
            "rel_spread": round(rel_spread(values), 4),
            # Recomputed against the CROSS-RUN reference median so this figure
            # and ref_gbs below are the same basis. Taking the last run's
            # pct_ref would silently mix bases.
            "pct_ref": None,
        }

    n_logical = os.cpu_count()
    grade, grade_reasons = box_grade(sidecars, os.path.getsize(model),
                                     n_logical)

    ref_med = statistics.median(ref_runs)
    for op, o in ops_out.items():
        gbs = (o["bytes"] / 1e9) / (o["med_us"] / 1e6) if o["med_us"] > 0 else 0.0
        o["gbs"] = round(gbs, 4)
        o["pct_ref"] = round(100.0 * gbs / ref_med, 3) if ref_med > 0 else None

    return {
        "schema": SCHEMA,
        "kind": "ops",
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "milestone": milestone,
        "label": label,
        "git": git_provenance(),
        "build": {
            "binary_sha256": sha256_of(exe),
            "binary_basename": os.path.basename(exe),
            # From the binary, not from this process. See the SCHEMA comment.
            "cflags": build_cflags,
        },
        # Model IDENTITY, not a path: models/ is gitignored and a committed
        # file must not point at something absent from the repo.
        "model": {
            "name": os.path.basename(model),
            "size_bytes": os.path.getsize(model),
        },
        # `grade` is MECHANICAL (header decision 3). It is the field that
        # decides whether this row may be compared to anything: "gate" rows
        # are quiesced by measurement, "ambient" rows describe a box that was
        # doing something else at the time and are not a diff base.
        "box": {
            "platform": sys.platform,
            "logical_cpus": n_logical,
            "free_ram_mb": free_ram_mb(),
            "grade": grade,
            "grade_reasons": grade_reasons,
            "sidecar": sidecars if sidecars else None,
        },
        "runs": len(runs),
        # THE DENOMINATOR: max(split8_f32, xor_u64) per run, then the
        # cross-run median. The variants are kept whole beside it so the
        # choice is re-derivable, and ref_gbs_serial is the conversion key
        # back to schema-1 rows (see the SCHEMA comment).
        "ref_gbs": round(ref_med, 4),
        "ref_gbs_runs": [round(v, 4) for v in ref_runs],
        "ref_rel_spread": round(rel_spread(ref_runs), 4),
        "ref_gbs_split8_runs": [round(r["ref_gbs_split8"], 4) for r in runs],
        "ref_gbs_xor_runs": [round(r["ref_gbs_xor"], 4) for r in runs],
        "ref_gbs_serial_runs": [round(r["ref_gbs_serial"], 4) for r in runs],
        "ops": ops_out,
    }


# ----------------------------------------------------------------- selftest


def selftest():
    checks, failed = 0, 0

    def ck(name, cond):
        nonlocal checks, failed
        checks += 1
        if not cond:
            failed += 1
            print(f"  FAIL {name}")
        else:
            print(f"  ok   {name}")

    c = cross_run([3.0, 1.0, 2.0])
    ck("cross_run median of 3", c["med"] == 2.0)
    ck("cross_run min/max", c["min"] == 1.0 and c["max"] == 3.0)
    ck("cross_run preserves run order", c["runs"] == [3.0, 1.0, 2.0])

    ck("rel_spread single run is 0", rel_spread([5.0]) == 0.0)
    ck("rel_spread equal runs is 0", rel_spread([5.0, 5.0, 5.0]) == 0.0)
    ck("rel_spread 8pct", abs(rel_spread([159.5, 162.8, 172.3]) - 0.0802) < 1e-3)

    # THE REAL DATA THAT MOTIVATED THE GUARD. Pre-fix LOGITS must be refused,
    # post-fix must be admitted -- if this check ever inverts, the guard has
    # stopped doing the one job it was calibrated for.
    pre = {"matvec LOGITS": [193.8e3, 463.8e3, 176.8e3]}
    post = {"matvec LOGITS": [162.8e3, 159.5e3, 172.3e3]}
    ck("guard REFUSES the pre-fix 2.6x swing", len(stability_violations(pre)) == 1)
    ck("guard ADMITS the post-fix 8pct spread", stability_violations(post) == [])

    # A MATCHED PAIR: identical 33% spread, opposite verdicts, and the ONLY
    # difference is op size. Both must be well ABOVE STABILITY_MAX_REL or the
    # pair proves nothing -- an earlier version of this check used 11% for
    # both, so the "exempt" case passed merely for being under the threshold
    # and would have kept passing with the exemption deleted.
    floor = {"vec_dot_q8_0_f32": [0.9, 1.2, 0.9]}      # ~33%, sub-timer-floor
    big = {"matvec ffn_up": [9000.0, 12000.0, 9000.0]}  # ~33%, real op
    ck("both halves of the pair exceed the threshold",
       rel_spread([0.9, 1.2, 0.9]) > STABILITY_MAX_REL
       and rel_spread([9000.0, 12000.0, 9000.0]) > STABILITY_MAX_REL)
    ck("timer-floor op exempt despite 33pct spread",
       stability_violations(floor) == [])
    ck("large op with the SAME 33pct spread is NOT exempt",
       len(stability_violations(big)) == 1)

    ck("violations sorted worst-first",
       [op for op, _ in stability_violations(
           {"a": [100.0, 130.0], "b": [100.0, 200.0]})] == ["b", "a"])

    # ---- schema 2: the build-flags guard ----
    # A row must be one build. These drive build_row directly rather than
    # inferring the behaviour from a full append, and they use this file as
    # a stand-in exe/model purely so sha256_of and getsize have something
    # real to read -- the flags logic is what is under test.
    def fake_run(cflags, ref=15.0):
        run = {
            "build_cflags": cflags,
            "ref_gbs": ref,
            "ref_gbs_split8": ref,
            "ref_gbs_xor": ref - 0.5,
            "ref_gbs_serial": 7.4,
            "ops": [{"op": "matvec ffn_up", "shape": "2560 x 9728",
                     "bytes": 26509312, "med_us": 4000.0, "reps": 95}],
        }
        if cflags is None:
            del run["build_cflags"]
        return run

    def refuses(runs):
        try:
            build_row(runs, "l", "M3", __file__, __file__)
            return False
        except SystemExit:
            return True

    flags = "-std=c11 -O3 -ffp-contract=off -Isrc"
    ck("schema 2 stamps the binary's flags into build.cflags",
       build_row([fake_run(flags), fake_run(flags)], "l", "M3", __file__,
                 __file__)["build"]["cflags"] == flags)
    ck("schema 2 REFUSES runs whose flags disagree (exe swapped mid-append)",
       refuses([fake_run(flags), fake_run("-std=c11 -O1 -Isrc")]))
    ck("schema 2 REFUSES a schema-1 binary that emits no flags",
       refuses([fake_run(None), fake_run(None)]))
    ck("schema 2 carries the serial variant as the schema-1 conversion key",
       build_row([fake_run(flags)], "l", "M3", __file__,
                 __file__)["ref_gbs_serial_runs"] == [7.4])
    ck("SCHEMA constant is 2", SCHEMA == 2)

    # ---- the box sidecar (header decision 3) ----
    def sample(t_ms, procs, free=8000):
        return {"t_ms": t_ms, "free_ram_mb": free,
                "procs": [{"pid": p, "name": n, "cpu_s": c,
                           "start_ms": (s[0] if s else None)}
                          for p, n, c, *s in procs]}

    # THE CASE THIS MECHANISM EXISTS FOR, in the day's real numbers: a game
    # holding ~5.08 cores across a 100 s window. If this ever grades "gate",
    # the guard has stopped doing the one job it was built for.
    quiet = sample(0, [(1, "idle", 0.0)])
    loaded_a = sample(0, [(14904, "FactoryGameSteam-Win64-Shipping", 1000.0)])
    loaded_b = sample(100_000, [(14904, "FactoryGameSteam-Win64-Shipping",
                                 1508.0)])
    d_loaded = sidecar_delta(loaded_a, loaded_b)
    ck("sidecar measures the game at 5.08 cores",
       abs(d_loaded["foreign_cpu_cores"] - 5.08) < 0.01)
    ck("sidecar names the culprit in the row",
       d_loaded["top_foreign"]["name"].startswith("FactoryGame"))
    ck("a 5-core co-load grades AMBIENT",
       box_grade([d_loaded], 4_280_405_216, 16)[0] == "ambient")

    d_quiet = sidecar_delta(quiet, sample(100_000, [(1, "idle", 0.4)]))
    ck("an idle box grades GATE",
       box_grade([d_quiet], 4_280_405_216, 16)[0] == "gate")
    ck("gate verdict carries no reasons",
       box_grade([d_quiet], 4_280_405_216, 16)[1] == [])

    # A row is only as comparable as its WORST run -- one loaded run out of
    # three is the case that produced the deleted 2026-08-03 baseline row.
    ck("one loaded run out of three condemns the row",
       box_grade([d_quiet, d_loaded, d_quiet], 4_280_405_216, 16)[0]
       == "ambient")

    # ABSENCE IS NOT QUIET. This is the distinction the whole mechanism turns
    # on, so it is asserted rather than assumed.
    ck("a missing sidecar grades AMBIENT, not gate",
       box_grade([None], 4_280_405_216, 16)[0] == "ambient")
    ck("no samples at all grades AMBIENT", box_grade([], 4_280_405_216, 16)[0]
       == "ambient")
    ck("sidecar_delta reports a missing sample as None",
       sidecar_delta(None, quiet) is None and sidecar_delta(quiet, None) is None)
    ck("sidecar_delta rejects a zero-length window",
       sidecar_delta(quiet, sample(0, [(1, "idle", 5.0)])) is None)

    # Free RAM under the model + headroom is disqualifying on its own, even on
    # a CPU-quiet box: the 4.28 GB mmap under ~5 GB free is the exact
    # condition two sessions argued about.
    tight = sidecar_delta(sample(0, [(1, "idle", 0.0)], free=5000),
                          sample(100_000, [(1, "idle", 0.4)], free=5000))
    g, why = box_grade([tight], 4_280_405_216, 16)
    ck("free RAM under model+2GB grades AMBIENT even when CPU is quiet",
       g == "ambient" and any("free RAM" in r for r in why))

    ck("a PID recycled mid-window is dropped, not subtracted",
       sidecar_delta(sample(0, [(7, "old", 900.0)]),
                     sample(10_000, [(7, "new", 1.0)]))
                    ["foreign_cpu_cores"] == 0.0)

    # A process seen only in the second sample is charged to the window only
    # if it STARTED there. Otherwise it was merely unreadable the first time,
    # and billing its whole lifetime here would break the floor guarantee --
    # the one property that lets "gate" be trusted.
    ck("a process that started mid-window is charged in full",
       sidecar_delta(sample(0, [(1, "idle", 0.0)]),
                     sample(10_000, [(1, "idle", 0.0), (99, "game", 20.0, 5000)]))
                    ["foreign_cpu_cores"] == 2.0)
    ck("a process merely unseen in the first sample is NOT charged",
       sidecar_delta(sample(50_000, [(1, "idle", 0.0)]),
                     sample(60_000, [(1, "idle", 0.0), (99, "old", 9999.0, 1000)]))
                    ["foreign_cpu_cores"] == 0.0)
    ck("an unreadable start time is treated as pre-existing, not as new",
       sidecar_delta(sample(50_000, [(1, "idle", 0.0)]),
                     sample(60_000, [(1, "idle", 0.0), (99, "sys", 9999.0)]))
                    ["foreign_cpu_cores"] == 0.0)

    # A BLIND SAMPLER MAY NOT CERTIFY QUIET. Measured under WSL 2026-08-03:
    # host at 5.35 foreign cores, WSL's /proc/stat view 0.01, verdict "gate".
    # The numbers below are that exact reading.
    blind_quiet = dict(d_quiet, blind="running under WSL -- sees the VM only")
    gb, gwhy = box_grade([blind_quiet], 4_280_405_216, 16)
    ck("a blind sampler cannot grade GATE however quiet it reads",
       gb == "ambient" and any("blind" in r for r in gwhy))

    # The power plan really did change from Performance to Turbo between two
    # checks 30 minutes apart on 2026-08-03, unprompted.
    swapped = dict(d_quiet, conditions_changed=True)
    ck("a power-plan change mid-run grades AMBIENT",
       box_grade([swapped], 4_280_405_216, 16)[0] == "ambient")
    ck("running on battery grades AMBIENT",
       box_grade([dict(d_quiet, on_ac=False)], 4_280_405_216, 16)[0]
       == "ambient")
    ck("a quiet run on AC with a stable plan still grades GATE",
       box_grade([dict(d_quiet, on_ac=True, conditions_changed=False,
                       power_plan="Turbo")], 4_280_405_216, 16)[0] == "gate")

    ck("build_row stamps a grade even with no sidecars",
       build_row([fake_run(flags)], "l", "M3", __file__, __file__)["box"]
       ["grade"] == "ambient")

    print(f"\nselftest: {checks - failed}/{checks} checks passed")
    return 1 if failed else 0


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path to the .gguf to bench")
    ap.add_argument("--exe", default=os.path.join(REPO, "build", "win",
                                                  "stoke.exe"))
    ap.add_argument("--runs", type=int, default=3,
                    help="separate bench RUNS to median across (default 3)")
    ap.add_argument("--label", default="", help="what this row records")
    ap.add_argument("--milestone", default="M3")
    ap.add_argument("--bench-args", nargs="*", default=[],
                    help="extra args passed through to stoke bench")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the row, append nothing")
    ap.add_argument("--force", action="store_true",
                    help="append even if the stability guard fails")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if not a.model:
        ap.error("--model is required (or use --selftest)")
    if not os.path.exists(a.exe):
        ap.error(f"binary not found: {a.exe}")
    if not os.path.exists(a.model):
        ap.error(f"model not found: {a.model}")
    if a.runs < 1:
        ap.error("--runs must be >= 1")
    if a.runs < 3 and not a.force:
        ap.error("--runs < 3 cannot detect sustained interference (see this "
                 "file's header, decision 2). Use --force if you mean it.")

    print(f"running {a.runs} bench runs ...")
    runs, sidecars = [], []
    for i in range(a.runs):
        r, sc = one_run(a.exe, a.model, a.bench_args)
        runs.append(r)
        sidecars.append(sc)
        load = (f", foreign {sc['foreign_cpu_cores']:.2f} cores"
                if sc else ", box UNSAMPLED")
        # Printed per run, not summarised at the end: an opt-out that failed on
        # run 2 of 3 is the one a reader most needs to see while it is happening.
        q = "" if sc is None or "qos_optout" not in sc else (
            ", qos opt-out ON" if sc["qos_optout"] else ", qos opt-out FAILED")
        print(f"  run {i + 1}/{a.runs}: ref {r['ref_gbs']:.2f} GB/s, "
              f"{len(r['ops'])} ops{load}{q}")

    row = build_row(runs, a.label, a.milestone, a.exe, a.model, sidecars)

    per_op = {op: o["med_us_runs"] for op, o in row["ops"].items()}
    violations = stability_violations(per_op)

    # The reference stream is checked separately and unconditionally: it is
    # the DENOMINATOR of every %ref in the row, so if it wobbles the whole
    # ratio basis wobbles with it, and no per-op exemption should apply.
    ref_spread = rel_spread(row["ref_gbs_runs"])
    if ref_spread > STABILITY_MAX_REL:
        violations.append(("reference stream", ref_spread))

    # Reported BEFORE the stability guard, because a passing guard on a loaded
    # box is precisely the trap this exists for: the 2026-08-03 row agreed with
    # itself across three runs and was 1.7x wrong (header decision 3).
    if row["box"]["grade"] == "gate":
        print("\nbox grade: GATE -- quiesced by measurement, comparison-grade.")
    else:
        print("\nbox grade: AMBIENT -- this row is NOT a diff base, NOT a "
              "referent, and may not be quoted against another row:")
        for r in row["box"]["grade_reasons"]:
            print(f"  - {r}")

    # ▶▶ THE THROTTLE GUARD, AND IT IS SEPARATE FROM THE STABILITY GUARD ON
    # PURPOSE. A throttled run is not an unstable one -- it is consistently
    # slow, so it sails through every spread check while being ~3x wrong (see
    # prefer_performance()). Nothing already in this file could have caught the
    # withdrawn 2026-08-28 row, which is the argument for a guard that asks a
    # different question: was the opt-out actually applied?
    qos_states = [sc.get("qos_optout") for sc in (sidecars or [])
                  if sc is not None and "qos_optout" in sc]
    if qos_states and not all(qos_states):
        print("\nTHROTTLE GUARD FAILED -- the EcoQoS opt-out did not apply to "
              f"{sum(1 for q in qos_states if not q)} of {len(qos_states)} run(s).")
        print("Windows was free to park bench on an efficiency core, which reads "
              "as a ~3x\nregression across EVERY op while ref_gbs -- being "
              "memory-bound -- barely moves.\nThat row would be a fabricated "
              "regression. Re-run, or pass --force if you accept it.")
        if not a.force:
            return 2

    if violations:
        print("\nSTABILITY GUARD FAILED -- this box is not measuring "
              "consistently right now:")
        for op, s in violations:
            print(f"  {op:<26} {s * 100:6.1f}% spread across runs")
        print("\nA row written from these runs would be a number a future "
              "regression hunt trusts and should not.\nQuiesce the box and "
              "re-run, or pass --force if you accept it.")
        if not a.force:
            return 2

    line = json.dumps(row, separators=(",", ":"), sort_keys=False)
    if a.dry_run:
        print("\n--dry-run, nothing appended. Row would be:\n")
        print(json.dumps(row, indent=2))
        return 0

    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    with open(HISTORY, "a", encoding="ascii", newline="\n") as fh:
        fh.write(line + "\n")
    n = sum(1 for _ in open(HISTORY, encoding="ascii"))
    print(f"\nappended to bench/history.jsonl ({n} row(s) total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
