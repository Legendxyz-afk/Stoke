#!/usr/bin/env python3
"""tests/tooling/run_e2e_interleaved.py -- clause 2's end-to-end ratio, measured
the way this box actually permits.

▶ WHY THIS EXISTS, AND IT IS A CORRECTION. The first attempt (2026-09-08,
`bench/results/2026-09-08-e2e-block`) ran five sequential cells of three runs
each and its pre-registered bracket FIRED: the same `llamacpp-tuned x
s2-closed-8` cell read 67.6 s at t+0 and 81.3 s at t+32, a **18.4% drift**
against a 5% threshold. Every cross-cell ratio from that block is therefore
VOID -- not wrong, unresolvable, because a 20% referent move swamps the ~3%
effect under test.

[[measurement-needs-interleaving]] already said this: *one unit per PROCESS,
counterbalance the order, detect drift from the arms' own repeats.* A
block-sequential matrix violates all three -- it couples three runs to one
thermal state and separates the arms by half an hour.

▶ WHAT THIS DOES INSTEAD.
  - ONE measured run per unit, one adapter lifetime per unit (fresh server).
  - Arms ALTERNATE (A B A B ...), so drift is shared rather than assigned.
  - A cool-down between units. NOTE: the drift that voided attempt 1 turned out
    to be Kabir opening Chrome and VS Code mid-block, NOT thermal load -- the
    cool-down is cheap insurance, not the fix for that. The fix for that is a
    quiesce gate, which bench/run_matrix.py still lacks (it prints a checklist
    and gates on nothing, where run_m5_bars.ps1 FAILS a unit that does not pass
    its own quiesce instrument).
  - Drift is read from EACH ARM'S OWN repeats across the sitting, which is the
    only control that works when both arms move together.
  - Every unit is banked as it completes: a run that dies at unit 7 keeps 6.

▶ IT ADDS NO MEASUREMENT LOGIC. Every number comes from bench.run_scenario via
run_matrix.run_cell -- the same code the referents are measured by, which is the
entire point of the exercise it serves.

▶ WHAT EACH UNIT NOW CARRIES BESIDE ITS RATE (2026-09-11). Attempt 2 found
stoke 0.2% repeatable inside one server lifetime and 11.7% apart across two,
and the bank could not say why, because it held rates and nothing about the
process that produced them. Every unit now banks: the WARMUP run's aggregate
(a within-lifetime repeat, free); the live process's CPU-seconds, page faults
and affinity/priority/EcoQoS state (`proc`, from bench/adapters.process_stats,
the two tells the 2026-08-17 EcoQoS finding named); and, for stoke, the pool's
placement trace (`STOKE_POOL_TRACE`, src/pool.c) -- per thread, the chunks it
claimed on P-cores against E-cores, sampled every --trace seconds, so a slow
process can be read as "its work sat on the efficiency cores from t=0" or
"something happened at t=90" rather than guessed at.

▶ AND A QUIESCE GATE PER UNIT. bench/run_matrix.py's checklist gated on
nothing; its quiesce_gate() now does, and a unit that cannot pass it is
REFUSED and the sitting stops (exit 2) -- a noisy box has a human on it.
--append resumes.

Usage:
  python tests/tooling/run_e2e_interleaved.py --scenario s2-closed-8 \
      --pairs 3 --cooldown 45 --out bench/results/<block>/interleaved-8.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "bench"))

import run_matrix  # noqa: E402
import scenarios as scen_mod  # noqa: E402

_TRACE_HDR = re.compile(
    r"pool trace: t=\+([\d.]+)s dispatches=(\d+) chunks=(\d+) cpu_user=([\d.]+)s "
    r"cpu_kernel=([\d.]+)s ecores=(\S+)")
_TRACE_THR = re.compile(r"pool trace: thr (\d+) tid=(\d+) ideal=(-?\d+) P=(\d+) E=(\d+) \|")
_SCHED_CHUNK = re.compile(r"sched: STOKE_SCHED_CHUNK=(\d+)")
_TRACE_ON = re.compile(r"pool trace: on, every [\d.]+s, (\d+) threads, spin=(\d+)")
# llama-server's own announcement of its slot count and per-slot context, so a
# `llamacpp-tuned@ctx=N` arm is verified from the server, not the label (the same
# rule `sched_chunk` follows for stoke). Both spellings are llama.cpp's server
# init lines and appear identically in ollama's embedded runner logs.
_LLAMA_SLOTS = re.compile(r"n_slots = (\d+)")
_LLAMA_CTX_SLOT = re.compile(r"n_ctx_slot = (\d+)")


def parse_trace(lines, startup=()):
    """The pool's periodic placement tables, compacted: one record per report,
    per-thread (ideal, P-chunks, E-chunks). Empty for engines that print none.
    The 'on' line's spin budget is attached to every report, so an A/B arm
    that asked for a spin can be checked against what the pool read.
    `startup` is the pre-healthy snapshot (adapters.Adapter.startup_lines): the
    'on' line lives there, and a long unit rotates it out of the 500-line ring
    `lines` comes from -- it is scanned for the header ONLY, never for reports,
    so a report that sits in both is not counted twice."""
    reports, cur, spin, nthr = [], None, None, None
    for ln in startup:
        m = _TRACE_ON.search(ln)
        if m:
            spin = int(m.group(2))
            nthr = int(m.group(1))
    for ln in lines:
        m = _TRACE_ON.search(ln)
        if m:
            spin = int(m.group(2))
            nthr = int(m.group(1))
            continue
        m = _TRACE_HDR.search(ln)
        if m:
            cur = {"t_s": float(m.group(1)), "dispatches": int(m.group(2)),
                   "chunks": int(m.group(3)), "cpu_user_s": float(m.group(4)),
                   "cpu_kernel_s": float(m.group(5)), "ecores": m.group(6), "spin": spin, "threads": nthr,
                   "thr": []}
            reports.append(cur)
            continue
        m = _TRACE_THR.search(ln)
        if m and cur is not None:
            cur["thr"].append({"w": int(m.group(1)), "ideal": int(m.group(3)),
                               "P": int(m.group(4)), "E": int(m.group(5))})
    return reports


def trace_summary(reports):
    """E-core share of all chunks over the whole lifetime, and over the LAST
    interval alone -- the pair that separates a startup state from a transient."""
    if not reports:
        return None
    last = reports[-1]
    p_all = sum(t["P"] for t in last["thr"])
    e_all = sum(t["E"] for t in last["thr"])
    out = {"reports": len(reports), "e_share_total": (e_all / (p_all + e_all)) if (p_all + e_all) else None,
           "ideal": [t["ideal"] for t in last["thr"]],
           "threads_with_e_ideal": None, "spin": last.get("spin"), "threads": last.get("threads"),
           # The dispatcher's chunk count against the helpers' median: the wake-latency
           # (or per-chunk speed) gap the 2026-09-11 sitting exposed at ~1.4x.
           "dispatcher_over_helper": None}
    if len(last["thr"]) >= 2:
        helpers = sorted(t["P"] + t["E"] for t in last["thr"][1:])
        med = helpers[len(helpers) // 2]
        d = last["thr"][0]["P"] + last["thr"][0]["E"]
        out["dispatcher_over_helper"] = round(d / med, 3) if med else None
    if last["ecores"].startswith("0x"):
        mask = int(last["ecores"], 16)
        out["threads_with_e_ideal"] = sum(1 for t in last["thr"] if t["ideal"] >= 0 and (mask >> t["ideal"]) & 1)
    if len(reports) >= 2:
        prev = reports[-2]
        dp = sum(t["P"] for t in last["thr"]) - sum(t["P"] for t in prev["thr"])
        de = sum(t["E"] for t in last["thr"]) - sum(t["E"] for t in prev["thr"])
        out["e_share_last_interval"] = (de / (dp + de)) if (dp + de) else None
    # Per-report E share, so a reader can see the time course without the raw lines.
    course = []
    for i, r in enumerate(reports):
        p = sum(t["P"] for t in r["thr"]); e = sum(t["E"] for t in r["thr"])
        if i:
            pp = sum(t["P"] for t in reports[i - 1]["thr"]); pe = sum(t["E"] for t in reports[i - 1]["thr"])
            p, e = p - pp, e - pe
        course.append(round(e / (p + e), 3) if (p + e) else None)
    out["e_share_course"] = course
    return out


def unit(config_label, scenario, timeout_s):
    """One adapter lifetime, one discarded warmup, ONE measured run."""
    cell = run_matrix.run_cell(config_label, scenario, runs=1, timeout_s=timeout_s, keep_log=True)
    if cell["error"]:
        return {"config": config_label, "error": cell["error"]}
    r = cell["runs"][0]
    pcs = r["per_client"]
    if r.get("errors"):
        return {"config": config_label, "error": str(r["errors"])}
    # ⚠ The verdict comes from reconstructing a number the harness also
    # publishes -- the one check that catches a unit error. A unit that cannot
    # reconstruct is refused, not banked.
    num = sum(p["n_tokens"] - 1 for p in pcs)
    span = max(p["last_token_abs"] for p in pcs) - min(p["first_token_abs"] for p in pcs)
    if abs(num / span - r["aggregate_decode_tps"]) > 1e-9:
        return {"config": config_label,
                "error": f"reconstruction failed: {num/span} vs {r['aggregate_decode_tps']}"}
    if {p["n_tokens"] for p in pcs} != {scenario.max_tokens}:
        return {"config": config_label,
                "error": f"token counts {sorted({p['n_tokens'] for p in pcs})} "
                         f"!= {scenario.max_tokens}"}
    ttfts = sorted(p["ttft_ms"] / 1000.0 for p in pcs)
    # Client-observed inter-token latency, pooled over every client of the run
    # (2026-09-16). Clause 4 was graded on exactly this quantity and ruled a
    # documented miss (memos/2026-09-16-clause4-itl-stall-DISPOSITION-KABIR.md);
    # the ruling asks for the REFERENT's own stall at @16/@32 as the re-anchor,
    # and a bank that drops the ITLs cannot supply it for either arm.
    itls = sorted(x for p in pcs for x in p["itl_ms"])
    itl = None
    if itls:
        itl = {"n": len(itls), "p50_ms": itls[len(itls) // 2],
               "p95_ms": itls[min(len(itls) - 1, int(0.95 * len(itls)))],
               "max_ms": itls[-1],
               # per-client maxima, so one stalled stream reads apart from all of them
               "client_max_ms": sorted(max(p["itl_ms"]) for p in pcs if p["itl_ms"])}
    log = cell.get("adapter_log") or []
    startup = cell.get("adapter_startup_log") or []
    reports = parse_trace(log, startup=startup)
    # The scheduler announces a STOKE_SCHED_CHUNK override at open (2026-09-15);
    # banked so a `stoke-tuned@chunk=N` arm is verified from the server's own log,
    # never from the label. None = nothing announced = the locked default -- which
    # is 128 in every bank up to `73b9246` (2026-09-15 afternoon) and 256 from the
    # commit that shipped the graded width the same evening; a bank does not yet
    # carry the binary's identity, so read None against the bank's date.
    # Read from the startup snapshot first (2026-09-16): every announcement below
    # is printed before the engine turns healthy, and a unit longer than the
    # ring's 500 lines would otherwise bank None for a value the server did say.
    sched_chunk = None
    server_slots = None
    server_ctx_slot = None
    for ln in list(startup) + list(log):
        m = _SCHED_CHUNK.search(ln)
        if m:
            sched_chunk = int(m.group(1))
        m = _LLAMA_SLOTS.search(ln)
        if m:
            server_slots = int(m.group(1))
        m = _LLAMA_CTX_SLOT.search(ln)
        if m:
            server_ctx_slot = int(m.group(1))
    return {
        "config": config_label,
        "scenario": scenario.name,
        "wall_s": r["wall_s"],
        "aggregate_decode_tps": r["aggregate_decode_tps"],
        "warmup_aggregate_decode_tps": (cell.get("warmup") or {}).get("aggregate_decode_tps"),
        "ttft50_s": ttfts[len(ttfts) // 2],
        "ttft_min_s": ttfts[0],
        "itl": itl,
        "completed": r["completed"],
        "adapter_log_head": (cell.get("adapter_log_tail") or "").splitlines()[:2],
        "sched_chunk": sched_chunk,
        # llama-server's announced slot count and per-slot ctx (None for stoke), so a
        # `llamacpp-tuned@ctx=N` unit carries the server's word beside the label.
        "server_slots": server_slots,
        "server_ctx_slot": server_ctx_slot,
        "proc": cell.get("proc"),
        "trace": trace_summary(reports),
        "trace_reports": reports,
        "error": None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--arm-a", default="llamacpp-tuned")
    p.add_argument("--arm-b", default="stoke-tuned")
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--cooldown", type=float, default=45.0)
    p.add_argument("--timeout-s", type=int, default=900)
    p.add_argument("--out", required=True)
    p.add_argument("--append", action="store_true",
                   help="add units to an existing bank instead of truncating it. "
                        "A stopped sitting keeps every unit it banked, and the "
                        "analysis reads the whole bank -- so resuming is the "
                        "normal case, not a special one.")
    p.add_argument("--trace", type=float, default=10.0,
                   help="STOKE_POOL_TRACE interval in seconds for the stoke arm's "
                        "server (0 = off). Inherited by the spawned process.")
    p.add_argument("--quiesce-tries", type=int, default=10,
                   help="attempts, 20 s apart, for the per-unit quiesce gate (0 = no gate; "
                        "every unit then banks quiesce.passed=false)")
    a = p.parse_args()

    if a.trace > 0:
        os.environ["STOKE_POOL_TRACE"] = str(a.trace)
    else:
        os.environ.pop("STOKE_POOL_TRACE", None)

    scenario = run_matrix.SCENARIOS_BY_NAME[a.scenario]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prior = []
    if a.append and out.exists():
        prior = [json.loads(L) for L in out.read_text(encoding="utf-8").splitlines() if L.strip()]
        print(f"resuming: {len(prior)} unit(s) already banked in {out}")
    fh = out.open("a" if a.append else "w", encoding="utf-8")

    banked = list(prior)
    t_start = time.time()
    refused = False
    # A sitting sliced into `--pairs 1` invocations (the foreground rule) must
    # keep flipping: the pair index continues from what the bank already holds,
    # or every invocation would run the same order and the counterbalance is gone.
    # ⚠ Not "the arm with fewer units goes first": a bank that is one unit
    # uneven (a stopped sitting) would then run the SAME order on every
    # invocation, because each pair preserves the imbalance. Pair parity only.
    k0 = sum(1 for r in prior if not r.get("error")) // 2
    for k in range(k0, k0 + a.pairs):
        # ABBA within a pair: the order flips every pair, so neither arm
        # systematically occupies the hotter half of the sitting.
        order = (a.arm_a, a.arm_b) if k % 2 == 0 else (a.arm_b, a.arm_a)
        for label in order:
            if a.cooldown > 0 and len(banked) > len(prior):
                time.sleep(a.cooldown)
            if a.quiesce_tries > 0:
                gate = run_matrix.quiesce_gate(tries=a.quiesce_tries)
                if not gate["passed"]:
                    rec = {"config": label, "error": "quiesce gate not met", "quiesce": gate,
                           "pair": k, "t_min": round((time.time() - t_start) / 60.0, 2)}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"  [{rec['t_min']:5.1f}m] {label:16s} REFUSED: box not quiet "
                          f"(reads {gate['reads'][-3:]}) -- resume with --append", flush=True)
                    refused = True
                    break
            else:
                gate = {"passed": False, "reads": [], "skipped": True}
            rec = unit(label, scenario, a.timeout_s)
            rec["quiesce"] = gate
            rec["pair"] = k
            rec["t_min"] = round((time.time() - t_start) / 60.0, 2)
            # Absolute rates belong to a SITTING; a bank that spans several needs
            # the wall-clock stamp to group them (t_min restarts per invocation).
            rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            banked.append(rec)
            if rec.get("error"):
                print(f"  [{rec['t_min']:5.1f}m] {label:16s} ERROR: {rec['error']}",
                      flush=True)
            else:
                proc = rec.get("proc") or {}
                tr = rec.get("trace") or {}
                extra = ""
                if "cpu_user_s" in proc:
                    extra += f" cpu={proc['cpu_user_s']:6.1f}s faults={proc.get('page_faults', '?')}"
                if tr.get("e_share_total") is not None:
                    extra += (f" E-share={tr['e_share_total']:.3f} d/h={tr.get('dispatcher_over_helper')}"
                              f" spin={tr.get('spin')} thr={tr.get('threads')} aff={proc.get('affinity_mask')}")
                if rec.get("sched_chunk") is not None:
                    extra += f" chunk={rec['sched_chunk']}"
                if rec.get("server_slots") is not None:
                    extra += f" slots={rec['server_slots']} ctx/slot={rec.get('server_ctx_slot')}"
                if "peak_ws_mb" in proc:
                    extra += f" peakws={proc['peak_ws_mb']:.0f}MB"
                if rec.get("itl"):
                    extra += f" itl50/95/max={rec['itl']['p50_ms']:.0f}/{rec['itl']['p95_ms']:.0f}/{rec['itl']['max_ms']:.0f}ms"
                print(f"  [{rec['t_min']:5.1f}m] {label:16s} wall={rec['wall_s']:7.1f}s "
                      f"agg={rec['aggregate_decode_tps']:6.3f} "
                      f"warm={rec['warmup_aggregate_decode_tps'] or 0:6.3f} "
                      f"ttft50={rec['ttft50_s']:5.1f}s{extra}", flush=True)
        if refused:
            break
    fh.close()

    ok = [r for r in banked if not r.get("error")]
    print("\n== per arm ==")
    walls = {}
    for label in (a.arm_a, a.arm_b):
        w = [r["wall_s"] for r in ok if r["config"] == label]
        if not w:
            print(f"  {label}: NO UNITS"); continue
        walls[label] = w
        drift = 100 * (max(w) - min(w)) / st.median(w)
        print(f"  {label:16s} n={len(w)} wall median={st.median(w):7.1f}s "
              f"own spread={drift:5.1f}%  units={[round(x,1) for x in w]}")
    if len(walls) == 2 and all(walls.values()):
        ma, mb = st.median(walls[a.arm_a]), st.median(walls[a.arm_b])
        print(f"\n  RATIO {a.arm_a} / {a.arm_b} = {ma/mb:.4f}x "
              f"(>1 means {a.arm_b} finishes the work first)")
        worst = max(100 * (max(w) - min(w)) / st.median(w) for w in walls.values())
        print(f"  ⚠ largest single-arm spread this sitting: {worst:.1f}% -- a ratio "
              f"is only readable if it exceeds this")
    # The per-process table: what each unit's process looked like beside its
    # rate, so a slow one can be read rather than guessed at.
    rows = [r for r in ok if r.get("proc") or r.get("trace")]
    if rows:
        print("\n== per unit: process state ==")
        print(f"  {'t+min':>6} {'arm':16s} {'wall':>7} {'agg':>7} {'warm':>7} {'cpu_u':>7} "
              f"{'faults':>8} {'E-share':>8} {'E-ideal':>8}")
        for r in rows:
            proc = r.get("proc") or {}
            tr = r.get("trace") or {}
            es = tr.get("e_share_total")
            print(f"  {r['t_min']:6.1f} {r['config']:16s} {r['wall_s']:7.1f} "
                  f"{r['aggregate_decode_tps']:7.3f} {(r.get('warmup_aggregate_decode_tps') or 0):7.3f} "
                  f"{proc.get('cpu_user_s', float('nan')):7.1f} {str(proc.get('page_faults', '?')):>8} "
                  f"{(f'{es:.3f}' if es is not None else '-'):>8} "
                  f"{str(tr.get('threads_with_e_ideal', '-')):>8}")
    return 2 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
