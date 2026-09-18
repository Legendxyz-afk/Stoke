#!/usr/bin/env python3
"""Read one block of the @16 / @32 / ollama clause-2 sitting and say what it says.

    python tests/tooling/e2e_block_summary.py bench/results/<dir>/block1.jsonl \
        [--log bench/results/<dir>/block1.log] [--band 1.10 1.35] [--merge out.json]

A bank from tests/tooling/run_e2e_interleaved.py, one record per unit. This
prints, for every unit, the fields the pre-registration's §6 tripwires read
(completed, quiesce, the configuration THE SERVER announced, page faults
against the working set, CPU-seconds against wall), then the per-arm medians,
the ratio, every pair, and the sitting-level tripwires (arm spread, the pp512
referent's move across the block from the log). Nothing here decides; it
lays the numbers beside the rules so the reader can.

The reconstruction of a unit's aggregate from its raw fields is the harness's
own refusal (unit() raises); this script trusts a record that exists.
"""
import argparse
import json
import re
import statistics as st
import sys


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def n_clients(scenario):
    m = re.search(r"-(\d+)$", scenario or "")
    return int(m.group(1)) if m else None


def unit_line(r, n):
    """One line per unit, plus the list of tripwires it trips (may be empty)."""
    trips = []
    label = r.get("config", "?")
    if r.get("error"):
        return f"  {label:28s} ERROR: {r['error']}", ["error"]
    proc = r.get("proc") or {}
    tr = r.get("trace") or {}
    q = r.get("quiesce") or {}
    wall = r["wall_s"]
    cpu = proc.get("cpu_user_s")
    faults = proc.get("page_faults")
    ws = proc.get("peak_ws_mb")
    if r.get("completed") != n:
        trips.append(f"completed {r.get('completed')} != {n}")
    if not q.get("passed"):
        trips.append("quiesce not passed")
    is_stoke = label.startswith("stoke")
    if is_stoke:
        if r.get("sched_chunk") != 256:
            trips.append(f"sched_chunk {r.get('sched_chunk')} != 256")
        if tr.get("threads") != 12:
            trips.append(f"threads {tr.get('threads')} != 12")
    if "@ctx=" in label:
        if r.get("server_slots") != n:
            trips.append(f"server_slots {r.get('server_slots')} != {n}")
        if r.get("server_ctx_slot") != 704:
            trips.append(f"ctx/slot {r.get('server_ctx_slot')} != 704")
    if label.startswith("llamacpp") and "@ctx=" not in label and faults is not None and ws:
        # paging on the policy-ctx llama arm: first-touch faults are ~WS/4 KB
        if faults > 2 * ws * 256:
            trips.append(f"faults {faults} > 2x WS/4KB ({int(2 * ws * 256)})")
        if cpu is not None and cpu / wall < 8:
            trips.append(f"cpu-s/wall {cpu / wall:.1f} < 8")
    itl = r.get("itl") or {}
    itl_s = ""
    if itl:
        itl_s = f" itl50/95/max={itl.get('p50_ms', 0):.0f}/{itl.get('p95_ms', 0):.0f}/{itl.get('max_ms', 0):.0f}ms"
    cfg = ""
    if is_stoke:
        cfg = f" chunk={r.get('sched_chunk')} thr={tr.get('threads')}"
    elif r.get("server_slots") is not None:
        cfg = f" slots={r.get('server_slots')} ctx/slot={r.get('server_ctx_slot')}"
    line = (f"  p{r.get('pair')} {label:28s} wall={wall:7.1f}s agg={r['aggregate_decode_tps']:6.3f} "
            f"ttft50={r['ttft50_s']:5.1f}s cpu={cpu if cpu is None else round(cpu, 1)} "
            f"faults={faults} peakws={ws}MB{cfg}{itl_s} quiesce={q.get('reads', [])[-2:]}")
    if trips:
        line += "\n" + " " * 6 + "⚠ TRIPWIRE: " + "; ".join(trips)
    return line, trips


def referent_lines(log_path):
    """The pp512 referent lines launch_block.sh prints before and after the block."""
    out = {}
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.search(r"=== referent (BEFORE|AFTER): .*?\|\s*pp512\s*\|\s*([0-9.]+) ± ([0-9.]+)", line)
                if m:
                    out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank")
    ap.add_argument("--log", help="the block's log, for the pp512 referent bracket")
    ap.add_argument("--band", nargs=2, type=float, metavar=("LO", "HI"),
                    help="the frozen band for this block's ratio (PASS if inside)")
    ap.add_argument("--merge", help="write a merged bank (summary + every record) here")
    a = ap.parse_args()

    rows = load(a.bank)
    ok = [r for r in rows if not r.get("error")]
    labels = []
    for r in rows:
        if r.get("config") not in labels:
            labels.append(r.get("config"))
    n = n_clients(rows[0].get("scenario")) if rows else None
    print(f"== {a.bank}: {len(rows)} records, {len(ok)} units, scenario {rows[0].get('scenario') if rows else '?'} (N={n}) ==")

    all_trips = {}
    for r in rows:
        line, trips = unit_line(r, n)
        print(line)
        if trips:
            all_trips[(r.get("pair"), r.get("config"))] = trips

    walls = {}
    print("\n== per arm ==")
    for label in labels:
        w = [r["wall_s"] for r in ok if r["config"] == label]
        if not w:
            print(f"  {label}: NO UNITS")
            continue
        walls[label] = w
        spread = 100 * (max(w) - min(w)) / st.median(w)
        cpu = [r["proc"]["cpu_user_s"] for r in ok if r["config"] == label and r.get("proc")]
        ttft = [r["ttft50_s"] for r in ok if r["config"] == label]
        agg = [r["aggregate_decode_tps"] for r in ok if r["config"] == label]
        itlmax = [r["itl"]["max_ms"] for r in ok if r["config"] == label and r.get("itl")]
        print(f"  {label:28s} n={len(w)} wall median={st.median(w):7.1f}s own spread={spread:5.1f}% "
              f"units={[round(x, 1) for x in w]} agg median={st.median(agg):.3f} ttft50 median={st.median(ttft):.1f}s "
              f"cpu median={st.median(cpu) if cpu else None} itl max median={st.median(itlmax) if itlmax else None}")
        if spread > 10:
            print(f"      ⚠ TRIPWIRE: own spread {spread:.1f}% > 10% -- the ratio is reported, graded only against this")

    summary = {}
    if len(walls) == 2:
        la, lb = labels[0], labels[1]
        ma, mb = st.median(walls[la]), st.median(walls[lb])
        ratio = ma / mb
        pairs = {}
        for r in ok:
            pairs.setdefault(r["pair"], {})[r["config"]] = r["wall_s"]
        pair_ratios = [(k, v[la] / v[lb]) for k, v in sorted(pairs.items()) if la in v and lb in v]
        print(f"\n  RATIO {la} / {lb} = {ratio:.4f}x  (medians {ma:.1f} / {mb:.1f} s; >1 means {lb} finishes first)")
        print("  pairs: " + "  ".join(f"p{k}={v:.3f}" for k, v in pair_ratios))
        every_gt1 = all(v > 1.0 for _, v in pair_ratios)
        print(f"  every pair > 1.0: {every_gt1}")
        if a.band:
            lo, hi = a.band
            inside = lo <= ratio <= hi
            print(f"  band [{lo}, {hi}]: {'INSIDE' if inside else 'OUTSIDE'} -- {'PASS' if inside and every_gt1 else 'MISS (recorded, not re-fitted)'}")
        # the secondary quantities the note banded (H5-H8)
        def med(label, f):
            v = [f(r) for r in ok if r["config"] == label and f(r) is not None]
            return st.median(v) if v else None
        cpu_a, cpu_b = med(la, lambda r: (r.get("proc") or {}).get("cpu_user_s")), med(lb, lambda r: (r.get("proc") or {}).get("cpu_user_s"))
        ttft_a, ttft_b = med(la, lambda r: r.get("ttft50_s")), med(lb, lambda r: r.get("ttft50_s"))
        itl_a, itl_b = med(la, lambda r: (r.get("itl") or {}).get("max_ms")), med(lb, lambda r: (r.get("itl") or {}).get("max_ms"))
        if cpu_a and cpu_b:
            print(f"  CPU-s {lb} / {la} = {cpu_b / cpu_a:.3f}x   TTFT p50 {la} / {lb} = {ttft_a / ttft_b:.2f}x")
        if itl_a is not None or itl_b is not None:
            print(f"  ITL max (median over units): {la} {itl_a} ms, {lb} {itl_b} ms")
        summary = {"arm_a": la, "arm_b": lb, "walls_a": walls[la], "walls_b": walls[lb],
                   "median_a": ma, "median_b": mb, "ratio_a_over_b": ratio,
                   "pairs": [{"pair": k, "ratio": v} for k, v in pair_ratios], "every_pair_gt1": every_gt1,
                   "spread_a_pct": 100 * (max(walls[la]) - min(walls[la])) / ma,
                   "spread_b_pct": 100 * (max(walls[lb]) - min(walls[lb])) / mb,
                   "cpu_b_over_a": (cpu_b / cpu_a) if cpu_a and cpu_b else None,
                   "ttft50_a_over_b": (ttft_a / ttft_b) if ttft_a and ttft_b else None,
                   "itl_max_ms_a": itl_a, "itl_max_ms_b": itl_b,
                   "band": a.band, "tripwires": {f"p{k[0]}/{k[1]}": v for k, v in all_trips.items()}}

    if a.log:
        ref = referent_lines(a.log)
        if "BEFORE" in ref and "AFTER" in ref:
            b, af = ref["BEFORE"][0], ref["AFTER"][0]
            move = 100 * (af - b) / b
            print(f"\n  pp512 referent: before {b:.2f} ± {ref['BEFORE'][1]:.2f}, after {af:.2f} ± {ref['AFTER'][1]:.2f} t/s -> moved {move:+.1f}%"
                  + ("  ⚠ TRIPWIRE: > 8%" if abs(move) > 8 else ""))
            summary["referent_pp512"] = {"before": b, "after": af, "move_pct": move}
        else:
            print(f"\n  pp512 referent: {ref or 'not found in the log'}")

    if a.merge:
        with open(a.merge, "w", encoding="utf-8") as fh:
            json.dump({"bank": a.bank, "summary": summary, "records": rows}, fh, indent=1)
        print(f"\n  merged -> {a.merge}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
