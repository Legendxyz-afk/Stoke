#!/usr/bin/env python3
"""M0 Task 3: aggregate run_matrix.py's raw per-cell JSON into the committed baseline report.

Reads <out-dir>/raw/environment.json and every <out-dir>/raw/<config>--<scenario>.json cell file
written by run_matrix.py, and produces two committed files, <out-dir>/<date>-baseline-qwen3-4b.json
and the matching .md: same content, json for machines, markdown for humans.

The JSON keeps one full Report dict (including per_client) for each cell's MEDIAN run only; the
other measured runs are represented solely by their aggregate_decode_tps (in runs_aggregate) so
the committed file stays small and readable -- the raw/ files (gitignored) keep everything.

Imports run_matrix (same-directory sys.path pattern as adapters.py) rather than re-declaring the
config/scenario canonical ordering or the tuned-config policy table, so both scripts share a
single source of truth and the policy table lands in the markdown report verbatim.
"""
import argparse
import datetime
import json
import pathlib
import sys

# Windows consoles default to a legacy code page (cp1252) that can't encode the em-dashes /
# multiply signs used below; force UTF-8 with a replace-on-failure fallback.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import run_matrix  # noqa: E402  (path insert above must run first; pulls in bench/scenarios/adapters too)

REPO_ROOT = run_matrix.REPO_ROOT
CONFIG_LABELS = run_matrix.CONFIG_LABELS
SCENARIO_NAMES = run_matrix.SCENARIO_NAMES

PHYSICS_CEILING_TPS = 160.0
PREMISE_THRESHOLD_PCT = 70.0
S2_NS = [1, 2, 4, 8, 16]

SCENARIO_TABLE_HEADERS = [
    "config", "aggregate tok/s", "TTFT p50 ms", "TTFT p95 ms", "ITL p50 ms", "ITL p95 ms",
    "completed", "notes",
]

PROTOCOL_PARAGRAPH = (
    "All four baseline configs (ollama-default, ollama-tuned, llamacpp-default, llamacpp-tuned) "
    "serve the same GGUF over the same locked scenario set, and the engine process is restarted "
    "between every cell so no cache or KV state leaks across configs or scenarios. Each cell runs "
    "1 discarded warmup request followed by N measured runs, and the numbers below are the median "
    "run by aggregate decode tok/s. Prompts are salted per run so engines with prefix/prompt "
    "caching cannot skip prefill on runs 2..N and silently inflate them, while S3's shared system "
    "prefix stays identical across clients within a single run (the capability S3 measures). "
    "Every request carries a per-request socket timeout. The published aggregate decode tok/s is "
    "decode-only -- it excludes each client's own first (TTFT/prefill) token -- and token counts "
    "come from the engine's own reported usage.completion_tokens (authoritative even when an "
    "engine coalesces multiple decode steps into one SSE chunk), while chunk arrival times still "
    "drive TTFT/ITL -- so ITL here is inter-arrival latency, and a coalesced multi-token chunk "
    "contributes a single gap. The underlying timing/measurement code (bench.run_scenario) is "
    "mock-verified against a known-exact server in test_harness.py before ever touching a real "
    "engine."
)


# -- generic markdown table rendering --------------------------------------------------------------
def render_md_table(headers, rows):
    """rows: list of dicts keyed by header name; a missing key renders as an empty cell."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _first_line(s):
    if not s:
        return ""
    s = str(s).strip()
    return s.splitlines()[0] if s else ""


def _fmt_ram_gb(ram_bytes):
    try:
        return f"{float(ram_bytes) / (1024 ** 3):.1f} GB"
    except (TypeError, ValueError):
        return str(ram_bytes)


def _fmt_model_size(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    return f"{n:,} bytes ({n / 1e9:.2f} GB)"


# -- median selection (JSON §Median rule) ----------------------------------------------------------
def _is_successful_run(report):
    per_client = report.get("per_client") or []
    return report.get("errors") == [] and report.get("completed") == len(per_client)


def select_median(runs):
    """Among `runs` (list of Report dicts), return the median run by aggregate_decode_tps among
    the successful ones (errors empty AND completed == len(per_client) -- run_scenario doesn't
    store scenario.n_clients on the Report itself). Even count picks the LOWER of the two middle
    values. No successful run -> None.
    """
    successful = [r for r in runs if _is_successful_run(r)]
    if not successful:
        return None
    ordered = sorted(successful, key=lambda r: r["aggregate_decode_tps"])
    n = len(ordered)
    idx = (n // 2 - 1) if n % 2 == 0 else (n // 2)
    return ordered[idx]


def build_cell_entry(cell):
    """Aggregate one raw cell dict (run_matrix.run_cell's output) into the committed-JSON shape:
    per-run per_client arrays are dropped except on the median run."""
    runs = cell.get("runs") or []
    median_report = select_median(runs)
    warmup = cell.get("warmup") or {}
    entry = {
        "config": cell.get("config"),
        "scenario": cell.get("scenario"),
        "error": cell.get("error"),
        "warmup_aggregate": warmup.get("aggregate_decode_tps"),
        "runs_aggregate": [r.get("aggregate_decode_tps") for r in runs],
        "median": median_report,
        "ollama_show": cell.get("ollama_show"),
    }
    if median_report is None:
        entry["failures"] = [r.get("errors", []) for r in runs]
    if cell.get("error"):
        entry["adapter_log_tail"] = cell.get("adapter_log_tail")
    return entry


# -- markdown sections ------------------------------------------------------------------------------
def environment_section(env):
    rows = [
        {"field": "CPU", "value": env.get("cpu", "")},
        {"field": "RAM", "value": _fmt_ram_gb(env.get("ram_bytes"))},
        {"field": "Platform", "value": env.get("platform", "")},
        {"field": "Power scheme", "value": _first_line(env.get("power_scheme_before", ""))},
        {"field": "Ollama version", "value": _first_line(env.get("ollama_version", ""))},
        {"field": "llama.cpp version", "value": _first_line(env.get("llamacpp_version", ""))},
        {"field": "Model file", "value": f"{env.get('model_file', '')} ({_fmt_model_size(env.get('model_file_bytes'))})"},
        {"field": "Git rev", "value": env.get("git_rev", "")},
    ]
    return "## Environment\n\n" + render_md_table(["field", "value"], rows)


def scenario_row(config_label, cell_entry):
    row = {"config": config_label}
    if cell_entry is None:
        row.update({
            "aggregate tok/s": "-", "TTFT p50 ms": "-", "TTFT p95 ms": "-",
            "ITL p50 ms": "-", "ITL p95 ms": "-", "completed": "-",
            "notes": "not run",
        })
        return row
    if cell_entry.get("error") or cell_entry.get("median") is None:
        note = _first_line(cell_entry.get("error") or "no successful run (all measured runs failed)")
        row.update({
            "aggregate tok/s": "FAILED", "TTFT p50 ms": "-", "TTFT p95 ms": "-",
            "ITL p50 ms": "-", "ITL p95 ms": "-", "completed": "-",
            "notes": note,
        })
        return row
    m = cell_entry["median"]
    per_client = m.get("per_client") or []
    coalesced = sum((c.get("n_tokens", 0) - c.get("n_arrivals", 0)) for c in per_client)
    row.update({
        "aggregate tok/s": f"{m['aggregate_decode_tps']:.1f}",
        "TTFT p50 ms": f"{m['ttft_ms']['p50']:.1f}",
        "TTFT p95 ms": f"{m['ttft_ms']['p95']:.1f}",
        "ITL p50 ms": f"{m['itl_ms']['p50']:.1f}",
        "ITL p95 ms": f"{m['itl_ms']['p95']:.1f}",
        "completed": f"{m['completed']}/{len(per_client)}",
        "notes": f"+{coalesced}" if coalesced > 0 else "",
    })
    return row


def run_to_run_line(config_label, cell_entry):
    aggs = (cell_entry or {}).get("runs_aggregate") or []
    if not aggs:
        return f"- {config_label}: (no data)"
    vals = ", ".join(f"{v:.1f}" if isinstance(v, (int, float)) else str(v) for v in aggs)
    return f"- {config_label}: {vals}"


def scenario_section(scenario_name, cells_by_key):
    lines = [f"## {scenario_name}", ""]
    rows = [scenario_row(cfg, cells_by_key.get(f"{cfg}--{scenario_name}")) for cfg in CONFIG_LABELS]
    lines.append(render_md_table(SCENARIO_TABLE_HEADERS, rows))
    lines.append("")
    lines.append("run-to-run aggregates:")
    for cfg in CONFIG_LABELS:
        lines.append(run_to_run_line(cfg, cells_by_key.get(f"{cfg}--{scenario_name}")))
    return "\n".join(lines)


def s2_aggregate(cells_by_key, config_label, n):
    entry = cells_by_key.get(f"{config_label}--s2-closed-{n}")
    if entry is None or entry.get("median") is None:
        return None
    return entry["median"]["aggregate_decode_tps"]


def s2_sweep_section(cells_by_key):
    headers = ["N"] + CONFIG_LABELS
    raw_rows = []
    for n in S2_NS:
        row = {"N": str(n)}
        for cfg in CONFIG_LABELS:
            v = s2_aggregate(cells_by_key, cfg, n)
            row[cfg] = f"{v:.1f}" if v is not None else "-"
        raw_rows.append(row)

    base = {cfg: s2_aggregate(cells_by_key, cfg, 1) for cfg in CONFIG_LABELS}
    scaling_rows = []
    for n in S2_NS:
        row = {"N": str(n)}
        for cfg in CONFIG_LABELS:
            v = s2_aggregate(cells_by_key, cfg, n)
            b = base[cfg]
            row[cfg] = f"{v / b:.2f}×" if (v is not None and b) else "-"
        scaling_rows.append(row)

    lines = ["## S2 concurrency sweep — aggregate decode tok/s vs N", ""]
    lines.append(render_md_table(headers, raw_rows))
    lines.append("")
    lines.append("Scaling, normalized to each config's N=1 (×, 2dp) -- the scaling curve is the wedge evidence:")
    lines.append("")
    lines.append(render_md_table(headers, scaling_rows))
    return "\n".join(lines)


def premise_gate_section(cells_by_key):
    lines = ["## Premise gate (kill-switch)", ""]
    lines.append(
        "Spec anchor: physics ceiling ≈ **160 tok/s aggregate** "
        "(spec `docs/superpowers/specs/2026-07-16-phase1-concurrent-cpu-core-design.md`, "
        "\"Physics budget\": batch-8 step ≈ 2.4GB weights + 8×74KB/tok×~1k KV ≈ "
        "3.0GB @ ~60GB/s real → ~20 steps/s → ~160 tok/s; threshold 70% = 112 tok/s)."
    )
    lines.append("")

    headers = ["scenario", "config", "median aggregate tok/s", "% of 160", "verdict"]
    rows = []
    breached = set()
    for scenario_name in ("s2-closed-8", "s2-closed-16"):
        for cfg in CONFIG_LABELS:
            entry = cells_by_key.get(f"{cfg}--{scenario_name}")
            if entry is None or entry.get("median") is None:
                rows.append({
                    "scenario": scenario_name, "config": cfg,
                    "median aggregate tok/s": "-", "% of 160": "-", "verdict": "-",
                })
                continue
            agg = entry["median"]["aggregate_decode_tps"]
            pct = agg / PHYSICS_CEILING_TPS * 100.0
            if pct >= PREMISE_THRESHOLD_PCT:
                verdict = "**≥70% BREACH**"
                breached.add(cfg)
            else:
                verdict = "below"
            rows.append({
                "scenario": scenario_name, "config": cfg,
                "median aggregate tok/s": f"{agg:.1f}", "% of 160": f"{pct:.1f}%",
                "verdict": verdict,
            })
    lines.append(render_md_table(headers, rows))
    lines.append("")

    if breached:
        final_line = (
            f"**PREMISE GATE: BREACHED by {', '.join(sorted(breached))} "
            f"— STOP, reconvene before kernel work**"
        )
    else:
        final_line = "**PREMISE GATE: CONFIRMED — all baselines below 70% of ceiling**"
    lines.append(final_line)
    return "\n".join(lines), final_line


def failures_section(raw_cells):
    """Brief spec: every cell error verbatim + its adapter_log_tail, plus any successful cell
    whose median run has non-empty errors (partial client failures), listed with counts.

    That second clause is unreachable as literally written: select_median() only ever chooses
    among runs with errors == [] (that's the definition of a "successful run" the median rule
    uses), so a chosen median can never itself carry non-empty errors -- the check is kept below
    anyway (harmless, guards against a future change to the median rule). What IS reachable, and
    would otherwise be invisible here, is a cell that ran to completion (no top-level cell.error)
    but had ZERO successful runs out of N -- every measured run had per-client failures, so
    median is None even though nothing raised. That cell already renders as a FAILED row in its
    scenario table; it belongs in this section too, so it's added as its own case.
    """
    lines = ["## Failures & anomalies", ""]
    any_content = False

    for cell in raw_cells:
        if not cell.get("error"):
            continue
        any_content = True
        key = f"{cell.get('config')}--{cell.get('scenario')}"
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"Error: {cell['error']}")
        lines.append("")
        lines.append("```")
        lines.append(cell.get("adapter_log_tail") or "(no output captured)")
        lines.append("```")
        lines.append("")

    for cell in raw_cells:
        if cell.get("error"):
            continue  # already covered above
        runs = cell.get("runs") or []
        median = select_median(runs)
        key = f"{cell.get('config')}--{cell.get('scenario')}"
        if median is None and runs:
            any_content = True
            lines.append(f"### {key} (no successful run)")
            lines.append("")
            lines.append(
                f"0/{len(runs)} measured runs succeeded. Per-run errors: "
                f"{[r.get('errors', []) for r in runs]}"
            )
            lines.append("")
        elif median is not None and median.get("errors"):
            any_content = True
            lines.append(f"### {key} (partial client failures in median run)")
            lines.append("")
            lines.append(f"{len(median['errors'])} client error(s): {median['errors']}")
            lines.append("")

    if not any_content:
        lines.append("None.")
        lines.append("")
    return "\n".join(lines)


def render_markdown(date_str, environment, raw_cells, cells_by_key):
    parts = [
        f"# stoke M0 baselines — Qwen3-4B-Instruct-2507 Q4_K_M — {date_str}",
        "",
        PROTOCOL_PARAGRAPH,
        "",
        environment_section(environment),
        "",
        "## Tuned-config policy",
        "",
        run_matrix.TUNED_CONFIG_POLICY_MD,
        "",
    ]
    for scenario_name in SCENARIO_NAMES:
        parts.append(scenario_section(scenario_name, cells_by_key))
        parts.append("")
    parts.append(s2_sweep_section(cells_by_key))
    parts.append("")
    gate_body, final_line = premise_gate_section(cells_by_key)
    parts.append(gate_body)
    parts.append("")
    parts.append(failures_section(raw_cells))
    return "\n".join(parts), final_line


# -- I/O + CLI ----------------------------------------------------------------------------------
def load_raw_cells(raw_dir):
    cells = []
    if not raw_dir.exists():
        return cells
    for path in sorted(raw_dir.glob("*--*.json")):
        with path.open("r", encoding="utf-8") as f:
            cells.append(json.load(f))
    return cells


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir", default="bench/results",
        help="results directory containing raw/, resolved relative to the repo root if not absolute "
             "(default bench/results)",
    )
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, default: today")
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    date_str = args.date or datetime.date.today().isoformat()

    env_path = raw_dir / "environment.json"
    environment = {}
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            environment = json.load(f)

    raw_cells = load_raw_cells(raw_dir)
    cells_by_key = {f"{c['config']}--{c['scenario']}": build_cell_entry(c) for c in raw_cells}

    report_json = {"environment": environment, "cells": cells_by_key}
    json_path = out_dir / f"{date_str}-baseline-qwen3-4b.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    md_text, final_line = render_markdown(date_str, environment, raw_cells, cells_by_key)
    md_path = out_dir / f"{date_str}-baseline-qwen3-4b.md"
    md_path.write_text(md_text, encoding="utf-8")

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(final_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
