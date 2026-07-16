#!/usr/bin/env python3
"""M0 Task 3: the config x scenario baseline matrix runner.

Runs every combination of the four baseline serving configs (ollama-default, ollama-tuned,
llamacpp-default, llamacpp-tuned) against the eight locked scenarios, ONE ENGINE AT A TIME --
adapters.py's Adapter.stop() contract guarantees port exclusivity (tree-kill + block until the
port is verified free), so this runner never has two servers occupying a port simultaneously.

Resumable: each cell writes its raw JSON file immediately on completion, under
<out-dir>/raw/<config>--<scenario>.json. A re-run of the same command skips any cell whose raw
file already exists (print `SKIP (exists): <name>`); pass --fresh to force re-execution.

Every published number in this matrix is produced by bench.run_scenario -- the same function
whose timing/token-accounting behavior is mock-verified against a known-exact mock server in
test_harness.py before it is ever pointed at a real engine. This runner adds no measurement logic
of its own; it only orchestrates adapters + run_scenario calls and records what comes back.

bench/report.py reads this script's raw/*.json output and aggregates it into the committed
baseline report (json + md); it also imports TUNED_CONFIG_POLICY_MD from this module so the
policy table appears verbatim in both places.
"""
import argparse
import atexit
import datetime
import json
import os
import pathlib
import platform
import re
import subprocess
import sys

# Windows consoles default to a legacy code page (cp1252) that can't encode the "x" (multiply)
# and other punctuation used in the console output this module prints; force UTF-8 with a
# replace-on-failure fallback so `python bench/run_matrix.py` never crashes on print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench, scenarios, adapters  # noqa: E402  (path insert above must run first)

REPO_ROOT = adapters.REPO_ROOT

CONFIG_LABELS = ["ollama-default", "ollama-tuned", "llamacpp-default", "llamacpp-tuned"]

SCENARIOS = [scenarios.S1] + scenarios.S2 + [scenarios.S3, scenarios.S4]
SCENARIO_NAMES = [s.name for s in SCENARIOS]
SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}

HIGH_PERF_GUID = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"  # stock Windows High-performance scheme
GUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


# -- tuned-config policy --------------------------------------------------------------------------
TUNED_CONFIG_POLICY_MD = """\
Let P(scenario): `s2-closed-N` → N; `s1-interactive` and `s3-agent-fanout` → 8;
`s4-longctx` → 1.

| config            | construction                                                                     |
|-------------------|-----------------------------------------------------------------------------------|
| ollama-default    | `adapters.OllamaAdapter()`                                                        |
| ollama-tuned      | `adapters.OllamaAdapter(num_parallel=P, context_length=(8192 if s4 else 2048))`   |
| llamacpp-default  | `adapters.LlamaCppAdapter()`                                                      |
| llamacpp-tuned    | `adapters.LlamaCppAdapter(parallel=P, ctx=(8192 if s4 else P*2048))`              |

Rationale: "tuned" = what a competent operator sets for the known workload -- slots matched to
concurrency, per-slot context matched to prompt+output size (2048 covers every scenario except
S4's 4096+128, which gets 8192 and P=1). Per-slot ctx also kept equal across the two tuned engines
for fairness. A fresh adapter instance is constructed per cell; the engine restarts per cell so no
cache/KV state leaks between cells.
"""

# The table above is also the load-bearing content report.py transcribes verbatim into the
# committed markdown's "## Tuned-config policy" section -- append it to __doc__ so it is
# genuinely part of this module's docstring (brief requirement) with a single source of truth
# (this same string, imported by report.py) rather than a hand-copied duplicate.
__doc__ = (__doc__ or "") + "\n### Tuned-config policy\n\n" + TUNED_CONFIG_POLICY_MD


def _parallelism_for(scenario):
    """P(scenario), per the tuned-config policy table above: s2-closed-N -> N; s1-interactive and
    s3-agent-fanout -> 8; s4-longctx -> 1."""
    if scenario.name.startswith("s2-closed-"):
        return scenario.n_clients
    if scenario.name in ("s1-interactive", "s3-agent-fanout"):
        return 8
    if scenario.name == "s4-longctx":
        return 1
    raise ValueError(f"no P() defined for scenario {scenario.name!r}")


def build_adapter(config_label, scenario):
    """Construct a fresh adapter for (config_label, scenario) per the tuned-config policy table.
    A new instance is built per cell; the engine restarts per cell so no cache/KV state leaks."""
    is_s4 = scenario.name == "s4-longctx"
    p = _parallelism_for(scenario)
    if config_label == "ollama-default":
        return adapters.OllamaAdapter()
    if config_label == "ollama-tuned":
        return adapters.OllamaAdapter(num_parallel=p, context_length=(8192 if is_s4 else 2048))
    if config_label == "llamacpp-default":
        return adapters.LlamaCppAdapter()
    if config_label == "llamacpp-tuned":
        return adapters.LlamaCppAdapter(parallel=p, ctx=(8192 if is_s4 else p * 2048))
    raise ValueError(f"unknown config label: {config_label!r}")


# -- small helpers ----------------------------------------------------------------------------
def _now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _capture(argv, cwd=None, timeout=30, combined=False):
    """subprocess.run(argv, capture_output=True, text=True, timeout=timeout[, cwd=cwd]); returns
    stdout (or stdout+stderr when combined=True) on success, or str(exception) on failure. Every
    best-effort environment.json field goes through this -- a missing tool or odd machine state
    degrades one field, not the whole run."""
    try:
        r = subprocess.run(argv, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=cwd)
        return (r.stdout + r.stderr) if combined else r.stdout
    except Exception as exc:
        return str(exc)


# -- environment capture -------------------------------------------------------------------------
def capture_environment():
    """One best-effort snapshot of the machine/repo state this run happened under. Every
    subprocess-based field degrades to the exception string on failure rather than raising --
    a missing tool must not abort the matrix. power_protocol() adds the power-plan decision
    fields afterward (mutating this same dict) once it knows --skip-power-check.
    """
    ram_raw = _capture(
        ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]
    ).strip()
    try:
        ram_bytes = int(ram_raw)
    except ValueError:
        ram_bytes = ram_raw  # capture failed / unparseable -- keep the raw text for diagnostics

    env = {
        "timestamp_iso": _now_iso(),
        "argv": sys.argv,
        "git_rev": _capture(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).strip(),
        "platform": platform.platform(),
        "cpu": _capture(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip(),
        "ram_bytes": ram_bytes,
        "power_scheme_before": _capture(["powercfg", "/getactivescheme"]),
        "battery": _capture(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"]
        ).strip(),
        "ollama_version": _capture([str(adapters.OLLAMA_EXE), "--version"], combined=True).strip(),
        "llamacpp_version": _capture([str(adapters.LLAMA_SERVER_EXE), "--version"], combined=True).strip(),
    }
    try:
        env["model_file"] = adapters.MODEL_GGUF.name
        env["model_file_bytes"] = os.path.getsize(adapters.MODEL_GGUF)
    except Exception as exc:
        env["model_file"] = str(exc)
        env["model_file_bytes"] = None
    return env


def battery_is_discharging(battery_value):
    """Win32_Battery.BatteryStatus == 1 means 'discharging'. Empty output means no battery
    entity was reported at all -- desktop/no battery, not an error."""
    return str(battery_value).strip() == "1"


def power_protocol(env, skip_power_check):
    """Runs before any cell. Refuses to proceed on battery power (a throttled CPU would poison
    every timing number the matrix produces) regardless of --skip-power-check. Unless
    --skip-power-check, switches to the stock Windows High-performance plan when the active
    scheme isn't already it, registers an atexit handler to restore the original plan, and
    records every decision into `env` (mutated in place) so environment.json shows the power
    state the run happened under.
    """
    if battery_is_discharging(env.get("battery", "")):
        print("fatal: on battery power -- plug in and re-run")
        sys.exit(1)

    scheme_line = env.get("power_scheme_before") or ""
    env["power_switched"] = False
    env["power_scheme_restored_to"] = None

    if not skip_power_check and "High performance" not in scheme_line:
        m = GUID_RE.search(scheme_line)
        if m is None:
            env["power_warning"] = (
                "could not parse a GUID out of `powercfg /getactivescheme` output -- "
                "not switching power plan"
            )
            print(f"WARNING: {env['power_warning']}")
        else:
            original_guid = m.group(0)
            _capture(["powercfg", "/setactive", HIGH_PERF_GUID])
            env["power_switched"] = True
            env["power_scheme_restored_to"] = original_guid

            def _restore():
                subprocess.run(
                    ["powercfg", "/setactive", original_guid],
                    capture_output=True, text=True, timeout=30,
                )

            atexit.register(_restore)

    print("=" * 78)
    print("MANUAL CHECKLIST -- confirm before the matrix proceeds:")
    print("  - machine is plugged in")
    print("  - close browsers and other heavy apps")
    print("  - don't use this machine until the matrix finishes (results are timing-sensitive)")
    print("=" * 78)


# -- cell protocol --------------------------------------------------------------------------------
def run_cell(config_label, scenario, runs, timeout_s):
    """Run one (config, scenario) cell: construct+start the adapter, one discarded warmup +
    `runs` measured bench.run_scenario calls (each salted so cross-run prefix/prompt caching
    can't inflate later runs), stop the adapter, return the cell dict. On any exception the error
    is recorded and the cell dict is still returned (a failed cell is data, not a reason to lose
    the rest of the matrix) -- adapter.stop() always runs, in a finally, so a mid-cell exception
    can never leak a server onto the port.
    """
    cell = {
        "config": config_label,
        "scenario": scenario.name,
        "started_iso": _now_iso(),
        "finished_iso": None,
        "warmup": None,
        "runs": [],
        "adapter_log_tail": None,
        "ollama_show": None,
        "error": None,
    }
    adapter = None
    try:
        adapter = build_adapter(config_label, scenario)
        adapter.start()

        if isinstance(adapter, adapters.OllamaAdapter):
            adapters.ensure_ollama_model(adapter.model_name, adapter.modelfile_path)
            try:
                show = subprocess.run(
                    [str(adapters.OLLAMA_EXE), "show", adapter.model_name],
                    env=adapters._ollama_env(), capture_output=True, encoding="utf-8", errors="replace", timeout=30,
                )
                cell["ollama_show"] = show.stdout
            except Exception as exc:
                cell["ollama_show"] = str(exc)

        warmup_report = bench.run_scenario(
            adapter.base_url, scenario, model=adapter.model_name,
            timeout_s=timeout_s, salt="w", use_usage=True,
        )
        cell["warmup"] = {
            "aggregate_decode_tps": warmup_report["aggregate_decode_tps"],
            "errors": warmup_report["errors"],
        }

        for r in range(1, runs + 1):
            report = bench.run_scenario(
                adapter.base_url, scenario, model=adapter.model_name,
                timeout_s=timeout_s, salt=f"r{r}", use_usage=True,
            )
            cell["runs"].append(report)
            print(
                f"run r{r}: aggregate={report['aggregate_decode_tps']:.1f} tok/s "
                f"completed={report['completed']}/{len(report['per_client'])}",
                flush=True,
            )
    except Exception as exc:
        cell["error"] = str(exc)
        print(f"CELL FAILED: {config_label} x {scenario.name}: {exc}", flush=True)
    finally:
        if adapter is not None:
            cell["adapter_log_tail"] = adapter._log_tail(40)
            try:
                adapter.stop()
            except Exception as exc:
                if cell["error"] is None:
                    cell["error"] = str(exc)
                print(f"CELL FAILED: {config_label} x {scenario.name}: stop() failed: {exc}", flush=True)
        cell["finished_iso"] = _now_iso()
    return cell


# -- CLI / main -------------------------------------------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="measured runs per cell (default 3)")
    parser.add_argument(
        "--only-config", action="append", choices=CONFIG_LABELS, default=None,
        help="restrict to this config label; repeatable (default: all four)",
    )
    parser.add_argument(
        "--only-scenario", action="append", choices=SCENARIO_NAMES, default=None,
        help="restrict to this scenario name; repeatable (default: all eight)",
    )
    parser.add_argument("--fresh", action="store_true", help="re-run cells even if their raw file exists")
    parser.add_argument(
        "--timeout-s", type=int, default=900,
        help="per-request socket timeout passed to run_scenario (default 900)",
    )
    parser.add_argument(
        "--out-dir", default="bench/results",
        help="output directory, resolved relative to the repo root if not absolute (default bench/results)",
    )
    parser.add_argument(
        "--skip-power-check", action="store_true",
        help="skip the power-plan switch (env capture still records the active plan)",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    only_configs = args.only_config or list(CONFIG_LABELS)
    only_scenarios = args.only_scenario or list(SCENARIO_NAMES)
    ordered_configs = [c for c in CONFIG_LABELS if c in only_configs]
    ordered_scenarios = [s for s in SCENARIO_NAMES if s in only_scenarios]

    env = capture_environment()
    power_protocol(env, args.skip_power_check)
    with open(raw_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)

    cells = [(c, s) for c in ordered_configs for s in ordered_scenarios]
    total = len(cells)

    for k, (config_label, scenario_name) in enumerate(cells, start=1):
        scenario = SCENARIOS_BY_NAME[scenario_name]
        name = f"{config_label}--{scenario_name}"
        raw_path = raw_dir / f"{name}.json"

        print(f"=== [{k}/{total}] {config_label} × {scenario_name} ===", flush=True)

        if raw_path.exists() and not args.fresh:
            print(f"SKIP (exists): {name}", flush=True)
            continue

        cell = run_cell(config_label, scenario, args.runs, args.timeout_s)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(cell, f, indent=2)

        status = "OK" if cell["error"] is None else f"ERROR: {cell['error']}"
        print(f"=== [{k}/{total}] {config_label} × {scenario_name}: {status} ===", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
