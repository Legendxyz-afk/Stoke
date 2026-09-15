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
import time

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

# stoke is deliberately NOT in CONFIG_LABELS. Those four ARE the M0 baseline
# matrix and its published numbers, and adapters.build_configs already refuses to
# add a fifth for the same reason: it would silently change what "all configs"
# means for every existing caller. These are selectable BY NAME only, and the
# default set below is unchanged.
#
# ▶ WHY THIS EXISTS (2026-09-08). stoke's serving numbers on record come from
# `stoke bench --serve`, which drives the scheduler IN-PROCESS, while every
# referent number comes through HTTP + SSE. Comparing them is comparing two
# things measured by different code over different transports. This label makes
# stoke a cell of the SAME matrix, so the comparison changes exactly one thing.
EXTRA_CONFIG_LABELS = ["stoke-default", "stoke-tuned"]
ALL_CONFIG_LABELS = CONFIG_LABELS + EXTRA_CONFIG_LABELS

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
    if config_label == "stoke-default":
        return adapters.StokeAdapter()
    if config_label.startswith("stoke-tuned@"):
        # ▶ A/B VARIANTS OF THE TUNED CELL (2026-09-11): `stoke-tuned@spin=20000`,
        # `stoke-tuned@pin=0xFFF`, `stoke-tuned@spin=2000,pin=0xFFF`. Everything else about the
        # cell is the banked arm's configuration, so the variant changes exactly the one runtime
        # setting it names. Only `stoke-tuned` takes a suffix: the referents are not ours to tune
        # here, and stoke-default exists to be the untouched default.
        env, affinity, threads = {}, None, None   # threads None = the shipping default
        for opt in config_label.split("@", 1)[1].split(","):
            key, _, val = opt.partition("=")
            if key == "spin" and val.isdigit():
                env["STOKE_POOL_SPIN"] = val
            elif key == "pin" and val:
                affinity = int(val, 0)
            elif key == "threads" and val.isdigit():
                threads = int(val)
            elif key == "chunk" and val.isdigit():
                # 2026-09-15: the prefill chunk width (STOKE_SCHED_CHUNK, locked at 128;
                # 256 read -3.4% wall in the in-process phase screen). The server
                # inherits the env; the scheduler REFUSES a value that is not a positive
                # multiple of the KV block, so a typo cannot silently run the default.
                env["STOKE_SCHED_CHUNK"] = val
            else:
                raise ValueError(f"unknown stoke-tuned option {opt!r} in {config_label!r}")
        return adapters.StokeAdapter(seqs=p, ctx=(8192 if is_s4 else 704), threads=threads,
                                     env=env, affinity=affinity, label=config_label)
    if config_label == "stoke-tuned":
        # ⚠ ctx 704 is NOT a fresh tuning choice -- it is the value the banked
        # in-process arm used (`run_m5_bars.ps1`: ctx 704, seqs N), and this
        # cell was built to change exactly ONE thing against that arm: the
        # transport. Re-tuning ctx here would confound the comparison.
        #
        # ▶ THREADS FOLLOW THE SHIPPING DEFAULT (2026-09-11, `bc95312`): this
        # cell no longer pins --threads 8. The serve default for a concurrent
        # engine is now 10 (stoke.c STOKE_DEFAULT_THREADS_CONCURRENT; measured
        # +5.8% wall over 8 at N=8, docs/superpowers/gate-reports/
        # 2026-09-11-pool-ab-8.md), and "tuned" must not mean "slower than the
        # default". ⚠ Banks BEFORE that commit labelled `stoke-tuned` ran 8
        # threads; `stoke-tuned@threads=8` reproduces that cell exactly, and
        # every record since 2026-09-11 carries the count in `trace.threads`
        # when the trace is on (run_e2e_interleaved.py turns it on).
        #
        # ⚠ stoke's --ctx is PER SEQUENCE where llama-server's is TOTAL across
        # slots, so llamacpp-tuned's `p * 2048` is 2048 per slot against stoke's
        # 704. S2 needs 525 + 128 = 653 positions, so both are sufficient and the
        # surplus is slack neither engine can spend -- but it is an asymmetry and
        # it is written down rather than absorbed.
        #
        # ⚠ CORRECTED 2026-09-11: prefix dedup is ON on this path, not off.
        # `stoke serve` sets dedup_prefix = 1 (main.c cmd_serve, since M6 A3
        # `b24066c`) and exposes no switch, so this cell has ALWAYS run dedup
        # on -- which is the like-for-like configuration against llama-server's
        # default `--slot-prompt-similarity 0.10` prompt cache, and NOT the
        # banked in-process arm's explicit --no-dedup as the earlier comment
        # claimed. It is immaterial for S2: the per-client salt
        # (`[run r1 client N] ...`) sits at the FRONT of every user message, so
        # the shareable prefix is the template header plus ~8 tokens of 525,
        # and prefix dedup cannot adopt past the first differing token.
        return adapters.StokeAdapter(seqs=p, ctx=(8192 if is_s4 else 704), threads=None)
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


def _version_of(argv):
    """`<engine> --version` banners can carry extra noise around the version line — ollama with no
    daemon up prints 'Warning: could not connect...' before 'Warning: client version is X.Y.Z' —
    so keep only the first line that names a version AND contains a digit (llama-server's build
    number '10037' has no dots, hence digit not dotted-triple). Falls back to the raw capture when
    nothing matches, so a weird banner degrades to diagnostics instead of vanishing."""
    raw = _capture(argv, combined=True).strip()
    for line in raw.splitlines():
        if re.search(r"version", line, re.IGNORECASE) and re.search(r"\d", line):
            return line.strip()
    return raw


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
        "ollama_version": _version_of([str(adapters.OLLAMA_EXE), "--version"]),
        "llamacpp_version": _version_of([str(adapters.LLAMA_SERVER_EXE), "--version"]),
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


# -- quiesce gate ---------------------------------------------------------------------------------
# ▶ WHY THIS EXISTS (2026-09-08 → 11). The checklist above is a request, not a gate: this runner
# printed "close browsers" and then measured whatever the box was doing. On 2026-09-08 a block was
# voided by its own bracket at 18.4% because Chrome and VS Code were opened mid-block -- and the
# same afternoon `tests/tooling/run_m5_bars.ps1`, which FAILS a unit that does not pass its quiesce
# instrument, would have refused to start it. A printed checklist discharges the author's honesty
# and nothing else. This is that instrument, ported: the CPU-busy fraction from the raw perf
# counter (two samples 800 ms apart), quiet meaning <= QUIET_CPU_PCT on two consecutive reads,
# tried up to `tries` times 20 s apart. A sample that cannot be taken reads as NOT quiet, never as
# 0 -- a failed measurement that reads as "perfectly quiet" is the one error a gate must not make.
QUIET_CPU_PCT = 12.0
_CPU_BUSY_PS = (
    "$a = Get-CimInstance Win32_PerfRawData_PerfOS_Processor -Filter \"Name='_Total'\";"
    "Start-Sleep -Milliseconds 800;"
    "$b = Get-CimInstance Win32_PerfRawData_PerfOS_Processor -Filter \"Name='_Total'\";"
    "$dI = [double]$b.PercentIdleTime - [double]$a.PercentIdleTime;"
    "$dT = [double]$b.Timestamp_Sys100NS - [double]$a.Timestamp_Sys100NS;"
    "if ($dT -le 0) { 'unusable' } else { [math]::Round(100.0 * (1.0 - $dI / $dT), 1) };"
    "(Get-Process | Where-Object { $_.ProcessName -match 'ISLC|Intelligent' } | Measure-Object).Count"
)


def cpu_busy_pct():
    """(busy_pct, islc_running). busy_pct is 999.0 when the sample is unusable, so it fails the
    gate rather than passing it."""
    if os.name != "nt":
        return 999.0, False
    out = _capture(["powershell", "-NoProfile", "-Command", _CPU_BUSY_PS], timeout=30)
    lines = [ln.strip() for ln in str(out).splitlines() if ln.strip()]
    try:
        busy = float(lines[0])
        islc = int(lines[1]) > 0
    except (IndexError, ValueError):
        return 999.0, False
    return busy, islc


def quiesce_gate(tries=10, wait_s=20.0, quiet_pct=QUIET_CPU_PCT, say=print):
    """Returns {"passed": bool, "reads": [...], "islc": bool}. Passes on two consecutive reads at or
    under `quiet_pct`; ISLC running fails it outright (it purges the standby cache the model load
    depends on -- see run_m5_bars.ps1's -AllowISLC for why that is a deliberate override)."""
    reads, quiet = [], 0
    for t in range(tries):
        busy, islc = cpu_busy_pct()
        reads.append(busy)
        if islc:
            say("  !! quiesce gate: ISLC is running -- refusing (stop it, or measure knowing the cache is purged)")
            return {"passed": False, "reads": reads, "islc": True}
        if busy <= quiet_pct:
            quiet += 1
            if quiet >= 2:
                say(f"  quiesce gate PASSED on cpu {reads[-2]}%, {reads[-1]}% (<= {quiet_pct}% twice)")
                return {"passed": True, "reads": reads, "islc": False}
        else:
            quiet = 0
            say(f"  waiting: cpu {busy}% > {quiet_pct}% (try {t + 1}/{tries})")
        if t + 1 < tries:
            time.sleep(wait_s)
    say(f"  !! quiesce gate NOT MET after {tries} tries -- last reads {reads[-2:]}")
    return {"passed": False, "reads": reads, "islc": False}


# -- cell protocol --------------------------------------------------------------------------------
def run_cell(config_label, scenario, runs, timeout_s, keep_log=False):
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
        "proc": None,
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
            if keep_log:
                cell["adapter_log"] = list(adapter._log_lines)
            # The process is still alive here, so its CPU-seconds and faults describe the
            # runs above; after stop() there is nothing left to ask.
            if adapter.proc is not None:
                cell["proc"] = adapters.process_stats(adapter.proc.pid)
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
        "--only-config", action="append", choices=ALL_CONFIG_LABELS, default=None,
        help="restrict to this config label; repeatable (default: the four M0 baselines; stoke-* are selectable but never in the default set)",
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
    parser.add_argument(
        "--no-quiesce-gate", action="store_true",
        help="measure without the per-cell CPU-quiet gate (every cell then banks quiesce.passed=false "
             "and its rates are INDICATIVE, not gate-grade)",
    )
    parser.add_argument(
        "--quiesce-tries", type=int, default=10,
        help="attempts, 20 s apart, for the per-cell quiesce gate before the cell is REFUSED (default 10)",
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
    ordered_configs = [c for c in ALL_CONFIG_LABELS if c in only_configs]
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

        # The gate is a START condition per cell and is banked WITH the cell, so a reader of
        # any raw file sees whether its numbers were taken on a quiet box. A cell that cannot
        # pass it is refused and the matrix stops: a noisy box has a human on it, and the
        # remaining cells would be measured under conditions this one already rejected.
        if args.no_quiesce_gate:
            gate = {"passed": False, "reads": [], "skipped": True}
        else:
            gate = quiesce_gate(tries=args.quiesce_tries)
            if not gate["passed"]:
                print(f"REFUSED: {name} -- the box is not quiet; re-run when it is "
                      f"(cells already banked are kept)", flush=True)
                return 2

        cell = run_cell(config_label, scenario, args.runs, args.timeout_s)
        cell["quiesce"] = gate
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(cell, f, indent=2)

        status = "OK" if cell["error"] is None else f"ERROR: {cell['error']}"
        print(f"=== [{k}/{total}] {config_label} × {scenario_name}: {status} ===", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
