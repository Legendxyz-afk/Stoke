#!/usr/bin/env python3
"""Mechanical verification of the bench harness's timing measurement code (M0 Task 1, Step 3).

The measurement code in bench.py is the load-bearing code every published benchmark number will
depend on -- a timing bug here would poison every number this project ever publishes. So before
bench.py ever talks to a real engine, it's driven against mock_server.py, whose token emission
timing is exactly known (perf_counter-scheduled, not naive sleep accumulation), and the harness's
own measurements are checked against a hand-derived expectation.

Hand-derived aggregate_decode_tps expectation for N=4, mock_p_ms=200, mock_d_ms=50, mock_t=20:
    Each client streams 20 tokens; the 1st is the TTFT/prefill token, the remaining 19 are
    decode-only tokens spaced d_ms=50ms apart -> 19 * 0.050s = 0.95s of decode time per client.
    The 4 client threads are independent (mock_server schedules each request's chunks off that
    request's own perf_counter t0, in its own handler thread), so their [first-token, last-token]
    windows overlap almost exactly -- the global window from min(first-token time) to
    max(last-token time) across all 4 clients is ~0.95s, same as any single client's window.
    aggregate_decode_tps = sum(n_tokens - 1 over clients) / global_window
                         = (4 * 19) / 0.95
                         ~= 80.0 tok/s
"""
import pathlib
import queue
import re
import subprocess
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bench import run_scenario
from scenarios import Scenario

MOCK_SERVER = pathlib.Path(__file__).resolve().parent / "mock_server.py"

failures = []


def check(label, cond, measured):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}: {measured}")
    if not cond:
        failures.append(label)


def start_mock_server():
    proc = subprocess.Popen(
        [sys.executable, str(MOCK_SERVER), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    line_q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=lambda: line_q.put(proc.stdout.readline()), daemon=True).start()

    try:
        line = line_q.get(timeout=10)
    except queue.Empty:
        proc.kill()
        raise RuntimeError("mock_server.py did not print a startup banner within 10s")

    m = re.search(r":(\d+)\s*$", line.strip())
    if not m:
        proc.terminate()
        raise RuntimeError(f"mock_server.py startup banner did not contain a port; got: {line!r}")
    return proc, int(m.group(1))


def main():
    proc, port = start_mock_server()
    base_url = f"http://127.0.0.1:{port}"

    try:
        scenario = Scenario(
            name="test-closed-4",
            n_clients=4,
            prompt_tokens=64,
            max_tokens=20,
            shared_prefix_tokens=0,
            mode="closed_loop",
        )
        report = run_scenario(
            base_url,
            scenario,
            model="mock-model",
            extra={"mock_p_ms": 200, "mock_d_ms": 50, "mock_t": 20},
        )

        token_counts = [c["n_tokens"] for c in report["per_client"]]
        check("every client streamed exactly 20 tokens", all(n == 20 for n in token_counts), token_counts)

        ttft_p50 = report["ttft_ms"]["p50"]
        check("TTFT p50 in [190, 280] ms", 190 <= ttft_p50 <= 280, f"{ttft_p50:.2f} ms")

        itl_p50 = report["itl_ms"]["p50"]
        check("ITL p50 in [45, 65] ms", 45 <= itl_p50 <= 65, f"{itl_p50:.2f} ms")

        expected_tps = 4 * 19 / 0.95  # == 80.0 tok/s, see module docstring
        measured_tps = report["aggregate_decode_tps"]
        lo, hi = expected_tps * 0.9, expected_tps * 1.1
        check(
            f"aggregate_decode_tps within +/-10% of {expected_tps:.2f} tok/s",
            lo <= measured_tps <= hi,
            f"{measured_tps:.2f} tok/s (allowed [{lo:.2f}, {hi:.2f}])",
        )

        check("no client errors", report["errors"] == [], report["errors"])

        # Non-token bookkeeping chunks (role preamble, finish_reason-only, usage trailer) must be
        # invisible to the measurement -- real engines emit them (smoke finding: llama-server +2
        # raw chunks, ollama +1). The mock's preamble is sent at t~0, BEFORE its prefill sleep, so
        # if bench._is_token_chunk ever counts it, TTFT p50 collapses toward 0 ms and the bound
        # below catches it; a counted trailer breaks the exact-20 count instead.
        report2 = run_scenario(
            base_url,
            scenario,
            model="mock-model",
            extra={"mock_p_ms": 200, "mock_d_ms": 50, "mock_t": 20,
                   "mock_preamble": 1, "mock_trailer": 2},
        )
        token_counts2 = [c["n_tokens"] for c in report2["per_client"]]
        check(
            "preamble/trailer chunks not counted as tokens (still exactly 20)",
            all(n == 20 for n in token_counts2),
            token_counts2,
        )
        ttft2_p50 = report2["ttft_ms"]["p50"]
        check("TTFT p50 ignores the t~0 preamble chunk", 190 <= ttft2_p50 <= 280, f"{ttft2_p50:.2f} ms")
        check("no client errors (preamble/trailer run)", report2["errors"] == [], report2["errors"])

        # The EOS path: a finish_reason="stop" chunk is one real decoded token (both real engines
        # count it in usage.completion_tokens) and MUST be counted; it arrives one d_ms step after
        # the last text chunk, so per-client decode tokens become 20 over a 20*d window -- the
        # aggregate expectation stays exactly 80 tok/s.
        report3 = run_scenario(
            base_url,
            scenario,
            model="mock-model",
            extra={"mock_p_ms": 200, "mock_d_ms": 50, "mock_t": 20,
                   "mock_preamble": 1, "mock_finish_stop": 1, "mock_trailer": 2},
        )
        token_counts3 = [c["n_tokens"] for c in report3["per_client"]]
        check(
            "EOS finish_reason=stop chunk counted as a token (exactly 21)",
            all(n == 21 for n in token_counts3),
            token_counts3,
        )
        measured_tps3 = report3["aggregate_decode_tps"]
        check(
            f"aggregate_decode_tps with EOS step within +/-10% of {expected_tps:.2f} tok/s",
            lo <= measured_tps3 <= hi,
            f"{measured_tps3:.2f} tok/s (allowed [{lo:.2f}, {hi:.2f}])",
        )
        check("no client errors (EOS run)", report3["errors"] == [], report3["errors"])

        # Measurement mode for real engines: n_tokens taken from the engine's own
        # usage.completion_tokens (authoritative under chunk coalescing -- observed on ollama),
        # arrivals still driving all timing. On the well-behaved mock (one chunk per token, usage
        # trailer reporting 21 = 20 text + EOS), both counts must agree and every number must
        # match the chunk-counted run.
        report4 = run_scenario(
            base_url,
            scenario,
            model="mock-model",
            extra={"mock_p_ms": 200, "mock_d_ms": 50, "mock_t": 20,
                   "mock_preamble": 1, "mock_finish_stop": 1, "mock_trailer": 2},
            use_usage=True,
        )
        token_counts4 = [c["n_tokens"] for c in report4["per_client"]]
        arrival_counts4 = [c["n_arrivals"] for c in report4["per_client"]]
        check(
            "use_usage mode: n_tokens from usage trailer (exactly 21)",
            all(n == 21 for n in token_counts4),
            token_counts4,
        )
        check(
            "use_usage mode: arrivals agree with usage on non-coalescing mock",
            arrival_counts4 == token_counts4,
            arrival_counts4,
        )
        check(
            "use_usage mode: no client marked usage_missing",
            not any(c["usage_missing"] for c in report4["per_client"]),
            [c["usage_missing"] for c in report4["per_client"]],
        )
        measured_tps4 = report4["aggregate_decode_tps"]
        check(
            f"use_usage mode: aggregate within +/-10% of {expected_tps:.2f} tok/s",
            lo <= measured_tps4 <= hi,
            f"{measured_tps4:.2f} tok/s (allowed [{lo:.2f}, {hi:.2f}])",
        )
        check("no client errors (use_usage run)", report4["errors"] == [], report4["errors"])

        print()
        print(f"completed={report['completed']}/{scenario.n_clients} wall_s={report['wall_s']:.3f}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if failures:
        print(f"\n{len(failures)} assertion(s) FAILED: {failures}")
        sys.exit(1)

    print("\nALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
