"""Bench harness core: HTTP/SSE timing measurement + scenario runners (M0 Task 1).

Stdlib only -- http.client, json, statistics, threading, time, plus the ordinary stdlib helpers
`random` (Poisson inter-arrival) and `urllib.parse` (splitting base_url). No httpx/requests/any
third-party install: the harness must run anywhere with zero deps.

Timing happens at SSE-chunk receipt (time.perf_counter() the instant a token-bearing `data: `
line is fully read off the wire) -- this is the load-bearing measurement code every published
number will depend on, which is why test_harness.py exists: to validate it against a mock server
with known-exact timing before it's ever pointed at a real engine.
"""
import http.client
import json
import random
import statistics
import threading
import time
import urllib.parse

import scenarios


def _split(base_url):
    parsed = urllib.parse.urlsplit(base_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname, port, (parsed.path or "")


def _is_token_chunk(parsed):
    """True iff this parsed SSE chunk represents one decoded token. Two shapes qualify:

    1. choices[0].delta.content is a NON-EMPTY string -- a visible text token.
    2. choices[0].finish_reason == "stop" -- the EOS token. When generation ends on a stop token,
       the engine decoded one final, real token (<|im_end|> etc) that produces no text; both
       ollama 0.30.11 and llama-server b10037 report it in usage.completion_tokens and announce
       it as an empty-content finish_reason="stop" chunk, whose arrival time IS that decode
       step's time. A finish_reason="length" chunk is NOT counted: hitting max_tokens decodes no
       extra token (the cap-hitting token already arrived as its own content chunk).

    Everything else is protocol bookkeeping, not a token: the role-announcement preamble
    ({"delta": {"role": "assistant", "content": ""}}), finish_reason="length" markers, and the
    choices-less usage trailer emitted under stream_options.include_usage. Empirical anchor
    (adapters.py --smoke, exact-match gated): content-only filtering undercounted BOTH engines by
    exactly 1 vs their own usage.completion_tokens on stop-finished generations; with rule 2
    added the counts match exactly. A chunk carrying both text and finish_reason="stop" would
    represent two tokens but count once -- neither engine does this; the smoke's exact gate
    would expose any engine that does."""
    if not isinstance(parsed, dict):
        return False
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    if choices[0].get("finish_reason") == "stop":
        return True
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return False
    content = delta.get("content")
    return isinstance(content, str) and content != ""


def timed_stream(base_url, model, messages, max_tokens, extra=None, capture_usage=False):
    """POST one streamed chat-completions request; return (ttft_s, itls_s, n_tokens, total_s), or
    (ttft_s, itls_s, n_tokens, total_s, usage) when capture_usage=True.

    ttft_s   -- time from just-before-send to the first TOKEN chunk fully received.
    itls_s   -- gaps between consecutive token-chunk arrivals (len == n_tokens - 1).
    n_tokens -- number of token chunks received before `data: [DONE]`.
    total_s  -- time from just-before-send to the last token chunk fully received.

    A "token chunk" is a `data:` line whose choices[0].delta.content is a non-empty string (see
    _is_token_chunk). Non-token bookkeeping lines -- role preambles, finish_reason-only chunks,
    usage trailers -- are received and parsed but never timestamped: real engines emit 1-2 of
    them per stream, and counting them would poison TTFT (preambles arrive pre-prefill) and every
    token count.

    `extra` is merged into the JSON payload's top level -- this is how mock_p_ms/mock_d_ms/mock_t
    ride along to mock_server.py; real adapters (Task 2+) simply pass extra=None.

    capture_usage (Task 2 addition; opt-in, default off so run_scenario's and test_harness.py's
    existing 4-tuple unpacking is untouched): when True, also returns the last non-empty `usage`
    object seen as a 5th tuple element (None if the engine never reports one). OpenAI-style
    streaming only emits `usage` on a final, choices-less chunk sent just before `data: [DONE]`,
    and only if the request carries `extra={"stream_options": {"include_usage": True}}`. That
    trailer has no delta.content, so it is never counted or timestamped -- usage capture does not
    perturb the measurement. adapters.py --smoke uses it to assert n_tokens ==
    usage.completion_tokens exactly, per engine.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if extra:
        payload.update(extra)
    body = json.dumps(payload).encode("utf-8")

    scheme, host, port, base_path = _split(base_url)
    conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=300)

    t0 = time.perf_counter()
    token_times = []
    usage = None
    try:
        conn.request(
            "POST",
            base_path + "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            resp.read()
            raise RuntimeError(f"{base_url}/v1/chat/completions returned HTTP {resp.status}")

        # HTTPResponse.readline() transparently undoes Transfer-Encoding: chunked, so this sees
        # the SSE payload as plain lines regardless of how the server chunked it on the wire.
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line or not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            t_arrival = time.perf_counter()
            try:
                parsed = json.loads(line[len("data: "):])
            except ValueError:
                parsed = None
            if _is_token_chunk(parsed):
                token_times.append(t_arrival)
            if capture_usage and isinstance(parsed, dict) and parsed.get("usage"):
                usage = parsed["usage"]
    finally:
        conn.close()

    if not token_times:
        raise RuntimeError("no tokens streamed")

    ttft = token_times[0] - t0
    itls = [b - a for a, b in zip(token_times, token_times[1:])]
    total = token_times[-1] - t0
    if capture_usage:
        return ttft, itls, len(token_times), total, usage
    return ttft, itls, len(token_times), total


def _percentiles(values_ms):
    """p50/p95 via statistics.quantiles(n=100); small-sample guard for len < 2."""
    if not values_ms:
        return {"p50": 0.0, "p95": 0.0}
    if len(values_ms) < 2:
        v = float(values_ms[0])
        return {"p50": v, "p95": v}
    q = statistics.quantiles(values_ms, n=100)
    return {"p50": q[49], "p95": q[94]}  # q[k-1] == k-th percentile


def run_scenario(base_url, scenario, model, extra=None, use_usage=False):
    """Run `scenario` against `base_url` and return a Report dict:
    {aggregate_decode_tps, ttft_ms: {p50,p95}, itl_ms: {p50,p95}, per_client: [...],
     wall_s, completed, errors}

    use_usage=False (Task 1 contract, mock-tested): per-client n_tokens = number of token chunks
    received (bench._is_token_chunk). use_usage=True (real-engine measurement mode): the request
    additionally carries stream_options.include_usage, and n_tokens comes from the engine's own
    usage.completion_tokens -- authoritative even when the engine coalesces several tokens into
    one SSE chunk, which ollama 0.30.11 demonstrably does on longer generations (probe_sse.py
    showed chunk-anatomy parity on short ones; the smoke's exact-count gate caught a 31-chunk /
    32-token generation). Chunk arrivals still drive ALL timing (TTFT, ITL, span) in both modes;
    ITL is therefore "inter-arrival latency" -- a coalesced chunk contributes one gap for two
    tokens. Each per_client record also carries n_arrivals (counted token chunks) and
    usage_missing so reports can quantify coalescing; if an engine omits usage, n_tokens falls
    back to n_arrivals with usage_missing=True.

    closed_loop: scenario.n_clients threads all start immediately, each firing
        scenario.requests_per_client requests back-to-back. Task 1 scope is exactly
        requests_per_client == 1 (one request per client per run); the loop/duration knob for
        >1 is a Task 2 extension point -- guarded below rather than silently mishandled.
    poisson: a single spawner thread starts one client thread per scenario.n_clients, sleeping
        an exponential(scenario.arrival_rate) inter-arrival gap before each spawn.

    aggregate_decode_tps = [sum over clients of (n_tokens - 1)] /
                           [max over clients of last-token-wallclock
                            - min over clients of first-token-wallclock]
    i.e. total decode-only tokens (excluding each client's own TTFT/prefill token) produced
    across the whole run, divided by the global wall-clock span from the earliest first-token to
    the latest last-token -- a concurrency-adjusted aggregate, not a per-client average.
    """
    if scenario.requests_per_client != 1:
        raise NotImplementedError(
            "requests_per_client > 1 is a Task 2 extension point; "
            "Task 1 scope is exactly one request per client per run"
        )

    per_client = [None] * scenario.n_clients
    errors = []
    errors_lock = threading.Lock()

    def worker(client_id):
        messages = scenarios.build_messages(scenario, client_id)
        t_req_start = time.perf_counter()
        try:
            if use_usage:
                req_extra = dict(extra) if extra else {}
                req_extra["stream_options"] = {"include_usage": True}
                ttft, itls, n_arrivals, total_s, usage = timed_stream(
                    base_url, model, messages, scenario.max_tokens,
                    extra=req_extra, capture_usage=True,
                )
                usage_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
                usage_missing = not isinstance(usage_tokens, int)
                n_tokens = n_arrivals if usage_missing else usage_tokens
            else:
                ttft, itls, n_arrivals, total_s = timed_stream(
                    base_url, model, messages, scenario.max_tokens, extra=extra
                )
                n_tokens, usage_missing = n_arrivals, False
            per_client[client_id] = {
                "client_id": client_id,
                "n_tokens": n_tokens,
                "n_arrivals": n_arrivals,
                "usage_missing": usage_missing,
                "ttft_ms": ttft * 1000.0,
                "itl_ms": [x * 1000.0 for x in itls],
                "first_token_abs": t_req_start + ttft,
                "last_token_abs": t_req_start + total_s,
                "error": None,
            }
        except Exception as exc:
            with errors_lock:
                errors.append(f"client {client_id}: {exc}")
            per_client[client_id] = {
                "client_id": client_id,
                "n_tokens": 0,
                "n_arrivals": 0,
                "usage_missing": False,
                "ttft_ms": None,
                "itl_ms": [],
                "first_token_abs": None,
                "last_token_abs": None,
                "error": str(exc),
            }

    wall_t0 = time.perf_counter()
    threads = []

    if scenario.mode == "closed_loop":
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(scenario.n_clients)]
        for th in threads:
            th.start()

    elif scenario.mode == "poisson":
        if not scenario.arrival_rate:
            raise ValueError("poisson mode requires scenario.arrival_rate")
        spawned = []

        def spawner():
            rng = random.Random()
            for i in range(scenario.n_clients):
                time.sleep(rng.expovariate(scenario.arrival_rate))
                th = threading.Thread(target=worker, args=(i,))
                spawned.append(th)
                th.start()

        sp = threading.Thread(target=spawner)
        sp.start()
        sp.join()
        threads = spawned

    else:
        raise ValueError(f"unknown scenario mode: {scenario.mode!r}")

    for th in threads:
        th.join()

    wall_s = time.perf_counter() - wall_t0
    completed = sum(1 for r in per_client if r and r["error"] is None)
    ok_records = [r for r in per_client if r and r["error"] is None]

    ttft_values = [r["ttft_ms"] for r in ok_records]
    itl_values = [v for r in ok_records for v in r["itl_ms"]]

    if ok_records:
        span = max(r["last_token_abs"] for r in ok_records) - min(r["first_token_abs"] for r in ok_records)
        total_decode_tokens = sum(r["n_tokens"] - 1 for r in ok_records)
        aggregate_decode_tps = (total_decode_tokens / span) if span > 0 else 0.0
    else:
        aggregate_decode_tps = 0.0

    return {
        "aggregate_decode_tps": aggregate_decode_tps,
        "ttft_ms": _percentiles(ttft_values),
        "itl_ms": _percentiles(itl_values),
        "per_client": per_client,
        "wall_s": wall_s,
        "completed": completed,
        "errors": errors,
    }
