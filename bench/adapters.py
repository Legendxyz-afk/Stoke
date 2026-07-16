#!/usr/bin/env python3
"""Engine adapters -- four baseline serving configs against the SAME GGUF (M0 Task 2).

Two engines, two configs each, all four pointed at models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf so
the M0 baseline matrix (Task 3) is an apples-to-apples same-weights comparison:

    ollama-default   -- `ollama serve`, no OLLAMA_NUM_PARALLEL (engine default)
    ollama-tuned     -- same + OLLAMA_NUM_PARALLEL=<n>
    llamacpp-default -- `llama-server -m <gguf> --port 8089 --host 127.0.0.1`, no other flags
    llamacpp-tuned   -- same + `--parallel <n> -c <n*2048>`

CRITICAL INVARIANT: only one engine may occupy its port at a time. Adapter.stop() is written to
be trustworthy (tree-kill + block-until-port-verified-free) specifically so callers -- this
module's --smoke driver, and the Task 3 matrix runner after it -- can rely on start()/use/stop()
as a hard sequential boundary. Never start a second adapter before the previous one's stop() has
returned successfully.

All constants below (exe paths, flag names, endpoint paths, env var names) were probed against
the real binaries at dispatch time -- see the paired verification notes in the Task 2 commit --
never recalled from memory, per the plan's Global Constraints #6/#20 (docs/superpowers/plans/
2026-07-16-phase1-implementation.md).
"""
import argparse
import collections
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench  # noqa: E402  (path insert above must run first)

# -- probed constants (dispatch-time, this machine) --------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OLLAMA_EXE = pathlib.Path(r"C:\Users\Kabir\AppData\Local\Programs\Ollama\ollama.exe")
LLAMA_SERVER_EXE = REPO_ROOT / "bench" / "vendor" / "llama-b10037" / "llama-server.exe"
MODEL_GGUF = REPO_ROOT / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MODELFILE_PATH = pathlib.Path(__file__).resolve().parent / "Modelfile.qwen3-4b"

OLLAMA_MODEL_NAME = "stoke-bench-qwen3"     # registered name; used as the `model` field in requests
LLAMACPP_MODEL_NAME = "qwen3-4b"             # llama-server serves one model; field value is arbitrary

OLLAMA_HOST_ADDR = "127.0.0.1:11434"        # forced explicitly so ambient OLLAMA_HOST can't drift us


# -- small stdlib-only helpers ------------------------------------------------------------------
def _port_free(host, port, timeout=0.3):
    """True iff nothing accepts a TCP connection on (host, port) right now."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) != 0
    finally:
        s.close()


def _whats_on_port(port):
    """Best-effort description of whatever process is LISTENING on `port`, via netstat + tasklist
    (Windows-only, stdlib subprocess -- no psutil/third-party dependency)."""
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception as exc:
        return f"(could not run netstat to identify the occupant: {exc})"

    pids = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
            pids.add(parts[-1])
    if not pids:
        return f"(netstat shows no LISTENING entry on port {port} -- may be TIME_WAIT or non-TCP)"

    descs = []
    for pid in pids:
        try:
            t = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            descs.append(f"pid {pid}: {t or '(tasklist returned nothing)'}")
        except Exception as exc:
            descs.append(f"pid {pid}: (tasklist lookup failed: {exc})")
    return "; ".join(descs)


def _probe_status(url, timeout=2.0):
    """GET url; return the HTTP status code, or None if the connection was refused/timed out (ie
    nothing is listening yet). A non-2xx HTTP response still means *something* answered, so HTTP
    error codes are returned rather than swallowed."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return None


def _ollama_env(extra=None):
    """Copy of the ambient environment with OLLAMA_HOST pinned to our expected address -- both the
    `ollama serve` subprocess and the `ollama create`/`ollama list` CLI calls in
    ensure_ollama_model() must agree on this or the CLI calls silently talk to the wrong server."""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = OLLAMA_HOST_ADDR
    if extra:
        env.update(extra)
    return env


# -- Adapter base: start()/stop()/label contract ------------------------------------------------
class Adapter:
    """Common lifecycle for a serving-engine subprocess.

    Subclasses implement `_spawn()` (return a started subprocess.Popen with stdout+stderr merged
    into one text pipe) and `_healthy()` (return True iff ready to serve inference).

    start() spawns, drains the child's combined stdout/stderr into a bounded ring buffer on a
    daemon thread (so a chatty child can never deadlock us by filling the pipe while we poll), and
    polls `_healthy()` until it returns True or `timeout_s` elapses. On timeout OR an early exit,
    it raises a loud RuntimeError carrying the captured output tail.

    stop() tree-kills the process (Windows: `taskkill /T /F`, since `ollama serve` forks a
    model-runner child that plain terminate() would orphan -- confirmed the port stays bound if
    only the parent dies otherwise is not the case here, but tree-killing is the correct general
    fix) and then BLOCKS until a fresh connect attempt to the port is refused. It raises loudly if
    either the process won't die or the port won't free up -- callers depend on stop() actually
    meaning "safe to start the next engine."
    """

    def __init__(self, label, base_url, host, port):
        self.label = label
        self.base_url = base_url
        self.host = host
        self.port = port
        self.proc = None
        self._log_lines = collections.deque(maxlen=500)

    # -- subclass contract --
    def _spawn(self):
        raise NotImplementedError

    def _healthy(self):
        raise NotImplementedError

    # -- shared machinery --
    def _drain_log(self):
        try:
            for line in self.proc.stdout:
                self._log_lines.append(line.rstrip("\n"))
        except Exception:
            pass  # pipe torn down under us during shutdown -- nothing useful to do

    def _log_tail(self, n=60):
        return "\n".join(list(self._log_lines)[-n:]) or "(no output captured)"

    def start(self, timeout_s=180):
        if not _port_free(self.host, self.port):
            who = _whats_on_port(self.port)
            raise RuntimeError(
                f"fatal: cannot start {self.label} -- port {self.port} is already occupied -- "
                f"{who} -- free the port (stop the offending process) before benchmarking"
            )

        self._log_lines.clear()
        try:
            self.proc = self._spawn()
        except FileNotFoundError as exc:
            raise RuntimeError(f"fatal: {self.label} -- executable not found: {exc}") from exc

        threading.Thread(target=self._drain_log, daemon=True).start()

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            code = self.proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"fatal: {self.label} exited early (code {code}) before becoming healthy -- "
                    f"captured output tail:\n{self._log_tail()}"
                )
            try:
                if self._healthy():
                    return self.base_url
            except Exception:
                pass  # not up yet -- keep polling
            time.sleep(1.0)

        # timed out: don't leak a half-started process holding the port
        self._tree_kill()
        raise RuntimeError(
            f"fatal: {self.label} did not become healthy within {timeout_s}s -- "
            f"captured output tail:\n{self._log_tail()}"
        )

    def _tree_kill(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                capture_output=True, text=True, timeout=20,
            )
        else:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if sys.platform != "win32":
                self.proc.kill()
            self.proc.wait(timeout=15)

    def stop(self, timeout_s=30):
        if self.proc is None:
            return  # never started (or already stopped) -- nothing to do
        self._tree_kill()

        if self.proc.poll() is None:
            raise RuntimeError(f"fatal: {self.label} (pid {self.proc.pid}) would not exit after tree-kill")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if _port_free(self.host, self.port):
                self.proc = None
                return
            time.sleep(0.5)

        who = _whats_on_port(self.port)
        raise RuntimeError(
            f"fatal: {self.label} process exited but port {self.port} is STILL occupied "
            f"{timeout_s}s later -- {who} -- refusing to declare stop() successful"
        )


# -- concrete adapters ---------------------------------------------------------------------------
class OllamaAdapter(Adapter):
    HOST = "127.0.0.1"
    PORT = 11434

    def __init__(self, num_parallel=None, context_length=None):
        label = "ollama-tuned" if num_parallel else "ollama-default"
        # base_url is the server ROOT: bench.timed_stream() appends /v1/chat/completions itself
        # (contract locked by the committed, mock-tested Task 1 harness).
        super().__init__(label, base_url=f"http://{self.HOST}:{self.PORT}", host=self.HOST, port=self.PORT)
        self.num_parallel = num_parallel
        self.context_length = context_length
        self.model_name = OLLAMA_MODEL_NAME

    def _spawn(self):
        extra_env = {"OLLAMA_KEEP_ALIVE": "1h", "OLLAMA_MAX_LOADED_MODELS": "1"}
        if self.num_parallel:
            extra_env["OLLAMA_NUM_PARALLEL"] = str(self.num_parallel)
        if self.context_length:
            extra_env["OLLAMA_CONTEXT_LENGTH"] = str(self.context_length)
        return subprocess.Popen(
            [str(OLLAMA_EXE), "serve"],
            env=_ollama_env(extra_env),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

    def _healthy(self):
        # Task spec: "Healthcheck: GET /v1/models == 200". Note this confirms the *server* is up,
        # not that the model is loaded -- ollama lazy-loads on first chat request (the task's
        # 180s-timeout / 300s-conn-timeout budget accounts for that first-request load cost).
        return _probe_status(self.base_url + "/v1/models") == 200


class LlamaCppAdapter(Adapter):
    HOST = "127.0.0.1"
    PORT = 8089

    def __init__(self, parallel=None, ctx=None):
        label = "llamacpp-tuned" if parallel else "llamacpp-default"
        self.root_url = f"http://{self.HOST}:{self.PORT}"
        # base_url is the server ROOT: bench.timed_stream() appends /v1/chat/completions itself.
        super().__init__(label, base_url=self.root_url, host=self.HOST, port=self.PORT)
        self.parallel = parallel
        self.ctx = ctx
        self.model_name = LLAMACPP_MODEL_NAME

    def _spawn(self):
        argv = [
            str(LLAMA_SERVER_EXE), "-m", str(MODEL_GGUF),
            "--port", str(self.PORT), "--host", self.HOST,
        ]
        # explicit ctx overrides the derived one; current behavior when ctx is omitted is
        # unchanged (parallel alone still derives -c from parallel * 2048).
        if self.ctx:
            argv += ["-c", str(self.ctx)]
        if self.parallel:
            argv += ["--parallel", str(self.parallel)]
        if self.parallel and not self.ctx:
            argv += ["-c", str(self.parallel * 2048)]
        return subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

    def _healthy(self):
        # Confirmed at dispatch (manual trial): /health returns 503 while the model is loading and
        # 200 {"status":"ok"} once ready -- a non-200 here means "not ready", NOT "endpoint
        # missing", so we keep polling /health rather than falling back on a mere non-200. Only a
        # 404 (this build genuinely lacks /health) triggers the /v1/models fallback the task asks
        # for; a connection refusal (None) means the process isn't listening yet.
        status = _probe_status(self.root_url + "/health")
        if status == 200:
            return True
        if status == 404:
            return _probe_status(self.root_url + "/v1/models") == 200
        return False


def build_configs(tuned_n=4, only=None):
    """The four M0 baseline configs, same GGUF, tuned variants parallel=tuned_n.

    only=None -> all four; only="ollama"/"llamacpp" -> just that engine's pair. The filter exists
    so smoke/bench runs can exercise llama-server without ever spawning `ollama serve` (launching
    Ollama loads the model into RAM and is permission-gated on this box)."""
    configs = [
        OllamaAdapter(num_parallel=None),
        OllamaAdapter(num_parallel=tuned_n),
        LlamaCppAdapter(parallel=None),
        LlamaCppAdapter(parallel=tuned_n),
    ]
    if only == "ollama":
        return [c for c in configs if isinstance(c, OllamaAdapter)]
    if only == "llamacpp":
        return [c for c in configs if isinstance(c, LlamaCppAdapter)]
    if only is not None:
        raise ValueError(f"unknown engine filter: {only!r} (expected 'ollama' or 'llamacpp')")
    return configs


# -- one-time setup: register the shared GGUF with Ollama ---------------------------------------
def ensure_ollama_model():
    """Register models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf with Ollama as `stoke-bench-qwen3`, via
    `ollama create -f bench/Modelfile.qwen3-4b`, so ollama-default/ollama-tuned serve the exact
    same weights llama-server does (same-GGUF fairness is the whole point of this bench).

    Requires `ollama serve` to already be running -- `ollama create`/`ollama list` are CLI clients
    that talk to the local server, they do nothing standalone. Idempotent: skips `ollama create`
    if `ollama list` already shows the model (so repeated --smoke runs don't re-copy 2.5GB).

    Modelfile path handling: confirmed at dispatch that `ollama create` accepts the UNQUOTED
    absolute Windows path with spaces (`FROM M:\\New Drive\\...\\Qwen3-...gguf`) exactly as written
    in bench/Modelfile.qwen3-4b -- no quoting workaround was needed. Kept as a fallback below only
    in case a future Ollama version regresses this.
    """
    def _list_stdout():
        r = subprocess.run(
            [str(OLLAMA_EXE), "list"], env=_ollama_env(), capture_output=True, text=True, timeout=30,
        )
        return r.stdout if r.returncode == 0 else ""

    if OLLAMA_MODEL_NAME in _list_stdout():
        return  # already registered (this run or a previous one)

    create = subprocess.run(
        [str(OLLAMA_EXE), "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE_PATH)],
        env=_ollama_env(), capture_output=True, text=True, timeout=600,  # first create hashes+copies 2.5GB
    )
    if create.returncode != 0:
        # confirmed-working form failed (env drift, moved file, etc) -- try a quoted-path Modelfile
        # once before giving up, and flag loudly which form was needed.
        quoted = f'FROM "{MODEL_GGUF}"\n'
        MODELFILE_PATH.write_text(quoted, encoding="utf-8")
        retry = subprocess.run(
            [str(OLLAMA_EXE), "create", OLLAMA_MODEL_NAME, "-f", str(MODELFILE_PATH)],
            env=_ollama_env(), capture_output=True, text=True, timeout=600,
        )
        if retry.returncode != 0:
            raise RuntimeError(
                f"fatal: `ollama create {OLLAMA_MODEL_NAME}` failed with BOTH unquoted and quoted "
                f"Modelfile paths.\nunquoted stderr:\n{create.stderr}\nquoted stderr:\n{retry.stderr}"
            )
        print(f"NOTE: ollama create needed the QUOTED Modelfile path form (unquoted failed) -- "
              f"bench/Modelfile.qwen3-4b has been rewritten to the quoted form.", file=sys.stderr)

    verify = _list_stdout()
    if OLLAMA_MODEL_NAME not in verify:
        raise RuntimeError(
            f"fatal: `ollama create` reported success but `ollama list` does not show "
            f"{OLLAMA_MODEL_NAME} -- stdout:\n{verify}"
        )


# -- mechanical verification: --smoke ------------------------------------------------------------
SMOKE_MESSAGES = [{"role": "user", "content": "Count from 1 to 10."}]
SMOKE_MAX_TOKENS = 32


def run_smoke(tuned_n=4, only=None):
    """Strictly sequentially, for each selected config: start(), one timed_stream() request,
    print label/ttft_ms/n_token_chunks/total_s/usage, then validate the token accounting that
    every published number depends on. FAILS if: the engine omits usage.completion_tokens
    (measurement runs count via usage -- run_scenario use_usage=True), or the filter counts MORE
    chunks than the engine decoded tokens (bench._is_token_chunk misclassifying bookkeeping as
    tokens). Counting FEWER is a printed NOTE, not a failure: engines may coalesce several
    tokens into one SSE chunk (observed on ollama 0.30.11), which is exactly why measurement
    counts via usage while chunk arrivals drive timing. Then stop() and confirm the port freed.
    Returns True iff every config streamed >0 token chunks with sane accounting and every stop()
    succeeded.
    """
    configs = build_configs(tuned_n, only=only)
    print("=" * 88)
    print(f"bench/adapters.py --smoke  (tuned configs at n={tuned_n}, "
          f"engines={only or 'all'}, {len(configs)} configs)")
    print("strictly sequential: one engine up at a time, start -> 1 request -> stop, per config")
    print("=" * 88)

    results = []
    overall_ok = True

    for adapter in configs:
        print(f"\n--- {adapter.label} ---", flush=True)
        entry = {"label": adapter.label}
        started = False
        try:
            base_url = adapter.start()
            started = True
            print(f"{adapter.label}: start() OK, base_url={base_url}", flush=True)

            if isinstance(adapter, OllamaAdapter):
                ensure_ollama_model()
                print(f"{adapter.label}: ollama model '{OLLAMA_MODEL_NAME}' confirmed via `ollama list`", flush=True)

            ttft, itls, n_tokens, total_s, usage = bench.timed_stream(
                adapter.base_url,
                adapter.model_name,
                SMOKE_MESSAGES,
                SMOKE_MAX_TOKENS,
                extra={"stream_options": {"include_usage": True}},
                capture_usage=True,
            )

            usage_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            usage_display = json.dumps(usage) if usage is not None else "<not reported by engine>"

            print(
                f"{adapter.label}: ttft_ms={ttft * 1000:.1f} n_token_chunks={n_tokens} "
                f"total_s={total_s:.3f} usage={usage_display}",
                flush=True,
            )

            entry.update(
                ok=True, n_token_chunks=n_tokens, ttft_ms=ttft * 1000,
                total_s=total_s, usage_completion_tokens=usage_tokens,
            )

            if usage_tokens is None:
                print(
                    f"FATAL [{adapter.label}]: engine did not report usage.completion_tokens "
                    f"under stream_options.include_usage -- real measurement runs "
                    f"(run_scenario use_usage=True) depend on it for authoritative token "
                    f"counts. Investigate before benchmarking this engine.",
                    flush=True,
                )
                overall_ok = False
            else:
                entry["count_vs_usage_diff"] = n_tokens - usage_tokens
                if n_tokens > usage_tokens:
                    print(
                        f"FATAL [{adapter.label}]: filter counted MORE token chunks ({n_tokens}) "
                        f"than the engine reports decoding ({usage_tokens}) -- "
                        f"bench._is_token_chunk is misclassifying a bookkeeping chunk as a "
                        f"token for this engine. Fix the filter before benchmarking.",
                        flush=True,
                    )
                    overall_ok = False
                elif n_tokens < usage_tokens:
                    print(
                        f"NOTE [{adapter.label}]: {usage_tokens - n_tokens} fewer token-chunk "
                        f"arrivals ({n_tokens}) than decoded tokens ({usage_tokens}) -- engine "
                        f"coalesced tokens into shared chunks. Expected occasionally (observed "
                        f"on ollama); measurement runs count via usage, so numbers stay exact.",
                        flush=True,
                    )

        except Exception as exc:
            print(f"FATAL [{adapter.label}]: {exc}", flush=True)
            entry.update(ok=False, error=str(exc))
            overall_ok = False
        finally:
            try:
                adapter.stop()
                if started:
                    print(f"{adapter.label}: stop() OK -- process exited, port {adapter.port} confirmed free", flush=True)
                else:
                    print(f"{adapter.label}: cleanup after failed start -- port {adapter.port} confirmed free", flush=True)
            except Exception as exc:
                print(f"FATAL [{adapter.label}]: stop() FAILED: {exc}", flush=True)
                entry["stop_error"] = str(exc)
                overall_ok = False

        results.append(entry)

    print("\n" + "=" * 88)
    print("SMOKE SUMMARY")
    print("=" * 88)
    for r in results:
        print(r)

    any_zero_chunks = any(r.get("ok") and not r.get("n_token_chunks") for r in results)
    overall_ok = overall_ok and not any_zero_chunks and len(results) == len(configs)

    print()
    if overall_ok:
        print(f"ALL {len(configs)} CONFIGS OK -- every config streamed >0 token chunks, "
              f"token accounting sane (usage reported, no filter overcounts), no stop() failures")
    else:
        print("SMOKE FAILED -- see FATAL/WARNING lines above")
    return overall_ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the strictly-sequential smoke test")
    parser.add_argument("--tuned-n", type=int, default=4, help="parallelism N for *-tuned configs during --smoke (default 4)")
    parser.add_argument("--only", choices=["ollama", "llamacpp"], default=None,
                        help="restrict to one engine's pair of configs (default: all four)")
    args = parser.parse_args(argv)

    if not args.smoke:
        parser.print_help()
        return 2

    ok = run_smoke(tuned_n=args.tuned_n, only=args.only)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
