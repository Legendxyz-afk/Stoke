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
# Same GGUF re-registered with `PARAMETER num_gpu 0` -- zero layers offloaded, pure CPU. This is
# the engine-level pin the M0 primary matrix uses: ollama 0.30.11's GPU discovery ignores
# CUDA_VISIBLE_DEVICES=-1 (verified empirically 2026-07-16 -- still a 5.4GB GPU compute process
# with GPU-class decode rates), so the env-var route is insufficient on this box.
OLLAMA_MODEL_NAME_CPU = "stoke-bench-qwen3-cpu"
MODELFILE_CPU_PATH = pathlib.Path(__file__).resolve().parent / "Modelfile.qwen3-4b-cpu"
LLAMACPP_MODEL_NAME = "qwen3-4b"             # llama-server serves one model; field value is arbitrary

OLLAMA_HOST_ADDR = "127.0.0.1:11434"        # forced explicitly so ambient OLLAMA_HOST can't drift us

# -- stoke itself (M6 rung A5) -------------------------------------------------------------------
# The whole point of gate item [3] is that NOTHING below this line's own class had to change: the
# harness's measurement machinery -- bench.timed_stream, bench.run_scenario, scenarios.py,
# report.py -- is untouched, and stoke is driven through exactly the same OpenAI-SSE path the two
# baselines are. If any of that had needed a special case, the compat surface would be the thing
# that is wrong, not the harness.
STOKE_EXE = REPO_ROOT / "build" / ("win/stoke.exe" if sys.platform == "win32" else "linux/stoke")
STOKE_MODEL_NAME = "stoke"                  # echoed back in every response; any string works
STOKE_PORT = 8091                           # not 8089 (llama-server) and not 11434 (ollama)


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


def process_stats(pid):
    """Best-effort snapshot of a LIVE engine process, taken by the harness before it is killed.

    ▶ WHY (2026-09-11). On 2026-09-08 two `stoke serve` lifetimes measured the same workload
    11.7% apart while llama-server beside them repeated to 0.8%, and nothing in the bank could
    say WHY: the harness recorded rates and nothing about the process that produced them. The
    2026-08-17 EcoQoS finding had already named the two tells that separate "the code was slow"
    from "the OS placed it badly" -- CPU-seconds (the same work costs ~4x more processor time on
    an efficiency core, so a sick run reads HIGH here) and page faults (0.8-0.9M healthy vs
    1.5-2.7M sick) -- and neither was being written down. Now both are, per cell, for every
    engine, so the next such unit carries its own explanation.

    Windows only (everything here is a kernel32 call; psapi's memory counters are exported by
    kernel32 as K32GetProcessMemoryInfo). Elsewhere, and on any failure, returns what it could
    get plus an `error` field -- a missing tool degrades one field, not the cell."""
    out = {"pid": pid}
    if os.name != "nt":
        out["error"] = "process_stats: Windows only"
        return out
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            out["error"] = f"OpenProcess failed: {ctypes.get_last_error()}"
            return out
        try:
            # CPU times: FILETIME in 100 ns units.
            c, e, kt, ut = (wintypes.FILETIME() for _ in range(4))
            if k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(kt), ctypes.byref(ut)):
                ft = lambda f: ((f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e7  # noqa: E731
                out["cpu_user_s"] = ft(ut)
                out["cpu_kernel_s"] = ft(kt)
                # Creation time is an absolute FILETIME (since 1601); the process's age is
                # what a reader wants beside its CPU-seconds.
                now = wintypes.FILETIME()
                k32.GetSystemTimeAsFileTime(ctypes.byref(now))
                out["age_s"] = ft(now) - ft(c)

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(pmc)
            if k32.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                out["page_faults"] = int(pmc.PageFaultCount)
                out["peak_ws_mb"] = round(pmc.PeakWorkingSetSize / 2**20, 1)
                out["ws_mb"] = round(pmc.WorkingSetSize / 2**20, 1)

            pmask, smask = ctypes.c_size_t(), ctypes.c_size_t()
            if k32.GetProcessAffinityMask(h, ctypes.byref(pmask), ctypes.byref(smask)):
                out["affinity_mask"] = hex(pmask.value)
            out["priority_class"] = hex(k32.GetPriorityClass(h))

            # Process-level EcoQoS: the state Windows may apply to a process it deems
            # background. ControlMask bit set + StateMask bit set == throttled.
            class _PPT(ctypes.Structure):
                _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong),
                            ("StateMask", ctypes.c_ulong)]
            ppt = _PPT(1, 0, 0)
            if k32.GetProcessInformation(h, 4, ctypes.byref(ppt), ctypes.sizeof(ppt)):
                out["power_throttling"] = {"control": ppt.ControlMask, "state": ppt.StateMask}
        finally:
            k32.CloseHandle(h)
    except Exception as exc:
        out["error"] = f"process_stats: {exc}"
    return out


def set_process_affinity(pid, mask):
    """Windows: SetProcessAffinityMask on a live process; threads created afterwards inherit it.
    Returns False off Windows or on any failure -- callers that need the pin must treat False as
    'the arm did not happen', never measure and label it pinned."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_SET_INFORMATION, PROCESS_QUERY_INFORMATION = 0x0200, 0x0400
        h = k32.OpenProcess(PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, int(pid))
        if not h:
            return False
        try:
            return bool(k32.SetProcessAffinityMask(h, ctypes.c_size_t(int(mask))))
        finally:
            k32.CloseHandle(h)
    except Exception:
        return False


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
        # ▶ The startup announcements, snapshotted the moment the engine turns healthy
        # (2026-09-16). The ring above holds 500 lines and a unit's log is not bounded:
        # stoke's pool trace alone is ~13 lines per report, so a ten-minute N=32 unit
        # evicts the lines a bank is VERIFIED from -- `sched: STOKE_SCHED_CHUNK=`, the
        # pool's `threads/spin` header, llama-server's `n_slots` / `n_ctx_slot`. The
        # bank reads the configuration from what the server SAID, never from the
        # label, so the saying has to survive the run.
        self.startup_lines = []

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
        self.startup_lines = []
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
                    self.startup_lines = list(self._log_lines)
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

    def __init__(self, num_parallel=None, context_length=None, cpu_only=True):
        label = "ollama-tuned" if num_parallel else "ollama-default"
        # base_url is the server ROOT: bench.timed_stream() appends /v1/chat/completions itself
        # (contract locked by the committed, mock-tested Task 1 harness).
        super().__init__(label, base_url=f"http://{self.HOST}:{self.PORT}", host=self.HOST, port=self.PORT)
        self.num_parallel = num_parallel
        self.context_length = context_length
        # cpu_only=True (the M0 primary-matrix default) serves the num_gpu=0 registration of the
        # same GGUF; False serves the unpinned one (ollama's own GPU/CPU choice -- the behavior
        # recorded in the report's gpu-sidebar appendix).
        self.cpu_only = cpu_only
        self.model_name = OLLAMA_MODEL_NAME_CPU if cpu_only else OLLAMA_MODEL_NAME
        self.modelfile_path = MODELFILE_CPU_PATH if cpu_only else MODELFILE_PATH

    def _spawn(self):
        # CUDA_VISIBLE_DEVICES=-1 hides every CUDA device from ollama's GPU discovery, forcing
        # CPU execution. Without it, ollama silently offloads the whole model to this box's RTX
        # 3050 (caught mid-matrix 2026-07-16: 53 tok/s "single-stream" -- double the ~25 tok/s
        # CPU bandwidth ceiling -- and nvidia-smi showed ollama's llama-server as a 5.4GB GPU
        # compute process). The M0 primary matrix is CPU-vs-CPU: stoke Phase 1 is a CPU engine,
        # the premise gate's 160 tok/s ceiling is CPU physics, and the vendored llama.cpp
        # baseline is a CPU-only build. Unforced GPU-default numbers live in the report's
        # gpu-sidebar appendix (bench/results/raw/gpu-sidebar/).
        extra_env = {
            "OLLAMA_KEEP_ALIVE": "1h",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
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

    def __init__(self, parallel=None, ctx=None, label=None):
        base = "llamacpp-tuned" if parallel else "llamacpp-default"
        self.root_url = f"http://{self.HOST}:{self.PORT}"
        # base_url is the server ROOT: bench.timed_stream() appends /v1/chat/completions itself.
        super().__init__(label or base, base_url=self.root_url, host=self.HOST, port=self.PORT)
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


class StokeAdapter(Adapter):
    """stoke's own HTTP server -- M6 rung A5, and the point of the whole rung order.

    This class is the ONLY new code the M0 harness needed in order to drive stoke. It implements
    the same three-method contract the two baselines do (_spawn, _healthy, a base_url), and every
    request that follows goes through bench.timed_stream unchanged. That is gate item [3] stated
    as a diff rather than as a claim.

    ⚠ THE `model` FIELD IS ECHOED, NOT VALIDATED, and that matches both baselines: llama-server
    serves one model and treats the field as arbitrary, and stoke serves the one GGUF it was
    started with. Nothing here should ever route on it.
    """
    HOST = "127.0.0.1"
    PORT = STOKE_PORT

    def __init__(self, seqs=None, ctx=None, threads=None, env=None, affinity=None, label=None):
        base = "stoke-tuned" if seqs else "stoke-default"
        self.root_url = f"http://{self.HOST}:{self.PORT}"
        # base_url is the server ROOT: bench.timed_stream() appends /v1/chat/completions itself.
        super().__init__(label or base, base_url=self.root_url, host=self.HOST, port=self.PORT)
        self.seqs = seqs
        self.ctx = ctx
        self.threads = threads
        self.model_name = STOKE_MODEL_NAME
        # ▶ A/B knobs (2026-09-11), so one binary can be measured under two runtime settings in
        # one interleaved sitting: extra environment for the server (STOKE_POOL_SPIN,
        # STOKE_POOL_TRACE, ...) and a process affinity mask applied right after spawn -- before
        # the model load finishes and long before the pool's threads exist, which inherit it.
        # Both are VERIFIABLE from the bank: process_stats() records the affinity mask, and the
        # pool trace header prints the spin budget it read.
        self.env = dict(env or {})
        self.affinity = affinity

    def _spawn(self):
        argv = [
            str(STOKE_EXE), "serve", str(MODEL_GGUF),
            "--host", self.HOST, "--port", str(self.PORT),
        ]
        if self.ctx:
            argv += ["--ctx", str(self.ctx)]
        if self.seqs:
            argv += ["--seqs", str(self.seqs)]
        if self.threads:
            argv += ["--threads", str(self.threads)]
        # No --max-requests: that option exists so the golden gate's server exits on its own, and
        # here the harness owns the lifecycle -- Adapter.stop() tree-kills and then blocks until
        # the port is verifiably free.
        env = None
        if self.env:
            env = os.environ.copy()
            env.update(self.env)
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        )
        if self.affinity is not None:
            if not set_process_affinity(proc.pid, self.affinity):
                proc.kill()
                raise RuntimeError(f"{self.label}: could not set affinity {self.affinity:#x}")
        return proc

    def _healthy(self):
        # stoke binds the port only AFTER the 2.5GB model is loaded, so a refused connection means
        # "still loading" and a 200 means genuinely ready -- there is no window where it accepts
        # connections it cannot serve. Adapter.start() polls this for up to timeout_s.
        return _probe_status(self.root_url + "/health") == 200


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
    # stoke is NOT in the default four. Those four ARE the M0 baseline matrix and its published
    # numbers; adding a fifth would silently change what "all configs" means for every existing
    # caller of this function. It is reachable only by asking for it by name.
    if only == "stoke":
        return [StokeAdapter(), StokeAdapter(seqs=tuned_n, ctx=1024)]
    if only is not None:
        raise ValueError(
            f"unknown engine filter: {only!r} (expected 'ollama', 'llamacpp' or 'stoke')")
    return configs


# -- one-time setup: register the shared GGUF with Ollama ---------------------------------------
def ensure_ollama_model(name=OLLAMA_MODEL_NAME_CPU, modelfile=MODELFILE_CPU_PATH):
    """Register models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf with Ollama under `name` via
    `ollama create -f <modelfile>`, so ollama-default/ollama-tuned serve the exact same weights
    llama-server does (same-GGUF fairness is the whole point of this bench). Defaults register
    the num_gpu=0 CPU-pinned variant the primary matrix uses; pass
    (OLLAMA_MODEL_NAME, MODELFILE_PATH) for the unpinned GPU-sidebar variant. Callers with an
    adapter in hand should pass (adapter.model_name, adapter.modelfile_path).

    Requires `ollama serve` to already be running -- `ollama create`/`ollama list` are CLI clients
    that talk to the local server, they do nothing standalone. Idempotent: skips `ollama create`
    if `ollama list` already shows the model (so repeated --smoke runs don't re-copy 2.5GB).

    Modelfile path handling: confirmed at dispatch that `ollama create` accepts the UNQUOTED
    absolute Windows path with spaces (`FROM M:\\New Drive\\...\\Qwen3-...gguf`) exactly as written
    in bench/Modelfile.qwen3-4b -- no quoting workaround was needed. Kept as a fallback below only
    in case a future Ollama version regresses this.
    """
    def _list_stdout():
        # encoding pinned: ollama's console output can carry bytes invalid in the Windows default
        # cp1252 codec (progress bars during create) -- a raw text=True read crashes the
        # subprocess reader thread with UnicodeDecodeError.
        r = subprocess.run(
            [str(OLLAMA_EXE), "list"], env=_ollama_env(), capture_output=True,
            encoding="utf-8", errors="replace", timeout=30,
        )
        return r.stdout if r.returncode == 0 else ""

    if name in _list_stdout():
        return  # already registered (this run or a previous one)

    create = subprocess.run(
        [str(OLLAMA_EXE), "create", name, "-f", str(modelfile)],
        env=_ollama_env(), capture_output=True, encoding="utf-8", errors="replace", timeout=600,  # first create hashes+copies 2.5GB
    )
    if create.returncode != 0:
        # confirmed-working form failed (env drift, moved file, etc) -- try a quoted-path FROM
        # once before giving up, preserving any PARAMETER lines, and flag loudly.
        original = modelfile.read_text(encoding="utf-8")
        quoted = "\n".join(
            f'FROM "{MODEL_GGUF}"' if line.startswith("FROM ") else line
            for line in original.splitlines()
        ) + "\n"
        modelfile.write_text(quoted, encoding="utf-8")
        retry = subprocess.run(
            [str(OLLAMA_EXE), "create", name, "-f", str(modelfile)],
            env=_ollama_env(), capture_output=True, encoding="utf-8", errors="replace", timeout=600,
        )
        if retry.returncode != 0:
            raise RuntimeError(
                f"fatal: `ollama create {name}` failed with BOTH unquoted and quoted "
                f"Modelfile paths.\nunquoted stderr:\n{create.stderr}\nquoted stderr:\n{retry.stderr}"
            )
        print(f"NOTE: ollama create needed the QUOTED Modelfile path form (unquoted failed) -- "
              f"{modelfile.name} has been rewritten to the quoted form.", file=sys.stderr)

    verify = _list_stdout()
    if name not in verify:
        raise RuntimeError(
            f"fatal: `ollama create` reported success but `ollama list` does not show "
            f"{name} -- stdout:\n{verify}"
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
                ensure_ollama_model(adapter.model_name, adapter.modelfile_path)
                print(f"{adapter.label}: ollama model '{adapter.model_name}' confirmed via `ollama list`", flush=True)

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


def run_scenario_check(scenario_name="s1", seqs=8, ctx=1024, threads=None, timeout_s=900):
    """M6 GATE ITEM [3]: drive one whole M0 scenario end to end, through the UNMODIFIED harness.

    The smoke above is one request. This is the real thing: S1 is eight clients arriving on a
    Poisson schedule, each with a 512-token prompt and 128 max_tokens, and every one of them is
    driven by bench.run_scenario -- the same function, with the same arguments, that produced the
    published M0 baseline numbers for ollama and llama-server. Nothing in bench.py, scenarios.py
    or report.py is touched or special-cased; the only stoke-shaped thing in the path is
    StokeAdapter's _spawn/_healthy.

    ⚠⚠ IT PRINTS AND IT DOES NOT BANK, AND THAT IS NOT A CONVENIENCE. A number printed next to a
    green banner gets quoted three weeks later as if it were a measurement. This runs one arm,
    once, un-interleaved, with no quiesce gate and no counterbalancing -- every safeguard the
    perf protocol requires is absent, deliberately, because this gate is asking whether the
    COMPAT SURFACE works end to end, not how fast anything is. The throughput line below is
    printed only so a run that silently produced nothing cannot look identical to one that
    worked. Do not put it in a report.
    """
    import scenarios  # local: only this entry point needs it

    table = {"s1": scenarios.S1, "s3": scenarios.S3, "s4": scenarios.S4}
    scenario = table.get(scenario_name)
    if scenario is None:
        raise ValueError(f"unknown scenario {scenario_name} (have: {sorted(table)})")

    adapter = StokeAdapter(seqs=seqs, ctx=ctx, threads=threads)
    print("=" * 88)
    print(f"M6 gate [3]: {scenario.name} -> {adapter.label} "
          f"(seqs={seqs} ctx={ctx} threads={threads or 'default'})")
    print(f"  {scenario.n_clients} clients, {scenario.prompt_tokens} prompt tokens, "
          f"{scenario.max_tokens} max_tokens, mode={scenario.mode}")
    print("  FUNCTIONAL GATE -- the numbers below are NOT a measurement and must not be quoted")
    print("=" * 88, flush=True)

    ok = False
    try:
        base_url = adapter.start()
        print(f"{adapter.label}: start() OK, base_url={base_url}", flush=True)
        rep = bench.run_scenario(
            base_url, scenario, adapter.model_name, use_usage=True, timeout_s=timeout_s,
            salt=str(time.time()),
        )
        print(json.dumps({k: v for k, v in rep.items() if k != "per_client"}, indent=2))

        n_clients = scenario.n_clients
        completed = rep.get("completed", 0)
        errors = rep.get("errors") or []
        short = [c for c in (rep.get("per_client") or [])
                 if not c or c.get("n_tokens", 0) < 1]
        missing_usage = [c for c in (rep.get("per_client") or []) if c and c.get("usage_missing")]

        if errors:
            print(f"FATAL: {len(errors)} client error(s): {errors[:3]}", flush=True)
        if completed != n_clients:
            print(f"FATAL: {completed}/{n_clients} clients completed", flush=True)
        if short:
            print(f"FATAL: {len(short)} client(s) streamed no tokens", flush=True)
        # The harness falls back to counting arrivals when an engine omits usage, and that
        # fallback is exactly what would hide a broken include_usage on our side.
        if missing_usage:
            print(f"FATAL: {len(missing_usage)} client(s) got no usage trailer -- the harness "
                  f"silently falls back to counting arrivals, so this must be checked, not "
                  f"assumed", flush=True)

        ok = (not errors and completed == n_clients and not short and not missing_usage)
    except Exception as exc:
        print(f"FATAL: {adapter.label}: {exc}", flush=True)
    finally:
        try:
            adapter.stop()
            print(f"{adapter.label}: stop() OK -- port {adapter.port} confirmed free", flush=True)
        except Exception as exc:
            print(f"FATAL: stop() FAILED: {exc}", flush=True)
            ok = False

    print()
    print(f"GATE [3] {'PASSED' if ok else 'FAILED'} -- {scenario.name} through the unmodified "
          f"M0 harness")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the strictly-sequential smoke test")
    parser.add_argument("--scenario", default=None, metavar="NAME",
                        help="M6 gate [3]: drive one M0 scenario (s1/s3/s4) against stoke through "
                             "the unmodified harness. Prints, never banks.")
    parser.add_argument("--seqs", type=int, default=8, help="--seqs for `stoke serve` under --scenario")
    parser.add_argument("--ctx", type=int, default=1024, help="--ctx for `stoke serve` under --scenario")
    parser.add_argument("--threads", type=int, default=None, help="--threads for `stoke serve`")
    parser.add_argument("--tuned-n", type=int, default=4, help="parallelism N for *-tuned configs during --smoke (default 4)")
    parser.add_argument("--only", choices=["ollama", "llamacpp", "stoke"], default=None,
                        help="restrict to one engine's pair of configs (default: all four; "
                             "'stoke' selects stoke's own pair, which is NOT in the default four)")
    args = parser.parse_args(argv)

    if args.scenario:
        ok = run_scenario_check(args.scenario, seqs=args.seqs, ctx=args.ctx,
                                threads=args.threads)
        return 0 if ok else 1

    if not args.smoke:
        parser.print_help()
        return 2

    ok = run_smoke(tuned_n=args.tuned_n, only=args.only)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
