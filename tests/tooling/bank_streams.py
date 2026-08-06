#!/usr/bin/env python3
"""Bank HEAD's generated streams for a model file, WITHOUT the oracle.

▶ WHY THIS EXISTS. [2v2-iii] needs HEAD's Q4_K_M streams on disk before any
probe can be teacher-forced against them (ruling §10 item 1). Every existing
launcher in this tree reaches that state by running the FULL gate, which starts
`llama-server` -- and the oracle is `llama-server`-authorization work that is
Kabir's to launch, retired from the criterion by the 2026-08-05 ruling and now
survey-only. Banking is the stoke-only half, so it gets a stoke-only entry
point rather than a gate invocation with the oracle talked out of running.

▶ IT REUSES parity_gate's CACHE RATHER THAN INVENTING A STORE. `fetch_stoke_result`
already writes `stoke_ids` under the four-part key (prompt hash, stoke.exe hash,
n_predict, invocation signature), so a stream banked here is the same artifact,
in the same place, under the same key, that a later gate or probe run will read.
That is the whole point: a second store would be a second thing to keep in sync
with `stoke_hash`, and the freeze rule already has one of those.

▶ RESUME IS FREE AND IS NOT A FEATURE OF THIS FILE. The cache is per prompt, so
an interrupted run costs the in-flight prompt only. This script adds no resume
logic of its own precisely so there is no second, subtly different notion of
"already done".

▶ THE ORACLE BINARY IS HASHED, NEVER RUN. `invocation_signature()` reads
ORACLE_EXE's sha256 into the cache key, so the file must exist -- but nothing
here can start a server: the only parity_gate entry point called is
`fetch_stoke_result`. The selftest asserts that by construction.
"""

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity_gate as pg  # noqa: E402


PER_JOB_GB = 1.3     # same figure m2_sweep.py and m2_classify_probes.py charge
                     # per concurrent stoke: KV cache at max_ctx plus scratch.


def _free_gb():
    """ullAvailPhys, matching the guard m2_sweep.py and m2_classify_probes.py use.

    NOT Win32_OperatingSystem's FreePhysicalMemory, which counts the free list
    only and read 3.31 GB against this function's 7.10 GB on the same box on
    2026-08-06. The difference is standby pages -- and for an mmapped model
    those are exactly where the model wants to live, so the free-list number
    understates headroom for this workload specifically.
    """
    if sys.platform != "win32":
        return None

    class _MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    ms = _MS()
    ms.dwLength = ctypes.sizeof(_MS)
    ctypes.WinDLL("kernel32").GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullAvailPhys / (1024.0 ** 3)


def bind_model(key: str):
    """Rebind parity_gate's module globals exactly as its main() does.

    Every consumer -- stoke_generate_argv, the cache key's invocation
    signature, the size pin -- reads these at call time, so this must happen
    before the first read or a prompt gets banked against the wrong file under
    the other file's key. parity_gate:2166 makes the same assignment for the
    same stated reason.
    """
    pg.MODEL_KEY = key
    pg.MODEL_PATH, pg.MODEL_SIZE_EXPECTED = pg.MODELS[key]


def pin_model_size():
    """Fail loudly if the model file is not the one the size constant names.

    The gate gets this check for free because it pins MODEL_SIZE_EXPECTED at
    every server start. There is no server here, so without this a truncated
    or swapped file would bank 50 streams that look exactly like real ones --
    and the cache key records the size it EXPECTED, not the size on disk, so
    the substitution would survive into the probe run and the gate report.
    """
    actual = pg.MODEL_PATH.stat().st_size
    if actual != pg.MODEL_SIZE_EXPECTED:
        pg.fatal(f"model size mismatch for {pg.MODEL_PATH.name}: expected "
                 f"{pg.MODEL_SIZE_EXPECTED} bytes, found {actual}")
    return actual


class _Args:
    """The attribute surface fetch_stoke_result actually reads."""

    def __init__(self, model, n_predict):
        self.model = model
        self.n_predict = n_predict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(pg.MODELS.keys()), default="q4_k_m",
                    help="model key to bank (default q4_k_m -- the leg that needs it)")
    ap.add_argument("--n-predict", type=int, default=pg.N_PREDICT_DEFAULT, dest="n_predict")
    ap.add_argument("--only", default=None, metavar="p00,p07,...",
                    help="comma-separated prompt ids instead of the whole corpus")
    ap.add_argument("--progress", default=None, metavar="PATH",
                    help="append one json line per completed prompt")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return run_selftest()

    bind_model(a.model)
    size = pin_model_size()

    model_gb = size / (1024.0 ** 3)
    free, need = _free_gb(), size / (1024.0 ** 3) + PER_JOB_GB
    if free is not None and free < need:
        # Charge the model every time even on a resume: a resumed run usually
        # finds it already cached, but "usually" is not a guarantee worth
        # trading a page-cache collapse for. Same reasoning as m2_sweep.py:71.
        sys.exit(f"refusing to launch: needs ~{need:.1f} GB "
                 f"(model {model_gb:.1f} + {PER_JOB_GB} per job), free {free:.2f} GB")

    ids = ([p.strip() for p in a.only.split(",") if p.strip()] if a.only
           else [p.stem for p in sorted(pg.CORPUS_DIR.glob("p*.txt"))])
    args = _Args(a.model, a.n_predict)

    print(f"model      : {pg.MODEL_PATH.name} ({size} bytes, pinned)")
    print(f"stoke.exe  : {pg._sha256_file(pg.STOKE_EXE)[:16]}...")
    print(f"n_predict  : {a.n_predict}")
    print(f"prompts    : {len(ids)}")
    print(f"free RAM   : {free:.2f} GB (need ~{need:.1f})" if free is not None else "free RAM   : n/a")
    print(flush=True)

    t0 = time.time()
    banked = cached = 0
    for i, pid in enumerate(ids, 1):
        # Cache hit vs miss is observed BEFORE the call, so the printed
        # tally reports what this run actually spent rather than what it found.
        key = pg._cache_key_hash(pg._cache_key_parts(pg.corpus_path(pid), args))
        had = "stoke_ids" in pg._load_cache_data(pg.corpus_path(pid), args)
        t = time.time()
        stream = pg.fetch_stoke_result(pid, args)
        dt = time.time() - t
        if had:
            cached += 1
        else:
            banked += 1
        print(f"[{i:2d}/{len(ids)}] {pid}  {len(stream):3d} ids  "
              f"{'cached' if had else f'{dt:6.1f}s'}  key={key[:12]}", flush=True)
        if a.progress:
            with open(a.progress, "a", encoding="utf-8") as f:
                f.write(json.dumps({"prompt": pid, "n_ids": len(stream),
                                    "cached": had, "secs": round(dt, 2),
                                    "model": a.model}) + "\n")

    print(f"\nbanked {banked}, already cached {cached}, "
          f"{time.time() - t0:.1f}s total", flush=True)
    return 0


def run_selftest() -> int:
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))

    # bind_model reaches the argv builder and the cache key, not just a global.
    bind_model("q8_0")
    q8_argv = pg.stoke_generate_argv(Path("<p>"), 64)
    q8_sig = pg.invocation_signature(_Args("q8_0", 64))
    bind_model("q4_k_m")
    q4_argv = pg.stoke_generate_argv(Path("<p>"), 64)
    q4_sig = pg.invocation_signature(_Args("q4_k_m", 64))
    ck("bind_model changes the stoke argv", q8_argv != q4_argv)
    ck("bind_model changes the cache key's invocation", q8_sig != q4_sig)
    ck("q4 argv names the q4 file", "Q4_K_M" in " ".join(q4_argv))
    ck("q8 argv names the q8 file", "Q8_0" in " ".join(q8_argv))

    # The two models must not share a cache key -- a collision would silently
    # report the first file's banked stream as the second file's result.
    bind_model("q8_0")
    k8 = pg._cache_key_hash(pg._cache_key_parts(pg.corpus_path("p00"), _Args("q8_0", 64)))
    bind_model("q4_k_m")
    k4 = pg._cache_key_hash(pg._cache_key_parts(pg.corpus_path("p00"), _Args("q4_k_m", 64)))
    ck("q8 and q4 cache keys differ for the same prompt", k8 != k4)

    # n_predict is in the key too, so a re-bank at another length cannot alias.
    kn = pg._cache_key_hash(pg._cache_key_parts(pg.corpus_path("p00"), _Args("q4_k_m", 32)))
    ck("n_predict is in the cache key", kn != k4)

    # The size pin must actually fatal, not warn.
    bind_model("q4_k_m")
    real = pg.MODEL_SIZE_EXPECTED
    try:
        pg.MODEL_SIZE_EXPECTED = real + 1
        try:
            pin_model_size()
            ck("pin_model_size fatals on a size mismatch", False)
        except pg.GateError:
            ck("pin_model_size fatals on a size mismatch", True)
    finally:
        pg.MODEL_SIZE_EXPECTED = real
    ck("pin_model_size passes on the real file", pin_model_size() == real)

    # ▶ NOTHING IN THE BANKING PATH MAY BE ABLE TO START A SERVER, and this is
    # checked against BYTECODE rather than source text. The first version of
    # this check grepped the module source and failed on its own docstring --
    # a test that reports on prose while claiming to report on behaviour, which
    # is the shape this project has been bitten by twice.
    #
    # The boundary being asserted is precise: MY functions never name an oracle
    # symbol. parity_gate's cache key does read ORACLE_EXE's sha256, which is a
    # file read -- OracleServer, the only thing that spawns a process, is
    # unreachable from here because nothing calls it.
    oracle_syms = {"OracleServer", "fetch_oracle_result", "oracle_argv",
                   "run_gate", "run_survey", "Popen"}
    referenced = set()
    for fn in (main, bind_model, pin_model_size, _free_gb):
        referenced |= set(fn.__code__.co_names)
    ck("banking path names no oracle/server symbol", not (referenced & oracle_syms))
    ck("...and the check can see symbols at all", "fetch_stoke_result" in referenced)

    ck("guard reads ullAvailPhys", _free_gb() is None or _free_gb() > 0)

    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {n}")
    print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
