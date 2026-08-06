"""Run the M2 binary over all 50 parity prompts and diff its token stream
against HEAD's, which leg 1 already banked.

WHY THIS ANSWERS SOMETHING WITHOUT THE ORACLE. Leg 1's cache holds HEAD's 64
ids for every prompt, keyed on HEAD's sha256. The M2 binary is on disk. So
"did K1-K5 change ANY token this corpus produces?" is answerable with no
llama-server at all -- and if the answer is "no", then leg 1's GATE FAIL is
inherited wholesale from M2 rather than caused by the K-work, for all 50
prompts rather than for the single position p02:16 already bisected.

Reads the cache but decides nothing from it: the banked ids are one SIDE of
the comparison, and the stoke_hash guard below refuses to run if they were
produced by a different binary than the one being credited.
"""
import argparse, ctypes, json, pathlib, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(r"M:\New Drive\Desktop\Programming\stoke")
CACHE = ROOT / "tests" / "tooling" / ".gate_cache"
MODEL = ROOT / "models" / "Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
M2EXE = ROOT / "build" / "stoke-m2-gate-d2b17c53.exe"
HEAD_SHA = "876ce53fd4a943fdd820f0bf9e8328b8aa751fa1269df9fd20f4fedbfb6390ff"

_ap = argparse.ArgumentParser()
_ap.add_argument("out")
_ap.add_argument("--jobs", type=int, default=1,
                 help="prompts to run CONCURRENTLY. Each is its own single-threaded "
                      "M2 process on an independent prompt, so concurrency cannot "
                      "change an id -- only how long the sweep takes.")
_args = _ap.parse_args()
OUT = pathlib.Path(_args.out)
JOBS = max(1, _args.jobs)

# ▶ WHY CONCURRENCY IS SAFE HERE AND WOULD NOT BE EVERYWHERE. The M2 binary is
# single-threaded (it predates K5's pool), each prompt is an independent
# process, and the ids are a deterministic function of (binary, model, prompt).
# So --jobs changes WALL TIME ONLY. What it does change is the meaning of the
# per-row `secs` field: under contention that is a wall time, not a clean
# per-prompt cost, and mixing the two bases in one file is exactly the error
# referent_interleave.py was fixed for. `jobs` is therefore recorded IN the
# state file, so a reader can tell which kind of timing they are holding.
#
# ▶ RAM IS THE CEILING, NOT CORES. Each M2 process holds ~1.2 GB private (the
# KV cache, allocated up front at max_ctx 4096) on top of ONE shared ~4.0 GB
# model page cache. The 2026-07-28 sizing recommended 2-3 on this 15.6 GB box
# and explicitly NOT 4. If the model is evicted the recorded ~3x rate collapse
# applies and the concurrency is self-defeating -- which is why this refuses
# rather than swaps.
MODEL_GB, PER_JOB_GB = 4.0, 1.3


def _free_gb():
    """Free physical RAM via GlobalMemoryStatusEx -- one syscall, no process
    walk. Returns None off Windows rather than guessing."""
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


if JOBS > 1:
    _free = _free_gb()
    # CHARGE THE MODEL EVERY TIME, even though a resumed sweep often finds it
    # already resident. Whether it is cached depends on what ran a minute ago,
    # which is not something a launch-time check can know -- and the failure it
    # guards against (evicting the model, then paying the recorded ~3x collapse)
    # is exactly what happens when the estimate is optimistic. Over-charging
    # costs at most one job of concurrency; under-charging costs the run.
    _need = MODEL_GB + JOBS * PER_JOB_GB
    if _free is not None and _free < _need:
        sys.exit(f"refusing --jobs {JOBS}: needs ~{_need:.1f} GB (model {MODEL_GB} "
                 f"+ {JOBS}x{PER_JOB_GB} private KV), {_free:.1f} GB free. A sweep "
                 f"that swaps is a sweep that thrashes -- lower --jobs or free RAM.")

banked = {}
for f in CACHE.glob("*.json"):
    if f.name.startswith("m3-gate"):
        continue
    d = json.load(open(f, encoding="utf-8"))
    if "stoke_ids" not in d.get("data", {}):
        continue
    if d["key"]["stoke_hash"] != HEAD_SHA:
        continue                      # not HEAD's -- never silently credited
    banked[re.search(r"(p\d+)\.txt", d["prompt_path"]).group(1)] = d["data"]["stoke_ids"]

print(f"banked HEAD streams: {len(banked)}", flush=True)
if len(banked) != 50:
    sys.exit(f"expected 50 HEAD streams under sha {HEAD_SHA[:8]}, got {len(banked)}")


# ▶ PLACEMENT AND PRIORITY ARE SET HERE, NOT LEFT TO THE SCHEDULER OR TO A
# LAUNCHER. Measured 2026-08-06 on this 6P+4E part: with the work left
# unplaced, two single-threaded prompts were bouncing across LP 1/5/13/15 --
# two of those are E-cores and two are the SECOND hyperthread of a P-core --
# and prompts took 448-607 s. Pinned to one thread per P-core they took
# 333-357 s. Affinity is inherited by child processes, so setting it on this
# process covers every stoke run it spawns; PRIORITY IS NOT RELIABLY INHERITED
# (observed: this process at High, its children still Normal), so the flag is
# passed explicitly at creation instead.
#
# 0x555 = LP 0,2,4,6,8,10, one thread on each of the six P-cores. Same mask the
# K6 gate legs use, for the same reason. It is deliberately NOT 0xFFF: giving
# the scheduler both hyperthreads of a core invites it to put two prompts on
# one physical core while another sits idle.
P_CORE_MASK = 0x555
_HIGH = getattr(subprocess, "HIGH_PRIORITY_CLASS", 0) if sys.platform == "win32" else 0

if sys.platform == "win32":
    # argtypes/restype are set explicitly: HANDLE and DWORD_PTR are 64-bit here
    # and ctypes defaults to c_int, which truncates the pseudo-handle silently.
    _k32 = ctypes.WinDLL("kernel32")
    _k32.GetCurrentProcess.restype = ctypes.c_void_p
    _k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not _k32.SetProcessAffinityMask(_k32.GetCurrentProcess(), P_CORE_MASK):
        sys.exit(f"fatal: could not pin to P-cores (mask 0x{P_CORE_MASK:X}); "
                 f"unpinned this job measured 448-607 s/prompt against 333-357 "
                 f"pinned, so proceeding would silently halve throughput")


def ids_of(exe, prompt):
    p = subprocess.run([str(exe), "run", str(MODEL), "--raw", "-f", str(prompt), "-n", "64"],
                       capture_output=True, timeout=1800, creationflags=_HIGH)
    if p.returncode != 0:
        return None, p.stderr.decode("utf-8", "replace")[-300:]
    out = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line == "---":
            break
        if line.lstrip("-").isdigit():
            out.append(int(line))
    return out, None


# RESUME. This exact run died once at 6/50 to a planned shutdown, and it is a
# ~3.5 h serial job -- so completed prompts are reloaded rather than redone.
# A row is only ever reloaded if it carries the ids it decided on, so a
# resumed verdict is never credited to a record that cannot show its work;
# ERROR rows are deliberately NOT reloaded (they are retried).
rows = []
done = {}
if OUT.exists():
    prior = json.loads(OUT.read_text())
    for r in (prior["rows"] if isinstance(prior, dict) else prior):
        if r.get("m2_ids") and not r.get("error"):
            done[r["prompt"]] = r
    rows = [done[p] for p in sorted(done)]
    print(f"resuming: {len(done)} prompts already recorded", flush=True)


def flush(rs, t_start):
    same = sum(1 for r in rs if r.get("identical"))
    OUT.write_text(json.dumps({"total_secs": round(time.time() - t_start, 1),
                               "identical": same, "of": len(rs), "jobs": JOBS,
                               "head_sha": HEAD_SHA, "rows": rs}, indent=1))


def run_one(pid):
    """Worker body: the subprocess and nothing else.

    Touches no shared state and never writes OUT. The parent stays the SINGLE
    writer -- flush() rewrites the whole file, so a second writer would not
    duplicate work, it would clobber rows."""
    prompt = ROOT / "tests" / "data" / "parity" / f"{pid}.txt"
    t0 = time.time()
    got, err = ids_of(M2EXE, prompt)
    return pid, got, err, time.time() - t0


for pid in sorted(done):
    r = done[pid]
    verdict = "IDENTICAL" if r["identical"] else "DIFFERS at %s" % r["first_diff"]
    print(f"{pid}: {verdict} (resumed)", flush=True)

todo = [pid for pid in sorted(banked) if pid not in done]
print(f"to run: {len(todo)} prompts at --jobs {JOBS}", flush=True)

t_all = time.time()
if JOBS == 1:
    # Serial stays a plain loop rather than a 1-worker pool: the sequential
    # path is the one every earlier row in this file was produced under, and
    # it should not silently acquire a scheduler.
    results = (run_one(pid) for pid in todo)
else:
    _pool = ThreadPoolExecutor(max_workers=JOBS)
    _futs = [_pool.submit(run_one, pid) for pid in todo]
    results = (f.result() for f in as_completed(_futs))

for pid, got, err, dt in results:
    if got is None:
        rows.append({"prompt": pid, "error": err, "secs": dt})
        print(f"{pid}: ERROR {err}", flush=True)
        rows.sort(key=lambda r: r["prompt"])
        flush(rows, t_all)
        continue
    head = banked[pid]
    first = next((i for i in range(min(len(got), len(head))) if got[i] != head[i]), None)
    if first is None and len(got) != len(head):
        first = min(len(got), len(head))
    # m2_ids is banked so the SEPARATE classification pass (ruling section 6)
    # never has to re-generate a multi-minute stream to look at one position.
    rows.append({"prompt": pid, "m2_len": len(got), "head_len": len(head),
                 "identical": first is None, "first_diff": first,
                 "secs": round(dt, 1), "m2_ids": got, "head_ids": head})
    print(f"{pid}: {'IDENTICAL' if first is None else f'DIFFERS at {first}'} "
          f"({len(got)} vs {len(head)} ids, {dt:.0f}s)", flush=True)
    rows.sort(key=lambda r: r["prompt"])
    flush(rows, t_all)

flush(rows, t_all)
same = sum(1 for r in rows if r.get("identical"))
print(f"\nM2 vs HEAD: {same}/{len(rows)} prompts token-identical, "
      f"{time.time() - t_all:.0f}s total", flush=True)
