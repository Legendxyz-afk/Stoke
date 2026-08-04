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
import json, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path(r"M:\New Drive\Desktop\Programming\stoke")
CACHE = ROOT / "tests" / "tooling" / ".gate_cache"
MODEL = ROOT / "models" / "Qwen_Qwen3-4B-Instruct-2507-Q8_0.gguf"
M2EXE = ROOT / "build" / "stoke-m2-gate-d2b17c53.exe"
HEAD_SHA = "876ce53fd4a943fdd820f0bf9e8328b8aa751fa1269df9fd20f4fedbfb6390ff"
OUT = pathlib.Path(sys.argv[1])

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


def ids_of(exe, prompt):
    p = subprocess.run([str(exe), "run", str(MODEL), "--raw", "-f", str(prompt), "-n", "64"],
                       capture_output=True, timeout=1800)
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


rows = []
t_all = time.time()
for pid in sorted(banked):
    prompt = ROOT / "tests" / "data" / "parity" / f"{pid}.txt"
    t0 = time.time()
    got, err = ids_of(M2EXE, prompt)
    dt = time.time() - t0
    if got is None:
        rows.append({"prompt": pid, "error": err, "secs": dt})
        print(f"{pid}: ERROR {err}", flush=True)
        continue
    head = banked[pid]
    first = next((i for i in range(min(len(got), len(head))) if got[i] != head[i]), None)
    if first is None and len(got) != len(head):
        first = min(len(got), len(head))
    rows.append({"prompt": pid, "m2_len": len(got), "head_len": len(head),
                 "identical": first is None, "first_diff": first, "secs": round(dt, 1)})
    print(f"{pid}: {'IDENTICAL' if first is None else f'DIFFERS at {first}'} "
          f"({len(got)} vs {len(head)} ids, {dt:.0f}s)", flush=True)
    OUT.write_text(json.dumps(rows, indent=1))

same = sum(1 for r in rows if r.get("identical"))
OUT.write_text(json.dumps({"total_secs": round(time.time() - t_all, 1),
                           "identical": same, "of": len(rows), "rows": rows}, indent=1))
print(f"\nM2 vs HEAD: {same}/{len(rows)} prompts token-identical, "
      f"{time.time() - t_all:.0f}s total", flush=True)
