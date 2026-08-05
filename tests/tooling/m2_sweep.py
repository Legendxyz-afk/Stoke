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
                               "identical": same, "of": len(rs),
                               "head_sha": HEAD_SHA, "rows": rs}, indent=1))


t_all = time.time()
for pid in sorted(banked):
    if pid in done:
        r = done[pid]
        verdict = "IDENTICAL" if r["identical"] else "DIFFERS at %s" % r["first_diff"]
        print(f"{pid}: {verdict} (resumed)", flush=True)
        continue
    prompt = ROOT / "tests" / "data" / "parity" / f"{pid}.txt"
    t0 = time.time()
    got, err = ids_of(M2EXE, prompt)
    dt = time.time() - t0
    if got is None:
        rows.append({"prompt": pid, "error": err, "secs": dt})
        print(f"{pid}: ERROR {err}", flush=True)
        flush(rows, t_all)
        continue
    head = banked[pid]
    first = next((i for i in range(min(len(got), len(head))) if got[i] != head[i]), None)
    if first is None and len(got) != len(head):
        first = min(len(got), len(head))
    # m2_ids is banked so the SEPARATE classification pass (ruling section 6)
    # never has to re-generate a 147-940 s stream to look at one position.
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
