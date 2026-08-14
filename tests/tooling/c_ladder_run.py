#!/usr/bin/env python3
"""Drive the L2/L3 ladder for every ESCALATE of a two-binary grading run.

Section 6: "Ladder <=1e-3 / no-STEP => ARITHMETIC-CLASS". The STEP leg is the
one that carries the verdict -- a layer whose diff jumps JUMP_FACTOR over the
running trend is the signature of a structural bug in one op, as against the
smooth compounding of a numeric difference. l2_diff.py owns that judgement;
this script only arranges the dumps and records what it said.

CONTROLS ARE NOT OPTIONAL. l2_diff.py's 1e-3 threshold is a
C-f32-vs-numpy-f32 constant, and a pair of engines that differ BY CONSTRUCTION
exceeds it identically whether they are healthy or broken -- M3 predicted this
and item [4] hit it. So the absolute leg is calibrated the way M3 calibrated
it: run the same ladder at positions where the two sides chose the SAME token,
and read the divergent positions against that band.

Divergent positions need a teacher-forced prompt (built first by
ladder_teacher_force.py, which verifies the round-trip id-for-id); controls
dump at their own last prefill position and need no construction.

Usage:
  python tests/tooling/c_ladder_run.py --stoke-a A.exe --stoke-b B.exe \\
      --workdir build/ladder-c --targets p08:54 p11:12 ... \\
      --controls p15 p16 p32 --out <...>-ladder.txt
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
PDIR = ROOT / "tests" / "data" / "parity"
L2 = ROOT / "tests" / "tooling" / "l2_diff.py"
L35_RE = re.compile(r"^L35\s+([\d.e+-]+)\s+([\d.e+-]+)", re.M)


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return (p.stdout.decode("utf-8", "replace"),
            p.stderr.decode("utf-8", "replace"), p.returncode)


def dump(stoke, model, prompt, out_dir, kv, threads):
    out_dir.mkdir(parents=True, exist_ok=True)
    so, se, rc = run([stoke, "run", str(model), "--raw", "-n", "1",
                      "-f", str(prompt), "--kv", kv, "--threads", str(threads),
                      "--dump", str(out_dir)])
    if rc != 0:
        raise SystemExit(f"dump failed ({rc}) for {prompt} -> {out_dir}\n"
                         f"{se[-400:]}")
    n = len(list(out_dir.glob("L*.f32")))
    if n != 36:
        raise SystemExit(f"{out_dir}: {n} layer files, expected 36")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stoke-a", required=True)
    ap.add_argument("--stoke-b", required=True)
    ap.add_argument("--kv-a", default="f32")
    ap.add_argument("--kv-b", default="f32")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--targets", nargs="*", default=[],
                    help="prompt:gp pairs; the teacher-forced file "
                         "<workdir>/<prompt>-tf<gp>.txt must already exist")
    ap.add_argument("--controls", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wd = pathlib.Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    lines, rows = [], []

    jobs = []
    for t in args.targets:
        name, gp = t.split(":")
        src = wd / f"{name}-tf{gp}.txt"
        if not src.exists():
            raise SystemExit(f"missing teacher-forced prompt {src} -- run "
                             f"ladder_teacher_force.py first")
        jobs.append((name, gp, src, "divergent"))
    for name in args.controls:
        jobs.append((name, "-", PDIR / f"{name}.txt", "control"))

    for name, gp, src, kind in jobs:
        tag = f"{name}-{gp}" if gp != "-" else f"{name}-ctl"
        da, db = wd / f"{tag}-A", wd / f"{tag}-B"
        dump(args.stoke_a, args.model, src, da, args.kv_a, args.threads)
        dump(args.stoke_b, args.model, src, db, args.kv_b, args.threads)
        so, se, rc = run([sys.executable, str(L2), str(da), str(db)])
        step = "STEP DETECTED" in so
        m = L35_RE.search(so)
        l35 = (float(m.group(1)), float(m.group(2))) if m else (None, None)
        rows.append({"prompt": name, "gp": gp, "kind": kind,
                     "L35_max_abs_diff": l35[0], "L35_mean_abs_diff": l35[1],
                     "step": step, "l2_exit": rc})
        print(f"{tag:14s} {kind:9s} L35 max={l35[0]:.3e} mean={l35[1]:.3e}  "
              f"STEP={'YES' if step else 'none'}")
        lines.append(f"===== {tag} ({kind}) =====\n{so}")

    pathlib.Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    pathlib.Path(args.out).with_suffix(".json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")

    steps = [r for r in rows if r["step"]]
    div = [r for r in rows if r["kind"] == "divergent"]
    ctl = [r for r in rows if r["kind"] == "control"]
    print(f"\n{len(rows)} curves: {len(div)} divergent, {len(ctl)} control")
    print(f"STEP detected in {len(steps)} -- "
          f"{', '.join(r['prompt'] for r in steps) if steps else 'NONE'}")
    if ctl:
        cm = [r["L35_mean_abs_diff"] for r in ctl]
        dm = [r["L35_mean_abs_diff"] for r in div]
        print(f"control mean_abs_diff  : {min(cm):.4f} - {max(cm):.4f}")
        if dm:
            print(f"divergent mean_abs_diff: {min(dm):.4f} - {max(dm):.4f}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
