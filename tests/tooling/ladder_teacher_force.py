"""Build a teacher-forced prompt for a ladder run at generated position gp.

`--dump` fires at the LAST PREFILL position, so to dump the forward pass that
produced generated token gp we must hand the engine prompt + the gp tokens
before it as prompt text. Both sides agree on those ids by construction (gp is
the FIRST divergence), so one text serves both.

THE RISK THIS SCRIPT REFUSES TO TAKE: detokenise-then-retokenise need not
round-trip, and a prompt that tokenises to different ids would silently dump a
DIFFERENT forward pass than the one being adjudicated. So the constructed text
is tokenised and compared id-for-id against prompt_ids + gen_ids[:gp]; a
mismatch REFUSES the position rather than proceeding. That check earned itself
on first use: the captured continuation carried the print's trailing newline
(id 198), so every prompt was one token long and would have dumped the
position AFTER the one under adjudication.

⚠ THE PINS BECAME PARAMETERS ON 2026-08-14, exactly as this note predicted --
M4.5 rung C needed ladders on a second pair, sooner than M7's Llama-3.2. The
OLD DEFAULTS ARE PRESERVED so item [4]'s invocation
(`ladder_teacher_force.py OUT p12 p03`) still resolves to the same binary,
sweep file and KV mode it did then; the record stays reproducible. Anything
else is passed with --stoke / --sweep / --kv.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

MODEL = r"M:\New Drive\Desktop\Programming\stoke\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
PDIR = pathlib.Path(r"M:\New Drive\Desktop\Programming\stoke\tests\data\parity")
GR = pathlib.Path(r"M:\New Drive\Desktop\Programming\stoke\docs\superpowers\gate-reports")

DEFAULT_STOKE = r"M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe"
DEFAULT_SWEEP = "2026-08-12-m4-g4-sweep.jsonl"

_ap = argparse.ArgumentParser()
_ap.add_argument("out", nargs="?", default=".")
_ap.add_argument("targets", nargs="*", default=None)
_ap.add_argument("--stoke", default=DEFAULT_STOKE)
_ap.add_argument("--sweep", default=DEFAULT_SWEEP)
_ap.add_argument("--kv", default="f32")
_ap.add_argument("--threads", default="6")
ARGS = _ap.parse_args()

STOKE = ARGS.stoke
OUT = pathlib.Path(ARGS.out)
# Item [4] happened to run into a directory that already existed; a fresh
# workdir made this fail at the FIRST write with a bare FileNotFoundError,
# after the engine work for that position had already been spent.
OUT.mkdir(parents=True, exist_ok=True)


def sh(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.decode("utf-8", "replace"), p.returncode


def tokenize(path):
    so, rc = sh([STOKE, "tokenize", MODEL, str(path)])
    m = re.search(r"\[([\d,\s]*)\]", so)
    if not m or rc != 0:
        raise SystemExit(f"tokenize failed for {path}: {so[:200]!r}")
    body = m.group(1).strip()
    return [] if not body else [int(x) for x in body.split(",")]


def main():
    rows = {json.loads(l)["prompt"]: json.loads(l)
            for l in (GR / ARGS.sweep).open(encoding="utf-8")}
    targets = ARGS.targets or ["p12", "p03"]
    for name in targets:
        row = rows[name]
        gp = row["first_divergence"]
        src = PDIR / f"{name}.txt"
        want = tokenize(src) + row["f32_ids"][:gp]
        assert row["f32_ids"][:gp] == row["q8_ids"][:gp], "sides differ before gp"

        # stoke prints ids, then "---", then the detokenised continuation
        so, rc = sh([STOKE, "run", MODEL, "--raw", "-n", str(gp), "-f", str(src),
                     "--kv", ARGS.kv, "--threads", ARGS.threads])
        if rc != 0 or "---" not in so:
            raise SystemExit(f"{name}: could not read a continuation")
        cont = so.split("---", 1)[1]
        if cont.startswith("\n"):
            cont = cont[1:]
        # the print's own trailing newline is not part of the continuation and
        # tokenises to a real id (198), which the round-trip check caught
        if cont.endswith("\n"):
            cont = cont[:-1]

        forced = OUT / f"{name}-tf{gp}.txt"
        forced.write_bytes(src.read_bytes() + cont.encode("utf-8"))
        got = tokenize(forced)
        ok = (got == want)
        print(f"{name} gp={gp}: prompt={len(want)-gp} + {gp} forced = {len(want)} ids; "
              f"round-trip {'EXACT' if ok else 'MISMATCH'} (got {len(got)})")
        if not ok:
            first = next((i for i in range(min(len(got), len(want)))
                          if got[i] != want[i]), min(len(got), len(want)))
            print(f"  first difference at index {first}: "
                  f"want {want[max(0,first-2):first+3]} got {got[max(0,first-2):first+3]}")
            print(f"  REFUSED -- this position cannot be laddered by teacher forcing")
            forced.unlink()


if __name__ == "__main__":
    main()
