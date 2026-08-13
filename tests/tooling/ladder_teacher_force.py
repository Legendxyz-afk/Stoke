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

⚠ PATHS AND THE SWEEP FILENAME ARE PINNED to the M4 item-[4] gate on this box.
M3's p41 resolution predicted M7 would need ladders again on Llama-3.2; when
that happens these become parameters, not constants.
"""
import json
import pathlib
import re
import subprocess
import sys

STOKE = r"M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe"
MODEL = r"M:\New Drive\Desktop\Programming\stoke\models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
PDIR = pathlib.Path(r"M:\New Drive\Desktop\Programming\stoke\tests\data\parity")
GR = pathlib.Path(r"M:\New Drive\Desktop\Programming\stoke\docs\superpowers\gate-reports")
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")


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
            for l in (GR / "2026-08-12-m4-g4-sweep.jsonl").open(encoding="utf-8")}
    targets = sys.argv[2:] or ["p12", "p03"]
    for name in targets:
        row = rows[name]
        gp = row["first_divergence"]
        src = PDIR / f"{name}.txt"
        want = tokenize(src) + row["f32_ids"][:gp]
        assert row["f32_ids"][:gp] == row["q8_ids"][:gp], "sides differ before gp"

        # stoke prints ids, then "---", then the detokenised continuation
        so, rc = sh([STOKE, "run", MODEL, "--raw", "-n", str(gp), "-f", str(src),
                     "--kv", "f32", "--threads", "6"])
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
