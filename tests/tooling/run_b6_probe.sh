#!/bin/sh
# tests/tooling/run_b6_probe.sh -- B6's forward half, end to end, in the order
# the pre-registration fixes:
#   docs/superpowers/design-notes/2026-09-14-m7-b6-unsloth-probe-preregistration.md
#
# Steps, each banked under build/b6/ and SKIPPED when its output already
# exists, so a killed run resumes where it stopped instead of re-spending an
# hour:
#   0  pre-flight: ullAvailPhys > model + 2 GB (the ruled launch condition,
#      quoted by its own number), sha256 of the model and the binary banked.
#   1  phase A for P-00 alone (stoke + the BATCHED reference) -- gives the
#      stream ids the controls run on.
#   2  the batched reference again on P-00 at EVERY position 0..n-1 (seconds),
#      so C1-Q compares at all of them, not only the 64 generated ones.
#   3  C1-Q's serial arm: the SERIAL reference at every position (the long
#      pole -- tens of seconds per position).
#   4  T6's second serial run, positions 0..39 only.
#   5  m7_control: C1-Q (serial vs batched, T5 threshold 1e-3) and T6 (serial
#      vs serial, --require-exact). ⚠ IF C1-Q FAILS THE RUN STOPS HERE: the
#      batched arm is not licensed for this pair, and the pre-registration
#      says the grid is redesigned, not re-run against a looser threshold.
#   6  phase A for the remaining 14 streams (batched).
#   7  phase B: freeze tau under f32_intcore_unsloth_q4km.
#   8  phase C: grade.
# The verdict of every step is taken from what it PRINTS and banks, never from
# an exit code alone. Every reference run goes through run_ref_probe.ps1,
# which pins itself (0xFFF, 6 threads) and throws if the pin does not take.
#
# Usage (from the repo root, under bash/sh -- NOT PowerShell):
#   sh tests/tooling/run_b6_probe.sh [outdir]
set -u
OUT=${1:-build/b6}
NULL_POS=40

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 90
model="models/unsloth-Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
stoke="build/win/stoke.exe"
probe="tests/tooling/m7_probe.py"
control="tests/tooling/m7_control.py"
launcher="tests/tooling/run_ref_probe.ps1"
grid="$OUT/grid"
mkdir -p "$grid"
LOG="$OUT/driver.log"
export PYTHONIOENCODING=utf-8
export MSYS_NO_PATHCONV=1

say() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
step() { say "=== $1 ==="; t0=$(date +%s); }
done_step() { say "=== $1 done in $(( $(date +%s) - t0 )) s ==="; }

# ---- 0. pre-flight ----
for f in "$model" "$stoke" "$probe" "$control" "$launcher"; do
    [ -e "$f" ] || { say "ABORT -- missing $f"; exit 1; }
done
avail_mb=$(powershell -NoProfile -Command '
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public class B6Mem { [StructLayout(LayoutKind.Sequential)] public struct MSX { public uint dwLength; public uint dwMemoryLoad; public ulong ullTotalPhys; public ulong ullAvailPhys; public ulong a; public ulong b; public ulong c; public ulong d; public ulong e; }
[DllImport("kernel32.dll", SetLastError=true)] static extern bool GlobalMemoryStatusEx(ref MSX lpBuffer);
public static ulong AvailPhys() { var m = new MSX(); m.dwLength = (uint)Marshal.SizeOf(m); GlobalMemoryStatusEx(ref m); return m.ullAvailPhys; } }
"@
[int]([B6Mem]::AvailPhys() / 1MB)' | tr -d '\r')
model_mb=$(( $(wc -c < "$model") / 1048576 ))
say "pre-flight: ullAvailPhys ${avail_mb:-?} MB, model $model_mb MB, need > $((model_mb + 2048)) MB"
if [ -z "${avail_mb:-}" ] || [ "$avail_mb" -le $((model_mb + 2048)) ]; then
    say "ABORT -- launch condition not met (free RAM > model + 2 GB by ullAvailPhys)"
    exit 1
fi
sha_model=$(sha256sum "$model" | cut -d' ' -f1)
sha_stoke=$(sha256sum "$stoke" | cut -d' ' -f1)
say "model sha256 $sha_model ($model)"
say "stoke sha256 $sha_stoke ($stoke)"
printf '{"model":"%s","model_sha256":"%s","stoke":"%s","stoke_sha256":"%s","started":"%s","avail_mb":%s}\n' \
    "$model" "$sha_model" "$stoke" "$sha_stoke" "$(date +%Y-%m-%dT%H:%M:%S)" "$avail_mb" > "$OUT/preflight.json"

# ---- 1. phase A, P-00 only ----
p00="$grid/P-00.grid.json"
if [ -e "$p00" ]; then say "skip: $p00 exists"; else
    step "1 phase A P-00 (batched)"
    python -u "$probe" --set unsloth --phase a --out-dir "$grid" --only P-00 2>&1 | tee -a "$LOG"
    [ -e "$p00" ] || { say "ABORT -- phase A banked no P-00 grid"; exit 1; }
    done_step "1 phase A P-00"
fi
n_prompt=$(python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['n_prompt'])" "$p00")
n_gen=$(python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['n_gen'])" "$p00")
n_total=$((n_prompt + n_gen))
ids="$grid/work/P-00.stream.ids"
[ -e "$ids" ] || { say "ABORT -- no ids file behind P-00: $ids"; exit 1; }
all_pos=$(seq -s, 0 $((n_total - 1)))
null_pos=$(seq -s, 0 $((NULL_POS - 1)))
say "P-00: n_prompt=$n_prompt n_gen=$n_gen -> $n_total positions; null arm $NULL_POS"

ref_run() { # $1 out json, $2 mode, $3 positions
    powershell -NoProfile -File "$launcher" -Gguf "$model" -IdsFile "$ids" -Out "$1" -Mode "$2" -ProbeAt "$3" 2>&1 | tee -a "$LOG"
    [ -e "$1" ] || { say "ABORT -- the reference wrote no file: $1"; exit 1; }
}

# ---- 2. batched, every position ----
batched_all="$OUT/P-00.batched.all.json"
if [ -e "$batched_all" ]; then say "skip: $batched_all exists"; else
    step "2 batched reference, all $n_total positions"
    ref_run "$batched_all" batched "$all_pos"
    done_step "2 batched-all"
fi

# ---- 3. serial, every position (C1-Q's other arm) ----
serial_all="$OUT/P-00.serial.all.json"
if [ -e "$serial_all" ]; then say "skip: $serial_all exists"; else
    step "3 serial reference, all $n_total positions (the long pole)"
    ref_run "$serial_all" serial "$all_pos"
    done_step "3 serial-all"
fi

# ---- 4. serial null arm ----
serial_null="$OUT/P-00.serial.null$NULL_POS.json"
if [ -e "$serial_null" ]; then say "skip: $serial_null exists"; else
    step "4 serial null arm, positions 0..$((NULL_POS - 1))"
    ref_run "$serial_null" serial "$null_pos"
    done_step "4 serial-null"
fi

# ---- 5. the controls ----
step "5 controls"
python -u "$control" --a "$serial_all" --b "$batched_all" --ids-a "$ids" --ids-b "$ids" \
    --label C1-Q --threshold 1e-3 --out "$OUT/C1Q.json" 2>&1 | tee -a "$LOG"
python -u "$control" --a "$serial_null" --b "$serial_all" --ids-a "$ids" --ids-b "$ids" \
    --label T6 --require-exact --out "$OUT/T6.json" 2>&1 | tee -a "$LOG"
c1_pass=$(python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['passed'])" "$OUT/C1Q.json" 2>/dev/null)
t6_pass=$(python -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8'))['passed'])" "$OUT/T6.json" 2>/dev/null)
say "C1-Q passed=${c1_pass:-?}  T6 passed=${t6_pass:-?}"
if [ "${c1_pass:-}" != "True" ]; then
    say "STOP -- C1-Q did not pass: the batched arm is NOT licensed for this pair. The grid"
    say "        is redesigned, not re-run against a looser threshold (pre-registration §2, T5)."
    exit 2
fi
if [ "${t6_pass:-}" != "True" ]; then
    say "⚠ T6: the serial arm is not bit-identical with itself -- the instrument has a noise"
    say "  floor; it is banked in T6.json and the report must read no delta below it."
fi
done_step "5 controls"

# ---- 6. phase A, the rest ----
missing=""
for s in P-10 P-20 P-30 P-45 P-41 L-160 L-240 L-660 L-900 L-1120 L-1300 L-1900 L-2500 L-2851; do
    [ -e "$grid/$s.grid.json" ] || missing="$missing${missing:+,}$s"
done
if [ -z "$missing" ]; then say "skip: all 14 remaining grids exist"; else
    step "6 phase A, streams: $missing"
    python -u "$probe" --set unsloth --phase a --out-dir "$grid" --only "$missing" 2>&1 | tee -a "$LOG"
    done_step "6 phase A"
fi
have=$(ls "$grid"/*.grid.json 2>/dev/null | wc -l | tr -d ' ')
say "grids banked: $have of 15"
[ "$have" -eq 15 ] || { say "ABORT -- phase A banked $have of 15 streams; not grading a partial grid"; exit 1; }

# ---- 7. phase B ----
step "7 phase B (freeze tau)"
python -u "$probe" --set unsloth --phase b --out-dir "$grid" --tau-out "$OUT/tau.json" 2>&1 | tee -a "$LOG"
[ -e "$OUT/tau.json" ] || { say "ABORT -- phase B wrote no tau file"; exit 1; }
done_step "7 phase B"

# ---- 8. phase C ----
step "8 phase C (grade)"
python -u "$probe" --set unsloth --phase c --out-dir "$grid" --tau "$OUT/tau.json" --verdicts "$OUT/verdicts.json" 2>&1 | tee -a "$LOG"
[ -e "$OUT/verdicts.json" ] || { say "ABORT -- phase C wrote no verdicts file"; exit 1; }
done_step "8 phase C"
say "B6 DRIVER: all steps banked under $OUT -- read verdicts.json, tau.json, C1Q.json, T6.json"
