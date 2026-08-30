#!/bin/sh
# tests/abi_firsttoken.sh -- script gate: a request submitted MID-STEP must not
# have its first token chosen greedily. M7 rung B3.
#
# Same policy as abi_smoke.sh: models/ and *.gguf are gitignored, so this SKIPS
# cleanly and loudly (exit 0) when the model is absent, and gates the build when
# it is present.
#
# Usage: sh tests/abi_firsttoken.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: abi_firsttoken.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$builddir/abi_firsttoken$exe"

golden_lib_require_model "$model" "abi_firsttoken"
golden_lib_require_binary "$bin" "abi_firsttoken"

out="$builddir/abi_firsttoken.out"
if ! "$bin" "$model" > "$out" 2>&1; then
    echo "abi_firsttoken: FAIL -- $bin exited nonzero" >&2
    sed 's/^/  /' "$out" >&2
    exit 1
fi

# ⚠ THE OK LINE IS NOT ENOUGH, and neither is "0 differing". A run in which
# the racing thread never stepped, or in which every request was drained before
# it could be raced, agrees with itself perfectly and proves nothing -- so the
# binary prints its step count and this reads it. Under the shipping build the
# window is nanoseconds and the pairs agree because nothing raced; the case that
# earns the gate is the mutant below.
if ! grep -qE "^abi_firsttoken: [0-9]+ pairs, 0 differing, [0-9]+ steps" "$out"; then
    echo "abi_firsttoken: FAIL -- no clean pair/step summary line" >&2
    sed 's/^/  /' "$out" >&2
    exit 1
fi
steps=$(sed -n 's/^abi_firsttoken: [0-9]* pairs, [0-9]* differing, \([0-9]*\) steps.*//p' "$out")
if [ "${steps:-0}" -lt 1 ]; then
    echo "abi_firsttoken: FAIL -- the racing thread took $steps steps; nothing was raced" >&2
    exit 1
fi
if ! grep -q "^abi_firsttoken: OK (a mid-step submit keeps its first token)$" "$out"; then
    echo "abi_firsttoken: FAIL -- exited 0 without its OK line" >&2
    sed 's/^/  /' "$out" >&2
    exit 1
fi

sed 's/^/  /' "$out"
echo "abi_firsttoken: OK (a mid-step submit keeps its first token)"
exit 0
