#!/bin/sh
# tests/abi_sample.sh -- script gate: drives SAMPLING through the public ABI
# (src/stoke.h) on the real Qwen3 model. M6 rung A6.
#
# Same policy as abi_smoke.sh: models/ and *.gguf are gitignored, so this SKIPS
# cleanly and loudly (exit 0) when the model is absent, and gates the build when
# it is present.
#
# Usage: sh tests/abi_sample.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: abi_sample.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$builddir/abi_sample$exe"

golden_lib_require_model "$model" "abi_sample"
golden_lib_require_binary "$bin" "abi_sample"

out="$builddir/abi_sample.out"
if ! "$bin" "$model" > "$out" 2>&1; then
    echo "abi_sample: FAIL -- $bin exited nonzero" >&2
    sed 's/^/  /' "$out" >&2
    exit 1
fi

# ⚠ EVERY CASE IS CHECKED BY NAME, not just the OK line. The binary counts its
# own cases and refuses to print OK unless all seven ran -- but a future edit that
# returns early from main would satisfy both that counter and a bare grep for
# OK. Seven greps cost nothing and cannot be satisfied by a shorter run.
for c in A B C D E F G; do
    if ! grep -q "^abi_sample: \[$c\]" "$out"; then
        echo "abi_sample: FAIL -- case [$c] did not report" >&2
        sed 's/^/  /' "$out" >&2
        exit 1
    fi
done
if ! grep -q "^abi_sample: OK (7 cases)$" "$out"; then
    echo "abi_sample: FAIL -- exited 0 without its OK line for all seven cases" >&2
    sed 's/^/  /' "$out" >&2
    exit 1
fi

sed 's/^/  /' "$out"
echo "abi_sample: OK (sampling ran through src/stoke.h)"
exit 0
