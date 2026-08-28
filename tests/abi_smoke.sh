#!/bin/sh
# tests/abi_smoke.sh -- script gate: drives a real generation through the
# PUBLIC ABI (src/stoke.h) on the real 2.5GB Qwen3 model.
#
# Same policy as the golden gates and for the same reason: models/ and *.gguf
# are gitignored, so the model exists on the reference box and not in a fresh
# clone. This SKIPS cleanly and loudly (exit 0) when the model is absent, and
# gates the build when it is present. tests/abi_tu.c and tests/abi_tu.cpp need
# no model and always run.
#
# Usage: sh tests/abi_smoke.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: abi_smoke.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$builddir/abi_smoke$exe"

golden_lib_require_model "$model" "abi_smoke"
golden_lib_require_binary "$bin" "abi_smoke"

out="$builddir/abi_smoke.out"
if ! "$bin" "$model" > "$out" 2>&1; then
    echo "abi_smoke: FAIL -- $bin exited nonzero" >&2
    sed 's/^/  /' "$out" >&2
    exit 1
fi

# The binary asserts its own invariants and exits nonzero on any of them; this
# only re-checks the two lines whose ABSENCE would mean it silently did less
# than it claims -- a pass that printed no generation is the failure mode a
# green exit code cannot distinguish on its own.
if ! grep -q "^abi_smoke: OK$" "$out"; then
    echo "abi_smoke: FAIL -- exited 0 without printing its OK line" >&2
    sed 's/^/  /' "$out" >&2
    exit 1
fi
if ! grep -q "^abi_smoke: bad path trapped" "$out"; then
    echo "abi_smoke: FAIL -- the trapped-open check did not run" >&2
    sed 's/^/  /' "$out" >&2
    exit 1
fi

sed 's/^/  /' "$out"
echo "abi_smoke: OK (a real generation ran through src/stoke.h)"
exit 0
