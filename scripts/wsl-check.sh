#!/bin/bash
# Sanitizer matrix, run inside WSL/Linux:  bash scripts/wsl-check.sh
# Assumes the native Windows leg (mingw32-make test) has already been run
# separately -- this script covers only the WSL/Linux leg: plain, ASan,
# TSan.
set -e
cd "$(dirname "$0")/.."

echo "=== [1/3] make test (plain WSL/Linux gcc build) ==="
make test

echo "=== [2/3] make SAN=asan test ==="
make SAN=asan test

echo "=== [3/3] make SAN=tsan test ==="
# setarch -R: disable ASLR for this invocation only. TSan's shadow-memory
# mapping can fail under some kernels' default high-entropy ASLR -- same
# lesson ../loom/scripts/wsl-check.sh already paid for on this exact box.
setarch x86_64 -R make SAN=tsan test

echo "ALL GREEN"
