#!/bin/sh
# tests/run_tests.sh — run every built tests/test_*.c binary, aggregate exit
# codes. The Makefile builds the binaries before invoking this; this script
# only runs and reports.
#
# Usage: sh tests/run_tests.sh <builddir> [exe-suffix]
set -u

builddir="${1:?usage: run_tests.sh <builddir> [exe-suffix]}"
exe="${2:-}"

echo "run_tests: builddir=$builddir exe='$exe'"

fail=0
ran=0

for src in tests/test_*.c; do
    [ -e "$src" ] || continue
    name=${src#tests/}
    name=${name%.c}
    bin="$builddir/$name$exe"

    if [ ! -f "$bin" ]; then
        echo "run_tests: missing binary for $src (expected $bin)" >&2
        fail=1
        continue
    fi

    ran=$((ran + 1))
    echo "=== $name ==="
    if "$bin"; then
        :
    else
        status=$?
        echo "run_tests: $name FAILED (exit $status)" >&2
        fail=1
    fi
done

if [ "$ran" -eq 0 ]; then
    echo "run_tests: no test binaries found (tests/test_*.c matched nothing built)" >&2
    exit 1
fi

if [ "$fail" -ne 0 ]; then
    echo "run_tests: FAILED"
    exit 1
fi

echo "run_tests: all $ran test binaries passed"
exit 0
