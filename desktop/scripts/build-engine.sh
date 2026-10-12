#!/usr/bin/env sh
# Builds lorren-engine.exe: the FROZEN engine (src/ exactly as at b1ee6b3), statically linked.
# Run from Git Bash (the Makefile needs sh + mkdir -p). Never edits src/ or the Makefile:
# BUILDDIR and LDFLAGS are command-line overrides.
set -eu
FREEZE=b1ee6b3
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
if ! git diff --quiet "$FREEZE" -- src/; then
    echo "build-engine: src/ differs from the frozen $FREEZE -- refusing" >&2
    exit 1
fi
PATH="/c/msys64/ucrt64/bin:$PATH" mingw32-make -j4 BUILDDIR=build/lorren-engine LDFLAGS=-static build/lorren-engine/stoke.exe
OUT="$ROOT/desktop/src-tauri/binaries/lorren-engine-x86_64-pc-windows-msvc.exe"
mkdir -p "$(dirname "$OUT")"
cp build/lorren-engine/stoke.exe "$OUT"
if PATH="/c/msys64/ucrt64/bin:$PATH" objdump -p "$OUT" | grep -qiE 'DLL Name: (libwinpthread|libgcc|libstdc)'; then
    echo "build-engine: still imports a MinGW runtime DLL -- not static" >&2
    exit 1
fi
case "$(file -b "$OUT")" in
    PE32+*) ;;
    *) echo "build-engine: $OUT is not a PE32+ executable" >&2; exit 1 ;;
esac
echo "build-engine: OK $OUT"
