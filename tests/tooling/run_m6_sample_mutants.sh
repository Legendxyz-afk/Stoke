#!/bin/sh
# tests/tooling/run_m6_sample_mutants.sh -- M6 rung A6's mutation leg.
#
# Usage (from the repo root, under bash/sh -- NOT PowerShell):
#     sh tests/tooling/run_m6_sample_mutants.sh
#
# ▶▶ WHY THIS EXISTS. tests/abi_sample.c passed all seven cases the first time
# it ran. That is exactly when a fixture is least trustworthy. Two of its cases
# ([F] and [G]) exist solely to catch wiring defects that produce FLUENT TEXT
# and no error, so neither can be checked by reading the output -- the only way
# to know they work is to break the thing they watch and see them go red.
#
# ▶ THE BANNER GATES THE VERDICT, and is checked before any result is read.
# Inferring "the -D took effect" from the failure alone is circular in the one
# direction that matters: a build that silently dropped the flag is a CLEAN
# binary, abi_sample passes, and this script would report "the fixture did not
# catch it" -- a false finding about the fixture caused by a mutant that was
# never there. Same argument as tests/tooling/run_m6_http_mutants.sh.
#
# ▶▶ AND THE VERDICT READS *WHICH* CASE WENT RED, not merely that something
# did. A mutant caught by the wrong case is a mutant whose intended grader is
# still unproven -- and on 2026-08-28 a seven-mutant leg in this project
# reported 7/7 CAUGHT while one of them had graded nothing at all, because a
# redundant check elsewhere rejected the input for a different reason.
#
# ▶ ONE BUILDDIR PER MUTANT, never shared: a command-line EXTRA_CFLAGS does not
# touch the Makefile's mtime, so a shared directory would relink stale objects.
set -u

MODEL="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
EXE=".exe"
[ "${OS:-}" = "Windows_NT" ] || EXE=""
MAKE="mingw32-make"
command -v "$MAKE" >/dev/null 2>&1 || MAKE="make"

if [ ! -f "$MODEL" ]; then
    echo "SKIP: $MODEL is absent (it is gitignored). This leg needs the real"
    echo "      model -- every case it grades is an end-to-end generation."
    exit 0
fi

MUTANTS="NO_SLOT_RELEASE STALE_HISTORY"

# What each mutant breaks, which case is supposed to notice, and therefore what
# its being caught proves.
claim_of() {
    case "$1" in
    NO_SLOT_RELEASE) echo "a finished request gives its sampling slot back" ;;
    STALE_HISTORY)   echo "a reused slot carries no history from its last occupant" ;;
    esac
}
grader_of() {
    case "$1" in
    NO_SLOT_RELEASE) echo "F" ;;
    STALE_HISTORY)   echo "G" ;;
    esac
}

echo "=== M6 rung A6: sampling mutation leg ==="
echo

# The clean arm first. A mutation leg run against a tree that is already red
# grades nothing.
echo "--- clean arm (no -D) ---"
$MAKE "build/win/abi_sample$EXE" >/dev/null 2>&1 || {
    echo "FATAL: the clean build failed; nothing below would mean anything"
    exit 1
}
clean_log="build/win/abi_sample.mut-clean.txt"
if ! ./"build/win/abi_sample$EXE" "$MODEL" > "$clean_log" 2>&1; then
    echo "FATAL: the clean binary is RED before any mutation:"
    grep 'FAIL' "$clean_log" | head
    exit 1
fi
if grep -q 'MUTANT' "$clean_log"; then
    echo "FATAL: the clean binary printed a mutant banner -- a -D leaked in"
    exit 1
fi
tail -1 "$clean_log"
echo "clean arm: green, and no banner (so the hooks are inert without a -D)"
echo

fail=0
printf '%-17s %-8s %-7s %s\n' MUTANT BANNER CAUGHT VERDICT
for m in $MUTANTS; do
    lower=$(echo "$m" | tr 'A-Z_' 'a-z-')
    dir="build/win-mut-sample-$lower"
    bin="$dir/abi_sample$EXE"
    log="$dir/abi_sample.out"

    $MAKE BUILDDIR="$dir" EXTRA_CFLAGS="-DSTOKE_MUT_$m" "$bin" \
        >"$dir.build.log" 2>&1
    if [ ! -f "$bin" ]; then
        printf '%-17s %-8s %-7s %s\n' "$m" "-" "-" "BUILD FAILED"
        tail -5 "$dir.build.log"
        fail=1
        continue
    fi

    ./"$bin" "$MODEL" >"$log" 2>&1
    rc=$?

    banner=NO
    grep -q "MUTANT src/stoke.c -- STOKE_MUT_$m" "$log" && banner=YES

    # WHICH case went red. abi_sample stops at its first failure, so the last
    # FAIL line is the grading one.
    want=$(grader_of "$m")
    # ⚠ sed, NOT `tr -d 'FAIL[] '`, which is what this line said first: tr
    # deletes every listed character EVERYWHERE, so "FAIL [F]" lost its F and
    # came back empty. It reported "[F] is still unproven" about a mutant [F]
    # had caught, at the one moment the leg exists to be trusted.
    got=$(grep -o 'FAIL \[[A-G]\]' "$log" | head -1 | sed 's/.*\[\(.\)\]/\1/')
    [ -n "$got" ] || got="-"

    if [ "$banner" != YES ]; then
        verdict="INVALID (no banner -- the -D did not take)"
        fail=1
    elif [ "$rc" -eq 0 ]; then
        verdict="NOT CAUGHT -- nothing checks that $(claim_of "$m")"
        fail=1
    elif [ "$got" != "$want" ]; then
        verdict="caught by [$got], NOT by its grader [$want] -- [$want] is still unproven"
        fail=1
    else
        verdict="caught by [$got]: $(claim_of "$m")"
    fi
    printf '%-17s %-8s %-7s %s\n' "$m" "$banner" "[$got]" "$verdict"
done

echo
if [ "$fail" -ne 0 ]; then
    echo "MUTATION LEG: FAILED -- abi_sample does not yet earn what it claims"
    exit 1
fi
echo "MUTATION LEG: GREEN -- every wiring claim can fail, and does, in the case"
echo "                       that is supposed to notice"
exit 0
