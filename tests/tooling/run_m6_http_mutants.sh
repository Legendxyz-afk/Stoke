#!/bin/sh
# tests/tooling/run_m6_http_mutants.sh -- M6 gate item [6]'s mutation leg.
#
# Usage (from the repo root, under bash/sh -- NOT PowerShell):
#     sh tests/tooling/run_m6_http_mutants.sh
#
# ▶▶ WHY THIS EXISTS. tests/data/http_corpus/ got all 51 of its predicted
# verdicts right the first time it ran. That is exactly when a fixture is least
# trustworthy: a corpus that agrees with the parser proves nothing until
# something proves it can DISAGREE. This project has the scar -- on 2026-08-25 a
# byte-compare over M0's own prompts would have passed while two mutants that
# destroyed the state under test produced byte-identical output.
#
# So each mutant below breaks ONE claim the corpus rests on, and the run is a
# pass only if EVERY one is caught.
#
# ▶ THE BANNER IS THE NON-CIRCULAR PROOF THE MUTATION IS LIVE, and it is checked
# before any verdict is read. Inferring "the -D took effect" from the failures
# alone is circular in the one direction that matters: a build that silently
# dropped the flag would produce a CLEAN binary, the corpus would pass, and this
# script would report "the corpus did not catch it" -- a false finding about the
# corpus caused by a mutant that was never there. Same argument as M5's
# STOKE_Q6K_KERNEL=not-a-kernel refusal.
#
# ▶ ONE BUILDDIR PER MUTANT, never shared. A command-line EXTRA_CFLAGS does not
# touch the Makefile's mtime, so reusing one directory would relink yesterday's
# objects and report on a binary that does not exist.
set -u

CLEAN_BIN="build/win/test_http.exe"
EXE=".exe"
[ "${OS:-}" = "Windows_NT" ] || EXE=""
MAKE="mingw32-make"
command -v "$MAKE" >/dev/null 2>&1 || MAKE="make"

MUTANTS="BARE_CR TE DUP_CL HEAD_CAP SPACE_COLON BODY_CAP NAME_LEN"

# What each mutant breaks, and therefore what its being caught proves.
claim_of() {
    case "$1" in
    BARE_CR)     echo "a lone CR in a head line is rejected" ;;
    TE)          echo "Transfer-Encoding is refused, never ignored" ;;
    DUP_CL)      echo "a repeated Content-Length is refused" ;;
    HEAD_CAP)    echo "the head byte budget is enforced" ;;
    SPACE_COLON) echo "the REASON is asserted, not just the status" ;;
    BODY_CAP)    echo "an over-large declared body is refused" ;;
    NAME_LEN)    echo "header name slices are exact" ;;
    esac
}

echo "=== M6 gate [6]: corpus mutation leg ==="
echo

# The clean arm first. A mutation leg run against a tree that is already red
# grades nothing.
echo "--- clean arm (no -D) ---"
$MAKE "$CLEAN_BIN" >/dev/null 2>&1 || {
    echo "FATAL: the clean build failed; nothing below would mean anything"
    exit 1
}
if ! ./"$CLEAN_BIN" >/tmp/m6_mut_clean.txt 2>&1; then
    echo "FATAL: the clean binary is RED before any mutation:"
    grep '^FAIL' /tmp/m6_mut_clean.txt | head
    exit 1
fi
tail -1 /tmp/m6_mut_clean.txt
if grep -q 'MUTANT' /tmp/m6_mut_clean.txt; then
    echo "FATAL: the clean binary printed a mutant banner -- a -D leaked in"
    exit 1
fi
echo "clean arm: green, and no banner (so the hooks are inert without a -D)"
echo

fail=0
printf '%-12s %-8s %-9s %-8s %s\n' MUTANT BANNER CORPUS SWEEP VERDICT
for m in $MUTANTS; do
    lower=$(echo "$m" | tr 'A-Z_' 'a-z-')
    dir="build/win-mut-http-$lower"
    bin="$dir/test_http$EXE"
    log="/tmp/m6_mut_$lower.txt"

    $MAKE BUILDDIR="$dir" EXTRA_CFLAGS="-DSTOKE_MUT_HTTP_$m" "$bin" \
        >"$log.build" 2>&1
    if [ ! -f "$bin" ]; then
        printf '%-12s %-8s %-9s %-8s %s\n' "$m" "-" "-" "-" "BUILD FAILED"
        tail -5 "$log.build"
        fail=1
        continue
    fi

    ./"$bin" >"$log" 2>&1
    rc=$?

    banner=NO
    grep -q "MUTANT src/http.c" "$log" && banner=YES
    corpus=miss
    grep -q "FAIL corpus" "$log" && corpus=CAUGHT
    sweep=miss
    grep -q "FAIL sweep" "$log" && sweep=CAUGHT

    # The banner gates the verdict: without it there is no mutant to grade.
    if [ "$banner" != YES ]; then
        verdict="INVALID (no banner -- the -D did not take)"
        fail=1
    elif [ "$rc" -eq 0 ]; then
        verdict="NOT CAUGHT -- $(claim_of "$m")"
        fail=1
    elif [ "$corpus" != CAUGHT ] && [ "$sweep" != CAUGHT ]; then
        verdict="caught by unit tests only (corpus blind)"
        fail=1
    else
        verdict="caught: $(claim_of "$m")"
    fi
    printf '%-12s %-8s %-9s %-8s %s\n' "$m" "$banner" "$corpus" "$sweep" "$verdict"
done

echo
if [ "$fail" -ne 0 ]; then
    echo "MUTATION LEG: FAILED -- the corpus does not yet earn gate item [6]"
    exit 1
fi
echo "MUTATION LEG: GREEN -- every claim the corpus rests on can fail, and does"
exit 0
