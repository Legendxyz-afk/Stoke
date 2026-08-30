#!/bin/sh
# tests/tooling/run_m6_server_sampling_mutants.sh -- A6's SERVER mutation leg.
#
# Usage (from the repo root, under bash/sh -- NOT PowerShell):
#     sh tests/tooling/run_m6_server_sampling_mutants.sh clean
#     sh tests/tooling/run_m6_server_sampling_mutants.sh TEMPERATURE
#     sh tests/tooling/run_m6_server_sampling_mutants.sh report
#     sh tests/tooling/run_m6_server_sampling_mutants.sh          (all of it)
#
# ▶▶ WHY THIS EXISTS, AND WHY IT IS NOT THE SAME AS run_m6_sample_mutants.sh.
# That one grades the ENGINE's sampling wiring through the C API. This one
# grades the SERVER's, and the defect it watches for is different in kind: a
# parameter that is parsed, range-checked, answered with a 200 -- and then
# dropped on the way into the request. Every response such a server sends is
# well-formed, fluent, and wrong only in a way nobody can see from the outside,
# because greedy text and sampled text are both text. The whole surface was a
# refusal table until A6 precisely because that failure mode is invisible.
#
# ▶ SLICED, ONE MUTANT PER INVOCATION, BANKED AS IT GOES. Each mutant needs two
# real servers and a 2.5 GB model load, and long background jobs on this box
# have been killed mid-run with no message three times. Rows land in
# build/m6-srv-mut/ the moment they are known, and `report` reads whatever is
# there.
#
# ⚠ EVERY ROW IS KEYED BY THE CLEAN BINARY'S sha256, and `report` REFUSES to
# combine rows from different ones. A stale binary has an mtime you can look at;
# a stale bank does not, and this project has published stale numbers under a
# fresh commit three times by trusting one.
#
# ▶ THE BANNER GATES THE VERDICT, checked against the BINARY rather than the
# run: a build that silently dropped its -D is a CLEAN binary, the gates pass,
# and this script would report "the fixture did not catch it" about a mutant
# that was never there. Reading it from the binary works whether or not the gate
# it ran got far enough to print anything -- the string exists only inside the
# #ifdef, so its presence is proof the flag took.
#
# ▶▶ AND THE VERDICT READS *WHICH* ASSERTION WENT RED, against a grader declared
# below BEFORE the run. A mutant caught by the wrong assertion leaves the
# assertion written for it still unproven, which is how a seven-mutant leg in
# this project reported 7/7 while one of them had graded nothing at all.
set -u

MODEL="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
EXE=".exe"
[ "${OS:-}" = "Windows_NT" ] || EXE=""
MAKE="mingw32-make"
command -v "$MAKE" >/dev/null 2>&1 || MAKE="make"

BANK="build/m6-srv-mut"
CLEAN_DIR="build/win"
CLEAN_BIN="$CLEAN_DIR/stoke$EXE"

ALL="TEMPERATURE TOPK TOPP PENALTY SEED"

if [ ! -f "$MODEL" ]; then
    echo "SKIP: $MODEL is absent (it is gitignored). Every case this leg grades"
    echo "      is an end-to-end generation through a real server."
    exit 0
fi
mkdir -p "$BANK"

# ---- what each mutant breaks, and which assertion is supposed to notice -----
#
# ⚠ THE GRADER FOR top_k IS THE openai SUITE, NOT THE GOLDEN, and the reason is
# worth writing down: the golden's top_k case sends temperature 0.7 against
# "Name one color.", whose answer is three tokens the model is very sure of. A
# sampler let loose on that prompt may well return "Blue." anyway, so the case
# pins the collapse rather than proving it can fail. The suite's version asks
# for sixteen tokens of open-ended prose at temperature 1.4, where agreeing with
# greedy by chance is not a thing that happens. The golden's verdict is still
# reported below -- as information, not as the grade.
claim_of() {
    case "$1" in
    TEMPERATURE) echo "a temperature reaches the sampler" ;;
    TOPK)        echo "top_k reaches the sampler" ;;
    TOPP)        echo "top_p reaches the sampler" ;;
    PENALTY)     echo "repeat_penalty reaches the sampler" ;;
    SEED)        echo "a client's seed is the seed that is used" ;;
    esac
}
grader_of() {
    case "$1" in
    TEMPERATURE) echo "openai:temperature" ;;
    TOPK)        echo "openai:topk" ;;
    TOPP)        echo "openai:topp" ;;
    PENALTY)     echo "golden:penalty" ;;
    SEED)        echo "openai:seed-repro" ;;
    esac
}

# Which named assertions fired, as tags, most specific first. The openai suite
# stops at its FIRST failure so it reports at most one; the golden accumulates.
tags_openai() { # <log>
    t=""
    grep -qF -- "the sampler is not being reached" "$1" && t="$t openai:temperature"
    grep -qF -- "a seed that does not reproduce is not a seed" "$1" && t="$t openai:seed-repro"
    grep -qF -- "the seed is not reaching the sampler" "$1" && t="$t openai:seed-sensitivity"
    grep -qF -- "top_k 1 leaves one candidate" "$1" && t="$t openai:topk"
    grep -qF -- "a nucleus smaller than the top token" "$1" && t="$t openai:topp"
    echo "${t# }"
}
tags_golden() { # <log>
    t=""
    grep -qF -- "top_k 1 leaves one candidate" "$1" && t="$t golden:topk"
    grep -qF -- "a nucleus cannot remove the most likely token" "$1" && t="$t golden:topp-greedy"
    grep -qF -- "a repeat penalty on an answer that repeats a token" "$1" && t="$t golden:penalty"
    echo "${t# }"
}

sha_of() { sha256sum "$1" 2> /dev/null | cut -c1-16; }

run_gates() { # <builddir> <port-offset> <log-prefix> -- echoes "<gtags>|<otags>"
    rg_dir="$1"
    rg_off="$2"
    rg_pre="$3"
    STOKE_SERVE_PORT=$((18139 + rg_off)) \
        sh tests/serve_golden.sh "$rg_dir" "$EXE" > "$rg_pre.golden.log" 2>&1
    # ⚠ The guard is per RUN and generous: a mutant makes the openai suite abort
    # early, and the gate now stops its server rather than waiting one out, so
    # this only ever bounds a genuine hang.
    STOKE_SERVE_OPENAI_PORT=$((18143 + rg_off)) STOKE_SERVE_GUARD_SECS=420 \
        sh tests/serve_openai.sh "$rg_dir" "$EXE" > "$rg_pre.openai.log" 2>&1
    echo "$(tags_golden "$rg_pre.golden.log")|$(tags_openai "$rg_pre.openai.log")"
}

do_clean() {
    echo "--- clean arm (no -D) ---"
    $MAKE "$CLEAN_BIN" > "$BANK/clean.build.log" 2>&1 || {
        echo "FATAL: the clean build failed; nothing below would mean anything"
        tail -5 "$BANK/clean.build.log"
        exit 1
    }
    if grep -q "MUTANT BUILD -- STOKE_MUT_SRV" "$CLEAN_BIN"; then
        echo "FATAL: the clean binary carries a mutant banner -- a -D leaked in"
        exit 1
    fi
    res=$(run_gates "$CLEAN_DIR" 0 "$BANK/clean")
    gt=${res%%|*}
    ot=${res##*|}
    if ! grep -q "^serve_golden: OK" "$BANK/clean.golden.log" ||
       ! grep -q "^serve_openai: OK (gate item" "$BANK/clean.openai.log"; then
        echo "FATAL: a gate is RED before any mutation -- this leg grades nothing"
        grep -E "FAIL" "$BANK/clean.golden.log" "$BANK/clean.openai.log" | head
        exit 1
    fi
    [ -z "$gt$ot" ] || {
        echo "FATAL: a named assertion fired on the clean arm: $gt $ot"
        exit 1
    }
    sha=$(sha_of "$CLEAN_BIN")
    echo "$sha" > "$BANK/CLEAN.sha"
    echo "clean arm: both gates green, no banner, no named assertion (sha $sha)"
    echo "           so the hooks are inert without a -D"
    echo
}

do_mutant() { # <MUTANT> <index>
    m="$1"
    idx="$2"
    lower=$(echo "$m" | tr 'A-Z_' 'a-z-')
    dir="build/win-mut-srv-$lower"
    bin="$dir/stoke$EXE"

    $MAKE BUILDDIR="$dir" EXTRA_CFLAGS="-DSTOKE_MUT_SRV_DROP_$m" "$bin" \
        > "$BANK/$m.build.log" 2>&1
    if [ ! -f "$bin" ]; then
        echo "$m|-|-|-|BUILD FAILED" > "$BANK/$m.row"
        tail -5 "$BANK/$m.build.log"
        return
    fi

    banner=NO
    grep -q "MUTANT BUILD -- STOKE_MUT_SRV_DROP_$m" "$bin" && banner=YES

    res=$(run_gates "$dir" "$idx" "$BANK/$m")
    gt=${res%%|*}
    ot=${res##*|}
    got="$gt $ot"
    got=$(echo "$got" | sed 's/^ *//; s/ *$//')
    [ -n "$got" ] || got="-"

    want=$(grader_of "$m")
    if [ "$banner" != YES ]; then
        verdict="INVALID (no banner -- the -D did not take)"
    elif [ "$got" = "-" ]; then
        verdict="NOT CAUGHT -- nothing checks that $(claim_of "$m")"
    else
        case " $got " in
        *" $want "*) verdict="caught by its grader: $(claim_of "$m")" ;;
        *) verdict="caught by [$got], NOT by $want -- that assertion is still unproven" ;;
        esac
    fi
    printf '%s|%s|%s|%s|%s\n' "$m" "$banner" "$want" "$got" "$verdict" \
        > "$BANK/$m.row"
    printf '%-12s %-7s %s\n' "$m" "$banner" "$verdict"
}

do_report() {
    echo
    printf '%-12s %-7s %-19s %s\n' MUTANT BANNER GRADER "FIRED / VERDICT"
    fail=0
    [ -f "$BANK/CLEAN.sha" ] || { echo "no clean arm banked -- run 'clean' first"; exit 1; }
    for m in $ALL; do
        if [ ! -f "$BANK/$m.row" ]; then
            printf '%-12s %-7s %-19s %s\n' "$m" "-" "$(grader_of "$m")" "NOT RUN"
            fail=1
            continue
        fi
        row=$(cat "$BANK/$m.row")
        b=$(echo "$row" | cut -d'|' -f2)
        w=$(echo "$row" | cut -d'|' -f3)
        g=$(echo "$row" | cut -d'|' -f4)
        v=$(echo "$row" | cut -d'|' -f5)
        printf '%-12s %-7s %-19s %s\n' "$m" "$b" "$w" "$g"
        printf '%-12s %-7s %-19s   %s\n' "" "" "" "$v"
        case "$v" in caught\ by\ its\ grader:*) ;; *) fail=1 ;; esac
    done
    echo
    echo "clean binary sha256[0:16] = $(cat "$BANK/CLEAN.sha")"
    if [ "$fail" -ne 0 ]; then
        echo "MUTATION LEG: FAILED -- the sampling surface does not yet earn what it claims"
        exit 1
    fi
    echo "MUTATION LEG: GREEN -- every parameter's delivery can fail, and does,"
    echo "                       in the assertion that is supposed to notice"
    exit 0
}

echo "=== M6 rung A6: server sampling mutation leg ==="
if [ $# -eq 0 ]; then
    do_clean
    i=1
    for m in $ALL; do do_mutant "$m" "$i"; i=$((i + 1)); done
    do_report
fi
for arg in "$@"; do
    case "$arg" in
    clean)  do_clean ;;
    report) do_report ;;
    *)
        i=1
        for m in $ALL; do
            [ "$m" = "$arg" ] && break
            i=$((i + 1))
        done
        case " $ALL " in
        *" $arg "*) do_mutant "$arg" "$i" ;;
        *) echo "unknown mutant '$arg' (have: $ALL clean report)"; exit 2 ;;
        esac
        ;;
    esac
done
