#!/bin/sh
# tests/att_simd_twins.sh -- attention's AVX2 arm against its scalar twin, on
# a real model, with the GQA group ASSERTED rather than assumed.
#
# ⚠⚠⚠ THE FACT THIS GATE EXISTS TO ENCODE, AND IT WAS FOUND BY MUTATION AFTER
# THE FIRST VERSION OF THIS FILE HAD ALREADY GONE GREEN:
#
#     ON Llama-3.2-3B (G=3) THE AVX2 ATTENTION ARM IS DEAD CODE.
#
# Both SIMD blocks in model.c's att_kvheads_run are `for (; g + 4u <= G; g +=
# 4u)`. At G=4 they carry everything; at G=3 they run ZERO times and the scalar
# remainder carries 100% of attention. So on Llama, `STOKE_ATT_SIMD=0` and the
# default arm are THE SAME CODE, and "the two arms agree byte for byte" is
# true, cheap and completely empty.
#
# Measured, not reasoned: perturbing att_score4_avx2's scale by 1% changed
# Llama's greedy ids NOT AT ALL and changed Qwen3's in 20 of 48 positions
# (2026-09-08). That is what makes leg 1 below evidence and leg 2 a
# change-detector, and it is why leg 1 runs on the QWEN file.
#
# ⚠ The 2026-09-05 B5 note says "STOKE_ATT_SIMD=0 and the AVX2 arm now produce
# byte-identical output" as part of showing the G=3 fallback is live. That
# sentence is TRUE and it is NOT twin evidence on that model -- the note's own
# neighbouring finding (that the remainder carries 100% at G=3) is exactly why.
# See [[fixture-can-be-blind-to-the-defect]].
#
# ⚠⚠ AND THE NON-VACUITY LEG IS NOT OPTIONAL. The arms are bit-identical by
# construction, so a green leg 1 is also exactly what you would see if
# STOKE_ATT_SIMD were IGNORED and both arms were the same one. Leg 3 proves the
# knob is read by requiring an out-of-range value to be REFUSED -- standing
# rule 10's idiom, and available only because src/model.c's env_u32 was made to
# refuse instead of silently defaulting on 2026-09-08.
#
# ⭐ Platform-independent by construction, so unlike llama_gen_golden.sh this is
# NOT opt-in: every comparison here is stoke against itself on one box with one
# libm, never against a golden generated somewhere else.
#
# Usage: sh tests/att_simd_twins.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: att_simd_twins.sh <builddir> [exe-suffix]}"
exe="${2:-}"
stoke_bin="$builddir/stoke$exe"
prompt="tests/data/llama_gen/short3.txt"
ngen=48
fail=0

golden_lib_require_binary "$stoke_bin" "att_simd_twins"
golden_lib_require_data_file "$prompt" "att_simd_twins" "prompt"

# gqa_group <model> -- head_count / head_count_kv, read from the FILE via
# `stoke info`. The whole gate turns on this number, so it is measured per run
# rather than written in a comment that a re-quantized file could falsify.
gqa_group() {
    _h=$("$stoke_bin" info "$1" 2>/dev/null \
         | sed -n 's/.*\.attention\.head_count (UINT32) = \([0-9]*\).*/\1/p' \
         | head -1)
    _k=$("$stoke_bin" info "$1" 2>/dev/null \
         | sed -n 's/.*\.attention\.head_count_kv (UINT32) = \([0-9]*\).*/\1/p' \
         | head -1)
    if [ -z "${_h:-}" ] || [ -z "${_k:-}" ] || [ "$_k" -eq 0 ]; then
        echo ""
        return 1
    fi
    echo $((_h / _k))
}

# ids <model> <simd-value-or-empty> <outfile>
ids() {
    if [ -z "$2" ]; then
        "$stoke_bin" run "$1" --raw -n "$ngen" -f "$prompt" 2>/dev/null \
            | sed -n '1,/^---$/p' | sed '$d' > "$3"
    else
        STOKE_ATT_SIMD="$2" "$stoke_bin" run "$1" --raw -n "$ngen" \
            -f "$prompt" 2>/dev/null | sed -n '1,/^---$/p' | sed '$d' > "$3"
    fi
    [ -s "$3" ]
}

# twin_leg <label> <model> <want-G> <meaning>
twin_leg() {
    _label="$1"; _model="$2"; _wantg="$3"; _meaning="$4"
    if [ ! -f "$_model" ]; then
        if [ "${STOKE_REQUIRE_GOLDEN:-0}" = "1" ]; then
            echo "att_simd_twins: FAILED -- $_model absent but" \
                 "STOKE_REQUIRE_GOLDEN=1" >&2
            return 1
        fi
        echo "att_simd_twins: SKIP $_label -- $_model not present"
        return 0
    fi
    _g=$(gqa_group "$_model") || {
        echo "att_simd_twins: FAILED -- could not read head counts from $_model" >&2
        return 1
    }
    if [ "$_g" != "$_wantg" ]; then
        # The leg's MEANING is a function of G. A file whose group changed must
        # stop the gate, not quietly re-label it.
        echo "att_simd_twins: FAILED -- $_label expects G=$_wantg, $_model" \
             "reports G=$_g. This gate's legs mean different things at" \
             "different G (the SIMD blocks are 'g + 4 <= G'), so the" \
             "assumption is checked, not assumed" >&2
        return 1
    fi
    ids "$_model" "" "$builddir/att_twins_${_label}_a.ids" || {
        echo "att_simd_twins: FAILED -- $_label default arm produced no ids" >&2
        return 1
    }
    ids "$_model" 0 "$builddir/att_twins_${_label}_b.ids" || {
        echo "att_simd_twins: FAILED -- $_label scalar arm produced no ids" >&2
        return 1
    }
    if diff -u "$builddir/att_twins_${_label}_a.ids" \
               "$builddir/att_twins_${_label}_b.ids" \
               > "$builddir/att_twins_${_label}.diff" 2>&1; then
        _n=$(grep -c . "$builddir/att_twins_${_label}_a.ids")
        echo "att_simd_twins: $_label OK (G=$_g, $_n ids) -- $_meaning"
        rm -f "$builddir/att_twins_${_label}_a.ids" \
              "$builddir/att_twins_${_label}_b.ids" \
              "$builddir/att_twins_${_label}.diff"
        return 0
    fi
    echo "att_simd_twins: FAILED -- $_label: the arms disagree at G=$_g." \
         "diff in $builddir/att_twins_${_label}.diff" >&2
    return 1
}

# ---- leg 1: the REAL twin, at G=4, where the AVX2 arm actually runs --------
twin_leg qwen4 "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" 4 \
    "the AVX2 score/accum kernels agree with their scalar twins" || fail=1

# ---- leg 2: G=3, a change-detector and explicitly NOT twin evidence --------
twin_leg llama3 "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf" 3 \
    "CHANGE-DETECTOR ONLY: at G=3 both arms are the same code, so this green is expected and carries no twin evidence" || fail=1

# ---- leg 3: the knob must be READ ------------------------------------------
live_model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
[ -f "$live_model" ] || live_model="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
if [ -f "$live_model" ]; then
    STOKE_ATT_SIMD=2 "$stoke_bin" run "$live_model" --raw -n 1 -f "$prompt" \
        > /dev/null 2> "$builddir/att_twins_live.err"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "att_simd_twins: FAILED -- STOKE_ATT_SIMD=2 was ACCEPTED (exit" \
             "0). The knob is not read, so leg 1 compared one arm with" \
             "itself and its green means nothing" >&2
        fail=1
    elif ! grep -q "STOKE_ATT_SIMD" "$builddir/att_twins_live.err"; then
        echo "att_simd_twins: FAILED -- STOKE_ATT_SIMD=2 exited $rc but the" \
             "message does not name the knob, so the refusal may be about" \
             "something else" >&2
        cat "$builddir/att_twins_live.err" >&2
        fail=1
    else
        echo "att_simd_twins: leg 3 OK -- STOKE_ATT_SIMD=2 refused (exit $rc)"
        rm -f "$builddir/att_twins_live.err"
    fi

    # ---- leg 4: the REMOVED knobs must be refused too ----------------------
    # att_env_init has promised since M5 that STOKE_ATT_LANES / STOKE_ATT_GQA
    # are "deliberately not accepted-and-ignored". Nothing enforced it until
    # 2026-09-08, and two tooling scripts in this repo still set them
    # (tests/tooling/run_att_par_verify.ps1, run_m5_e2_verify.ps1), so the trap
    # was armed rather than hypothetical.
    for dead in STOKE_ATT_LANES STOKE_ATT_GQA; do
        env "$dead=1" "$stoke_bin" run "$live_model" --raw -n 1 -f "$prompt" \
            > /dev/null 2> "$builddir/att_twins_dead.err"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "att_simd_twins: FAILED -- $dead was ACCEPTED (exit 0)" >&2
            fail=1
        elif ! grep -q "$dead" "$builddir/att_twins_dead.err"; then
            echo "att_simd_twins: FAILED -- $dead exited $rc without naming" \
                 "the knob" >&2
            fail=1
        fi
    done
    rm -f "$builddir/att_twins_dead.err"
    [ "$fail" -eq 0 ] && echo "att_simd_twins: leg 4 OK -- both removed knobs refused"
else
    echo "att_simd_twins: SKIP legs 3-4 -- no model present"
fi

if [ "$fail" -ne 0 ]; then
    echo "att_simd_twins: FAILED" >&2
    exit 1
fi
echo "att_simd_twins: OK"
exit 0
