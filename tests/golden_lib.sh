# tests/golden_lib.sh -- shared body for the four golden-comparison gate
# scripts (golden_info.sh, tok_golden.sh, tok_special_golden.sh,
# unicode_sweep_golden.sh): the model-check / STOKE_REQUIRE_GOLDEN
# SKIP-vs-FAIL policy, the stoke-binary/corpus/golden presence checks, and
# the run+diff+cleanup body they'd otherwise each carry as their own
# ~100-line near-verbatim copy. `.`-sourced (POSIX `.`, not bash-only
# `source`) by each gate script, which then calls these functions
# supplying its own name/corpus/golden/oracle args -- adding a fifth gate
# means sourcing this file and calling these functions, never copying or
# re-touching the skip/require policy itself (one policy site).
#
# POSIX `sh` ONLY -- this is sourced by both msys `sh` (Windows) and
# WSL/Linux `sh` (dash-ish), so no bash-only syntax anywhere in it: no
# `[[`, no `local`, no arrays, no `$RANDOM`/`$(({}))`-style bashisms. Every
# function below uses `gl_`-prefixed variable names specifically because
# POSIX shell functions share the caller's global namespace (no `local`) --
# the prefix is this file's only guard against clobbering a gate script's
# own `model`/`golden`/`corpus`/... variables.
#
# Every message shape a gate script prints while GREEN (the loud SKIP line
# and each gate's own `<name>: OK (...)` line) is reproduced byte-for-byte
# from before this refactor -- those lines are the readable record of the
# green bar. The FAILED-path messages keep the same shape (`<name>: FAILED
# -- ...`) but are no longer hand-duplicated per gate; tok_special_golden's
# two diff-mismatch messages lose their old "plain tokenize "/"--special "
# prefix words specifically (now the same generic "output differs from
# <golden>" every gate uses) -- the golden path in that message already
# says unambiguously which of the two directions failed, so nothing
# diagnostic is lost, and every gate's failure message is now one shape
# instead of two.

# golden_lib_require_model <model-path> <gate-name>
# Model-absence policy, identical across every gate: a missing model is a
# loud SKIP (exit 0) UNLESS STOKE_REQUIRE_GOLDEN=1, in which case it's a
# hard FAILED (exit 1) instead -- scripts/wsl-check.sh exports
# STOKE_REQUIRE_GOLDEN=1 on the reference box (which always has the real
# model) specifically so a moved/renamed model can never silently
# downgrade every golden gate to a no-op. Exits the whole gate script
# directly in both branches when the model is absent -- there is nothing
# left for any caller to do either way -- and simply returns when present.
golden_lib_require_model() {
    gl_model="$1"
    gl_name="$2"
    if [ ! -f "$gl_model" ]; then
        if [ "${STOKE_REQUIRE_GOLDEN:-0}" = "1" ]; then
            echo "$gl_name: FAILED -- $gl_model not present but" \
                 "STOKE_REQUIRE_GOLDEN=1 (this box is supposed to have the" \
                 "real model; a moved/missing model must not silently" \
                 "downgrade this gate to a no-op)" >&2
            exit 1
        fi
        echo "$gl_name: SKIP -- $gl_model not present (models/ is" \
             "gitignored; this gate only runs on boxes that have the real" \
             "model file)"
        exit 0
    fi
}

# golden_lib_require_binary <stoke-bin-path> <gate-name>
# Exits 1 if the built `stoke` binary isn't there yet.
golden_lib_require_binary() {
    if [ ! -f "$1" ]; then
        echo "$2: missing binary $1 (build it before running this gate)" >&2
        exit 1
    fi
}

# golden_lib_require_data_file <path> <gate-name> <label>
# Exits 1 if a corpus/golden input file is missing: "<gate-name>: missing
# <label> <path>" (<label> e.g. "corpus file" / "golden file").
golden_lib_require_data_file() {
    if [ ! -f "$1" ]; then
        echo "$2: missing $3 $1" >&2
        exit 1
    fi
}

# golden_lib_run_and_diff <gate-name> <golden> <out-file> <diff-file> <exec-desc> <argv...>
# Runs <argv...> with stdout redirected to <out-file>. A nonzero exit
# prints "<gate-name>: FAILED -- '<exec-desc>' exited <rc>" and returns 1
# (matching every gate's original run-step message exactly). On a zero
# exit, diffs <out-file> against <golden> into <diff-file>; a mismatch
# prints "<gate-name>: FAILED -- output differs from <golden>" plus the
# diff itself, and returns 1; a clean match removes both temp files and
# returns 0.
#
# Never exits on failure -- only returns 1 -- so a gate with a single
# invocation can `|| exit 1` immediately, while a gate comparing MULTIPLE
# (cmd,golden) pairs (tok_special_golden.sh's plain + --special legs) can
# run every leg, accumulate a `fail` flag, and report every failing leg in
# one pass instead of stopping at the first.
golden_lib_run_and_diff() {
    gl_name="$1"
    gl_golden="$2"
    gl_out="$3"
    gl_diff="$4"
    gl_desc="$5"
    shift 5

    "$@" > "$gl_out"
    gl_rc=$?
    if [ "$gl_rc" -ne 0 ]; then
        echo "$gl_name: FAILED -- '$gl_desc' exited $gl_rc" >&2
        return 1
    fi

    if ! diff -u "$gl_golden" "$gl_out" > "$gl_diff" 2>&1; then
        echo "$gl_name: FAILED -- output differs from $gl_golden" >&2
        cat "$gl_diff" >&2
        return 1
    fi

    rm -f "$gl_out" "$gl_diff"
    return 0
}
