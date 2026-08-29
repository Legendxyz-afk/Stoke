#!/bin/sh
# tests/abi_checklist_mutants.sh -- proves tests/abi_checklist.sh can FAIL.
#
# ▶▶▶ WHY THIS EXISTS. abi_checklist.sh went all-boxes green, and nine of those
# boxes had never been observed failing. A checklist that cannot fail is not a
# checklist; it is a green light, and this one gates an ABI FREEZE -- the single
# decision in this project that cannot be walked back. So each box gets a
# deliberately broken header and has to notice.
#
# ⚠⚠ EACH MUTANT ASSERTS ITS OWN BOX, NOT MERELY A NON-ZERO EXIT. That
# distinction is the whole lesson of the A2 corpus's BARE_CR mutant: a mutation
# caught by the WRONG check goes red, the table reads green, and the property
# it claimed to test was never exercised. Every case below names the box it
# expects to see fail, and a mutant that fails a DIFFERENT box is reported as a
# failure of this leg.
#
# ⚠ The checklist's first run really did read eleven boxes "ok" against ZERO
# extracted functions -- only its anti-vacuity control caught it. That is what
# a checker looks like when it has stopped checking, and it is why this file is
# not optional.
#
# Usage: sh tests/abi_checklist_mutants.sh
set -u

REAL="src/stoke.h"
CHECK="tests/abi_checklist.sh"

[ -f "$REAL" ] || { echo "abi_mutants: FAIL -- $REAL missing" >&2; exit 1; }
[ -f "$CHECK" ] || { echo "abi_mutants: FAIL -- $CHECK missing" >&2; exit 1; }

tmp="${TMPDIR:-/tmp}/abi_mutants.$$"
mkdir -p "$tmp" || exit 1
trap 'rm -rf "$tmp"' EXIT

fail=0
n_ok=0

# run_mutant <name> <expected-box-substring> <sed-program>
#
# Applies <sed-program> to a copy of the header, runs the checklist against it,
# and requires BOTH that the checklist failed AND that the named box is the one
# that failed.
run_mutant() {
    m_name="$1"
    m_box="$2"
    m_sed="$3"

    m_hdr="$tmp/$m_name.h"
    sed "$m_sed" "$REAL" > "$m_hdr"

    if cmp -s "$m_hdr" "$REAL"; then
        echo "  [BROKEN] $m_name -- the mutation changed NOTHING, so it graded" \
             "nothing; the sed program no longer matches the header" >&2
        fail=1
        return
    fi

    STOKE_ABI_HDR="$m_hdr" sh "$CHECK" > "$tmp/$m_name.out" 2>&1
    m_rc=$?

    if [ "$m_rc" -eq 0 ]; then
        echo "  [MISSED]  $m_name -- the checklist PASSED a header that violates:" \
             "$m_box" >&2
        fail=1
        return
    fi
    if ! grep -F '[FAIL]' "$tmp/$m_name.out" | grep -qF "$m_box"; then
        echo "  [WRONG]   $m_name -- the checklist failed, but NOT on the box" \
             "this mutant breaks ($m_box). It went red for another reason, which" \
             "means the box under test was never exercised:" >&2
        grep -F '[FAIL]' "$tmp/$m_name.out" | sed 's/^/            /' >&2
        fail=1
        return
    fi
    echo "  [caught]  $m_name -> $m_box"
    n_ok=$((n_ok + 1))
}

echo "abi_mutants: proving each box of $CHECK can fail"
echo

# 1. An unprefixed public function.
run_mutant "unprefixed" "stoke_-prefixed" \
    's/^uint32_t stoke_n_vocab/uint32_t engine_n_vocab/'

# 2. A varargs entry point.
run_mutant "varargs" "no varargs" \
    's/^void stoke_close(stoke_engine\* e);/void stoke_close(stoke_engine* e, ...);/'

# 3. A public struct crossing the boundary by value.
#    ⚠ MUTATES AN EXISTING DECLARATION rather than adding one. The first version
#    added a new function, which of course was not in the THREAD SAFETY block --
#    so THAT box fired, the leg went red, and the by-value box was never
#    exercised at all. A mutant caught by the wrong check grades nothing, and
#    this leg is built to say so out loud rather than count it.
run_mutant "byvalue" "by value" \
    's/const stoke_chat_message\* msgs/stoke_chat_message msgs/'

# 4. An enum inside a public struct.
run_mutant "enum-in-struct" "no enum inside" \
    's/^    int32_t  kv_q8;/    enum { X } bad; int32_t  kv_q8;/'

# 5. An enumerator with no explicit value.
run_mutant "unnumbered" "explicitly numbered" \
    's/^    STOKE_ERR_TOO_LONG      = -6,/    STOKE_ERR_SNEAKY,\n    STOKE_ERR_TOO_LONG      = -6,/'

# 6. A buffer-writing entry point with no cap -- the cap taken OFF an existing
#    one, for the same reason as case 3.
#    ⚠ Targets a declaration that fits on ONE line. The first version aimed at
#    stoke_detokenize, whose parameters wrap across two -- sed is line-oriented,
#    so it matched nothing and the mutation graded nothing. The run_mutant
#    helper reports that as [BROKEN] rather than counting it, which is the only
#    reason it was noticed.
run_mutant "nocap" "explicit cap" \
    's/uint32_t\* out, int32_t cap);/uint32_t* out);/'

# 7. A public function missing from the THREAD SAFETY block.
run_mutant "no-thread-note" "THREAD SAFETY" \
    's/^ \*   stoke_abi_version  .*$//'

# 8. The C++ guard removed.
run_mutant "no-extern-c" "extern" \
    's/^extern "C" {$//'

# 9. The text encoding no longer stated.
run_mutant "no-utf8" "text encoding" \
    's/UTF-8/ascii-ish/g'

echo
if [ "$fail" -ne 0 ]; then
    echo "abi_mutants: FAILED -- see above; $n_ok caught"
    exit 1
fi
echo "abi_mutants: OK ($n_ok mutants, each caught by the box it breaks)"
exit 0
