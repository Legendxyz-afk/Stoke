#!/bin/sh
# tests/abi_checklist.sh -- M6 GATE ITEM [4]: the ABI freeze checklist, run as a
# MECHANICAL pass.
#
# ▶▶ THE FORWARD NOTE ASKS FOR A MECHANICAL PASS, NOT A READING, and the
# difference is the whole value. A checklist a person walks is a checklist that
# passes on the afternoon somebody is confident; a checklist that is a script
# fails the build the day the header stops satisfying it, which is the day
# before the freeze rather than the year after. Every box below is either
# checked against the header's own text or explicitly recorded as unautomatable
# WITH the reason -- there is no third category and nothing is waved through.
#
# ⚠ THE HEADER IS READ WITH COMMENTS STRIPPED, for the reason tests/
# abi_isolation.sh learned by failing on its own documentation: a rule about
# what the code SAYS must not be satisfiable, or breakable, by prose about the
# rule. `gcc -fpreprocessed -E` removes comments and expands nothing, so the
# declarations survive verbatim and only the commentary goes.
#
# Usage: sh tests/abi_checklist.sh
set -u

# Overridable ONLY so tests/abi_checklist_mutants.sh can point this at a
# deliberately broken copy and prove each box can fail. Nothing else should set
# it; the default is the header this gate exists to check.
HDR="${STOKE_ABI_HDR:-src/stoke.h}"
IMPL="src/stoke.c"
CONTROL="src/sched.h"   # an INTERNAL header: not stoke_-prefixed, and it is the
                        # positive control that proves these patterns can fire
CC="${CC:-gcc}"

fail=0
box() { printf '  [%s] %s\n' "$1" "$2"; }
bad() {
    box "FAIL" "$1"
    shift
    for line in "$@"; do printf '         %s\n' "$line"; done
    fail=1
}

for f in "$HDR" "$IMPL" "$CONTROL"; do
    if [ ! -f "$f" ]; then
        echo "abi_checklist: FAIL -- $f does not exist" >&2
        exit 1
    fi
done

tmp="${TMPDIR:-/tmp}/abi_checklist.$$"
mkdir -p "$tmp" || exit 1
trap 'rm -rf "$tmp"' EXIT

# ⚠ CR IS STRIPPED HERE AND THE FIRST VERSION OF THIS SCRIPT DID NOT DO IT.
# This repo is core.autocrlf=true, so src/stoke.h has CRLF in the working
# tree, and the compiler preserves it. Splitting the joined text on ';'
# then leaves every declaration beginning with a carriage return, so the
# "starts with an identifier" pattern matched NOTHING -- the script reported
# eleven green boxes against zero extracted functions. The anti-vacuity
# control at the bottom is the only thing that caught it.
strip() { "$CC" -fpreprocessed -dD -E "$1" 2> /dev/null | tr -d '\r'; }
strip "$HDR"     > "$tmp/hdr"     || { echo "abi_checklist: FAIL -- cannot strip $HDR" >&2; exit 1; }
strip "$CONTROL" > "$tmp/control" || { echo "abi_checklist: FAIL -- cannot strip $CONTROL" >&2; exit 1; }

# Every function DECLARATION in the stripped header, one per line, normalized to
# a single space. A declaration is a line ending in ');' possibly continued over
# several lines, so the file is joined and re-split on ';'.
#
# ⚠ PREPROCESSOR LINES ARE REMOVED BEFORE THE JOIN, NOT FILTERED AFTER IT, and
# the difference silently ate five of sixteen functions in the first version. A
# `#define` has no semicolon, so joining first GLUES it to the declaration that
# follows -- `#define STOKE_ABI_VERSION 0 int32_t stoke_abi_version(void)` is
# one chunk, and a filter that drops chunks beginning with '#' drops the
# declaration with it. The count printed above is the guard: sixteen functions
# reading as eleven is visible, where each individually missing box would not
# have been.
grep -v '^ *#' "$tmp/hdr" | tr '\n' ' ' | tr ';' '\n' \
    | sed 's/  */ /g; s/^ //' > "$tmp/decls"
grep -E '^[A-Za-z_].*\(' "$tmp/decls" > "$tmp/funcs" || true

# The public function NAMES: the identifier immediately before the '('.
sed 's/(.*//; s/.*[^A-Za-z0-9_]\([A-Za-z0-9_]*\)$/\1/' "$tmp/funcs" \
    | grep -E '^[A-Za-z_]' | sort -u > "$tmp/names"

n_funcs=$(wc -l < "$tmp/names")
echo "abi_checklist: M6 gate item [4] -- $HDR, $n_funcs public functions"
echo

# ---- 1. every public symbol is stoke_-prefixed ------------------------------
stray=$(grep -v '^stoke_' "$tmp/names" || true)
if [ -n "$stray" ]; then
    bad "every public function is stoke_-prefixed" "unprefixed: $stray"
else
    box "ok" "every public function is stoke_-prefixed ($n_funcs checked)"
fi

stray_t=$(grep -E '^ *} *[A-Za-z_]' "$tmp/hdr" | sed 's/.*} *//; s/ *$//' \
          | grep -v '^stoke_' || true)
if [ -n "$stray_t" ]; then
    bad "every public typedef is stoke_-prefixed" "unprefixed: $stray_t"
else
    box "ok" "every public typedef is stoke_-prefixed"
fi

stray_m=$(grep -E '^ *# *define ' "$tmp/hdr" | awk '{print $2}' \
          | sed 's/(.*//' | grep -vE '^(STOKE_|stoke_)' || true)
if [ -n "$stray_m" ]; then
    bad "every public macro is STOKE_-prefixed" "unprefixed: $stray_m"
else
    box "ok" "every public macro is STOKE_/stoke_-prefixed"
fi

# ---- 2. no varargs ----------------------------------------------------------
# A varargs entry point cannot be called from ctypes, cgo or a Rust binding
# without per-platform assembly knowledge, which is the whole point of refusing
# it rather than merely avoiding it.
if grep -qF '...' "$tmp/funcs"; then
    bad "no varargs in the public surface" "$(grep -F '...' "$tmp/funcs")"
else
    box "ok" "no varargs in the public surface"
fi

# ---- 3. no structs by value across the boundary -----------------------------
# Checked by NAME: every public struct type must appear only as a pointer. A
# by-value struct fixes its layout into every caller's binary forever.
byval=""
# ⚠ THE ';' MUST COME OFF THE TYPEDEF NAME. `} stoke_chat_message;` left the
# semicolon attached, so every pattern built from it read
# "stoke_chat_message; " and could not match anything -- the box was green
# because it was searching for a string that cannot occur. Found by the
# mutation leg, not by this box ever failing.
for t in $(grep -E '^ *} *stoke_' "$tmp/hdr" | sed 's/.*} *//; s/[; ]*$//'); do
    # `<type> <name>` with no '*' between them, anywhere in a declaration.
    if grep -E "(^|[^*A-Za-z0-9_])$t +[A-Za-z_]" "$tmp/funcs" > /dev/null 2>&1; then
        byval="$byval $t"
    fi
done
if [ -n "$byval" ]; then
    bad "no public struct crosses the boundary by value" "by value:$byval"
else
    box "ok" "no public struct crosses the boundary by value"
fi

# ---- 4. no enums inside public structs --------------------------------------
# An enum's size is implementation-defined, so one inside a struct whose layout
# crosses an ABI boundary makes that layout compiler-dependent.
if awk '/^ *typedef struct \{/{inb=1} inb&&/ enum /{print; bad=1} /^ *} *stoke_/{inb=0} END{exit !bad}' \
        "$tmp/hdr" > "$tmp/enums" 2>/dev/null; then
    bad "no enum inside a public struct" "$(cat "$tmp/enums")"
else
    box "ok" "no enum inside a public struct"
fi

# ---- 5. error codes are explicitly numbered ---------------------------------
# "Errors are never renumbered after the tag" cannot be checked before the tag.
# What CAN be checked is that renumbering requires an edit that is visible:
# every enumerator carries its own value, so inserting one cannot silently
# shift the rest.
unnumbered=$(awk '/^ *enum \{/{inb=1;next} inb&&/^ *\}/{inb=0} inb' "$tmp/hdr" \
             | grep -E '^ *STOKE_[A-Z_]+ *,? *$' || true)
if [ -n "$unnumbered" ]; then
    bad "every public enumerator is explicitly numbered" "implicit: $unnumbered"
else
    box "ok" "every public enumerator is explicitly numbered"
fi

# ---- 6. the ABI version is declared AND checked at open ---------------------
if ! grep -qF 'STOKE_ABI_VERSION' "$tmp/hdr"; then
    bad "STOKE_ABI_VERSION is defined in the header"
elif ! grep -qF 'stoke_abi_version' "$tmp/names"; then
    bad "stoke_abi_version() is declared"
elif ! grep -qF 'STOKE_ABI_VERSION' "$IMPL"; then
    bad "the open path compares the caller's version against STOKE_ABI_VERSION" \
        "a version macro nothing verifies is a comment"
else
    box "ok" "ABI version declared, exposed as a function, and checked in $IMPL"
fi

# ---- 7. caller-owns-buffer with an explicit cap -----------------------------
# Every entry point that WRITES into a caller's buffer takes its capacity.
# `out_rows` is exempt by name and the exemption is stated here rather than
# hidden in a pattern: it is a single uint32_t, not a buffer.
nocap=""
while IFS= read -r d; do
    case "$d" in
        *out_rows*) continue ;;
    esac
    case "$d" in
        *"char* "*|*"char *"*|*"uint32_t* "*|*"uint32_t *"*)
            case "$d" in
                *const\ char*|*const\ uint32_t*)
                    # const in: not a buffer we write. Only flag if it ALSO has
                    # a writable one.
                    case "$d" in
                        *"* out"*|*"* buf"*) ;;
                        *) continue ;;
                    esac
                    ;;
            esac
            case "$d" in
                *cap*) ;;
                *) nocap="$nocap
    $d" ;;
            esac
            ;;
    esac
done < "$tmp/funcs"
if [ -n "$nocap" ]; then
    bad "every buffer-writing entry point takes an explicit cap" "$nocap"
else
    box "ok" "every buffer-writing entry point takes an explicit cap"
fi

# ---- 8. every public function's thread safety is stated ---------------------
# ⚠ Checked against the header's own THREAD SAFETY block -- ONE place, so an
# auditor reads one list rather than hunting per-function prose. This is the
# box most likely to rot: a function added later is documented where it is
# declared and forgotten here.
missing=""
for n in $(cat "$tmp/names"); do
    if ! grep -qF "$n" "$HDR"; then continue; fi
    if ! awk '/THREAD SAFETY/{inb=1} inb&&/^ \*\//{inb=0} inb' "$HDR" \
            | grep -qF "$n"; then
        missing="$missing $n"
    fi
done
if [ -n "$missing" ]; then
    bad "every public function appears in the THREAD SAFETY block" \
        "missing:$missing"
else
    box "ok" "every public function appears in the THREAD SAFETY block"
fi

# ---- 9. every public function is documented at its declaration --------------
undocumented=""
for n in $(cat "$tmp/names"); do
    ln=$(grep -nE "(^|[^A-Za-z0-9_])$n *\(" "$HDR" | head -1 | cut -d: -f1)
    [ -n "$ln" ] || continue
    prev=$(sed -n "$((ln - 1))p;$((ln - 2))p;$((ln - 3))p" "$HDR")
    case "$prev" in
        *"*"*) ;;                       # inside or just after a comment block
        *) undocumented="$undocumented $n" ;;
    esac
done
if [ -n "$undocumented" ]; then
    bad "every public function is documented in the header" \
        "undocumented:$undocumented"
else
    box "ok" "every public function is documented in the header"
fi

# ---- 10. the C++ guard, and the two TUs that prove it ----------------------
# ⚠ CHECKED ON THE STRIPPED HEADER AND FOR THE OPENING BRACE. Against the raw
# text, `grep 'extern "C"'` is satisfied by the CLOSING COMMENT
# `} /* extern "C" */` -- so deleting the guard itself leaves the box green.
# Found by the mutation leg, which is the third time in this project that a
# rule was satisfiable by prose about the rule.
if ! grep -qF 'extern "C" {' "$tmp/hdr"; then
    bad "the header carries an extern \"C\" guard"
elif [ ! -f tests/abi_tu.c ] || [ ! -f tests/abi_tu.cpp ]; then
    bad "the standalone C11 and C++ translation units exist" \
        "gate item [5] is what runs them; this box only checks they are here"
else
    box "ok" "extern \"C\" guard present, with abi_tu.c and abi_tu.cpp (gate [5])"
fi

# ---- 11. UTF-8 is stated ----------------------------------------------------
if grep -qF 'UTF-8' "$HDR"; then
    box "ok" "the header states its text encoding (UTF-8 in and out)"
else
    bad "the header states its text encoding"
fi

# ---- ANTI-VACUITY: prove the patterns can fire ------------------------------
# ⚠ Without this, one typo turns the whole script into a green light that
# examines nothing. src/sched.h is an INTERNAL header: its symbols are not
# stoke_-prefixed by design, so the prefix pattern MUST match there. If it
# stops matching, the pattern is broken, not the header.
echo
ctl_names=$(tr '\n' ' ' < "$tmp/control" | tr ';' '\n' | sed 's/  */ /g; s/^ //' \
            | grep -E '^[A-Za-z_].*\(' | sed 's/(.*//; s/.*[^A-Za-z0-9_]\([A-Za-z0-9_]*\)$/\1/' \
            | grep -E '^[A-Za-z_]' | grep -vc '^stoke_' || true)
if [ "${ctl_names:-0}" -lt 3 ]; then
    bad "ANTI-VACUITY: the prefix pattern found only ${ctl_names:-0} non-stoke_ functions in $CONTROL" \
        "that header is full of them -- the CHECK is broken, not the code"
else
    box "ok" "anti-vacuity: the prefix pattern finds $ctl_names unprefixed functions in $CONTROL"
fi

echo
if [ "$fail" -ne 0 ]; then
    echo "abi_checklist: FAILED -- the boxes above are gate item [4] and it is not all-boxes yet"
    exit 1
fi
echo "abi_checklist: OK (gate item [4] -- every box mechanically checked)"
exit 0
