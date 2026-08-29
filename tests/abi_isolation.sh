#!/bin/sh
# tests/abi_isolation.sh -- M6: the server reaches the engine ONLY through the
# public ABI.
#
# ▶▶ WHY THIS IS A TEST AND NOT A COMMENT. The M6 entry note's ordering decision
# rests entirely on it: `server.c` could include sched.h and model.h -- it lives
# in the same tree and nothing stops it -- and if it ever does, the ABI ships
# having been exercised by nothing and the A7 freeze becomes a guess dressed as
# a checklist. A rule that is only written down decays on the first afternoon
# somebody needs one field. This one fails the build.
#
# It has already paid for itself once: `stoke_apply_chat_template` exists
# because server.c needed it and stoke.h could not express a chat request. That
# is the rule working -- a missing capability surfaced as a header change
# instead of a quiet #include.
#
# ▶ WHAT COUNTS AS "THE ENGINE" is listed explicitly rather than left to a
# guess. http.h, json.h and net.h are NOT the engine: they are the server's own
# support code, built at rungs A2 and A3 for exactly this caller. common.h is
# not either -- it is fixed-width types and timing, and every .c in this project
# includes it first because on the Linux leg it carries the feature-test macro.
# What common.h ALSO carries is fatal(), so this script checks separately that
# the server never calls it: a network endpoint that can exit the process is the
# one thing the whole M6 error split exists to prevent.
#
# ⚠⚠ EVERY PATTERN BELOW RUNS AGAINST THE SOURCE WITH COMMENTS STRIPPED, and
# that is not tidiness. Grepping the raw file makes a rule that reads
# "src/server.c must never call fatal()" fail on the SENTENCE "this file never
# calls fatal()" -- which is exactly what happened the first time this ran
# (2026-08-29). A checker whose verdict changes when you DOCUMENT the property
# it checks is worse than no checker: it trains you to edit prose to keep a
# build green. `gcc -fpreprocessed -E` removes comments and expands nothing, so
# the #include lines survive verbatim and only the commentary goes.
#
# Usage: sh tests/abi_isolation.sh
set -u

TARGET="src/server.c"
CONTROL="src/main.c"
ENGINE_HEADERS="model.h sched.h kv.h gguf.h pool.h blockpool.h serve.h tok.h bench.h"
CC="${CC:-gcc}"

fail=0

if [ ! -f "$TARGET" ]; then
    echo "abi_isolation: FAIL $TARGET does not exist" >&2
    exit 1
fi

strip_comments() {
    if ! "$CC" -fpreprocessed -dD -E "$1" 2> /dev/null; then
        echo "abi_isolation: FAIL could not strip comments from $1 with" \
             "'$CC -fpreprocessed -dD -E' -- the checks below would then run" \
             "against commentary and could pass or fail for the wrong reason" >&2
        exit 1
    fi
}

target_src="${TMPDIR:-/tmp}/abi_isolation_target.$$"
control_src="${TMPDIR:-/tmp}/abi_isolation_control.$$"
strip_comments "$TARGET" > "$target_src"
strip_comments "$CONTROL" > "$control_src"
trap 'rm -f "$target_src" "$control_src"' EXIT

# --- 1. no engine header --------------------------------------------------
for h in $ENGINE_HEADERS; do
    if grep -Eq "^[[:space:]]*#[[:space:]]*include[[:space:]]+\"$h\"" "$target_src"; then
        echo "abi_isolation: FAIL $TARGET includes \"$h\" -- reach the engine" \
             "through stoke.h, or add what is missing TO stoke.h" >&2
        fail=1
    fi
done

# --- 2. stoke.h is actually included --------------------------------------
# Without this the check above passes trivially for a file that talks to
# nothing, which is the shape a refactor could leave behind.
if ! grep -Eq '^[[:space:]]*#[[:space:]]*include[[:space:]]+"stoke.h"' "$target_src"; then
    echo "abi_isolation: FAIL $TARGET does not include \"stoke.h\"" >&2
    fail=1
fi

# --- 3. the server never calls fatal() ------------------------------------
# common.h brings it into scope; a client-triggerable fatal() is a remote
# process kill. Comments are already gone by here, so a sentence ABOUT this
# rule cannot trip it -- which is exactly what it did before the strip.
if grep -Eq '(^|[^_[:alnum:]])fatal[[:space:]]*\(' "$target_src"; then
    echo "abi_isolation: FAIL $TARGET calls fatal() -- a network endpoint must" \
         "answer with a status code, never exit the host process" >&2
    grep -nE '(^|[^_[:alnum:]])fatal[[:space:]]*\(' "$target_src" >&2
    fail=1
fi

# --- 4. ANTI-VACUITY: prove the checks can fire ---------------------------
# A typo in any pattern above turns this script into a green light that
# examines nothing. $CONTROL includes the engine headers and calls fatal() by
# design, so it is a standing positive control: if the patterns stop matching
# THERE, they are broken, not satisfied.
control_hits=0
for h in $ENGINE_HEADERS; do
    if grep -Eq "^[[:space:]]*#[[:space:]]*include[[:space:]]+\"$h\"" "$control_src"; then
        control_hits=$((control_hits + 1))
    fi
done
if [ "$control_hits" -lt 3 ]; then
    echo "abi_isolation: FAIL the include pattern matched only $control_hits" \
         "engine headers in $CONTROL, which includes many -- the CHECK is" \
         "broken, not the code" >&2
    fail=1
fi
if ! grep -Eq '(^|[^_[:alnum:]])fatal[[:space:]]*\(' "$control_src"; then
    echo "abi_isolation: FAIL the fatal() pattern does not match $CONTROL," \
         "which calls it -- the CHECK is broken, not the code" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "abi_isolation: FAILED"
    exit 1
fi
echo "abi_isolation: OK ($TARGET reaches the engine only through stoke.h," \
     "and never calls fatal(); patterns verified against $CONTROL)"
exit 0
