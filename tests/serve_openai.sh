#!/bin/sh
# tests/serve_openai.sh -- M6 GATE ITEM [1]: the openai-python suite.
#
# Starts `stoke serve` and points the ecosystem's own client at it. The golden
# gate already pins every byte of every response; this asks the different
# question a typed client answers -- would openai-python, which parses each
# response into a pydantic model and raises on anything that does not fit,
# ACCEPT what we send? A response can match a recorded transcript exactly and
# still be a shape no real client can use.
#
# ▶ THE CLIENT LIVES IN A REPO-LOCAL VENV, not in the box's site-packages, so
# running this gate changes nothing else on the machine and the gate names its
# own client version. It is gitignored and regenerable:
#     python -m venv bench/.venv-openai
#     bench/.venv-openai/Scripts/pip install openai
# Absent, this SKIPS loudly rather than failing -- the same policy the model
# check below uses, and for the same reason: a missing local tool must not read
# as a broken server.
#
# Usage: sh tests/serve_openai.sh <builddir> [exe-suffix]
set -u

. "$(dirname "$0")/golden_lib.sh"

builddir="${1:?usage: serve_openai.sh <builddir> [exe-suffix]}"
exe="${2:-}"

model="models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
bin="$builddir/stoke$exe"
suite="tests/serve_openai.py"
port="${STOKE_SERVE_OPENAI_PORT:-$(golden_lib_port 18143 "$builddir")}"

golden_lib_require_model "$model" "serve_openai"
golden_lib_require_binary "$bin" "serve_openai"
golden_lib_require_data_file "$suite" "serve_openai" "suite"

# The venv's interpreter, by platform. Nothing here uses the ambient `python`:
# the whole point of the venv is that this gate's client version is a property
# of the repo and not of whatever happens to be on PATH.
py="bench/.venv-openai/Scripts/python.exe"
[ -x "$py" ] || py="bench/.venv-openai/bin/python"
if [ ! -x "$py" ]; then
    # ⚠ ESCALATES ON STOKE_REQUIRE_OPENAI, NOT STOKE_REQUIRE_GOLDEN, and the
    # distinction is not pedantry. REQUIRE_GOLDEN means "this box has the real
    # model, so a missing one is a moved file rather than a fresh clone" -- and
    # scripts/wsl-check.sh sets it for the Linux legs of the matrix. A venv is
    # NOT like a model: it is per-interpreter and per-platform, so the Windows
    # one is legitimately absent under WSL and escalating there would fail the
    # matrix over a file that is not supposed to exist. The Windows gate run
    # sets REQUIRE_OPENAI instead, which keeps "the venv quietly vanished" a
    # failure exactly where the venv is supposed to be.
    if [ "${STOKE_REQUIRE_OPENAI:-0}" = "1" ]; then
        echo "serve_openai: FAILED -- no openai venv found but" \
             "STOKE_REQUIRE_OPENAI=1. Recreate it with:  python -m venv" \
             "bench/.venv-openai && bench/.venv-openai/Scripts/pip install openai" >&2
        exit 1
    fi
    echo "serve_openai: SKIP -- bench/.venv-openai not present (gitignored;" \
         "create it with 'python -m venv bench/.venv-openai' then" \
         "'bench/.venv-openai/Scripts/pip install openai')"
    exit 0
fi

srvlog="$builddir/serve_openai.srv"
rm -f "$srvlog"

guard=$(golden_lib_guard "$builddir")

# One readiness probe plus the suite's six requests. The count is asserted
# below: the SDK is constructed with max_retries=0 precisely so this number is
# exact, and a retry storm cannot pass as a clean run.
n_requests=7

$guard "$bin" serve "$model" --port "$port" --ctx 1024 --seqs 4 --threads 8 \
    --max-requests "$n_requests" -v > "$srvlog" 2>&1 &
srv_pid=$!

if ! curl -sS --retry 400 --retry-delay 1 --retry-connrefused \
        "http://127.0.0.1:$port/health" > /dev/null; then
    echo "serve_openai: FAILED -- the server never answered /health" >&2
    kill "$srv_pid" 2> /dev/null
    exit 1
fi

"$py" "$suite" "http://127.0.0.1:$port/v1"
suite_rc=$?

wait "$srv_pid"
srv_rc=$?

fail=0
[ "$suite_rc" -eq 0 ] || fail=1
if [ "$srv_rc" -ne 0 ]; then
    echo "serve_openai: FAILED -- the server exited $srv_rc" >&2
    fail=1
fi
if ! grep -q "^stoke serve: $n_requests requests completed, exiting$" "$srvlog"; then
    echo "serve_openai: FAILED -- the server did not report $n_requests completed" \
         "requests; the SDK is configured with max_retries=0, so any other" \
         "count means a request was retried or never arrived" >&2
    sed 's/^/  /' "$srvlog" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi

rm -f "$srvlog"
echo "serve_openai: OK (gate item [1] -- openai-python accepts every shape)"
exit 0
