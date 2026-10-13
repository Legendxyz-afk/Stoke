#!/bin/sh
# tests/tooling/guest_clock_check.sh -- M7 gate [3] guard, design note
# docs/superpowers/design-notes/2026-10-12-m7-soak-4x6h-rule.md section 9.
#
# The WSL guest's CLOCK_MONOTONIC times serve_fuzz's chunks and the server's
# --max-seconds. On 2026-10-13 it ran 8.6% FAST against the host (clocksource
# `tsc`, the kernel having detected the TSC at 2611.197 MHz), so a "60-min"
# chunk lasted ~55 real minutes and run 1 carried ~5.5 h of traffic, not 6 h.
# The guest's WALL clock is synced from the host and stayed right, which is
# why the timestamps looked fine. This measures the guest's monotonic rate
# against the HOST clock, read through WSL interop, and refuses a run when the
# two differ by more than 1%.
#
#   sh tests/tooling/guest_clock_check.sh [window_seconds]     (default 240)
#   sh tests/tooling/guest_clock_check.sh --judge WINDOW < samples
#       (the verdict alone, on "g0 h0 e0 g1 h1 e1" in ms -- for its tests)
#
# The verdict is the printed "guest clock:" line. Exit 0 = within 1%;
# exit 2 = refused: more than 1% off, OR the measurement itself was too
# uncertain to say -- a check that cannot evaluate must not read as a pass.

judge() {
    awk -v win="$1" '
    {
        if (NF != 6) { print "guest clock: REFUSED -- a reading failed (" NF " of 6 numbers: " $0 ")"; bad = 1; exit }
        for (i = 1; i <= 6; i++) if ($i !~ /^[0-9]+$/) { print "guest clock: REFUSED -- reading " i " is not a number: " $i; bad = 1; exit }
        g0 = $1; h0 = $2; e0 = $3; g1 = $4; h1 = $5; e1 = $6
        dh = h1 - h0; dg = g1 - g0
        if (dh < win * 900) { print "guest clock: REFUSED -- the host interval (" dh " ms) is shorter than the window"; bad = 1; exit }
        ratio = dg / dh
        unc   = ((e0 + e1) / 2) / dh
        dev   = ratio - 1; if (dev < 0) dev = -dev
        line  = sprintf("guest monotonic %.1f s over host %.1f s = ratio %.4f (+/- %.4f)", dg / 1000, dh / 1000, ratio, unc)
        if (unc > 0.005) { print "guest clock: REFUSED -- " line ": too uncertain to judge"; bad = 1; exit }
        if (dev > 0.01)  { print "guest clock: REFUSED -- " line ": more than 1% off; pin the clocksource and restart the VM (design note section 9)"; bad = 1; exit }
        print "guest clock: OK -- " line
    }
    END { if (NR == 0) { print "guest clock: REFUSED -- no samples"; exit 2 } exit bad ? 2 : 0 }'
}

if [ "${1:-}" = "--judge" ]; then
    judge "${2:-240}"
    exit $?
fi

WIN=${1:-240}
case "$WIN" in
    ''|*[!0-9]*) echo "guest clock: REFUSED -- window '$WIN' is not a whole number of seconds"; exit 2 ;;
esac
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
if [ ! -x "$PS" ]; then
    echo "guest clock: REFUSED -- cannot run $PS (WSL interop off?), so the host clock cannot be read"
    exit 2
fi

up_ms()   { awk '{ printf "%d", $1 * 1000 }' /proc/uptime; }
host_ms() { "$PS" -NoProfile -NonInteractive -Command '[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()' 2>/dev/null | tr -d '\r\n '; }

# One sample: the host's time bracketed by two guest readings. The guest time
# it pairs with is their midpoint, uncertain by their gap.
sample() {
    a=$(up_ms); h=$(host_ms); b=$(up_ms)
    echo "$(( (a + b) / 2 )) ${h:-X} $(( b - a ))"
}

s0=$(sample)
sleep "$WIN"
s1=$(sample)
echo "$s0 $s1" | judge "$WIN"
