#!/bin/sh
# tests/tooling/guest_clock_settle.sh -- M7 gate [3], design note
# docs/superpowers/design-notes/2026-10-12-m7-soak-4x6h-rule.md section 11.
#
# The WSL guest's clock runs 8.3% FAST whenever this distro's
# systemd-timesyncd runs. timesyncd steps the guest to NTP (ntp.ubuntu.com);
# this host's clock sits ~2.5 s from NTP; and WSL's own time sync -- outside
# the distro, over /dev/ptp_hyperv -- slews the guest back to the host at its
# maximum rate for as long as the two disagree, which is forever. The kernel
# shows it: adjtimex reads tick 10833 instead of 10000 and freq +33.333 ppm,
# 1 + 1/12. With timesyncd stopped, WSL's sync converges in about a minute and
# then holds tick 10000 and a few ppm (2026-10-15).
#
# This stops timesyncd for the run -- not disabled: the distro starts it again
# at its next boot -- then waits until the kernel holds tick 10000 and
# |freq| <= 100 ppm in three readings 15 s apart. Root.
#
#   sh tests/tooling/guest_clock_settle.sh [max_seconds]      (default 300)
#
# Each reading is printed ("  +N s: tick T, freq F ppm"); the verdict is the
# one "guest clock:" line, last: SETTLED, exit 0; REFUSED, exit 2 -- including
# a reading that cannot be taken, since a check that cannot evaluate must not
# read as a pass.

MAX=${1:-300}
case "$MAX" in
    ''|*[!0-9]*) echo "guest clock: REFUSED -- '$MAX' is not a whole number of seconds"; exit 2 ;;
esac
if [ "$(id -u)" -ne 0 ]; then
    echo "guest clock: REFUSED -- not root: systemd-timesyncd cannot be stopped"
    exit 2
fi

PY=$(mktemp)
cat > "$PY" <<'PY'
import ctypes, ctypes.util
class timex(ctypes.Structure):
    _fields_ = [("modes", ctypes.c_uint), ("offset", ctypes.c_long), ("freq", ctypes.c_long),
                ("maxerror", ctypes.c_long), ("esterror", ctypes.c_long), ("status", ctypes.c_int),
                ("constant", ctypes.c_long), ("precision", ctypes.c_long), ("tolerance", ctypes.c_long),
                ("time_sec", ctypes.c_long), ("time_usec", ctypes.c_long), ("tick", ctypes.c_long),
                ("ppsfreq", ctypes.c_long), ("jitter", ctypes.c_long), ("shift", ctypes.c_int),
                ("stabil", ctypes.c_long), ("jitcnt", ctypes.c_long), ("calcnt", ctypes.c_long),
                ("errcnt", ctypes.c_long), ("stbcnt", ctypes.c_long), ("tai", ctypes.c_int),
                ("pad", ctypes.c_int * 11)]
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
t = timex()
if libc.adjtimex(ctypes.byref(t)) >= 0:
    print(t.tick, int(round(t.freq / 65536)))
PY

if systemctl is-active --quiet systemd-timesyncd 2>/dev/null; then
    if ! systemctl stop systemd-timesyncd; then
        echo "guest clock: REFUSED -- systemd-timesyncd is running and could not be stopped"
        rm -f "$PY"; exit 2
    fi
    how="stopped systemd-timesyncd"
else
    how="systemd-timesyncd was not running"
fi

good=0; waited=0
while :; do
    set -- $(python3 "$PY" 2>/dev/null)
    tick=${1:-?}; freq=${2:-?}
    echo "  +$waited s: tick $tick, freq $freq ppm"
    if [ "$tick" = 10000 ] && [ "$freq" -ge -100 ] 2>/dev/null && [ "$freq" -le 100 ]; then
        good=$((good + 1))
    else
        good=0
    fi
    [ "$good" -ge 3 ] && break
    if [ "$waited" -ge "$MAX" ]; then
        echo "guest clock: REFUSED -- the kernel did not hold tick 10000 within $MAX s (last: tick $tick, freq $freq ppm; $how)"
        rm -f "$PY"; exit 2
    fi
    sleep 15
    waited=$((waited + 15))
done
rm -f "$PY"
echo "guest clock: SETTLED -- tick $tick, freq $freq ppm in 3 readings 15 s apart, after $waited s ($how)"
