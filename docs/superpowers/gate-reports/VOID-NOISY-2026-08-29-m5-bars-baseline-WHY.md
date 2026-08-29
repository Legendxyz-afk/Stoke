# VOIDED: the box did not meet the quiesce bar

This stamp holds two banked units and neither may be used, for either an
absolute or a comparison. It is renamed rather than deleted so the numbers stay
readable and so the clean stamp name cannot be reused by accident.

    b1-8f32-S   7.223 tok/s   quiesce gate PASSED (12%, 12%)
    b1-8f32-B  26.671 tok/s   quiesce gate FAILED (8%, 19%) -- banked
                              quiesced:false, "absolutes do not survive"

## Why it is void

ISLC was stopped first, so that is not the cause. Measured afterwards with the
gate's OWN instrument (`Win32_PerfFormattedData_PerfOS_Processor`, `_Total`),
twelve readings two seconds apart:

    11, 10, 16, 22, 15, 13, 10, 31, 18, 30, 25, 32   -> median 18%, max 32%

The gate wants two consecutive reads at or below 12%. A median of 18% is not a
quiet box, and the first unit's pass at exactly 12% was the tail of that
distribution rather than a quiet box briefly observed.

Per-process CPU deltas over a 12-second window account for only ~2.4% across 16
logical CPUs (conhost, dwm, claude, System, svchost), and WSL was Stopped with
no vmmem resident -- so the load is not one nameable process. The most likely
residue is this session's own: a great many builds, three sanitizer suites and
several 2.5 GB model loads, leaving Defender and the page cache busy.

## What it would take

Two consecutive `_Total` reads at or below 12%. Re-sample before starting; if
the median is still near 18%, something is running that a per-process delta does
not attribute, and the honest options are to wait for the box to settle or to
find it.

## What must NOT be done with these numbers

`b1-8f32-S` at 7.223 is 0.81x M5's banked 8.960 at the same cell. That gap is
NOT a finding. It is one block, on a box that does not meet the quiesce bar,
against a banked value taken over three blocks on one that did -- and block 1 is
where a warming transient lives. Comparing them would be exactly the
cross-sitting attribution the standing rules forbid.
