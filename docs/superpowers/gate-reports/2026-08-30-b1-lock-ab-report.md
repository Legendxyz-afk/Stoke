# M7 rung B1 — locking the scheduler's cross-thread surface, and what it cost

**Date:** 2026-08-30 · **Metric:** `AGGREGATE(M0)` · **Bank:**
`2026-08-30-b1-lock-ab-bank.jsonl` · sliced one unit per invocation.

⚠ **Arm labels in the bank are positional and mean this here:** `M5` = the
REFERENCE arm, which for B1 is the **tag-`m6` binary** (`bec348e0c11583c4`);
`A6` = today's locked `HEAD` (`303f4e3adb32586a`); `NULL` = that same locked
binary launched a second time. The `bin` sha in every banked row is what
actually identifies an arm, and it is unambiguous.

## Why there is a measurement at all

B1 locked the scheduler's cross-thread surface. That is a change to `sched.c`,
and M6's standing constraint is that a scheduler touch is re-measured against
the tag it is trying not to move, or it does not land. The instrument is the
same as A6's rate half: a same-sitting A/B, arms interleaved inside each block,
order counterbalanced, **with a null arm** whose spread is the only thing that
makes the result mean anything.

## Pre-flight

Both binaries driven with `--ids` on this cell: **byte-identical**, 1534 bytes,
`cmp` clean. Locking changes when things happen, not what is computed, and this
is where that claim is checked rather than assumed.

## The box, for once, was quiet

Per-unit quiesce readings (the gate's own `_Total` instrument, two samples per
unit) ran `0,4 · 15,9 · 18,13 · 20,10 · 31,13 · 0,11 · 0,15 · 23,15 · 0,15`.
Several units started from a genuinely idle box — which is why the spreads below
are roughly five times tighter than the same harness produced this morning at
median 14.5%. **That is a fact about the sitting, not about the code**, and it is
the reason nothing here is compared to this morning's numbers.

## 8f32 — the primary cell

| arm | n | values (tok/s) | median |
|---|---|---|---|
| reference (`m6`) | 3 | 6.418 6.446 6.528 | **6.446** |
| locked (`HEAD`) | 3 | 6.365 6.440 6.485 | **6.440** |
| null (locked, again) | 3 | 6.420 6.424 6.463 | **6.424** |

Block-paired, order counterbalanced:

| block | locked/reference | locked/null |
|---|---|---|
| b1 | 0.9874 | 0.9848 |
| b2 | 1.0034 | 1.0025 |
| b3 | 0.9934 | 1.0101 |
| **span** | **0.987 – 1.003 (1.6 pp)** | **0.985 – 1.010 (2.5 pp)** |

**Median reference → locked: 6.446 → 6.440, a difference of 0.09%, against a
null-arm floor of 0.25% on medians. The locked-vs-reference spread (1.6 pp) is
NARROWER than the locked-vs-ITSELF spread (2.5 pp)** — the same shape A6's rate
half produced, on a quieter box and an order of magnitude tighter.

## 16f32 — the cell where contention would show, one block

⚠ **This cell was added for a reason, not for symmetry.** The new lock is taken
once per poll, and the bars driver polls once per live request per step — so
whatever contention the change introduces scales with `--seqs`, and 8 clients is
the cell least able to show it. 16 is the largest cell the bars gate defines.

| arm | value (tok/s) |
|---|---|
| reference (`m6`) | 6.188 |
| locked (`HEAD`) | 6.190 |
| null (locked, again) | 6.011 |

| | ratio |
|---|---|
| locked / reference | **1.0003** |
| locked / null | 1.0298 |

**0.03% between the binaries, against a 3.0% floor from the null arm at the same
cell.** One block, so the floor is one sample rather than a distribution — that
is stated rather than dressed up, and it is still 100× the effect being looked
for.

## Verdict

**The locking costs nothing this harness can resolve.** That is the expected
result and the reason is arithmetic: the mutex is taken twice per step and once
per poll, against a step that runs a full forward pass. What matters is that
the lock is **released across `host->step`** — holding it there would have
serialised every caller's poll against a ~100 ms forward pass, which is not
thread safety but a stall, and it would have shown up here immediately.

At both cells the difference between the two BINARIES is far smaller than the
difference between one binary and a second copy of itself:

| cell | locked vs reference | null floor |
|---|---|---|
| 8f32 | 0.09% (medians of 3) | 0.25% (medians), 2.5 pp per block |
| 16f32 | 0.03% (one block) | 3.0% (one block) |

⚠ **What this does not say.** It does not say the lock is free — no box-level
instrument resolves two uncontended acquires against a forward pass. It says the
cost is below the floor at the cells measured, and it says nothing about
absolute rate: the bars gate's absolutes are still M5's, which is M7 rung B2.
