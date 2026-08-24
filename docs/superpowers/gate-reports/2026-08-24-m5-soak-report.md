# M5 gate item [5] — the 1 h randomized soak under ASan and TSan

**2026-08-24, Opus 5, branch `m5-entry`, commit under test `21bf06e`.**
Runner `tests/tooling/run_m5_soak.sh`; banks
`2026-08-24-m5-soak-{asan,tsan}.jsonl`, logs `…{asan,tsan}.txt`.

The gate asks for "1 h randomized soak ASan+TSan clean" — 30 min per leg. The
soak is `tests/test_sched.c`'s seed walk over the **real scheduler and the real
block pool**, with every invariant asserted after **every step**, never sampled:
M4's I1–I12 via `bp_check`, plus M5's S1–S8.

## Result

| leg | wall | chunks | steps | preemptions | seeds covered | failures |
|---|---|---|---|---|---|---|
| **ASan + UBSan** | 1799 s | 2,090 | **418,237,658** | 2,155,823 | 1 … 4,968,585 | **0** |
| **TSan** | 1798 s | 313 | **62,636,849** | 322,901 | 1 … 744,113 | **0** |

**BOTH LEGS COMPLETE AND GREEN.** 1 h of wall clock, **480,874,507 steps over
5.71 M distinct traces, 2,478,724 preemptions, zero findings** — `418×` and `62×`
the gate's 1e6 step floor respectively.

⇒ **GATE ITEM [5] IS MET.**

*(An earlier revision of this report recorded the TSan leg as a partial at
1347 s, written when the box was about to be shut down. It was allowed to finish;
the numbers above are the complete run. The partial is left described here rather
than erased, because the runner's resumability is the reason a truncated leg
would have cost 8 minutes instead of 30 — and that is the property worth
recording.)*

## Why the numbers are this large, and why that is the point

`test_sched` at its default runs 1e6 steps in ~1.2 s. The gate's floor was
written as a step count; an hour of wall time buys three orders more than that,
so the binding question stopped being "enough steps" and became "enough distinct
traces". The seed walk answers it: seed *n* generates a different trace, so
**4.97 M distinct traces on the ASan leg** rather than one trace run many times.

Preemptions are the number to watch, not steps — gate [1]'s note records that the
watermark is effective enough that a `pressure` trace had to be made deliberately
adversarial or it would bank a passing S6 test proving nothing. **2,478,724
preemptions across the two legs** says the preemption path was genuinely
exercised, not merely reachable.

## The runner, and the one thing it is built around

An hour-long job that prints only when it finishes banks nothing if it is killed
— the shape that cost this project 76 minutes of probe compute on 2026-08-12.
So:

- **Chunked and resumable.** Each chunk appends its next seed; a kill costs one
  chunk, not the leg. That is exactly what let this report exist at all.
- **A failure replays in seconds.** `test_sched <steps> <seed>` reproduces any
  chunk, which is the whole reason the walk is deterministic rather than
  time-seeded. A soak whose failures cannot be replayed reports only "something
  broke".
- **A chunk with no parsable progress is a FAILURE, not a zero.** A sanitizer
  abort leaves `rc != 0` and no `next=`; treating that as "0 steps, carry on" is
  how a runner reports an hour of green over a binary that died in minute one.
- **One writer per file.** M0's ops lesson; the bank is read back by the resume
  logic.

## What changed in `tests/test_sched.c`, and why it is inert

`soak()` gained a `seed0` and `main` an optional second argv (it refuses 0, so a
mis-parsed or empty bank cannot masquerade as "start from the top"). **The
default path reproduces gate [1]'s banked result exactly — 1,000,130 steps over
11,870 traces, 5,142 preemptions, 18 checks, 0 failures** — so the per-commit
suite and gate [1]'s evidence are the same run they were.

## Scope

ASan/UBSan answers the memory and arithmetic questions; TSan the concurrency one.
**Neither is evidence for the other**, which is why they are separate legs and
why a complete ASan leg does not let the partial TSan leg be rounded up.
