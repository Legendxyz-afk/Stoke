# M5 gate [6] on the shipping default -- preempt-lossless with prefix dedup ON

> !! **-Quick PLUMBING PRE-FLIGHT, NOT A GATE.** At 2 seqs x n 16 the private
> growth is ~2 blocks and NO pool both admits every client and starves it: the
> preemption window is EMPTY BY ARITHMETIC. Claims 2, 4, 5, 6 and 7 are
> **ungradeable at this shape**, and this report renders them `n/a` rather
> than grading them. What it DOES check: the binaries run, the flags are
> accepted, adoption fires (claim 3), the reference is reproducible (P == R
> at an unsqueezed pool), and the adopted-K/V mutant has power (claim 8).

Run `2026-08-27-m5-gate6-dedup-quick`. Fixture **fanout-hetero** (2 clients, pairwise-distinct streams
over a load-bearing shared prefix of 1162 tokens = 72 whole blocks),
16 generated, ctx 1600, threads 8, kv f32, **dedup ON (the serve default)**.
Raw `2026-08-27-m5-gate6-dedup-quick-raw.txt`.

| arm | binary | pool | preempt | steps | rows | adopted | peak blocks | ids sha256 |
|---|---|---|---|---|---|---|---|---|
| R (reference) | `win` | derived | 0 | 26 | 1215 | 1152 | 76 | `FFC6715DA42A9DEC` |
| P (pressure) | `win` | 200 | 0 | 26 | 1215 | 1152 | 76 | `FFC6715DA42A9DEC` |
| MP (mutant) | `win-mut-preempt` | 200 | 0 | 26 | 1215 | 1152 | 76 | `FFC6715DA42A9DEC` |
| MK (mutant) | `win-mut-adoptkv` | 200 | 0 | 26 | 1215 | 1152 | 76 | `94CDCA18BADBAA13` |

## Pre-registered before the first arm ran

| # | prediction | band | measured | result |
|---|---|---|---|---|
| P1 | R's peak = SUM blocks(p_i + n) - adopted/16 | [76, 77] | 76 | **HELD** |
| P2 | R adopts >= (k-1) x 1152 positions | >= 1152 | 1152 | **HELD** |
| P3 | pool 200 preempts at least once | > 0 | 0 | n/a -- ungradeable at this shape |
| P4 | a readmitted victim RE-ADOPTS (P adopts > R) | > 1152 | 1152 | n/a -- ungradeable at this shape |

## The eight claims, each able to fail on its own

| # | claim | result |
|---|---|---|
| 1 | the reference is genuinely unpressured (preemptions == 0) | **PASS** (0) |
| 2 | **the squeeze is real BY MEASUREMENT**: R's peak residency (76) exceeds the squeezed pool (200) | n/a -- ungradeable at this shape |
| 3 | adoption fired in the reference (>= (k-1) x 1152 positions) | **PASS** (1152 of 2337) |
| 4 | the pressure arm genuinely preempted (preemptions > 0) | n/a -- ungradeable at this shape |
| 5 | **a readmitted victim RE-ADOPTED**: P adopted more than R | n/a -- ungradeable at this shape |
| 6 | **preemption lost nothing**: P's stream == R's, byte for byte | n/a -- ungradeable at this shape |
| 7 | the compare sees a REPLAY defect: `MUT_PREEMPT` differs | n/a -- ungradeable at this shape |
| 8 | the compare sees an ADOPTED-KV defect: `MUT_ADOPT_KV` differs | **PASS** |

## VERDICT: preempt-losslessness on the DEDUP path is **NOT GRADED -- this was a pre-flight**

Claims 7 and 8 are why 1-6 mean anything, and there are two of them because the
two failures are different. `MUT_PREEMPT` replays one row too FEW, so the final
replayed row reproduces the logits of the wrong position. `MUT_ADOPT_KV` zeroes
the K/V under the whole adopted prefix at admission -- through the SHARED blocks,
so it damages the donor too, which is the shape a refcount bug would have. Both
leave the run alive and the output merely wrong: the class `kv.h` names as the
dangerous one, and the class a weaker mutant could not show -- `skip16` (16
backed-but-unwritten positions) was MEASURED SAME on 2026-08-25.

## Why the dedup path needed a different window, not a smaller pool

Gate [6] squeezed to 170 with 4 clients and no dedup. The same reasoning at
8 clients WITH dedup put the pool at 84-88 and got serialisation and
**preemptions 0**, because admission and residency measure different things:

- **admission is blind to adoption** -- `admit_if_possible` checks
  `free >= blocks(target) + W` before the `Kv` exists, so every client is
  admitted as if it paid for its whole ~74-block prompt.
- **residency is not** -- the 2nd..kth clients adopt 72 of those blocks by
  refcount and cost the pool ~2 blocks each.

A pool below the RESIDENCY is therefore also below the ADMISSION ceiling, and
admission just runs the clients one at a time. What reopens the window is
**private growth**: the shared prefix is paid once, every generated token is
private, so k clients x n tokens is the term that outgrows the admitted
footprint. That is gate [6]'s own guard #2 ("growth must exceed W*16")
generalised -- widen k and n, do not shrink the pool.

| quantity | blocks |
|---|---|
| footprint once all 2 are admitted, before a token is generated (SUM blocks(p_i) - adopted/16) | 75 |
| **residency at completion -- R's measured peak** | **76** |
| squeezed pool | 200 |
| P's own peak (it cannot exceed its pool) | 76 |

## What this does NOT cover

- **Q8 KV.** Every arm is `--kv f32`. Q8's re-adoption is arithmetically the same
  path, but that is an argument, not this measurement.
- **Staggered arrivals.** All 2 requests are submitted at once (`--stagger 0`),
  which is the shape that makes the pool bind. A late arrival preempting an
  already-decoding cohort is a different trace.
- **The admission asymmetry itself is UNFIXED and is a finding, not a defect
  fixed here.** With dedup on, admission reserves the un-adopted prompt, so under
  memory pressure the shipping default admits fewer sequences than the pool can
  hold, and readmits a victim only when `free >= blocks(target) + W` although it
  then occupies `blocks(target) - adopted`. It is deliberate (sched.c: "admitting
  on the un-adopted size is conservative in the only direction that matters") and
  it is why this trace exists at all. Whether it should hold is an M6 question.
