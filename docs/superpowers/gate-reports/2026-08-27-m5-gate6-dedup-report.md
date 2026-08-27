# M5 gate [6] on the shipping default -- preempt-lossless with prefix dedup ON

Run `2026-08-27-m5-gate6-dedup`. Fixture **fanout-hetero** (8 clients, pairwise-distinct streams
over a load-bearing shared prefix of 1162 tokens = 72 whole blocks),
300 generated, ctx 1600, threads 8, kv f32, **dedup ON (the serve default)**.
Raw `2026-08-27-m5-gate6-dedup-raw.txt`.

| arm | binary | pool | preempt | steps | rows | adopted | peak blocks | ids sha256 |
|---|---|---|---|---|---|---|---|---|
| R (reference) | `win` | derived | 0 | 316 | 3685 | 8064 | 232 | `5F221697F4CEFD81` |
| P (pressure) | `win` | 200 | 2 | 388 | 4214 | 10368 | 200 | `5F221697F4CEFD81` |
| MP (mutant) | `win-mut-preempt` | 200 | 2 | 388 | 4212 | 10368 | 200 | `CBDC34288D99B42A` |
| MK (mutant) | `win-mut-adoptkv` | 200 | 2 | 388 | 4214 | 10368 | 200 | `83F9E8F3971B7ADC` |

### The same comparison without the hash

A sha256 says SAME or DIFFERENT and nothing else. This is the id streams read
back and compared position by position, which is what separates a real DIFFER
from an arm that merely died early:

| arm | ids produced (expect 2400) | first id that differs from R |
|---|---|---|
| P (pressure) | 2400 | none -- identical |
| MP (mutant) | 2400 | #2068 of 2400 |
| MK (mutant) | 2400 | #2 of 2400 |

## Pre-registered before the first arm ran

| # | prediction | band | measured | result |
|---|---|---|---|---|
| P1 | R's peak = SUM blocks(p_i + n) - adopted/16 | [232, 233] | 232 | **HELD** |
| P2 | R adopts >= (k-1) x 1152 positions | >= 8064 | 8064 | **HELD** |
| P3 | pool 200 preempts at least once | > 0 | 2 | **HELD** |
| P4 | a readmitted victim RE-ADOPTS (P adopts > R) | > 8064 | 10368 | **HELD** |

## The eight claims, each able to fail on its own

| # | claim | result |
|---|---|---|
| 1 | the reference is genuinely unpressured (preemptions == 0) | **PASS** (0) |
| 2 | **the squeeze is real BY MEASUREMENT**: R's peak residency (232) exceeds the squeezed pool (200) | **PASS** |
| 3 | adoption fired in the reference (>= (k-1) x 1152 positions) | **PASS** (8064 of 9357) |
| 4 | the pressure arm genuinely preempted (preemptions > 0) | **PASS** (2) |
| 5 | **a readmitted victim RE-ADOPTED**: P adopted more than R | **PASS** (10368 of 11695 vs 8064 of 9357) |
| 6 | **preemption lost nothing**: P's stream == R's, byte for byte | **PASS** |
| 7 | the compare sees a REPLAY defect: `MUT_PREEMPT` differs | **PASS** |
| 8 | the compare sees an ADOPTED-KV defect: `MUT_ADOPT_KV` differs | **PASS** |

## VERDICT: preempt-losslessness on the DEDUP path is **DEMONSTRATED**

Claims 7 and 8 are why 1-6 mean anything, and there are two of them because the
two failures are different. `MUT_PREEMPT` replays one row too FEW, so the final
replayed row reproduces the logits of the wrong position. `MUT_ADOPT_KV` zeroes
the K/V under the whole adopted prefix at admission -- through the SHARED blocks,
so it damages the donor too, which is the shape a refcount bug would have. Both
leave the run alive and the output merely wrong: the class `kv.h` names as the
dangerous one, and the class a weaker mutant could not show -- `skip16` (16
backed-but-unwritten positions) was MEASURED SAME on 2026-08-25.

MP's defect is legible in the LEDGER as well as in the stream: it stepped 4212
rows against P's 4214, exactly one fewer per preemption. That is the off-by-one
counted rather than argued, and it is also why the mutant is not merely a
different binary that happened to disagree.

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
| footprint once all 8 are admitted, before a token is generated (SUM blocks(p_i) - adopted/16) | 86 |
| **residency at completion -- R's measured peak** | **232** |
| squeezed pool | 200 |
| P's own peak (it cannot exceed its pool) | 200 |

## A finding this trace hands over for free: adoption makes PREEMPTION cheap

Decode rows are the SAME in both arms -- every request emits its first id from
its first prefill and a replay re-emits nothing -- so **P's extra rows are
entirely re-prefill**: 4214 - 3685 = **529 rows over 2 preemptions = ~264.5 each**.
Each readmission adopted 1152 positions (P's adoption total less R's, over the
same preemptions), so the victim replayed a target of ~1416.5 positions and
**recomputed only 264.5 of them**.

Prefix dedup is normally argued for at ADMISSION. This measures it at
PREEMPTION: the re-prefill a victim pays is cut by the whole shared prefix,
because `preempt()` closes the Kv and resets `n_done` to 0, so readmission
arrives at `kv_adopt_prefix` as a FRESH sequence -- the one state it accepts.
The victim's blocks came back and the shared ones never left.

## What this does NOT cover

- **Q8 KV.** Every arm is `--kv f32`. Q8's re-adoption is arithmetically the same
  path, but that is an argument, not this measurement.
- **Staggered arrivals.** All 8 requests are submitted at once (`--stagger 0`),
  which is the shape that makes the pool bind. A late arrival preempting an
  already-decoding cohort is a different trace.
- **The admission asymmetry itself is UNFIXED and is a finding, not a defect
  fixed here.** With dedup on, admission reserves the un-adopted prompt, so under
  memory pressure the shipping default admits fewer sequences than the pool can
  hold, and readmits a victim only when `free >= blocks(target) + W` although it
  then occupies `blocks(target) - adopted`. It is deliberate (sched.c: "admitting
  on the un-adopted size is conservative in the only direction that matters") and
  it is why this trace exists at all. Whether it should hold is an M6 question.
