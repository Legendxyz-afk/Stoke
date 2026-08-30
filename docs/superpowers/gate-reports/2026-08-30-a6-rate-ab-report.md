# A6's RATE HALF — the M5-vs-A6 A/B, with a null arm

**Date:** 2026-08-30 · **Cell:** 8f32 (S2, 8 clients, ctx 704, n 128, threads 8,
`--no-dedup`, kv f32) · **Metric:** `AGGREGATE(M0)` · **6 blocks × 3 arms = 18
units**, sliced one unit per invocation and banked as each landed.
**Bank:** `2026-08-30-a6-rate-ab-bank.jsonl`.

## What was asked, and what was deliberately NOT asked

M6's standing constraint: *if a refactor for `stoke.h` touches the scheduler,
the M5 bars are re-measured or the refactor does not land.*

That constraint protects against A6 silently costing rate. It is discharged here
by an **attribution** — did this milestone's code change the rate? — and not by
re-running the bars gate, for a stated reason:

- The bars gate answers *"is the ABSOLUTE rate still above the threshold"*, and
  its numbers are meaningful only on a box that meets its quiesce bar.
- **This box does not meet it today.** Measured with the gate's own instrument
  (`Win32_PerfFormattedData_PerfOS_Processor`, `_Total`, 12 reads 2 s apart):
  median **21.5%**, max 44%, zero consecutive reads ≤ 12%. After stopping ISLC
  (which `run_m5_bars.ps1` throws on): median **14.5%**, max 27%, one
  consecutive pair ≤ 12%. Per-process deltas attribute only ~6.9% of it, the
  same unattributed residue the 2026-08-29 attempt found before voiding itself.
  That attempt's own note is the precedent: a pass at exactly 12% is *"the tail
  of that distribution rather than a quiet box briefly observed"*.
- **An A/B with a null arm is the instrument that survives this**, because the
  noise lands in the null and bounds the claim. Nothing below is compared to the
  banked 8.960; that would be exactly the cross-sitting attribution the standing
  rules forbid.

## The arms

| arm | binary | sha256[0:16] |
|---|---|---|
| **M5** | tag `m5` (`bc969b9`), built in a worktree with **identical CFLAGS** (only `CXXFLAGS` and new link rules changed since) | `f8e51b4c73df504a` |
| **A6** | today's `HEAD` | `1e265cf22e26ed99` |
| **NULL** | today's `HEAD` **again** — a second process, same code | `1e265cf22e26ed99` |

**NULL is the denominator.** Whatever it reports is what this harness cannot
resolve *at the level the comparison is made* — process against process — and
M5-vs-A6 means nothing except against it.

## Pre-registered prediction (written before the first unit ran)

A **null result**. From the diff rather than from memory, A6 adds to this path
exactly:

- `serve_step` gains `if (h->sample)` — **one perfectly-predicted branch, outside
  the token loop** — and `ServeHost` gains two pointers;
- nothing else. `sample_prologue` (the mutex and the slot sweep) lives in
  `stoke.c`, and **`bench --serve` never enters `stoke.c`** — it drives
  `serve_host`/`sched` directly. `sched.c` gained only `sched_is_live`, which
  nothing on this path calls. `pool_prefer_performance_here` is called only from
  `bench --ops`, not from here.

## Two pre-flights, because arms that do different work are not comparable

1. **Byte-identical output.** Both binaries driven with `--ids` on this cell:
   `cmp` clean, 1534 bytes. Greedy decoding being unchanged is itself an A6
   claim, and this is where it is checked.
2. **Identical scheduler work.** `steps` is **167 in all 18 units** — not a
   median, an exact match in every run of both binaries.

## Results

| arm | n | values (tok/s) | median |
|---|---|---|---|
| M5 | 6 | 5.572 5.703 5.846 5.861 5.888 5.977 | **5.846** |
| A6 | 6 | 5.258 5.726 5.853 6.038 6.052 6.138 | **5.853** |
| NULL | 6 | 5.604 5.720 6.008 6.029 6.114 6.202 | **6.008** |

Block-paired ratios (arms ran back to back inside each block, order
counterbalanced across blocks so no arm sits in the warming position twice):

| block | A6/M5 | A6/NULL |
|---|---|---|
| b1 | 1.0302 | 1.0556 |
| b2 | 0.9436 | 0.9383 |
| b3 | 1.0499 | 0.9897 |
| b4 | 1.0279 | 0.9899 |
| b5 | 0.9580 | 0.9497 |
| b6 | 1.0263 | 0.9742 |
| **span** | **0.944 – 1.050 (10.6 pp)** | **0.938 – 1.056 (11.7 pp)** |

## Verdict

**No measurable rate change. The prediction held.**

- Median M5 → A6: **5.846 → 5.853, a difference of 0.12%.**
- The floor, from the null arm: **2.6% on medians of six, and ±5–6% per block.**
- **The A6-vs-M5 spread (10.6 pp) is NARROWER than the A6-vs-itself spread
  (11.7 pp).** Comparing A6 to the M5 binary looks exactly like comparing A6 to
  a second copy of A6. The observed effect is ~20× below what this harness can
  resolve.

⚠ **What this does NOT say.** It does not say A6 costs zero — no box-level
instrument can resolve one predicted branch per step against a step that is a
full forward pass. It says the cost is below 2.6% on medians / ~5% per block at
this cell, and the diff says why it should be far below that. It also says
nothing about absolute rate: **the 5.8–6.0 tok/s here must not be compared to
the banked 8.960**, which was taken on a box that met a quiesce bar this one
does not.

## Open

- **The bars gate's ABSOLUTES have not been re-taken** and still want a box that
  passes the quiesce bar. That is a separate sitting and Kabir's call; the
  constraint's purpose is discharged above, but the gate's literal numbers are
  from M5.
- The 16f32 / q8 cells were not run. Nothing on this path scales with `--seqs`
  (the slot sweep that does is in `stoke.c`), so a second cell was judged to add
  little against ~20 more minutes; that judgement is recorded rather than hidden.
