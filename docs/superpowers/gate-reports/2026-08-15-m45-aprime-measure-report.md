# M4.5 overnight P-BAR evaluation

Run 2026-08-15-m45-aprime-measure. Arms: A = A maddubs (main@650c47c), B = B A-prime (m45-aprime), L = llama-bench b10037.

- A: `M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe`
- B: `M:\New Drive\Desktop\Programming\stoke\build\win-ap\stoke.exe`
N=1 via `batch --seqs 1`, p00, -n 128, 6 threads, P-cores. Arm order rotates per cycle.

## Correctness (owed re-runs)


## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 12.375 | 11.586 | 18.39 | 0.936x | 0.630x | 0.51 âš  disk-bound | 0 |
| 2 | 12.605 | 11.381 | 18.29 | 0.903x | 0.622x | 0.67 âš  disk-bound | 59 |
| 3 | 12.612 | 11.519 | 18.37 | 0.913x | 0.627x | 0.7 âš  disk-bound | 20 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 12.611
- B maddubs     : 11.435
- L llama-bench : 18.37

## The anchor guard

Arm A measured **12.611** against its quiesced anchor of 12.57, a deviation of **0.3%**.

A reproduced its anchor within 10%, so the B/A ratios rest on a baseline that is behaving as it did when the anchor was frozen.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.61 tok/s**. B measured **11.435**.

**P-BAR NOT MET** -- short by 1.366x. A' (row tiling) is the next lever and is not optional.

## Stability across blocks

Arm B across every run: min 11.048, max 11.602, spread 1.05x over 9 runs spanning the night.

A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure.

Bank: `2026-08-15-m45-aprime-measure-bank.jsonl`. Raw log: `2026-08-15-m45-aprime-measure-raw.txt`.

---

## Addendum, written after the run by the session that set it up

**THE HEADLINE: A' IS A REGRESSION AND MUST NOT BE KEPT.** Arm A is maddubs
(`main@650c47c`), arm B is A' (`m45-aprime`) -- the table's column headers
are stale generator text, see (1) below.

    A' = 11.435 / 12.611 = 0.907x, a 9.3% SLOWDOWN
    per block: 0.936x, 0.903x, 0.913x -- consistent, no overlap in the ranges

**And this measurement is sound, unlike the P-BAR run's.** The anchor guard was
set to 12.57 -- maddubs' own median from the P-BAR run at the identical
workload -- and arm A came back at **12.611, a deviation of +0.3%**. A guard
calibrated on the workload it grades, passing. llama also reproduced (18.37 vs
its 18.05 anchor).

⇒ **§7.3 fires: "keep whichever variant is fastest, INCLUDING KEEPING NONE."**
A' is not merely under the 5% threshold, it is negative. **The single-row path
stays; A' is recorded as TRIED AND REJECTED ON MEASUREMENT.** Reverting is one
line in `rows_q4_k_avx2`. The kernel, its test and its docs are kept on the
branch as the record of what was tried.

Likely cause, hypothesis and unmeasured: the Q8_0 x4 win came when the inner
loop ran 8 products/uop and was latency-bound. maddubs changed the regime --
four concurrent weight streams, four accumulators and per-super-block
`scales8[4][8]`/`mins8[4][8]` setup now add register and cache pressure to a
kernel that is already dense. **Row tiling solved a problem maddubs had already
removed.**

### Three stale lines in the generated text above

1. **The table headers say "A pre-maddubs | B maddubs".** Those are hardcoded
   in the report generator; the `-ArmALabel`/`-ArmBLabel` parameters reach
   the "Arms:" line but not the table. Cosmetic, but read the Arms line, not
   the headers. Worth fixing before the generator is used a third time.
2. **"A' (row tiling) is the next lever and is not optional"** is hardcoded
   P-BAR advice and is now moot -- this run IS the A' measurement, and it
   rejected A'. With A' eliminated the remaining levers are Q6_K-on-maddubs
   (sized at 1.20x, a floor) and D (repack, RAM-feasible, gate is occupancy).
3. **The "disk-bound" flags on all three blocks are the known non-issue** --
   P-BAR blocks 1 and 2 gave identical ratios at 0.19 GB and 8.08 GB standby.
   The model is mmap'd and `--timing` reports load separately, so an empty
   standby costs the fault-in, not the decode rate.

### What it says about P-BAR

B/L here is A' against llama: 0.622-0.630, versus maddubs' 0.692. A' moves the
engine AWAY from the bar. The P-BAR verdict stands where the previous run left
it: **maddubs 12.523, bar 15.39, short by 1.229x.**
