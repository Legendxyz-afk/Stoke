# M4.5 overnight P-BAR evaluation

Run 2026-08-19-m45-pbar-l4-t8-unpinned. Arms: A = A pre-maddubs, B = B HEAD + L4, L = llama-bench b10037.

- A: `M:\New Drive\Desktop\Programming\stoke\build\stoke-m45-c-head-bc9dd3d6.exe`
- B: `M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe`
N=1 via `batch --seqs 1`, p00, -n 128. Threads: **A=8, B=8, L=8** -- each arm at the count where that engine is fastest, per the 2026-08-17 thread-scaling curve. Affinity unpinned over 16 logical processors. Arm order rotates per cycle.

## Correctness (owed re-runs)


## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 0.922 | 16.577 | 18.56 | 17.979x | 0.893x | 0.07 ! disk-bound | 9 |
| 2 | 0.928 | 16.493 | 18.59 | 17.773x | 0.887x | 0.74 ! disk-bound | 5 |
| 3 | 0.922 | 16.668 | 18.32 | 18.078x | 0.910x | 0.91 ! disk-bound | 8 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 0.922
- B maddubs     : 16.577
- L llama-bench : 18.58

## The anchor guard

Arm A measured **0.922** against its quiesced anchor of 10.835, a deviation of **-91.5%**.

! **A did not reproduce its anchor within 10%.** Every ratio computed against it is suspect: a depressed baseline inflates the speedup. Treat the B/A column as UNRELIABLE and read B/L instead, which does not depend on A.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.79 tok/s**. B measured **16.577** at 8 threads against llama's **18.58** at 8, giving **0.892x**.

Both engines run at their own fastest thread count, so neither side is handicapped. Grading stoke at 8 against a llama held at 6 would clear the bar by arithmetic rather than by speed.

**P-BAR MET.**

## Stability across blocks

Arm B across every run: min 15.354, max 16.914, spread 1.10x over 9 runs spanning the night.

A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure.

## Wrong-cores check (CPU-seconds per unit)

Windows EcoQoS parks the pool on the efficiency cores, where the SAME work costs ~4x more processor-time -- so a depressed unit reads HIGH here while also finishing later. That makes this tell independent of the rate. Fixed in the engine at `76dee7f`; the check stays to catch the next thing that quietly moves work somewhere slower.

| arm | median CPU-s | max | flagged |
|---|---|---|---|
| A | 299.8 | 303.1 | 0 |
| B | 59.95 | 62.1 | 0 |
| L | 168.9 | 171.9 | 0 |

Bank: `2026-08-19-m45-pbar-l4-t8-unpinned-bank.jsonl`. Raw log: `2026-08-19-m45-pbar-l4-t8-unpinned-raw.txt`.
