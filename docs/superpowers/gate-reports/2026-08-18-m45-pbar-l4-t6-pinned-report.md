# M4.5 overnight P-BAR evaluation

Run 2026-08-18-m45-pbar-l4-t6-pinned. Arms: A = A pre-maddubs, B = B HEAD + L4, L = llama-bench b10037.

- A: `M:\New Drive\Desktop\Programming\stoke\build\stoke-m45-c-head-bc9dd3d6.exe`
- B: `M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe`
N=1 via `batch --seqs 1`, p00, -n 128. Threads: **A=6, B=6, L=6** -- each arm at the count where that engine is fastest, per the 2026-08-17 thread-scaling curve. Affinity PINNED 0xFFF (P-core siblings) over 16 logical processors. Arm order rotates per cycle.

## Correctness (owed re-runs)

- **IDENTICAL** -- Q4_K_M p41 relinked vs pre-maddubs
- **IDENTICAL** -- Q8_0 p41 anchor vs m3 archive

## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 5.293 | 16.369 | 18.04 | 3.093x | 0.907x | 4.71 | 9 |
| 2 | 5.303 | 16.672 | 18.13 | 3.144x | 0.920x | 4.87 | 0 |
| 3 | 5.298 | 16.529 | 18.21 | 3.120x | 0.908x | 5.17 | 12 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 5.298
- B maddubs     : 16.529
- L llama-bench : 18.13

## The anchor guard

Arm A measured **5.298** against its quiesced anchor of 9.989, a deviation of **-47.0%**.

! **A did not reproduce its anchor within 10%.** Every ratio computed against it is suspect: a depressed baseline inflates the speedup. Treat the B/A column as UNRELIABLE and read B/L instead, which does not depend on A.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.41 tok/s**. B measured **16.529** at 6 threads against llama's **18.13** at 6, giving **0.912x**.

Both engines run at their own fastest thread count, so neither side is handicapped. Grading stoke at 8 against a llama held at 6 would clear the bar by arithmetic rather than by speed.

**P-BAR MET.**

## Stability across blocks

Arm B across every run: min 15.198, max 16.734, spread 1.10x over 9 runs spanning the night.

A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure.

## Wrong-cores check (CPU-seconds per unit)

Windows EcoQoS parks the pool on the efficiency cores, where the SAME work costs ~4x more processor-time -- so a depressed unit reads HIGH here while also finishing later. That makes this tell independent of the rate. Fixed in the engine at `76dee7f`; the check stays to catch the next thing that quietly moves work somewhere slower.

| arm | median CPU-s | max | flagged |
|---|---|---|---|
| A | 130.7 | 132.1 | 0 |
| B | 47.8 | 49 | 0 |
| L | 130.2 | 131.1 | 0 |

Bank: `2026-08-18-m45-pbar-l4-t6-pinned-bank.jsonl`. Raw log: `2026-08-18-m45-pbar-l4-t6-pinned-raw.txt`.
