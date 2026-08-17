# M4.5 overnight P-BAR evaluation

Run 2026-08-17-m45-pbar-same-threads. Arms: A = A pre-maddubs, B = B HEAD maddubs+Q6_K, L = llama-bench b10037.

- A: `M:\New Drive\Desktop\Programming\stoke\build\stoke-m45-c-head-bc9dd3d6.exe`
- B: `M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe`
N=1 via `batch --seqs 1`, p00, -n 128. Threads: **A=6, B=6, L=6** -- each arm at the count where that engine is fastest, per the 2026-08-17 thread-scaling curve. Affinity PINNED 0xFFF (P-core siblings) over 16 logical processors. Arm order rotates per cycle.

## Correctness (owed re-runs)

- **IDENTICAL** -- Q4_K_M p41 relinked vs pre-maddubs
- **IDENTICAL** -- Q8_0 p41 anchor vs m3 archive

## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 10.031 | 15.436 | 18.41 | 1.539x | 0.838x | 8.82 | 1 |
| 2 | 5.393 | 9.978 | 18.73 | 1.850x | 0.533x | 7.28 | 3 |
| 3 | 10.048 | 15.508 | 18.62 | 1.543x | 0.833x | 7.31 | 28 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 9.991
- B maddubs     : 15.369
- L llama-bench : 18.62

## The anchor guard

Arm A measured **9.991** against its quiesced anchor of 8.262, a deviation of **20.9%**.

âš  **A did not reproduce its anchor within 10%.** Every ratio computed against it is suspect: a depressed baseline inflates the speedup. Treat the B/A column as UNRELIABLE and read B/L instead, which does not depend on A.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.83 tok/s**. B measured **15.369** at 6 threads against llama's **18.62** at 6, giving **0.825x**.

Both engines run at their own fastest thread count, so neither side is handicapped. Grading stoke at 8 against a llama held at 6 would clear the bar by arithmetic rather than by speed.

**P-BAR NOT MET** -- short by 1.030x. Remaining sized levers: Q6_K is already landed, so what is left is D (repack, gate is occupancy) and the per-thread deficit the curve located at low t.

## Stability across blocks

Arm B across every run: min 9.957, max 15.606, spread 1.57x over 9 runs spanning the night.

A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure.

Bank: `2026-08-17-m45-pbar-same-threads-bank.jsonl`. Raw log: `2026-08-17-m45-pbar-same-threads-raw.txt`.
