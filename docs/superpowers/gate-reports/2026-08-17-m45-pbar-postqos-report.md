# M4.5 overnight P-BAR evaluation

Run 2026-08-17-m45-pbar-postqos. Arms: A = A pre-maddubs (no QoS fix), B = B HEAD maddubs+Q6_K+QoS, L = llama-bench b10037.

- A: `M:\New Drive\Desktop\Programming\stoke\build\stoke-m45-c-head-bc9dd3d6.exe`
- B: `M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe`
N=1 via `batch --seqs 1`, p00, -n 128. Threads: **A=6, B=6, L=6** -- each arm at the count where that engine is fastest, per the 2026-08-17 thread-scaling curve. Affinity PINNED 0xFFF (P-core siblings) over 16 logical processors. Arm order rotates per cycle.

## Correctness (owed re-runs)

- **IDENTICAL** -- Q4_K_M p41 relinked vs pre-maddubs
- **IDENTICAL** -- Q8_0 p41 anchor vs m3 archive

## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 9.985 | 15.42 | 18.69 | 1.544x | 0.825x | 7.59 | 20 |
| 2 | 9.989 | 15.451 | 18.66 | 1.547x | 0.828x | 7.6 | 35 |
| 3 | 9.997 | 15.489 | 18.71 | 1.549x | 0.828x | 7.6 | 2 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 9.989
- B maddubs     : 15.451
- L llama-bench : 18.69

## The anchor guard

Arm A measured **9.989** against its quiesced anchor of 8.262, a deviation of **20.9%**.

âš  **A did not reproduce its anchor within 10%.** Every ratio computed against it is suspect: a depressed baseline inflates the speedup. Treat the B/A column as UNRELIABLE and read B/L instead, which does not depend on A.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.89 tok/s**. B measured **15.451** at 6 threads against llama's **18.69** at 6, giving **0.827x**.

Both engines run at their own fastest thread count, so neither side is handicapped. Grading stoke at 8 against a llama held at 6 would clear the bar by arithmetic rather than by speed.

**P-BAR NOT MET** -- short by 1.028x. Remaining sized levers: Q6_K is already landed, so what is left is D (repack, gate is occupancy) and the per-thread deficit the curve located at low t.

## Stability across blocks

Arm B across every run: min 15.043, max 15.497, spread 1.03x over 9 runs spanning the night.

A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure.

## Wrong-cores check (CPU-seconds per unit)

Windows EcoQoS parks the pool on the efficiency cores, where the SAME work costs ~4x more processor-time -- so a depressed unit reads HIGH here while also finishing later. That makes this tell independent of the rate. Fixed in the engine at `76dee7f`; the check stays to catch the next thing that quietly moves work somewhere slower.

| arm | median CPU-s | max | flagged |
|---|---|---|---|
| A | 75.6 | 76.2 | 0 |
| B | 46.2 | 47.6 | 0 |
| L | 126.35 | 126.6 | 0 |

Bank: `2026-08-17-m45-pbar-postqos-bank.jsonl`. Raw log: `2026-08-17-m45-pbar-postqos-raw.txt`.
