# M4.5 overnight P-BAR evaluation

Run 2026-08-19-m45-armA-repro. Arms: A = A pre-maddubs, B = B maddubs, L = llama-bench b10037.

- A: `M:\New Drive\Desktop\Programming\stoke\build\stoke-m45-c-head-bc9dd3d6.exe`
- B: `M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe`
N=1 via `batch --seqs 1`, p00, -n 64. Threads: **A=8, B=8, L=8** -- each arm at the count where that engine is fastest, per the 2026-08-17 thread-scaling curve. Affinity unpinned over 16 logical processors. Arm order rotates per cycle.

## Correctness (owed re-runs)


## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 8.61 | 15.53 | 18.63 | 1.804x | 0.834x | 2.53 | 29 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 8.61
- B maddubs     : 15.53
- L llama-bench : 18.63

## The anchor guard

Arm A measured **8.61** against its quiesced anchor of 10.835, a deviation of **-20.5%**.

! **A did not reproduce its anchor within 10%.** Every ratio computed against it is suspect: a depressed baseline inflates the speedup. Treat the B/A column as UNRELIABLE and read B/L instead, which does not depend on A.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.84 tok/s**. B measured **15.53** at 8 threads against llama's **18.63** at 8, giving **0.834x**.

Both engines run at their own fastest thread count, so neither side is handicapped. Grading stoke at 8 against a llama held at 6 would clear the bar by arithmetic rather than by speed.

**P-BAR NOT MET** -- short by 1.020x. Remaining sized levers: Q6_K is already landed, so what is left is D (repack, gate is occupancy) and the per-thread deficit the curve located at low t.

## Stability across blocks


## Wrong-cores check (CPU-seconds per unit)

Windows EcoQoS parks the pool on the efficiency cores, where the SAME work costs ~4x more processor-time -- so a depressed unit reads HIGH here while also finishing later. That makes this tell independent of the rate. Fixed in the engine at `76dee7f`; the check stays to catch the next thing that quietly moves work somewhere slower.

| arm | median CPU-s | max | flagged |
|---|---|---|---|
| A | 60 | 60 | 0 |
| B | 36.1 | 36.1 | 0 |
| L | 86.2 | 86.2 | 0 |

Bank: `2026-08-19-m45-armA-repro-bank.jsonl`. Raw log: `2026-08-19-m45-armA-repro-raw.txt`.
