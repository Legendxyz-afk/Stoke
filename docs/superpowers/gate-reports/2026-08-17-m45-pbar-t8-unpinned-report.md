# M4.5 overnight P-BAR evaluation

Run 2026-08-17-m45-pbar-t8-unpinned. Arms: A = A pre-maddubs @t=8 unpinned, B = B HEAD @t=8 unpinned, L = llama-bench b10037.

- A: `M:\New Drive\Desktop\Programming\stoke\build\stoke-m45-c-head-bc9dd3d6.exe`
- B: `M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe`
N=1 via `batch --seqs 1`, p00, -n 128. Threads: **A=8, B=8, L=8** -- each arm at the count where that engine is fastest, per the 2026-08-17 thread-scaling curve. Affinity unpinned over 16 logical processors. Arm order rotates per cycle.

## Correctness (owed re-runs)


## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 10.842 | 15.492 | 18.74 | 1.429x | 0.827x | 7.61 | 13 |
| 2 | 10.825 | 15.475 | 18.7 | 1.430x | 0.828x | 7.64 | 8 |
| 3 | 10.835 | 15.516 | 18.59 | 1.432x | 0.835x | 7.65 | 7 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 10.835
- B maddubs     : 15.492
- L llama-bench : 18.7

## The anchor guard

Arm A measured **10.835** against its quiesced anchor of 10.01, a deviation of **8.2%**.

A reproduced its anchor within 10%, so the B/A ratios rest on a baseline that is behaving as it did when the anchor was frozen.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.90 tok/s**. B measured **15.492** at 8 threads against llama's **18.7** at 8, giving **0.828x**.

Both engines run at their own fastest thread count, so neither side is handicapped. Grading stoke at 8 against a llama held at 6 would clear the bar by arithmetic rather than by speed.

**P-BAR NOT MET** -- short by 1.026x. Remaining sized levers: Q6_K is already landed, so what is left is D (repack, gate is occupancy) and the per-thread deficit the curve located at low t.

## Stability across blocks

Arm B across every run: min 15.294, max 15.554, spread 1.02x over 9 runs spanning the night.

A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure.

## Wrong-cores check (CPU-seconds per unit)

Windows EcoQoS parks the pool on the efficiency cores, where the SAME work costs ~4x more processor-time -- so a depressed unit reads HIGH here while also finishing later. That makes this tell independent of the rate. Fixed in the engine at `76dee7f`; the check stays to catch the next thing that quietly moves work somewhere slower.

| arm | median CPU-s | max | flagged |
|---|---|---|---|
| A | 88.4 | 92.3 | 0 |
| B | 58 | 59.8 | 0 |
| L | 168.05 | 170.4 | 0 |

Bank: `2026-08-17-m45-pbar-t8-unpinned-bank.jsonl`. Raw log: `2026-08-17-m45-pbar-t8-unpinned-raw.txt`.
