# M4.5 overnight P-BAR evaluation

Run 2026-08-15-smoke-ap. Arms: A = A maddubs, B = B A-prime, L = llama-bench b10037.

- A: `M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe`
- B: `M:\New Drive\Desktop\Programming\stoke\build\win-ap\stoke.exe`
N=1 via `batch --seqs 1`, p00, -n 16, 6 threads, P-cores. Arm order rotates per cycle.

## Correctness (owed re-runs)


## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 14.98 | 13.518 | 18.48 | 0.902x | 0.731x | 7.54 | 23 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 14.98
- B maddubs     : 13.518
- L llama-bench : 18.48

## The anchor guard

Arm A measured **14.98** against its quiesced anchor of 12.5, a deviation of **19.8%**.

âš  **A did not reproduce its anchor within 10%.** Every ratio computed against it is suspect: a depressed baseline inflates the speedup. Treat the B/A column as UNRELIABLE and read B/L instead, which does not depend on A.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.71 tok/s**. B measured **13.518**.

**P-BAR NOT MET** -- short by 1.162x. A' (row tiling) is the next lever and is not optional.

## Stability across blocks


Bank: `2026-08-15-smoke-ap-bank.jsonl`. Raw log: `2026-08-15-smoke-ap-raw.txt`.
