# M4.5 overnight P-BAR evaluation

Run 2026-08-15-m45-xn-measure. Arms: A = A widening _xn (main), B = B maddubs _xn (branch), L = llama-bench b10037.

- A: `M:\New Drive\Desktop\Programming\stoke\build\win\stoke.exe`
- B: `M:\New Drive\Desktop\Programming\stoke\build\win-ap\stoke.exe`
N=1 via `batch --seqs 1`, p00, -n 32, 6 threads, P-cores. Arm order rotates per cycle.

## Correctness (owed re-runs)


## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 7.258 | 12.131 | 18.5 | 1.671x | 0.656x | 0.87 âš  disk-bound | 16 |
| 2 | 7.686 | 12.467 | 18.58 | 1.622x | 0.671x | 1.09 âš  disk-bound | 0 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 7.258
- B maddubs     : 12.393
- L llama-bench : 18.5

## The anchor guard

Arm A measured **7.258** against its quiesced anchor of 12.646, a deviation of **-42.6%**.

âš  **A did not reproduce its anchor within 10%.** Every ratio computed against it is suspect: a depressed baseline inflates the speedup. Treat the B/A column as UNRELIABLE and read B/L instead, which does not depend on A.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.73 tok/s**. B measured **12.393**.

**P-BAR NOT MET** -- short by 1.269x. A' (row tiling) is the next lever and is not optional.

## Stability across blocks

Arm B across every run: min 11.897, max 12.632, spread 1.06x over 6 runs spanning the night.

A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure.

Bank: `2026-08-15-m45-xn-measure-bank.jsonl`. Raw log: `2026-08-15-m45-xn-measure-raw.txt`.
