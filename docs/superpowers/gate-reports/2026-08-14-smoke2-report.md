# M4.5 overnight P-BAR evaluation

Run 2026-08-14-smoke2. Arms: A = pre-maddubs (`bc9dd3d6`), B = maddubs (HEAD), L = llama-bench b10037.
N=1 via `batch --seqs 1`, p00, -n 16, 6 threads, P-cores. Arm order rotates per cycle.

## Correctness (owed re-runs)

- **IDENTICAL** -- Q4_K_M p41 relinked vs pre-maddubs
- **IDENTICAL** -- Q8_0 p41 anchor vs m3 archive

## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L |
|---|---|---|---|---|---|
| 1 | 9.725 | 14.675 | 17.94 | 1.509x | 0.818x |

## Overall medians

- A pre-maddubs : 9.725
- B maddubs     : 14.675
- L llama-bench : 17.94

## The anchor guard

Arm A measured **9.725** against its quiesced anchor of 10.01, a deviation of **-2.8%**.

A reproduced its anchor within 10%, so the B/A ratios rest on a baseline that is behaving as it did when the anchor was frozen.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **15.25 tok/s**. B measured **14.675**.

**P-BAR NOT MET** -- short by 1.039x. A' (row tiling) is the next lever and is not optional.

## Stability across blocks


Bank: `2026-08-14-smoke2-bank.jsonl`. Raw log: `2026-08-14-smoke2-raw.txt`.
