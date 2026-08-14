# M4.5 overnight P-BAR evaluation

Run 2026-08-14-smoke3. Arms: A = pre-maddubs (`bc9dd3d6`), B = maddubs (HEAD), L = llama-bench b10037.
N=1 via `batch --seqs 1`, p00, -n 16, 6 threads, P-cores. Arm order rotates per cycle.

## Correctness (owed re-runs)

- **IDENTICAL** -- Q4_K_M p41 relinked vs pre-maddubs
- **IDENTICAL** -- Q8_0 p41 anchor vs m3 archive

## Rates by block

| block | A pre-maddubs | B maddubs | L llama | B/A | B/L | standby GB | cpu % |
|---|---|---|---|---|---|---|---|
| 1 | 9.737 | 14.91 | 16.67 | 1.531x | 0.894x | 6.42 | 22 |
| 2 | 9.7 | 14.584 | 18.36 | 1.504x | 0.794x | 6.43 | 0 |

A block whose standby cache is under 2.0 GB against a 2.33 GB model was reading the model from DISK and its rates are not comparable -- free RAM reads healthy while the model is entirely absent from cache. Drop such a block and say so rather than averaging it in.

## Overall medians

- A pre-maddubs : 9.7
- B maddubs     : 14.584
- L llama-bench : 16.67

## The anchor guard

Arm A measured **9.7** against its quiesced anchor of 10.01, a deviation of **-3.1%**.

A reproduced its anchor within 10%, so the B/A ratios rest on a baseline that is behaving as it did when the anchor was frozen.

## P-BAR

P-BAR is 0.85x llama measured in the same sitting: **14.17 tok/s**. B measured **14.584**.

**P-BAR MET.**

## Stability across blocks

Arm B across every run: min 14.584, max 14.91, spread 1.02x over 2 runs spanning the night.

A spread materially wider than a few percent means the box drifted and the medians are the honest summary rather than any single figure.

Bank: `2026-08-14-smoke3-bank.jsonl`. Raw log: `2026-08-14-smoke3-raw.txt`.
