# M4.5 -- thread-scaling curve, stoke vs llama

Run `2026-08-17-m45-thread-curve`. UNPINNED over 16 logical processors (6P+4E). stoke = `batch --seqs 1 --kv f32 -n 64` on p00; llama = `llama-bench -p 0 -n 64 -r 3`, b10037. 4 cycles, thread direction and engine order both rotate. Median per cell.

## The curve

| t | stoke tok/s | llama tok/s | stoke/llama | stoke speedup vs t=1 | llama speedup vs t=1 | llama anchor 08-11 | anchor dev |
|---|---|---|---|---|---|---|---|
| 1 | 3.694 | 5.88 | 0.628x | 1.00x | 1.00x | 4.98 | 18.1% |
| 2 | 7.157 | 10.78 | 0.664x | 1.94x | 1.83x | 9.84 | 9.6% |
| 4 | 13.040 | 16.775 | 0.777x | 3.53x | 2.85x | 16.07 | 4.4% |
| 6 | 16.391 | 18.795 | 0.872x | 4.44x | 3.20x | 17.45 | 7.7% |
| 8 | 16.716 | 18.73 | 0.892x | 4.53x | 3.19x | 17.85 | 4.9% |
| 10 | 16.276 | 19.315 | 0.843x | 4.41x | 3.28x | 17.93 | 7.7% |

## The anchor guard

Six independent llama points, same binary and flags as the 2026-08-11 quiesced curve. A drifted box cannot reproduce six at once; if several deviate in the same direction, read the stoke/llama column (measured minutes apart on the same box) and not the absolutes.

**Bridge cell** -- stoke t=6, n=128: **15.685** against the post-Q6_K ladder value of 14.48, a deviation of **8.3%**. This ties the curve to the P-BAR ladder without re-running it.

## Per-cell spread

| cell | runs | min | max | spread |
|---|---|---|---|---|
| t=1 S | 4 | 2.513 | 3.844 | 52.96% |
| t=1 L | 4 | 5.64 | 6.17 | 9.40% |
| t=2 S | 4 | 6.048 | 7.82 | 29.30% |
| t=2 L | 4 | 10.23 | 11.15 | 8.99% |
| t=4 S | 4 | 12.945 | 13.4 | 3.51% |
| t=4 L | 4 | 16.47 | 17.11 | 3.89% |
| t=6 S | 4 | 16.338 | 16.546 | 1.27% |
| t=6 L | 4 | 18.33 | 19.22 | 4.86% |
| t=8 S | 4 | 16.591 | 17.065 | 2.86% |
| t=8 L | 4 | 18.5 | 18.76 | 1.41% |
| t=10 S | 4 | 16.147 | 16.326 | 1.11% |
| t=10 L | 4 | 19.05 | 19.45 | 2.10% |

A cell whose spread is wide relative to the differences being read is not evidence, however good its median looks.

Bank: `2026-08-17-m45-thread-curve-bank.jsonl`. Raw log: `2026-08-17-m45-thread-curve-raw.txt`.
