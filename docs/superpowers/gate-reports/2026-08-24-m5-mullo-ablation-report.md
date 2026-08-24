# What is `vpmulld` costing the Q4_K batched kernel?

Run `2026-08-24-m5-mullo-ablation`. **A = the shipping SIMD kernel. B = the same kernel with the
integer scale multiply replaced by an add -- B COMPUTES THE WRONG ANSWER and
exists only to be timed.** Prediction pre-registered before the first cell:
`memos/2026-08-24-m5-mullo-ablation-PREREGISTERED.md` (1.15x, accept 1.10-1.25x;
>=1.25x => p01-throughput-bound, <=1.08x => the scale is a red herring).

PREFILL of a 525-token prompt, ctx 704, kv f32, arms interleaved with the order
alternating per cycle. B/A is an UPPER BOUND on what removing the integer scale
from the critical path could buy: a real redesign still has to apply the scale.

| threads | A tok/s | B tok/s | B/A | A GFLOP/s | B GFLOP/s | pairings B won |
|---|---|---|---|---|---|---|
| 1 | 6.0950 | 6.8328 | **1.121x** | 51.6 | 57.8 | 4/4 |
| 8 | 30.4322 | 31.9998 | **1.052x** | 257.5 | 270.7 | 4/4 |

| unit | arm | t | tok/s | GFLOP/s | wall s |
|---|---|---|---|---|---|
| t8-c1-A | A | 8 | 30.4457 | 257.6 | 20.1 |
| t8-c1-B | B | 8 | 31.6088 | 267.4 | 20 |
| t8-c2-B | B | 8 | 31.6639 | 267.9 | 20 |
| t8-c2-A | A | 8 | 30.7217 | 259.9 | 19 |
| t8-c3-A | A | 8 | 30.3626 | 256.9 | 20 |
| t8-c3-B | B | 8 | 32.3356 | 273.6 | 18 |
| t8-c4-B | B | 8 | 33.6923 | 285 | 19.1 |
| t8-c4-A | A | 8 | 30.4187 | 257.3 | 20 |
| t1-c1-A | A | 1 | 6.208 | 52.5 | 90.1 |
| t1-c1-B | B | 1 | 6.8625 | 58.1 | 82.1 |
| t1-c2-B | B | 1 | 6.8225 | 57.7 | 82.1 |
| t1-c2-A | A | 1 | 5.982 | 50.6 | 92.1 |
| t1-c3-A | A | 1 | 6.2497 | 52.9 | 88.1 |
| t1-c3-B | B | 1 | 6.8432 | 57.9 | 80.1 |
| t1-c4-B | B | 1 | 6.7832 | 57.4 | 81.1 |
| t1-c4-A | A | 1 | 5.912 | 50 | 92.1 |
