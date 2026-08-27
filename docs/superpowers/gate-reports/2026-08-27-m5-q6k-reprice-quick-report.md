# The Q6_K `vpmaddubsw` kernel, re-priced on a real prefill

> !! **-Quick, NOT the full rung** -- 2 points.

7.68x on the isolated kernel at N=16. This is what it is worth END TO END.

1 cycles, point-order and arm-order both rotating, 6 threads, affinity **0xFFF (P-cores)**. One binary, `STOKE_Q6K_KERNEL` selecting the arm, so no code-layout confound. Commit `418f88f`, bank key `B1E99988`. Medians over cycles.

| prompt | prefill tok | ref ms | maddubs ms | **speedup** | null (ref2/ref1) | resolved? |
|---|---|---|---|---|---|---|
| `p00` | 23 | 4554.7 | 4708.3 | **0.9674x** | 16.48% | **NO -- inside the null** |
| `p40` | 490 | 20084 | 17013.7 | **1.1805x** | 3.31% | yes |

## The pre-registered predictions

| # | prediction | result |
|---|---|---|
| P1 | speedup RISES with prompt length (16-tile share grows) | **HELD** (p40 1.1805x > p00 0.9674x) |
| P2 | end-to-end prefill speedup is BELOW the 1.71x matvec-portion ceiling | **HELD** -- no cell reaches 1.71x |
| -- | every arm of a cell prefilled the SAME token count | **yes** |

**The null column is the point.** `ref2/ref1` is the same kernel measured twice
as two processes, so it is what this harness cannot resolve for THAT cell. It is
reported per cell and never pooled: a between-process null on this box read 30% at
the smallest cell and 1.0% at the largest, and one pooled floor would bury a real
result under a floor belonging to a different cell. A speedup smaller than its own
cell's null reports NOTHING, however tidy it looks.

A cell above the 1.71x ceiling would mean the measurement is wrong, not the kernel
brilliant: prefill is strictly more than its matvecs, so an end-to-end number cannot
exceed the matvec-portion bound. That bound is ADDENDUM 19's 1.67x RE-DERIVED at the
kernel's MEASURED 83.68 GF/s rather than the 70 GF/s it assumed (0.0107 + 0.0907 =
0.1014 units against a 0.173 baseline); the asymptote, with Q6_K free, is 1.91x.
The band was fixed BEFORE the first run -- re-deriving from better inputs beforehand
is getting it right, changing it afterwards is re-fitting. Predictions are recorded
as written; a falsified one stays on the page with the right measurement beside it.
