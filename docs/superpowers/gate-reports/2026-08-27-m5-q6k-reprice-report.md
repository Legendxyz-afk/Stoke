# The Q6_K `vpmaddubsw` kernel, re-priced on a real prefill

7.68x on the isolated kernel at N=16. This is what it is worth END TO END.

3 cycles, point-order and arm-order both rotating, 6 threads, affinity **0xFFF (P-cores)**. One binary, `STOKE_Q6K_KERNEL` selecting the arm, so no code-layout confound. Commit `418f88f`, bank key `B1E99988`. Medians over cycles.

| prompt | prefill tok | ref ms | maddubs ms | **speedup** | null (ref2/ref1) | ref spread | mad spread | floor | resolved? |
|---|---|---|---|---|---|---|---|---|---|
| `p00` | 23 | 4743 | 3541.8 | **1.3391x** | 0.46% | 1.23% | 32.76% | 32.76% | **NO -- inside the floor** |
| `p40` | 490 | 20146.7 | 16648.3 | **1.2101x** | 0.28% | 3.84% | 1.7% | 3.84% | yes |
| `long` | 1436 | 64392.3 | 53461.7 | **1.2045x** | 1.42% | 2.87% | 2.29% | 2.87% | yes |

## The pre-registered predictions

| # | prediction | result |
|---|---|---|
| P1 | speedup RISES with prompt length (16-tile share grows) | **NOT SUPPORTED -- FLAT.** `p40` 1.2101x vs `long` 1.2045x is -0.46%, INSIDE the 3.84% floor: the benefit SATURATES, it does not rise -- graded on the 2 RESOLVED cells only; `p00` cannot anchor a trend |
| P2 | end-to-end prefill speedup is BELOW the 1.71x matvec-portion ceiling | **HELD** -- no cell reaches 1.71x |
| -- | every arm of a cell prefilled the SAME token count | **yes** |

**The floor column is the point, and THE NULL ALONE WAS NOT ENOUGH.** `ref2/ref1`
is the same kernel measured twice as two processes -- but it is built from the REF
arm, so it bounds REF's noise and says nothing about MAD's. On this rung's first
full run that gap produced a **1.3391x at `p00` that was not real**: the null read
0.46% while a 12-run probe put BOTH arms' true spread at 16-18%. Three samples of
`ref` agreed by luck and `mad`'s median-of-3 landed low twice. **A null computed
from n=3 can report a tiny floor purely by chance.**

So the floor is now `max(null, ref spread, mad spread)`: an arm that cannot
reproduce ITSELF cannot measure anything, and that is now visible per arm instead of
hidden inside a ratio. It is still never pooled across cells -- a between-process
null on this box read 30% at the smallest cell and 1.0% at the largest.

!! Short prompts are STRUCTURALLY unresolvable here. A fixed ~500-700 ms of startup
jitter is ~15% of `p00`'s 4.7 s prefill and under 4% of `p40`'s 20 s, so the
smallest cell cannot see a kernel effect at all. More cycles do not fix that; only a
longer prompt does.

A cell above the 1.71x ceiling would mean the measurement is wrong, not the kernel
brilliant: prefill is strictly more than its matvecs, so an end-to-end number cannot
exceed the matvec-portion bound. That bound is ADDENDUM 19's 1.67x RE-DERIVED at the
kernel's MEASURED 83.68 GF/s rather than the 70 GF/s it assumed (0.0107 + 0.0907 =
0.1014 units against a 0.173 baseline); the asymptote, with Q6_K free, is 1.91x.
The band was fixed BEFORE the first run -- re-deriving from better inputs beforehand
is getting it right, changing it afterwards is re-fitting. Predictions are recorded
as written; a falsified one stays on the page with the right measurement beside it.
