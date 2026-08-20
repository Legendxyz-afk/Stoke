# Does vectorising the Q4_K mins term buy anything?

Run `2026-08-20-m5-mins-ab`. A = `STOKE_MINS_SIMD=0` (shipping default), B = `=1`.
PREFILL of a 525-token prompt, ctx 704, kv f32, arms interleaved with the
order alternating per cycle. The two are already BIT-EXACT on the real model
through the batch path, and that comparison is mutation-verified.

| threads | A tok/s | B tok/s | B/A | A GFLOP/s | B GFLOP/s |
|---|---|---|---|---|---|
| 1 | 5.4639 | 5.8081 | **1.063x** | 46.2 | 49.1 |
| 8 | 24.1512 | 26.0538 | **1.079x** | 204.3 | 220.4 |

**t=1 is the headline** -- no pool, no barrier, no dispatch, so the ratio is the
kernel's own. objdump put the scalar spelling at ~40 uops against the vector
dot's ~57; the kernel runs at ~2.5 uops/cycle against a 6-wide front end, so the
uop ratio was an upper bound and never a prediction.

| unit | arm | t | tok/s | GFLOP/s | wall s |
|---|---|---|---|---|---|
| t8-c1-A | A | 8 | 24.6641 | 208.7 | 25.2 |
| t8-c1-B | B | 8 | 25.9717 | 219.7 | 24.1 |
| t8-c2-B | B | 8 | 26.3178 | 222.6 | 24.1 |
| t8-c2-A | A | 8 | 24.289 | 205.5 | 24.1 |
| t8-c3-A | A | 8 | 22.6843 | 191.9 | 25.1 |
| t8-c3-B | B | 8 | 26.136 | 221.1 | 22.1 |
| t8-c4-B | B | 8 | 24.8776 | 210.5 | 23.1 |
| t8-c4-A | A | 8 | 24.0134 | 203.2 | 24.1 |
| t1-c1-A | A | 1 | 5.4689 | 46.3 | 100.2 |
| t1-c1-B | B | 1 | 5.771 | 48.8 | 95.2 |
| t1-c2-B | B | 1 | 5.8453 | 49.5 | 94.2 |
| t1-c2-A | A | 1 | 5.4589 | 46.2 | 99.2 |
