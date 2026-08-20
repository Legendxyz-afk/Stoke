# Does vectorising the Q4_K mins term buy anything?

Run `2026-08-20-m5-mins-ab-v2`. A = `STOKE_MINS_SIMD=0` (shipping default), B = `=1`.
PREFILL of a 525-token prompt, ctx 704, kv f32, arms interleaved with the
order alternating per cycle. The two are already BIT-EXACT on the real model
through the batch path, and that comparison is mutation-verified.

| threads | A tok/s | B tok/s | B/A | A GFLOP/s | B GFLOP/s |
|---|---|---|---|---|---|
| 1 | 5.2372 | 5.7574 | **1.099x** | 44.3 | 48.7 |
| 8 | 24.7158 | 26.4741 | **1.071x** | 209.1 | 224.0 |

**t=1 is the headline** -- no pool, no barrier, no dispatch, so the ratio is the
kernel's own. objdump put the scalar spelling at ~40 uops against the vector
dot's ~57; the kernel runs at ~2.5 uops/cycle against a 6-wide front end, so the
uop ratio was an upper bound and never a prediction.

| unit | arm | t | tok/s | GFLOP/s | wall s |
|---|---|---|---|---|---|
| t8-c1-A | A | 8 | 25.7607 | 217.9 | 24.1 |
| t8-c1-B | B | 8 | 26.4733 | 224 | 21.1 |
| t8-c2-B | B | 8 | 26.4749 | 224 | 21 |
| t8-c2-A | A | 8 | 23.5501 | 199.2 | 24.1 |
| t8-c3-A | A | 8 | 24.7992 | 209.8 | 23.1 |
| t8-c3-B | B | 8 | 25.0672 | 212.1 | 23.1 |
| t8-c4-B | B | 8 | 26.4825 | 224 | 24.1 |
| t8-c4-A | A | 8 | 24.6323 | 208.4 | 24.1 |
| t1-c1-A | A | 1 | 5.1628 | 43.7 | 105.2 |
| t1-c1-B | B | 1 | 5.7139 | 48.3 | 96.2 |
| t1-c2-B | B | 1 | 5.8009 | 49.1 | 95.1 |
| t1-c2-A | A | 1 | 5.3117 | 44.9 | 102.1 |
