# Does vectorising the Q4_K mins term buy anything?

Run `PREFLIGHT-m5-mins-ab`. A = `STOKE_MINS_SIMD=0` (shipping default), B = `=1`.
PREFILL of a 525-token prompt, ctx 704, kv f32, arms interleaved with the
order alternating per cycle. The two are already BIT-EXACT on the real model
through the batch path, and that comparison is mutation-verified.

| threads | A tok/s | B tok/s | B/A | A GFLOP/s | B GFLOP/s |
|---|---|---|---|---|---|
| 8 | 25.4680 | 24.3821 | **0.957x** | 215.5 | 206.3 |

**t=1 is the headline** -- no pool, no barrier, no dispatch, so the ratio is the
kernel's own. objdump put the scalar spelling at ~40 uops against the vector
dot's ~57; the kernel runs at ~2.5 uops/cycle against a 6-wide front end, so the
uop ratio was an upper bound and never a prediction.

| unit | arm | t | tok/s | GFLOP/s | wall s |
|---|---|---|---|---|---|
| t8-c1-A | A | 8 | 25.468 | 215.5 | 24.1 |
| t8-c1-B | B | 8 | 24.3821 | 206.3 | 23.1 |
