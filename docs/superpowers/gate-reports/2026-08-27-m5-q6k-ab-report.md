# The batched Q6_K kernel: reference vs `vpmaddubsw`

Run `2026-08-27-m5-q6k-ab`, 4 blocks x 24 cycles, tensor `ffn_down` (the model's Q6_K
matvec), one tensor per process, pinned to one P-core (mask 0x1), arm order
reversed on even blocks.

**A** = `vec_dot_q6_k_q8_k_xn_avx2` -- widens i8 to i32, 8 products per operation,
32 iterations per sequence per super-block.
**B** = `vec_dot_q6_k_q8_k_xn_maddubs_avx2` -- 32 int8 products per `vpmaddubsw`,
8 iterations, with the -32 offset carried on Q8_K's `bsums` exactly as Q4_K
already carries its `mins`.

**Both arms are the SAME BINARY** -- `STOKE_Q6K_KERNEL` selects at runtime, so this
A/B carries none of the code-layout confound that forced the 2026-08-24 `vpmulld`
arms into equal-length build directories. An unrecognised value is REFUSED, and
the runner checks that before measuring anything.

| N | A ref GF/s | B maddubs GF/s | **B/A** | ref2 (null) GF/s | ref2/A |
|---|---|---|---|---|---|
| 1 (CONTROL -- same code path) | 36.62 | 31.56 | **0.86x** | 25.63 | 0.700 |
| 2 | 17.17 | 42.11 | **2.45x** | 17.59 | 1.024 |
| 4 | 19.13 | 56.80 | **2.97x** | 18.96 | 0.991 |
| 8 | 18.66 | 75.36 | **4.04x** | 18.02 | 0.966 |
| 16 | 10.90 | 83.68 | **7.68x** | 10.79 | 0.990 |

## The pre-registered predictions

| # | prediction | measured | result |
|---|---|---|---|
| P1 | the N=1 CONTROL is within +-7% (the null-arm floor measured 2026-08-27) | 0.862x | **FALSIFIED** |
| P1b | the BETWEEN-PROCESS null at the SAME CELL brackets the control | N=1 null 30.0% | **HELD** |
| P2 | B/A at N=16 exceeds 2.0x | 7.68x | **HELD** |
| P3 | B climbs monotonically with N where A does not | monotone | **HELD** |

!! **P1 IS FALSIFIED AND THE BAND WAS MINE AND WRONG.** 7% is the floor of an
A/C null arm taken INSIDE one timing loop; this comparison is BETWEEN PROCESSES,
so process start, page placement and thermal state at launch are all in it and
none of them are in the 7%. **Applying an in-process denominator to a
between-process ratio is the error [[derived-quantity-is-a-claim]] names**, and the
prediction is left standing as falsified rather than re-fitted after the fact.

Arm `ref2` measures the floor that should have been used: the SAME kernel and
the SAME env value in a DIFFERENT process. **And it must be read PER CELL, not as
a max** -- N=1 is the FIRST cell each process measures and absorbs model load,
tensor warm and page placement, while N>=2 runs on a settled process. One number
for both would mix two populations and understate the cell that matters:

- **N=1 null 30.0%** -- and the N=1 control's 0.862x sits INSIDE it.
- **N=16 null 1.0%** -- against which B/A's 7.68x is not close to a noise
  question at all.

**N=1 is a free control and it costs nothing to carry**: `vec.c` routes
`n_seq == 1` to the SOLO worker, so neither arm's kernel is reached there and the
two arms execute IDENTICAL CODE. Whatever it reports is this harness's noise under
the exact conditions of the cells that matter, and a speedup at N>=2 means
something only to the extent it exceeds it.

## Correctness

**Not measured here, and not assumed either.** `tests/test_simd.c` asserts the twin
BIT-IDENTICAL to both the reference twin AND the solo kernel, per sequence, over 600
iterations spanning n_seq 1/2/3/8/15/16 and n 256/512/768/2560/9728 -- with an
anti-vacuity check that the sequences actually differ from each other, because a
kernel that ignored `s` and returned sequence 0's answer would otherwise pass
every assertion in it.

The exactness is by CONSTRUCTION, not by testing: `SUM (u-32).a.s == SUM u.a.s -
32.SUM a.s` is exact in integers, `SUM a` per scale group is what `bsums` already
stores, and every intermediate is inside its type (63x127x2 = 16,002 against i16;
x127 x2 = 4,064,508 and 260,128,512 against i32). See `simd.h`.

**The default is still `ref`.** An end-to-end stream gate against the ARCHIVES,
with a mutant that must differ, is what has to pass before any flip -- the
`STOKE_MINS_SIMD` sequence, not a shortcut around it.
