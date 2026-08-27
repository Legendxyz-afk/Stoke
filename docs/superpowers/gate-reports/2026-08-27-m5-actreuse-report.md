# What are the activation RE-READS costing the batched Q4_K kernel?

Run `2026-08-27-m5-actreuse`, 3 blocks x 24 interleaved A/B cycles per cell, single-threaded,
pinned to one P-core (mask 0x1). Instrument `stoke bench --matvec-batch`.
Prediction and discriminators pre-registered BEFORE the first cell:
`memos/2026-08-27-m5-activation-reuse-PREREGISTERED.md`.

**A** = `matvec_batch` with N distinct activation buffers -- the real case.
**B** = the same call with every activation pointer ALIASED to one buffer:
same loads, same arithmetic, one cache-resident working set.
**C** = the NULL arm: a SECOND set of distinct buffers -- the same computation
as A at different addresses. **A/C is the harness's own floor**, measured rather
than assumed, and A/B only means something to the extent it EXCEEDS it.

Each cell refuses to run unless **B differs from A** (the alias took effect),
**C agrees with A byte for byte** (addresses alone cannot change an answer), and
**the N activations differ from each other** (or both other checks pass
vacuously). A broken cell stops the run; it never reports a ratio.

`act KiB` is the activation tile arm A RE-READS FOR EVERY OUTPUT ROW.
**A/B is an upper bound**: a real redesign still has to read each activation once.

| tensor | n_in x n_out | N | act KiB | A med ms | B med ms | **A/B** | A/C (floor) | A GF/s | B won | C won |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_q | 2560 x 4096 | 1 | 2.9 | 0.585 | 0.578 | **1.007** | 0.962 | 35.85 | 42/72 | 30/72 |
| attn_q | 2560 x 4096 | 2 | 5.7 | 1.027 | 1.001 | **1.037** | 0.930 | 40.84 | 38/72 | 31/72 |
| attn_q | 2560 x 4096 | 4 | 11.4 | 1.524 | 1.461 | **1.041** | 1.046 | 55.05 | 46/72 | 40/72 |
| attn_q | 2560 x 4096 | 8 | 22.8 | 2.396 | 2.739 | **0.949** | 1.009 | 70.02 | 29/72 | 30/72 |
| attn_q | 2560 x 4096 | 16 | 45.6 | 5.097 | 4.737 | **1.088** | 0.988 | 65.84 | 52/72 | 32/72 |
| ffn_gate | 2560 x 9728 | 1 | 2.9 | 1.346 | 1.424 | **0.940** | 0.967 | 37.01 | 34/72 | 29/72 |
| ffn_gate | 2560 x 9728 | 2 | 5.7 | 2.497 | 2.533 | **0.986** | 0.984 | 39.90 | 32/72 | 29/72 |
| ffn_gate | 2560 x 9728 | 4 | 11.4 | 3.483 | 3.468 | **0.995** | 0.998 | 57.20 | 38/72 | 34/72 |
| ffn_gate | 2560 x 9728 | 8 | 22.8 | 6.268 | 6.061 | **1.014** | 1.023 | 63.57 | 40/72 | 46/72 |
| ffn_gate | 2560 x 9728 | 16 | 45.6 | 11.353 | 10.916 | **1.052** | 0.992 | 70.19 | 52/72 | 26/72 |
| attn_out | 4096 x 2560 | 1 | 4.6 | 1.073 | 1.058 | **1.007** | 1.006 | 19.54 | 38/72 | 42/72 |
| attn_out | 4096 x 2560 | 2 | 9.1 | 1.754 | 1.789 | **1.004** | 0.992 | 23.91 | 31/72 | 34/72 |
| attn_out | 4096 x 2560 | 4 | 18.2 | 2.648 | 2.738 | **0.988** | 0.967 | 31.68 | 31/72 | 34/72 |
| attn_out | 4096 x 2560 | 8 | 36.5 | 4.707 | 4.714 | **1.019** | 0.985 | 35.64 | 41/72 | 35/72 |
| attn_out | 4096 x 2560 | 16 | 73.0 | 9.083 | 8.219 | **1.111** | 1.025 | 36.94 | 55/72 | 35/72 |
| ffn_down | 9728 x 2560 | 1 | 10.8 | 2.444 | 2.451 | **0.990** | 0.968 | 20.38 | 41/72 | 29/72 |
| ffn_down | 9728 x 2560 | 2 | 21.7 | 10.327 | 10.108 | **1.002** | 1.005 | 9.65 | 34/72 | 34/72 |
| ffn_down | 9728 x 2560 | 4 | 43.3 | 18.703 | 18.849 | **0.992** | 1.000 | 10.65 | 31/72 | 33/72 |
| ffn_down | 9728 x 2560 | 8 | 86.7 | 37.015 | 37.506 | **1.002** | 1.003 | 10.76 | 38/72 | 37/72 |
| ffn_down | 9728 x 2560 | 16 | 173.4 | 73.587 | 73.911 | **0.996** | 0.998 | 10.83 | 33/72 | 34/72 |

**Measured harness floor: the null arm deviates from A by up to 7.0% across every cell.**
An A/B inside that is not an effect. A pairing count near half the cycles is not
one either, whatever the median says.

## The pre-registered predictions

| # | cell | predicted | measured | result |
|---|---|---|---|---|
| P1 | `attn_q` N=16 | A/B <= 1.15 | 1.088 | **HELD** |
| P2 | `ffn_down` N=16 | A/B <= 1.25 | 0.996 | **HELD** |
| P3 | N=1 control | 1.00 +- 0.03 | see table | **FALSIFIED** (ffn_gate N=1 A/B 0.940) |

## Verdict against the discriminators fixed before the run

Largest A/B over every batched cell: **1.111** (attn_out N=16).

| if | then | fired |
|---|---|---|
| A/B >= 2.0 anywhere | the memory hypothesis IS the gap | no |
| A/B in [1.25, 2.0) | real but partial | no |
| A/B <= 1.15 everywhere | the hypothesis is DEAD | **YES** |

**=> the memory hypothesis is DEAD -- no named lever remains.**

> !! **WITHDRAWAL CONDITION 3 FIRED**: the N=1 control is not 1.00 +- 0.03
> (ffn_gate N=1 A/B 0.940). `vec.c` routes `n_seq == 1` to the solo worker, so
> there is nothing to alias and this deviation is HARNESS BIAS, not physics.
> Read it against the measured floor above before reading any other cell:
> if it sits inside +-7.0%, the control is reporting the harness and not the code.

## POST-HOC OBSERVATION -- NOT PRE-REGISTERED, NOT GRADED, AND NOT YET A FINDING

The `A GF/s` column is the thing the instrument gap was hiding, and it shows
something the ablation was not looking for. Read the N=16 rows against `n_in`:

| tensor | n_in | n_out | super-blocks per row | FLOP | weight bytes | A GF/s @ N=16 |
|---|---|---|---|---|---|---|
| attn_q | 2560 | 4096 | 10 | 336 M | 5.9 MB | **65.84** |
| ffn_gate | 2560 | 9728 | 10 | 797 M | 14.0 MB | **70.19** |
| attn_out | 4096 | 2560 | 16 | 336 M | 5.9 MB | **36.94** |
| ffn_down | 9728 | 2560 | 38 | 797 M | 14.0 MB | **10.83** |

`attn_q` and `attn_out` are TRANSPOSES of each other: identical FLOP, identical
weight bytes, identical (row x super-block x sequence) body count. So are
`ffn_gate` and `ffn_down`. Within each pair the only thing that differs is the
LOOP SHAPE -- how many super-blocks sit inside one row -- and the one with more
of them is much slower.

**The activation side is already RULED OUT as the cause by arm B**, which collapses
the activation working set to 1/N and does not move any of these cells. Whatever
this is, it is not activation locality, not the activation TLB footprint, and not
the number of activation streams.

!! **AND ITS MAGNITUDE IS NOT PINNED DOWN.** An earlier configuration of this same
instrument (3 tensors, 2 arms) measured `ffn_down` at N=16 at **19.76 GF/s**;
this one (4 tensors, 3 arms) reads **10.83**. The A/B RATIO is robust because the arms are
interleaved inside the timing loop -- the ABSOLUTE rate is not, and it moved by
1.8x on a harness change alone. **The direction reproduces in both runs; the size
does not, and nothing should be built on the size until it does.**

Next experiment, and it is cheap: sweep `n_out` at fixed `n_in` to separate
weight-cache residency from per-row structure, and re-measure with one tensor per
process so no earlier tensor can be sitting in L3.

## What this measures, and what it does not

- It is an **upper bound**, not a projected speedup. Perfect reuse is unbuildable;
  a real redesign blocks the row loop over a tile that still has to be read once.
- **Single-threaded.** t=1 is the headline because the 3.22x gap was measured at
  every thread count and t=1 has no pool, no barrier and no dispatch. At t=8 the
  same tile is contended by 8 cores against a shared L3, which can only make the
  re-reads matter MORE -- so a small ratio here is the conservative reading.
- **A/B is measured, the CAUSE is not.** A ratio near 1 says the re-reads are not
  costing time; it does not say which cache level served them.
