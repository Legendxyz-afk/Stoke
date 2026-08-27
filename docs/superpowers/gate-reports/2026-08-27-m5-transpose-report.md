# Is the `n_in` collapse real, or was it the harness?

Run `2026-08-27-m5-transpose`, 4 blocks x 24 cycles, **one tensor per PROCESS** with the
activation scratch sized to that tensor, pinned to one P-core (mask 0x1),
order reversed on even blocks.

## First: one row of the 2026-08-27 table is WITHDRAWN

`blk.0.ffn_down.weight` is **Q6_K**, not Q4_K -- in a Q4_K_M model it is upgraded.
210 bytes per super-block against 144, **20.4 MB against 14.0**, and a different
kernel (`vec_dot_q6_k_q8_k_xn_avx2`). The `ffn_gate` vs `ffn_down` 6.5x was
never like-for-like and **is withdrawn**. The instrument now prints the quant type
and weight bytes on every row: a fixture drawn BY NAME from a real model can
differ in the property a comparison rests on, and nothing in that table said so.

## What survives: one exact transpose pair

`attn_q` (2560 x 4096) and `attn_output` (4096 x 2560) are both Q4_K, both
5,898,240 bytes, same FLOP, and the same 40,960 (row x super-block) unpacks.
Only the loop shape differs: 10 super-blocks inside a row against 16.

| tensor | type | n_in x n_out | N | A GF/s | A/B (alias) | A/C (floor) |
|---|---|---|---|---|---|---|
| attn_q | Q4_K | 2560 x 4096 | 1 | **38.32** | 1.001 | 1.002 |
| attn_q | Q4_K | 2560 x 4096 | 2 | **45.02** | 0.986 | 0.996 |
| attn_q | Q4_K | 2560 x 4096 | 4 | **61.42** | 0.984 | 1.004 |
| attn_q | Q4_K | 2560 x 4096 | 8 | **71.59** | 1.007 | 0.992 |
| attn_q | Q4_K | 2560 x 4096 | 16 | **71.75** | 1.084 | 1.003 |
| attn_out | Q4_K | 4096 x 2560 | 1 | **37.45** | 1.000 | 1.000 |
| attn_out | Q4_K | 4096 x 2560 | 2 | **46.44** | 0.995 | 0.998 |
| attn_out | Q4_K | 4096 x 2560 | 4 | **58.20** | 0.994 | 1.002 |
| attn_out | Q4_K | 4096 x 2560 | 8 | **70.79** | 1.012 | 1.005 |
| attn_out | Q4_K | 4096 x 2560 | 16 | **67.88** | 1.072 | 0.994 |
| ffn_gate | Q4_K | 2560 x 9728 | 1 | **32.38** | 1.001 | 0.979 |
| ffn_gate | Q4_K | 2560 x 9728 | 2 | **39.68** | 0.998 | 1.025 |
| ffn_gate | Q4_K | 2560 x 9728 | 4 | **55.82** | 1.005 | 1.013 |
| ffn_gate | Q4_K | 2560 x 9728 | 8 | **62.18** | 1.019 | 1.023 |
| ffn_gate | Q4_K | 2560 x 9728 | 16 | **69.69** | 1.055 | 0.998 |
| ffn_down | Q6_K | 9728 x 2560 | 1 | **33.42** | 0.994 | 1.003 |
| ffn_down | Q6_K | 9728 x 2560 | 2 | **17.64** | 0.998 | 1.001 |
| ffn_down | Q6_K | 9728 x 2560 | 4 | **19.27** | 0.994 | 1.011 |
| ffn_down | Q6_K | 9728 x 2560 | 8 | **20.44** | 1.004 | 0.971 |
| ffn_down | Q6_K | 9728 x 2560 | 16 | **10.89** | 0.997 | 1.007 |

## The pre-registered outcomes

Transpose ratio at N=16, **isolated**: `attn_q` 71.75 / `attn_out` 67.88 = **1.057x**.
In the contaminated four-tensor run it was **1.78x** (65.84 / 36.94).

| outcome | band | fired |
|---|---|---|
| H-ARTIFACT: contamination, observation dies | < 1.30x | **YES** |
| BETWEEN: partly both, needs an n_out sweep | 1.30-1.60x | no |
| H-REAL: the loop shape is a real lever | >= 1.60x | no |

**=> H-ARTIFACT -- the four-tensor table's rates were contaminated and the observation dies with them.**

## AND THE ISOLATED TABLE SURFACES A REAL ONE: THE BATCHED Q6_K KERNEL GOES BACKWARDS

Every Q4_K tensor CLIMBS with N -- 38 -> 70 GF/s -- and the three of them land
within 5% of each other at N=16 regardless of shape. `ffn_down` (Q6_K) does the
opposite:

| N | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| `ffn_down` Q6_K GF/s | 33.42 | 17.64 | 19.27 | 20.44 | 10.89 |
| `attn_q` Q4_K GF/s | 38.32 | 45.02 | 61.42 | 71.59 | 71.75 |

**The SOLO Q6_K kernel is fine** (N=1: 33.42 against Q4_K's 38.32). It is the BATCHED one
that collapses, and batching makes it ~2x WORSE where it makes Q4_K ~2x better.
This is isolated, like-for-like, and reproduces across all 4 blocks.

**The cause is legible in src/kernels/simd.c.** Per sequence per super-block:

- `vec_dot_q4_k_q8_k_xn_avx2` runs 8 iterations of `vpmaddubsw` + `vpmaddwd`,
  **32 int8 products per instruction**.
- `vec_dot_q6_k_q8_k_xn_avx2` runs 32 iterations of `kq_load_i8x8_to_i32` +
  `kq_mac8` -- it WIDENS i8 to i32 and does **8 products per operation**.

Same 256 elements, ~4x the uops, and the widened `aux8` is re-read per sequence
on top of the activation. The per-sequence body is the part batching multiplies
by N, which is why the gap only appears at N>1.

**The prize, and it is arithmetic rather than a hope.** `ffn_down` is Q6_K in
**18 of this model's 36 layers** (Q4_K_M upgrades half of them) and is 24.7% of
the per-layer matvec MACs. Holding everything else at ~70 GF/s, moving those
0.897 GFLOP/token from 10.9 to 70 takes the matvec portion of prefill from
0.173 to 0.104 arbitrary units = **~1.67x**. Against the ~2.75x the gate [2] bars
need, that is the largest live lever this project has, and the first one in weeks
that is a measured defect rather than a hypothesis.

**A bit-exact fix exists on paper.** `vpmaddubsw` takes its first operand
UNSIGNED, so keep the 6-bit weight as its raw [0,63] value and carry the -32
offset as a separate term -- exactly the shape Q4_K already uses for its `mins`:
`SUM (w-32).a.s == SUM w.a.s - 32.SUM a.s`, and `SUM a` is what Q8_K's `bsums`
already stores. Overflow is inside the bounds: 63x127x2 = 16,002 against i16's
32,767, then x127 x2 = 4,064,508 against i32. A 32-byte `vpmaddubsw` spans TWO of
Q6_K's 16-element scale groups, which a following `vpmaddwd` against
`[s_j x8, s_j+1 x8]` applies exactly, since `(a+b).s == a.s + b.s` in integers --
the same argument `simd.c` already makes for the Q4_K mins term.

**NOT ATTEMPTED HERE.** It is a shipping kernel with a bit-exactness acceptance
criterion and it needs its own mutation-backed gate. This report prices it.

The `A/B` and `A/C` columns are carried through unchanged: the activation-reuse
ablation's conclusion does not depend on any of this, because it is a RATIO
between arms measured inside one timing loop, and that is exactly the quantity
the contamination above could not touch.
