# Pricing `vpmulld`: the prediction, committed before the first cell

**2026-08-24, Opus 5, branch `m5-entry`.** Written and committed BEFORE the
ablation arm is built or run. Follows `2026-08-20-m5-attribution-RESULTS.md`,
whose §2 left `vpmulld` as the first named-but-unmeasured candidate, and whose
closing line asks for exactly this number: *"price `vpmulld`'s contribution by
ablation — that costs an afternoon and tells you what the bit-exactness line is
actually worth in tok/s."*

## 1. The experiment

In `vec_dot_q4_k_q8_k_xn_avx2`'s per-sequence inner loop, replace

```c
acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(sub, _mm256_set1_epi32((int)scales8[j])));
```

with the same shape and a cheaper op:

```c
acc = _mm256_add_epi32(acc, _mm256_add_epi32(sub, _mm256_set1_epi32((int)scales8[j])));
```

**The result is deliberately WRONG.** The binary is an instrument, is named
`mullo-ablate`, is never installed as an arm, and is deleted after the run.

**Why `add` and not deletion.** Deleting the multiply entirely would let GCC
dead-code the `scales8` unpack and the `vpbroadcastd`, crediting the ablation
with savings a real change could never collect. Substituting `vpaddd` keeps
`scales8[j]` live, keeps the broadcast, keeps the operand count and keeps the
dependency edge into `acc` — so the delta prices the multiply and nothing else.

**What the number means.** It is an **upper bound** on what removing the integer
scale from the critical path could buy. A real redesign still has to apply the
scale somewhere; llama.cpp defers it to a float FMA at the super-block boundary,
which is cheaper but not free.

## 2. What the disassembly says, and what it does not

`objdump` on the `STOKE_MINS_SIMD=1` arm (the verify gate that decides whether
it becomes the default was still running when this was written), whole function:

| mnemonic | count | note |
|---|---|---|
| `vmovdqu` / `vmovups` | 37 / 20 | incl. the Windows-ABI xmm6–15 save/restore |
| `vpaddd` | 13 | 8 accumulate + reduction |
| `vpmaddwd` | 9 | 8 × ones16 fold + **1 = the mins term now in SIMD** |
| `vpmulld` | **8** | **the scale, once per sub-block per sequence** |
| `vpmaddubsw` | 8 | the nibble × activation product |
| `vpbroadcastd` | 9 | `_mm256_set1_epi32(scales8[j])` |

**Not spill-bound.** The only stack-relative vector traffic inside the loop is
two `wnib` reloads (`0xc0`/`0x100(%rsp)`, twice each); every other `%rsp` move
is the ABI prologue/epilogue saving xmm6–xmm15 once. §2 of the 08-20 memo listed
register pressure as a candidate; on this evidence it is a weak one.

**The port arithmetic, which is the interesting part.** Per sub-block the
per-sequence inner loop issues to the p0/p1 vector-ALU cluster:
`vpmaddubsw` 1 µop + `vpmaddwd` 1 µop + **`vpmulld` 2 µops** + `vpaddd` 1 µop.
So **`vpmulld` is half of the p01 pressure in the hot loop.** And "2.5 µops/cycle
against a 6-wide front end" — the 08-20 memo's reason for killing the µop-count
theory — is *also* what a p01-saturated kernel looks like, because p01 is two
ports wide. That reading was available on 08-20 and was not taken.

⚠ **AND THE ROOFLINE ARGUES AGAINST IT, WHICH IS WHY THIS IS A MEASUREMENT.**
~34 p01 µops per (super-block × sequence) at 2/cycle is ~17 cycles, i.e. ~120
GFLOP/s at ~4 GHz. The kernel was measured at **46.3 GFLOP/s single-threaded** —
~39% of its own port roofline, so p01 has slack and something else binds.
That estimate is soft: 46.3 is an END-TO-END prefill rate including attention
and the norms, not this kernel alone, so the true fraction is higher than 39%.
Soft enough that neither model can be argued to a conclusion — which is the
definition of a measurement worth taking.

## 3. THE PREDICTION

**t=1 prefill, ablation vs the shipping SIMD arm, interleaved with the order
alternating per cycle: I predict 1.15×, and I will accept anything in
1.10×–1.25× as "as predicted".**

Discriminators, committed now:

- **≥ 1.25× ⇒ the kernel IS p01-throughput-bound and the scale is the biggest
  remaining lever.** Then §5's bit-exactness question is really a question about
  the scale, and the candidate in §4 below becomes the priority over any
  float-deferred redesign.
- **≤ 1.08× ⇒ the scale is a red herring.** The remaining ~3× is memory or
  latency, `vpmulld` is not what holds this kernel, and Kabir's decision is not
  about the integer scale at all. **This would be the more useful result**, on
  the same logic that made the 08-20 negative result worth more than its 7%.
- Between 1.08× and 1.10×, or between 1.25× and anything higher: report the
  number, claim neither model.

**Prior discipline: price it at half.** Three kernel arguments in this project
have run 1.5–2× hot (L4 predicted 317→63 ms and measured 106.75; A′ argued for
more than 0.907×; Q6_K argued 1.20× and measured 1.108×). 1.15× is already the
halved figure; the un-halved instinct from "half the p01 µops" was ~1.4×.

## 4. THE CANDIDATE THIS MAKES URGENT, AND IT IS ON THE **NEAR** SIDE OF THE LINE

If the ablation reads high, the obvious next move looks like llama's — defer the
scales to a float FMA — and that is **not** bit-exact. **There is a bit-exact
alternative, and it was not on the 08-20 list.**

`sub` is bounded: `maddubs` gives |lane| ≤ 15·128·2 = 3840 as i16, and the
`ones16` fold gives |sub| ≤ 7680 as i32 — which **fits i16**. So two sub-blocks'
`sub` vectors can be packed to i16 with `vpackssdw` (no saturation at 7680, plus
one `vpermq` to undo AVX2's per-128-bit-lane packing) and multiplied-and-added
by **one `vpmaddwd`** against the two packed scales: ≤ 7680 · 63 · 2 = 967,680,
three orders inside i32.

Per PAIR of sub-blocks that trades **2 `vpmulld` + 2 `vpaddd` (6 p01 µops)** for
**1 `vpmaddwd` + 1 `vpaddd` (2 p01 µops) + 2 shuffle µops on p5** — it moves work
OFF the contended cluster rather than merely reducing it. And it is bit-exact by
exactly the argument that licensed the mins term: every term integer, integer
addition associative, regrouping inside a super-block free, bound derived rather
than assumed (`simd.c:75`'s founding principle).

⚠ **This is a candidate, not a claim, and it is deliberately NOT built yet.**
The order is: price the ceiling by ablation first, then decide whether any of
this is worth building. The 08-20 session's lesson was that the free integer-core
win existed and was worth only ~7%; the same could be true here, and the ablation
is what says so before an afternoon is spent.

## 5. What would make me withdraw this memo

- If the ablation binary is not bit-DIFFERENT from the shipping arm on a real
  model, it did not take effect and every number from it is meaningless.
- If the ablation and the shipping arm agree within the run's own repeat spread,
  the run had no resolution and the answer is "not measurable here", not "1.00×".
- If the arms are not interleaved with the order alternating, the number is
  withdrawn regardless of what it says. A single unpaired pre-flight sample read
  0.957× on 2026-08-20 for a change that wins 6/6 interleaved.
