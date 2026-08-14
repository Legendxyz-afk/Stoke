# Q6_K on maddubs — the lever that may be larger than A′

*2026-08-15, written overnight while the P-BAR run measured. **Unimplemented.**
Design and arithmetic only; the sizing step below must run before anyone writes
the kernel.*

## Why look here at all

The M4.5 record attributed the llama gap to the inner loop: *"the 2.2× llama
gap is the inner loop (8 vs 32 products/uop)"*. Rung C's maddubs kernel closed
exactly that — Q4_K now retires **32 products per instruction**, the same as
llama.

**And the gap did not close.** Blocks 1–2 of the overnight run put B/L at
**0.691**, needing 1.23× more. So whatever remains is *not* the Q4_K inner
loop, and the ladder's next scheduled lever (A′, row tiling) had already come
back *slower* than maddubs on a first noisy sample.

The obvious unexamined suspect: **Q6_K still runs the OLD widening path**, 8
products per `vpmulld`, and Q6_K is `ffn_down` — by the M4.5 record the largest
single weight class at **24.9%**. A quarter of the weight traffic never got
rung C at all.

## Why Q6_K was skipped, and why the reason dissolves

`vpmaddubsw` multiplies **unsigned** bytes by **signed** bytes. Q4_K's nibbles
are naturally unsigned 0..15, so it fits. Q6_K's quants are **centred**: the
kernel builds `aux8` as `(ql | qh<<4) - 32`, giving signed −32..+31. That is
why the port was deferred as "too subtle to write unverified".

But the centring is a subtraction, and a subtraction distributes:

    S_b = Σ_j scale_j · Σ_{k∈j} (q6_k − 32)·q8_k
        = Σ_j scale_j · [ Σ_{k∈j} q6_k·q8_k  −  32·Σ_{k∈j} q8_k ]

with `q6_k` the **raw unsigned 0..63** value, never centred. The first term is
exactly what `vpmaddubsw` computes. The second is 32 times the sub-block's
activation sum.

## The part that makes it clean rather than merely possible

**That activation sum is already computed, already stored, and currently
thrown away.**

- Q6_K has **16 sub-blocks of 16 elements**, one i8 scale each (`simd.c`).
- Q8_K's `bsums` has **16 entries, each summing 16 elements** (`kquant.c`).
- The granularities are **identical**, so sub-block `j` corresponds to
  `bsums[j]` with no regrouping.
- And the Q6_K path states in its own comment that it *"never touches the
  activation's bsums"* — the data is free.

So the correction is `32 · bsums[j]`, a 16-term integer reduction per
super-block of exactly the shape Q4_K's min term already uses.

    S_b = Σ_j scale_j · ( maddubs_j − 32·bsums[j] )

**Exact in integers.** No approximation, no new rounding, and therefore
CONTRACT v2 compliant by the same argument the Q4_K port used: any
implementation may reach `S_b` however it likes, then execute the fold
verbatim.

## Bounds, computed not assumed

- `vpmaddubsw` pair, unsigned q6 × signed q8: `2 · 63 · 127 = 16,002` against
  an i16 ceiling of 32,767. **Safe**, with roughly half the headroom spare.
  (Note this is tighter than Q4_K's 3,810 — worth restating if the quant range
  ever changes.)
- The correction term: `32 · |bsums| ≤ 32 · 16 · 127 = 65,024`, i32 territory,
  applied after the i16→i32 widening.
- Q6_K's existing reachable worst case for `S_b` is **133,169,152** (measured
  by `test_c_oracle`), unchanged by this rewrite since the value is identical.

## Size it BEFORE writing it

The Amdahl case is suggestive but **not measured**: if Q6_K is ~25% of weight
bytes running a ~4× slower inner loop, it plausibly dominates what remains —
but "plausibly" is exactly the reasoning this project keeps having to retract.

The instrument already exists and the ruling already points at it: §7.2's
C-SLOW branch prescribes *"one `bench --ops` comparison of per-matvec cost"* to
separate kernel from elsewhere. **Run `bench --ops` split by weight type
first.** If Q6_K is not a large share of matvec time, this proposal is worth
nothing and should be dropped rather than written.

## If it is written

Same verification chain as A′, no shortcuts: `test_simd`'s twin-vs-scalar case
over 1e4 tensors (the scalar must be ported too, or the twin has nothing exact
to agree with), `test_kquant`, batch identity, and byte-compare — **but note
this one WILL move the streams**, because it changes the integer core's
grouping for Q6_K. Unlike A′ and the Q4_K maddubs port, it is a stream-changing
change and needs grading.

⚠ And grading now means the **decided** criterion, not the retired one: zero
BUG-class plus full §6 classification plus laddered ESCALATEs, with the
first-divergence count reported and not bounded
(`memos/2026-08-14-m45-first-divergence-criterion-DECISION-OPUS.md`).
