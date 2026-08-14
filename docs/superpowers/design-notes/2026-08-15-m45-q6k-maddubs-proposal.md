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

## How big, from exact figures

The share is no longer a recollection. Read from **GGUF metadata only** (no
tensor data touched, so this costs nothing and cannot be wrong about the file):

| type | bytes | share | tensors |
|---|---|---|---|
| Q4_K | 1.644 GiB | 70.85% | 216 |
| Q6_K | **0.676 GiB** | **29.12%** | 37 |
| F32 | 0.001 GiB | 0.03% | 145 |

But **bytes are the wrong unit** — the inner loop costs per *element*, and the
two formats pack differently (144 B per 256 elements against 210 B). Converted:

    Q4_K  3.138e9 elements  78.0%
    Q6_K  0.885e9 elements  22.0%

So Q6_K is 22% of the work by element count, and it runs the slow path. What
that costs depends on the throughput ratio between the two inner loops, which
is **assumed, not measured** — so here is the sensitivity rather than a single
number:

| Q4_K inner loop faster by | Q6_K's share of inner-loop time | gain from porting it |
|---|---|---|
| 4× (32 vs 8 products/uop) | 53% | **1.66×** |
| 3× | 46% | **1.44×** |
| 2× | 36% | **1.22×** |

**P-BAR needs 1.23×.** Even the pessimistic end of that range lands on it, and
the middle clears it comfortably. That is why this outranks A′ as the next
lever: A′ reorganises how rows are fed, while this fixes a quarter of the work
still running at a quarter of the speed.

⚠ The ratio is the load-bearing assumption and it is an *upper* bound —
products-per-instruction only translates to time if the loop is compute-bound,
and this box's Q4_K_M matvec tops out at 44.8 GB/s, so bandwidth may bind
first. That is exactly what the sizing run below settles.

## ▶▶ SIZED, 2026-08-15 04:1x — AND THE ANSWER IS SOBERER THAN THE ESTIMATE

The sizing run below was attempted and **`bench --ops` crashed with
STATUS_HEAP_CORRUPTION**. Bisecting the archived binaries put the fault back to
the m3 gate archive: the harness sized its activation scratch with Q8_0
constants while a K-quant matvec quantizes to Q8_K, overflowing by 200 bytes
per call at n_in=2560. **The instrument the ruling prescribes had been broken
for two milestones**, unnoticed because `bench --ops` is not in the test matrix.
Fixed in `src/bench.c` (size for the larger of the two formats).

With the instrument working, measured on the Q4_K_M model:

| projection | type | GB/s | %ref |
|---|---|---|---|
| attn_q | Q4_K | 9.62 | 41.5% |
| attn_k | Q4_K | 9.71 | 41.9% |
| attn_out | Q4_K | 9.92 | 42.8% |
| ffn_gate | Q4_K | 9.36 | 40.4% |
| ffn_up | Q4_K | 7.57 | 32.6% |
| **attn_v** | **Q6_K** | **7.24** | 31.2% |
| **ffn_down** | **Q6_K** | **7.06** | 30.5% |
| **LOGITS** | **Q6_K** | **6.78** | 29.2% |

GB/s is the wrong unit again — Q6_K carries more bits per element. Converted at
0.5625 and 0.8203 B/element:

    Q4_K  9.24 GB/s  ->  16.42 G elements/s
    Q6_K  7.03 GB/s  ->   8.57 G elements/s
    measured inner-loop ratio: 1.92x

**Not 4×. 1.92×.** Feeding that back through the same table:

    Q6_K share of inner-loop time:  35%   (estimated 53%)
    gain from porting Q6_K:        1.20x  (estimated 1.66x)
    P-BAR needs:                   1.23x

⇒ **Q6_K-on-maddubs lands just SHORT of P-BAR on its own.** It is still the
largest single lever identified, and combined with any real A′ gain it would
clear the bar — but the earlier 1.66× was optimism, and the reason is visible
in the %ref column: every projection runs at **29–43% of the reference
stream**, so these matvecs are partly memory-bound and instruction throughput
does not convert one-for-one into time.

⚠ Note `ffn_up` (Q4_K, 7.57) sits with the Q6_K group despite being the same
shape as `ffn_gate` (9.36). Unexplained; one run, not investigated. It drags
the Q4_K mean down and therefore makes the 1.92× ratio, if anything, an
UNDER-estimate of the gap — so treat 1.20× as a floor rather than a point
estimate.

## Size it BEFORE writing it

The Amdahl case above is arithmetic on top of one assumed ratio, and
"plausibly" is exactly the reasoning this project keeps having to retract.

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
