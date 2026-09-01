/* src/kernels/simd.h -- K3: AVX2 + FMA + F16C + AVX-VNNI twins of the
 * kernels.h scalar kernels, plus the runtime CPU probe that decides whether
 * they may be called.
 *
 * WHAT THE M3 GATE DEMANDS, and why it shapes this interface more than
 * performance does. Gate [1]: "scalar<->SIMD exact-diff (integer paths) /
 * <=1e-5 rel (f32 paths) on 1e4 random tensors per kernel". Bit-exactness on
 * the integer paths is not an aspiration here, it is the acceptance
 * criterion, and it constrains the implementation in one specific way that
 * is easy to get wrong -- see the note on vec_dot_q8_0_q8_0_avx2 below.
 *
 * THE TWINS ARE DECLARED SEPARATELY AND CALLABLE DIRECTLY, ON PURPOSE. The
 * obvious design is a single dispatching entry point that quietly picks SIMD
 * or scalar. That design cannot be tested: on a box without AVX-VNNI the
 * exact-diff harness would compare scalar against scalar, pass with a full
 * green bar, and prove nothing at all. Exposing the twin lets the test call
 * it directly and assert separately that this box really has the feature --
 * so a vacuous pass is impossible rather than merely unlikely. Same family
 * as the %ref denominator and the max/med = 1.33 blindness: a check is most
 * dangerous where it is silently trivial.
 *
 * CALLERS MUST GUARD. Calling a twin on a CPU without the feature is
 * undefined (SIGILL in practice), so every call site is
 * `if (kernels_cpu_has_avx_vnni()) twin(...) else scalar(...)`. The scalar
 * kernels in kernels.h remain the reference implementation and are never
 * removed -- they are what the twins are DEFINED as agreeing with.
 */
#ifndef STOKE_KERNELS_SIMD_H
#define STOKE_KERNELS_SIMD_H

#include "common.h"

#include <stdbool.h>

/* True iff this CPU supports AVX-VNNI (VEX-encoded, 256-bit): CPUID leaf 7
 * subleaf 1, EAX bit 4. Probed once and cached; safe to call in a hot loop.
 *
 * This is the VEX form (`vpdpbusd` under `-mavxvnni`), NOT AVX-512-VNNI.
 * The distinction is not academic on this box: K0's P-C probe measured
 * AVX-VNNI = 1 and AVX-VNNI-INT8 = 0, which is why the dpbusd sign trick is
 * mandatory rather than optional -- `_mm256_dpbssd_epi32` would do signed x
 * signed directly and does not exist here.
 *
 * Returns false on non-x86-64 builds, where the twins are not compiled. */
bool kernels_cpu_has_avx_vnni(void);

/* AVX2+VNNI twin of vec_dot_q8_0_q8_0. BIT-IDENTICAL to the scalar kernel
 * for every input, which is asserted over 1e4 random tensors in
 * tests/test_simd.c. Fatal on the same conditions as the scalar version (n
 * not a multiple of 32, inf/nan block scale).
 *
 * ▶▶ HOW BIT-EXACTNESS IS ACHIEVED, AND WHY COPYING ggml HERE WOULD BREAK
 * IT. The vectorized part is the INTEGER inner dot only. Integer addition is
 * associative, so summing the eight i32 lanes of a `vpdpbusd` accumulator in
 * any order gives exactly the scalar loop's `sumi` -- the SIMD freedom is
 * free of consequence there. The f32 part is NOT associative, so the
 * per-block contribution `sum += (f32)sumi * (d_w * d_a)` is accumulated
 * SEQUENTIALLY, block by block, in precisely the scalar order.
 *
 * ggml does the opposite: it keeps a __m256 accumulator across blocks with
 * `_mm256_fmadd_ps` and horizontally reduces once at the end. That is faster
 * and it is CORRECT FOR ggml -- but it sums the block contributions in a
 * different order, so it cannot be bit-identical to a sequential scalar
 * reference, and adopting it would forfeit gate [1] for a few percent. The
 * probe notes say to copy ggml's IDIOM (the sign trick, the intrinsic
 * spelling, the macro guard); they do not say to copy its accumulation
 * strategy, and here the two requirements point in opposite directions.
 *
 * The cost of the exact ordering is one horizontal i32 reduction per 32
 * elements. That is real but small against the 32 scalar multiply-adds it
 * replaces, and gate [1] is not negotiable. */
f32 vec_dot_q8_0_q8_0_avx2(const void* wrow, const void* arow, u64 n);

/* FOUR rows of a Q8_0 matvec against one shared Q8_0-quantized activation.
 * out[k] is BIT-IDENTICAL to vec_dot_q8_0_q8_0(wrows[k], arow, n).
 *
 * WHY THIS EXISTS, measured 2026-08-03. The single-row twin closed only
 * 1.07x of the 1.997x gap to llama at one thread, and the attribution ladder
 * put the remainder OUTSIDE the kernel: a ggml-style vec_dot that abandons
 * bit-exactness entirely still left 1.54x on the table. What one row at a
 * time cannot express is memory-level parallelism -- a single row is one
 * load stream feeding one serial f32 accumulation chain, so the core spends
 * most of its time waiting rather than computing, at 8.3 GB/s against a
 * ~20 GB/s measured single-thread ceiling.
 *
 * ▶▶ AND IT COSTS NO GATE, WHICH IS THE WHOLE POINT. The rows of a matvec
 * are INDEPENDENT dot products -- nothing reduces across them -- so each
 * out[k] runs exactly the block loop and exactly the sequential f32
 * accumulation the single-row kernel runs. Bit-exactness is preserved per
 * row by construction rather than by tolerance, and gate [1] is untouched.
 * Four rows was chosen because the activation block is then loaded once
 * instead of four times while four independent chains are in flight.
 *
 * The shared activation is validated ONCE per block rather than once per
 * row. For valid tensors that is invisible. For a corrupt one, both paths
 * still fatal on a corrupt block -- but this one can name a different block
 * than the row-major scalar would reach first, since it walks blocks
 * outermost. Loud refusal either way; the index in the message may differ. */
void vec_dot_q8_0_q8_0_x4_avx2(const void* const wrows[4], const void* arow,
                                u64 n, f32 out[4]);

/* AVX2 twin of dequant_row_q8_0. BIT-IDENTICAL to the scalar kernel.
 *
 * Trivially so, and worth saying why: dequant is ELEMENTWISE (`out[i] = d *
 * (f32)q[i]`) with no reduction anywhere, so there is no summation order to
 * preserve and IEEE754 gives the same correctly-rounded product per lane as
 * per scalar iteration. Exactness here is a property of the operation, not
 * of care taken in the implementation. */
void dequant_row_q8_0_avx2(const void* row, u64 n, f32* out);

/* AVX2 twin of argmax. Returns the IDENTICAL index to the scalar kernel for
 * every input, including ties.
 *
 * THE TIE RULE IS THE WHOLE DIFFICULTY. The scalar kernel uses a strict `>`,
 * so equal values never displace an earlier index and the answer is the
 * FIRST occurrence of the maximum. A vector max-reduction loses that
 * ordering -- it finds the max VALUE across lanes with no notion of which
 * lane came first -- so the index has to be recovered in a second pass that
 * re-imposes "lowest index wins". A twin that returns any valid maximum
 * would pass a careless test and silently change generated tokens on the
 * first tie, which on a 151,936-wide logits vector is not a rare event.
 *
 * NaN: the scalar kernel's `>` is false against NaN, so a NaN never becomes
 * best and a leading NaN keeps index 0 unless a later real value exceeds it.
 * The twin reproduces that rather than the hardware's own NaN preference. */
u32 argmax_avx2(const f32* logits, u64 n);

/* K4: AVX2 twins of the K-quant integer dots. BIT-IDENTICAL to
 * vec_dot_q4_k_q8_k / vec_dot_q6_k_q8_k, asserted over random super-blocks
 * in tests/test_simd.c.
 *
 * WHY THESE ARE NOT OPTIONAL, measured 2026-08-04. Wiring the scalar
 * K-quant core made stoke run Q4_K_M at 0.560 tok/s against Q8_0's 2.002 --
 * the SMALLER file was 3.6x SLOWER, because the scalar dot has to widen
 * every 4-bit quant into an int8 scratch before it can multiply anything
 * while K3's twins covered Q8_0 only. The bar is quoted on Q4_K_M, so these
 * twins are load-bearing for it rather than an optimization layered on a
 * working path.
 *
 * ▶▶ BIT-EXACTNESS COMES FROM CONTRACT v2 (ruling 2026-08-12 §3), NOT from
 * the scalar's lane layout. This paragraph described the opposite until
 * 2026-08-14 and was wrong from `1f78ebe` onward -- recorded because the
 * distinction is exactly what the maddubs kernel depends on.
 *
 * The OLD contract was an accident of transcribing ggml: the scalar kept
 * eight accumulators where lane l collected elements congruent to l mod 8,
 * the twins inherited that assignment, and the eight f32 lanes were summed
 * 0..7 in a pinned order. Inheriting it made the twins exact -- and forbade
 * every wide-ISA kernel, because a kernel that partitions the elements
 * differently cannot reproduce a per-lane f32 partition. That is why the
 * original 8-lane x4 port was cancelled rather than written.
 *
 * Under v2 the contract is the super-block's exact INTEGER total S_b, plus
 * two f32 operations executed verbatim. Any implementation may reach S_b
 * however it likes -- integer addition is associative, so every partition
 * gives the same bits -- and only then runs the pinned f32 lines. So
 * `vec_dot_q4_k_q8_k_avx2`'s `vpmaddubsw` inner loop, which sums ADJACENT
 * PAIRS and therefore partitions the elements completely differently from
 * the scalar, is bit-exact by construction. tests/test_simd.c asserts it
 * against the scalar over 1e4 tensors rather than trusting this note.
 *
 * The f32 side is still left strictly alone: multiply and add stay two
 * separate roundings rather than an FMA. */
f32 vec_dot_q4_k_q8_k_avx2(const void* wrow, const void* arow, u64 n);
f32 vec_dot_q6_k_q8_k_avx2(const void* wrow, const void* arow, u64 n);

/* NOTE: a Q4_K x4 twin (rung A') was written and MEASURED at 0.907x -- a 9.3%
 * regression -- and removed on 2026-08-15. It was bit-exact; it was simply
 * slower, because maddubs had already removed the latency-bound regime the
 * Q8_0 x4 kernel above exploits. Preserved at 01184eb if anyone wants to
 * re-measure it on different hardware. */

/* M4.5 rung B (ruling 2026-08-12 §7.1): ONE weight row against N activations,
 * unpacking the row's weight-only work once instead of once per sequence.
 *
 * WHAT IS HOISTED, and it is exactly the work that does not depend on the
 * activation: the scale/min unpack, the 256 nibble widenings, and the two
 * f16→f32 conversions. WHAT IS NOT TOUCHED is every operation that touches
 * an activation, in the order it already ran.
 *
 * BIT-EXACTNESS IS BY CONSTRUCTION, NOT BY ARGUMENT. Each sequence keeps its
 * own (sums, sumf) pair across the super-block loop and the per-sequence
 * arithmetic executes in the same sequence, over the same values, as
 * vec_dot_q4_k_q8_k_avx2 would execute alone. Only the LOOP ORDER changes;
 * nothing sums across sequences and no grouping is regrouped. `out[s]` must
 * equal the solo kernel's return for every s, which tests/test_simd.c
 * asserts directly rather than trusting this comment.
 *
 * n_seq is tiled by the caller at KERNELS_BATCH_TILE so the per-sequence
 * accumulators stay a small stack array. */
/* Overridable at build time so the tile can be A/B'd against the shipping one
 * without editing a header -- see the M7 prefill investigation. */
#ifndef KERNELS_BATCH_TILE
#define KERNELS_BATCH_TILE 16u
#endif

/* ▶▶▶ AND n_rows IS TILED TOO, WHICH IS ABOUT CACHE AND NOTHING ELSE.
 *
 * With one weight row per call, a call touches every sequence's activation
 * before the next row starts: n_seq * (n/256) * 292 B, 46.7 KiB at n_seq=16
 * and n=2560, against a 48 KiB L1 -- so each of the thousands of rows re-reads
 * all of it from L2. Taking KERNELS_ROW_TILE rows per call and putting the ROW
 * loop INSIDE the super-block loop makes the live activation ONE super-block's
 * worth, 16 * 292 B = 4.7 KiB, shared by every row in the tile.
 *
 * MEASURED, not assumed: `-DSTOKE_MUT_ALIAS_ACT` -- which collapses the
 * activation to a single sequence's -- prices an L1-resident tile at 1.29x of
 * matmul time. ⚠ The SAME probe measured 5.8% on 2026-08-30, before the
 * per-sequence bookkeeping came out of these kernels. An ablation's verdict is
 * scoped to the version it ran on; removing the dominant cost promoted this
 * one.
 *
 * ⚠ THE WEIGHT SIDE DOES NOT PAY FOR IT. Prep stays once per (row,
 * super-block) exactly as before -- which is why this works where shrinking
 * KERNELS_BATCH_TILE does not (=8 measured 1.7% WORSE end-to-end, =4 8% worse,
 * because a smaller tile amortizes that prep over fewer sequences). A tile's
 * rows span n_rows*row_bytes and are swept super-block by super-block, so each
 * weight byte is still fetched once and every line is fully used.
 *
 * BIT-EXACT BY THE SAME CONSTRUCTION AS THE SEQUENCE TILING: each (row,
 * sequence) still accumulates its own pair over the super-blocks in ascending
 * order. Only the order of independent work changes.
 *
 * `out` is ROW-MAJOR, out[rr * n_seq + s], and `wrows` is the first of n_rows
 * rows spaced `row_bytes` apart. */
/* 64 IS MEASURED, NOT CHOSEN -- AND IT WAS 32 UNTIL THE KERNEL UNDER IT
 * CHANGED. Larger tiles amortize the super-block's activation slice over more
 * rows (the whole point), while a tile's rows span n_rows*row_bytes of weight
 * that must stream through L1 alongside it. Where those two cross is a
 * property of the KERNEL, not of the machine, so it moves when the kernel does.
 *
 * Swept against the maddubs kernel (2026-08-31), matmul ms on the 525-token
 * prefill: 8 -> 6014, 16 -> 5649, **32 -> 5434**, 64 -> 5545.
 *
 * Re-swept against the AVX-VNNI kernel (2026-09-01), which spends 9 uops on
 * ports 0-1 per (super-block, sequence) where maddubs spends 16 and carries a
 * 4 KiB permuted-activation buffer besides: 16 -> 4487.6, 32 -> 4325.1,
 * **64 -> 4255.9**, 128 -> 4356.3 (flat with 64, inside that block's 1.1%
 * control drift). **The optimum moved UP: 64 was 2% worse and is now 1.6%
 * better.** Four units per arm, ABBA, ranges DISJOINT (worst 64 beats best 32),
 * with the untouched `attention` bucket as a null arm at 1.003x.
 *
 * ⏹ AND IT IS BETTER ON ALL THREE AXES -- batched decode at 8 sequences reads
 * 63.638 -> 65.240 tok/s (1.025x, ranges disjoint), so this is not a
 * prefill-for-decode trade; and the GATE METRIC agrees, `bench --serve` 8f32
 * AGGREGATE(M0) 16.314 -> 16.784 (1.029x, four units per arm over two ABBA
 * blocks). ⚠ THE SERVE AXIS WAS CHECKED BECAUSE IT HAD NOT BEEN: the tile was
 * swept on `run` (prefill) and `batch` (decode), which are different mixes
 * from the one the gate measures, and the first post-change serve reading
 * looked LOW against a number taken hours earlier. It was block drift -- the
 * arms interleaved in one block disagree with that reading and agree with the
 * other two axes. A constant validated on the axes you happened to sweep is
 * not validated on the axis you are graded by. Bit-exact, checked rather than assumed:
 * `test_matvec_batch` 180/180 on both Q4_K kernels, and id streams `cmp`-equal
 * to the ROW_TILE=32 binary at n_seq=1 and at 4-way batch.
 *
 * ⚠ A SWEPT CONSTANT IS SCOPED TO THE KERNEL IT WAS SWEPT AGAINST. Re-run this
 * after any change to the inner loop's instruction mix or working set. */
#ifndef KERNELS_ROW_TILE
#define KERNELS_ROW_TILE 64u
#endif
void vec_dot_q4_k_q8_k_xn_avx2(const void* wrows, u64 row_bytes, u32 n_rows,
                               const void* const* arows, u32 n_seq, u64 n,
                               f32* out);
void vec_dot_q6_k_q8_k_xn_avx2(const void* wrows, u64 row_bytes, u32 n_rows,
                               const void* const* arows, u32 n_seq, u64 n,
                               f32* out);

/* ---- the Q4_K batched kernel on AVX-VNNI ---------------------------------
 *
 * A BIT-IDENTICAL twin of `vec_dot_q4_k_q8_k_xn_avx2` -- same integer total by
 * CONTRACT v2 -- that spends 9 uops on ports 0-1 per (super-block, sequence)
 * where the maddubs form spends 16. The trick is a cheap permutation, applied
 * identically to the nibbles and to the activations, that puts four DIFFERENT
 * sub-blocks in each `vpdpbusd` accumulator's lanes: the eight per-sub-block
 * scales then arrive in ONE `vpmaddwd` at the end of the super-block instead of
 * one per sub-block, which is the cost that made `vpdpbusd` look unusable here.
 * The full argument, the overflow ladder and the measured alternatives are at
 * the definition in simd.c.
 *
 * ⚠ CALLERS MUST CHECK `kernels_cpu_has_avx_vnni()`, like every other kernel
 * in this file -- `vec.c` gates the whole batched AVX2 path on it already.
 *
 * ⚠ WHICH ARM SHIPS IS `STOKE_Q4K_VNNI`; `STOKE_Q4K_KERNEL=maddubs|vnni`
 * overrides it at runtime so both arms come from ONE binary in ONE sitting,
 * for the reason `STOKE_Q6K_KERNEL` exists. */
void vec_dot_q4_k_q8_k_xn_vnni_avx2(const void* wrows, u64 row_bytes,
                                    u32 n_rows, const void* const* arows,
                                    u32 n_seq, u64 n, f32* out);
/* The mins-term A/B knob. IT LIVES IN THE HEADER because both simd.c (which
 * has the two spellings) and vec.c (which must not offer the VNNI kernel when
 * the SIMD spelling is off) have to agree on its value. As a `#define` in
 * simd.c it was UNDEFINED in vec.c, where `#if !STOKE_MINS_SIMD` is `#if !0` --
 * always true -- so the guard fired in every build and switched the VNNI kernel
 * silently off. Caught by `STOKE_Q4K_KERNEL=bogus` no longer being REFUSED,
 * which is the anti-vacuity check that function already carried. */
#ifndef STOKE_MINS_SIMD
#define STOKE_MINS_SIMD 1
#endif

/* SHIPS ON, 2026-09-01. Prefill 6359.7 -> 5443.6 ms = 1.168x, ten units per
 * arm from ONE binary, ABBA-counterbalanced on a box the user flagged as noisy
 * -- readable anyway because the arms' RANGES DO NOT OVERLAP (worst vnni 5608.9
 * beats best maddubs 6285.0) and because the untouched `attention` bucket, a
 * within-process control, reads 873.7 vs 874.1 = 1.000x across the arms. The
 * Q4_K matmuls alone are 1.258x, which is the whole of the 1.272x an
 * instruction-mix probe said was there: the two permutations cost nothing
 * measurable. Decode is untouched structurally (n_seq==1 never reaches an xn
 * kernel) and checked. */
#ifndef STOKE_Q4K_VNNI
#define STOKE_Q4K_VNNI 1
#endif

/* ---- the Q6_K batched kernel, re-formulated on vpmaddubsw -----------------
 *
 * ▶▶ WHY IT EXISTS: MEASURED 2026-08-27, THE BATCHED Q6_K KERNEL GOES BACKWARDS.
 * `stoke bench --matvec-batch`, one tensor per process: every Q4_K tensor CLIMBS
 * with N (38 -> 70 GF/s) while `ffn_down` (Q6_K) reads 33.42 at N=1 and then
 * 17.64 / 19.27 / 20.44 / 10.89. The SOLO kernel is fine; batching makes this one
 * ~2x WORSE where it makes Q4_K ~2x better.
 *
 * The reason is density, and it is visible in the source. Per sequence per
 * super-block, `vec_dot_q4_k_q8_k_xn_avx2` runs 8 iterations of
 * `vpmaddubsw` + `vpmaddwd` -- 32 int8 products per instruction. The Q6_K twin
 * above runs 32 iterations of `kq_load_i8x8_to_i32` + `kq_mac8`: it WIDENS i8 to
 * i32 and does 8 products per operation. Same 256 elements, ~4x the uops -- in
 * the per-sequence body, which is exactly what batching multiplies by N.
 *
 * ▶ WHY vpmaddubsw APPLIES TO Q6_K AT ALL, given its quants are SIGNED.
 * `vpmaddubsw` takes its first operand UNSIGNED. Q6_K's stored value is a 6-bit
 * unsigned `u` in [0,63] which the reference offsets to `u - 32`. Keep `u` raw
 * and carry the offset as a separate term -- exactly the shape Q4_K already uses
 * for its `mins`:
 *
 *     SUM (u-32)*a*s  ==  SUM u*a*s  -  32 * SUM a*s
 *
 * and `SUM a` over each 16-element scale group is precisely what Q8_K's `bsums`
 * already stores (see quantize_row_q8_k). So the second term is one `vpmaddwd`
 * of the scales against the bsums, and nothing has to be recomputed.
 *
 * ▶ BIT-EXACTNESS IS BY CONSTRUCTION, NOT BY TESTING, and the bounds are the
 * argument. Every term is an integer and integer arithmetic is associative, so
 * the regrouping is exact PROVIDED nothing overflows:
 *   - `vpmaddubsw` pairs: 2 * 63 * 127 = 16,002 against i16's 32,767.
 *   - the following `vpmaddwd` against the scales: 16,002 * 127 * 2 = 4,064,508.
 *   - accumulated over 8 chunks and 8 lanes: 260,128,512 against i32's 2^31.
 *   - the bsums term: 127 * 2032 * 2 = 516,128, and 32 * that over 8 lanes is
 *     132,128,768.
 * `S_b` is therefore the SAME INTEGER the twin above computes, and the one f32
 * expression that consumes it is byte-identical -- which is what tests/test_simd.c
 * asserts directly rather than inheriting from this comment.
 *
 * ▶ A 32-BYTE vpmaddubsw SPANS TWO OF Q6_K's SCALE GROUPS, which is the one
 * structural difference from Q4_K (whose 8 scales over 256 elements line up with
 * its chunks exactly). Q6_K has 16 scales, so a chunk covers groups 2c and 2c+1;
 * the following `vpmaddwd` against `[s_2c x8, s_2c+1 x8]` applies each to its own
 * half, since within a 128-bit lane every scale lane is the same value and
 * `(a+b)*s == a*s + b*s`.
 *
 * DECLARED SEPARATELY AND CALLABLE DIRECTLY, for the reason this header gives at
 * the top: a dispatching entry point cannot be tested against the thing it
 * dispatches to. `vec.c` chooses between the two. */
/* The two attention inner loops, AVX2. simd.c says why the score kernel stays
 * 128-bit (its lanes are the GQA group's q-heads, so widening would reassociate
 * the sum over d) while the accumulate kernel is 256-bit (its lanes are
 * independent d's, so the width is free). Both are bit-exact against model.c's
 * SSE2 arm; the caller gates on kernels_cpu_has_avx_vnni().
 *
 * `qt` is the TRANSPOSED q block: element (d, t) is qt[d*qstride + t]. */
void att_score4_avx2(const f32* qt, u64 qstride, const f32* kj, u32 hd,
                     f32 scale, f32* out4);
void att_accum4_avx2(f32* o0, f32* o1, f32* o2, f32* o3, const f32* vj,
                     const f32* w4, u32 hd);

void vec_dot_q6_k_q8_k_xn_maddubs_avx2(const void* wrows, u64 row_bytes,
                                       u32 n_rows, const void* const* arows,
                                       u32 n_seq, u64 n, f32* out);

/* Which Q6_K batched kernel `matvec_batch` dispatches to.
 *
 * The COMPILE-TIME default is what ships; `STOKE_Q6K_KERNEL=ref|maddubs`
 * overrides it at runtime so both arms come from ONE binary in ONE sitting --
 * the same reasoning `sched.c` gives for `STOKE_SCHED_CHUNK`, and it matters
 * more here: the 2026-08-24 `vpmulld` ablation had to build its two arms into
 * equal-length directories because mingw embeds object paths and PE here is not
 * position-independent, so a longer build directory shifts `.text` and code
 * layout alone can move a hot loop by a few percent.
 *
 * An unrecognised value is REFUSED rather than ignored: a typo that silently
 * selected the default would make an A/B measure one kernel twice.
 *
 * ▶▶ FLIPPED TO 1 ON 2026-08-27, AND WHAT PAID FOR IT.
 * `tests/test_simd.c` asserts the maddubs twin bit-identical to the reference
 * twin AND to the solo kernel over random tensors -- but that grades the KERNEL,
 * not the ENGINE, and a perfect twin reached through the wrong dispatch is still
 * a wrong engine. So the flip waited on a STREAM gate:
 * gate-reports/2026-08-27-m5-q6k-gate-report.md, 11 cases (both prompts, kv f32
 * and q8, 2 and 6 threads, `batch --seqs` 2/8/16), every one byte-compared to
 * build/stoke-m45-gate-fddd12cd.exe -- a binary built BEFORE this kernel
 * existed. ref MATCH 11/11 (so nothing else moved), maddubs MATCH 11/11, the
 * lane-swapped mutant DIFFERS 11/11, and the mutant's banner proves the batched
 * Q6_K kernel was DISPATCHED in all 11 -- a case that never reached it would
 * have compared EQUAL and graded nothing. */
#ifndef STOKE_Q6K_MADDUBS
#define STOKE_Q6K_MADDUBS 1
#endif

#endif /* STOKE_KERNELS_SIMD_H */
