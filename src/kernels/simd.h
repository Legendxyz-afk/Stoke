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
#define KERNELS_BATCH_TILE 16u
void vec_dot_q4_k_q8_k_xn_avx2(const void* wrow, const void* const* arows,
                               u32 n_seq, u64 n, f32* out);
void vec_dot_q6_k_q8_k_xn_avx2(const void* wrow, const void* const* arows,
                               u32 n_seq, u64 n, f32* out);

#endif /* STOKE_KERNELS_SIMD_H */
