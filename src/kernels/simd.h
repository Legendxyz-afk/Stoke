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
 * BIT-EXACTNESS COMES FROM THE SCALAR'S OWN SHAPE, which is a happier
 * accident than Q8_0's was. The scalar K-quant kernels already keep eight
 * independent accumulators (aux32[0..7] in i32, sums[0..7] in f32) where
 * lane l collects the elements at indices congruent to l mod 8 -- that IS a
 * 256-bit register's lane layout, so the twins inherit the assignment rather
 * than having to reconstruct it. Integer reordering is then free (exact and
 * associative, bounds checked in simd.c), and the f32 side is left strictly
 * alone: multiply and add stay two separate roundings rather than an FMA,
 * and the final horizontal sum runs lane 0..7 in order because that is the
 * scalar's order. */
f32 vec_dot_q4_k_q8_k_avx2(const void* wrow, const void* arow, u64 n);
f32 vec_dot_q6_k_q8_k_avx2(const void* wrow, const void* arow, u64 n);

#endif /* STOKE_KERNELS_SIMD_H */
