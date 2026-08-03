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

#endif /* STOKE_KERNELS_SIMD_H */
