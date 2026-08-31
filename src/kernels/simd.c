/* src/kernels/simd.c -- K3 SIMD twins. See simd.h for the contract and for
 * why bit-exactness, not speed, dictates the structure.
 *
 * Compiled with -mavx2 -mfma -mf16c -mavxvnni (see the Makefile rule that
 * applies them to THIS FILE ONLY). Nothing here may be called without
 * kernels_cpu_has_avx_vnni() returning true first.
 */
#include "common.h"

#include "kernels/kernels.h"
#include "kernels/simd.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)

#include <cpuid.h>
#include <immintrin.h>

/* K0's P-B probe found that the WRONG spelling (`_mm256_dpbusd_epi32`, the
 * AVX-512 name) compiles AND runs on both of this project's toolchains and
 * emits the identical `{vex} vpdpbusd`. The forward note's promised compile
 * error does not exist, so there is no compile-time protection anywhere in
 * the matrix. ggml guards by macro and never relies on the compiler
 * rejecting a spelling; this copies that structure and adds the #error the
 * toolchains will not give us. */
#if !defined(__AVXVNNI__)
#error "simd.c requires -mavxvnni (K0 P-B: neither toolchain rejects the wrong intrinsic spelling, so this guard is the only protection)"
#endif

/* ▶ THE MINS TERM'S SPELLING. IT DEFAULTS TO SIMD SINCE 2026-08-24.
 * `vec_dot_q4_k_q8_k_xn_avx2` computed its per-sequence mins term with a scalar
 * loop that objdump showed was ~40-45% of the emitted per-sequence work. The
 * SIMD spelling is bit-exact by construction (integer, associative, proved
 * non-overflowing at the site) -- but "bit-exact by construction" is an
 * ARGUMENT, and this project ships on measurement. Both stated bars were met
 * before this default moved:
 *
 *   SPEED -- interleaved A/B, arm order alternating per cycle: 1.063x (v1) and
 *   1.099x (v2) at t=1, 1.079x / 1.071x at t=8, B winning 6 of 6 pairings in
 *   each run. `gate-reports/2026-08-20-m5-mins-ab-v2-report.md`.
 *
 *   EXACTNESS -- the 10-case lineage grid on the `run` path plus 4 `batch`
 *   cases, every arm against an ARCHIVED binary rather than against each other:
 *   14 cases x 2 arms, all bit-identical.
 *   `gate-reports/2026-08-24-m5-mins-verify.txt`.
 *
 * ⚠ AND THAT GATE CARRIES ITS OWN PROOF THAT IT COULD HAVE FAILED. A third
 * binary built from this file with ONE WRONG LANE in `dup` ran inside the gate
 * on one case per leg and DIFFERED on both. Without that, 28 SAMEs would have
 * been consistent with a kernel that never executed -- which was a live worry,
 * because the 08-20 memo believed `stoke run` could not reach this kernel at
 * all. It can: E3 chunks prefill through `model_forward_batch` (default
 * STOKE_PREFILL_CHUNK=128) and vec.c tiles at KERNELS_BATCH_TILE=16, so `run`
 * enters here at n_seq=16 and then the chunk remainder. Decode does not.
 *
 * Build the old arm with -DSTOKE_MINS_SIMD=0 (Makefile: EXTRA_CFLAGS). It is
 * kept so the A/B can be re-run and so a toolchain regression has a referent. */
#ifndef STOKE_MINS_SIMD
#define STOKE_MINS_SIMD 1
#endif

/* ---------------------------------------------------------------- feature */

/* Cached across calls: CPUID is serializing and this is read per matvec row.
 * -1 = not yet probed. No lock: the probe is idempotent and writes a value
 * every racing thread would compute identically, so the worst a data race
 * can do is run it twice. (K5 adds the pool; this stays correct there.) */
static int g_has_avx_vnni = -1;

static bool probe_avx_vnni(void) {
    unsigned eax, ebx, ecx, edx;

    /* The OS must have enabled YMM state saving, or every AVX instruction
     * below faults regardless of what the CPU reports supporting. Checking
     * CPUID alone is the classic way to ship a binary that SIGILLs on a
     * machine whose CPU is perfectly capable -- and it would defeat the
     * scalar fallback this whole file is guarded by. */
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    const unsigned OSXSAVE = 1u << 27;
    const unsigned AVX     = 1u << 28;
    if ((ecx & (OSXSAVE | AVX)) != (OSXSAVE | AVX)) return false;
    /* XCR0 bits 1 (SSE/XMM) and 2 (AVX/YMM) both set. */
    if ((_xgetbv(0) & 0x6u) != 0x6u) return false;

    /* AVX2 itself: leaf 7 subleaf 0, EBX bit 5. The twins use AVX2 integer
     * ops beyond VNNI, so VNNI alone is not sufficient. */
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    if ((ebx & (1u << 5)) == 0u) return false;

    /* AVX-VNNI (VEX, 256-bit): leaf 7 subleaf 1, EAX bit 4. This is the bit
     * the forward note names, and it is NOT AVX-512-VNNI -- K0's P-C probe
     * measured this box at AVX-VNNI = 1, AVX-VNNI-INT8 = 0. */
    if (!__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) return false;
    return (eax & (1u << 4)) != 0u;
}

bool kernels_cpu_has_avx_vnni(void) {
    if (g_has_avx_vnni < 0) g_has_avx_vnni = probe_avx_vnni() ? 1 : 0;
    return g_has_avx_vnni == 1;
}

/* ------------------------------------------------------------------ twins */

/* Sum of eight i32 lanes. Exact at any order -- integer addition is
 * associative, which is the entire reason the integer core makes a
 * bit-exact SIMD twin possible at all.
 *
 * ▶ ONE REDUCTION PER BLOCK LOOKS WASTEFUL AND WAS MEASURED TO BE FINE.
 * 2026-08-03: a 4-block unrolled variant was written and verified bit-exact,
 * holding four accumulators in registers and folding them with three
 * `vphaddd` so the four cross-lane chains became one. Confirmed from objdump
 * that it did what was intended -- the reduction dropped from ~24 ops per
 * four blocks to ~11 (and the first attempt, which used `for (k<4)` over an
 * `__m256i acc[4]`, did NOT: gcc kept the loop rolled and spilled the
 * accumulators, emitting 4 `vpdpbusd` where 8 were wanted).
 *
 * It bought nothing. Measured A,B,A,B against this version: **1.0125x CPU
 * time, i.e. 1.25% WORSE**, over 4 interleaved cycles. The ALU savings are
 * invisible because this loop is not ALU-bound. So the simple version
 * stands, and the unroll is recorded here rather than re-attempted.
 *
 * ▶▶ AND THE MEASUREMENT METHOD IS THE MORE REUSABLE FINDING. Wall-clock
 * A/B on a busy box was useless: 117% spread across cycles, ratios from 0.51
 * to 1.12, on a box carrying ~8 foreign cores. The SAME comparison by
 * PROCESS CPU TIME spread only 9.9%. Interleaving controls for slow drift
 * BETWEEN cycles; it does nothing about high-frequency contention WITHIN
 * one, and wall time absorbs every descheduling event while CPU time does
 * not. When the box cannot be quiesced, compare CPU time. */
static inline i32 hsum_i32_8(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* Mirrors q8_0.c's block_scale_or_fatal, including the message, so a corrupt
 * tensor fails identically whichever path ran. */
static f32 simd_block_scale_or_fatal(const u8* blk, u64 b, const char* who) {
    u16 raw;
    memcpy(&raw, blk, sizeof raw);
    f32 d = f16_to_f32(raw);
    if (isinf(d) || isnan(d)) {
        fatal("kernels: %s -- block %llu scale decodes to inf/nan -- "
              "corrupt or hostile Q8_0 tensor (a sane GGUF cannot contain "
              "one)",
              who, (unsigned long long)b);
    }
    return d;
}

/* One block's integer dot, as eight partial i32 lanes. Shared by the
 * single-row and four-row kernels so the idiom has exactly one definition. */
static inline __m256i block_acc(const u8* wblk, const u8* ablk,
                                 __m256i bias, __m256i ones) {
    /* loadu, never load: these point into the GGUF mmap at a 34-byte block
     * stride + 2, so they are essentially never 32-byte aligned. */
    __m256i wv = _mm256_loadu_si256((const __m256i*)(const void*)(wblk + 2));
    __m256i av = _mm256_loadu_si256((const __m256i*)(const void*)(ablk + 2));
    /* ▶▶ WHY THIS IS *NOT* ggml's SIGN TRICK, which the probe notes told
     * us to copy. `vpdpbusd` multiplies UNSIGNED bytes by SIGNED bytes,
     * so one side has to be made unsigned. ggml does it by moving the
     * weight's sign onto the activation:
     *     ax = |w| ; sy = sign_epi8(a, w) ; dot = dpbusd(ax, sy)
     * That is exact for every input ggml can actually produce, because
     * quantize_row_q8_0 computes roundf(x * 127/absmax) and therefore
     * emits quants in [-127, 127] -- never -128.
     *
     * It is NOT exact at a = -128 with w < 0. PSIGNB negates in two's
     * complement, and -(-128) wraps back to -128, so the product comes
     * out with the WRONG SIGN: true w*a = +|w|*128, the trick yields
     * -|w|*128. Our scalar reference handles that input correctly, so
     * adopting the trick would put a real scalar<->SIMD divergence into
     * the milestone whose gate is bit-exact agreement -- reachable by a
     * corrupt or hostile GGUF, which is a file we do not write.
     *
     * The bias decomposition below is exact for the FULL i8 range and
     * costs one extra dpbusd plus two cheap integer ops per 32 elements:
     *     w_u = w XOR 0x80          (maps [-128,127] -> [0,255] = w+128)
     *     p1  = dpbusd(w_u, a)      lanes: sum (w+128)*a
     *     p2  = dpbusd(ones, a)     lanes: sum a
     *     sumi = hsum(p1 - 128*p2)  = sum w*a
     * Both intermediates stay far inside i32 (p1 lane <= 4*255*127 =
     * 129,540; 128*p2 lane <= 128*4*127 = 65,024).
     *
     * The probe notes said "match ggml, then let the exact-diff gate
     * arbitrate". The gate arbitrated: this is the deviation it bought. */
    __m256i wu = _mm256_xor_si256(wv, bias);
    __m256i p1 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), wu, av);
    __m256i p2 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ones, av);
    return _mm256_sub_epi32(p1, _mm256_slli_epi32(p2, 7));
}

f32 vec_dot_q8_0_q8_0_avx2(const void* wrow, const void* arow, u64 n) {
    if (n % KERNELS_Q8_0_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q8_0_q8_0_avx2 -- n=%llu is not a multiple "
              "of %u",
              (unsigned long long)n, KERNELS_Q8_0_BLOCK_ELEMS);
    }
    const u8* wbase = (const u8*)wrow;
    const u8* abase = (const u8*)arow;
    u64       num_blocks = n / KERNELS_Q8_0_BLOCK_ELEMS;
    f32       sum = 0.0f;

    /* The +128 bias vector and the all-ones multiplier, hoisted. */
    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i ones = _mm256_set1_epi8(1);

    for (u64 b = 0; b < num_blocks; b++) {
        const u8* wblk = wbase + b * KERNELS_Q8_0_BLOCK_BYTES;
        const u8* ablk = abase + b * KERNELS_Q8_0_BLOCK_BYTES;
        f32       dw = simd_block_scale_or_fatal(wblk, b,
                                                 "vec_dot_q8_0_q8_0 (weight)");
        f32       da = simd_block_scale_or_fatal(
            ablk, b, "vec_dot_q8_0_q8_0 (activation)");

        __m256i acc = block_acc(wblk, ablk, bias, ones);

        /* Sequential f32 accumulation in the scalar's exact order, and the
         * two scales multiplying each other FIRST -- see simd.h. */
        sum += (f32)hsum_i32_8(acc) * (dw * da);
    }
    return sum;
}

void vec_dot_q8_0_q8_0_x4_avx2(const void* const wrows[4], const void* arow,
                                u64 n, f32 out[4]) {
    if (n % KERNELS_Q8_0_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q8_0_q8_0_x4_avx2 -- n=%llu is not a multiple "
              "of %u",
              (unsigned long long)n, KERNELS_Q8_0_BLOCK_ELEMS);
    }
    const u8* w0 = (const u8*)wrows[0];
    const u8* w1 = (const u8*)wrows[1];
    const u8* w2 = (const u8*)wrows[2];
    const u8* w3 = (const u8*)wrows[3];
    const u8* ab = (const u8*)arow;
    u64       num_blocks = n / KERNELS_Q8_0_BLOCK_ELEMS;

    const __m256i bias = _mm256_set1_epi8((char)0x80);
    const __m256i ones = _mm256_set1_epi8(1);

    /* FOUR INDEPENDENT ACCUMULATORS, four independent chains. Each is the
     * same running f32 sum the single-row kernel keeps, stepped in the same
     * block order -- the rows never interact, which is exactly why this can
     * be four times the work without being a numerics change. */
    f32 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    for (u64 b = 0; b < num_blocks; b++) {
        u64 off = b * KERNELS_Q8_0_BLOCK_BYTES;
        /* Shared by all four rows, so decoded and validated once per block
         * rather than once per row -- see simd.h on what that means for the
         * message a corrupt tensor produces. */
        f32 da = simd_block_scale_or_fatal(ab + off, b,
                                           "vec_dot_q8_0_q8_0 (activation)");
        f32 d0 = simd_block_scale_or_fatal(w0 + off, b,
                                           "vec_dot_q8_0_q8_0 (weight)");
        f32 d1 = simd_block_scale_or_fatal(w1 + off, b,
                                           "vec_dot_q8_0_q8_0 (weight)");
        f32 d2 = simd_block_scale_or_fatal(w2 + off, b,
                                           "vec_dot_q8_0_q8_0 (weight)");
        f32 d3 = simd_block_scale_or_fatal(w3 + off, b,
                                           "vec_dot_q8_0_q8_0 (weight)");

        __m256i a0 = block_acc(w0 + off, ab + off, bias, ones);
        __m256i a1 = block_acc(w1 + off, ab + off, bias, ones);
        __m256i a2 = block_acc(w2 + off, ab + off, bias, ones);
        __m256i a3 = block_acc(w3 + off, ab + off, bias, ones);

        /* Same grouping as the scalar: the two scales multiply each other
         * first, then the integer sum. */
        s0 += (f32)hsum_i32_8(a0) * (d0 * da);
        s1 += (f32)hsum_i32_8(a1) * (d1 * da);
        s2 += (f32)hsum_i32_8(a2) * (d2 * da);
        s3 += (f32)hsum_i32_8(a3) * (d3 * da);
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

void dequant_row_q8_0_avx2(const void* row, u64 n, f32* out) {
    if (n % KERNELS_Q8_0_BLOCK_ELEMS != 0) {
        fatal("kernels: dequant_row_q8_0_avx2 -- n=%llu is not a multiple of "
              "%u",
              (unsigned long long)n, KERNELS_Q8_0_BLOCK_ELEMS);
    }
    const u8* base = (const u8*)row;
    u64       num_blocks = n / KERNELS_Q8_0_BLOCK_ELEMS;

    for (u64 b = 0; b < num_blocks; b++) {
        const u8* blk = base + b * KERNELS_Q8_0_BLOCK_BYTES;
        f32       d   = simd_block_scale_or_fatal(blk, b, "dequant_row_q8_0");
        const u8* qs  = blk + 2;
        f32*      o   = out + b * KERNELS_Q8_0_BLOCK_ELEMS;
        __m256    vd  = _mm256_set1_ps(d);

        /* Eight at a time: cvtepi8_epi32 widens i8 -> i32 exactly (the whole
         * i8 range is representable), cvtepi32_ps then converts exactly (all
         * of [-128,127] is representable in f32), and mul_ps is correctly
         * rounded. So each lane computes precisely the scalar's `d * (f32)q`
         * with no reduction anywhere to reorder. */
        for (u32 g = 0; g < KERNELS_Q8_0_BLOCK_ELEMS; g += 8u) {
            __m128i q8  = _mm_loadl_epi64((const __m128i*)(const void*)(qs + g));
            __m256i q32 = _mm256_cvtepi8_epi32(q8);
            _mm256_storeu_ps(o + g,
                             _mm256_mul_ps(vd, _mm256_cvtepi32_ps(q32)));
        }
    }
}

u32 argmax_avx2(const f32* logits, u64 n) {
    if (n == 0u) return 0u;

    /* THE SCALAR'S NaN SEMANTICS, REPRODUCED RATHER THAN APPROXIMATED. Its
     * test is a strict `>`, which is false in both directions against NaN.
     * So if logits[0] is NaN nothing can ever displace it and the answer is
     * 0 no matter what follows -- a fact no max-reduction would arrive at by
     * itself. */
    if (isnan(logits[0])) return 0u;

    u64    i    = 0;
    __m256 vmax = _mm256_set1_ps(logits[0]);
    for (; i + 8u <= n; i += 8u) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(logits + i));
    }
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(vmax),
                          _mm256_extractf128_ps(vmax, 1));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    f32 best = _mm_cvtss_f32(m);
    for (; i < n; i++) {
        if (logits[i] > best) best = logits[i];
    }

    /* MAXPS returns its SECOND operand when either is NaN, so a NaN anywhere
     * past index 0 can land in `best`. Rather than reason about which lane
     * won, hand those inputs to the reference implementation: it is the
     * definition of the answer, and this cannot happen on logits a working
     * forward pass produced. */
    if (isnan(best)) return argmax(logits, n);

    /* SECOND PASS, AND IT IS WHAT MAKES THE TIE RULE COME OUT RIGHT. The
     * reduction above found the maximum VALUE; it has no idea which lane held
     * it, and lanes carry no ordering. Scanning forward for the FIRST element
     * equal to that maximum re-imposes "lowest index wins" exactly as the
     * scalar `>` does. Vectorized too, so the pass is cheap: movemask plus a
     * count-trailing-zeros finds the first matching lane directly. */
    __m256 vb = _mm256_set1_ps(best);
    u64    j  = 0;
    for (; j + 8u <= n; j += 8u) {
        __m256 eq = _mm256_cmp_ps(_mm256_loadu_ps(logits + j), vb,
                                  _CMP_EQ_OQ);
        int    mask = _mm256_movemask_ps(eq);
        if (mask != 0) {
            return (u32)(j + (u64)__builtin_ctz((unsigned)mask));
        }
    }
    for (; j < n; j++) {
        if (logits[j] == best) return (u32)j;
    }

    /* `best` was taken FROM the array, so some element equals it. */
    fatal("kernels: argmax_avx2 -- internal: maximum %.9g not found on the "
          "second pass over %llu elements",
          (double)best, (unsigned long long)n);
    return 0u;
}

/* ---------------------------------------------------------- K4: K-quants
 *
 * HOW THESE STAY BIT-EXACT, and it is a different argument from Q8_0's.
 *
 * The scalar K-quant kernels already keep EIGHT independent accumulators --
 * aux32[0..7] in i32 and sums[0..7] in f32 -- where lane l collects every
 * element whose index is congruent to l mod 8. That structure is already the
 * shape of a 256-bit register, so the twin holds aux32 in one __m256i and
 * sums in one __m256 and the lane assignment is IDENTICAL by construction.
 *
 * In the INTEGER domain reordering is free: i32 addition is exact and
 * associative, so folding the four per-8 groups of a sub-block before
 * applying the scale gives precisely the scalar's value (scale*sum == sum of
 * scale*x, exactly, in integers). Bounds checked rather than assumed: the
 * largest |product| is 127*127 = 16129, a Q4_K scale is 6-bit (<=63) and a
 * Q6_K scale is i8 (<=127), so the worst-case |aux32| is ~32.5M (Q4_K) and
 * ~65.5M (Q6_K) -- both far inside i32.
 *
 * In the f32 domain nothing may move. `sums[l] += d * (f32)aux32[l]` is a
 * multiply THEN an add -- two roundings -- so it is written as a separate
 * _mm256_mul_ps and _mm256_add_ps and NOT as an FMA, which would round once
 * and produce a different number. This is the concrete reason
 * -ffp-contract=off is pinned globally: without it the compiler is free to
 * fuse those two into an FMA and silently break gate [1] on whichever
 * toolchain happens to default that way.
 *
 * The final horizontal reduction is likewise sequential in lane order --
 * sumf += sums[0], then [1], ... [7] -- because that is the scalar's order,
 * and a tree reduction would be a different (equally valid, non-identical)
 * f32 sum. Same trap K3 documented for ggml's cross-block accumulator.
 */

/* Multiply-accumulate 8 elements: acc += cvt(w8) * cvt(a8), i32 lanes.
 * `w` values are 0..15 (Q4_K nibbles) or already-centred i8 (Q6_K), so the
 * caller picks the right widening. */
static inline __m256i kq_mac8(__m256i acc, __m256i w32, __m256i a32) {
    return _mm256_add_epi32(acc, _mm256_mullo_epi32(w32, a32));
}

static inline __m256i kq_load_i8x8_to_i32(const i8* p) {
    return _mm256_cvtepi8_epi32(
        _mm_loadl_epi64((const __m128i*)(const void*)p));
}

/* ▶ THREE HELPERS WERE DELETED HERE, and what they were is the point.
 * `kq_accum_sums` and `kq_finish` implemented the OLD contract's f32 fold --
 * eight lanes multiplied and summed 0..7 in a pinned order -- and lost their
 * last call site when CONTRACT v2 landed in 1f78ebe. `kq_load_u8x8_to_i32`
 * widened Q4_K's nibbles to i32 lanes and lost its last call site when the
 * maddubs kernels started consuming them as bytes. They survived -Werror only
 * because `static inline` is exempt from -Wunused-function, so nothing
 * complained; they were removed rather than left because code describing an
 * abandoned contract is worse than no code -- simd.h's note describing that
 * same contract had already misled for three days. */

/* CONTRACT v2: reduce the eight i32 lanes to the super-block's single exact
 * total. Integer adds are associative, so ANY order gives the same bits --
 * that associativity is precisely what lets a twin of any width meet the
 * scalar's fold without inheriting its lane partition. */
static inline i32 kq_hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* CONTRACT v2, FOUR SUPER-BLOCK TOTALS AT ONCE -- lane k of the result is
 * exactly what `kq_hsum_i32` would return for the k-th argument, and it is
 * exact for the reason stated just above: the totals are INTEGERS, integer
 * addition is associative, so the tree shape is free and a wider one inherits
 * the scalar's answer rather than approximating it.
 *
 * ▶ WHY IT EXISTS. In the BATCHED kernels this reduction runs once per
 * (super-block, SEQUENCE) -- twice, counting the mins term -- and `objdump`
 * puts the pair at 14 of the 55 instructions in that loop body. Four separate
 * calls are 28 instructions; this is 11.
 *
 * `vphaddd` is per 128-bit lane, so hadd(a,b) = [a01,a23,b01,b23 |
 * a45,a67,b45,b67]. After the second one each 128-bit lane holds one partial
 * per input in input order, and adding the two lanes finishes all four. */
static inline __m128i kq_hsum4_i32(__m256i a, __m256i b, __m256i c,
                                   __m256i d) {
    __m256i ab   = _mm256_hadd_epi32(a, b);
    __m256i cd   = _mm256_hadd_epi32(c, d);
    __m256i abcd = _mm256_hadd_epi32(ab, cd);
    return _mm_add_epi32(_mm256_castsi256_si128(abcd),
                         _mm256_extracti128_si256(abcd, 1));
}

/* One super-block of the Q4_K x Q8_K dot, for ONE sequence: the 23 of the 55
 * instructions that are the dot product. Factored out so the grouped path and
 * the remainder path below cannot drift apart -- they are the same arithmetic
 * in the same order, and the only thing the grouping changes is how the
 * results are reduced and folded. */
static inline __m256i kq4_dot_block(const __m256i* wnib, const u8* scales8,
                                    const i8* q8) {
    __m256i acc = _mm256_setzero_si256();
    for (u32 j = 0; j < 8u; ++j) {
        __m256i av =
            _mm256_loadu_si256((const __m256i*)(const void*)(q8 + j * 32u));
        acc = _mm256_add_epi32(
            acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wnib[j], av),
                                   _mm256_set1_epi16((short)scales8[j])));
    }
    return acc;
}

f32 vec_dot_q4_k_q8_k_avx2(const void* wrow, const void* arow, u64 n) {
    if (n % KERNELS_Q4_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q4_k_q8_k_avx2 -- n=%llu is not a multiple "
              "of %u",
              (unsigned long long)n, KERNELS_Q4_K_BLOCK_ELEMS);
    }
    const u8* wb = (const u8*)wrow;
    const u8* ab = (const u8*)arow;
    u64 nb = n / KERNELS_Q4_K_BLOCK_ELEMS;

    /* BYTE-wise mask, not the epi32 one the widening kernel used: the whole
     * point below is that nothing is widened to i32 before the multiply. */
    const __m256i lownib8 = _mm256_set1_epi8(0xF);
    f32 sumf = 0.0f;

    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q4_K_BLOCK_BYTES;
        const u8* q4 = blk + 16;
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        const u8* bsums = ablk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
        f32 ad;
        memcpy(&ad, ablk, sizeof ad);

        u8 scales8[8], mins8[8];
        kernels_q4_k_unpack_scales(blk + 4, scales8, mins8);

        /* The min term: 16 i16 bsums against 8 mins, mins8[j/2]. Sixteen
         * integer terms per super-block -- left scalar because vectorising
         * it would cost more shuffling than it saves. */
        i32 sumi = 0;
        for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS / 16u; ++j) {
            i16 bs;
            memcpy(&bs, bsums + 2u * j, sizeof bs);
            sumi += (i32)bs * (i32)mins8[j / 2u];
        }

        __m256i acc = _mm256_setzero_si256();
        /* ── M4.5 rung C, step 3: the maddubs inner loop. THIS is where C's
         * throughput lives; the contract re-anchor exists to permit it.
         *
         * ONE `vpmaddubsw` retires 32 products. The kernel it replaces widened
         * every quant to an i32 lane first and got 8 per `vpmulld` -- the 4x
         * inner-loop deficit the attribution ladder measured against llama.
         *
         * ▶▶ WHY THIS IS BIT-EXACT AND WHY IT WAS ILLEGAL BEFORE CONTRACT v2.
         * maddubs sums ADJACENT PAIRS, so lane l here holds elements
         * {4l..4l+3} of the sub-block where the widening kernel's lane l held
         * {l, l+8, l+16, l+24}. Under the OLD contract those eight lanes WERE
         * the contract -- they fed eight f32 accumulators in a pinned order --
         * so a kernel that partitioned them differently could not be
         * bit-exact, and that is precisely why A-as-originally-scoped was
         * cancelled. Under v2 the contract is the exact integer total S_b, and
         * integer addition is associative, so ANY partition reaching the same
         * S_b is exact BY CONSTRUCTION. Same value, fewer instructions.
         *
         * Sub-block j takes its 32 quants from q4[(j/2)*32 ...]: the LOW
         * nibbles when j is even, the HIGH nibbles when j is odd -- exactly
         * the order the scalar's aux8 is filled in, so scales8 still runs
         * 0..7 straight down.
         *
         * NO SATURATION, computed not assumed: `vpmaddubsw` saturates its i16
         * result, and the largest pair here is 2 x 15 x 127 = 3,810 against an
         * i16 ceiling of 32,767. (This is why the same trick needs care for
         * Q8_0, where a pair reaches 2 x 127 x 127 and does saturate.) The
         * `vpmaddwd` widening then tops out at 7,620 per i32 lane, and after
         * the scale (<=63) and eight sub-blocks the lane holds at most
         * 3,840,480 -- three orders inside i32. */
        for (u32 j = 0; j < 8u; ++j) {
            const u8* qsrc = q4 + (j / 2u) * 32u;
            __m256i qb =
                _mm256_loadu_si256((const __m256i*)(const void*)qsrc);
            /* No byte-granular shift exists, so shift 16-bit lanes and mask:
             * for each byte pair the low byte becomes b0>>4 with b1's low
             * nibble above it, and the mask discards exactly that. */
            __m256i wv = (j & 1u)
                             ? _mm256_and_si256(_mm256_srli_epi16(qb, 4),
                                                lownib8)
                             : _mm256_and_si256(qb, lownib8);
            __m256i av = _mm256_loadu_si256(
                (const __m256i*)(const void*)(q8 + j * 32u));
            /* Unsigned nibbles x signed quants -- the argument order is load
             * bearing, maddubs treats its FIRST operand as unsigned.
             *
             * ▶ The scale is folded into the pair-sum, as in the xn twin and as
             * the Q6_K maddubs kernel already did: one `madd_epi16` against a
             * broadcast scale replaces `madd_epi16(...,ones16)` plus a 2-uop
             * `vpmulld`. scale*(p[2m]+p[2m+1]) == scale*p[2m] + scale*p[2m+1]
             * exactly in integers, and neither form overflows -- nibbles 0..15
             * against |q8| <= 127 give a maddubs pair <= 3810 (no i16
             * saturation), a 6-bit scale takes each product to <= 240k and each
             * pair-sum to <= 480k, both inside i32.
             *
             * ⚠ THIS IS THE DECODE PATH. vec.c routes n_seq==1 here, so it is
             * what every generated token runs through, while the xn twin is
             * prefill's. */
            acc = _mm256_add_epi32(
                acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wv, av),
                                       _mm256_set1_epi16((short)scales8[j])));
        }

        u16 draw, dmraw;
        memcpy(&draw, blk, sizeof draw);
        memcpy(&dmraw, blk + 2, sizeof dmraw);
        /* The two CONTRACT lines, verbatim from the scalar. */
        const i32 S_b = kq_hsum_i32(acc);
        const f32 d = f16_to_f32(draw) * ad;
        const f32 dmin = f16_to_f32(dmraw) * ad;
        sumf = sumf + d * (f32)S_b;
        sumf = sumf - dmin * (f32)sumi;
    }
    return sumf;
}

void vec_dot_q4_k_q8_k_xn_avx2(const void* wrow, const void* const* arows,
                               u32 n_seq, u64 n, f32* out) {
    if (n % KERNELS_Q4_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q4_k_q8_k_xn_avx2 -- n=%llu is not a multiple "
              "of %u",
              (unsigned long long)n, KERNELS_Q4_K_BLOCK_ELEMS);
    }
    if (n_seq == 0u || n_seq > KERNELS_BATCH_TILE) {
        fatal("kernels: vec_dot_q4_k_q8_k_xn_avx2 -- n_seq=%u outside 1..%u; "
              "the caller tiles", n_seq, KERNELS_BATCH_TILE);
    }
    const u8* wb = (const u8*)wrow;
    u64 nb = n / KERNELS_Q4_K_BLOCK_ELEMS;

    const __m256i lownib8 = _mm256_set1_epi8(0xF);
#ifdef STOKE_MUT_NO_BOOKKEEP
    /* ▶▶ MUTANT, SPEED PROBE ONLY -- THE OUTPUT IS DELIBERATELY WRONG, in the
     * same family as -DSTOKE_MUT_ALIAS_ACT and built only by an explicit -D.
     *
     * WHAT IT ASKS. `objdump` on the shipping kernel says the per-sequence loop
     * body is 55 instructions, of which 23 are the dot product (8 vpmaddubsw,
     * 8 vpmaddwd, 7 vpaddd) and 32 are BOOKKEEPING: two 7-uop horizontal sums,
     * a 10-uop scalar-float epilogue, the mins load+vpmaddwd, and the loop's
     * own pointer arithmetic. If that reading is right, deleting the
     * bookkeeping should buy ~1.9x on the matmuls. If it is not, the dot loop
     * itself is the wall and no reduction-batching rewrite can help.
     *
     * ⚠ The per-column SCALE is deliberately NOT stripped, which is what the
     * previous commit said the next probe would do: GCC already hoists the
     * eight vpbroadcastw out of the sequence loop and folds the reload into
     * vpmaddwd's memory operand, so that probe would have measured the removal
     * of a cost that is not being paid. Reading the disassembly cost ten
     * minutes and moved the target. */
    __m256i vacc[KERNELS_BATCH_TILE];
    for (u32 s = 0; s < n_seq; ++s) vacc[s] = _mm256_setzero_si256();
#else
    f32 sumf[KERNELS_BATCH_TILE];
    for (u32 s = 0; s < n_seq; ++s) sumf[s] = 0.0f;
#endif

    /* ▶ THE HOIST GOT FOUR TIMES SMALLER WHEN THE INNER LOOP BECAME MADDUBS.
     * Rung B hoisted the nibbles WIDENED to i32 -- 8 sub-blocks x 4 groups x
     * 32 B = 1 KiB. maddubs consumes the nibbles as BYTES, so the same hoist
     * is now 8 x 32 B = 256 B. That direction matters: rung B measured that
     * over-hoisting Q6_K (materialising a 1 KiB __m256i wv[32] whose values
     * were already ready) made N=16 SLOWER, 13.565 -> 11.207, because it
     * quadrupled the bytes re-read per sequence. Here the bytes re-read per
     * sequence drop by the same factor. */
    __m256i wnib[8];

    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q4_K_BLOCK_BYTES;
        const u8* q4 = blk + 16;

        /* ── WEIGHT-ONLY WORK, ONCE PER ROW instead of once per sequence.
         * This is the whole of rung B, now carrying rung C's byte layout. */
        u8 scales8[8], mins8[8];
        kernels_q4_k_unpack_scales(blk + 4, scales8, mins8);
        for (u32 j = 0; j < 8u; ++j) {
            const u8* qsrc = q4 + (j / 2u) * 32u;
            __m256i qb =
                _mm256_loadu_si256((const __m256i*)(const void*)qsrc);
            wnib[j] = (j & 1u)
                          ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), lownib8)
                          : _mm256_and_si256(qb, lownib8);
        }
        u16 draw, dmraw;
        memcpy(&draw, blk, sizeof draw);
        memcpy(&dmraw, blk + 2, sizeof dmraw);
        const f32 dw = f16_to_f32(draw);
        const f32 dm = f16_to_f32(dmraw);
#if STOKE_MINS_SIMD
        /* ── HOISTED, because `mins8` IS WEIGHT-ONLY. The duplicated vector
         * depends on nothing per-sequence, so building it inside the sequence
         * loop rebuilt an identical value N times -- and this loop already
         * holds `wnib[8]`, half the register file, live across it, so the extra
         * live vectors were a plausible way to CAUSE spills while removing
         * uops. [m0,m0,m1,m1,...] so vpmaddwd's pair-sum is exactly
         * mins8[k]*(bs[2k]+bs[2k+1]). */
        static const i8 dup[16] = {0, 0, 1, 1, 2, 2, 3, 3,
                                   4, 4, 5, 5, 6, 6, 7, 7};
        const __m256i m16 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(
            _mm_loadl_epi64((const __m128i*)(const void*)mins8),
            _mm_loadu_si128((const __m128i*)(const void*)dup)));
#endif
#ifdef STOKE_MUT_NO_BOOKKEEP
        /* The weight-only halves of the deleted bookkeeping stay COMPUTED --
         * they are once per block, not per sequence, so removing them would
         * confound the probe with a cost it is not asking about. */
        (void)dw;
        (void)dm;
#if STOKE_MINS_SIMD
        (void)m16;
#endif
#endif

        u32 s = 0;
#if STOKE_MINS_SIMD && !defined(STOKE_MUT_NO_BOOKKEEP)
        /* ▶▶▶ FOUR SEQUENCES AT A TIME, AND WHAT IT IS FOR. The dot product
         * itself is unchanged and un-regrouped -- `kq4_dot_block` is the same
         * eight instructions in the same order the loop below runs. What the
         * grouping removes is the part of the body that is NOT the dot:
         * `objdump` on the 2026-08-30 kernel counts 55 instructions per
         * (super-block, sequence), of which 23 are the dot and 32 are
         * bookkeeping. Two horizontal sums become one `kq_hsum4_i32` each (14
         * instructions -> 22 per FOUR sequences), and the ten-instruction
         * scalar float epilogue becomes one four-wide one.
         *
         * ⚠⚠ BIT-EXACT, AND THE TWO HALVES ARE EXACT FOR DIFFERENT REASONS.
         * The reductions are INTEGER, so associativity makes the tree shape
         * free -- that is CONTRACT v2, stated at kq_hsum_i32 and relied on here
         * rather than re-derived. The float fold is exact because it is not
         * reassociated at all: lane k executes `sumf + (dw*ad)*S` then
         * `- (dm*ad)*M`, the same operations on the same values in the same
         * order as the scalar statement below, and `-ffp-contract=off` is
         * pinned globally so neither spelling can grow an FMA the other lacks.
         * Four lanes of the identical computation, not a wider accumulator.
         *
         * ⚠ The MINS group is reduced before the DOT group is built, not
         * alongside it: eight live accumulators plus `wnib[8]` would be the
         * whole register file, and rung B measured that over-hoisting this
         * kernel's sibling made N=16 SLOWER. Peak liveness here is four.
         *
         * The remainder falls through to the loop below -- production tiles at
         * KERNELS_BATCH_TILE=16, so the remainder is the prefill chunk's tail
         * and decode's n_seq=1, neither of which is where the time is. */
        const __m128 dwv = _mm_set1_ps(dw);
        const __m128 dmv = _mm_set1_ps(dm);
        for (; s + 4u <= n_seq; s += 4u) {
            const u8* ab[4];
            f32       d0, d1, d2, d3;
            for (u32 k = 0; k < 4u; ++k) {
                ab[k] = (const u8*)arows[s + k] + i * KERNELS_Q8_K_BLOCK_BYTES;
            }
            /* Four separate scalars rather than an `f32[4]` written by memcpy
             * and read back with `_mm_loadu_ps`: gcc turned that array into a
             * GPR pack -- 19 `mov`, 4 `or`, 2 `shl` per group building four
             * floats it already had in memory. `_mm_setr_ps` of four scalars
             * gets movss/unpcklps, which is what the value is. */
            memcpy(&d0, ab[0], sizeof d0);
            memcpy(&d1, ab[1], sizeof d1);
            memcpy(&d2, ab[2], sizeof d2);
            memcpy(&d3, ab[3], sizeof d3);
            const __m128i M4 = kq_hsum4_i32(
                _mm256_madd_epi16(m16, _mm256_loadu_si256(
                    (const __m256i*)(const void*)(ab[0] + 4 +
                                                  KERNELS_Q8_K_BLOCK_ELEMS))),
                _mm256_madd_epi16(m16, _mm256_loadu_si256(
                    (const __m256i*)(const void*)(ab[1] + 4 +
                                                  KERNELS_Q8_K_BLOCK_ELEMS))),
                _mm256_madd_epi16(m16, _mm256_loadu_si256(
                    (const __m256i*)(const void*)(ab[2] + 4 +
                                                  KERNELS_Q8_K_BLOCK_ELEMS))),
                _mm256_madd_epi16(m16, _mm256_loadu_si256(
                    (const __m256i*)(const void*)(ab[3] + 4 +
                                                  KERNELS_Q8_K_BLOCK_ELEMS))));
            /* ⚠⚠ THE SUB-BLOCK LOOP IS OUTSIDE THE SEQUENCE LOOP, AND THAT
             * NEST IS THE WHOLE POINT -- the first spelling of this ran
             * `kq4_dot_block` four times, which keeps FOUR accumulators live
             * across `wnib[8]` and spilled: `objdump` counted 66 `vmovdqu` in
             * the group body, 57 instructions per sequence against the 55 it
             * was replacing, and it measured 1.06x when the ablation said 1.32x
             * was available. Here only ONE weight vector and ONE scale are live
             * at a time and they are shared by all four sequences, so peak
             * liveness is four accumulators plus two -- no spill, and the
             * weight-side loads amortize four ways instead of one.
             *
             * Per sequence this is the same eight sub-blocks in the same
             * ascending order accumulating into the same lanes as the loop
             * below; `test_matvec_batch` holds the two to bit-identity against
             * the untouched solo kernel at n=4 and n=5, which is what makes the
             * duplication safe to have. */
            const i8* q8g[4];
            for (u32 k = 0; k < 4u; ++k) q8g[k] = (const i8*)(ab[k] + 4);
            __m256i a0 = _mm256_setzero_si256();
            __m256i a1 = _mm256_setzero_si256();
            __m256i a2 = _mm256_setzero_si256();
            __m256i a3 = _mm256_setzero_si256();
            /* ⚠ THE PRAGMA IS LOAD BEARING, and it is here because `objdump`
             * said so. Fully unrolled, gcc REASSOCIATES the four integer
             * accumulator chains into trees -- it is entitled to, integer
             * addition being associative -- which needs all 32 products live
             * at once and spilled every one: 32 `vmovdqu` stores to the stack
             * per group, 8 per sequence, on top of the work. Keeping the loop
             * rolled keeps four accumulators and one weight vector live and
             * costs one compare and branch per sub-block, shared by four
             * sequences. */
            for (u32 j = 0; j < 8u; ++j) {
                const __m256i w  = wnib[j];
                const __m256i sc = _mm256_set1_epi16((short)scales8[j]);
                a0 = _mm256_add_epi32(
                    a0, _mm256_madd_epi16(
                            _mm256_maddubs_epi16(
                                w, _mm256_loadu_si256(
                                       (const __m256i*)(const void*)(q8g[0] +
                                                                     j * 32u))),
                            sc));
                a1 = _mm256_add_epi32(
                    a1, _mm256_madd_epi16(
                            _mm256_maddubs_epi16(
                                w, _mm256_loadu_si256(
                                       (const __m256i*)(const void*)(q8g[1] +
                                                                     j * 32u))),
                            sc));
                a2 = _mm256_add_epi32(
                    a2, _mm256_madd_epi16(
                            _mm256_maddubs_epi16(
                                w, _mm256_loadu_si256(
                                       (const __m256i*)(const void*)(q8g[2] +
                                                                     j * 32u))),
                            sc));
                a3 = _mm256_add_epi32(
                    a3, _mm256_madd_epi16(
                            _mm256_maddubs_epi16(
                                w, _mm256_loadu_si256(
                                       (const __m256i*)(const void*)(q8g[3] +
                                                                     j * 32u))),
                            sc));
            }
            const __m128i S4 = kq_hsum4_i32(a0, a1, a2, a3);
            const __m128 a4 = _mm_setr_ps(d0, d1, d2, d3);
            __m128       v  = _mm_loadu_ps(sumf + s);
            v = _mm_add_ps(v, _mm_mul_ps(_mm_mul_ps(dwv, a4),
                                         _mm_cvtepi32_ps(S4)));
            v = _mm_sub_ps(v, _mm_mul_ps(_mm_mul_ps(dmv, a4),
                                         _mm_cvtepi32_ps(M4)));
            _mm_storeu_ps(sumf + s, v);
        }
#endif

        /* ── PER-SEQUENCE WORK, byte-for-byte the solo kernel's, in its
         * order. Nothing below regroups anything; it only runs N times. */
        for (; s < n_seq; ++s) {
            const u8* ablk = (const u8*)arows[s] + i * KERNELS_Q8_K_BLOCK_BYTES;
            const i8* q8 = (const i8*)(ablk + 4);
            const u8* bsums = ablk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
            f32 ad;
            memcpy(&ad, ablk, sizeof ad);

            /* ── THE MINS TERM. Two spellings of ONE integer sum, and the
             * SIMD one is exact by construction rather than by testing.
             *
             * WHY IT IS HERE (2026-08-20). Gate [2]'s attribution measured the
             * prefill gap to llama.cpp at ~3.2x AT EVERY THREAD COUNT, with
             * stoke's thread scaling (4.54x at t=8) BETTER than llama's (4.30x)
             * -- so the gap is per-thread instruction efficiency, not the pool.
             * `objdump` on this function then found the scalar spelling below
             * fully unrolled and NOT vectorized by GCC: 16 `movswl` + 8 `add` +
             * 8 `imul` + 8 `add` is ~40 uops beside the ~57 the actual vector
             * dot product costs. Roughly 40-45% of the emitted per-sequence
             * work was a loop that is not the dot product.
             *
             * WHY REGROUPING IT IS BIT-EXACT, not merely tested to be. Every
             * term is an integer, integer addition is associative, and nothing
             * overflows: `bsums[j]` sums 16 quants each clamped to +-128 so
             * |bs| <= 2048; a pair is <= 4096 against an i16 ceiling of 32767;
             * times a 6-bit min (<= 63) is <= 258048, and `vpmaddwd` sums two
             * of those to <= 516096 against an i32 ceiling. That is `simd.c`'s
             * founding principle (see kq_hsum_i32 above) applied to the one
             * term in this kernel that had never been given it.
             *
             * AND GCC ALREADY PROVED THE REGROUPING SOUND. The disassembly
             * shows `add %edx,%r15d` feeding a SINGLE `imul` -- the compiler
             * noticed mins8[j/2] is shared by j=2k and j=2k+1 and folded
             * bs[2k]+bs[2k+1] before multiplying. The SIMD form computes the
             * identical pairing, which is why the two spellings agree bit for
             * bit rather than merely closely. */
#ifndef STOKE_MUT_NO_BOOKKEEP
            i32 sumi;
#if STOKE_MINS_SIMD
            /* `m16` is hoisted above with the weight-only work; all that is
             * per-sequence here is the bsums load, one vpmaddwd and the
             * reduction. */
            sumi = kq_hsum_i32(_mm256_madd_epi16(
                m16, _mm256_loadu_si256((const __m256i*)(const void*)bsums)));
#else
            sumi = 0;
            for (u32 j = 0; j < KERNELS_Q8_K_BLOCK_ELEMS / 16u; ++j) {
                i16 bs;
                memcpy(&bs, bsums + 2u * j, sizeof bs);
                sumi += (i32)bs * (i32)mins8[j / 2u];
            }
#endif
#endif

            /* ▶▶▶ THE SCALE IS FOLDED INTO THE PAIR-SUM, WHICH DELETES A
             * vpmulld FROM THE INNER LOOP. It used to pair-sum with `ones16`
             * and then multiply the i32 result by the scale --
             * `_mm256_mullo_epi32` is 2 uops with ~10-cycle latency and was the
             * most expensive instruction here. `madd_epi16` against a broadcast
             * SCALE does the pair-sum and the multiply in one 1-uop
             * instruction.
             *
             * ⚠ BIT-EXACT BY ARITHMETIC, NOT BY TESTING -- the same standard
             * the mins term above is held to. Old: scale*(p[2m] + p[2m+1]).
             * New: scale*p[2m] + scale*p[2m+1]. Equal by distributivity over
             * the INTEGERS, and neither form overflows: the nibbles are 0..15
             * and the activations |q8| <= 127, so a maddubs pair is <= 3810
             * (no i16 saturation); a 6-bit scale <= 63 makes each product
             * <= 240k and each pair-sum <= 480k, both inside i32.
             *
             * The broadcast stays INSIDE the loop rather than being hoisted
             * with the other weight-only work: eight more live __m256i beside
             * `wnib[8]` would be the whole register file, and rung B measured
             * that over-hoisting this kernel's siblings made N=16 SLOWER.
             *
             * The eight are `kq4_dot_block`, CALLED rather than restated, so
             * the grouped path above and this one cannot drift apart in a
             * kernel where a one-instruction divergence is a silent wrong
             * answer. */
            __m256i acc = kq4_dot_block(wnib, scales8, q8);
#ifdef STOKE_MUT_NO_BOOKKEEP
            /* The ONLY thing left per sequence: fold the block's raw i32 lanes
             * into a running vector. The dot loop above is untouched, and it
             * cannot be optimised away because `vacc` reaches `out`. */
            vacc[s] = _mm256_add_epi32(vacc[s], acc);
            (void)ad;
            (void)bsums;
#else
            const i32 S_b = kq_hsum_i32(acc);
            sumf[s] = sumf[s] + (dw * ad) * (f32)S_b;
            sumf[s] = sumf[s] - (dm * ad) * (f32)sumi;
#endif
        }
    }

#ifdef STOKE_MUT_NO_BOOKKEEP
    for (u32 s = 0; s < n_seq; ++s) out[s] = (f32)kq_hsum_i32(vacc[s]);
#else
    for (u32 s = 0; s < n_seq; ++s) out[s] = sumf[s];
#endif
}

f32 vec_dot_q6_k_q8_k_avx2(const void* wrow, const void* arow, u64 n) {
    if (n % KERNELS_Q6_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q6_k_q8_k_avx2 -- n=%llu is not a multiple "
              "of %u",
              (unsigned long long)n, KERNELS_Q6_K_BLOCK_ELEMS);
    }
    const u8* wb = (const u8*)wrow;
    const u8* ab = (const u8*)arow;
    u64 nb = n / KERNELS_Q6_K_BLOCK_ELEMS;

    f32 sumf = 0.0f;

    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q6_K_BLOCK_BYTES;
        const u8* ql = blk;
        const u8* qh = blk + 128;
        const i8* sc = (const i8*)(blk + 192);
        const u8* ablk = ab + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        const u8* bsums = ablk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
        f32 ad;
        memcpy(&ad, ablk, sizeof ad);

        /* ▶▶ Q6_K ON MADDUBS (2026-08-15). `vpmaddubsw` needs UNSIGNED bytes
         * against signed ones, and Q6_K's quants are CENTRED to -32..+31,
         * which is why this port was deferred when Q4_K got one. But the
         * centring is a subtraction, and subtraction distributes:
         *
         *   S_b = sum_j sc_j * sum_k (q6_k - 32) * q8_k
         *       = sum_j sc_j * [ sum_k q6_k*q8_k  -  32 * sum_k q8_k ]
         *
         * with q6_k the RAW UNSIGNED 0..63 value. The first term is exactly
         * what maddubs computes; the second is 32 times the sub-block's
         * activation sum.
         *
         * ▶ AND THAT SUM IS ALREADY COMPUTED AND CURRENTLY THROWN AWAY.
         * Q6_K has 16 sub-blocks of 16 elements; Q8_K's bsums has 16 entries
         * each summing 16 elements. IDENTICAL granularity, no regrouping --
         * and until now this kernel never touched bsums at all.
         *
         * ⚠ NOT A STREAM CHANGE. S_b is an exact INTEGER total, so reaching
         * it by this identity gives the same VALUE, and the f32 fold below is
         * untouched. Bit-exactness holds by the same CONTRACT v2 argument the
         * Q4_K maddubs port used, and test_simd asserts it against the scalar
         * rather than trusting this note. The scalar is deliberately NOT
         * ported: it already computes the exact S_b, and a twin reaching it
         * differently is precisely what v2 exists to permit.
         *
         * NO SATURATION, computed: a maddubs pair is at most 2*63*127 =
         * 16,002 against an i16 ceiling of 32,767 -- tighter than Q4_K's
         * 3,810, and still half the headroom spare. */
        u8 raw6[KERNELS_Q6_K_BLOCK_ELEMS];
        for (u32 h = 0; h < 2u; ++h) {
            const u8* q4h = ql + h * 64u;
            const u8* qhh = qh + h * 32u;
            u8* a = raw6 + h * 128u;
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = (u8)((q4h[l + 0] & 0xF) | (((qhh[l] >> 0) & 3) << 4));
                a[l + 32] = (u8)((q4h[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4));
                a[l + 64] = (u8)((q4h[l + 0] >> 4) | (((qhh[l] >> 4) & 3) << 4));
                a[l + 96] = (u8)((q4h[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4));
            }
        }

        /* The centring correction, sixteen integer terms at exactly the
         * sub-block granularity bsums already uses. */
        i32 corr = 0;
        for (u32 j = 0; j < 16u; ++j) {
            i16 bs;
            memcpy(&bs, bsums + 2u * j, sizeof bs);
            corr += (i32)sc[j] * (i32)bs;
        }

        __m256i acc = _mm256_setzero_si256();
        /* TWO sub-blocks per pass: 16 elements is half a ymm, so pairing them
         * fills the register and one maddubs retires 32 products. The two
         * halves carry different scales, which is why the multiplier is a
         * built vector rather than a broadcast: vpmaddwd keeps i32 lanes 0..3
         * over bytes 0..15 (sub-block j) and 4..7 over bytes 16..31 (j+1). */
        for (u32 j = 0; j < 16u; j += 2u) {
            __m256i wv = _mm256_loadu_si256(
                (const __m256i*)(const void*)(raw6 + j * 16u));
            __m256i av = _mm256_loadu_si256(
                (const __m256i*)(const void*)(q8 + j * 16u));
            /* ▶ THE SCALE FOLDED INTO THE PAIR-SUM, as in the Q4_K twin: one
             * `madd_epi16` replaces `madd_epi16(...,ones16)` + `mullo_epi32`,
             * deleting a 2-uop vpmulld per pass. The multiplier is BUILT
             * rather than broadcast because the two halves carry different
             * scales -- maddubs puts sub-block j in i16 lanes 0..7 and j+1 in
             * 8..15, so an i16 vector of [sc[j] x8, sc[j+1] x8] reproduces the
             * i32 lane layout the old `sv` had exactly.
             *
             * ⚠ BIT-EXACT BY ARITHMETIC: scale*(p[2m]+p[2m+1]) becomes
             * scale*p[2m] + scale*p[2m+1], equal over the integers. No
             * overflow: 6-bit weights (<=63) against |q8| <= 127 give a
             * maddubs pair <= 16002 (inside i16, so no saturation), an i8
             * scale takes each product to <= 2.03M and each pair-sum to
             * <= 4.06M, both inside i32. */
            __m256i sv16 = _mm256_set_m128i(_mm_set1_epi16((short)sc[j + 1]),
                                            _mm_set1_epi16((short)sc[j]));
            acc = _mm256_add_epi32(
                acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wv, av), sv16));
        }

        u16 draw;
        memcpy(&draw, blk + 208, sizeof draw);
        /* The CONTRACT line, verbatim from the scalar -- with the centring
         * folded back in as exact integer arithmetic before it. */
        const i32 S_b = kq_hsum_i32(acc) - 32 * corr;
        sumf = sumf + (f16_to_f32(draw) * ad) * (f32)S_b;
    }
    return sumf;
}

void vec_dot_q6_k_q8_k_xn_avx2(const void* wrow, const void* const* arows,
                               u32 n_seq, u64 n, f32* out) {
    if (n % KERNELS_Q6_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q6_k_q8_k_xn_avx2 -- n=%llu is not a multiple "
              "of %u",
              (unsigned long long)n, KERNELS_Q6_K_BLOCK_ELEMS);
    }
    if (n_seq == 0u || n_seq > KERNELS_BATCH_TILE) {
        fatal("kernels: vec_dot_q6_k_q8_k_xn_avx2 -- n_seq=%u outside 1..%u; "
              "the caller tiles", n_seq, KERNELS_BATCH_TILE);
    }
    const u8* wb = (const u8*)wrow;
    u64 nb = n / KERNELS_Q6_K_BLOCK_ELEMS;

    f32 sumf[KERNELS_BATCH_TILE];
    for (u32 s = 0; s < n_seq; ++s) sumf[s] = 0.0f;

    /* ⚠ MEASURED, NOT ASSUMED: hoisting the WIDENING too (a __m256i wv[32]
     * alongside this) made N=16 SLOWER -- 13.565 -> 11.207 tok/s. Q4_K's
     * widening masks and shifts nibbles, which is arithmetic worth hoisting;
     * Q6_K's aux8 already holds ready i8 values, so `kq_load_i8x8_to_i32` is
     * a load-and-convert and materialising it quadruples the bytes re-read
     * per sequence (256 -> 1024) to save nothing. Only the interleaved
     * rebuild is hoisted here. The two K-quants do NOT want the same hoist. */
    i8 aux8[KERNELS_Q6_K_BLOCK_ELEMS];

    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q6_K_BLOCK_BYTES;
        const u8* ql = blk;
        const u8* qh = blk + 128;
        const i8* sc = (const i8*)(blk + 192);

        /* ── WEIGHT-ONLY WORK, ONCE PER ROW. Q6_K gains more from the hoist
         * than Q4_K does: the whole interleaved aux8 rebuild below used to
         * run once per sequence, and it is the reason the scalar K-quant
         * path was 3.6x slower than Q8_0 on a smaller file. */
        for (u32 h = 0; h < 2u; ++h) {
            const u8* q4h = ql + h * 64u;
            const u8* qhh = qh + h * 32u;
            i8* a = aux8 + h * 128u;
            for (int l = 0; l < 32; ++l) {
                a[l + 0] = (i8)((q4h[l + 0] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (i8)((q4h[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (i8)((q4h[l + 0] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (i8)((q4h[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
            }
        }
        u16 draw;
        memcpy(&draw, blk + 208, sizeof draw);
        const f32 dw = f16_to_f32(draw);

        /* ── PER-SEQUENCE WORK, the solo kernel's operations in its order. */
        for (u32 s = 0; s < n_seq; ++s) {
            const u8* ablk = (const u8*)arows[s] + i * KERNELS_Q8_K_BLOCK_BYTES;
            const i8* q8 = (const i8*)(ablk + 4);
            f32 ad;
            memcpy(&ad, ablk, sizeof ad);

            __m256i acc = _mm256_setzero_si256();
            for (u32 j = 0; j < 16u; ++j) {
                __m256i sub = _mm256_setzero_si256();
                for (u32 g = 0; g < 2u; ++g) {
                    __m256i wv = kq_load_i8x8_to_i32(aux8 + j * 16u + g * 8u);
                    __m256i av = kq_load_i8x8_to_i32(q8 + j * 16u + g * 8u);
                    sub = kq_mac8(sub, wv, av);
                }
                acc = _mm256_add_epi32(
                    acc, _mm256_mullo_epi32(sub, _mm256_set1_epi32((int)sc[j])));
            }
            const i32 S_b = kq_hsum_i32(acc);
            sumf[s] = sumf[s] + (dw * ad) * (f32)S_b;
        }
    }

    for (u32 s = 0; s < n_seq; ++s) out[s] = sumf[s];
}

/* See simd.h for WHY this exists, why vpmaddubsw applies to a signed quant, and
 * the overflow bounds that make the regrouping bit-exact rather than merely
 * close. This file implements it and argues nothing the header already argued.
 *
 * The structure is deliberately Q4_K's, line for line where it can be: weight-
 * only work hoisted to once per row, then a per-sequence body that is nothing
 * but loads, one vpmaddubsw, one vpmaddwd and an accumulate. */
void vec_dot_q6_k_q8_k_xn_maddubs_avx2(const void* wrow,
                                       const void* const* arows, u32 n_seq,
                                       u64 n, f32* out) {
    if (n % KERNELS_Q6_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q6_k_q8_k_xn_maddubs_avx2 -- n=%llu is not a "
              "multiple of %u",
              (unsigned long long)n, KERNELS_Q6_K_BLOCK_ELEMS);
    }
    if (n_seq == 0u || n_seq > KERNELS_BATCH_TILE) {
        fatal("kernels: vec_dot_q6_k_q8_k_xn_maddubs_avx2 -- n_seq=%u outside "
              "1..%u; the caller tiles",
              n_seq, KERNELS_BATCH_TILE);
    }
    const u8* wb = (const u8*)wrow;
    u64       nb = n / KERNELS_Q6_K_BLOCK_ELEMS;

#ifdef STOKE_MUT_NO_BOOKKEEP
    /* The Q4_K twin's probe, applied here too: Q4_K_M carries Q6_K tensors and
     * the profiler's matmul total covers both, so mutating only one arm would
     * under-read the ceiling by whatever share Q6_K holds. */
    __m256i vacc[KERNELS_BATCH_TILE];
    for (u32 s = 0; s < n_seq; ++s) vacc[s] = _mm256_setzero_si256();
#else
    f32 sumf[KERNELS_BATCH_TILE];
    for (u32 s = 0; s < n_seq; ++s) sumf[s] = 0.0f;
#endif

    /* ▶ THE RAW 6-BIT VALUE, NOT THE OFFSET ONE. This is the only difference
     * from the twin's `aux8` rebuild: no `- 32`, because vpmaddubsw needs an
     * UNSIGNED first operand and the offset is carried by the bsums term below.
     * Element order is identical, so scale group j still covers [16j, 16j+16). */
    u8      auxu[KERNELS_Q6_K_BLOCK_ELEMS];
    __m256i scv[KERNELS_Q6_K_BLOCK_ELEMS / 32u]; /* [s_2c x8, s_2c+1 x8], i16 */

    for (u64 i = 0; i < nb; ++i) {
        const u8* blk = wb + i * KERNELS_Q6_K_BLOCK_BYTES;
        const u8* ql  = blk;
        const u8* qh  = blk + 128;
        const i8* sc  = (const i8*)(blk + 192);

        /* ── WEIGHT-ONLY WORK, ONCE PER ROW. */
        for (u32 h = 0; h < 2u; ++h) {
            const u8* q4h = ql + h * 64u;
            const u8* qhh = qh + h * 32u;
            u8*       a   = auxu + h * 128u;
            for (int l = 0; l < 32; ++l) {
                a[l + 0]  = (u8)((q4h[l + 0] & 0xF) | (((qhh[l] >> 0) & 3) << 4));
                a[l + 32] = (u8)((q4h[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4));
                a[l + 64] = (u8)((q4h[l + 0] >> 4) | (((qhh[l] >> 4) & 3) << 4));
                a[l + 96] = (u8)((q4h[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4));
            }
        }
        /* A chunk of 32 elements spans scale groups 2c and 2c+1, and within a
         * 128-bit lane every scale lane holds the same value -- see simd.h. */
        for (u32 c = 0; c < KERNELS_Q6_K_BLOCK_ELEMS / 32u; ++c) {
#ifdef STOKE_MUT_Q6K
            /* ▶▶ A DELIBERATE LANE-ORDER ERROR, BUILT ONLY VIA EXTRA_CFLAGS, AND
             * IT TARGETS THE ONE STRUCTURALLY NEW RISK IN THIS KERNEL. Q4_K's
             * eight scales line up with its eight 32-byte chunks exactly; Q6_K
             * has SIXTEEN, so a chunk spans TWO scale groups and the halves of
             * `scv[c]` must match the halves `vpmaddubsw` produced. Swapping
             * them is the mistake this formulation invites, it is invisible in
             * a diff that does not know the lane layout, and it leaves the run
             * ALIVE AND MERELY WRONG -- the class kv.h names as dangerous.
             *
             * It can never be in a shipping binary: nothing defines this macro,
             * the default build does not, and the banner in sched_open's
             * neighbour below announces a mutant on every run. */
            const i16 s0 = (i16)sc[2u * c + 1u];
            const i16 s1 = (i16)sc[2u * c];
#else
            const i16 s0 = (i16)sc[2u * c];
            const i16 s1 = (i16)sc[2u * c + 1u];
#endif
            scv[c] = _mm256_set_epi16(s1, s1, s1, s1, s1, s1, s1, s1,
                                      s0, s0, s0, s0, s0, s0, s0, s0);
        }
        /* The 16 scales as i16, for the bsums term. Weight-only, so hoisted. */
        const __m256i sc16 = _mm256_cvtepi8_epi16(
            _mm_loadu_si128((const __m128i*)(const void*)sc));

        u16 draw;
        memcpy(&draw, blk + 208, sizeof draw);
        const f32 dw = f16_to_f32(draw);

        /* ── PER-SEQUENCE WORK: two loads, one vpmaddubsw, one vpmaddwd and an
         * add, eight times -- against the twin's thirty-two widening loads. */
        for (u32 s = 0; s < n_seq; ++s) {
            const u8* ablk = (const u8*)arows[s] + i * KERNELS_Q8_K_BLOCK_BYTES;
            const i8* q8    = (const i8*)(ablk + 4);
            const u8* bsums = ablk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
            f32       ad;
            memcpy(&ad, ablk, sizeof ad);

            __m256i acc = _mm256_setzero_si256();
            for (u32 c = 0; c < KERNELS_Q6_K_BLOCK_ELEMS / 32u; ++c) {
                __m256i uv = _mm256_loadu_si256(
                    (const __m256i*)(const void*)(auxu + c * 32u));
                __m256i av = _mm256_loadu_si256(
                    (const __m256i*)(const void*)(q8 + c * 32u));
                /* UNSIGNED first, SIGNED second -- the argument order is load
                 * bearing, exactly as in the Q4_K twin. */
                acc = _mm256_add_epi32(
                    acc, _mm256_madd_epi16(_mm256_maddubs_epi16(uv, av),
                                           scv[c]));
            }
#ifdef STOKE_MUT_NO_BOOKKEEP
            vacc[s] = _mm256_add_epi32(vacc[s], acc);
            (void)ad;
            (void)bsums;
            (void)sc16;
            (void)dw;
#else
            /* SUM u*a*s */
            const i32 P = kq_hsum_i32(acc);
            /* SUM a*s over each scale group, straight out of the bsums. */
            const i32 M = kq_hsum_i32(_mm256_madd_epi16(
                sc16,
                _mm256_loadu_si256((const __m256i*)(const void*)bsums)));
            /* SUM (u-32)*a*s, the twin's S_b as the SAME INTEGER. */
            const i32 S_b = P - 32 * M;
            sumf[s] = sumf[s] + (dw * ad) * (f32)S_b;
#endif
        }
    }

#ifdef STOKE_MUT_NO_BOOKKEEP
    for (u32 s = 0; s < n_seq; ++s) out[s] = (f32)kq_hsum_i32(vacc[s]);
#else
    for (u32 s = 0; s < n_seq; ++s) out[s] = sumf[s];
#endif
}

#else /* !x86-64 */

bool kernels_cpu_has_avx_vnni(void) { return false; }

void dequant_row_q8_0_avx2(const void* row, u64 n, f32* out) {
    (void)row;
    (void)n;
    (void)out;
    fatal("kernels: dequant_row_q8_0_avx2 on a non-x86-64 build");
}

void vec_dot_q8_0_q8_0_x4_avx2(const void* const wrows[4], const void* arow,
                                u64 n, f32 out[4]) {
    (void)wrows;
    (void)arow;
    (void)n;
    (void)out;
    fatal("kernels: vec_dot_q8_0_q8_0_x4_avx2 on a non-x86-64 build");
}

u32 argmax_avx2(const f32* logits, u64 n) {
    (void)logits;
    (void)n;
    fatal("kernels: argmax_avx2 on a non-x86-64 build");
    return 0u;
}

f32 vec_dot_q8_0_q8_0_avx2(const void* wrow, const void* arow, u64 n) {
    (void)wrow;
    (void)arow;
    (void)n;
    fatal("kernels: vec_dot_q8_0_q8_0_avx2 called on a non-x86-64 build -- "
          "callers must check kernels_cpu_has_avx_vnni() first");
    return 0.0f;
}

f32 vec_dot_q4_k_q8_k_avx2(const void* wrow, const void* arow, u64 n) {
    (void)wrow;
    (void)arow;
    (void)n;
    fatal("kernels: vec_dot_q4_k_q8_k_avx2 on a non-x86-64 build -- callers "
          "must check kernels_cpu_has_avx_vnni() first");
    return 0.0f;
}

f32 vec_dot_q6_k_q8_k_avx2(const void* wrow, const void* arow, u64 n) {
    (void)wrow;
    (void)arow;
    (void)n;
    fatal("kernels: vec_dot_q6_k_q8_k_avx2 on a non-x86-64 build -- callers "
          "must check kernels_cpu_has_avx_vnni() first");
    return 0.0f;
}

void vec_dot_q4_k_q8_k_xn_avx2(const void* wrow, const void* const* arows,
                               u32 n_seq, u64 n, f32* out) {
    (void)wrow;
    (void)arows;
    (void)n_seq;
    (void)n;
    (void)out;
    fatal("kernels: vec_dot_q4_k_q8_k_xn_avx2 on a non-x86-64 build -- "
          "callers must check kernels_cpu_has_avx_vnni() first");
}

void vec_dot_q6_k_q8_k_xn_avx2(const void* wrow, const void* const* arows,
                               u32 n_seq, u64 n, f32* out) {
    (void)wrow;
    (void)arows;
    (void)n_seq;
    (void)n;
    (void)out;
    fatal("kernels: vec_dot_q6_k_q8_k_xn_avx2 on a non-x86-64 build -- "
          "callers must check kernels_cpu_has_avx_vnni() first");
}


void vec_dot_q6_k_q8_k_xn_maddubs_avx2(const void* wrow,
                                       const void* const* arows, u32 n_seq,
                                       u64 n, f32* out) {
    (void)wrow;
    (void)arows;
    (void)n_seq;
    (void)n;
    (void)out;
    fatal("kernels: vec_dot_q6_k_q8_k_xn_maddubs_avx2 on a non-x86-64 build -- "
          "callers must check kernels_cpu_has_avx_vnni() first");
}

#endif
