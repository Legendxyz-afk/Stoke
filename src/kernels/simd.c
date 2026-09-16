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
 * kept so the A/B can be re-run and so a toolchain regression has a referent.
 *
 * ⚠⚠ THE DEFAULT LIVES IN simd.h, NOT HERE, AND THAT COST A BUILD TO LEARN
 * (2026-09-01). It used to be `#ifndef/#define` in this .c file, so vec.c --
 * which has to know, because the VNNI kernel has only the SIMD mins spelling
 * and must not be selectable when this arm is off -- saw the name UNDEFINED.
 * `#if !STOKE_MINS_SIMD` is then `#if !0`, i.e. ALWAYS TRUE, so the guard fired
 * in every build and switched the VNNI kernel silently off. A per-TU default
 * for a build-wide knob is not a default, it is two different knobs. */

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

/* ▶▶ THE PREAMBLE'S THREE CALLS, INLINED (2026-09-16). `objdump` on the
 * shipping VNNI kernel showed every (row, super-block) paying three
 * out-of-line calls -- `kernels_q4_k_unpack_scales` and `f16_to_f32` twice --
 * each behind a `vzeroupper`, with the eight unpacked nibble vectors stored to
 * the stack before the calls and reloaded after them. At N=8 (two groups per
 * row) that preamble was of the same order as the dot product it prepares.
 *
 * `kq_f16c` is `vcvtph2ps`: ONE uop, and exact for every input, because every
 * f16 value is representable in f32 -- normals, subnormals (the instruction
 * does not honour DAZ for its f16 source and the results are f32 normals, so
 * FTZ cannot touch them), zeros, infinities. A NaN keeps its payload; the one
 * difference from the software converter is that an sNaN arrives already
 * quiet, which the other spelling's first multiply does to it anyway, so the
 * fold's bits are the same either way (test_simd walks all 65536 inputs and
 * asserts exactly that relation). `kq_unpack_scales_inl` is kquant.c's
 * `kq_unpack_q4_k_scales`, byte for byte in its arithmetic; the same file's
 * pair-agreement test is extended to this copy. */
static inline f32 kq_f16c(u16 h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

static inline u32 kq_ld_u32(const u8* p) {
    u32 v;
    memcpy(&v, p, sizeof v);   /* x86-64 only: this file is, see line 17 */
    return v;
}

static inline void kq_unpack_scales_inl(const u8* packed, u8* scales8,
                                        u8* mins8) {
    const u32 kmask1 = 0x3f3f3f3fu;
    const u32 kmask2 = 0x0f0f0f0fu;
    const u32 kmask3 = 0x03030303u;
    u32 utmp0 = kq_ld_u32(packed + 0);
    u32 utmp1 = kq_ld_u32(packed + 4);
    u32 utmp2 = kq_ld_u32(packed + 8);
    const u32 utmp3 = ((utmp2 >> 4) & kmask2) | (((utmp1 >> 6) & kmask3) << 4);
    const u32 uaux  = utmp1 & kmask1;
    utmp1 = (utmp2 & kmask2) | (((utmp0 >> 6) & kmask3) << 4);
    utmp2 = uaux;
    utmp0 &= kmask1;
    for (u32 k = 0; k < 4u; k++) {
        scales8[k]      = (u8)((utmp0 >> (8u * k)) & 0xFFu);
        scales8[k + 4u] = (u8)((utmp1 >> (8u * k)) & 0xFFu);
        mins8[k]        = (u8)((utmp2 >> (8u * k)) & 0xFFu);
        mins8[k + 4u]   = (u8)((utmp3 >> (8u * k)) & 0xFFu);
    }
}

/* The two helpers above, exposed out of line FOR THE TESTS ONLY (simd.h). */
f32 kernels_f16_to_f32_f16c_probe(u16 h) { return kq_f16c(h); }
void kernels_q4_k_unpack_scales_inl_probe(const void* packed12, u8* scales8,
                                          u8* mins8) {
    kq_unpack_scales_inl((const u8*)packed12, scales8, mins8);
}

/* Mirrors q8_0.c's block_scale_or_fatal, including the message, so a corrupt
 * tensor fails identically whichever path ran. */
static f32 simd_block_scale_or_fatal(const u8* blk, u64 b, const char* who) {
    u16 raw;
    memcpy(&raw, blk, sizeof raw);
    f32 d = kq_f16c(raw);
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

/* ---- the two attention inner loops ----------------------------------------
 *
 * ▶▶ WHY THEY LIVE HERE. `att_kvheads_run` is in model.c, which the Makefile
 * deliberately compiles WITHOUT -mavx2 so gcc cannot emit AVX2 into the scalar
 * reference twins. Attention is 15% of prefill and BOTH of its passes are
 * limited by the same thing on SSE2: there is no single-uop
 * broadcast-from-memory, so `_mm_load1_ps` is movss+shufps and the score loop
 * ends up three quarters broadcast overhead. `vbroadcastss` is one uop.
 * MEASURED before either of these was written: -mavx2 on that file is worth
 * 1.06x against a scalar score pass and 1.218x against the vectorized one.
 *
 * ⚠⚠ BIT-EXACT AGAINST THE SSE2 SPELLING, for two different reasons.
 * `att_score4_avx2` STAYS 128-BIT ON PURPOSE: its four lanes are the GQA
 * group's four q-heads, so widening would mean either eight q-heads (there are
 * four) or two d per lane group -- and the latter sums even and odd d
 * separately and combines at the end, which is a reassociation. Only the
 * ENCODING changes, VEX for the cheap broadcast. `att_accum4_avx2` DOES widen
 * to 256 bits, because there the lanes are independent d's: each o[d] still
 * accumulates over j in ascending order whatever the vector width, so the lane
 * count is free.
 *
 * ⚠ -ffp-contract=off is pinned in the global CFLAGS, and that is what stops
 * this file's -mfma from fusing a mul+add here into an FMA the SSE2 arm cannot
 * spell. That flag is load bearing for these two functions.
 *
 * The caller gates on kernels_cpu_has_avx_vnni(), the same probe vec.c uses to
 * pick its AVX2 kernels, and keeps its SSE2 arm for everything else. */
void att_score4_avx2(const f32* qt, u64 qstride, const f32* kj, u32 hd,
                     f32 scale, f32* out4) {
    __m128 acc = _mm_setzero_ps();
    for (u32 d = 0; d < hd; ++d) {
        acc = _mm_add_ps(
            acc, _mm_mul_ps(_mm_loadu_ps(qt + (u64)d * qstride),
                            _mm_broadcast_ss(kj + d)));
    }
    _mm_storeu_ps(out4, _mm_mul_ps(acc, _mm_set1_ps(scale)));
}

void att_accum4_avx2(f32* o0, f32* o1, f32* o2, f32* o3, const f32* vj,
                     const f32* w4, u32 hd) {
    const __m256 b0 = _mm256_broadcast_ss(w4 + 0);
    const __m256 b1 = _mm256_broadcast_ss(w4 + 1);
    const __m256 b2 = _mm256_broadcast_ss(w4 + 2);
    const __m256 b3 = _mm256_broadcast_ss(w4 + 3);
    u32 d = 0;
#ifdef STOKE_MUT_NO_ACC
    /* MUTANT, SPEED PROBE ONLY -- OUTPUT DELIBERATELY WRONG. Stores instead of
     * accumulating, which deletes the LOAD half of the read-modify-write and
     * nothing else: same multiplies, same stores, same traffic in vj. It asks
     * whether re-reading the 512-byte output row for every j is what this pass
     * costs, before anyone rewrites it to hold that row in registers. */
    for (; d + 8u <= hd; d += 8u) {
        const __m256 v = _mm256_loadu_ps(vj + d);
        _mm256_storeu_ps(o0 + d, _mm256_mul_ps(b0, v));
        _mm256_storeu_ps(o1 + d, _mm256_mul_ps(b1, v));
        _mm256_storeu_ps(o2 + d, _mm256_mul_ps(b2, v));
        _mm256_storeu_ps(o3 + d, _mm256_mul_ps(b3, v));
    }
    for (; d < hd; ++d) {
        o0[d] = w4[0] * vj[d];
        o1[d] = w4[1] * vj[d];
        o2[d] = w4[2] * vj[d];
        o3[d] = w4[3] * vj[d];
    }
    return;
#endif
    for (; d + 8u <= hd; d += 8u) {
        const __m256 v = _mm256_loadu_ps(vj + d);
        _mm256_storeu_ps(o0 + d, _mm256_add_ps(_mm256_loadu_ps(o0 + d),
                                               _mm256_mul_ps(b0, v)));
        _mm256_storeu_ps(o1 + d, _mm256_add_ps(_mm256_loadu_ps(o1 + d),
                                               _mm256_mul_ps(b1, v)));
        _mm256_storeu_ps(o2 + d, _mm256_add_ps(_mm256_loadu_ps(o2 + d),
                                               _mm256_mul_ps(b2, v)));
        _mm256_storeu_ps(o3 + d, _mm256_add_ps(_mm256_loadu_ps(o3 + d),
                                               _mm256_mul_ps(b3, v)));
    }
    /* The tail is the same per-element operation, not a different one: for a
     * given d the value is o[d] + w*vj[d] whether it is computed in a lane or
     * in a register. Nothing crosses lanes here. */
    for (; d < hd; ++d) {
        o0[d] += w4[0] * vj[d];
        o1[d] += w4[1] * vj[d];
        o2[d] += w4[2] * vj[d];
        o3[d] += w4[3] * vj[d];
    }
}

/* ▶▶ FOUR KEYS AT ONCE, SAME ARITHMETIC PER KEY (2026-09-15).
 *
 * att_score4_avx2 is ONE dependent chain: 128 mul+add pairs on a single xmm
 * accumulator, ~512 cycles of add latency per key while the load ports sit
 * idle, and at decode every key's K row also comes from DRAM behind that
 * chain. The step trace priced it: an N=8 decode step reads 146 ms at a
 * 20-token context and 231 ms at ~600, so attention over ~600 keys is ~85 ms
 * of the step -- a third of it -- at ~3 GF/s per thread.
 *
 * This runs four keys' chains INTERLEAVED, one accumulator each. Each key's
 * accumulator sees exactly the sequence att_score4_avx2 would give it -- the
 * same q vector, the same broadcast of k[d], mul then add in ascending d --
 * so every output is BIT-IDENTICAL to four calls of the one-key kernel; the
 * only thing that changed is that the four chains' latencies overlap and the
 * four K rows stream in parallel. -ffp-contract=off keeps the mul+add
 * unfused here exactly as it does in the one-key kernel. Graded by
 * tests/att_simd_twins.sh (AVX2 arm vs the scalar twin, bit-identical) and by
 * the byte-exact goldens, never by tolerance. */
void att_score4x4_avx2(const f32* qt, u64 qstride, const f32* k0,
                       const f32* k1, const f32* k2, const f32* k3, u32 hd,
                       f32 scale, f32* out16) {
    __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
    __m128 a2 = _mm_setzero_ps(), a3 = _mm_setzero_ps();
    for (u32 d = 0; d < hd; ++d) {
        const __m128 q = _mm_loadu_ps(qt + (u64)d * qstride);
        a0 = _mm_add_ps(a0, _mm_mul_ps(q, _mm_broadcast_ss(k0 + d)));
        a1 = _mm_add_ps(a1, _mm_mul_ps(q, _mm_broadcast_ss(k1 + d)));
        a2 = _mm_add_ps(a2, _mm_mul_ps(q, _mm_broadcast_ss(k2 + d)));
        a3 = _mm_add_ps(a3, _mm_mul_ps(q, _mm_broadcast_ss(k3 + d)));
    }
    const __m128 s = _mm_set1_ps(scale);
    _mm_storeu_ps(out16 + 0, _mm_mul_ps(a0, s));
    _mm_storeu_ps(out16 + 4, _mm_mul_ps(a1, s));
    _mm_storeu_ps(out16 + 8, _mm_mul_ps(a2, s));
    _mm_storeu_ps(out16 + 12, _mm_mul_ps(a3, s));
}

/* ▶▶ EIGHT KEYS PER OUTPUT LOAD, SAME ARITHMETIC PER ELEMENT (2026-09-15).
 *
 * att_accum4_avx2 read-modify-writes the four 512-byte output rows once PER
 * KEY: for 600 keys that is 600 loads and 600 stores of every output lane
 * against 600 multiplies. Here the outputs stay in registers across eight
 * keys and touch memory once per eight. Per output element the operation
 * sequence is unchanged -- o = o + w_j * v_j[d] for j ascending, mul then
 * add, keys in order within the group and groups in order -- so the result is
 * BIT-IDENTICAL to eight calls of the one-key kernel. `v8` holds the eight V
 * rows, `w32[k*4 + g]` the eight keys' weights for the four outputs. */
void att_accum4x8_avx2(f32* o0, f32* o1, f32* o2, f32* o3,
                       const f32* const* v8, const f32* w32, u32 hd) {
    u32 d = 0;
    for (; d + 8u <= hd; d += 8u) {
        __m256 r0 = _mm256_loadu_ps(o0 + d), r1 = _mm256_loadu_ps(o1 + d);
        __m256 r2 = _mm256_loadu_ps(o2 + d), r3 = _mm256_loadu_ps(o3 + d);
        for (u32 k = 0; k < 8u; k++) {
            const __m256 v = _mm256_loadu_ps(v8[k] + d);
            r0 = _mm256_add_ps(r0, _mm256_mul_ps(_mm256_broadcast_ss(w32 + k * 4u + 0u), v));
            r1 = _mm256_add_ps(r1, _mm256_mul_ps(_mm256_broadcast_ss(w32 + k * 4u + 1u), v));
            r2 = _mm256_add_ps(r2, _mm256_mul_ps(_mm256_broadcast_ss(w32 + k * 4u + 2u), v));
            r3 = _mm256_add_ps(r3, _mm256_mul_ps(_mm256_broadcast_ss(w32 + k * 4u + 3u), v));
        }
        _mm256_storeu_ps(o0 + d, r0); _mm256_storeu_ps(o1 + d, r1);
        _mm256_storeu_ps(o2 + d, r2); _mm256_storeu_ps(o3 + d, r3);
    }
    /* Tail: the same per-element operation in the same key order. */
    for (; d < hd; ++d) {
        for (u32 k = 0; k < 8u; k++) {
            o0[d] += w32[k * 4u + 0u] * v8[k][d];
            o1[d] += w32[k * 4u + 1u] * v8[k][d];
            o2[d] += w32[k * 4u + 2u] * v8[k][d];
            o3[d] += w32[k * 4u + 3u] * v8[k][d];
        }
    }
}

/* An empty asm that CLAIMS the accumulator forces gcc to materialise it in a
 * register at this point, which stops it reassociating N accumulator chains
 * into trees. It emits nothing. Without it `objdump` counts 34 stack spills in
 * the Q4_K grouped kernel: the tree form needs all 32 products of an unrolled
 * sub-block loop live at once.
 *
 * ⚠ FILE SCOPE, and it was not always -- fixed 2026-09-01. It used to sit
 * inside `vec_dot_q4_k_q8_k_xn_avx2`, within that function's
 * `#if STOKE_MINS_SIMD` block, while the Q6_K grouped kernel below uses it
 * unconditionally. A `#define` is file-scoped, so Q6_K was silently depending
 * on the Q4_K grouped path being compiled in, and `-DSTOKE_MINS_SIMD=0` -- a
 * RETAINED A/B ARM -- had quietly stopped building at all. A knob that no
 * longer compiles is not a knob, it is a claim. */
#if defined(__GNUC__)
#define KQ_KEEP(v) __asm__("" : "+x"(v))
#else
#define KQ_KEEP(v) ((void)0)
#endif

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
        kq_unpack_scales_inl(blk + 4, scales8, mins8);

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
        const f32 d = kq_f16c(draw) * ad;
        const f32 dmin = kq_f16c(dmraw) * ad;
        sumf = sumf + d * (f32)S_b;
        sumf = sumf - dmin * (f32)sumi;
    }
    return sumf;
}

void vec_dot_q4_k_q8_k_xn_avx2(const void* wrows, u64 row_bytes, u32 n_rows,
                               const void* const* arows, u32 n_seq, u64 n,
                               f32* out) {
    if (n % KERNELS_Q4_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q4_k_q8_k_xn_avx2 -- n=%llu is not a multiple "
              "of %u",
              (unsigned long long)n, KERNELS_Q4_K_BLOCK_ELEMS);
    }
    if (n_seq == 0u || n_seq > KERNELS_BATCH_TILE) {
        fatal("kernels: vec_dot_q4_k_q8_k_xn_avx2 -- n_seq=%u outside 1..%u; "
              "the caller tiles", n_seq, KERNELS_BATCH_TILE);
    }
    if (n_rows == 0u || n_rows > KERNELS_ROW_TILE) {
        fatal("kernels: vec_dot_q4_k_q8_k_xn_avx2 -- n_rows=%u outside 1..%u; "
              "the caller tiles", n_rows, KERNELS_ROW_TILE);
    }
    const u8* wb = (const u8*)wrows;
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
    __m256i vacc[KERNELS_ROW_TILE * KERNELS_BATCH_TILE];
    for (u32 t = 0; t < n_rows * n_seq; ++t) vacc[t] = _mm256_setzero_si256();
#else
    f32 sumf[KERNELS_ROW_TILE * KERNELS_BATCH_TILE];
    for (u32 t = 0; t < n_rows * n_seq; ++t) sumf[t] = 0.0f;
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

    /* ⚠ SUPER-BLOCK OUTSIDE, ROW INSIDE — see simd.h's KERNELS_ROW_TILE note.
     * Every row in the tile consumes THE SAME super-block of activations
     * before the loop moves on, so the live activation is 16 * 292 B rather
     * than the whole row's worth. Nothing about a given (row, sequence)
     * accumulation changes: it still walks i in ascending order. */
    for (u64 i = 0; i < nb; ++i) {
      for (u32 rr = 0; rr < n_rows; ++rr) {
        const u8* blk = wb + (u64)rr * row_bytes + i * KERNELS_Q4_K_BLOCK_BYTES;
        const u8* q4 = blk + 16;
#ifdef STOKE_MUT_NO_BOOKKEEP
        __m256i* vacc_r = vacc + (size_t)rr * n_seq;
#else
        f32* sumf_r = sumf + (size_t)rr * n_seq;
#endif

        /* ── WEIGHT-ONLY WORK, ONCE PER ROW instead of once per sequence.
         * This is the whole of rung B, now carrying rung C's byte layout. */
        u8 scales8[8], mins8[8];
        kq_unpack_scales_inl(blk + 4, scales8, mins8);
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
        const f32 dw = kq_f16c(draw);
        const f32 dm = kq_f16c(dmraw);
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
#define KQ_ACT(k, jj) \
    _mm256_loadu_si256((const __m256i*)(const void*)(q8g[(k)] + (jj) * 32u))
#if defined(STOKE_MUT_G4_VNNI_CEIL) || defined(STOKE_MUT_G4_VNNI_B) || \
    defined(STOKE_MUT_G4_VNNI_A)
            /* ▶▶ MUTANTS, SPEED PROBES ONLY -- THE OUTPUT IS DELIBERATELY
             * WRONG (the per-sub-block scale is dropped or applied to the
             * wrong lanes), in the family of -DSTOKE_MUT_NO_BOOKKEEP. Built
             * only by an explicit -D, and vec.c's q4k_select announces them.
             *
             * WHAT THEY ASK. `objdump` on the shipping group loop counts 157
             * instructions per (super-block, row, FOUR sequences): 32
             * vpmaddubsw + 32 vpmaddwd + 28 vpaddd of dot, and the rest
             * reduction/epilogue/pointer work. The multiplies are the whole of
             * ports 0-1 -- 68 of them, so 34 cycles at 2/cycle -- while the
             * front end at 6 uops/cycle wants only 26. If that reading is
             * right the loop is PORT-BOUND ON THE MULTIPLIERS, and AVX-VNNI's
             * `vpdpbusd` retires 32 MACs in ONE p01 uop where maddubs+maddwd
             * take two.
             *
             * The recorded objection is that `vpdpbusd` cannot carry the
             * per-sub-block scale: it accumulates 4 byte-products per i32 lane
             * and there is nowhere to put a 6-bit multiplier. That was checked
             * on paper, never measured. These two probes bracket what the
             * answer is worth BEFORE any of the layout work that would be
             * needed to collect it:
             *
             *   _CEIL -- eight `vpdpbusd` straight into the four accumulators,
             *   no scale at all. 8 p01 uops per (super-block, sequence) where
             *   the shipping kernel spends 16. This is the ABSOLUTE ceiling:
             *   no scale-application scheme can beat it.
             *
             *   _B -- the instruction mix of the kernel that WOULD be exact if
             *   the activations and nibbles were permuted so that lane l of a
             *   `vpdpbusd` accumulator holds four elements of sub-block l:
             *   eight dpbusd into TWO accumulators (elements 0-15 and 16-31 of
             *   every sub-block, each bounded by 16*15*128 = 30,720 so
             *   `vpackssdw` cannot saturate), one pack, one `vpmaddwd`
             *   carrying all eight scales at once. 9 p01 uops. The lane
             *   semantics here are wrong -- the operands are NOT permuted, so
             *   the answer is garbage -- but every instruction, its port and
             *   its dependency are the ones that kernel would issue.
             *
             * ⚠ Read these UPSTREAM of the garbage: the profiler's matmul
             * bucket, never prefill wall time. Wrong activations cost ~2.1 s in
             * elementwise ops that no PROF bucket covers, which is how
             * -DSTOKE_MUT_NO_BOOKKEEP came back 1.22x on matmul and ~1.00x on
             * the wall. */
#ifdef STOKE_MUT_G4_VNNI_CEIL
            for (u32 j = 0; j < 8u; ++j) {
                const __m256i w = wnib[j];
                a0 = _mm256_dpbusd_avx_epi32(a0, w, KQ_ACT(0, j));
                a1 = _mm256_dpbusd_avx_epi32(a1, w, KQ_ACT(1, j));
                a2 = _mm256_dpbusd_avx_epi32(a2, w, KQ_ACT(2, j));
                a3 = _mm256_dpbusd_avx_epi32(a3, w, KQ_ACT(3, j));
                KQ_KEEP(a0);
                KQ_KEEP(a1);
                KQ_KEEP(a2);
                KQ_KEEP(a3);
            }
#elif defined(STOKE_MUT_G4_VNNI_A)
            /*   _A -- the mix of the kernel that needs NO LAYOUT CHANGE:
             * sub-blocks taken in PAIRS, two `vpdpbusd` into two zeroed
             * accumulators, `vpackssdw` to i16 (a lane holds at most
             * 4*15*128 = 7,680, so no saturation), one `vpmaddwd` carrying the
             * pair's two scales, one `vpaddd`. Four times per super-block
             * instead of once: 12 p01 uops against _B's 9 and the shipping
             * kernel's 16, and 8 zeroing idioms per (super-block, sequence)
             * that _B does not pay. If this reads like _B, the whole
             * permuted-layout project is unnecessary. */
            for (u32 j = 0; j < 8u; j += 2u) {
                const __m256i wl = wnib[j];
                const __m256i wh = wnib[j + 1u];
                __m256i t0, t1;
#define KQ_PAIR(acc, k)                                                       \
    t0 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), wl, KQ_ACT(k, j));   \
    t1 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), wh,                  \
                                 KQ_ACT(k, j + 1u));                          \
    acc = _mm256_add_epi32(                                                   \
        acc, _mm256_madd_epi16(_mm256_packs_epi32(t0, t1), m16));             \
    KQ_KEEP(acc)
                KQ_PAIR(a0, 0);
                KQ_PAIR(a1, 1);
                KQ_PAIR(a2, 2);
                KQ_PAIR(a3, 3);
#undef KQ_PAIR
            }
#else
            {
                __m256i b0 = _mm256_setzero_si256();
                __m256i b1 = _mm256_setzero_si256();
                __m256i b2 = _mm256_setzero_si256();
                __m256i b3 = _mm256_setzero_si256();
                for (u32 j = 0; j < 4u; ++j) {
                    const __m256i w = wnib[j];
                    a0 = _mm256_dpbusd_avx_epi32(a0, w, KQ_ACT(0, j));
                    a1 = _mm256_dpbusd_avx_epi32(a1, w, KQ_ACT(1, j));
                    a2 = _mm256_dpbusd_avx_epi32(a2, w, KQ_ACT(2, j));
                    a3 = _mm256_dpbusd_avx_epi32(a3, w, KQ_ACT(3, j));
                    KQ_KEEP(a0);
                    KQ_KEEP(a1);
                    KQ_KEEP(a2);
                    KQ_KEEP(a3);
                }
                for (u32 j = 4u; j < 8u; ++j) {
                    const __m256i w = wnib[j];
                    b0 = _mm256_dpbusd_avx_epi32(b0, w, KQ_ACT(0, j));
                    b1 = _mm256_dpbusd_avx_epi32(b1, w, KQ_ACT(1, j));
                    b2 = _mm256_dpbusd_avx_epi32(b2, w, KQ_ACT(2, j));
                    b3 = _mm256_dpbusd_avx_epi32(b3, w, KQ_ACT(3, j));
                    KQ_KEEP(b0);
                    KQ_KEEP(b1);
                    KQ_KEEP(b2);
                    KQ_KEEP(b3);
                }
                /* `m16` stands in for the eight-scale vector: it is hoisted to
                 * the same place, is the same width, and feeds the same
                 * instruction, so the mix and the operand source match. */
                a0 = _mm256_madd_epi16(_mm256_packs_epi32(a0, b0), m16);
                a1 = _mm256_madd_epi16(_mm256_packs_epi32(a1, b1), m16);
                a2 = _mm256_madd_epi16(_mm256_packs_epi32(a2, b2), m16);
                a3 = _mm256_madd_epi16(_mm256_packs_epi32(a3, b3), m16);
            }
#endif
            (void)scales8;
#else
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
                KQ_KEEP(a0);
                KQ_KEEP(a1);
                KQ_KEEP(a2);
                KQ_KEEP(a3);
            }
#endif
#undef KQ_ACT
#ifdef STOKE_MUT_G4_DOTONLY
            /* MUTANT, SPEED PROBE ONLY -- OUTPUT DELIBERATELY WRONG. Keeps the
             * 8-sub-block dot for all four sequences and deletes everything
             * else the group path does: the mins term, both kq_hsum4_i32
             * reductions and the four-wide float fold. It answers the only
             * question left about this kernel -- whether any BIT-EXACT
             * headroom remains in it, or whether what is left is the dot
             * product itself. */
            _mm_storeu_ps(sumf_r + s,
                          _mm_add_ps(_mm_loadu_ps(sumf_r + s),
                                     _mm_cvtepi32_ps(kq_hsum4_i32(a0, a1, a2,
                                                                  a3))));
            (void)d0; (void)d1; (void)d2; (void)d3; (void)M4;
            (void)dwv; (void)dmv;
            continue;
#endif
            const __m128i S4 = kq_hsum4_i32(a0, a1, a2, a3);
            const __m128 a4 = _mm_setr_ps(d0, d1, d2, d3);
            __m128       v  = _mm_loadu_ps(sumf_r + s);
            v = _mm_add_ps(v, _mm_mul_ps(_mm_mul_ps(dwv, a4),
                                         _mm_cvtepi32_ps(S4)));
            v = _mm_sub_ps(v, _mm_mul_ps(_mm_mul_ps(dmv, a4),
                                         _mm_cvtepi32_ps(M4)));
            _mm_storeu_ps(sumf_r + s, v);
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
            vacc_r[s] = _mm256_add_epi32(vacc_r[s], acc);
            (void)ad;
            (void)bsums;
#else
            const i32 S_b = kq_hsum_i32(acc);
            sumf_r[s] = sumf_r[s] + (dw * ad) * (f32)S_b;
            sumf_r[s] = sumf_r[s] - (dm * ad) * (f32)sumi;
#endif
        }
      }
    }

#ifdef STOKE_MUT_NO_BOOKKEEP
    for (u32 t = 0; t < n_rows * n_seq; ++t) out[t] = (f32)kq_hsum_i32(vacc[t]);
#else
    for (u32 t = 0; t < n_rows * n_seq; ++t) out[t] = sumf[t];
#endif
}

/* ── FOUR 32-BYTE SUB-BLOCKS -> FOUR VECTORS WHOSE LANE l BELONGS TO SUB-BLOCK
 * (l mod 4). Eight `vpunpck`s, no cross-lane permute, and it is the SAME eight
 * on the weight side and the activation side -- which is the only reason the
 * VNNI kernel below is affordable.
 *
 * ▶ WHAT IT IS FOR. `vpdpbusd` accumulates four byte-products into each i32
 * lane and has nowhere to put a multiplier, so the recorded objection to using
 * it here was that the per-sub-block scale would need a `vpmulld` costing more
 * than the instruction saves. That is true of the NATURAL layout, where all
 * eight lanes of an accumulator belong to ONE sub-block. It stops being true
 * once the lanes SPAN sub-blocks: then one `vpmaddwd` against a 16-lane scale
 * vector applies all eight scales and reduces, once per super-block.
 *
 * ▶ WHY THIS EXACT SHUFFLE. `vpunpckldq(a,b)` interleaves dwords WITHIN each
 * 128-bit half, and `vpunpcklqdq` then pairs the halves, so four inputs land
 * as [a_m, b_m, c_m, d_m] in the low half and [a_{m+4}, b_{m+4}, c_{m+4},
 * d_{m+4}] in the high half. Lane l therefore holds sub-block (l mod 4), and
 * the four outputs together cover every one of the eight dwords of all four
 * inputs EXACTLY ONCE -- which is what makes the partition a permutation and
 * therefore exact. The obvious alternative, a full 8x8 dword transpose, needs
 * eight `vpermd` on top of this and buys nothing: the lane-to-sub-block map is
 * ours to choose, and `scales8` is reordered to match for two instructions.
 *
 * ⚠ Lane bound, computed not assumed: each lane accumulates FOUR dwords = 16
 * elements, so |lane| <= 16 * 15 * 128 = 30,720 against the i16 ceiling of
 * 32,767 that `vpackssdw` saturates at. Sixteen elements is the most a lane
 * may hold; a layout putting all 32 in one lane would saturate and be wrong. */
static inline void kq_perm4(__m256i s0, __m256i s1, __m256i s2, __m256i s3,
                            __m256i* outv) {
    const __m256i t0 = _mm256_unpacklo_epi32(s0, s1);
    const __m256i t1 = _mm256_unpackhi_epi32(s0, s1);
    const __m256i t2 = _mm256_unpacklo_epi32(s2, s3);
    const __m256i t3 = _mm256_unpackhi_epi32(s2, s3);
    outv[0] = _mm256_unpacklo_epi64(t0, t2);
    outv[1] = _mm256_unpackhi_epi64(t0, t2);
    outv[2] = _mm256_unpacklo_epi64(t1, t3);
    outv[3] = _mm256_unpackhi_epi64(t1, t3);
}

/* ▶▶▶ THE AVX-VNNI TWIN OF `vec_dot_q4_k_q8_k_xn_avx2`, BIT-IDENTICAL TO IT.
 *
 * WHAT CHANGES AND WHAT DOES NOT. The mins term, the horizontal reductions,
 * the four-wide float fold, the tiling and the per-sequence tail are the twin's
 * verbatim. Only the eight-sub-block dot is different:
 *
 *   maddubs twin   8 vpmaddubsw + 8 vpmaddwd + 8 vpaddd   16 uops on ports 0-1
 *   this kernel    8 vpdpbusd + 1 vpackssdw + 1 vpmaddwd   9 uops on ports 0-1
 *
 * per (super-block, sequence). `vpdpbusd` retires 32 unsigned-by-signed MACs in
 * one uop where the pair takes two, and because `kq_perm4` has put four
 * DIFFERENT sub-blocks in each accumulator's lanes, the eight per-sub-block
 * scales arrive in a single `vpmaddwd` at the end instead of one per sub-block.
 *
 * ⚠⚠ BIT-EXACT BY CONTRACT v2, NOT BY TESTING. The contract is the exact
 * integer total S_b = sum_j scale_j * (sum_e q_e * a_e), and integer addition
 * is associative, so ANY partition of those products that reaches the same
 * integer is the same answer. `kq_perm4` is a permutation (every dword of every
 * input appears in exactly one output lane), `vpdpbusd` sums four products per
 * lane, `vpackssdw` moves lanes without changing values because no lane can
 * reach the i16 ceiling, and `vpmaddwd` against `sc16` multiplies lane l's
 * partial by its OWN sub-block's scale and adds two of them. Summing the eight
 * i32 lanes therefore yields sum_j scale_j * dot_j, which is S_b. Nothing here
 * is a float, so nothing here can round differently.
 *
 * ⚠ THE OVERFLOW LADDER, computed at every step rather than at the end:
 * a `vpdpbusd` lane holds 16 elements, |lane| <= 16*15*128 = 30,720 (inside
 * i16, so `vpackssdw` cannot saturate); `vpmaddwd` scales two of those by a
 * 6-bit scale and adds, <= 2*63*30,720 = 3,870,720; eight such lanes sum to
 * <= 30,965,760, two orders inside i32. The accumulators are reset every
 * super-block, exactly as the twin's are, so the ladder never grows past this.
 *
 * ⚠ WHY THE PERMUTED ACTIVATION IS BUILT HERE AND NOT AT QUANTIZE TIME. It is
 * a SECOND view of the same bytes, and the Q8_K block layout is shared with the
 * solo kernel, both Q6_K kernels and the two scalar references -- a layout
 * change would touch all of them for a win that is only in this one. Built at
 * the top of the super-block loop it is paid once per (super-block, sequence)
 * and read by every row in the tile: about 3% of the kernel's work against the
 * 27% the probe measured. Hoisting it to `matvec_batch` would amortise it over
 * all n_out rows instead of the tile's 32 and is the next thing to try if this
 * is ever re-priced.
 *
 * ⚠ THE TAIL IS NOT CONVERTED. `n_seq % 4` sequences fall through to the same
 * per-sequence path the twin uses, on the ORIGINAL activation layout via
 * `kq4_dot_block`. Production tiles at 16 so the tail is decode's n_seq==1 and
 * the prefill chunk's remainder, neither of which is where the time is -- and
 * leaving it shared means the two kernels cannot drift apart there. */
void vec_dot_q4_k_q8_k_xn_vnni_avx2(const void* wrows, u64 row_bytes,
                                    u32 n_rows, const void* const* arows,
                                    u32 n_seq, u64 n, f32* out) {
    if (n % KERNELS_Q4_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q4_k_q8_k_xn_vnni_avx2 -- n=%llu is not a "
              "multiple of %u",
              (unsigned long long)n, KERNELS_Q4_K_BLOCK_ELEMS);
    }
    if (n_seq == 0u || n_seq > KERNELS_BATCH_TILE) {
        fatal("kernels: vec_dot_q4_k_q8_k_xn_vnni_avx2 -- n_seq=%u outside "
              "1..%u; the caller tiles", n_seq, KERNELS_BATCH_TILE);
    }
    if (n_rows == 0u || n_rows > KERNELS_ROW_TILE) {
        fatal("kernels: vec_dot_q4_k_q8_k_xn_vnni_avx2 -- n_rows=%u outside "
              "1..%u; the caller tiles", n_rows, KERNELS_ROW_TILE);
    }
    const u8* wb = (const u8*)wrows;
    u64 nb = n / KERNELS_Q4_K_BLOCK_ELEMS;

    const __m256i lownib8 = _mm256_set1_epi8(0xF);
    f32 sumf[KERNELS_ROW_TILE * KERNELS_BATCH_TILE];
    for (u32 t = 0; t < n_rows * n_seq; ++t) sumf[t] = 0.0f;

    __m256i wnib[8];
    __m256i wperm[8];
    /* One super-block of every grouped sequence's activations, permuted. 4 KiB
     * at the production tile, live only across one iteration of the loop
     * below, and read n_rows times. */
    u8 aperm[KERNELS_BATCH_TILE][KERNELS_Q8_K_BLOCK_ELEMS];

    for (u64 i = 0; i < nb; ++i) {
      /* Only the sequences the grouped path will take: the tail reads the
       * original layout, so permuting for it would be work thrown away. */
      const u32 n_grp = n_seq & ~3u;
      for (u32 s = 0; s < n_grp; ++s) {
        const u8* ablk = (const u8*)arows[s] + i * KERNELS_Q8_K_BLOCK_BYTES;
        const i8* q8 = (const i8*)(ablk + 4);
        __m256i av[8], pv[8];
        for (u32 j = 0; j < 8u; ++j) {
            av[j] = _mm256_loadu_si256(
                (const __m256i*)(const void*)(q8 + j * 32u));
        }
        kq_perm4(av[0], av[1], av[2], av[3], pv);
        kq_perm4(av[4], av[5], av[6], av[7], pv + 4);
        for (u32 m = 0; m < 8u; ++m) {
            _mm256_storeu_si256((__m256i*)(void*)(aperm[s] + m * 32u), pv[m]);
        }
      }
      for (u32 rr = 0; rr < n_rows; ++rr) {
        const u8* blk = wb + (u64)rr * row_bytes + i * KERNELS_Q4_K_BLOCK_BYTES;
        const u8* q4 = blk + 16;
        f32* sumf_r = sumf + (size_t)rr * n_seq;

        u8 scales8[8], mins8[8];
        kq_unpack_scales_inl(blk + 4, scales8, mins8);
        for (u32 j = 0; j < 8u; ++j) {
            const u8* qsrc = q4 + (j / 2u) * 32u;
            __m256i qb =
                _mm256_loadu_si256((const __m256i*)(const void*)qsrc);
            wnib[j] = (j & 1u)
                          ? _mm256_and_si256(_mm256_srli_epi16(qb, 4), lownib8)
                          : _mm256_and_si256(qb, lownib8);
        }
        kq_perm4(wnib[0], wnib[1], wnib[2], wnib[3], wperm);
#ifdef STOKE_MUT_Q4K_VNNI
        /* MUTANT (`-DSTOKE_MUT_Q4K_VNNI`), and it exists to make this kernel's
         * grader NON-VACUOUS. Sub-blocks 4 and 5 are swapped on the WEIGHT side
         * only, so `sc16` applies s4 to sub-block 5 and s5 to sub-block 4 while
         * every instruction, lane count and bound stays exactly as it was.
         *
         * ⚠ It has to be a one-SIDED swap. Permuting both sides identically is
         * a relabelling and stays bit-exact -- which is the whole reason the
         * kernel is allowed to permute at all, and therefore exactly the mutant
         * that would pass and prove nothing. `test_matvec_batch` must FAIL
         * under `STOKE_Q4K_KERNEL=vnni` and PASS under `maddubs`: the second
         * half is what shows the failure is this kernel and not the fixture. */
        kq_perm4(wnib[5], wnib[4], wnib[6], wnib[7], wperm + 4);
#else
        kq_perm4(wnib[4], wnib[5], wnib[6], wnib[7], wperm + 4);
#endif

        /* The eight scales as i16, duplicated across the 128-bit halves --
         * which is exactly the order `vpackssdw` leaves the lanes in, so no
         * further shuffling is needed. Two instructions, once per (row,
         * super-block). */
        const __m128i s8 =
            _mm_loadl_epi64((const __m128i*)(const void*)scales8);
        const __m256i sc16 =
            _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(s8, s8));

        u16 draw, dmraw;
        memcpy(&draw, blk, sizeof draw);
        memcpy(&dmraw, blk + 2, sizeof dmraw);
        const f32 dw = kq_f16c(draw);
        const f32 dm = kq_f16c(dmraw);
        static const i8 dup[16] = {0, 0, 1, 1, 2, 2, 3, 3,
                                   4, 4, 5, 5, 6, 6, 7, 7};
        const __m256i m16 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(
            _mm_loadl_epi64((const __m128i*)(const void*)mins8),
            _mm_loadu_si128((const __m128i*)(const void*)dup)));

        const __m128 dwv = _mm_set1_ps(dw);
        const __m128 dmv = _mm_set1_ps(dm);
        u32 s = 0;
        for (; s + 4u <= n_seq; s += 4u) {
            const u8* ab[4];
            f32       d0, d1, d2, d3;
            for (u32 k = 0; k < 4u; ++k) {
                ab[k] = (const u8*)arows[s + k] + i * KERNELS_Q8_K_BLOCK_BYTES;
            }
            memcpy(&d0, ab[0], sizeof d0);
            memcpy(&d1, ab[1], sizeof d1);
            memcpy(&d2, ab[2], sizeof d2);
            memcpy(&d3, ab[3], sizeof d3);
            /* The mins term reads `bsums`, which the permutation does not
             * touch: it is a per-16-element sum computed at quantize time and
             * stored beside the quants. */
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

            /* Sub-blocks 0-3 into one accumulator per sequence, 4-7 into
             * another: two accumulators are what keeps every lane under the
             * i16 ceiling `vpackssdw` would otherwise saturate at. Eight live
             * accumulators plus one weight vector, so no spill. */
            __m256i a0 = _mm256_setzero_si256();
            __m256i a1 = _mm256_setzero_si256();
            __m256i a2 = _mm256_setzero_si256();
            __m256i a3 = _mm256_setzero_si256();
            __m256i b0 = _mm256_setzero_si256();
            __m256i b1 = _mm256_setzero_si256();
            __m256i b2 = _mm256_setzero_si256();
            __m256i b3 = _mm256_setzero_si256();
#define KQ_PV(k, m) \
    _mm256_loadu_si256((const __m256i*)(const void*)(aperm[s + (k)] + (m) * 32u))
            for (u32 m = 0; m < 4u; ++m) {
                const __m256i w = wperm[m];
                a0 = _mm256_dpbusd_avx_epi32(a0, w, KQ_PV(0, m));
                a1 = _mm256_dpbusd_avx_epi32(a1, w, KQ_PV(1, m));
                a2 = _mm256_dpbusd_avx_epi32(a2, w, KQ_PV(2, m));
                a3 = _mm256_dpbusd_avx_epi32(a3, w, KQ_PV(3, m));
            }
            for (u32 m = 4u; m < 8u; ++m) {
                const __m256i w = wperm[m];
                b0 = _mm256_dpbusd_avx_epi32(b0, w, KQ_PV(0, m));
                b1 = _mm256_dpbusd_avx_epi32(b1, w, KQ_PV(1, m));
                b2 = _mm256_dpbusd_avx_epi32(b2, w, KQ_PV(2, m));
                b3 = _mm256_dpbusd_avx_epi32(b3, w, KQ_PV(3, m));
            }
#undef KQ_PV
            a0 = _mm256_madd_epi16(_mm256_packs_epi32(a0, b0), sc16);
            a1 = _mm256_madd_epi16(_mm256_packs_epi32(a1, b1), sc16);
            a2 = _mm256_madd_epi16(_mm256_packs_epi32(a2, b2), sc16);
            a3 = _mm256_madd_epi16(_mm256_packs_epi32(a3, b3), sc16);

            /* From here down this is the maddubs twin's epilogue verbatim --
             * same reduction, same four-wide float fold, same order. */
            const __m128i S4 = kq_hsum4_i32(a0, a1, a2, a3);
            const __m128 a4 = _mm_setr_ps(d0, d1, d2, d3);
            __m128       v  = _mm_loadu_ps(sumf_r + s);
            v = _mm_add_ps(v, _mm_mul_ps(_mm_mul_ps(dwv, a4),
                                         _mm_cvtepi32_ps(S4)));
            v = _mm_sub_ps(v, _mm_mul_ps(_mm_mul_ps(dmv, a4),
                                         _mm_cvtepi32_ps(M4)));
            _mm_storeu_ps(sumf_r + s, v);
        }

        /* ── THE TAIL, on the original layout, through the shared block dot. */
        for (; s < n_seq; ++s) {
            const u8* ablk = (const u8*)arows[s] + i * KERNELS_Q8_K_BLOCK_BYTES;
            const i8* q8 = (const i8*)(ablk + 4);
            const u8* bsums = ablk + 4 + KERNELS_Q8_K_BLOCK_ELEMS;
            f32 ad;
            memcpy(&ad, ablk, sizeof ad);
            const i32 sumi = kq_hsum_i32(_mm256_madd_epi16(
                m16, _mm256_loadu_si256((const __m256i*)(const void*)bsums)));
            const i32 S_b = kq_hsum_i32(kq4_dot_block(wnib, scales8, q8));
            sumf_r[s] = sumf_r[s] + (dw * ad) * (f32)S_b;
            sumf_r[s] = sumf_r[s] - (dm * ad) * (f32)sumi;
        }
      }
    }

    for (u32 t = 0; t < n_rows * n_seq; ++t) out[t] = sumf[t];
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
        sumf = sumf + (kq_f16c(draw) * ad) * (f32)S_b;
    }
    return sumf;
}

void vec_dot_q6_k_q8_k_xn_avx2(const void* wrows, u64 row_bytes, u32 n_rows,
                               const void* const* arows, u32 n_seq, u64 n,
                               f32* out) {
    if (n % KERNELS_Q6_K_BLOCK_ELEMS != 0) {
        fatal("kernels: vec_dot_q6_k_q8_k_xn_avx2 -- n=%llu is not a multiple "
              "of %u",
              (unsigned long long)n, KERNELS_Q6_K_BLOCK_ELEMS);
    }
    if (n_seq == 0u || n_seq > KERNELS_BATCH_TILE) {
        fatal("kernels: vec_dot_q6_k_q8_k_xn_avx2 -- n_seq=%u outside 1..%u; "
              "the caller tiles", n_seq, KERNELS_BATCH_TILE);
    }
    if (n_rows == 0u || n_rows > KERNELS_ROW_TILE) {
        fatal("kernels: vec_dot_q6_k_q8_k_xn_avx2 -- n_rows=%u outside 1..%u; "
              "the caller tiles", n_rows, KERNELS_ROW_TILE);
    }
    const u8* wb = (const u8*)wrows;
    u64 nb = n / KERNELS_Q6_K_BLOCK_ELEMS;

    f32 sumf[KERNELS_ROW_TILE * KERNELS_BATCH_TILE];
    for (u32 t = 0; t < n_rows * n_seq; ++t) sumf[t] = 0.0f;

    /* ⚠ MEASURED, NOT ASSUMED: hoisting the WIDENING too (a __m256i wv[32]
     * alongside this) made N=16 SLOWER -- 13.565 -> 11.207 tok/s. Q4_K's
     * widening masks and shifts nibbles, which is arithmetic worth hoisting;
     * Q6_K's aux8 already holds ready i8 values, so `kq_load_i8x8_to_i32` is
     * a load-and-convert and materialising it quadruples the bytes re-read
     * per sequence (256 -> 1024) to save nothing. Only the interleaved
     * rebuild is hoisted here. The two K-quants do NOT want the same hoist. */
    i8 aux8[KERNELS_Q6_K_BLOCK_ELEMS];

    for (u64 i = 0; i < nb; ++i) {
      for (u32 rr = 0; rr < n_rows; ++rr) {
        const u8* blk = wb + (u64)rr * row_bytes + i * KERNELS_Q6_K_BLOCK_BYTES;
        const u8* ql = blk;
        const u8* qh = blk + 128;
        const i8* sc = (const i8*)(blk + 192);
        f32* sumf_r = sumf + (size_t)rr * n_seq;

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
        const f32 dw = kq_f16c(draw);

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
            sumf_r[s] = sumf_r[s] + (dw * ad) * (f32)S_b;
        }
      }
    }

    for (u32 t = 0; t < n_rows * n_seq; ++t) out[t] = sumf[t];
}

/* See simd.h for WHY this exists, why vpmaddubsw applies to a signed quant, and
 * the overflow bounds that make the regrouping bit-exact rather than merely
 * close. This file implements it and argues nothing the header already argued.
 *
 * The structure is deliberately Q4_K's, line for line where it can be: weight-
 * only work hoisted to once per row, then a per-sequence body that is nothing
 * but loads, one vpmaddubsw, one vpmaddwd and an accumulate. */
void vec_dot_q6_k_q8_k_xn_maddubs_avx2(const void* wrows, u64 row_bytes,
                                       u32 n_rows, const void* const* arows,
                                       u32 n_seq, u64 n, f32* out) {
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
    if (n_rows == 0u || n_rows > KERNELS_ROW_TILE) {
        fatal("kernels: vec_dot_q6_k_q8_k_xn_maddubs_avx2 -- n_rows=%u outside "
              "1..%u; the caller tiles",
              n_rows, KERNELS_ROW_TILE);
    }
    const u8* wb = (const u8*)wrows;
    u64       nb = n / KERNELS_Q6_K_BLOCK_ELEMS;

#ifdef STOKE_MUT_NO_BOOKKEEP
    /* The Q4_K twin's probe, applied here too: Q4_K_M carries Q6_K tensors and
     * the profiler's matmul total covers both, so mutating only one arm would
     * under-read the ceiling by whatever share Q6_K holds. */
    __m256i vacc[KERNELS_ROW_TILE * KERNELS_BATCH_TILE];
    for (u32 t = 0; t < n_rows * n_seq; ++t) vacc[t] = _mm256_setzero_si256();
#else
    f32 sumf[KERNELS_ROW_TILE * KERNELS_BATCH_TILE];
    for (u32 t = 0; t < n_rows * n_seq; ++t) sumf[t] = 0.0f;
#endif

    /* ▶ THE RAW 6-BIT VALUE, NOT THE OFFSET ONE. This is the only difference
     * from the twin's `aux8` rebuild: no `- 32`, because vpmaddubsw needs an
     * UNSIGNED first operand and the offset is carried by the bsums term below.
     * Element order is identical, so scale group j still covers [16j, 16j+16). */
    u8      auxu[KERNELS_Q6_K_BLOCK_ELEMS];
    __m256i scv[KERNELS_Q6_K_BLOCK_ELEMS / 32u]; /* [s_2c x8, s_2c+1 x8], i16 */

    for (u64 i = 0; i < nb; ++i) {
      for (u32 rr = 0; rr < n_rows; ++rr) {
        const u8* blk = wb + (u64)rr * row_bytes + i * KERNELS_Q6_K_BLOCK_BYTES;
        const u8* ql  = blk;
        const u8* qh  = blk + 128;
        const i8* sc  = (const i8*)(blk + 192);
#ifdef STOKE_MUT_NO_BOOKKEEP
        __m256i* vacc_r = vacc + (size_t)rr * n_seq;
#else
        f32* sumf_r = sumf + (size_t)rr * n_seq;
#endif

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
        const f32 dw = kq_f16c(draw);

        u32 s = 0;
#ifndef STOKE_MUT_NO_BOOKKEEP
        /* ▶▶▶ FOUR SEQUENCES AT A TIME, the Q4_K twin's grouping applied to
         * the twin's shape. Same two exactness arguments, unchanged: the
         * horizontal sums are INTEGER so `kq_hsum4_i32` inherits the scalar's
         * answer (CONTRACT v2), and `P - 32*M` is integer too -- a left shift
         * by 5 is the same i32 product, and no term is anywhere near the
         * bounds simd.h proves for this kernel. The float fold is one add per
         * super-block per lane, the same operation on the same values in the
         * same order the scalar statement below runs.
         *
         * ⚠ THE SUB-BLOCK LOOP IS OUTSIDE THE SEQUENCE LOOP for the reason the
         * twin records at length: with the sequence loop outside, four live
         * accumulators across the weight vectors spill, and the version that
         * spills is SLOWER THAN THE ONE IT REPLACES. Here `uv` and `scv[c]`
         * are live one at a time and shared by all four sequences. */
        for (; s + 4u <= n_seq; s += 4u) {
            const u8* ab[4];
            f32       d0, d1, d2, d3;
            for (u32 k = 0; k < 4u; ++k) {
                ab[k] = (const u8*)arows[s + k] + i * KERNELS_Q8_K_BLOCK_BYTES;
            }
            memcpy(&d0, ab[0], sizeof d0);
            memcpy(&d1, ab[1], sizeof d1);
            memcpy(&d2, ab[2], sizeof d2);
            memcpy(&d3, ab[3], sizeof d3);

            const __m128i M4 = kq_hsum4_i32(
                _mm256_madd_epi16(sc16, _mm256_loadu_si256(
                    (const __m256i*)(const void*)(ab[0] + 4 +
                                                  KERNELS_Q8_K_BLOCK_ELEMS))),
                _mm256_madd_epi16(sc16, _mm256_loadu_si256(
                    (const __m256i*)(const void*)(ab[1] + 4 +
                                                  KERNELS_Q8_K_BLOCK_ELEMS))),
                _mm256_madd_epi16(sc16, _mm256_loadu_si256(
                    (const __m256i*)(const void*)(ab[2] + 4 +
                                                  KERNELS_Q8_K_BLOCK_ELEMS))),
                _mm256_madd_epi16(sc16, _mm256_loadu_si256(
                    (const __m256i*)(const void*)(ab[3] + 4 +
                                                  KERNELS_Q8_K_BLOCK_ELEMS))));

            const i8* q8g[4];
            for (u32 k = 0; k < 4u; ++k) q8g[k] = (const i8*)(ab[k] + 4);
            __m256i a0 = _mm256_setzero_si256();
            __m256i a1 = _mm256_setzero_si256();
            __m256i a2 = _mm256_setzero_si256();
            __m256i a3 = _mm256_setzero_si256();
            for (u32 c = 0; c < KERNELS_Q6_K_BLOCK_ELEMS / 32u; ++c) {
                const __m256i uv = _mm256_loadu_si256(
                    (const __m256i*)(const void*)(auxu + c * 32u));
                const __m256i sc = scv[c];
                a0 = _mm256_add_epi32(
                    a0, _mm256_madd_epi16(
                            _mm256_maddubs_epi16(
                                uv, _mm256_loadu_si256(
                                        (const __m256i*)(const void*)(q8g[0] +
                                                                      c * 32u))),
                            sc));
                a1 = _mm256_add_epi32(
                    a1, _mm256_madd_epi16(
                            _mm256_maddubs_epi16(
                                uv, _mm256_loadu_si256(
                                        (const __m256i*)(const void*)(q8g[1] +
                                                                      c * 32u))),
                            sc));
                a2 = _mm256_add_epi32(
                    a2, _mm256_madd_epi16(
                            _mm256_maddubs_epi16(
                                uv, _mm256_loadu_si256(
                                        (const __m256i*)(const void*)(q8g[2] +
                                                                      c * 32u))),
                            sc));
                a3 = _mm256_add_epi32(
                    a3, _mm256_madd_epi16(
                            _mm256_maddubs_epi16(
                                uv, _mm256_loadu_si256(
                                        (const __m256i*)(const void*)(q8g[3] +
                                                                      c * 32u))),
                            sc));
                /* Same reassociation barrier as the Q4_K twin, and this kernel
                 * needed it more: `objdump` counted 40 stack spills here. */
                KQ_KEEP(a0);
                KQ_KEEP(a1);
                KQ_KEEP(a2);
                KQ_KEEP(a3);
            }
            const __m128i P4  = kq_hsum4_i32(a0, a1, a2, a3);
            const __m128i Sb4 = _mm_sub_epi32(P4, _mm_slli_epi32(M4, 5));
            const __m128  a4  = _mm_setr_ps(d0, d1, d2, d3);
            __m128        v   = _mm_loadu_ps(sumf_r + s);
            v = _mm_add_ps(v, _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(dw), a4),
                                         _mm_cvtepi32_ps(Sb4)));
            _mm_storeu_ps(sumf_r + s, v);
        }
#endif

        /* ── PER-SEQUENCE WORK: two loads, one vpmaddubsw, one vpmaddwd and an
         * add, eight times -- against the twin's thirty-two widening loads. */
        for (; s < n_seq; ++s) {
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
            vacc_r[s] = _mm256_add_epi32(vacc_r[s], acc);
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
            sumf_r[s] = sumf_r[s] + (dw * ad) * (f32)S_b;
#endif
        }
      }
    }

#ifdef STOKE_MUT_NO_BOOKKEEP
    for (u32 t = 0; t < n_rows * n_seq; ++t) out[t] = (f32)kq_hsum_i32(vacc[t]);
#else
    for (u32 t = 0; t < n_rows * n_seq; ++t) out[t] = sumf[t];
#endif
}

#else /* !x86-64 */

bool kernels_cpu_has_avx_vnni(void) { return false; }

f32 kernels_f16_to_f32_f16c_probe(u16 h) {
    (void)h;
    fatal("kernels: kernels_f16_to_f32_f16c_probe on a non-x86-64 build");
}

void kernels_q4_k_unpack_scales_inl_probe(const void* packed12, u8* scales8,
                                          u8* mins8) {
    (void)packed12;
    (void)scales8;
    (void)mins8;
    fatal("kernels: kernels_q4_k_unpack_scales_inl_probe on a non-x86-64 "
          "build");
}

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

void att_score4_avx2(const f32* qt, u64 qstride, const f32* kj, u32 hd,
                     f32 scale, f32* out4) {
    (void)qt;
    (void)qstride;
    (void)kj;
    (void)hd;
    (void)scale;
    (void)out4;
    fatal("kernels: att_score4_avx2 on a non-x86-64 build -- callers must "
          "check kernels_cpu_has_avx_vnni() first");
}

void att_accum4_avx2(f32* o0, f32* o1, f32* o2, f32* o3, const f32* vj,
                     const f32* w4, u32 hd) {
    (void)o0;
    (void)o1;
    (void)o2;
    (void)o3;
    (void)vj;
    (void)w4;
    (void)hd;
    fatal("kernels: att_accum4_avx2 on a non-x86-64 build -- callers must "
          "check kernels_cpu_has_avx_vnni() first");
}

void att_score4x4_avx2(const f32* qt, u64 qstride, const f32* k0,
                       const f32* k1, const f32* k2, const f32* k3, u32 hd,
                       f32 scale, f32* out16) {
    (void)qt; (void)qstride; (void)k0; (void)k1; (void)k2; (void)k3;
    (void)hd; (void)scale; (void)out16;
    fatal("kernels: att_score4x4_avx2 on a non-x86-64 build -- callers must "
          "check kernels_cpu_has_avx_vnni() first");
}

void att_accum4x8_avx2(f32* o0, f32* o1, f32* o2, f32* o3,
                       const f32* const* v8, const f32* w32, u32 hd) {
    (void)o0; (void)o1; (void)o2; (void)o3; (void)v8; (void)w32; (void)hd;
    fatal("kernels: att_accum4x8_avx2 on a non-x86-64 build -- callers must "
          "check kernels_cpu_has_avx_vnni() first");
}

void vec_dot_q4_k_q8_k_xn_avx2(const void* wrows, u64 row_bytes, u32 n_rows,
                               const void* const* arows, u32 n_seq, u64 n,
                               f32* out) {
    (void)wrows;
    (void)row_bytes;
    (void)n_rows;
    (void)arows;
    (void)n_seq;
    (void)n;
    (void)out;
    fatal("kernels: vec_dot_q4_k_q8_k_xn_avx2 on a non-x86-64 build -- "
          "callers must check kernels_cpu_has_avx_vnni() first");
}

void vec_dot_q4_k_q8_k_xn_vnni_avx2(const void* wrows, u64 row_bytes,
                                    u32 n_rows, const void* const* arows,
                                    u32 n_seq, u64 n, f32* out) {
    (void)wrows;
    (void)row_bytes;
    (void)n_rows;
    (void)arows;
    (void)n_seq;
    (void)n;
    (void)out;
    fatal("kernels: vec_dot_q4_k_q8_k_xn_vnni_avx2 on a non-x86-64 build -- "
          "callers must check kernels_cpu_has_avx_vnni() first");
}

void vec_dot_q6_k_q8_k_xn_avx2(const void* wrows, u64 row_bytes, u32 n_rows,
                               const void* const* arows, u32 n_seq, u64 n,
                               f32* out) {
    (void)wrows;
    (void)row_bytes;
    (void)n_rows;
    (void)arows;
    (void)n_seq;
    (void)n;
    (void)out;
    fatal("kernels: vec_dot_q6_k_q8_k_xn_avx2 on a non-x86-64 build -- "
          "callers must check kernels_cpu_has_avx_vnni() first");
}


void vec_dot_q6_k_q8_k_xn_maddubs_avx2(const void* wrows, u64 row_bytes,
                                       u32 n_rows, const void* const* arows,
                                       u32 n_seq, u64 n, f32* out) {
    (void)wrows;
    (void)row_bytes;
    (void)n_rows;
    (void)arows;
    (void)n_seq;
    (void)n;
    (void)out;
    fatal("kernels: vec_dot_q6_k_q8_k_xn_maddubs_avx2 on a non-x86-64 build -- "
          "callers must check kernels_cpu_has_avx_vnni() first");
}

#endif
