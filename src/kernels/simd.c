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
 * bit-exact SIMD twin possible at all. */
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

        /* loadu, never load: these point into the GGUF mmap at a 34-byte
         * block stride + 2, so they are essentially never 32-byte aligned
         * and an aligned load would fault on the first row. */
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
        __m256i acc = _mm256_sub_epi32(p1, _mm256_slli_epi32(p2, 7));

        /* Sequential f32 accumulation in the scalar's exact order, and the
         * two scales multiplying each other FIRST -- see simd.h. */
        sum += (f32)hsum_i32_8(acc) * (dw * da);
    }
    return sum;
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

#else /* !x86-64 */

bool kernels_cpu_has_avx_vnni(void) { return false; }

void dequant_row_q8_0_avx2(const void* row, u64 n, f32* out) {
    (void)row;
    (void)n;
    (void)out;
    fatal("kernels: dequant_row_q8_0_avx2 on a non-x86-64 build");
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

#endif
