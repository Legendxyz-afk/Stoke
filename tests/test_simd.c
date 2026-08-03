/* tests/test_simd.c -- M3 gate [1]: the scalar<->SIMD exact-diff harness.
 *
 * The forward note's gate is "scalar<->SIMD exact-diff (integer paths) /
 * <=1e-5 rel (f32 paths) on 1e4 random tensors per kernel, all quant kinds".
 * This file IS that gate for the kernels K3 twins, and it is written to fail
 * in every way the property can actually be violated:
 *
 *   - INTEGER PATHS ARE COMPARED AS BIT PATTERNS, not with an epsilon. The
 *     whole point of LOCKED decision #2 (both paths run the same i32 integer
 *     core) is that no tolerance is needed, so accepting one would hide
 *     exactly the reordering it exists to forbid -- a __m256 accumulator
 *     summing blocks in a different order lands within 1e-6 while being
 *     precisely the defect the gate forbids.
 *
 *   - THE SUITE REFUSES TO PASS VACUOUSLY. Every twin is called DIRECTLY,
 *     never through a dispatcher that could fall back to scalar, and
 *     simd_cpu_probe_finds_avx_vnni_on_this_box asserts the feature is
 *     actually present. Without that assertion this file would go green on a
 *     CPU where nothing under test ever executed, which is the failure mode
 *     this project has paid for three times in measurement and would
 *     otherwise now import into its correctness gate.
 *
 * Random inputs are drawn from a FIXED seed: a gate that samples a different
 * 1e4 tensors on every run cannot be bisected, and a failure that does not
 * reproduce is not evidence. The generator is a local xorshift rather than
 * rand(), whose sequence is implementation-defined and would differ between
 * the Windows and WSL legs -- the two legs must test the same tensors.
 */
#include "common.h"

#include "kernels/kernels.h"
#include "kernels/simd.h"

#include <string.h>

#include "test.h"

/* Block layout, mirrored from kernels.h's Q8_0 contract: f16 scale then 32
 * i8 quants. Declared here rather than reaching into the kernel's internals
 * so the test states its own understanding of the format it is feeding. */
#define BLK_ELEMS 32u
#define BLK_BYTES 34u

/* xorshift64*, fixed seed. Deterministic and identical on both legs. */
static u64 g_rng = 0x9E3779B97F4A7C15ull;

static void rng_reset(void) { g_rng = 0x9E3779B97F4A7C15ull; }

static u32 rng_next(void) {
    g_rng ^= g_rng >> 12;
    g_rng ^= g_rng << 25;
    g_rng ^= g_rng >> 27;
    return (u32)((g_rng * 0x2545F4914F6CDD1Dull) >> 32);
}

/* Fills one Q8_0 block: an f16 scale and 32 signed quants.
 *
 * The quants deliberately include the endpoints -128 and 127 rather than a
 * comfortable interior range. -128 is the value that makes the dpbusd sign
 * trick interesting: _mm256_sign_epi8(x, x) cannot represent |-128| in i8
 * and leaves it as -128, so if the scalar and SIMD paths disagree anywhere
 * it is most likely there, and a test drawing only from [-127,127] would
 * miss it completely. */
static void fill_block(u8* blk, f32 scale) {
    u16 h = f32_to_f16(scale);
    memcpy(blk, &h, 2);
    for (u32 i = 0; i < BLK_ELEMS; i++) {
        blk[2 + i] = (u8)(i8)(i32)((rng_next() % 256u) - 128u);
    }
}

/* Scales spanning several binades, including denormal-ish and large, so the
 * f32 product d_w*d_a is not always in a benign range. Kept finite: inf/nan
 * scales are a fatal() path on both implementations, tested elsewhere. */
static f32 random_scale(void) {
    static const f32 mags[] = {1.0f / 4096.0f, 1.0f / 127.0f, 0.00731f,
                               0.5f, 1.0f, 7.25f, 133.0f};
    u32 i = rng_next() % (u32)(sizeof mags / sizeof mags[0]);
    f32 s = mags[i];
    return (rng_next() & 1u) ? -s : s;
}

static bool bit_identical(f32 a, f32 b) {
    return memcmp(&a, &b, sizeof(f32)) == 0;
}

/* memcpy, not a pointer cast: type-punning an f32 through a u32* is a
 * strict-aliasing violation that -Werror rejects outright here, and at -O3
 * it is the kind gcc is entitled to miscompile silently. */
static u32 f32_bits(f32 v) {
    u32 b;
    memcpy(&b, &v, sizeof b);
    return b;
}

/* ------------------------------------------------------------------ tests */

/* THE GUARD THAT KEEPS EVERY OTHER TEST IN THIS FILE HONEST. If this fails,
 * the twins below are not being exercised on this box and their green bars
 * mean nothing -- so this is a hard failure, not a skip. */
TEST(simd_cpu_probe_finds_avx_vnni_on_this_box) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
}

TEST(simd_vec_dot_q8_0_q8_0_is_bit_exact_vs_scalar_over_1e4_tensors) {
    if (!kernels_cpu_has_avx_vnni()) {
        fprintf(stderr, "FAIL %s:%d: no AVX-VNNI -- the exact-diff gate "
                        "cannot run on this box\n", __FILE__, __LINE__);
        g_test_case_failed = 1;
        return;
    }

    /* Row lengths cover one block, a couple of blocks, and lengths that are
     * NOT multiples of the 4-block unrolling a vectorized loop naturally
     * wants -- a twin that handles only whole groups of 4 blocks would pass
     * on 128 and fail on 96, so both are here. */
    static const u64 lens[] = {32, 64, 96, 128, 160, 256, 2560};
    const u32        n_lens = (u32)(sizeof lens / sizeof lens[0]);
    const u32        iters  = 10000;

    rng_reset();
    u8* w = (u8*)xmalloc((size_t)(2560u / BLK_ELEMS) * BLK_BYTES);
    u8* a = (u8*)xmalloc((size_t)(2560u / BLK_ELEMS) * BLK_BYTES);

    u32 mismatches = 0;
    for (u32 it = 0; it < iters; it++) {
        u64 n      = lens[it % n_lens];
        u64 blocks = n / BLK_ELEMS;
        for (u64 b = 0; b < blocks; b++) {
            fill_block(w + b * BLK_BYTES, random_scale());
            fill_block(a + b * BLK_BYTES, random_scale());
        }

        f32 want = vec_dot_q8_0_q8_0(w, a, n);
        f32 got  = vec_dot_q8_0_q8_0_avx2(w, a, n);

        if (!bit_identical(want, got)) {
            if (mismatches < 3u) {
                fprintf(stderr,
                        "FAIL %s:%d: iter %u n=%llu scalar=%.9g simd=%.9g "
                        "(bits %08x vs %08x)\n",
                        __FILE__, __LINE__, it, (unsigned long long)n,
                        (double)want, (double)got,
                        f32_bits(want), f32_bits(got));
            }
            mismatches++;
        }
    }
    if (mismatches != 0u) {
        fprintf(stderr, "FAIL %s:%d: %u of %u tensors differ\n", __FILE__,
                __LINE__, mismatches, iters);
        g_test_case_failed = 1;
    }

    free(w);
    free(a);
}

/* The all-zero block is the one input where the scalar kernel's f32 grouping
 * could plausibly be replicated by a wrong-but-close SIMD path (0 * anything
 * is 0 in both), so it is asserted separately rather than left to the
 * random sweep to hit by chance. */
TEST(simd_vec_dot_q8_0_q8_0_matches_scalar_on_all_zero_quants) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
    if (!kernels_cpu_has_avx_vnni()) return;

    u8 w[BLK_BYTES * 2];
    u8 a[BLK_BYTES * 2];
    memset(w, 0, sizeof w);
    memset(a, 0, sizeof a);
    u16 h = f32_to_f16(0.125f);
    memcpy(w, &h, 2);
    memcpy(w + BLK_BYTES, &h, 2);
    memcpy(a, &h, 2);
    memcpy(a + BLK_BYTES, &h, 2);

    ASSERT_TRUE(bit_identical(vec_dot_q8_0_q8_0(w, a, 64),
                              vec_dot_q8_0_q8_0_avx2(w, a, 64)));
}

/* -128 cannot be negated in i8, which is exactly what the sign trick does to
 * the activation side. Driven to saturation on BOTH sides so the worst case
 * is a real input rather than a hypothetical. */
TEST(simd_vec_dot_q8_0_q8_0_matches_scalar_at_i8_min_saturation) {
    ASSERT_TRUE(kernels_cpu_has_avx_vnni());
    if (!kernels_cpu_has_avx_vnni()) return;

    u8 w[BLK_BYTES];
    u8 a[BLK_BYTES];
    u16 h = f32_to_f16(1.0f);
    memcpy(w, &h, 2);
    memcpy(a, &h, 2);
    for (u32 i = 0; i < BLK_ELEMS; i++) {
        w[2 + i] = (u8)(i8)-128;
        a[2 + i] = (u8)(i8)-128;
    }

    ASSERT_TRUE(bit_identical(vec_dot_q8_0_q8_0(w, a, 32),
                              vec_dot_q8_0_q8_0_avx2(w, a, 32)));
}
